import torch
import torch.nn as nn
import torch.nn.functional as F

from config import block_size, n_embd, n_head, n_layer, dropout

#--------------------------------
#single attention head
#--------------------------------
class Head(nn.Module):
    #single self-attention head
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) #W_k
        self.query = nn.Linear(n_embd, head_size, bias=False) #W_q
        self.value = nn.Linear(n_embd, head_size, bias=False) #W_v
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #lower triangular matrix to mask out future tokens
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) #k vector: (B,T,C) -> (B,T,head_size)
        q = self.query(x) #q vector: (B,T,C) -> (B,T,head_size)

        #compute (weighted) attention scores (a_ij -> how much should token i attend to token j)
        wei = q @ k.transpose(-2,-1) * C**-0.5 #(B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) #mask out future tokens
        wei = F.softmax(wei, dim=-1) #softmax for weighted distribution

        #dropout for regularization
        wei = self.dropout(wei)

        #compute z_i (a_ij * v_j -> weighted value for old embedding) *this is not the final contextually rich embedding
        v = self.value(x) #(B,T,C) -> (B,T,head_size)
        out = wei @ v #(B,T,T) @ (B,T,C) -> (B,T,C)
        return out

    #kv caching version of forward() above
    #conceptual note about kv caching: q isnt stored because each token only asks "what am i looking for?" once, when it is first generated. k and v are stored because each token is asked "what do you have to offer?" every time a new token is generated.
    #there are 4 layers (abstractions) of the cache, each defined separately in Head, MultiHeadAttention, Block, and Model
    #this is the bottom layer, where the actual K/V cache is stored and updated: cache[layer][head]
    def forward_cached(self, x, cache, pad_mask=None):
        #inputs:
        #x: (B, T_new, C) just the newest tokens
        #cache: None (first call) or (k_cache, v_cache), each with shape (B, T_cache, head_size); note: head_size = n_embd // n_head (distribution of embedding budget across heads)
        
        k_new = self.key(x) #(B, T_new, head_size)
        q_new = self.query(x) #(B, T_new, head_size) 
        v_new = self.value(x) #(B, T_new, head_size)

        if cache is None:
            k, v = k_new, v_new
            T_cache = 0
        else:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k_new], dim=1) #(B, T_cache+T_new, head_size) *append, don't recompute
            v = torch.cat([v_cache, v_new], dim=1)
            T_cache = k_cache.shape[1]

        T_new = x.shape[1]
        T_key = T_cache + T_new
        C = x.shape[-1] #matches forward()'s scaling exactly (n_embd, not head_size) so cached and uncached paths stay numerically identical
        wei = q_new @ k.transpose(-2, -1) * C**-0.5 #(B,T_new,head_size) @ (B,head_size,T_key) -> (B,T_new,T_key)

        #causal mask
        causal = torch.tril(torch.ones(T_new, T_key, device=x.device, dtype=torch.bool), diagonal=T_cache)

        #part of batching implementation: handle the padding mask to ensure that queries do not attend to padded (blank) positions
        if pad_mask is not None:
            B = x.shape[0]
            known_len = pad_mask.shape[1] #max_prompt_len
            if T_key <= known_len:
                key_real = pad_mask[:, :T_key]
            else:
                generated_real = torch.ones(B, T_key - known_len, device=x.device, dtype=torch.bool)
                key_real = torch.cat([pad_mask, generated_real], dim=1) #(B, T_key)
            allowed = causal.unsqueeze(0) & key_real.unsqueeze(1) #(B, T_new, T_key)

            #correctness fix
            query_pos = torch.arange(T_new, device=x.device).unsqueeze(1) + T_cache
            key_pos = torch.arange(T_key, device=x.device).unsqueeze(0)
            self_diagonal = (query_pos == key_pos)
            allowed = allowed | self_diagonal.unsqueeze(0)
        else:
            allowed = causal

        wei = wei.masked_fill(~allowed, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v #(B,T_new,T_key) @ (B,T_key,head_size) -> (B,T_new,head_size)
        return out, (k, v) #return the grown cache so the caller can pass it to the next step

#--------------------------------
#multi attention head
#--------------------------------
#MultiHeadAttention holds n_head Heads
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #list of heads
        self.proj = nn.Linear(num_heads * head_size, n_embd) #projection layer to combine heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #concatenate heads along the last dimension
        out = self.proj(out) #project back to original embedding size
        out = self.dropout(out) #dropout for regularization
        return out

    #kv caching version of forward() above
    #second layer of cache: cache[layer] is generated here
    def forward_cached(self, x, layer_cache, pad_mask=None):
        outs = []
        new_layer_cache = []
        for head_idx, h in enumerate(self.heads):
            head_cache = layer_cache[head_idx] if layer_cache is not None else None
            out, new_head_cache = h.forward_cached(x, head_cache, pad_mask)
            outs.append(out)
            new_layer_cache.append(new_head_cache)

        out = torch.cat(outs, dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out, new_layer_cache

#--------------------------------
#MLP (feed forward)
#--------------------------------
class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), #x''_i * W_up = x'''_i (each row in W_up represents a pattern or feature)
            nn.ReLU(), #negative values are set to 0, positive values are kept (x'''_i -> ReLU -> x''''_i)
            nn.Linear(4 * n_embd, n_embd), #x''''_i * W_down (each column in W_down represents what is added to the embedding given a certain pattern is found)
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

#--------------------------------
#full block with MLP
#--------------------------------
#Block holds a single MultiHeadAttention and a single MLP, with residual connections and layer normalization
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) #turns regular token embeddings into contextually rich token embeddings
        self.mlp = MLP(n_embd) #pattern matching memory mechanism (encodes updates into contextually rich token embeddings if certain patterns are found)
        self.ln1 = nn.LayerNorm(n_embd) #normalization layer to stabilize training
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) #communicate (pre-norm + residual)
        x = x + self.mlp(self.ln2(x)) #compute (pre-norm + residual)
        return x

    #third layer of cache: nothing is generated here (still at cache[layer] abstraction)
    #the block holds an MLP in addition to the MultiHeadAttention, but the MLP doesn't attend to other positions, so it needs no cache 
    def forward_cached(self, x, layer_cache, pad_mask=None):
        sa_out, new_layer_cache = self.sa.forward_cached(self.ln1(x), layer_cache, pad_mask)
        x = x + sa_out
        x = x + self.mlp(self.ln2(x))
        return x, new_layer_cache

#--------------------------------
#full model
#--------------------------------
class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #lookup table for token embeddings (vocab_size rows, each an n_embd vector)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) #positional encoding lookup table (block_size rows, each an n_embd vector)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)]) #the tower: n_layer independent Block instances where each block's output is the next block's input
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb #positionally aware token embeddings
        x = self.blocks(x) #positionally aware contextually rich token embeddings
        x = self.ln_f(x)
        logits = self.lm_head(x) #converts (B,T, n_embd) -> (B,T,vocab_size): each token now has a raw score for what the next token should be

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    #final layer of cache: generates cache (final abstraction)
    def forward_cached(self, idx, start_pos, cache, pad_mask=None, position_ids=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        if position_ids is None: #if no batching
            pos_emb = self.position_embedding_table(torch.arange(start_pos, start_pos + T, device=idx.device))
        else: #if batching (true position_ids are provided)
            pos_emb = self.position_embedding_table(position_ids) #(B,T) -> (B,T,n_embd) directly
        x = tok_emb + pos_emb #each token now carries what it is and where it is

        new_cache = []
        for layer_idx, block in enumerate(self.blocks):
            layer_cache = cache[layer_idx] if cache is not None else None
            x, new_layer_cache = block.forward_cached(x, layer_cache, pad_mask) #note: pad_mask gets passed through each layer since the kv cache keeps notes on the blank (padded) positions -> pad_mask is the reminder to ignore those positions during attention computation
            new_cache.append(new_layer_cache)

        x = self.ln_f(x) #final layer normalization before the language model head
        logits = self.lm_head(x) #lm.head is the final linear layer that creates the final distribution over the vocabulary
        return logits, new_cache #return the final logits and the updated cache for the next incremental step
    
    #original, uncached generation loop (kept same for benchmarking purposes)
    def generate_naive(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] #crops the context to the last block_size tokens
            logits, _ = self(idx_cond) #this is the model thinking about what the next token should be -> it generates the raw opinion (logits) for the next token
            logits = logits[:, -1, :] #only care about the last time step
            probs = F.softmax(logits, dim=-1) #convert logits to probabilities
            idx_next = torch.multinomial(probs, num_samples=1) #samples from the distribution to get the next token
            idx = torch.cat((idx, idx_next), dim=1) #add the new token to the sequence
        return idx

    #same sampling loop as generate_naive, but the prompt is processed once ("prefill"), then each new token only runs a single incremental step through forward_cached instead of a full recompute
    def generate_with_cache(self, idx, max_new_tokens):
        assert idx.shape[1] + max_new_tokens <= block_size, (
            f"generate_with_cache is bounded by block_size ({block_size}): "
            f"prompt ({idx.shape[1]}) + max_new_tokens ({max_new_tokens}) exceeds it. "
            f"use generate_naive for longer generations (see scope note above)."
        )

        #prefill: run the whole prompt through once, building the initial cache
        cache = [None] * n_layer
        logits, cache = self.forward_cached(idx, start_pos=0, cache=cache)
        logits = logits[:, -1, :]

        for t in range(max_new_tokens):
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if t == max_new_tokens - 1:
                break #sampled the last token we need, no point running the model again

            #idx_next is the only new token, start_pos is its position in the full sequence
            logits, cache = self.forward_cached(idx_next, start_pos=idx.shape[1] - 1, cache=cache)
            logits = logits[:, -1, :]

        return idx

    #process several independent requests as one batched generation instead of one generate_with_cache() call per request
    #few things happen here:
    #1. the prompts are left-padded to the same length
    #2. a pad_mask is created to distinguish real tokens from padding tokens
    #3. the model is called (forward_cached) with the padded input and pad_mask to generate tokens for all requests in the batch
    def generate_batched(self, idx_list, max_new_tokens_list, pad_id=0):
        #2/3 things that batching.py unpacked are collected here: idx_list and max_new_tokens_list
        B = len(idx_list) #list of prompts (the token vectors for each prompt) in the batch
        device = idx_list[0].device
        prompt_lens = [t.shape[0] for t in idx_list] #length of each prompt (token vector) in the batch
        max_prompt_len = max(prompt_lens) #maximum length of any prompt in the batch
        total_steps = max(max_new_tokens_list) #maximum number of new tokens any request in the batch wants to generate

        #ensure that the total sequence length (prompt + generated tokens) does not exceed the model's max length
        assert max_prompt_len + total_steps <= block_size, (
            f"generate_batched is bounded by block_size ({block_size}): longest prompt "
            f"({max_prompt_len}) + longest requested max_new_tokens ({total_steps}) exceeds it "
            f"(same reasoning as generate_with_cache's scope note)."
        )

        #left padding is used to align all prompts to the same physical length in the batch tensor by adding blank tokens at the beginning of each prompt to fill empty space in shorter prompts
        padded = torch.full((B, max_prompt_len), pad_id, dtype=torch.long, device=device) #padded starts as a grid of padding tokens (blanks)
        pad_mask = torch.zeros((B, max_prompt_len), dtype=torch.bool, device=device) #pad_mask distinguishes real tokens from padding tokens (True = real token, False = padding), everything starts as False
        for b, t in enumerate(idx_list):
            L = prompt_lens[b] #length of the current prompt (number of real tokens)
            padded[b, max_prompt_len - L:] = t #place the real tokens into the padded tensor
            pad_mask[b, max_prompt_len - L:] = True #mark the positions of the real tokens in the pad mask

        #model relies on absolute position embeddings, so we need to track the true position of each token separately from its physical slot in the batch tensor
        #left padding can shift a 3 token prompt to physical positions 7,8,9 in a batch of width 10 when we want its true positions to be 0,1,2 -> position_ids tracks the true positions separately to maintain quality for shorter prompts batched with longer prompts
        position_ids = torch.zeros((B, max_prompt_len), dtype=torch.long, device=device) #starts off as a grid of zeros
        for b in range(B):
            L = prompt_lens[b]
            position_ids[b, max_prompt_len - L:] = torch.arange(L, device=device) #number the true positions of the tokens in the current prompt

        #prefill: one batched forward pass over all B (padded) prompts at once
        cache = [None] * n_layer #this is the kv cache for each layer
        logits, cache = self.forward_cached(padded, start_pos=0, cache=cache, pad_mask=pad_mask, position_ids=position_ids) #model reads entire grid in one go, then fills kv cache, and generates logits (score for every possible next word) for every token
        logits = logits[:, -1, :] #keep only the logits for the last token of each prompt since these are the scores for the next token to be generated

        #decode: iteratively generate tokens for each prompt in the batch until all prompts have been completed or the maximum number of steps is reached
        out = padded.clone() #output tensor starts off as a copy of the padded input prompts
        prompt_lens_t = torch.tensor(prompt_lens, device=device) #for per-row decode position ids below

        for t in range(total_steps):
            probs = F.softmax(logits, dim=-1) #softmax to convert logits to probabilities to sample the next token
            next_tok = torch.multinomial(probs, num_samples=1) #sample next token from the distribution just generated for each prompt in the batch
            out = torch.cat([out, next_tok], dim=1) #append newly sampled token to the output tensor for each prompt in the batch

            if t == total_steps - 1:
                break #every row has its last needed token -> no point running the model again

            step_position_ids = (prompt_lens_t + t).unsqueeze(1) #position ids for the newly generated token in each row, accounting for the row's prompt length and the current step
            logits, cache = self.forward_cached(next_tok, start_pos=0, cache=cache, pad_mask=pad_mask, position_ids=step_position_ids) #generate logits for the newly generated token given the current kv cache and position ids
            logits = logits[:, -1, :] #keep only the logits for the last token of each prompt since these are the scores for the next token to be generated

        #trim per row: drop this row's left-padding, and cap its completion at what it actually asked for
        results = []
        for b in range(B):
            L = prompt_lens[b]
            row = out[b]
            prompt_part = row[max_prompt_len - L : max_prompt_len]
            completion_part = row[max_prompt_len : max_prompt_len + max_new_tokens_list[b]]
            results.append(torch.cat([prompt_part, completion_part]))
        return results

    #new default entry point: dispatches to the cached or naive path
    def generate(self, idx, max_new_tokens, use_cache=True):
        if use_cache:
            return self.generate_with_cache(idx, max_new_tokens)
        return self.generate_naive(idx, max_new_tokens)
