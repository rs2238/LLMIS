import json

import torch

from config import block_size, batch_size, learning_rate, max_iters, eval_interval, eval_iters, device
from model import Model


def main():
    torch.manual_seed(1337)

    #--------------------------------
    #data pipeline
    #--------------------------------
    #load the corpus
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    #vocab
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    #encode (tokenizer): decode isn't needed for training, only serve.py generates text
    stoi = {ch: i for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s] #string to list of integers

    #split
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data)) #first 90% train
    train_data = data[:n]
    val_data = data[n:] #last 10% validation

    #--------------------------------
    #batch sampler
    #--------------------------------
    #adds stochasticity to training process (sample random windows of text from training data and feed them to optimizer)
    def get_batch(split):
        data = train_data if split == 'train' else val_data #split -> string input -> tells us which data to sample from
        ix = torch.randint(len(data) - block_size, (batch_size,)) #len(data) - block_size -> highest index we can sample from, (batch_size,) -> # windows we look at concurrently
        x = torch.stack([data[i:i+block_size] for i in ix]) #creates tensor of shape (B, T) where B = batch size, T = block size
        y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #offset by 1 produces the "correct answer" for each char
        x, y = x.to(device), y.to(device)
        return x, y

    #--------------------------------
    #training loop
    #--------------------------------
    model = Model(vocab_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    best_val_loss = float('inf')

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if losses['val'] < best_val_loss: #only save on improvement, avoids checkpointing an overfit final step
                best_val_loss = losses['val']
                torch.save(model.state_dict(), 'model.pt')
                print(f"  saved (val {best_val_loss:.4f})")

        xb, yb = get_batch('train')
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    #--------------------------------
    #persist artifacts
    #--------------------------------
    with open('vocab.json', 'w') as f:
        json.dump(chars, f)


if __name__ == "__main__":
    main()
