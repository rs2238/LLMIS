import json

import torch

from config import device
from model import Model


def load_model():
    with open('vocab.json', 'r') as f:
        chars = json.load(f)
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s] #string to list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) #integer list to string

    model = Model(vocab_size)
    model.load_state_dict(torch.load('model.pt', map_location=device))
    model.eval()
    model.to(device)

    return model, encode, decode, stoi


if __name__ == "__main__":
    model, encode, decode, _ = load_model()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    #500 tokens exceeds block_size (256), so this needs the sliding-window naive path --
    #see the scope note on Model.generate_with_cache in model.py
    print(decode(model.generate(context, max_new_tokens=500, use_cache=False)[0].tolist()))
