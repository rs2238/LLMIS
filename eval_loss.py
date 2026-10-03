#quantifies accuracy loss due to quantization (specifically FP32 vs INT8 cross-entropy loss on held-out validation data)
import torch

from config import batch_size, block_size, eval_iters
from quantize import quantize_model
from serve import load_model


def get_val_batch(val_data):
    ix = torch.randint(len(val_data) - block_size, (batch_size,))
    x = torch.stack([val_data[i:i + block_size] for i in ix])
    y = torch.stack([val_data[i + 1:i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def eval_loss(model, val_batches):
    #evaluated on the same pre-sampled batches for every model passed in, so any loss difference reflects the model, not which random windows happened to get sampled
    model.eval()
    losses = torch.zeros(len(val_batches))
    for k, (x, y) in enumerate(val_batches):
        _, loss = model(x, y)
        losses[k] = loss.item()
    return losses.mean().item()


if __name__ == "__main__":
    torch.manual_seed(1337) #matches train.py's seed, for a reproducible val split + sample

    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    model, encode, decode, stoi = load_model()
    model_fp32 = model.to("cpu")

    #same 90/10 split convention as train.py
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    val_data = data[n:]

    val_batches = [get_val_batch(val_data) for _ in range(eval_iters)]

    fp32_loss = eval_loss(model_fp32, val_batches)

    model_int8 = quantize_model(model_fp32)
    int8_loss = eval_loss(model_int8, val_batches)

    print(f"FP32 val loss: {fp32_loss:.4f}")
    print(f"INT8 val loss: {int8_loss:.4f}")
    print(f"delta:         {int8_loss - fp32_loss:+.4f}  ({(int8_loss / fp32_loss - 1) * 100:+.2f}%)")
