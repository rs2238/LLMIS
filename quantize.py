#quantization utilities for converting a model's linear layers from FP32 to INT8
#note: dynamic quantization is used here, meaning weights are quantized once at conversion time and activations are quantized on the fly during forward passes
import copy

import torch
import torch.nn as nn

if "fbgemm" in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = "fbgemm" #intel/amd
elif "qnnpack" in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = "qnnpack" #apple silicon, ARM linux
else:
    raise RuntimeError(
        "no quantized CPU backend (fbgemm/qnnpack) available in this torch build"
    )

#function to quantize a model's linear layers from FP32 to INT8
def quantize_model(model):
    model = copy.deepcopy(model).to("cpu").eval() #create a deep copy of the model, move it to CPU, and set it to evaluation mode before quantization
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8) #quantize the model's linear layers to INT8 while keeping activations in FP32
