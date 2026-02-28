import torch
from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", dtype=torch.float16)
layer = model.model.layers[0]

torch.manual_seed(42)
x = torch.randn(1, 128, 4096, dtype=torch.float16)

# Save input
x.detach().numpy().tofile("test_data/rmsnorm_input.bin")

# Save weights
layer.input_layernorm.weight.data.numpy().tofile("test_data/rmsnorm_weight.bin")

# Save expected output
normed = layer.input_layernorm(x)
normed.detach().numpy().tofile("test_data/rmsnorm_output.bin")