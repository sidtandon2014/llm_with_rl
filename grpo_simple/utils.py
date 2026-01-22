# References: https://huggingface.co/blog/train_memory

import torch
from transformers import AutoModelForCausalLM

DEVICE = "cuda:2"
MODEL_DTYPE = torch.bfloat16
INPUT_DTYPE = torch.int64
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=MODEL_DTYPE).to(DEVICE)

activation_sizes = []

def forward_hook(module, input, output):
    """
    Hook to calculate activation size for each module.
    """
    if isinstance(output, torch.Tensor):
        activation_sizes.append(output.numel() * output.element_size())
    elif isinstance(output, (tuple, list)):
        for tensor in output:
            if isinstance(tensor, torch.Tensor):
                activation_sizes.append(tensor.numel() * tensor.element_size())

# Register hooks for each submodule
hooks = []
for submodule in model.modules():
    hooks.append(submodule.register_forward_hook(forward_hook))

# Perform a forward pass with a dummy input
dummy_input = torch.zeros((1, 1), dtype=INPUT_DTYPE, device=DEVICE)
model.eval()  # No gradients needed for memory measurement
with torch.no_grad():
    model(dummy_input)

# Clean up hooks
for hook in hooks:
    hook.remove()

print(sum(activation_sizes))  # Output: 5065216
