import torch
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4000"

torch.cuda.empty_cache()

# torch.cuda.set_per_process_memory_fraction(0.9, device=0)

torch.cuda.memory_summary(device=None, abbreviated=False)

print(torch.cuda.is_available())