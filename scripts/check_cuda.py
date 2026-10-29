"""Check CUDA initialization inside an allocated GPU container."""

import torch


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the allocated container.")
    print("CUDA preflight:", torch.cuda.get_device_name(0))
