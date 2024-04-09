import torch


def conv_str_to_dtype(s: str) -> torch.dtype:
    if s == "torch.float32":
        return torch.float32
    elif s == "torch.bfloat16":
        return torch.bfloat16
    elif s == "torch.float16":
        return torch.float16
    raise ValueError(f"Unknown dtype {s}")
