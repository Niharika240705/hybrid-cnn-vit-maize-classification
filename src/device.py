import torch


def select_device(requested_device="auto"):
    """Select the requested accelerator, falling back safely when unavailable."""
    requested = str(requested_device or "auto").lower()

    if requested in {"cuda", "auto"} and torch.cuda.is_available():
        return torch.device("cuda")

    if requested in {"mps", "auto"} and torch.backends.mps.is_available():
        return torch.device("mps")

    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(
            f"Unsupported device '{requested_device}'. Use 'cuda', 'mps', 'cpu', or 'auto'."
        )

    return torch.device("cpu")
