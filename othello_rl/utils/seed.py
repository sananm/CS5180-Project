import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True):
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value
        deterministic: If True, enforce deterministic CUDA algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
