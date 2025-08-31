import os
import random
from typing import Optional

try:
    import numpy as np  # type: ignore
except Exception:  # numpy optional
    np = None  # type: ignore

import torch


def set_seed(seed: Optional[int] = None, deterministic: bool = False) -> int:
    """Set seeds for python, numpy, and torch for reproducibility.

    If no seed is provided, reads from env SEED or defaults to 42.
    Returns the seed used.
    """
    used_seed = seed if seed is not None else int(os.getenv("SEED", 42))

    random.seed(used_seed)
    if np is not None:
        np.random.seed(used_seed)  # type: ignore
    torch.manual_seed(used_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(used_seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return used_seed

