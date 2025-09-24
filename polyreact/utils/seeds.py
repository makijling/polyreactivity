"""Random seed utilities."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


@dataclass(slots=True)
class SeedState:
    """Record of RNG seeds applied across libraries."""

    python: int
    numpy: int
    torch: int | None = None


def set_global_seeds(seed: int) -> SeedState:
    """Seed ``random``, ``numpy`` and ``torch`` (if available)."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch_seed: int | None = None
    if "torch" in globals() and torch is not None:  # pragma: no branch
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - GPU specific
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch_seed = seed

    return SeedState(python=seed, numpy=seed, torch=torch_seed)
