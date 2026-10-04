"""Pytest configuration and shared fixtures for the PROIC project."""

import numpy as np
import pytest

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False


@pytest.fixture(autouse=True)
def set_random_seeds() -> None:
    """Ensure reproducibility by fixing random seeds in all tests."""
    np.random.seed(42)
    if _HAS_TORCH and torch is not None:
        torch.manual_seed(42)
