"""Unit tests for performance metrics computation."""

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics


def test_perfect_control_has_zero_errors() -> None:
    """Perfect tracking (zero error) should yield ISE=IAE=ITAE=0."""
    t = np.linspace(0, 100, 1000)
    sp = np.ones_like(t) * 50.0
    output = sp.copy()
    error = sp - output
    metrics = compute_metrics(t, error, output, sp)
    assert metrics.ise == pytest.approx(0.0, abs=1e-9)
    assert metrics.iae == pytest.approx(0.0, abs=1e-9)
    assert metrics.itae == pytest.approx(0.0, abs=1e-9)


def test_mismatched_arrays_raise_value_error() -> None:
    """Mismatched array lengths should raise ValueError."""
    t = np.linspace(0, 10, 100)
    with pytest.raises(ValueError):
        compute_metrics(t, np.zeros(50), np.zeros(100), np.zeros(100))
