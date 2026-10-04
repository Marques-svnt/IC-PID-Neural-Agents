"""Neural network based controllers for non-linear process systems."""

from src.controllers.neural.nn_pid import (
    AdaptiveGainNetwork,
    NNPIDController,
    gains_parallel_to_standard,
    gains_standard_to_parallel,
)

__all__ = [
    "AdaptiveGainNetwork",
    "NNPIDController",
    "gains_parallel_to_standard",
    "gains_standard_to_parallel",
]
