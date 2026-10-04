"""Performance metrics for control system evaluation."""

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Standard control performance metrics.

    Attributes:
        ise: Integral Squared Error  [error_unit^2 * s].
        iae: Integral Absolute Error [error_unit * s].
        itae: Integral Time-weighted Absolute Error [error_unit * s^2].
        overshoot_pct: Maximum overshoot relative to setpoint step [%].
        rise_time: Time to first reach 90% of setpoint change [s].
        settling_time: Time to remain within ±2% of setpoint [s].
        steady_state_error: Final error at end of simulation [error_unit].
    """

    ise: float
    iae: float
    itae: float
    overshoot_pct: float
    rise_time: float
    settling_time: float
    steady_state_error: float

    def __str__(self) -> str:
        return (
            f"ISE={self.ise:.4f} | IAE={self.iae:.4f} | ITAE={self.itae:.4f} | "
            f"OS={self.overshoot_pct:.2f}% | tr={self.rise_time:.2f}s | "
            f"ts={self.settling_time:.2f}s | ess={self.steady_state_error:.4f}"
        )


def compute_metrics(
    time: NDArray[np.float64],
    error: NDArray[np.float64],
    output: NDArray[np.float64],
    setpoint: NDArray[np.float64],
    tolerance_pct: float = 2.0,
) -> PerformanceMetrics:
    """Compute standard control performance metrics from simulation data.

    Args:
        time: Time vector [s].
        error: Tracking error e(t) = setpoint - output [same units].
        output: System output y(t).
        setpoint: Reference signal r(t).
        tolerance_pct: Settling band as percentage of final setpoint value.

    Returns:
        PerformanceMetrics dataclass with all computed metrics.

    Raises:
        ValueError: If input arrays have inconsistent lengths.
    """
    if not (len(time) == len(error) == len(output) == len(setpoint)):
        raise ValueError("All input arrays must have the same length.")

    step_size = float(setpoint[-1] - setpoint[0])

    # Integral metrics (NumPy 2.0+ compatibility)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    ise = float(trapz_fn(error**2, time))
    iae = float(trapz_fn(np.abs(error), time))
    itae = float(trapz_fn(time * np.abs(error), time))

    # Overshoot
    if step_size != 0:
        peak = float(np.max(output) if step_size > 0 else np.min(output))
        overshoot_pct = max(0.0, (peak - setpoint[-1]) / abs(step_size) * 100)
    else:
        overshoot_pct = 0.0

    # Rise time (10% -> 90% of step, or first crossing of 90%)
    target_90 = setpoint[0] + 0.9 * step_size
    crossed = (
        np.where(output >= target_90)[0]
        if step_size > 0
        else np.where(output <= target_90)[0]
    )
    rise_time = float(time[crossed[0]]) if len(crossed) > 0 else float("nan")

    # Settling time (±tolerance band around final setpoint)
    tol = abs(step_size) * tolerance_pct / 100.0
    within_band = np.abs(output - setpoint[-1]) <= tol
    # Find last time output was OUTSIDE the band
    outside = np.where(~within_band)[0]
    settling_time = float(time[outside[-1]]) if len(outside) > 0 else 0.0

    # Steady-state error (mean of last 10% of simulation)
    tail_idx = int(0.9 * len(error))
    steady_state_error = float(np.mean(error[tail_idx:]))

    metrics = PerformanceMetrics(
        ise=ise,
        iae=iae,
        itae=itae,
        overshoot_pct=overshoot_pct,
        rise_time=rise_time,
        settling_time=settling_time,
        steady_state_error=steady_state_error,
    )
    logger.info("Metrics computed: %s", metrics)
    return metrics
