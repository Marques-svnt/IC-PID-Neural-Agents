"""Mathematical performance criteria for closed-loop benchmark analysis."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ControlPerformanceMetrics:
    """Benchmark quantitative evaluation metrics for control loop performance.

    Attributes:
        iae: Integral of Absolute Error (total tracking deviation).
        itae: Integral of Time-weighted Absolute Error (penalizes lingering oscillations).
        ise: Integral of Squared Error (penalizes large transients).
        tv: Total Variation of the control action (actuator wear metric).
        overshoot_pct: Maximum peak overshoot percentage [%].
        settling_time: Time taken to enter and remain within +/- 2% band [min].
        rise_time: Time taken to rise from 10% to 90% of setpoint step [min].
    """

    iae: float
    itae: float
    ise: float
    tv: float
    overshoot_pct: float
    settling_time: Optional[float]
    rise_time: Optional[float]


def evaluate_performance(
    t: np.ndarray,
    y: np.ndarray,
    setpoint: np.ndarray,
    u: np.ndarray,
    settling_band: float = 0.02,
) -> ControlPerformanceMetrics:
    """Computes standard academic and industrial control performance criteria.

    Args:
        t: Array of time points [min] (monotonically increasing).
        y: Array of controlled process variable values (PV).
        setpoint: Array of target setpoint values (SP).
        u: Array of manipulated control actions (MV).
        settling_band: Fractional band for settling time calculation (default 2% or 0.02).

    Returns:
        Populated ControlPerformanceMetrics dataclass.

    Raises:
        ValueError: If array dimensions do not match or have fewer than 2 elements.
    """
    n = len(t)
    if not (len(y) == len(setpoint) == len(u) == n):
        raise ValueError(
            f"Array length mismatch: t={len(t)}, y={len(y)}, sp={len(setpoint)}, u={len(u)}"
        )
    if n < 2:
        raise ValueError("At least 2 points are required to calculate performance metrics")

    dt = np.diff(t)
    error = setpoint - y
    abs_error = np.abs(error)

    # Trapezoidal numerical integration for time-domain criteria
    iae = float(np.sum(0.5 * (abs_error[:-1] + abs_error[1:]) * dt))
    itae = float(np.sum(0.5 * (t[:-1] * abs_error[:-1] + t[1:] * abs_error[1:]) * dt))
    ise = float(np.sum(0.5 * (error[:-1] ** 2 + error[1:] ** 2) * dt))

    # Total Variation (TV) of control effort: TV = sum |u_k - u_{k-1}|
    tv = float(np.sum(np.abs(np.diff(u))))

    # Overshoot calculation
    sp_final = setpoint[-1]
    y_init = y[0]
    step_amplitude = abs(sp_final - y_init)

    if step_amplitude > 1e-6:
        if sp_final > y_init:
            peak = np.max(y)
            overshoot = max(0.0, (peak - sp_final) / step_amplitude * 100.0)
        else:
            trough = np.min(y)
            overshoot = max(0.0, (sp_final - trough) / step_amplitude * 100.0)
    else:
        overshoot = 0.0

    # Settling time (time after which |y - sp_final| <= band * step_amplitude)
    settling_threshold = max(settling_band * step_amplitude, 1e-4)
    outside_band_indices = np.where(np.abs(y - sp_final) > settling_threshold)[0]

    if len(outside_band_indices) == 0:
        settling_time = float(t[0])
    elif outside_band_indices[-1] == n - 1:
        # Never settled inside the band
        settling_time = None
    else:
        settling_time = float(t[outside_band_indices[-1] + 1])

    # Rise time (10% to 90% of transition)
    rise_time: Optional[float] = None
    if step_amplitude > 1e-6:
        val_10 = y_init + 0.10 * (sp_final - y_init)
        val_90 = y_init + 0.90 * (sp_final - y_init)

        if sp_final > y_init:
            idx_10 = np.where(y >= val_10)[0]
            idx_90 = np.where(y >= val_90)[0]
        else:
            idx_10 = np.where(y <= val_10)[0]
            idx_90 = np.where(y <= val_90)[0]

        if len(idx_10) > 0 and len(idx_90) > 0 and idx_90[0] >= idx_10[0]:
            rise_time = float(t[idx_90[0]] - t[idx_10[0]])

    logger.debug(
        "Performance calculated: IAE=%.3f, ITAE=%.3f, TV=%.3f, Overshoot=%.1f%%",
        iae,
        itae,
        tv,
        overshoot,
    )

    return ControlPerformanceMetrics(
        iae=iae,
        itae=itae,
        ise=ise,
        tv=tv,
        overshoot_pct=float(overshoot),
        settling_time=settling_time,
        rise_time=rise_time,
    )
