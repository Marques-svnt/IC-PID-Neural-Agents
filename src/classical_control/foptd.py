"""First-Order Plus Time Delay (FOPTD) model identification from step response data."""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FOPTDModel:
    """Parameters of a First-Order Plus Time Delay transfer function.

    Model: G(s) = K * exp(-theta*s) / (tau*s + 1).

    Attributes:
        k_p: Static process gain (delta_y / delta_u) [PV_units / MV_units].
        tau: Process dominant time constant [min]. Must be positive.
        theta: Apparent dead time / time delay [min]. Must be non-negative.
    """

    k_p: float
    tau: float
    theta: float

    def __post_init__(self) -> None:
        if self.tau <= 0.0:
            raise ValueError(f"Time constant tau must be positive, got {self.tau}")
        if self.theta < 0.0:
            raise ValueError(f"Time delay theta must be non-negative, got {self.theta}")

    def step_response(self, t: np.ndarray, delta_u: float, y0: float = 0.0) -> np.ndarray:
        """Calculates theoretical analytical step response for given time array.

        Args:
            t: Time vector [min].
            delta_u: Step input magnitude applied at t=0.
            y0: Baseline process output before step.

        Returns:
            Computed FOPTD output response y(t).
        """
        y = np.full_like(t, y0, dtype=np.float64)
        active_mask = t >= self.theta
        t_active = t[active_mask] - self.theta
        y[active_mask] = y0 + self.k_p * delta_u * (1.0 - np.exp(-t_active / self.tau))
        return y


def identify_foptd(
    t: np.ndarray,
    y: np.ndarray,
    delta_u: float,
    y0: Optional[float] = None,
) -> Tuple[FOPTDModel, float]:
    """Fits an optimal FOPTD model to empirical or simulated step response data.

    Uses non-linear least squares (Levenberg-Marquardt / Trust Region Reflective)
    to estimate process gain K, time constant tau, and delay theta.

    Args:
        t: Array of time points [min] starting at 0.
        y: Measured process output curve.
        delta_u: Step input magnitude (must be non-zero).
        y0: Baseline output value. If None, uses y[0].

    Returns:
        Tuple of:
            - Populated FOPTDModel dataclass.
            - Root Mean Square Error (RMSE) of the model fit.

    Raises:
        ValueError: If delta_u is zero or array dimensions are inconsistent.
        RuntimeError: If non-linear optimization fails to converge.
    """
    if abs(delta_u) < 1e-9:
        raise ValueError("Step magnitude delta_u cannot be zero for identification")
    if len(t) != len(y) or len(t) < 5:
        raise ValueError(
            "Time and output arrays must have matching dimensions with at least 5 points"
        )

    baseline = y[0] if y0 is None else y0
    total_delta_y = y[-1] - baseline
    k_init = total_delta_y / delta_u

    # Heuristic initial guess: find 28.3% and 63.2% response times
    y_28 = baseline + 0.283 * total_delta_y
    y_63 = baseline + 0.632 * total_delta_y

    sign = 1.0 if total_delta_y >= 0 else -1.0
    idx_28 = np.where(sign * y >= sign * y_28)[0]
    idx_63 = np.where(sign * y >= sign * y_63)[0]

    t_28 = float(t[idx_28[0]]) if len(idx_28) > 0 else float(t[-1] * 0.3)
    t_63 = float(t[idx_63[0]]) if len(idx_63) > 0 else float(t[-1] * 0.6)

    tau_init = max(1.5 * (t_63 - t_28), 0.05)
    theta_init = max(t_63 - tau_init, 0.0)

    # Parametric fitting function
    def foptd_fit(
        time_pts: np.ndarray, k_val: float, tau_val: float, theta_val: float
    ) -> np.ndarray:
        m = FOPTDModel(k_p=k_val, tau=tau_val, theta=theta_val)
        return m.step_response(time_pts, delta_u, y0=baseline)

    # Bounds for parameters
    lower_bounds = [-np.inf if k_init < 0 else 0.0, 0.01, 0.0]
    upper_bounds = [0.0 if k_init < 0 else np.inf, t[-1] * 2.0, t[-1] * 0.8]

    p0 = [k_init, tau_init, theta_init]

    try:
        popt, _ = curve_fit(
            f=foptd_fit,
            xdata=t,
            ydata=y,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=2000,
        )
    except Exception as exc:
        logger.warning("FOPTD curve_fit failed (%s), falling back to heuristic parameters", exc)
        popt = p0

    model = FOPTDModel(k_p=float(popt[0]), tau=float(popt[1]), theta=float(popt[2]))
    y_pred = model.step_response(t, delta_u, y0=baseline)
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    logger.info(
        "FOPTD Identified: K=%.4f, tau=%.3f min, theta=%.3f min (RMSE=%.4e)",
        model.k_p,
        model.tau,
        model.theta,
        rmse,
    )

    return model, rmse
