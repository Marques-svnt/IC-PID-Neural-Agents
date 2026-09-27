"""Analytical tuning rules for PID controllers based on FOPTD reaction curves.

Implements classic Ziegler-Nichols, Cohen-Coon, and Skogestad SIMC formulae
with sign awareness for direct-acting and reverse-acting chemical processes.
"""

import logging
from enum import Enum
from typing import Optional

from src.classical_control.foptd import FOPTDModel
from src.classical_control.pid_controller import PIDGains

logger = logging.getLogger(__name__)


class TuningRule(str, Enum):
    """Analytical tuning methods for PID controllers."""

    ZIEGLER_NICHOLS_PID = "ziegler_nichols_pid"
    COHEN_COON_PID = "cohen_coon_pid"
    SKOGESTAD_SIMC_PID = "skogestad_simc_pid"
    SKOGESTAD_SIMC_PI = "skogestad_simc_pi"


def tune_analytical(
    model: FOPTDModel,
    rule: TuningRule,
    tau_c: Optional[float] = None,
    n_filter: float = 10.0,
) -> PIDGains:
    """Calculates PID gains using standard analytical rules from literature.

    Args:
        model: Identified or nominal FOPTD transfer function model.
        rule: TuningRule selection (Ziegler-Nichols, Cohen-Coon, or SIMC).
        tau_c: Desired closed-loop time constant for SIMC [min].
            If None, defaults to max(theta, 0.1*tau).
        n_filter: Derivative filter constant N.

    Returns:
        Configured PIDGains instance.

    Raises:
        ValueError: If model parameters violate stability assumptions.
    """
    k = model.k_p
    tau = model.tau
    # Guard against pure zero delay to avoid numerical division by zero
    theta = max(model.theta, 0.02 * tau)
    sign_k = 1.0 if k >= 0 else -1.0
    abs_k = abs(k)

    r_ratio = theta / tau

    if rule == TuningRule.ZIEGLER_NICHOLS_PID:
        # Ziegler-Nichols Open-Loop Reaction Curve PID
        kp = sign_k * (1.2 * tau) / (abs_k * theta)
        ti = 2.0 * theta
        td = 0.5 * theta

    elif rule == TuningRule.COHEN_COON_PID:
        # Cohen-Coon PID (better compensation for large theta/tau ratios)
        kp = sign_k * (tau / (abs_k * theta)) * ((4.0 / 3.0) + (r_ratio / 4.0))
        ti = theta * (32.0 + 6.0 * r_ratio) / (13.0 + 8.0 * r_ratio)
        td = theta * 4.0 / (11.0 + 2.0 * r_ratio)

    elif rule == TuningRule.SKOGESTAD_SIMC_PID:
        # Skogestad Simple IMC (SIMC) for PID
        tc = tau_c if tau_c is not None else max(theta, 0.1 * tau)
        kp = sign_k * (1.0 / abs_k) * (tau / (tc + theta))
        ti = min(tau, 4.0 * (tc + theta))
        td = 0.33 * theta

    elif rule == TuningRule.SKOGESTAD_SIMC_PI:
        # Skogestad Simple IMC (SIMC) for PI (no derivative action)
        tc = tau_c if tau_c is not None else max(theta, 0.1 * tau)
        kp = sign_k * (1.0 / abs_k) * (tau / (tc + theta))
        ti = min(tau, 4.0 * (tc + theta))
        td = 0.0

    else:
        raise ValueError(f"Unknown tuning rule: {rule}")

    gains = PIDGains(
        kp=float(kp),
        ti=float(ti),
        td=float(td),
        n_filter=float(n_filter),
    )

    logger.info(
        "Analytical Tuning [%s]: Kp=%.3f, Ti=%.3f min, Td=%.3f min",
        rule.value,
        gains.kp,
        gains.ti,
        gains.td,
    )

    return gains
