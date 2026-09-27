"""Numerical optimization-based PID tuning subject to robustness constraints (Ms).

Directly minimizes integral performance criteria (ITAE, IAE, ISE) evaluated over
the non-linear plant, subject to maximum sensitivity peak constraints Ms <= 1.6.
"""

import logging
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from src.classical_control.closed_loop import ClosedLoopSimulator
from src.classical_control.pid_controller import AntiWindupMethod, PIDController, PIDGains
from src.sim_core.actuators import ActuatorLimits
from src.sim_core.cstr_plant import CSTRPlant, CSTRState

logger = logging.getLogger(__name__)


class OptimizationCriterion(str, Enum):
    """Performance criteria for numerical optimization."""

    ITAE = "itae"
    IAE = "iae"
    ISE = "ise"


def calculate_maximum_sensitivity(
    plant: CSTRPlant,
    nominal_state: CSTRState,
    q_j_ss: float,
    gains: PIDGains,
    omega_range: Tuple[float, float] = (1e-3, 1e2),
    n_freqs: int = 300,
) -> float:
    """Calculates peak sensitivity Ms = max_w |1 / (1 + L(jw))| from linearized model.

    Args:
        plant: CSTR plant model instance.
        nominal_state: Steady-state operating point.
        q_j_ss: Steady-state coolant flow rate.
        gains: PID gains candidate.
        omega_range: (w_min, w_max) in rad/min.
        n_freqs: Number of logarithmically spaced frequency sample points.

    Returns:
        Maximum sensitivity peak Ms (dimensionless). Typically 1.2 to 2.0.
    """
    A, B = plant.linearize(nominal_state, q_j_ss)
    C = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)  # Measure reactor temperature T

    omegas = np.logspace(np.log10(omega_range[0]), np.log10(omega_range[1]), n_freqs)
    sensitivities = np.zeros(n_freqs, dtype=np.float64)

    tau_f = gains.td / gains.n_filter if gains.td > 0 else 0.0

    for i, w in enumerate(omegas):
        s = 1j * w

        # 1. Linearized plant frequency response G(jw) = C (sI - A)^(-1) B
        sI_minus_A = s * np.eye(3) - A
        try:
            inv_sI_minus_A = np.linalg.inv(sI_minus_A)
            G_jw = complex((C @ inv_sI_minus_A @ B)[0, 0])
        except np.linalg.LinAlgError:
            return 1e3

        # 2. Parallel PID controller frequency response C(jw)
        p_term = gains.kp
        i_term = gains.kp / (s * gains.ti)
        d_term = (gains.kp * gains.td * s) / (1.0 + tau_f * s) if gains.td > 0 else 0.0
        C_pid = p_term + i_term + d_term

        # 3. Open loop L(jw) = G(jw) * C_pid(jw)
        L_jw = G_jw * C_pid

        # 4. Sensitivity S(jw) = 1 / (1 + L(jw))
        S_jw = 1.0 / (1.0 + L_jw)
        sensitivities[i] = abs(S_jw)

    ms = float(np.max(sensitivities))
    return ms


def tune_by_optimization(
    plant: CSTRPlant,
    nominal_state: CSTRState,
    q_j_ss: float,
    target_dT: float = -2.0,
    initial_gains: Optional[PIDGains] = None,
    criterion: OptimizationCriterion = OptimizationCriterion.ITAE,
    max_ms: float = 1.6,
    t_sim: float = 8.0,
    dt: float = 0.02,
) -> PIDGains:
    """Finds optimal PID gains by minimizing closed-loop integral error subject to Ms <= max_ms.

    Args:
        plant: CSTR plant model instance.
        nominal_state: Starting equilibrium point.
        q_j_ss: Nominal steady-state actuator command.
        target_dT: Step change in temperature setpoint [K].
        initial_gains: Starting point for optimization. If None, uses conservative defaults.
        criterion: Objective criterion to minimize (ITAE, IAE, ISE).
        max_ms: Maximum permissible sensitivity peak (default 1.6 for robustness).
        t_sim: Simulation duration in minutes for each objective function evaluation.
        dt: Simulation time step in minutes.

    Returns:
        Optimal tuned PIDGains dataclass.
    """
    if initial_gains is None:
        # Default starting point: moderate proportional gain, reasonable Ti and Td
        p0 = np.array([-5.0, 1.5, 0.2], dtype=np.float64)
    else:
        p0 = np.array([initial_gains.kp, initial_gains.ti, initial_gains.td], dtype=np.float64)

    target_T = nominal_state.T + target_dT
    sim = ClosedLoopSimulator()
    limits = ActuatorLimits(u_min=0.0, u_max=300.0, max_slew_rate=100.0)

    def objective(params: np.ndarray) -> float:
        kp, ti, td = float(params[0]), float(params[1]), float(params[2])

        # Physical viability bounds
        if kp >= 0.0 or ti <= 0.05 or td < 0.0:
            return 1e6

        candidate_gains = PIDGains(kp=kp, ti=ti, td=td)

        # 1. Robustness check via maximum sensitivity Ms
        ms = calculate_maximum_sensitivity(plant, nominal_state, q_j_ss, candidate_gains)
        ms_penalty = 1e4 * max(0.0, ms - max_ms) ** 2 if ms > max_ms else 0.0

        # 2. Closed-loop simulation
        pid = PIDController(
            gains=candidate_gains,
            actuator_limits=limits,
            anti_windup=AntiWindupMethod.CLAMPING,
            u_bias=q_j_ss,
        )

        try:
            res = sim.run(
                plant=plant,
                controller=pid,
                initial_state=nominal_state,
                t_span=(0.0, t_sim),
                dt=dt,
                setpoint_func=lambda t: target_T,
            )
        except Exception:
            return 1e6

        if criterion == OptimizationCriterion.ITAE:
            perf = res.metrics.itae
        elif criterion == OptimizationCriterion.IAE:
            perf = res.metrics.iae
        else:
            perf = res.metrics.ise

        # Check final error to prevent settling offset
        final_err = abs(res.t_pv[-1] - target_T)
        offset_penalty = 1e3 * final_err if final_err > 0.5 else 0.0

        total_cost = perf + ms_penalty + offset_penalty
        return float(total_cost)

    logger.info(
        "Starting PID optimization: criterion=%s (max_ms=%.2f)",
        criterion.value,
        max_ms,
    )

    res = minimize(
        fun=objective,
        x0=p0,
        method="Nelder-Mead",
        options={"maxiter": 120, "xatol": 1e-2, "fatol": 1e-2, "disp": False},
    )

    opt_kp, opt_ti, opt_td = float(res.x[0]), float(res.x[1]), float(res.x[2])
    opt_gains = PIDGains(kp=opt_kp, ti=max(opt_ti, 0.05), td=max(opt_td, 0.0))
    final_ms = calculate_maximum_sensitivity(plant, nominal_state, q_j_ss, opt_gains)

    logger.info(
        "Optimal PID found: Kp=%.3f, Ti=%.3f min, Td=%.3f min (Ms=%.2f, Cost=%.2f)",
        opt_gains.kp,
        opt_gains.ti,
        opt_gains.td,
        final_ms,
        res.fun,
    )

    return opt_gains
