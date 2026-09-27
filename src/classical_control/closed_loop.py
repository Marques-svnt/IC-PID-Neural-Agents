"""Closed-loop CSTR simulation engine with PID control and perturbation injection."""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from src.classical_control.pid_controller import PIDController
from src.evaluation.metrics import ControlPerformanceMetrics, evaluate_performance
from src.sim_core.cstr_plant import CSTRPlant, CSTRState
from src.sim_core.integrator import NumericalIntegrator

logger = logging.getLogger(__name__)


@dataclass
class ClosedLoopResult:
    """Simulation trajectory results for closed-loop evaluation.

    Attributes:
        t: Array of time points [min].
        setpoint: Target trajectory values [K].
        t_pv: Measured reactor temperature trajectory [K].
        states: Full state trajectory matrix (N, 3) [C_A, T, T_j].
        u_applied: Commanded physical flow rate to cooling jacket [L/min].
        u_unconstrained: Unbounded controller output before saturation [L/min].
        metrics: Academic benchmark performance metrics.
    """

    t: np.ndarray
    setpoint: np.ndarray
    t_pv: np.ndarray
    states: np.ndarray
    u_applied: np.ndarray
    u_unconstrained: np.ndarray
    metrics: ControlPerformanceMetrics


class ClosedLoopSimulator:
    """Simulates closed-loop feedback control of the CSTR plant."""

    def __init__(self, integrator: Optional[NumericalIntegrator] = None) -> None:
        """Initializes the closed-loop simulator.

        Args:
            integrator: Numerical integration engine. Defaults to RK45.
        """
        self.integrator = integrator or NumericalIntegrator(method="RK45")

    def run(
        self,
        plant: CSTRPlant,
        controller: PIDController,
        initial_state: CSTRState,
        t_span: tuple[float, float],
        dt: float,
        setpoint_func: Callable[[float], float],
        noise_std: float = 0.0,
        seed: Optional[int] = None,
    ) -> ClosedLoopResult:
        """Executes closed-loop simulation over time span.

        Args:
            plant: CSTR physical plant model.
            controller: Configured PID controller.
            initial_state: Plant starting state at t_span[0].
            t_span: Tuple of (t_start, t_end) in minutes.
            dt: Sampling period in minutes.
            setpoint_func: Callable mapping time t -> setpoint temperature [K].
            noise_std: Standard deviation of sensor noise [K].
            seed: Optional seed for reproducible noise generation.

        Returns:
            ClosedLoopResult with complete time-series and computed metrics.
        """
        rng = np.random.default_rng(seed)
        t_points = np.arange(t_span[0], t_span[1] + 1e-9, dt)
        n_steps = len(t_points)

        states = np.zeros((n_steps, 3), dtype=np.float64)
        sp_arr = np.zeros(n_steps, dtype=np.float64)
        t_pv_arr = np.zeros(n_steps, dtype=np.float64)
        u_app_arr = np.zeros(n_steps, dtype=np.float64)
        u_uncon_arr = np.zeros(n_steps, dtype=np.float64)

        current_state = initial_state
        controller.reset(initial_measurement=current_state.T, initial_u=controller.u_bias)

        logger.info(
            "Starting closed-loop simulation: t_span=%s, dt=%.3f min, steps=%d",
            t_span,
            dt,
            n_steps,
        )

        for i in range(n_steps):
            t_now = t_points[i]
            sp = setpoint_func(t_now)
            sp_arr[i] = sp
            states[i] = current_state.to_array()

            noise = rng.normal(0.0, noise_std) if noise_std > 0.0 else 0.0
            pv_measured = current_state.T + noise
            t_pv_arr[i] = pv_measured

            # Compute PID action
            u_act, diag = controller.compute(setpoint=sp, measurement=pv_measured, dt=dt)
            u_app_arr[i] = u_act
            u_uncon_arr[i] = diag["u_unconstrained"]

            # Advance plant state across interval dt
            if i < n_steps - 1:
                current_state = self.integrator.step(
                    plant=plant,
                    current_state=current_state,
                    q_j=u_act,
                    t_current=t_now,
                    dt=dt,
                )

        # Evaluate performance metrics across full trajectory
        metrics = evaluate_performance(
            t=t_points,
            y=t_pv_arr,
            setpoint=sp_arr,
            u=u_app_arr,
        )

        logger.info(
            "Closed-loop complete: IAE=%.2f, ITAE=%.2f, TV=%.2f, Mp=%.1f%%",
            metrics.iae,
            metrics.itae,
            metrics.tv,
            metrics.overshoot_pct,
        )

        return ClosedLoopResult(
            t=t_points,
            setpoint=sp_arr,
            t_pv=t_pv_arr,
            states=states,
            u_applied=u_app_arr,
            u_unconstrained=u_uncon_arr,
            metrics=metrics,
        )
