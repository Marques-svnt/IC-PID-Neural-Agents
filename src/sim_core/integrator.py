"""Numerical integrator for stiff and non-linear chemical process simulations."""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from src.sim_core.cstr_plant import CSTRPlant, CSTRState

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Container for simulation trajectories.

    Attributes:
        t: Array of time points [min].
        states: Array of state trajectories shape (N, 3) [C_A, T, T_j].
        u: Array of applied control inputs [L/min].
        T_measured: Array of measured reactor temperature with noise [K].
    """

    t: np.ndarray
    states: np.ndarray
    u: np.ndarray
    T_measured: np.ndarray


class NumericalIntegrator:
    """Handles continuous-time integration of CSTR dynamics across discrete control intervals."""

    def __init__(self, method: str = "RK45", rtol: float = 1e-6, atol: float = 1e-8) -> None:
        """Initializes the numerical integrator.

        Args:
            method: Scipy integration method ('RK45', 'Radau', 'BDF', 'LSODA').
            rtol: Relative error tolerance.
            atol: Absolute error tolerance.
        """
        self.method = method
        self.rtol = rtol
        self.atol = atol
        logger.debug("NumericalIntegrator initialized with method=%s", method)

    def step(
        self,
        plant: CSTRPlant,
        current_state: CSTRState,
        q_j: float,
        t_current: float,
        dt: float,
    ) -> CSTRState:
        """Integrates plant dynamics across a single sampling interval dt.

        Args:
            plant: CSTR plant model instance.
            current_state: Initial state at t_current.
            q_j: Constant coolant flow applied over interval [t_current, t_current + dt].
            t_current: Start time in minutes.
            dt: Sampling period in minutes.

        Returns:
            Updated CSTRState at t_current + dt.

        Raises:
            RuntimeError: If integration fails or produces non-finite states.
        """
        y0 = current_state.to_array()
        t_span = (t_current, t_current + dt)

        sol = solve_ivp(
            fun=lambda t, y: plant.derivatives(t, y, q_j),
            t_span=t_span,
            y0=y0,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )

        if not sol.success:
            logger.error("Integration step failed at t=%.2f: %s", t_current, sol.message)
            raise RuntimeError(f"ODE integration failure: {sol.message}")

        y_next = sol.y[:, -1]
        if np.any(np.isnan(y_next)) or np.any(np.isinf(y_next)):
            raise RuntimeError(f"Non-finite state encountered at t={t_current + dt}: {y_next}")

        return CSTRState.from_array(y_next)

    def simulate_open_loop(
        self,
        plant: CSTRPlant,
        initial_state: CSTRState,
        t_span: tuple[float, float],
        dt: float,
        u_func: Callable[[float], float],
        noise_std: float = 0.0,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """Simulates open-loop trajectory with user-defined control profile.

        Args:
            plant: CSTR plant model instance.
            initial_state: State at t = t_span[0].
            t_span: (t_start, t_end) in minutes.
            dt: Sampling interval in minutes.
            u_func: Function mapping t -> q_j(t).
            noise_std: Standard deviation of additive Gaussian noise on temperature [K].
            seed: Optional random seed for noise reproducibility.

        Returns:
            SimulationResult containing all computed trajectories.
        """
        rng = np.random.default_rng(seed)
        t_points = np.arange(t_span[0], t_span[1] + 1e-9, dt)
        n_steps = len(t_points)

        states = np.zeros((n_steps, 3), dtype=np.float64)
        u_vals = np.zeros(n_steps, dtype=np.float64)
        T_meas = np.zeros(n_steps, dtype=np.float64)

        current_state = initial_state
        states[0] = current_state.to_array()
        u_vals[0] = u_func(t_points[0])
        noise_0 = rng.normal(0.0, noise_std) if noise_std > 0 else 0.0
        T_meas[0] = current_state.T + noise_0

        for i in range(n_steps - 1):
            t_now = t_points[i]
            q_j = u_func(t_now)
            u_vals[i] = q_j

            current_state = self.step(plant, current_state, q_j, t_now, dt)
            states[i + 1] = current_state.to_array()

            noise_i = rng.normal(0.0, noise_std) if noise_std > 0 else 0.0
            T_meas[i + 1] = current_state.T + noise_i

        u_vals[-1] = u_func(t_points[-1])

        return SimulationResult(
            t=t_points,
            states=states,
            u=u_vals,
            T_measured=T_meas,
        )
