"""Nonlinear Continuous Stirred-Tank Reactor (CSTR) thermal system model.

This module provides the phenomenological CSTR model with jacket cooling
and first-order irreversible exothermic kinetics (A -> B), based on the
benchmark formulation from Bequette (2003).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from src.systems.base import SimulationResult, SystemParameters, ThermalSystem

logger = logging.getLogger(__name__)


@dataclass
class CSTRParameters(SystemParameters):
    """Physical and operational parameters for the jacketed CSTR.

    Attributes:
        v_reactor: Reactor liquid volume [L].
        v_jacket: Cooling jacket volume [L].
        q_feed: Process reactant feed volumetric flow rate [L/min].
        c_af: Nominal feed reactant concentration [mol/L].
        t_f0: Nominal reactant feed inlet temperature [K].
        t_jf: Coolant jacket feed inlet temperature [K].
        k0: Pre-exponential Arrhenius frequency factor [min^-1].
        e_over_r: Activation energy normalized by universal gas constant [K].
        delta_h: Heat of reaction [cal/mol].
        rho_cp: Liquid reactant volumetric heat capacity [cal/(L*K)].
        rho_j_cp_j: Coolant liquid volumetric heat capacity [cal/(L*K)].
        ua_nom: Nominal overall heat transfer coefficient product [cal/(min*K)].
        u_min: Minimum allowable coolant flow rate [L/min].
        u_max: Maximum allowable coolant flow rate [L/min].
        s_max: Maximum actuator slew rate |du/dt| [L/min^2].
        c_a_ss: Nominal steady-state reactant concentration [mol/L].
        t_ss: Nominal steady-state reactor temperature [K].
        t_j_ss: Nominal steady-state jacket temperature [K].
        q_j_ss: Nominal steady-state coolant flow rate [L/min].
    """

    v_reactor: float = 100.0
    v_jacket: float = 20.0
    q_feed: float = 100.0
    c_af: float = 1.0
    t_f0: float = 350.0
    t_jf: float = 300.0
    k0: float = 7.2e10
    e_over_r: float = 8750.0
    delta_h: float = -5.0e4
    rho_cp: float = 500.0
    rho_j_cp_j: float = 500.0
    ua_nom: float = 5.0e4
    u_min: float = 0.0
    u_max: float = 300.0
    s_max: float = 80.0
    c_a_ss: float = 0.0502
    t_ss: float = 396.65
    t_j_ss: float = 348.325
    q_j_ss: float = 100.0


class CSTRSystem(ThermalSystem):
    """Nonlinear Jacketed CSTR thermal system model.

    The state vector is defined as x = [C_A, T, T_j]^T:
        - C_A: Reactant concentration [mol/L]
        - T: Reactor mixture temperature [K]
        - T_j: Cooling jacket temperature [K]

    The manipulated control variable u is the coolant volumetric flow q_j [L/min].
    """

    def __init__(
        self,
        params: Optional[CSTRParameters] = None,
        dt: float = 0.02,
    ) -> None:
        """Initializes the CSTR model with specified parameters.

        Args:
            params: CSTRParameters dataclass. If None, default nominal values are used.
            dt: Sampling and integration step size [min].
        """
        cstr_params = params if params is not None else CSTRParameters()
        super().__init__(params=cstr_params, dt=dt)
        self.cstr_params = cstr_params

        logger.info(
            "CSTRSystem initialized | V=%.1f L, dt=%.3f min, T_ss=%.2f K",
            self.cstr_params.v_reactor,
            self.dt,
            self.cstr_params.t_ss,
        )

    def get_nominal_steady_state(self) -> NDArray[np.float64]:
        """Returns the nominal operating point state vector [C_A_ss, T_ss, T_j_ss].

        Returns:
            NumPy array of shape (3,) with steady-state values.
        """
        return np.array(
            [
                self.cstr_params.c_a_ss,
                self.cstr_params.t_ss,
                self.cstr_params.t_j_ss,
            ],
            dtype=np.float64,
        )

    def dynamics(
        self,
        t: float,
        state: NDArray[np.float64],
        u: float,
        t_f: Optional[float] = None,
        c_af: Optional[float] = None,
        ua: Optional[float] = None,
    ) -> NDArray[np.float64]:
        """Calculates state derivatives dx/dt for the CSTR system.

        Args:
            t: Current simulation time [min].
            state: Array [C_A, T, T_j] of current states.
            u: Actuator command (coolant flow rate q_j) [L/min].
            t_f: Reactant feed temperature disturbance [K]. If None, uses nominal t_f0.
            c_af: Feed concentration disturbance [mol/L]. If None, uses nominal c_af.
            ua: Overall heat transfer coefficient [cal/(min*K)]. If None, uses nominal ua_nom.

        Returns:
            Array [dC_A/dt, dT/dt, dT_j/dt] of state derivatives.
        """
        p = self.cstr_params
        c_a = max(0.0, float(state[0]))
        temp = max(100.0, float(state[1]))
        temp_j = max(100.0, float(state[2]))

        q_j = float(np.clip(u, p.u_min, p.u_max))
        feed_temp = float(t_f) if t_f is not None else p.t_f0
        feed_conc = float(c_af) if c_af is not None else p.c_af
        heat_transfer = float(ua) if ua is not None else p.ua_nom

        # Arrhenius kinetic rate coefficient
        reaction_rate = p.k0 * np.exp(-p.e_over_r / temp)

        # 1. Mass balance: dC_A/dt
        d_ca = (p.q_feed / p.v_reactor) * (feed_conc - c_a) - reaction_rate * c_a

        # 2. Reactor energy balance: dT/dt
        heat_generation = (-p.delta_h / p.rho_cp) * reaction_rate * c_a
        heat_exchange_reactor = (heat_transfer / (p.v_reactor * p.rho_cp)) * (
            temp - temp_j
        )
        convective_inflow = (p.q_feed / p.v_reactor) * (feed_temp - temp)
        d_temp = convective_inflow + heat_generation - heat_exchange_reactor

        # 3. Jacket energy balance: dT_j/dt
        convective_jacket = (q_j / p.v_jacket) * (p.t_jf - temp_j)
        heat_exchange_jacket = (heat_transfer / (p.v_jacket * p.rho_j_cp_j)) * (
            temp - temp_j
        )
        d_temp_j = convective_jacket + heat_exchange_jacket

        return np.array([d_ca, d_temp, d_temp_j], dtype=np.float64)

    def integrate_step(
        self,
        state: NDArray[np.float64],
        u: float,
        prev_u: Optional[float] = None,
        dt: Optional[float] = None,
        t_f: Optional[float] = None,
        c_af: Optional[float] = None,
        ua: Optional[float] = None,
    ) -> Tuple[NDArray[np.float64], float]:
        """Integrates system states over one discrete sampling interval dt.

        Applies actuator slew-rate limits and amplitude saturation.

        Args:
            state: Current state vector [C_A, T, T_j].
            u: Desired control action (coolant flow q_j) [L/min].
            prev_u: Previous actual control signal [L/min].
            dt: Integration interval [min]. Defaults to self.dt.
            t_f: Feed temperature disturbance [K].
            c_af: Feed concentration disturbance [mol/L].
            ua: Heat transfer coefficient [cal/(min*K)].

        Returns:
            Tuple of (next_state_vector, saturated_applied_control_signal).
        """
        step_dt = float(dt) if dt is not None else self.dt
        p = self.cstr_params

        # Slew-rate limiting
        if prev_u is not None:
            max_delta = p.s_max * step_dt
            delta_u = np.clip(u - prev_u, -max_delta, max_delta)
            u_command = prev_u + delta_u
        else:
            u_command = u

        # Amplitude saturation
        u_applied = float(np.clip(u_command, p.u_min, p.u_max))

        sol = solve_ivp(
            fun=lambda t_var, x_var: self.dynamics(
                t=t_var,
                state=x_var,
                u=u_applied,
                t_f=t_f,
                c_af=c_af,
                ua=ua,
            ),
            t_span=(0.0, step_dt),
            y0=state,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )

        next_state = np.array(sol.y[:, -1], dtype=np.float64)

        # Numerical safety clamping
        next_state[0] = max(0.0, float(next_state[0]))
        next_state[1] = np.clip(float(next_state[1]), 100.0, 800.0)
        next_state[2] = np.clip(float(next_state[2]), 100.0, 800.0)

        return next_state, u_applied

    def step_response(
        self,
        step_magnitude: float = 10.0,
        duration: float = 20.0,
    ) -> SimulationResult:
        """Simulates open-loop step response starting from nominal steady state.

        Args:
            step_magnitude: Step change in coolant flow rate q_j [L/min].
            duration: Simulation duration [min].

        Returns:
            SimulationResult object containing time series of states and control.
        """
        num_steps = int(duration / self.dt)
        time_vec = np.linspace(0.0, duration, num_steps + 1)
        state = self.get_nominal_steady_state()

        u_applied = self.cstr_params.q_j_ss + step_magnitude
        u_applied = float(
            np.clip(u_applied, self.cstr_params.u_min, self.cstr_params.u_max)
        )

        output_arr = np.zeros(num_steps + 1, dtype=np.float64)
        control_arr = np.zeros(num_steps + 1, dtype=np.float64)
        setpoint_arr = np.full(num_steps + 1, self.cstr_params.t_ss, dtype=np.float64)

        output_arr[0] = state[1]
        control_arr[0] = self.cstr_params.q_j_ss

        current_u = self.cstr_params.q_j_ss
        for i in range(num_steps):
            state, current_u = self.integrate_step(
                state=state,
                u=u_applied,
                prev_u=current_u,
            )
            output_arr[i + 1] = state[1]
            control_arr[i + 1] = current_u

        error_arr = setpoint_arr - output_arr

        return SimulationResult(
            time=time_vec,
            output=output_arr,
            control_signal=control_arr,
            setpoint=setpoint_arr,
            error=error_arr,
        )
