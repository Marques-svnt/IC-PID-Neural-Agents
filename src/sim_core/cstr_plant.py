"""Non-isothermal Continuous Stirred Tank Reactor (CSTR) plant model.

Implements the standard benchmark ODEs with non-linear Arrhenius kinetics
and thermal cooling jacket dynamics, per Seborg et al. and Ogunnaike.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import root

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CSTRParameters:
    """Phenomenological parameters for the non-isothermal CSTR.

    Attributes:
        q: Process stream volumetric flow rate [L/min].
        V: Reactor fluid volume [L].
        k_0: Arrhenius pre-exponential kinetic constant [1/min].
        E_over_R: Activation energy divided by universal gas constant [K].
        delta_H: Heat of reaction [cal/mol] (negative for exothermic).
        rho_cp: Reactor mixture volumetric heat capacity [cal/(L*K)].
        UA: Overall heat transfer coefficient times area [cal/(min*K)].
        V_j: Cooling jacket volume [L].
        rho_j_cp_j: Cooling jacket fluid volumetric heat capacity [cal/(L*K)].
        C_Af: Feed reactant concentration [mol/L].
        T_f: Feed stream temperature [K].
        T_jf: Coolant supply inlet temperature [K].
    """

    q: float = 100.0
    V: float = 100.0
    k_0: float = 7.2e10
    E_over_R: float = 8750.0
    delta_H: float = -5.0e4
    rho_cp: float = 500.0
    UA: float = 5.0e4
    V_j: float = 20.0
    rho_j_cp_j: float = 500.0
    C_Af: float = 1.0
    T_f: float = 350.0
    T_jf: float = 300.0


@dataclass
class CSTRState:
    """State vector of the CSTR.

    Attributes:
        C_A: Reactant concentration in reactor [mol/L].
        T: Reactor temperature [K].
        T_j: Cooling jacket temperature [K].
    """

    C_A: float
    T: float
    T_j: float

    def to_array(self) -> np.ndarray:
        """Converts state to 1D numpy array [C_A, T, T_j]."""
        return np.array([self.C_A, self.T, self.T_j], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "CSTRState":
        """Builds state from 1D numpy array."""
        return cls(C_A=float(arr[0]), T=float(arr[1]), T_j=float(arr[2]))


class CSTRPlant:
    """Non-linear continuous stirred tank reactor with first-order exothermic reaction."""

    def __init__(self, params: Optional[CSTRParameters] = None) -> None:
        """Initializes the CSTR plant with thermodynamic parameters.

        Args:
            params: Plant parameter specification. Defaults to standard literature benchmark.
        """
        self.params = params or CSTRParameters()
        logger.info(
            "CSTRPlant initialized: V=%.1f L, q=%.1f L/min, UA=%.1f cal/(min*K)",
            self.params.V,
            self.params.q,
            self.params.UA,
        )

    def reaction_rate(self, C_A: float, T: float) -> float:
        """Calculates reaction consumption rate r_A = k(T) * C_A.

        Args:
            C_A: Reactant concentration [mol/L].
            T: Absolute temperature [K].

        Returns:
            Reaction rate in [mol/(L*min)].
        """
        # Guard against non-physical temperature values
        T_safe = max(100.0, float(T))
        k = self.params.k_0 * np.exp(-self.params.E_over_R / T_safe)
        return float(k * max(0.0, float(C_A)))

    def derivatives(self, t: float, state: np.ndarray, q_j: float) -> np.ndarray:
        """Computes time derivatives of the CSTR states dx/dt = f(x, u).

        Args:
            t: Current time in minutes (unused in autonomous ODE, required by ODE solvers).
            state: Array containing [C_A, T, T_j].
            q_j: Coolant volumetric flow rate in jacket [L/min].

        Returns:
            1D array containing [dC_A/dt, dT/dt, dT_j/dt].
        """
        C_A, T, T_j = float(state[0]), float(state[1]), float(state[2])
        p = self.params

        r_A = self.reaction_rate(C_A, T)

        # 1. Mass Balance: dC_A/dt
        dC_A_dt = (p.q / p.V) * (p.C_Af - C_A) - r_A

        # 2. Reactor Energy Balance: dT/dt
        heat_gen = (-p.delta_H / p.rho_cp) * r_A
        heat_removal = (p.UA / (p.V * p.rho_cp)) * (T - T_j)
        dT_dt = (p.q / p.V) * (p.T_f - T) + heat_gen - heat_removal

        # 3. Cooling Jacket Energy Balance: dT_j/dt
        jacket_exchange = (p.UA / (p.V_j * p.rho_j_cp_j)) * (T - T_j)
        dT_j_dt = (q_j / p.V_j) * (p.T_jf - T_j) + jacket_exchange

        return np.array([dC_A_dt, dT_dt, dT_j_dt], dtype=np.float64)

    def find_steady_state(
        self, q_j: float, initial_guess: Optional[CSTRState] = None
    ) -> CSTRState:
        """Finds equilibrium state x_ss where f(x_ss, q_j) = 0.

        Args:
            q_j: Steady-state coolant flow rate [L/min].
            initial_guess: Initial guess for root solver.

        Returns:
            Computed equilibrium CSTRState.

        Raises:
            RuntimeError: If the numerical solver fails to converge.
        """
        guess = (
            initial_guess.to_array()
            if initial_guess is not None
            else np.array([0.1, 350.0, 300.0], dtype=np.float64)
        )

        def residual(x: np.ndarray) -> np.ndarray:
            return self.derivatives(0.0, x, q_j)

        res = root(residual, guess, method="hybr")
        if not res.success:
            logger.error("Failed to find steady state for q_j=%.2f: %s", q_j, res.message)
            raise RuntimeError(f"Steady-state convergence failed: {res.message}")

        ss_state = CSTRState.from_array(res.x)
        logger.info(
            "Steady state found for q_j=%.2f: C_A=%.4f mol/L, T=%.2f K, T_j=%.2f K",
            q_j,
            ss_state.C_A,
            ss_state.T,
            ss_state.T_j,
        )
        return ss_state

    def linearize(
        self, state_ss: CSTRState, q_j_ss: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates state-space Jacobian matrices A = df/dx and B = df/du at steady state.

        Args:
            state_ss: Equilibrium operating point state.
            q_j_ss: Equilibrium coolant flow rate.

        Returns:
            Tuple (A, B) representing linearized continuous system dx/dt = A x + B u.
        """
        p = self.params
        C_A = state_ss.C_A
        T = state_ss.T
        T_j = state_ss.T_j

        k = p.k_0 * np.exp(-p.E_over_R / T)
        dk_dT = k * (p.E_over_R / (T**2))

        # Partial derivatives for A matrix
        # Row 1: d(dC_A/dt) / d[C_A, T, T_j]
        a11 = -(p.q / p.V) - k
        a12 = -dk_dT * C_A
        a13 = 0.0

        # Row 2: d(dT/dt) / d[C_A, T, T_j]
        a21 = (-p.delta_H / p.rho_cp) * k
        a22 = -(p.q / p.V) + (-p.delta_H / p.rho_cp) * (dk_dT * C_A) - (p.UA / (p.V * p.rho_cp))
        a23 = p.UA / (p.V * p.rho_cp)

        # Row 3: d(dT_j/dt) / d[C_A, T, T_j]
        a31 = 0.0
        a32 = p.UA / (p.V_j * p.rho_j_cp_j)
        a33 = -(q_j_ss / p.V_j) - (p.UA / (p.V_j * p.rho_j_cp_j))

        A = np.array([
            [a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33],
        ], dtype=np.float64)

        # Partial derivatives for B matrix: d(df/dt) / dq_j
        b1 = 0.0
        b2 = 0.0
        b3 = (p.T_jf - T_j) / p.V_j

        B = np.array([[b1], [b2], [b3]], dtype=np.float64)

        logger.debug("Linearized A eigenvalues: %s", np.linalg.eigvals(A))
        return A, B
