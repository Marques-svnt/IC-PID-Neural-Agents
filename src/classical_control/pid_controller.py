"""Industrial-grade discrete PID controller implementation with anti-windup strategies.

Supports parallel PID formulation, derivative filter (N-filter), derivative on measurement,
and comparative anti-windup architectures (Conditional Clamping vs Back-Calculation).
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from src.sim_core.actuators import ActuatorLimits, ValveActuator

logger = logging.getLogger(__name__)


class AntiWindupMethod(str, Enum):
    """Supported anti-windup compensation mechanisms."""

    NONE = "none"
    CLAMPING = "clamping"
    BACK_CALCULATION = "back_calculation"


@dataclass
class PIDGains:
    """PID controller parameters.

    Attributes:
        kp: Proportional gain (dimensionless or L/(min*K)).
        ti: Integral time constant [min]. Must be positive.
        td: Derivative time constant [min]. Must be non-negative.
        n_filter: Derivative filter coefficient (tau_f = td / n_filter). Typically 5 to 20.
        tt: Tracking time constant for back-calculation anti-windup [min].
            If None, defaults to sqrt(ti * td) if td > 0 else ti.
    """

    kp: float
    ti: float
    td: float = 0.0
    n_filter: float = 10.0
    tt: Optional[float] = None

    def __post_init__(self) -> None:
        if self.ti <= 0.0:
            raise ValueError(f"Integral time constant ti must be positive, got {self.ti}")
        if self.td < 0.0:
            raise ValueError(f"Derivative time constant td cannot be negative, got {self.td}")
        if self.n_filter <= 0.0:
            raise ValueError(f"Filter coefficient n_filter must be positive, got {self.n_filter}")
        if self.tt is None:
            self.tt = np.sqrt(self.ti * self.td) if self.td > 0.0 else self.ti


class PIDController:
    """Discrete Parallel PID Controller with realistic industrial features."""

    def __init__(
        self,
        gains: PIDGains,
        actuator_limits: Optional[ActuatorLimits] = None,
        anti_windup: AntiWindupMethod = AntiWindupMethod.CLAMPING,
        derivative_on_measurement: bool = True,
        u_bias: float = 100.0,
    ) -> None:
        """Initializes the PID controller.

        Args:
            gains: PID parameter specification.
            actuator_limits: Physical actuator bounds and slew-rate limits.
            anti_windup: Anti-windup method to apply ('none', 'clamping', 'back_calculation').
            derivative_on_measurement: If True, derivative action acts on PV to avoid derivative
                                       kick upon setpoint changes. If False, acts on error.
            u_bias: Nominal operating baseline action (steady-state feedforward).
        """
        self.gains = gains
        self.limits = actuator_limits or ActuatorLimits()
        self.valve = ValveActuator(limits=self.limits, initial_position=u_bias)
        self.anti_windup = anti_windup
        self.derivative_on_measurement = derivative_on_measurement
        self.u_bias = float(u_bias)

        # Internal state memory
        self._integral_state: float = 0.0
        self._prev_error: float = 0.0
        self._prev_measurement: Optional[float] = None
        self._filtered_deriv_state: float = 0.0
        self._last_u_unconstrained: float = self.u_bias
        self._last_u_applied: float = self.u_bias
        self._is_saturated: bool = False

        logger.info(
            "PIDController initialized: Kp=%.3f, Ti=%.3f, Td=%.3f, AntiWindup=%s",
            self.gains.kp,
            self.gains.ti,
            self.gains.td,
            self.anti_windup.value,
        )

    def reset(self, initial_measurement: float, initial_u: Optional[float] = None) -> None:
        """Resets controller internal states for bumpless initialization.

        Args:
            initial_measurement: Starting process variable measurement.
            initial_u: Desired initial control output. Defaults to u_bias.
        """
        u_init = initial_u if initial_u is not None else self.u_bias
        self.valve.reset(u_init)
        self._integral_state = 0.0
        self._prev_error = 0.0
        self._prev_measurement = float(initial_measurement)
        self._filtered_deriv_state = 0.0
        self._last_u_unconstrained = u_init
        self._last_u_applied = u_init
        self._is_saturated = False
        logger.debug("PIDController reset with initial measurement=%.2f", initial_measurement)

    def compute(
        self,
        setpoint: float,
        measurement: float,
        dt: float,
    ) -> Tuple[float, dict[str, float]]:
        """Calculates discrete control action u_k for the current time step.

        Args:
            setpoint: Desired target value (SP).
            measurement: Current measured process variable (PV).
            dt: Sampling period in minutes.

        Returns:
            Tuple of:
                - u_actual: Physical constrained control action applied to plant.
                - diag: Dictionary of controller components ('p', 'i', 'd', 'u_unconstrained',
                        'is_saturated').

        Raises:
            ValueError: If dt <= 0.
        """
        if dt <= 0.0:
            raise ValueError(f"Sampling time dt must be positive, got {dt}")

        error = setpoint - measurement

        if self._prev_measurement is None:
            self._prev_measurement = measurement

        # 1. Proportional Term: P = Kp * e
        p_term = self.gains.kp * error

        # 2. Derivative Term with first-order low-pass filter (N-filter)
        if self.gains.td > 0.0:
            tau_f = self.gains.td / self.gains.n_filter
            alpha = dt / (tau_f + dt)

            if self.derivative_on_measurement:
                # Derivative on PV: -d(PV)/dt avoids setpoint kick
                d_input = -(measurement - self._prev_measurement) / dt
            else:
                # Derivative on Error: d(e)/dt
                d_input = (error - self._prev_error) / dt

            # Low-pass filter update
            self._filtered_deriv_state = (
                (1.0 - alpha) * self._filtered_deriv_state + alpha * d_input
            )
            d_term = self.gains.kp * self.gains.td * self._filtered_deriv_state
        else:
            d_term = 0.0

        # 3. Integral Term Update considering Anti-Windup
        i_gain = self.gains.kp / self.gains.ti
        should_integrate = True

        if self.anti_windup == AntiWindupMethod.CLAMPING and self._is_saturated:
            # Clamping rule: freeze integrator if error attempts to drive deeper into saturation
            if self._last_u_unconstrained >= self.limits.u_max and error > 0.0:
                should_integrate = False
            elif self._last_u_unconstrained <= self.limits.u_min and error < 0.0:
                should_integrate = False

        if should_integrate:
            # Trapezoidal integration for higher accuracy
            delta_i = i_gain * 0.5 * (error + self._prev_error) * dt
            if self.anti_windup == AntiWindupMethod.BACK_CALCULATION:
                # Add tracking correction: (u_actual - u_unconstrained) / Tt
                tracking_error = self._last_u_applied - self._last_u_unconstrained
                tt = self.gains.tt or self.gains.ti
                delta_i += (tracking_error / tt) * dt

            self._integral_state += delta_i

        i_term = self._integral_state

        # 4. Unconstrained control calculation
        u_unconstrained = self.u_bias + p_term + i_term + d_term

        # 5. Actuator saturation and rate-limiting
        u_actual, is_saturated = self.valve.apply(u_unconstrained, dt=dt)

        # Store states for next iteration
        self._last_u_unconstrained = u_unconstrained
        self._last_u_applied = u_actual
        self._is_saturated = is_saturated
        self._prev_error = error
        self._prev_measurement = measurement

        diag = {
            "p": float(p_term),
            "i": float(i_term),
            "d": float(d_term),
            "u_unconstrained": float(u_unconstrained),
            "is_saturated": 1.0 if is_saturated else 0.0,
        }

        return u_actual, diag
