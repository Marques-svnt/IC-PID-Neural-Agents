"""Actuator modeling module with saturation, slew rate limits, and status flags."""

import logging
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger(__name__)


@dataclass
class ActuatorLimits:
    """Configuration limits for control valve actuators.

    Attributes:
        u_min: Minimum physical opening / flow rate limit.
        u_max: Maximum physical opening / flow rate limit.
        max_slew_rate: Maximum permissible rate of change per minute (L/min^2).
                       Set to None or inf for unrestricted valve speed.
    """

    u_min: float = 0.0
    u_max: float = 300.0
    max_slew_rate: float = 100.0


class ValveActuator:
    """Simulates physical valve dynamics including saturation and slew-rate limiting."""

    def __init__(self, limits: ActuatorLimits, initial_position: float = 100.0) -> None:
        """Initializes the valve actuator.

        Args:
            limits: Configured physical limits of the valve.
            initial_position: Initial steady-state position / flow rate.
        """
        self.limits = limits
        self._current_position = float(
            max(self.limits.u_min, min(self.limits.u_max, initial_position))
        )
        logger.info(
            "ValveActuator initialized with u_min=%.2f, u_max=%.2f, current=%.2f",
            self.limits.u_min,
            self.limits.u_max,
            self._current_position,
        )

    @property
    def current_position(self) -> float:
        """Returns the current valve position / flow rate."""
        return self._current_position

    def apply(self, u_desired: float, dt: float) -> Tuple[float, bool]:
        """Calculates actual output considering slew rate limit and absolute saturation.

        Args:
            u_desired: Desired control action demanded by controller.
            dt: Sampling or integration time step in minutes.

        Returns:
            Tuple of:
                - actual_u: Constrained actual flow rate applied to plant.
                - is_saturated: True if requested action exceeded physical limits.

        Raises:
            ValueError: If dt is non-positive.
        """
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")

        is_saturated = False

        # 1. Apply absolute clamping to desired value
        if u_desired > self.limits.u_max:
            u_clamped = self.limits.u_max
            is_saturated = True
        elif u_desired < self.limits.u_min:
            u_clamped = self.limits.u_min
            is_saturated = True
        else:
            u_clamped = u_desired

        # 2. Apply slew-rate limitation
        if self.limits.max_slew_rate is not None and self.limits.max_slew_rate > 0:
            delta_max = self.limits.max_slew_rate * dt
            delta_req = u_clamped - self._current_position
            if abs(delta_req) > delta_max:
                u_actual = self._current_position + (
                    delta_max if delta_req > 0 else -delta_max
                )
                is_saturated = True
            else:
                u_actual = u_clamped
        else:
            u_actual = u_clamped

        self._current_position = u_actual
        return self._current_position, is_saturated

    def reset(self, position: float) -> None:
        """Resets valve position to a specified level.

        Args:
            position: Target position to reset valve to.
        """
        self._current_position = float(
            max(self.limits.u_min, min(self.limits.u_max, position))
        )
        logger.debug("ValveActuator reset to %.2f", self._current_position)
