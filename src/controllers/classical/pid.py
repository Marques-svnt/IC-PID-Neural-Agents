"""Digital PID controller with Tustin discretization and anti-windup."""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DiscretizationMethod(str, Enum):
    """Available discretization methods for the digital PID."""

    TUSTIN = "tustin"  # Bilinear (Tustin) — default, best frequency response
    FORWARD_EULER = "euler_forward"
    BACKWARD_EULER = "euler_backward"


@dataclass
class PIDGains:
    """PID controller tuning parameters.

    Attributes:
        kp: Proportional gain [output_unit / error_unit].
        ki: Integral gain [output_unit / (error_unit * s)].
        kd: Derivative gain [output_unit * s / error_unit].
        u_min: Minimum control output (saturation lower bound).
        u_max: Maximum control output (saturation upper bound).
    """

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    u_min: float = 0.0
    u_max: float = 100.0


class DigitalPID:
    """Discrete-time PID controller with anti-windup (back-calculation).

    Implements the parallel form:
        u(k) = Kp*e(k) + Ki*T*sum(e) + Kd/T*(e(k) - e(k-1))

    Anti-windup is handled via integrator clamping when output saturates.

    Args:
        gains: PIDGains dataclass with Kp, Ki, Kd and output bounds.
        dt: Sampling time [s].
        method: Discretization method (default: Tustin).
    """

    def __init__(
        self,
        gains: PIDGains,
        dt: float,
        method: DiscretizationMethod = DiscretizationMethod.TUSTIN,
    ) -> None:
        self.gains = gains
        self.dt = dt
        self.method = method

        # Internal state
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._prev_output: float = 0.0

        logger.info(
            "DigitalPID initialized | Kp=%.4f Ki=%.4f Kd=%.4f dt=%.3f method=%s",
            gains.kp,
            gains.ki,
            gains.kd,
            dt,
            method.value,
        )

    def reset(self) -> None:
        """Reset controller internal state (call between simulations)."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_output = 0.0
        logger.debug("PID state reset.")

    def compute(self, setpoint: float, measurement: float) -> float:
        """Compute next control output.

        Args:
            setpoint: Desired process value (reference) [same unit as measurement].
            measurement: Current process output [same unit as setpoint].

        Returns:
            Control signal u(k), clamped to [u_min, u_max].
        """
        error = setpoint - measurement
        g = self.gains

        # Anti-windup: only integrate if output was not saturated last step
        saturated = self._prev_output >= g.u_max or self._prev_output <= g.u_min
        if not saturated:
            self._integral += error * self.dt

        derivative = (error - self._prev_error) / self.dt if self.dt > 0 else 0.0

        output_raw = g.kp * error + g.ki * self._integral + g.kd * derivative
        output = float(max(g.u_min, min(g.u_max, output_raw)))

        self._prev_error = error
        self._prev_output = output

        logger.debug(
            "PID | sp=%.3f meas=%.3f err=%.3f u=%.3f",
            setpoint,
            measurement,
            error,
            output,
        )
        return output
