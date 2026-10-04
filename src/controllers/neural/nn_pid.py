"""Neural PID Controller implementation using PyTorch for adaptive gain tuning.

This module provides the neural network architecture that computes adaptive
PID parameters (Kp, Ki, Kd) in real-time based on tracking error, its rate
of change, and process states, guaranteeing strictly positive gains within
physically safe boundaries.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def gains_parallel_to_standard(
    kp: float,
    ki: float,
    kd: float,
    eps: float = 1e-6,
) -> Tuple[float, float, float]:
    """Converts parallel PID gains (Kp, Ki, Kd) to standard series time constants (Kp, Ti, Td).

    Formulas:
        Ti = Kp / Ki
        Td = Kd / Kp

    Args:
        kp: Proportional gain.
        ki: Integral gain.
        kd: Derivative gain.
        eps: Minimum denominator threshold to prevent division by zero.

    Returns:
        Tuple of (Kp, Ti, Td) with integral and derivative time constants [min].
    """
    ti = kp / max(abs(ki), eps)
    td = kd / max(abs(kp), eps)
    return kp, ti, td


def gains_standard_to_parallel(
    kp: float,
    ti: float,
    td: float,
    eps: float = 1e-6,
) -> Tuple[float, float, float]:
    """Converts standard series PID parameters (Kp, Ti, Td) to parallel gains (Kp, Ki, Kd).

    Formulas:
        Ki = Kp / Ti
        Kd = Kp * Td

    Args:
        kp: Proportional gain.
        ti: Integral reset time constant [min].
        td: Derivative rate time constant [min].
        eps: Minimum denominator threshold to prevent division by zero.

    Returns:
        Tuple of parallel gains (Kp, Ki, Kd).
    """
    ki = kp / max(abs(ti), eps)
    kd = kp * td
    return kp, ki, kd


class AdaptiveGainNetwork(nn.Module):
    """Feedforward neural network predicting adaptive PID gains.

    The network processes normalized error signals and process temperatures,
    outputting scale factors constrained by Sigmoid activations to ensure
    gains stay within predefined stability boundaries.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 32,
        kp_bounds: Tuple[float, float] = (5.0, 60.0),
        ki_bounds: Tuple[float, float] = (1.0, 40.0),
        kd_bounds: Tuple[float, float] = (0.5, 15.0),
    ) -> None:
        """Initializes the adaptive gain neural network.

        Args:
            input_dim: Dimension of input feature vector [error, d_error, integral, temp_norm].
            hidden_dim: Number of neurons in hidden layers.
            kp_bounds: (min, max) allowable range for proportional gain Kp.
            ki_bounds: (min, max) allowable range for integral gain Ki.
            kd_bounds: (min, max) allowable range for derivative gain Kd.
        """
        super().__init__()
        self.kp_min, self.kp_max = kp_bounds
        self.ki_min, self.ki_max = ki_bounds
        self.kd_min, self.kd_max = kd_bounds

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

        logger.info(
            "AdaptiveGainNetwork initialized: Kp in [%.1f, %.1f], Ki in [%.1f, %.1f], Kd in [%.1f, %.1f]",
            self.kp_min,
            self.kp_max,
            self.ki_min,
            self.ki_max,
            self.kd_min,
            self.kd_max,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes positive, bounded PID gains from input features.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,).

        Returns:
            Tensor of shape (batch_size, 3) or (3,) containing [Kp, Ki, Kd].
        """
        raw_scale = self.net(x)
        # Scale Sigmoid outputs (0, 1) into respective physical boundaries
        kp = self.kp_min + raw_scale[..., 0] * (self.kp_max - self.kp_min)
        ki = self.ki_min + raw_scale[..., 1] * (self.ki_max - self.ki_min)
        kd = self.kd_min + raw_scale[..., 2] * (self.kd_max - self.kd_min)

        return torch.stack([kp, ki, kd], dim=-1)


class NNPIDController:
    """Discrete-time Neural PID Controller with online neural gain scheduling.

    Combines classical digital PID filtering and anti-windup clamping with
    neural network parameter modulation.
    """

    def __init__(
        self,
        dt: float = 0.05,
        u_min: float = 0.0,
        u_max: float = 300.0,
        slew_rate_max: Optional[float] = 80.0,
        u_nominal: float = 100.0,
        kp_bounds: Tuple[float, float] = (5.0, 60.0),
        ki_bounds: Tuple[float, float] = (1.0, 40.0),
        kd_bounds: Tuple[float, float] = (0.5, 15.0),
        model_path: Optional[str] = None,
    ) -> None:
        """Initializes the Neural PID Controller.

        Args:
            dt: Sampling interval [min].
            u_min: Minimum control effort limit (e.g. min cooling flow).
            u_max: Maximum control effort limit (e.g. max cooling flow).
            slew_rate_max: Maximum allowed rate of change |du/dt| [units/min].
            u_nominal: Baseline/bias control action.
            kp_bounds: Minimum and maximum proportional gain.
            ki_bounds: Minimum and maximum integral gain.
            kd_bounds: Minimum and maximum derivative gain.
            model_path: Optional path to pre-trained PyTorch weights.
        """
        if dt <= 0:
            raise ValueError(f"Sampling time dt must be positive, got {dt}")

        self.dt = float(dt)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.slew_rate_max = float(slew_rate_max) if slew_rate_max is not None else None
        self.u_nominal = float(u_nominal)

        self.model = AdaptiveGainNetwork(
            input_dim=4,
            hidden_dim=32,
            kp_bounds=kp_bounds,
            ki_bounds=ki_bounds,
            kd_bounds=kd_bounds,
        )

        if model_path is not None:
            self.load_weights(model_path)
        else:
            self.model.eval()

        # Controller state variables
        self.integral_error: float = 0.0
        self.previous_error: float = 0.0
        self.previous_u: float = self.u_nominal

    def reset(self) -> None:
        """Resets controller internal states (integral and previous error)."""
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.previous_u = self.u_nominal
        logger.debug("NNPIDController internal states reset.")

    def compute_gains(
        self,
        error: float,
        d_error: float,
        integral: float,
        temp_val: float,
    ) -> Tuple[float, float, float]:
        """Infers adaptive gains from current process states.

        Args:
            error: Current tracking error (e.g., T_reactor - setpoint) [K].
            d_error: Error derivative (de/dt) [K/min].
            integral: Accumulated error integral [K*min].
            temp_val: Current reactor temperature [K].

        Returns:
            Tuple of (Kp, Ki, Kd).
        """
        # Normalization heuristics for neural stability
        features = torch.tensor(
            [
                np.clip(error / 10.0, -5.0, 5.0),
                np.clip(d_error / 50.0, -5.0, 5.0),
                np.clip(integral / 20.0, -5.0, 5.0),
                (temp_val - 350.0) / 100.0,
            ],
            dtype=torch.float32,
        )

        with torch.no_grad():
            gains = self.model(features)

        kp = float(gains[0].item())
        ki = float(gains[1].item())
        kd = float(gains[2].item())
        return kp, ki, kd

    def update(
        self,
        setpoint: float,
        measured_value: float,
        reverse_acting: bool = True,
    ) -> Dict[str, float]:
        """Calculates control action with adaptive neural gains and anti-windup.

        For exothermic cooling (reverse acting default):
        When measured_value > setpoint, error > 0, calling for INCREASED cooling.

        Args:
            setpoint: Desired temperature setpoint [K].
            measured_value: Current measured reactor temperature [K].
            reverse_acting: True if control effort increases when measured > setpoint.

        Returns:
            Dictionary containing 'u', 'kp', 'ki', 'kd', 'error'.
        """
        error = (
            (measured_value - setpoint)
            if reverse_acting
            else (setpoint - measured_value)
        )
        d_error = (error - self.previous_error) / self.dt if self.dt > 0 else 0.0

        # Anti-windup conditional integration
        is_saturated_high = (self.previous_u >= self.u_max) and (error > 0)
        is_saturated_low = (self.previous_u <= self.u_min) and (error < 0)

        if not (is_saturated_high or is_saturated_low):
            self.integral_error += error * self.dt

        # Neural gain inference
        kp, ki, kd = self.compute_gains(
            error=error,
            d_error=d_error,
            integral=self.integral_error,
            temp_val=measured_value,
        )

        # PID control equation
        p_term = kp * error
        i_term = ki * self.integral_error
        d_term = kd * d_error
        u_target = self.u_nominal + p_term + i_term + d_term

        # Rate limiting (slew-rate)
        if self.slew_rate_max is not None:
            max_delta = self.slew_rate_max * self.dt
            delta_u = np.clip(u_target - self.previous_u, -max_delta, max_delta)
            u_computed = self.previous_u + delta_u
        else:
            u_computed = u_target

        # Hard saturation
        u_saturated = float(np.clip(u_computed, self.u_min, self.u_max))

        # Update memories
        self.previous_error = error
        self.previous_u = u_saturated

        return {
            "u": u_saturated,
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "error": error,
        }

    def compute_standard_gains(
        self,
        error: float,
        d_error: float,
        integral: float,
        temp_val: float,
    ) -> Tuple[float, float, float]:
        """Infers adaptive standard series parameters (Kp, Ti, Td).

        Args:
            error: Current tracking error [K].
            d_error: Error derivative [K/min].
            integral: Accumulated error integral [K*min].
            temp_val: Current reactor temperature [K].

        Returns:
            Tuple of (Kp, Ti, Td) with time constants Ti and Td in minutes.
        """
        kp, ki, kd = self.compute_gains(error, d_error, integral, temp_val)
        return gains_parallel_to_standard(kp, ki, kd)

    def save_weights(self, filepath: Union[str, Path]) -> None:
        """Saves current PyTorch model weights to disk.

        Args:
            filepath: Destination file path for .pt model weights.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(path))
        logger.info("Saved NNPIDController weights to %s", path)

    def load_weights(self, filepath: Union[str, Path]) -> None:
        """Loads pre-trained PyTorch weights from disk.

        Args:
            filepath: Source file path for .pt model weights.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Weight file not found: {path}")
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info("Loaded NNPIDController weights from %s", path)
