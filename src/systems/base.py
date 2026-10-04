"""Abstract base class for all thermal system models."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class SystemParameters:
    """Base dataclass for system parameters. Subclass for each system."""

    pass


@dataclass
class SimulationResult:
    """Container for simulation output data.

    Attributes:
        time: Time vector [s].
        output: System output (e.g., temperature) [K or degC].
        control_signal: Control effort applied [W or %].
        setpoint: Reference setpoint vector [K or degC].
        error: Tracking error (setpoint - output).
    """

    time: NDArray[np.float64]
    output: NDArray[np.float64]
    control_signal: NDArray[np.float64]
    setpoint: NDArray[np.float64]
    error: NDArray[np.float64]


class ThermalSystem(ABC):
    """Abstract base class for thermal process models.

    All physical units must be specified in subclass docstrings.
    Subclasses must implement `dynamics()` and `step_response()`.
    """

    def __init__(self, params: SystemParameters, dt: float = 1.0) -> None:
        """Initialize the thermal system.

        Args:
            params: System-specific parameters dataclass.
            dt: Sampling time [s].
        """
        self.params = params
        self.dt = dt
        logger.info(
            "Initialized %s with dt=%.3f s",
            self.__class__.__name__,
            dt,
        )

    @abstractmethod
    def dynamics(
        self,
        t: float,
        state: NDArray[np.float64],
        u: float,
    ) -> NDArray[np.float64]:
        """Compute state derivatives (ODE right-hand side).

        Args:
            t: Current time [s].
            state: Current state vector.
            u: Control input [W or normalized].

        Returns:
            State derivatives dx/dt.
        """
        ...

    @abstractmethod
    def step_response(
        self,
        step_magnitude: float = 1.0,
        duration: float = 200.0,
    ) -> SimulationResult:
        """Simulate open-loop step response for model validation.

        Args:
            step_magnitude: Amplitude of the step input.
            duration: Simulation duration [s].

        Returns:
            SimulationResult with time, output, and control signal.
        """
        ...
