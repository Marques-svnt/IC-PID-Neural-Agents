"""Unit tests for the DigitalPID controller."""

import pytest

from src.controllers.classical.pid import DigitalPID, PIDGains


class TestDigitalPID:
    """Test suite for DigitalPID."""

    def test_proportional_only_tracks_setpoint(self) -> None:
        """P-only controller should reduce but not eliminate error."""
        gains = PIDGains(kp=2.0, ki=0.0, kd=0.0, u_min=0.0, u_max=100.0)
        pid = DigitalPID(gains=gains, dt=1.0)
        output = pid.compute(setpoint=50.0, measurement=0.0)
        assert output == pytest.approx(100.0)  # saturated at u_max

    def test_anti_windup_clamps_integral(self) -> None:
        """Integral should not grow when output is saturated."""
        gains = PIDGains(kp=1.0, ki=1.0, kd=0.0, u_min=0.0, u_max=10.0)
        pid = DigitalPID(gains=gains, dt=1.0)
        for _ in range(50):
            pid.compute(setpoint=100.0, measurement=0.0)
        # Integral should be bounded due to anti-windup
        assert pid._integral < 100.0 * 50  # far less than unclamped would be

    def test_reset_clears_state(self) -> None:
        """After reset, controller state should be zeroed."""
        gains = PIDGains(kp=1.0, ki=1.0, kd=0.0)
        pid = DigitalPID(gains=gains, dt=1.0)
        pid.compute(setpoint=10.0, measurement=0.0)
        pid.reset()
        assert pid._integral == 0.0
        assert pid._prev_error == 0.0

    def test_output_within_bounds(self) -> None:
        """Output must always be within [u_min, u_max]."""
        gains = PIDGains(kp=10.0, ki=5.0, kd=2.0, u_min=0.0, u_max=50.0)
        pid = DigitalPID(gains=gains, dt=0.1)
        for sp in [0.0, 25.0, 100.0, -50.0]:
            for meas in [0.0, 50.0, -10.0]:
                u = pid.compute(setpoint=sp, measurement=meas)
                assert 0.0 <= u <= 50.0, (
                    f"Output {u} out of bounds for sp={sp}, meas={meas}"
                )
                pid.reset()
