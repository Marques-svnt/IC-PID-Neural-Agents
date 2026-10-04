"""Unit tests for the Neural PID Controller (NNPIDController)."""

import numpy as np
import pytest
import torch

from src.controllers.neural.nn_pid import AdaptiveGainNetwork, NNPIDController


class TestAdaptiveGainNetwork:
    """Test suite for the AdaptiveGainNetwork PyTorch module."""

    def test_forward_output_shape_and_bounds(self) -> None:
        """Verifies tensor shapes and boundary satisfaction of neural gains."""
        kp_bounds = (10.0, 50.0)
        ki_bounds = (2.0, 20.0)
        kd_bounds = (1.0, 10.0)

        net = AdaptiveGainNetwork(
            input_dim=4,
            hidden_dim=16,
            kp_bounds=kp_bounds,
            ki_bounds=ki_bounds,
            kd_bounds=kd_bounds,
        )

        batch_x = torch.randn(8, 4)
        out = net(batch_x)

        assert out.shape == (8, 3)
        # Check that all gains are strictly within specified bounds
        assert torch.all(out[:, 0] >= kp_bounds[0]) and torch.all(
            out[:, 0] <= kp_bounds[1]
        )
        assert torch.all(out[:, 1] >= ki_bounds[0]) and torch.all(
            out[:, 1] <= ki_bounds[1]
        )
        assert torch.all(out[:, 2] >= kd_bounds[0]) and torch.all(
            out[:, 2] <= kd_bounds[1]
        )


class TestNNPIDController:
    """Test suite for the discrete NNPIDController."""

    def test_invalid_dt_raises_value_error(self) -> None:
        """Ensures that non-positive dt values raise ValueError."""
        with pytest.raises(ValueError, match="dt must be positive"):
            NNPIDController(dt=0.0)

        with pytest.raises(ValueError, match="dt must be positive"):
            NNPIDController(dt=-0.1)

    def test_gain_boundaries_satisfied(self) -> None:
        """Verifies that computed gains are strictly within assigned boundaries."""
        controller = NNPIDController(
            dt=0.05,
            kp_bounds=(15.0, 45.0),
            ki_bounds=(5.0, 25.0),
            kd_bounds=(1.0, 8.0),
        )

        kp, ki, kd = controller.compute_gains(
            error=5.0,
            d_error=2.0,
            integral=10.0,
            temp_val=390.0,
        )

        assert 15.0 <= kp <= 45.0
        assert 5.0 <= ki <= 25.0
        assert 1.0 <= kd <= 8.0

    def test_output_clamped_to_actuator_limits(self) -> None:
        """Ensures control action never violates physical saturation limits."""
        u_min = 10.0
        u_max = 200.0
        controller = NNPIDController(
            dt=0.05,
            u_min=u_min,
            u_max=u_max,
            slew_rate_max=1000.0,
            u_nominal=100.0,
        )

        # Huge error should saturate to u_max
        step1 = controller.update(
            setpoint=350.0, measured_value=500.0, reverse_acting=True
        )
        assert step1["u"] <= u_max

        # Huge negative error should saturate to u_min
        controller.reset()
        step2 = controller.update(
            setpoint=500.0, measured_value=300.0, reverse_acting=True
        )
        assert step2["u"] >= u_min

    def test_anti_windup_prevents_integral_growth_when_saturated(self) -> None:
        """Checks anti-windup clamping behavior during prolonged saturation."""
        controller = NNPIDController(
            dt=0.1,
            u_min=0.0,
            u_max=150.0,
            u_nominal=150.0,  # Already at max limit
        )

        # Apply high temperature that demands more cooling (reverse acting)
        for _ in range(5):
            res = controller.update(
                setpoint=350.0, measured_value=400.0, reverse_acting=True
            )
            assert res["u"] == 150.0

        # Integral error should remain 0 because previous_u was at u_max and error > 0
        assert controller.integral_error == 0.0

    def test_reset_clears_internal_state(self) -> None:
        """Verifies that reset returns memories to initial states."""
        controller = NNPIDController(dt=0.05, u_nominal=80.0)
        controller.update(setpoint=350.0, measured_value=360.0)

        assert controller.previous_u != 80.0 or controller.integral_error != 0.0

        controller.reset()
        assert controller.integral_error == 0.0
        assert controller.previous_error == 0.0
        assert controller.previous_u == 80.0

    def test_gains_conversion_bidirectional(self) -> None:
        """Verifies mathematical consistency of parallel <-> standard conversions."""
        from src.controllers.neural.nn_pid import (
            gains_parallel_to_standard,
            gains_standard_to_parallel,
        )

        kp_orig = 25.0
        ki_orig = 18.0
        kd_orig = 3.5

        kp_std, ti_std, td_std = gains_parallel_to_standard(kp_orig, ki_orig, kd_orig)
        assert np.isclose(kp_std, 25.0)
        assert np.isclose(ti_std, 25.0 / 18.0)
        assert np.isclose(td_std, 3.5 / 25.0)

        kp_recon, ki_recon, kd_recon = gains_standard_to_parallel(
            kp_std, ti_std, td_std
        )
        assert np.isclose(kp_recon, kp_orig)
        assert np.isclose(ki_recon, ki_orig)
        assert np.isclose(kd_recon, kd_orig)

    def test_compute_standard_gains(self) -> None:
        """Verifies that compute_standard_gains yields valid time constants."""
        controller = NNPIDController(dt=0.05)
        kp, ti, td = controller.compute_standard_gains(
            error=2.0,
            d_error=0.5,
            integral=1.0,
            temp_val=395.0,
        )
        assert kp > 0.0
        assert ti > 0.0
        assert td > 0.0

    def test_save_and_load_weights(self, tmp_path) -> None:
        """Verifies weight serialization and faithful reloading."""
        controller1 = NNPIDController(dt=0.05)
        weight_file = tmp_path / "model_weights.pt"
        controller1.save_weights(weight_file)
        assert weight_file.exists()

        controller2 = NNPIDController(dt=0.05, model_path=str(weight_file))
        g1 = controller1.compute_gains(1.0, 0.2, 0.5, 396.0)
        g2 = controller2.compute_gains(1.0, 0.2, 0.5, 396.0)
        assert np.allclose(g1, g2)
