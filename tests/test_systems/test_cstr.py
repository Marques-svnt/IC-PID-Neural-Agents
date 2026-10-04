"""Unit tests for the CSTR phenomenological system model (CSTRSystem)."""

import numpy as np
import pytest

from src.systems.cstr import CSTRSystem


class TestCSTRSystem:
    """Test suite for CSTR nonlinear dynamics and simulation."""

    @pytest.fixture
    def cstr(self) -> CSTRSystem:
        """Fixture providing default initialized CSTRSystem."""
        return CSTRSystem(dt=0.02)

    def test_nominal_steady_state_has_near_zero_derivatives(
        self,
        cstr: CSTRSystem,
    ) -> None:
        """Verifies that the nominal state corresponds to an equilibrium operating point."""
        ss_state = cstr.get_nominal_steady_state()
        u_ss = cstr.cstr_params.q_j_ss

        derivatives = cstr.dynamics(t=0.0, state=ss_state, u=u_ss)

        # Balances should be close to zero at steady-state
        assert np.isclose(derivatives[0], 0.0, atol=1e-3), f"dC_A/dt = {derivatives[0]}"
        assert np.isclose(derivatives[1], 0.0, atol=1e-2), f"dT/dt = {derivatives[1]}"
        assert np.isclose(derivatives[2], 0.0, atol=1e-2), f"dT_j/dt = {derivatives[2]}"

    def test_step_response_shape_and_consistency(
        self,
        cstr: CSTRSystem,
    ) -> None:
        """Ensures that step_response produces correct vectors and dimensions."""
        duration = 5.0
        step_mag = 15.0
        res = cstr.step_response(step_magnitude=step_mag, duration=duration)

        expected_len = int(duration / cstr.dt) + 1
        assert len(res.time) == expected_len
        assert len(res.output) == expected_len
        assert len(res.control_signal) == expected_len
        assert len(res.setpoint) == expected_len
        assert len(res.error) == expected_len

        # Increased cooling water should lower reactor temperature
        assert res.output[-1] < res.output[0]

    def test_actuator_slew_rate_and_saturation(
        self,
        cstr: CSTRSystem,
    ) -> None:
        """Verifies that actuator rate limiting (slew-rate) and clipping are enforced."""
        state = cstr.get_nominal_steady_state()
        dt = cstr.dt
        s_max = cstr.cstr_params.s_max

        # Request massive instantaneous jump from 100 to 300
        prev_u = 100.0
        desired_u = 300.0

        _, applied_u = cstr.integrate_step(
            state=state,
            u=desired_u,
            prev_u=prev_u,
        )

        max_allowed_delta = s_max * dt
        assert applied_u <= prev_u + max_allowed_delta + 1e-6
        assert applied_u <= cstr.cstr_params.u_max

    def test_feed_disturbances_affect_dynamics(
        self,
        cstr: CSTRSystem,
    ) -> None:
        """Verifies that disturbances in feed temperature and concentration alter derivatives."""
        state = cstr.get_nominal_steady_state()
        u_ss = cstr.cstr_params.q_j_ss

        nom_derivs = cstr.dynamics(t=0.0, state=state, u=u_ss)
        hot_feed_derivs = cstr.dynamics(t=0.0, state=state, u=u_ss, t_f=360.0)

        # Higher feed temperature directly increases thermal energy derivative dT/dt
        assert hot_feed_derivs[1] > nom_derivs[1]

        concentrated_derivs = cstr.dynamics(t=0.0, state=state, u=u_ss, c_af=1.2)
        # Higher reactant feed concentration directly increases mass balance derivative dC_A/dt
        assert concentrated_derivs[0] > nom_derivs[0]
