"""Unit tests for the Control MCP server and numerical simulation tools.

Covers:
    - tune_classical_pid: IMC, Ziegler-Nichols, Cohen-Coon, and edge/error cases.
    - compute_transient_metrics: Standard, zero overshoot, constant control, negative step,
      and dimensional mismatch edge cases.
    - simulate_thermal_plant: Jacketed CSTR and linear FOPTD simulations, actuator limits,
      and invalid argument guards.
    - LangGraph and Langfuse integration: Workflow orchestration and telemetry handling.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.agents.orchestrator import (
    ControlWorkflowState,
    get_langfuse_callback,
    run_control_workflow,
)
from src.mcp.control_mcp import (
    compute_transient_metrics,
    simulate_thermal_plant,
    tune_classical_pid,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Unit Tests for `tune_classical_pid`
# =============================================================================
class TestTuneClassicalPID:
    """Test suite for analytical PID controller tuning relations."""

    def test_tune_imc_valid(self) -> None:
        """Verify IMC / Skogestad SIMC tuning relations on a standard FOPTD plant."""
        k = 2.0
        tau = 10.0
        theta = 2.0
        result = tune_classical_pid(k=k, tau=tau, theta=theta, method="imc")

        assert isinstance(result, dict)
        assert "kp" in result
        assert "ti" in result
        assert "td" in result
        assert "ki" in result
        assert "kd" in result

        # Expected SIMC: tau_c = max(theta, 0.1*tau) = 2.0
        # Kp = (1 / k) * (tau / (tau_c + theta)) = (1 / 2) * (10 / (2 + 2)) = 1.25
        # Ti = min(tau, 8*(tau_c + theta)) = min(10, 8*4) = 10.0
        # Td = 0.0
        assert np.isclose(result["kp"], 1.25, rtol=1e-3)
        assert np.isclose(result["ti"], 10.0, rtol=1e-3)
        assert np.isclose(result["td"], 0.0, atol=1e-6)
        assert np.isclose(result["ki"], 1.25 / 10.0, rtol=1e-3)
        assert np.isclose(result["kd"], 0.0, atol=1e-6)

    def test_tune_ziegler_nichols_valid(self) -> None:
        """Verify open-loop Ziegler-Nichols reaction curve formulas."""
        k = 1.5
        tau = 6.0
        theta = 1.0
        result = tune_classical_pid(k=k, tau=tau, theta=theta, method="ziegler_nichols")

        # Kp = 1.2 * tau / (k * theta) = 1.2 * 6.0 / (1.5 * 1.0) = 4.8
        # Ti = 2.0 * theta = 2.0
        # Td = 0.5 * theta = 0.5
        assert np.isclose(result["kp"], 4.8, rtol=1e-3)
        assert np.isclose(result["ti"], 2.0, rtol=1e-3)
        assert np.isclose(result["td"], 0.5, rtol=1e-3)
        assert np.isclose(result["ki"], 4.8 / 2.0, rtol=1e-3)
        assert np.isclose(result["kd"], 4.8 * 0.5, rtol=1e-3)

    def test_tune_cohen_coon_valid(self) -> None:
        """Verify Cohen-Coon tuning parameters."""
        k = 2.0
        tau = 8.0
        theta = 2.0
        result = tune_classical_pid(k=k, tau=tau, theta=theta, method="cohen_coon")

        # R = theta / tau = 2 / 8 = 0.25
        # Kp = (1 / 2) * (8 / 2) * (4/3 + 0.25/4) = 2.0 * (1.3333 + 0.0625) = 2.79166...
        r_ratio = 0.25
        expected_kp = (1.0 / 2.0) * (8.0 / 2.0) * (4.0 / 3.0 + 0.25 * r_ratio)
        expected_ti = theta * (32.0 + 6.0 * r_ratio) / (13.0 + 8.0 * r_ratio)
        expected_td = theta * 4.0 / (11.0 + 2.0 * r_ratio)

        assert np.isclose(result["kp"], expected_kp, rtol=1e-3)
        assert np.isclose(result["ti"], expected_ti, rtol=1e-3)
        assert np.isclose(result["td"], expected_td, rtol=1e-3)

    def test_tune_zero_gain_raises_error(self) -> None:
        """Zero process gain must raise ValueError to prevent division by zero."""
        with pytest.raises(ValueError, match="gain 'k' cannot be zero"):
            tune_classical_pid(k=0.0, tau=5.0, theta=1.0, method="imc")

    def test_tune_negative_tau_raises_error(self) -> None:
        """Negative time constant must raise ValueError."""
        with pytest.raises(
            ValueError, match=r"(?i)time constant 'tau' must be strictly positive"
        ):
            tune_classical_pid(k=1.0, tau=-2.0, theta=1.0, method="imc")

    def test_tune_negative_theta_raises_error(self) -> None:
        """Negative dead time must raise ValueError."""
        with pytest.raises(
            ValueError, match=r"(?i)dead time 'theta' cannot be negative"
        ):
            tune_classical_pid(k=1.0, tau=5.0, theta=-0.5, method="imc")

    def test_tune_zero_theta_singularity_raises_error(self) -> None:
        """Zero dead time in Z-N or Cohen-Coon must raise ValueError to prevent division by zero."""
        with pytest.raises(ValueError, match="strictly positive for 'ziegler_nichols'"):
            tune_classical_pid(k=1.0, tau=5.0, theta=0.0, method="ziegler_nichols")

        with pytest.raises(ValueError, match="strictly positive for 'cohen_coon'"):
            tune_classical_pid(k=1.0, tau=5.0, theta=0.0, method="cohen_coon")

    def test_tune_invalid_method_raises_error(self) -> None:
        """Unrecognized tuning method name must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported tuning method"):
            tune_classical_pid(k=1.0, tau=5.0, theta=1.0, method="arbitrary_heuristic")


# =============================================================================
# Unit Tests for `compute_transient_metrics`
# =============================================================================
class TestComputeTransientMetrics:
    """Test suite for transient, integral, and total variation metrics."""

    def test_compute_metrics_standard_response(self) -> None:
        """Validate metrics calculation on a synthetic second-order damped step response."""
        t = list(np.linspace(0.0, 10.0, 1001))
        # Simulated underdamped response with known overshoot
        setpoint = 100.0
        omega_n = 2.0
        zeta = 0.4
        wd = omega_n * np.sqrt(1.0 - zeta**2)
        y = [
            float(
                setpoint
                * (
                    1.0
                    - np.exp(-zeta * omega_n * ti)
                    * (
                        np.cos(wd * ti)
                        + (zeta / np.sqrt(1.0 - zeta**2)) * np.sin(wd * ti)
                    )
                )
            )
            for ti in t
        ]
        u = [float(50.0 + 10.0 * np.exp(-ti)) for ti in t]

        metrics = compute_transient_metrics(
            time_series=t,
            output_series=y,
            setpoint=setpoint,
            control_series=u,
        )

        assert "rise_time" in metrics
        assert "settling_time" in metrics
        assert "overshoot_pct" in metrics
        assert "iae" in metrics
        assert "ise" in metrics
        assert "itae" in metrics
        assert "total_variation" in metrics

        # Underdamped system with zeta=0.4 has theoretical overshoot around 25.4%
        assert 20.0 < metrics["overshoot_pct"] < 30.0
        assert metrics["rise_time"] > 0.0
        assert metrics["settling_time"] > metrics["rise_time"]
        assert metrics["iae"] > 0.0
        assert metrics["ise"] > 0.0
        assert metrics["total_variation"] > 0.0

    def test_compute_metrics_zero_overshoot(self) -> None:
        """Overdamped exponential response must produce exactly 0.0% overshoot."""
        t = list(np.linspace(0.0, 10.0, 501))
        setpoint = 50.0
        # Monotonically increasing response: y(t) = 50 * (1 - exp(-t))
        y = [float(setpoint * (1.0 - np.exp(-ti))) for ti in t]
        u = [float(10.0 * (1.0 - np.exp(-ti))) for ti in t]

        metrics = compute_transient_metrics(
            time_series=t,
            output_series=y,
            setpoint=setpoint,
            control_series=u,
        )

        assert metrics["overshoot_pct"] == 0.0
        assert metrics["settling_time"] > 0.0
        assert metrics["rise_time"] > 0.0

    def test_compute_metrics_constant_control_zero_tv(self) -> None:
        """Constant control action must have a Total Variation (TV) of 0.0."""
        t = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [20.0, 25.0, 29.0, 30.0, 30.0]
        setpoint = 30.0
        u_const = [42.0, 42.0, 42.0, 42.0, 42.0]

        metrics = compute_transient_metrics(
            time_series=t,
            output_series=y,
            setpoint=setpoint,
            control_series=u_const,
        )

        assert np.isclose(metrics["total_variation"], 0.0, atol=1e-9)

    def test_compute_metrics_mismatched_lengths_raises(self) -> None:
        """Mismatched series lengths must raise ValueError."""
        t = [0.0, 1.0, 2.0]
        y = [10.0, 20.0]  # length 2
        u = [1.0, 1.0, 1.0]

        with pytest.raises(ValueError, match="Input series lengths must match"):
            compute_transient_metrics(
                time_series=t,
                output_series=y,
                setpoint=20.0,
                control_series=u,
            )

    def test_compute_metrics_single_point_raises(self) -> None:
        """Fewer than 2 points must raise ValueError for lack of interval."""
        with pytest.raises(ValueError, match="At least 2 points are required"):
            compute_transient_metrics(
                time_series=[0.0],
                output_series=[10.0],
                setpoint=10.0,
                control_series=[1.0],
            )

    def test_compute_metrics_negative_step(self) -> None:
        """Cooling step (negative delta) computes correct settling time and overshoot."""
        t = list(np.linspace(0.0, 10.0, 101))
        # Initial temp 400K, target 350K (-50K step)
        y = [float(350.0 + 50.0 * np.exp(-ti)) for ti in t]
        u = [float(100.0 + 50.0 * (1.0 - np.exp(-ti))) for ti in t]

        metrics = compute_transient_metrics(
            time_series=t,
            output_series=y,
            setpoint=350.0,
            control_series=u,
        )

        assert metrics["overshoot_pct"] == 0.0
        assert metrics["iae"] > 0.0
        assert metrics["settling_time"] > 0.0


# =============================================================================
# Unit Tests for `simulate_thermal_plant`
# =============================================================================
class TestSimulateThermalPlant:
    """Test suite for ODE integration of thermal models."""

    def test_simulate_cstr_nominal_execution(self) -> None:
        """Simulate jacketed CSTR under closed loop and verify output structures."""
        result = simulate_thermal_plant(
            system_id="cstr_jacketed",
            setpoint=401.65,  # +5K step from nominal 396.65K
            duration_s=2.0,
            dt_s=0.1,
        )

        assert isinstance(result, dict)
        assert result["system_id"] == "cstr_jacketed"
        assert len(result["time"]) == 21
        assert len(result["output"]) == 21
        assert len(result["control_signal"]) == 21
        assert len(result["setpoint"]) == 21

        # Check physical sanity
        for temp in result["output"]:
            assert 300.0 < temp < 500.0  # Safe temperature boundaries

    def test_simulate_actuator_saturation_boundaries(self) -> None:
        """Verify control signal respects [u_min, u_max] saturation limits."""
        # Demand large cooling request
        result = simulate_thermal_plant(
            system_id="cstr",
            setpoint=350.0,  # Far below steady state, requiring high cooling flow
            duration_s=3.0,
            dt_s=0.05,
        )

        u_arr = np.array(result["control_signal"])
        assert np.all(u_arr >= 0.0), "Actuator violated lower limit u_min=0.0"
        assert np.all(u_arr <= 300.0), "Actuator violated upper limit u_max=300.0"

    def test_simulate_linear_foptd_system(self) -> None:
        """Simulate linear FOPTD plant option."""
        result = simulate_thermal_plant(
            system_id="linear_foptd",
            setpoint=320.0,
            duration_s=1.0,
            dt_s=0.1,
        )

        assert len(result["output"]) == 11
        assert result["output"][0] == 300.0
        assert result["output"][-1] > 300.0

    def test_simulate_invalid_duration_or_dt(self) -> None:
        """Invalid time settings must raise ValueError."""
        with pytest.raises(ValueError, match="strictly positive"):
            simulate_thermal_plant("cstr", setpoint=400.0, duration_s=-1.0, dt_s=0.1)

        with pytest.raises(ValueError, match="strictly positive"):
            simulate_thermal_plant("cstr", setpoint=400.0, duration_s=10.0, dt_s=0.0)

        with pytest.raises(ValueError, match="greater than or equal to 'dt_s'"):
            simulate_thermal_plant("cstr", setpoint=400.0, duration_s=0.05, dt_s=0.1)

    def test_simulate_unknown_system_raises_error(self) -> None:
        """Unrecognized system identifier must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported system_id"):
            simulate_thermal_plant(
                "unrecognized_plant_xyz", setpoint=400.0, duration_s=1.0
            )


# =============================================================================
# Unit Tests for Multi-Agent Orchestration & Observability
# =============================================================================
class TestMultiAgentOrchestration:
    """Test suite for LangGraph orchestration and Langfuse callback configuration."""

    def test_get_langfuse_callback_without_credentials(self) -> None:
        """Without valid environment keys, callback should be None gracefully."""
        callback = get_langfuse_callback()
        assert callback is None or hasattr(callback, "on_chain_start")

    def test_build_control_graph_and_execution(self) -> None:
        """Execute full multi-agent pipeline and assert complete report generation."""
        initial_state: ControlWorkflowState = {
            "system_id": "cstr_jacketed",
            "setpoint": 401.65,
            "duration_s": 1.0,
            "dt_s": 0.1,
            "history": [],
        }

        final_state = run_control_workflow(initial_state)

        assert "plant_parameters" in final_state
        assert "tuning_results" in final_state
        assert "neural_config" in final_state
        assert "simulation_results" in final_state
        assert "metrics" in final_state
        assert "report" in final_state
        assert "Relatório Executivo" in final_state["report"]
        assert len(final_state["history"]) >= 5
