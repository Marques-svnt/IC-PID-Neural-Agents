"""Model Context Protocol (MCP) server for thermal simulation and control.

Exposes deterministic FastMCP tools for numerical integration of thermal plants,
transient and integral performance metric computation, and classical PID tuning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)

# Attempt importing FastMCP from mcp.server.fastmcp
try:
    from mcp.server.fastmcp import FastMCP

    _HAS_FASTMCP = True
except ImportError:
    FastMCP = None  # type: ignore[assignment,misc]
    _HAS_FASTMCP = False
    logger.warning(
        "mcp.server.fastmcp not available in environment. Running standalone."
    )

# Instantiate FastMCP server if available
if _HAS_FASTMCP and FastMCP is not None:
    mcp = FastMCP("ThermalControlServer")
else:

    class DummyMCP:
        """Fallback decorator provider when FastMCP is not installed."""

        def tool(self):  # type: ignore[no-untyped-def]
            def decorator(fn):  # type: ignore[no-untyped-def]
                return fn

            return decorator

    mcp = DummyMCP()  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Tool 1: simulate_thermal_plant
# -----------------------------------------------------------------------------
@mcp.tool()
def simulate_thermal_plant(
    system_id: str,
    setpoint: float,
    duration_s: float,
    dt_s: float = 0.1,
) -> Dict[str, Any]:
    """Integrates a thermal process model under closed-loop control via solve_ivp.

    Simulates the non-linear jacketed CSTR or standard thermal plant dynamics
    using explicit Runge-Kutta (RK45) integration. The closed-loop controller
    manipulates coolant flow rate q_j subject to physical actuator limits and slew rate.

    Args:
        system_id: Plant identifier ('cstr_jacketed', 'cstr', 'thermal_plant', 'linear_foptd').
        setpoint: Target process temperature [K].
        duration_s: Total simulation duration in seconds (or minutes for CSTR timescale).
        dt_s: Numerical discretization time step in seconds (or minutes).

    Returns:
        Dictionary containing:
            - system_id: Process model identifier.
            - time: List of time coordinates [s or min].
            - output: List of reactor temperature states T(t) [K].
            - control_signal: List of cooling flow rates q_j(t) [L/min].
            - setpoint: List of setpoint values [K].

    Raises:
        ValueError: If duration_s <= 0, dt_s <= 0, duration_s < dt_s, or invalid system_id.
    """
    logger.info(
        "simulate_thermal_plant requested | system_id=%s, setpoint=%.2f, duration=%.2f, dt=%.3f",
        system_id,
        setpoint,
        duration_s,
        dt_s,
    )

    if duration_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("Both 'duration_s' and 'dt_s' must be strictly positive.")
    if duration_s < dt_s:
        raise ValueError("'duration_s' must be greater than or equal to 'dt_s'.")

    valid_systems = {"cstr", "cstr_jacketed", "thermal_plant", "linear_foptd"}
    norm_id = system_id.lower().strip()
    if norm_id not in valid_systems:
        raise ValueError(
            f"Unsupported system_id '{system_id}'. Supported systems: {sorted(list(valid_systems))}."
        )

    num_steps = int(np.floor(duration_s / dt_s))
    time_points = [float(k * dt_s) for k in range(num_steps + 1)]

    if norm_id == "linear_foptd":
        # First-order plus lag thermal plant: dy/dt = (1/tau) * (-(y - T_amb) + K*u)
        tau = 2.0
        k_gain = 1.0
        t_ambient = 300.0
        y_val = t_ambient  # Initial ambient temp
        u_val = 0.0
        kp, ki = 2.0, 0.5
        integral = 0.0
        u_min, u_max = 0.0, 100.0

        y_out: List[float] = [y_val]
        u_out: List[float] = [u_val]

        for step in range(num_steps):
            t_curr = step * dt_s
            error = setpoint - y_val
            integral += error * dt_s
            u_raw = kp * error + ki * integral
            u_val = float(np.clip(u_raw, u_min, u_max))

            sol = solve_ivp(
                fun=lambda t, y: [(1.0 / tau) * (-(y[0] - t_ambient) + k_gain * u_val)],
                t_span=(t_curr, t_curr + dt_s),
                y0=[y_val],
                method="RK45",
                rtol=1e-6,
                atol=1e-8,
            )
            y_val = float(sol.y[0, -1])
            y_out.append(y_val)
            u_out.append(u_val)

        return {
            "system_id": system_id,
            "time": time_points,
            "output": y_out,
            "control_signal": u_out,
            "setpoint": [float(setpoint)] * len(time_points),
        }

    # Jacketed CSTR phenomenological model (Bequette, 2003)
    # Parameters
    v_reactor = 100.0  # [L]
    v_jacket = 20.0  # [L]
    q_feed = 100.0  # [L/min]
    c_af = 1.0  # [mol/L]
    t_f = 350.0  # [K]
    t_jf = 300.0  # [K]
    k0 = 7.2e10  # [min^-1]
    e_over_r = 8750.0  # [K]
    delta_h = -5.0e4  # [cal/mol]
    rho_cp = 500.0  # [cal/(L*K)]
    rho_j_cp_j = 500.0  # [cal/(L*K)]
    ua = 5.0e4  # [cal/(min*K)]

    # Steady-state nominal operating point
    c_a_ss = 0.0502  # [mol/L]
    t_ss = 396.65  # [K]
    t_j_ss = 348.33  # [K]
    q_j_ss = 100.0  # [L/min]

    # Actuator limits
    u_min = 0.0
    u_max = 300.0
    s_max = 80.0  # Max rate of change L/min^2

    # Controller tuning for closed loop
    kp = 25.0
    ki = 18.0
    kd = 4.0

    state = np.array([c_a_ss, t_ss, t_j_ss], dtype=np.float64)
    q_j_current = q_j_ss
    integral_err = 0.0
    prev_err = setpoint - t_ss

    output_t: List[float] = [float(state[1])]
    control_u: List[float] = [float(q_j_current)]

    for step in range(num_steps):
        t_curr = step * dt_s
        temp_reactor = state[1]
        # In a cooling jacket, error = T_reactor - setpoint
        # When temp > setpoint, we want MORE cooling flow q_j
        err_temp = temp_reactor - setpoint

        # Anti-windup clamping
        saturated = (q_j_current >= u_max and err_temp > 0) or (
            q_j_current <= u_min and err_temp < 0
        )
        if not saturated:
            integral_err += err_temp * dt_s

        d_err = (err_temp - prev_err) / dt_s if dt_s > 0 else 0.0
        prev_err = err_temp

        q_j_target = q_j_ss + kp * err_temp + ki * integral_err + kd * d_err
        # Slew-rate limiting
        max_delta = s_max * dt_s
        delta_q = np.clip(q_j_target - q_j_current, -max_delta, max_delta)
        q_j_current = float(np.clip(q_j_current + delta_q, u_min, u_max))

        # ODE system RHS
        def cstr_ode(t: float, y: np.ndarray) -> List[float]:
            c_a, t_r, t_j = y[0], y[1], y[2]
            # Arrhenius rate law
            rate_k = k0 * np.exp(-e_over_r / t_r)
            reaction_rate = rate_k * c_a

            dc_a_dt = (q_feed / v_reactor) * (c_af - c_a) - reaction_rate
            dt_r_dt = (
                (q_feed / v_reactor) * (t_f - t_r)
                + (-delta_h / rho_cp) * reaction_rate
                - (ua / (v_reactor * rho_cp)) * (t_r - t_j)
            )
            dt_j_dt = (q_j_current / v_jacket) * (t_jf - t_j) + (
                ua / (v_jacket * rho_j_cp_j)
            ) * (t_r - t_j)
            return [dc_a_dt, dt_r_dt, dt_j_dt]

        sol = solve_ivp(
            fun=cstr_ode,
            t_span=(t_curr, t_curr + dt_s),
            y0=state,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
        )
        state = sol.y[:, -1]
        output_t.append(float(state[1]))
        control_u.append(float(q_j_current))

    logger.info("Simulation completed: final T=%.2f K", output_t[-1])
    return {
        "system_id": system_id,
        "time": time_points,
        "output": output_t,
        "control_signal": control_u,
        "setpoint": [float(setpoint)] * len(time_points),
    }


# -----------------------------------------------------------------------------
# Tool 2: compute_transient_metrics
# -----------------------------------------------------------------------------
@mcp.tool()
def compute_transient_metrics(
    time_series: List[float],
    output_series: List[float],
    setpoint: float,
    control_series: List[float],
) -> Dict[str, float]:
    """Computes comprehensive control transient, integral error, and actuator metrics.

    Evaluates:
        - Rise Time (tr): Time to first reach 90% of setpoint change.
        - Settling Time (ts): Time after which output permanently stays within +-2% band.
        - Percent Overshoot (Mp): Maximum percentage deviation above setpoint step.
        - IAE: Integral of Absolute Error.
        - ISE: Integral of Squared Error.
        - ITAE: Integral of Time-weighted Absolute Error.
        - Total Variation (TV): Sum of absolute successive changes in control effort.

    Args:
        time_series: Array of discrete time points [s].
        output_series: System output response values y(t).
        setpoint: Target steady-state reference value.
        control_series: Control input signal values u(t).

    Returns:
        Dictionary mapping metric names to their calculated float values.

    Raises:
        ValueError: If series lengths do not match or are shorter than 2 points.
    """
    logger.info("compute_transient_metrics invoked with %d points", len(time_series))

    if not (len(time_series) == len(output_series) == len(control_series)):
        raise ValueError(
            f"Input series lengths must match: time ({len(time_series)}), "
            f"output ({len(output_series)}), control ({len(control_series)})."
        )
    if len(time_series) < 2:
        raise ValueError("At least 2 points are required to compute transient metrics.")

    t_arr = np.array(time_series, dtype=np.float64)
    y_arr = np.array(output_series, dtype=np.float64)
    u_arr = np.array(control_series, dtype=np.float64)

    error = setpoint - y_arr
    step_delta = setpoint - y_arr[0]

    # Integral error metrics via trapezoidal rule
    iae = float(np.trapezoid(np.abs(error), t_arr))
    ise = float(np.trapezoid(error**2, t_arr))
    itae = float(np.trapezoid(t_arr * np.abs(error), t_arr))

    # Total Variation (TV) of control effort
    total_variation = float(np.sum(np.abs(np.diff(u_arr))))

    # Percent Overshoot (Mp)
    if abs(step_delta) > 1e-9:
        if step_delta > 0:
            peak_val = float(np.max(y_arr))
            overshoot = max(0.0, (peak_val - setpoint) / abs(step_delta) * 100.0)
        else:
            min_val = float(np.min(y_arr))
            overshoot = max(0.0, (setpoint - min_val) / abs(step_delta) * 100.0)
    else:
        overshoot = 0.0

    # Rise Time (tr): 10% to 90% of setpoint step
    if abs(step_delta) > 1e-9:
        target_10 = y_arr[0] + 0.1 * step_delta
        target_90 = y_arr[0] + 0.9 * step_delta

        if step_delta > 0:
            idx_10 = np.where(y_arr >= target_10)[0]
            idx_90 = np.where(y_arr >= target_90)[0]
        else:
            idx_10 = np.where(y_arr <= target_10)[0]
            idx_90 = np.where(y_arr <= target_90)[0]

        if len(idx_10) > 0 and len(idx_90) > 0:
            rise_time = float(t_arr[idx_90[0]] - t_arr[idx_10[0]])
            if rise_time < 0.0:
                rise_time = float(t_arr[idx_90[0]])
        elif len(idx_90) > 0:
            rise_time = float(t_arr[idx_90[0]])
        else:
            rise_time = float(t_arr[-1])
    else:
        rise_time = 0.0

    # Settling Time (ts, 2% tolerance band)
    band = (
        0.02 * abs(step_delta)
        if abs(step_delta) > 1e-9
        else 0.02 * max(abs(setpoint), 1.0)
    )
    outside_band = np.where(np.abs(y_arr - setpoint) > band)[0]
    if len(outside_band) == 0:
        settling_time = 0.0
    else:
        last_outside_idx = outside_band[-1]
        settling_time = float(t_arr[last_outside_idx])

    metrics = {
        "rise_time": rise_time,
        "settling_time": settling_time,
        "overshoot_pct": overshoot,
        "iae": iae,
        "ise": ise,
        "itae": itae,
        "total_variation": total_variation,
    }
    logger.info("Computed metrics: %s", metrics)
    return metrics


# -----------------------------------------------------------------------------
# Tool 3: tune_classical_pid
# -----------------------------------------------------------------------------
@mcp.tool()
def tune_classical_pid(
    k: float,
    tau: float,
    theta: float,
    method: str = "imc",
) -> Dict[str, float]:
    """Calculates analytical PID controller gains for a First-Order Plus Dead Time (FOPTD) model.

    Supports IMC (Internal Model Control / Skogestad SIMC), Ziegler-Nichols (Z-N),
    and Cohen-Coon (C-C) tuning relations. Guarded against negative dead times,
    zero process gains, and division by zero singularities.

    FOPTD Model Transfer Function:
        G(s) = [k / (tau * s + 1)] * exp(-theta * s)

    Args:
        k: Static process gain (non-zero).
        tau: Dominant process time constant [time_unit] (strictly positive).
        theta: Apparent process dead time / transport delay [time_unit] (non-negative).
        method: Tuning rule ('imc', 'ziegler_nichols', 'cohen_coon').

    Returns:
        Dictionary containing:
            - kp: Proportional gain.
            - ti: Integral time constant [time_unit].
            - td: Derivative time constant [time_unit].
            - ki: Integral gain (kp / ti).
            - kd: Derivative gain (kp * td).

    Raises:
        ValueError: On zero gain, non-positive tau, negative theta, zero theta in
            formulas requiring division by theta, or unsupported method.
    """
    logger.info(
        "tune_classical_pid called | k=%.4f, tau=%.4f, theta=%.4f, method=%s",
        k,
        tau,
        theta,
        method,
    )

    if abs(k) < 1e-12:
        raise ValueError("Process gain 'k' cannot be zero.")
    if tau <= 0.0:
        raise ValueError("Time constant 'tau' must be strictly positive.")
    if theta < 0.0:
        raise ValueError("Dead time 'theta' cannot be negative.")

    norm_method = method.lower().strip()
    valid_methods = {"imc", "ziegler_nichols", "cohen_coon"}
    if norm_method not in valid_methods:
        raise ValueError(
            f"Unsupported tuning method '{method}'. Supported methods: {sorted(list(valid_methods))}."
        )

    # Singularities when theta == 0 for formulas dividing by theta
    if theta == 0.0 and norm_method in {"ziegler_nichols", "cohen_coon"}:
        raise ValueError(
            f"Dead time 'theta' must be strictly positive for '{norm_method}' tuning to avoid division by zero."
        )

    abs_k = abs(k)

    if norm_method == "imc":
        # Skogestad SIMC rule: tau_c = max(theta, 0.1 * tau) or standard tau_c = theta
        tau_c = max(theta, 0.1 * tau) if theta > 0.0 else 0.1 * tau
        kp = float((1.0 / abs_k) * (tau / (tau_c + theta)))
        ti = float(min(tau, 8.0 * (tau_c + theta)))
        td = 0.0

    elif norm_method == "ziegler_nichols":
        # Ziegler-Nichols Open-Loop Reaction Curve
        kp = float((1.2 / abs_k) * (tau / theta))
        ti = float(2.0 * theta)
        td = float(0.5 * theta)

    elif norm_method == "cohen_coon":
        # Cohen-Coon 3-term relations
        r_ratio = theta / tau
        kp = float((1.0 / abs_k) * (tau / theta) * (4.0 / 3.0 + (1.0 / 4.0) * r_ratio))
        ti = float(theta * (32.0 + 6.0 * r_ratio) / (13.0 + 8.0 * r_ratio))
        td = float(theta * 4.0 / (11.0 + 2.0 * r_ratio))

    ki = float(kp / ti) if ti > 0.0 else 0.0
    kd = float(kp * td)

    tuning_gains = {
        "kp": kp,
        "ti": ti,
        "td": td,
        "ki": ki,
        "kd": kd,
    }
    logger.info("Synthesized gains: %s", tuning_gains)
    return tuning_gains


if __name__ == "__main__":
    if _HAS_FASTMCP and FastMCP is not None:
        logger.info("Starting ThermalControlServer FastMCP...")
        mcp.run()
    else:
        logger.info("FastMCP not installed. Exiting.")
