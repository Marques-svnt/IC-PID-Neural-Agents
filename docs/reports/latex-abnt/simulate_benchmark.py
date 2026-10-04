"""
Simulacao completa do benchmark e geracao dos dados para visualizacao PGFPlots / TikZ.
"""

import numpy as np
from scipy.integrate import solve_ivp

# Parametros nominais
V = 100.0
V_j = 20.0
q = 100.0
C_Af = 1.0
T_f0 = 350.0
T_jf = 300.0
k0 = 7.2e10
E_over_R = 8750.0
dH = -5.0e4
rho_Cp = 500.0
rho_j_Cp_j = 500.0
UA_nom = 5.0e4

C_A_ss = 0.0502
T_ss = 396.65
T_j_ss = 348.33
q_j_ss = 100.0

u_min = 0.0
u_max = 300.0
S_max = 80.0 # L/min^2 -> delta_u_max = S_max * dt = 1.6 L/min
dt = 0.02
t_final = 20.0
steps = int(t_final / dt)
time_vec = np.linspace(0, t_final, steps + 1)

def run_simulation(kp, ki, kd, name="Controller"):
    # Initial state
    y = np.array([C_A_ss, T_ss, T_j_ss])

    # Storage
    T_arr = np.zeros(steps + 1)
    qj_arr = np.zeros(steps + 1)
    Tsp_arr = np.zeros(steps + 1)

    T_arr[0] = y[1]
    qj_arr[0] = q_j_ss
    Tsp_arr[0] = T_ss + 5.0 # step at t=0

    # PID states
    integral = 0.0
    prev_error = (T_ss + 5.0) - y[1]
    prev_qj = q_j_ss
    D_filter = 0.0
    N = 10.0

    for k in range(steps):
        t = k * dt
        # Setpoint and disturbance
        T_sp = T_ss + 5.0 # step of +5K
        T_f = T_f0 if t < 10.0 else T_f0 + 10.0 # disturbance +10K at t=10 min
        Tsp_arr[k] = T_sp

        # Current temp
        T_curr = y[1]
        error = T_sp - T_curr

        # Clamping anti-windup:
        # Check if saturated
        is_saturated = (prev_qj >= u_max and error < 0) or (prev_qj <= u_min and error > 0)
        # Note reverse action: if T > T_sp (error < 0), controller wants MORE cooling (qj increases towards u_max)
        if not is_saturated:
            integral += error * dt

        # Derivative filtered
        deriv = (error - prev_error) / dt if dt > 0 else 0.0

        # Raw PID command
        qj_raw = q_j_ss + kp * error + ki * integral + kd * deriv

        # Slew-rate and amplitude saturation
        delta_qj = np.clip(qj_raw - prev_qj, -S_max * dt, S_max * dt)
        qj_slew = prev_qj + delta_qj
        qj_sat = np.clip(qj_slew, u_min, u_max)

        # ODE integration for one step dt
        sol = solve_ivp(
            lambda tau, state: [
                (q / V) * (C_Af - state[0]) - k0 * np.exp(-E_over_R / state[1]) * state[0],
                (q / V) * (T_f - state[1]) + (-dH / rho_Cp) * k0 * np.exp(-E_over_R / state[1]) * state[0] - (UA_nom / (V * rho_Cp)) * (state[1] - state[2]),
                (qj_sat / V_j) * (T_jf - state[2]) + (UA_nom / (V_j * rho_j_Cp_j)) * (state[1] - state[2])
            ],
            [t, t + dt],
            y,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        y = sol.y[:, -1]

        # Clamp temperature to avoid numerical overflow if unstable
        if y[1] > 600.0 or y[1] < 200.0:
            y[1] = 600.0 if y[1] > 600.0 else 200.0

        prev_error = error
        prev_qj = qj_sat

        T_arr[k+1] = y[1]
        qj_arr[k+1] = qj_sat

    Tsp_arr[-1] = Tsp_arr[-2]

    # Compute metrics
    err = np.abs(Tsp_arr - T_arr)
    iae = np.trapezoid(err, time_vec)
    itae = np.trapezoid(time_vec * err, time_vec)
    ise = np.trapezoid((Tsp_arr - T_arr)**2, time_vec)
    tv = np.sum(np.abs(np.diff(qj_arr)))
    peak_T = np.max(T_arr[:int(10.0/dt)])
    overshoot = max(0.0, (peak_T - (T_ss + 5.0)) / 5.0 * 100.0)

    print(f"[{name}] IAE={iae:.2f}, ITAE={itae:.2f}, ISE={ise:.2f}, Mp={overshoot:.1f}%, TV={tv:.1f}, Max T={np.max(T_arr):.1f}K")
    return time_vec, T_arr, qj_arr

print("Running controllers...")
t, T_simc, q_simc = run_simulation(-7.03, -10.98, -0.22, "SIMC (tau_c=0.40)")
t, T_itae, q_itae = run_simulation(-14.21, -14.21, -0.01, "ITAE Otimo")
t, T_zn, q_zn = run_simulation(-44.56, -234.53, -2.09, "Ziegler-Nichols")
t, T_cc, q_cc = run_simulation(-50.86, -231.18, -1.68, "Cohen-Coon")
