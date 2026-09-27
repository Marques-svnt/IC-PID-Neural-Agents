"""Benchmark script comparing classical PID tuning methods on non-linear CSTR.

Generates publication figures and LaTeX tables for Article 1:
- Ziegler-Nichols (ZN)
- Cohen-Coon (CC)
- Skogestad SIMC
- Numerical Optimization (ITAE with Ms <= 1.6)
"""

import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.classical_control.closed_loop import ClosedLoopSimulator
from src.classical_control.foptd import identify_foptd
from src.classical_control.pid_controller import AntiWindupMethod, PIDController, PIDGains
from src.classical_control.tuning_analytical import TuningRule, tune_analytical
from src.classical_control.tuning_optimization import (
    OptimizationCriterion,
    calculate_maximum_sensitivity,
    tune_by_optimization,
)
from src.evaluation.plot_styles import (
    IEEE_LINESTYLES,
    IEEE_PALETTE,
    get_figure_dimensions,
    setup_ieee_style,
)
from src.sim_core.actuators import ActuatorLimits
from src.sim_core.cstr_plant import CSTRPlant
from src.sim_core.integrator import NumericalIntegrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("exp_01_benchmark")


def run_benchmark() -> None:
    """Executes full comparative campaign, producing figures and LaTeX tables."""
    project_root = Path(__file__).resolve().parent.parent
    paper_dir = project_root / "Artigos_Rascunhos" / "Artigo_01_PID_Neural_Comparativo"
    figures_dir = paper_dir / "figures"
    tables_dir = paper_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    plant = CSTRPlant()
    integrator = NumericalIntegrator(method="RK45")
    q_j_ss = 100.0
    ss = plant.find_steady_state(q_j=q_j_ss)

    logger.info("Nominal steady state: T=%.2f K, C_A=%.4f mol/L", ss.T, ss.C_A)

    # 1. Open-loop step identification
    logger.info("Executing open-loop identification step (+10 L/min)...")
    delta_q = 10.0
    t_id_span = (0.0, 6.0)
    dt = 0.02

    id_res = integrator.simulate_open_loop(
        plant=plant,
        initial_state=ss,
        t_span=t_id_span,
        dt=dt,
        u_func=lambda t: q_j_ss + delta_q,
    )

    foptd, rmse = identify_foptd(
        t=id_res.t,
        y=id_res.states[:, 1],
        delta_u=delta_q,
        y0=ss.T,
    )
    logger.info(
        "Identified FOPTD: K=%.4f, tau=%.3f min, theta=%.3f min",
        foptd.k_p,
        foptd.tau,
        foptd.theta,
    )

    # 2. Compute PID gains for each method
    controllers: dict[str, PIDGains] = {
        "Ziegler-Nichols": tune_analytical(foptd, rule=TuningRule.ZIEGLER_NICHOLS_PID),
        "Cohen-Coon": tune_analytical(foptd, rule=TuningRule.COHEN_COON_PID),
        "Skogestad SIMC": tune_analytical(foptd, rule=TuningRule.SKOGESTAD_SIMC_PID, tau_c=0.4),
        "Optimal ITAE ($M_s \\leq 1.6$)": tune_by_optimization(
            plant=plant,
            nominal_state=ss,
            q_j_ss=q_j_ss,
            criterion=OptimizationCriterion.ITAE,
            max_ms=1.6,
        ),
    }

    # 3. Simulate Closed Loop for each controller (Deterministic Nominal Benchmark)
    sim = ClosedLoopSimulator(integrator=integrator)
    t_sim_span = (0.0, 10.0)
    target_T = ss.T - 2.0  # Setpoint step of -2 K

    def setpoint_profile(t: float) -> float:
        return target_T

    sim_results = {}
    table_rows = []

    limits = ActuatorLimits(u_min=0.0, u_max=300.0, max_slew_rate=80.0)

    for name, gains in controllers.items():
        logger.info("Simulating closed loop for: %s", name)
        pid = PIDController(
            gains=gains,
            actuator_limits=limits,
            anti_windup=AntiWindupMethod.CLAMPING,
            u_bias=q_j_ss,
        )

        res = sim.run(
            plant=plant,
            controller=pid,
            initial_state=ss,
            t_span=t_sim_span,
            dt=dt,
            setpoint_func=setpoint_profile,
            noise_std=0.0,  # Pure deterministic tracking to isolate nominal dynamics
            seed=42,
        )
        sim_results[name] = res

        ms = calculate_maximum_sensitivity(plant, ss, q_j_ss, gains)
        table_rows.append({
            "Method": name,
            r"\(K_p\)": f"{gains.kp:.2f}",
            r"\(T_i\text{ (min)}\)": f"{gains.ti:.2f}",
            r"\(T_d\text{ (min)}\)": f"{gains.td:.3f}",
            "IAE": f"{res.metrics.iae:.2f}",
            "ITAE": f"{res.metrics.itae:.2f}",
            "TV (L/min)": f"{res.metrics.tv:.1f}",
            r"\(M_p\text{ (\%)}\)": f"{res.metrics.overshoot_pct:.1f}",
            r"\(t_s\text{ (min)}\)": (
                f"{res.metrics.settling_time:.2f}"
                if res.metrics.settling_time is not None
                else "N/A"
            ),
            r"\(M_s\)": f"{ms:.2f}",
        })

    # 4. Generate LaTeX Table using clean booktabs
    df = pd.DataFrame(table_rows)
    cols = " & ".join(df.columns)
    rows_list = []
    for _, row in df.iterrows():
        rows_list.append(" & ".join(str(val) for val in row.values) + r" \\")
    rows_str = "\n".join(rows_list)
    col_align = "l" + "c" * (len(df.columns) - 1)

    latex_table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Comparative performance benchmark of classical PID tuning methods on non-linear CSTR.}}
\\label{{tab:classical_benchmark}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{{col_align}}}
\\toprule
{cols} \\\\
\\midrule
{rows_str}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}
"""
    table_file = tables_dir / "benchmark_table.tex"
    table_file.write_text(latex_table, encoding="utf-8")
    logger.info("Saved LaTeX table to: %s", table_file)

    # 5. Plot Publication-Quality Figures (IEEE Single-Column Standard)
    setup_ieee_style(single_column=True)
    fig_w, fig_h = get_figure_dimensions(columns=1, height_override=3.6)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True, dpi=300)

    # Plot setpoint
    sample_res = next(iter(sim_results.values()))
    ax1.plot(
        sample_res.t,
        sample_res.setpoint,
        color=IEEE_PALETTE["Setpoint"],
        linestyle=IEEE_LINESTYLES["Setpoint"],
        label="Setpoint ($T_{sp}$)",
        linewidth=1.2,
    )

    for name, res in sim_results.items():
        color = IEEE_PALETTE.get(name, "#333333")
        linestyle = IEEE_LINESTYLES.get(name, "-")
        ax1.plot(res.t, res.t_pv, label=name, color=color, linestyle=linestyle, linewidth=1.2)
        ax2.plot(res.t, res.u_applied, label=name, color=color, linestyle=linestyle, linewidth=1.2)

    ax1.set_ylabel("Reactor Temp. $T$ (K)", fontsize=8.5)
    ax1.legend(loc="lower right", frameon=True, fontsize=6.8, framealpha=0.9)
    ax1.set_title("Classical PID Benchmark: Setpoint Tracking", fontsize=9.0)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.axhline(
        limits.u_max,
        color=IEEE_PALETTE["Constraint"],
        linestyle=IEEE_LINESTYLES["Constraint"],
        label="Limits ($u_{\\min}, u_{\\max}$)",
        linewidth=1.0,
    )
    ax2.axhline(
        limits.u_min,
        color=IEEE_PALETTE["Constraint"],
        linestyle=IEEE_LINESTYLES["Constraint"],
        linewidth=1.0,
    )
    ax2.set_ylabel("Coolant Flow $q_j$ (L/min)", fontsize=8.5)
    ax2.set_xlabel("Time $t$ (min)", fontsize=8.5)
    ax2.legend(loc="upper right", frameon=True, fontsize=6.8, framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_png = figures_dir / "fig1_classical_benchmark.png"
    fig_pdf = figures_dir / "fig1_classical_benchmark.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    logger.info("Saved benchmark figures: %s and %s", fig_png, fig_pdf)


if __name__ == "__main__":
    run_benchmark()
