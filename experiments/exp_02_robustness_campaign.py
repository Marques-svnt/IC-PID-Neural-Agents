"""Comprehensive robustness, parameter uncertainty, and disturbance campaign for CSTR.

Evaluates classical PID controllers (Ziegler-Nichols, Cohen-Coon, Skogestad SIMC,
and Constrained Optimal ITAE) across:
1. Unmeasured load disturbance rejection (feed temperature and concentration shocks).
2. Heat exchanger fouling and parameter drift (UA degradation from 100% to 60%).
3. Sensor measurement noise sweep and actuator wear Pareto frontier (IAE vs TV).
4. Wide-range multi-operating point non-linear setpoint tracking.

Generates publication-quality figures and booktabs LaTeX tables for Article 1.
"""

import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.classical_control.closed_loop import ClosedLoopSimulator
from src.classical_control.foptd import identify_foptd
from src.classical_control.pid_controller import AntiWindupMethod, PIDController, PIDGains
from src.classical_control.tuning_analytical import TuningRule, tune_analytical
from src.classical_control.tuning_optimization import (
    OptimizationCriterion,
    tune_by_optimization,
)
from src.sim_core.actuators import ActuatorLimits
from src.sim_core.cstr_plant import CSTRPlant, CSTRState
from src.sim_core.integrator import NumericalIntegrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("exp_02_robustness")

# Matplotlib formatting setup
AVAILABLE_STYLES = plt.style.available
STYLE = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in AVAILABLE_STYLES else "default"
plt.style.use(STYLE)

CONTROLLER_COLORS = {
    "Ziegler-Nichols": "#1f77b4",
    "Cohen-Coon": "#ff7f0e",
    "Skogestad SIMC": "#2ca02c",
    "Optimal ITAE ($M_s \\leq 1.6$)": "#d62728",
}


def setup_controllers(
    plant: CSTRPlant, ss: CSTRState, q_j_ss: float
) -> dict[str, PIDGains]:
    """Identifies FOPTD model and tunes all four PID strategies.

    Args:
        plant: CSTR plant model.
        ss: Nominal steady state.
        q_j_ss: Nominal steady-state coolant flow rate [L/min].

    Returns:
        Dictionary mapping controller name to PIDGains.
    """
    logger.info("Performing step test identification (+10 L/min)...")
    integrator = NumericalIntegrator(method="RK45")
    delta_q = 10.0
    id_res = integrator.simulate_open_loop(
        plant=plant,
        initial_state=ss,
        t_span=(0.0, 6.0),
        dt=0.02,
        u_func=lambda t: q_j_ss + delta_q,
    )

    foptd, _ = identify_foptd(
        t=id_res.t,
        y=id_res.states[:, 1],
        delta_u=delta_q,
        y0=ss.T,
    )
    logger.info(
        "FOPTD: K=%.4f K/(L/min), tau=%.3f min, theta=%.3f min",
        foptd.k_p,
        foptd.tau,
        foptd.theta,
    )

    return {
        "Ziegler-Nichols": tune_analytical(foptd, rule=TuningRule.ZIEGLER_NICHOLS_PID),
        "Cohen-Coon": tune_analytical(foptd, rule=TuningRule.COHEN_COON_PID),
        "Skogestad SIMC": tune_analytical(foptd, rule=TuningRule.SKOGESTAD_SIMC_PID, tau_c=0.4),
        "Optimal ITAE ($M_s \\leq 1.6$)": tune_by_optimization(
            plant=plant,
            nominal_state=ss,
            q_j_ss=q_j_ss,
            target_dT=-2.0,
            criterion=OptimizationCriterion.ITAE,
            max_ms=1.6,
            t_sim=6.0,
            dt=0.02,
        ),
    }


def export_latex_table(
    data: list[dict[str, Any]],
    output_path: Path,
    caption: str,
    label: str,
) -> None:
    """Exports structured data to publication-grade LaTeX booktabs table.

    Args:
        data: List of dictionary records.
        output_path: Target path for .tex file.
        caption: Table caption.
        label: LaTeX label reference.
    """
    df = pd.DataFrame(data)
    cols = " & ".join(df.columns)
    rows_list = []
    for _, row in df.iterrows():
        rows_list.append(" & ".join(str(val) for val in row.values) + r" \\")
    rows_str = "\n".join(rows_list)
    col_align = "l" + "c" * (len(df.columns) - 1)

    latex_table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{col_align}}}
\\toprule
{cols} \\\\
\\midrule
{rows_str}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_table, encoding="utf-8")
    logger.info("Saved LaTeX table to: %s", output_path)


def run_load_disturbance_campaign(
    plant: CSTRPlant,
    ss: CSTRState,
    q_j_ss: float,
    controllers: dict[str, PIDGains],
    limits: ActuatorLimits,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    """Evaluates regulatory response against unmeasured feed shocks.

    Args:
        plant: CSTR physical plant.
        ss: Nominal steady-state.
        q_j_ss: Baseline coolant flow [L/min].
        controllers: Tuned PID gains dict.
        limits: Actuator physical saturation limits.
        figures_dir: Output path for plots.
        tables_dir: Output path for LaTeX tables.
    """
    logger.info("=== Running Load Disturbance Rejection Campaign ===")
    sim = ClosedLoopSimulator()
    t_span = (0.0, 15.0)
    dt = 0.02

    # Disturbance timeline:
    # t in [0, 2): Nominal steady state
    # t in [2, 8): Feed temperature surge Delta T_f = +5.0 K
    # t in [8, 15]: Additional feed concentration surge Delta C_Af = +0.2 mol/L (+20%)
    def disturbance_profile(t: float) -> dict[str, float]:
        dist = {}
        if 2.0 <= t < 8.0:
            dist["T_f"] = plant.params.T_f + 5.0
        elif t >= 8.0:
            dist["T_f"] = plant.params.T_f + 5.0
            dist["C_Af"] = plant.params.C_Af + 0.2
        return dist

    sim_results = {}
    table_rows = []

    for name, gains in controllers.items():
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
            t_span=t_span,
            dt=dt,
            setpoint_func=lambda t: ss.T,  # Strictly regulatory control at nominal T
            disturbance_func=disturbance_profile,
            noise_std=0.05,
            seed=42,
        )
        sim_results[name] = res

        # Peak deviation from setpoint during disturbance
        dev = np.abs(res.t_pv - ss.T)
        peak_dev = np.max(dev[res.t >= 2.0])

        table_rows.append({
            "Method": name,
            "IAE": f"{res.metrics.iae:.2f}",
            "ITAE": f"{res.metrics.itae:.2f}",
            "TV (L/min)": f"{res.metrics.tv:.1f}",
            "Peak Dev (K)": f"{peak_dev:.2f}",
            "Max Valve (L/min)": f"{np.max(res.u_applied):.1f}",
        })

    export_latex_table(
        data=table_rows,
        output_path=tables_dir / "disturbance_rejection_table.tex",
        caption="Regulatory performance under combined feed temperature ($+5$~K) "
        "and concentration ($+20\\%$) load disturbances.",
        label="tab:load_disturbance",
    )

    # Plot trajectories
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True, dpi=300)

    # Reference nominal line
    sample_res = next(iter(sim_results.values()))
    ax1.plot(sample_res.t, sample_res.setpoint, "k--", label="Target $T_{sp}$", linewidth=1.5)

    # Annotate disturbance injection points
    ax1.axvline(2.0, color="gray", linestyle="--", alpha=0.7)
    ax1.text(2.1, 401.0, "$\\Delta T_f = +5$ K", fontsize=9, color="gray", fontweight="bold")
    ax1.axvline(8.0, color="gray", linestyle="--", alpha=0.7)
    ax1.text(8.1, 401.0, "$\\Delta C_{Af} = +20\\%$", fontsize=9, color="gray", fontweight="bold")

    for name, res in sim_results.items():
        color = CONTROLLER_COLORS[name]
        ax1.plot(res.t, res.t_pv, label=name, color=color, linewidth=1.8)
        ax2.plot(res.t, res.u_applied, label=name, color=color, linewidth=1.8)

    ax1.set_ylabel("Reactor Temperature $T$ (K)", fontsize=11)
    ax1.set_title(
        "Load Disturbance Rejection: $+5$ K Feed Temp Surge & $+20\\%$ Feed Conc Surge",
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend(loc="upper right", frameon=True, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.axhline(limits.u_max, color="red", linestyle=":", label="Valve Limits ($u_{max}, u_{min}$)")
    ax2.axhline(limits.u_min, color="red", linestyle=":")
    ax2.set_ylabel("Coolant Flow $q_j$ (L/min)", fontsize=11)
    ax2.set_xlabel("Time $t$ (min)", fontsize=11)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_png = figures_dir / "fig2_disturbance_rejection.png"
    fig_pdf = figures_dir / "fig2_disturbance_rejection.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    logger.info("Saved fig2: %s and %s", fig_png, fig_pdf)


def run_fouling_campaign(
    plant: CSTRPlant,
    ss: CSTRState,
    q_j_ss: float,
    controllers: dict[str, PIDGains],
    limits: ActuatorLimits,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    """Evaluates stability and performance degradation under jacket thermal fouling (UA loss).

    Args:
        plant: CSTR physical plant.
        ss: Nominal steady-state.
        q_j_ss: Baseline coolant flow [L/min].
        controllers: Tuned PID gains dict.
        limits: Actuator physical saturation limits.
        figures_dir: Output path for plots.
        tables_dir: Output path for LaTeX tables.
    """
    logger.info("=== Running Thermal Fouling Campaign (UA Degradation) ===")
    sim = ClosedLoopSimulator()
    t_span = (0.0, 10.0)
    dt = 0.02
    target_T = ss.T - 2.0

    fouling_levels = [1.0, 0.9, 0.8, 0.7, 0.6]  # UA / UA_nominal
    table_rows = []

    # Store trajectories for nominal (1.0) and severe fouling (0.7) for plotting
    plot_data: dict[str, dict[str, Any]] = {"1.0": {}, "0.7": {}}

    for name, gains in controllers.items():
        for f_ratio in fouling_levels:
            ua_actual = plant.params.UA * f_ratio
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
                t_span=t_span,
                dt=dt,
                setpoint_func=lambda t: target_T if t >= 0.5 else ss.T,
                disturbance_func=lambda t: {"UA": ua_actual},
                noise_std=0.05,
                seed=42,
            )

            if f_ratio in [1.0, 0.7]:
                plot_data[str(f_ratio)][name] = res

            table_rows.append({
                "Method": name,
                "$UA / UA_{\\text{nom}}$": f"{int(f_ratio * 100)}\\%",
                "IAE": f"{res.metrics.iae:.2f}",
                "ITAE": f"{res.metrics.itae:.2f}",
                "TV (L/min)": f"{res.metrics.tv:.1f}",
                "$M_p$ (\\%)": f"{res.metrics.overshoot_pct:.1f}",
            })

    export_latex_table(
        data=table_rows,
        output_path=tables_dir / "fouling_sensitivity_table.tex",
        caption="Impact of heat exchanger fouling ($UA$ reduction from $100\\%$ to $60\\%$) "
        "on closed-loop tracking metrics.",
        label="tab:fouling_sensitivity",
    )

    # Plot comparison: Nominal (100% UA) vs Severe Fouling (70% UA)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, dpi=300)

    sample_res = next(iter(plot_data["1.0"].values()))

    # Panel (a): T at 100% UA
    ax1.plot(sample_res.t, sample_res.setpoint, "k--", label="Target SP", linewidth=1.5)
    for name, res in plot_data["1.0"].items():
        ax1.plot(res.t, res.t_pv, label=name, color=CONTROLLER_COLORS[name], linewidth=1.8)
    ax1.set_title("(a) Clean Jacket ($100\\% UA$): Temperature $T$", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Temperature $T$ (K)", fontsize=10)
    ax1.legend(loc="lower right", frameon=True, fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel (b): T at 70% UA
    ax2.plot(sample_res.t, sample_res.setpoint, "k--", label="Target SP", linewidth=1.5)
    for name, res in plot_data["0.7"].items():
        ax2.plot(res.t, res.t_pv, label=name, color=CONTROLLER_COLORS[name], linewidth=1.8)
    ax2.set_title("(b) Fouled Jacket ($70\\% UA$): Temperature $T$", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True, fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Panel (c): Coolant Flow at 100% UA
    for name, res in plot_data["1.0"].items():
        ax3.plot(res.t, res.u_applied, label=name, color=CONTROLLER_COLORS[name], linewidth=1.8)
    ax3.axhline(limits.u_max, color="red", linestyle=":", label="Valve Limits")
    ax3.set_title("(c) Clean Jacket ($100\\% UA$): Coolant $q_j$", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Coolant Flow $q_j$ (L/min)", fontsize=10)
    ax3.set_xlabel("Time $t$ (min)", fontsize=10)
    ax3.legend(loc="upper right", frameon=True, fontsize=8)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # Panel (d): Coolant Flow at 70% UA
    for name, res in plot_data["0.7"].items():
        ax4.plot(res.t, res.u_applied, label=name, color=CONTROLLER_COLORS[name], linewidth=1.8)
    ax4.axhline(limits.u_max, color="red", linestyle=":", label="Valve Limits")
    ax4.set_title("(d) Fouled Jacket ($70\\% UA$): Coolant $q_j$", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Time $t$ (min)", fontsize=10)
    ax4.legend(loc="upper right", frameon=True, fontsize=8)
    ax4.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_png = figures_dir / "fig3_fouling_robustness.png"
    fig_pdf = figures_dir / "fig3_fouling_robustness.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    logger.info("Saved fig3: %s and %s", fig_png, fig_pdf)


def run_noise_and_pareto_campaign(
    plant: CSTRPlant,
    ss: CSTRState,
    q_j_ss: float,
    controllers: dict[str, PIDGains],
    limits: ActuatorLimits,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    """Quantifies sensor noise sensitivity and maps the IAE vs TV Pareto frontier.

    Args:
        plant: CSTR physical plant.
        ss: Nominal steady-state.
        q_j_ss: Baseline coolant flow [L/min].
        controllers: Tuned PID gains dict.
        limits: Actuator physical saturation limits.
        figures_dir: Output path for plots.
        tables_dir: Output path for LaTeX tables.
    """
    logger.info("=== Running Sensor Noise & Actuator Chattering (Pareto) Campaign ===")
    sim = ClosedLoopSimulator()
    t_span = (0.0, 10.0)
    dt = 0.02
    target_T = ss.T - 2.0

    noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0]  # Standard deviation of temperature measurement [K]
    n_monte_carlo = 5

    # Structure to hold metrics: results[name][sigma] = {"iae": [...], "tv": [...]}
    results: dict[str, dict[float, dict[str, list[float]]]] = {
        name: {sigma: {"iae": [], "tv": []} for sigma in noise_levels}
        for name in controllers
    }

    for name, gains in controllers.items():
        for sigma in noise_levels:
            for seed in range(n_monte_carlo):
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
                    t_span=t_span,
                    dt=dt,
                    setpoint_func=lambda t: target_T if t >= 0.5 else ss.T,
                    noise_std=sigma,
                    seed=100 + seed,
                )
                results[name][sigma]["iae"].append(res.metrics.iae)
                results[name][sigma]["tv"].append(res.metrics.tv)

    # Aggregate into summary table
    table_rows = []
    for name in controllers:
        for sigma in noise_levels:
            mean_iae = float(np.mean(results[name][sigma]["iae"]))
            std_iae = float(np.std(results[name][sigma]["iae"]))
            mean_tv = float(np.mean(results[name][sigma]["tv"]))
            std_tv = float(np.std(results[name][sigma]["tv"]))

            table_rows.append({
                "Method": name,
                "Noise $\\sigma$ (K)": f"{sigma:.1f}",
                "IAE (mean $\\pm$ std)": f"{mean_iae:.2f} $\\pm$ {std_iae:.2f}",
                "TV (L/min)": f"{mean_tv:.1f} $\\pm$ {std_tv:.1f}",
            })

    export_latex_table(
        data=table_rows,
        output_path=tables_dir / "noise_chattering_table.tex",
        caption="Sensitivity of tracking error (IAE) and actuator effort (TV) "
        "to measurement noise levels across $5$ Monte Carlo runs.",
        label="tab:noise_sensitivity",
    )

    # Plot Pareto Frontier: IAE vs TV
    plt.figure(figsize=(8.0, 6.0), dpi=300)
    markers = ["o", "s", "^", "D"]

    for (name, marker) in zip(controllers.keys(), markers):
        iaes = [np.mean(results[name][sig]["iae"]) for sig in noise_levels]
        tvs = [np.mean(results[name][sig]["tv"]) for sig in noise_levels]
        color = CONTROLLER_COLORS[name]

        plt.plot(tvs, iaes, marker=marker, linestyle="-", label=name, color=color, markersize=7)

        # Annotate noise level endpoints
        plt.annotate(
            f"$\\sigma={noise_levels[0]}$",
            (tvs[0], iaes[0]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color=color,
        )
        plt.annotate(
            f"$\\sigma={noise_levels[-1]}$",
            (tvs[-1], iaes[-1]),
            textcoords="offset points",
            xytext=(5, -10),
            fontsize=8,
            color=color,
        )

    plt.xlabel("Actuator Total Variation $TV$ (L/min) [Wear / Chattering]", fontsize=11)
    plt.ylabel("Integral Absolute Error $IAE$ (K$\\cdot$min) [Tracking Error]", fontsize=11)
    plt.title(
        "Pareto Trade-off: Tracking Quality vs Actuator Wear across Noise Levels",
        fontsize=12,
        fontweight="bold",
    )
    plt.legend(loc="upper left", frameon=True, fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    fig_png = figures_dir / "fig4_pareto_iae_tv.png"
    fig_pdf = figures_dir / "fig4_pareto_iae_tv.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    logger.info("Saved fig4: %s and %s", fig_png, fig_pdf)


def run_multistep_campaign(
    plant: CSTRPlant,
    ss: CSTRState,
    q_j_ss: float,
    controllers: dict[str, PIDGains],
    limits: ActuatorLimits,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    """Evaluates non-linear multi-operating point tracking across distinct reaction regimes.

    Args:
        plant: CSTR physical plant.
        ss: Nominal steady-state.
        q_j_ss: Baseline coolant flow [L/min].
        controllers: Tuned PID gains dict.
        limits: Actuator physical saturation limits.
        figures_dir: Output path for plots.
        tables_dir: Output path for LaTeX tables.
    """
    logger.info("=== Running Multi-Operating Point Non-linear Tracking Campaign ===")
    sim = ClosedLoopSimulator()
    t_span = (0.0, 24.0)
    dt = 0.02

    # Setpoint steps:
    # 0 to 4 min: Nominal (396.65 K)
    # 4 to 10 min: Step down to 390.0 K (cooling regime, lower reaction rate)
    # 10 to 17 min: Step up to 402.0 K (strongly exothermic runaway-sensitive regime)
    # 17 to 24 min: Return to nominal (396.65 K)
    def multistep_profile(t: float) -> float:
        if t < 4.0:
            return ss.T
        elif t < 10.0:
            return 390.0
        elif t < 17.0:
            return 402.0
        else:
            return ss.T

    sim_results = {}
    table_rows = []

    for name, gains in controllers.items():
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
            t_span=t_span,
            dt=dt,
            setpoint_func=multistep_profile,
            noise_std=0.05,
            seed=42,
        )
        sim_results[name] = res

        table_rows.append({
            "Method": name,
            "Total IAE": f"{res.metrics.iae:.2f}",
            "Total ITAE": f"{res.metrics.itae:.2f}",
            "Total TV (L/min)": f"{res.metrics.tv:.1f}",
            "Max Valve (L/min)": f"{np.max(res.u_applied):.1f}",
            "Min Valve (L/min)": f"{np.min(res.u_applied):.1f}",
        })

    export_latex_table(
        data=table_rows,
        output_path=tables_dir / "multistep_tracking_table.tex",
        caption="Multi-operating point tracking across moderate (390~K), nominal (396.65~K), "
        "and highly exothermic (402~K) reaction regimes.",
        label="tab:multistep_tracking",
    )

    # Plot multi-step trajectories
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True, dpi=300)

    sample_res = next(iter(sim_results.values()))
    ax1.plot(sample_res.t, sample_res.setpoint, "k--", label="Target $T_{sp}$", linewidth=1.5)

    for name, res in sim_results.items():
        color = CONTROLLER_COLORS[name]
        ax1.plot(res.t, res.t_pv, label=name, color=color, linewidth=1.8)
        ax2.plot(res.t, res.u_applied, label=name, color=color, linewidth=1.8)

    ax1.set_ylabel("Reactor Temperature $T$ (K)", fontsize=11)
    ax1.set_title(
        "Non-linear Wide-Range Tracking: $396.65\\text{ K} \\rightarrow 390\\text{ K} "
        "\\rightarrow 402\\text{ K} \\rightarrow 396.65\\text{ K}$",
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend(loc="lower right", frameon=True, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.axhline(limits.u_max, color="red", linestyle=":", label="Valve Limits ($u_{max}, u_{min}$)")
    ax2.axhline(limits.u_min, color="red", linestyle=":")
    ax2.set_ylabel("Coolant Flow $q_j$ (L/min)", fontsize=11)
    ax2.set_xlabel("Time $t$ (min)", fontsize=11)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_png = figures_dir / "fig5_multistep_tracking.png"
    fig_pdf = figures_dir / "fig5_multistep_tracking.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    logger.info("Saved fig5: %s and %s", fig_png, fig_pdf)


def run_campaign() -> None:
    """Orchestrates all four experimental campaigns for Article 1."""
    project_root = Path(__file__).resolve().parent.parent
    paper_dir = project_root / "Artigos_Rascunhos" / "Artigo_01_PID_Neural_Comparativo"
    figures_dir = paper_dir / "figures"
    tables_dir = paper_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    plant = CSTRPlant()
    q_j_ss = 100.0
    ss = plant.find_steady_state(q_j=q_j_ss)
    limits = ActuatorLimits(u_min=0.0, u_max=300.0, max_slew_rate=80.0)

    controllers = setup_controllers(plant, ss, q_j_ss)

    logger.info("Starting comprehensive robustness campaign...")
    run_load_disturbance_campaign(plant, ss, q_j_ss, controllers, limits, figures_dir, tables_dir)
    run_fouling_campaign(plant, ss, q_j_ss, controllers, limits, figures_dir, tables_dir)
    run_noise_and_pareto_campaign(plant, ss, q_j_ss, controllers, limits, figures_dir, tables_dir)
    run_multistep_campaign(plant, ss, q_j_ss, controllers, limits, figures_dir, tables_dir)
    logger.info("Robustness campaign finished successfully!")


if __name__ == "__main__":
    run_campaign()
