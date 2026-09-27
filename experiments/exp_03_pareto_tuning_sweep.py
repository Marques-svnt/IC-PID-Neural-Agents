"""Pareto frontier exploration via continuous tuning parameter sweep on non-linear CSTR.

Generates the authentic Pareto trade-off between setpoint tracking accuracy (IAE)
and actuator physical wear (TV) by:
1. Sweeping the Skogestad SIMC closed-loop time constant tau_c in [0.015, 2.5] min.
2. Performing scalarized multi-objective optimization across lambda in [0.05, 0.95].
3. Evaluating peak sensitivity Ms to delineate robustness boundaries (Ms <= 1.6).
4. Mapping discrete classical tunings (ZN, CC, SIMC, ITAE-opt) on the design space.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.classical_control.closed_loop import ClosedLoopSimulator
from src.classical_control.foptd import FOPTDModel, identify_foptd
from src.classical_control.pid_controller import AntiWindupMethod, PIDController, PIDGains
from src.classical_control.tuning_analytical import TuningRule, tune_analytical
from src.classical_control.tuning_optimization import (
    OptimizationCriterion,
    calculate_maximum_sensitivity,
    tune_by_optimization,
)
from src.evaluation.plot_styles import (
    IEEE_MARKERS,
    IEEE_PALETTE,
    get_figure_dimensions,
    setup_ieee_style,
)
from src.sim_core.actuators import ActuatorLimits
from src.sim_core.cstr_plant import CSTRPlant, CSTRState
from src.sim_core.integrator import NumericalIntegrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("exp_03_pareto")


def export_latex_table(
    data: List[Dict[str, Any]],
    output_path: Path,
    caption: str,
    label: str,
) -> None:
    """Exports structured data to a clean LaTeX booktabs table.

    Args:
        data: List of dictionary rows.
        output_path: Target .tex file path.
        caption: Table caption.
        label: LaTeX label for cross-referencing.
    """
    df = pd.DataFrame(data)
    cols = " & ".join(df.columns)
    rows_list = []
    for _, row in df.iterrows():
        rows_list.append(" & ".join(str(val) for val in row.values) + r" \\")
    rows_str = "\n".join(rows_list)
    col_align = "l" + "c" * (len(df.columns) - 1)

    latex_code = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
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
    output_path.write_text(latex_code, encoding="utf-8")
    logger.info("Saved LaTeX table to: %s", output_path)


def run_simc_sweep(
    plant: CSTRPlant,
    ss: CSTRState,
    q_j_ss: float,
    foptd: FOPTDModel,
    limits: ActuatorLimits,
    tau_c_values: np.ndarray,
) -> List[Dict[str, Any]]:
    """Sweeps the closed-loop time constant tau_c for Skogestad SIMC.

    Args:
        plant: CSTR physical plant.
        ss: Nominal steady state.
        q_j_ss: Steady-state coolant flow rate.
        foptd: Identified FOPTD model.
        limits: Valve physical limits.
        tau_c_values: Array of tau_c evaluation points [min].

    Returns:
        List of result dictionaries containing gains and performance metrics.
    """
    logger.info("Running Skogestad SIMC parameter sweep across %d points...", len(tau_c_values))
    sim = ClosedLoopSimulator()
    results = []

    for tc in tau_c_values:
        gains = tune_analytical(foptd, TuningRule.SKOGESTAD_SIMC_PID, tau_c=float(tc))
        ms = calculate_maximum_sensitivity(plant, ss, q_j_ss, gains)

        pid = PIDController(
            gains=gains,
            actuator_limits=limits,
            anti_windup=AntiWindupMethod.CLAMPING,
            u_bias=q_j_ss,
        )

        try:
            res = sim.run(
                plant=plant,
                controller=pid,
                initial_state=ss,
                t_span=(0.0, 10.0),
                dt=0.02,
                setpoint_func=lambda t: ss.T - 2.0,
            )
            iae = res.metrics.iae
            itae = res.metrics.itae
            tv = res.metrics.tv
            overshoot = res.metrics.overshoot_pct
            ts = res.metrics.settling_time if res.metrics.settling_time is not None else 10.0
        except Exception as e:
            logger.warning("Simulation failed for tau_c=%.3f: %s", tc, e)
            continue

        results.append({
            "tau_c": float(tc),
            "kp": gains.kp,
            "ti": gains.ti,
            "td": gains.td,
            "ms": ms,
            "iae": iae,
            "itae": itae,
            "tv": tv,
            "overshoot": overshoot,
            "settling_time": ts,
        })

    return results


def run_scalarized_multiobjective_sweep(
    plant: CSTRPlant,
    ss: CSTRState,
    q_j_ss: float,
    limits: ActuatorLimits,
    lambdas: np.ndarray,
    ref_iae: float,
    ref_tv: float,
) -> List[Dict[str, Any]]:
    """Runs scalarized multi-objective optimization: min lambda*(IAE/IAE0) + (1-lambda)*(TV/TV0).

    Args:
        plant: CSTR physical plant.
        ss: Nominal steady state.
        q_j_ss: Steady-state coolant flow rate.
        limits: Valve physical limits.
        lambdas: Array of trade-off weights in (0, 1).
        ref_iae: Baseline IAE for normalization.
        ref_tv: Baseline TV for normalization.

    Returns:
        List of Pareto-optimal result dictionaries.
    """
    logger.info(
        "Running scalarized multi-objective optimization across %d weights...",
        len(lambdas),
    )
    sim = ClosedLoopSimulator()
    results = []

    # Initial guess: moderate PID
    x0 = np.array([-4.5, 1.2, 0.15], dtype=np.float64)

    for lam in lambdas:
        def objective(params: np.ndarray) -> float:
            kp, ti, td = float(params[0]), float(params[1]), float(params[2])
            if kp >= 0.0 or ti <= 0.05 or td < 0.0:
                return 1e6

            candidate = PIDGains(kp=kp, ti=ti, td=td)
            ms = calculate_maximum_sensitivity(plant, ss, q_j_ss, candidate)
            # Penalty for violating robustness boundary Ms <= 2.0
            ms_pen = 1e3 * (ms - 2.0) ** 2 if ms > 2.0 else 0.0

            pid = PIDController(
                gains=candidate,
                actuator_limits=limits,
                anti_windup=AntiWindupMethod.CLAMPING,
                u_bias=q_j_ss,
            )

            try:
                res = sim.run(
                    plant=plant,
                    controller=pid,
                    initial_state=ss,
                    t_span=(0.0, 8.0),
                    dt=0.02,
                    setpoint_func=lambda t: ss.T - 2.0,
                )
            except Exception:
                return 1e6

            norm_iae = res.metrics.iae / ref_iae
            norm_tv = res.metrics.tv / ref_tv
            final_err = abs(res.t_pv[-1] - (ss.T - 2.0))
            offset_pen = 1e3 * final_err if final_err > 0.3 else 0.0

            cost = float(lam * norm_iae + (1.0 - lam) * norm_tv + ms_pen + offset_pen)
            return cost

        opt = minimize(
            fun=objective,
            x0=x0,
            method="Nelder-Mead",
            options={"maxiter": 80, "xatol": 1e-2, "fatol": 1e-2, "disp": False},
        )

        opt_gains = PIDGains(kp=float(opt.x[0]), ti=float(opt.x[1]), td=float(opt.x[2]))
        ms_opt = calculate_maximum_sensitivity(plant, ss, q_j_ss, opt_gains)

        pid_opt = PIDController(
            gains=opt_gains,
            actuator_limits=limits,
            anti_windup=AntiWindupMethod.CLAMPING,
            u_bias=q_j_ss,
        )
        res_opt = sim.run(
            plant=plant,
            controller=pid_opt,
            initial_state=ss,
            t_span=(0.0, 10.0),
            dt=0.02,
            setpoint_func=lambda t: ss.T - 2.0,
        )

        results.append({
            "lambda": float(lam),
            "kp": opt_gains.kp,
            "ti": opt_gains.ti,
            "td": opt_gains.td,
            "ms": ms_opt,
            "iae": res_opt.metrics.iae,
            "itae": res_opt.metrics.itae,
            "tv": res_opt.metrics.tv,
            "overshoot": res_opt.metrics.overshoot_pct,
            "settling_time": res_opt.metrics.settling_time or 10.0,
        })
        # Warm start for next weight
        x0 = np.array([opt_gains.kp, opt_gains.ti, opt_gains.td], dtype=np.float64)

    return results


def run_benchmark_points(
    plant: CSTRPlant,
    ss: CSTRState,
    q_j_ss: float,
    foptd: FOPTDModel,
    limits: ActuatorLimits,
) -> Dict[str, Dict[str, Any]]:
    """Evaluates the discrete classical tuning methods as reference points."""
    sim = ClosedLoopSimulator()
    gains_dict = {
        "Ziegler-Nichols": tune_analytical(foptd, TuningRule.ZIEGLER_NICHOLS_PID),
        "Cohen-Coon": tune_analytical(foptd, TuningRule.COHEN_COON_PID),
        "Skogestad SIMC": tune_analytical(foptd, TuningRule.SKOGESTAD_SIMC_PID, tau_c=0.40),
        "Optimal ITAE (Ms<=1.6)": tune_by_optimization(
            plant=plant,
            nominal_state=ss,
            q_j_ss=q_j_ss,
            criterion=OptimizationCriterion.ITAE,
            max_ms=1.6,
        ),
    }

    discrete_points = {}
    for name, gains in gains_dict.items():
        ms = calculate_maximum_sensitivity(plant, ss, q_j_ss, gains)
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
            t_span=(0.0, 10.0),
            dt=0.02,
            setpoint_func=lambda t: ss.T - 2.0,
        )
        discrete_points[name] = {
            "gains": gains,
            "ms": ms,
            "iae": res.metrics.iae,
            "itae": res.metrics.itae,
            "tv": res.metrics.tv,
            "overshoot": res.metrics.overshoot_pct,
        }
    return discrete_points


def plot_true_pareto_frontier(
    simc_results: List[Dict[str, Any]],
    opt_results: List[Dict[str, Any]],
    discrete_points: Dict[str, Dict[str, Any]],
    figures_dir: Path,
) -> None:
    """Renders the true continuous Pareto trade-off figure with IEEE editorial formatting."""
    setup_ieee_style(single_column=False)
    fig_w, fig_h = get_figure_dimensions(columns=2, height_override=3.8)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=300)

    # ----------------------------------------------------
    # Panel (a): Pareto Frontier in the (TV, IAE) Plane
    # ----------------------------------------------------
    simc_tv = [r["tv"] for r in simc_results]
    simc_iae = [r["iae"] for r in simc_results]
    simc_ms = np.array([r["ms"] for r in simc_results])

    # Plot continuous SIMC curve with colormap based on Ms
    scatter = ax1.scatter(
        simc_tv,
        simc_iae,
        c=simc_ms,
        cmap="viridis_r",
        vmin=1.1,
        vmax=2.5,
        s=28,
        zorder=3,
        edgecolor="none",
        label=r"SIMC $\tau_c$-sweep locus",
    )
    ax1.plot(simc_tv, simc_iae, color="#555555", linestyle="-", linewidth=1.0, alpha=0.7)

    # Scalarized Multi-objective frontier
    opt_tv = [r["tv"] for r in opt_results]
    opt_iae = [r["iae"] for r in opt_results]
    # Sort by TV
    sorted_opt = sorted(zip(opt_tv, opt_iae), key=lambda x: x[0])
    ax1.plot(
        [x[0] for x in sorted_opt],
        [x[1] for x in sorted_opt],
        color="#1B7837",
        linestyle="--",
        linewidth=1.4,
        label=r"Pareto optimal front ($\lambda$-opt)",
    )

    # Plot discrete benchmark points
    for name, pt in discrete_points.items():
        color = IEEE_PALETTE.get(name, "#000000")
        marker = IEEE_MARKERS.get(name, "o")
        ax1.scatter(
            pt["tv"],
            pt["iae"],
            color=color,
            marker=marker,
            s=65,
            zorder=5,
            edgecolors="black",
            linewidth=0.8,
            label=f"{name} ($M_s={pt['ms']:.2f}$)",
        )

    # Callout annotations for ZN / CC if off-scale or boundary
    zn_pt = discrete_points["Ziegler-Nichols"]
    if zn_pt["tv"] > 300 or zn_pt["iae"] > 5.0:
        ax1.annotate(
            f"ZN (TV={zn_pt['tv']:.0f}, IAE={zn_pt['iae']:.1f})",
            xy=(min(zn_pt["tv"], 280), min(zn_pt["iae"], 5.0)),
            xytext=(160, 4.3),
            arrowprops=dict(facecolor="#D95F02", arrowstyle="->", lw=0.9),
            fontsize=7.5,
            color="#D95F02",
            fontweight="bold",
        )

    ax1.set_xlim(-5, 300)
    ax1.set_ylim(1.0, 5.5)
    ax1.set_xlabel("Actuator Total Variation $TV$ (L/min) [Wear / Effort]")
    ax1.set_ylabel("Integral Absolute Error $IAE$ (K$\\cdot$min) [Tracking Error]")
    ax1.set_title("(a) True Pareto Trade-off: $IAE$ vs $TV$", fontsize=9.0)
    ax1.legend(loc="upper right", fontsize=7.0, framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Colorbar for sensitivity Ms
    cbar = plt.colorbar(scatter, ax=ax1, pad=0.02, aspect=20)
    cbar.set_label("Maximum Sensitivity $M_s = \\|S\\|_\\infty$", fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.5)

    # ----------------------------------------------------
    # Panel (b): Tuning Parameter tau_c vs Robustness Ms & IAE
    # ----------------------------------------------------
    tau_cs = [r["tau_c"] for r in simc_results]
    ms_vals = [r["ms"] for r in simc_results]
    iae_vals = [r["iae"] for r in simc_results]

    color_ms = "#1F78B4"
    color_iae = "#E31A1C"

    ax2.plot(
        tau_cs, ms_vals, color=color_ms, linestyle="-", linewidth=1.5, label="Max Sensitivity $M_s$"
    )
    ax2.axhline(
        1.6, color="#33A02C", linestyle="--", linewidth=1.1, label="Robust Target $M_s = 1.6$"
    )
    ax2.axhline(
        2.0, color="#E31A1C", linestyle=":", linewidth=1.1, label="Marginal Limit $M_s = 2.0$"
    )

    ax2.set_xscale("log")
    ax2.set_xlabel(r"Closed-loop Tuning Parameter $\tau_c$ (min) [Log Scale]")
    ax2.set_ylabel(r"Maximum Sensitivity Peak $M_s$", color=color_ms)
    ax2.tick_params(axis="y", labelcolor=color_ms)
    ax2.set_ylim(1.0, 3.5)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Twin axis for IAE
    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        tau_cs,
        iae_vals,
        color=color_iae,
        linestyle="-.",
        linewidth=1.5,
        label="Tracking Error $IAE$",
    )
    ax2_twin.set_ylabel(r"Integral Absolute Error $IAE$ (K$\cdot$min)", color=color_iae)
    ax2_twin.tick_params(axis="y", labelcolor=color_iae)
    ax2_twin.set_ylim(1.0, 6.0)

    # Combined legend for panel (b)
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper right",
        fontsize=7.0,
        framealpha=0.9,
    )
    ax2.set_title(r"(b) Robustness ($M_s$) and Performance ($IAE$) vs $\tau_c$", fontsize=9.0)

    plt.tight_layout()

    fig_png = figures_dir / "fig4_pareto_iae_tv.png"
    fig_pdf = figures_dir / "fig4_pareto_iae_tv.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    logger.info("Saved true Pareto figures to: %s and %s", fig_png, fig_pdf)


def run_experiment() -> None:
    """Executes the full true Pareto exploration and generates artifacts."""
    paper_dir = PROJECT_ROOT / "Artigos_Rascunhos" / "Artigo_01_PID_Neural_Comparativo"
    figures_dir = paper_dir / "figures"
    tables_dir = paper_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    plant = CSTRPlant()
    integrator = NumericalIntegrator(method="RK45")
    q_j_ss = 100.0
    ss = plant.find_steady_state(q_j=q_j_ss)
    limits = ActuatorLimits(u_min=0.0, u_max=300.0, max_slew_rate=80.0)

    # 1. System Identification
    logger.info("Identifying nominal FOPTD transfer function...")
    delta_q = 10.0
    id_res = integrator.simulate_open_loop(
        plant=plant,
        initial_state=ss,
        t_span=(0.0, 6.0),
        dt=0.02,
        u_func=lambda t: q_j_ss + delta_q,
    )
    foptd, rmse = identify_foptd(
        t=id_res.t,
        y=id_res.states[:, 1],
        delta_u=delta_q,
        y0=ss.T,
    )
    logger.info(
        "FOPTD: K=%.4f, tau=%.4f min, theta=%.4f min (RMSE=%.4f K)",
        foptd.k_p,
        foptd.tau,
        foptd.theta,
        rmse,
    )

    # 2. Continuous Skogestad SIMC Sweep
    tau_c_vals = np.logspace(np.log10(0.015), np.log10(2.5), 45)
    simc_results = run_simc_sweep(plant, ss, q_j_ss, foptd, limits, tau_c_vals)

    # 3. Discrete Classical Benchmark Points
    discrete_points = run_benchmark_points(plant, ss, q_j_ss, foptd, limits)

    # Reference values for scalarization
    ref_iae = discrete_points["Skogestad SIMC"]["iae"]
    ref_tv = discrete_points["Skogestad SIMC"]["tv"]

    # 4. Multi-objective scalarized optimization sweep
    lambdas = np.linspace(0.05, 0.95, 12)
    opt_results = run_scalarized_multiobjective_sweep(
        plant=plant,
        ss=ss,
        q_j_ss=q_j_ss,
        limits=limits,
        lambdas=lambdas,
        ref_iae=ref_iae,
        ref_tv=ref_tv,
    )

    # 5. Generate True Pareto Plot
    plot_true_pareto_frontier(simc_results, opt_results, discrete_points, figures_dir)

    # 6. Generate LaTeX Summary Table
    # Sample 6 representative tau_c values plus optimal points
    table_rows = []
    target_tcs = [0.03, 0.10, 0.25, 0.40, 0.80, 1.50]
    for target in target_tcs:
        closest = min(simc_results, key=lambda r: abs(r["tau_c"] - target))
        table_rows.append({
            "Strategy": f"SIMC ($\\tau_c={closest['tau_c']:.2f}$)",
            "$K_p$": f"{closest['kp']:.2f}",
            "$T_i$ (min)": f"{closest['ti']:.2f}",
            "$T_d$ (min)": f"{closest['td']:.2f}",
            "$M_s$": f"{closest['ms']:.2f}",
            "IAE (K$\\cdot$min)": f"{closest['iae']:.2f}",
            "TV (L/min)": f"{closest['tv']:.1f}",
            "$M_p$ (\\%)": f"{closest['overshoot']:.1f}",
        })

    for name, pt in discrete_points.items():
        table_rows.append({
            "Strategy": name,
            "$K_p$": f"{pt['gains'].kp:.2f}",
            "$T_i$ (min)": f"{pt['gains'].ti:.2f}",
            "$T_d$ (min)": f"{pt['gains'].td:.2f}",
            "$M_s$": f"{pt['ms']:.2f}",
            "IAE (K$\\cdot$min)": f"{pt['iae']:.2f}",
            "TV (L/min)": f"{pt['tv']:.1f}",
            "$M_p$ (\\%)": f"{pt['overshoot']:.1f}",
        })

    export_latex_table(
        data=table_rows,
        output_path=tables_dir / "pareto_tradeoff_table.tex",
        caption=(
            "Quantitative trade-off along the continuous Pareto frontier: "
            "impact of closed-loop tuning parameter $\\tau_c$ and classical rules "
            "on robustness ($M_s$) and actuator effort ($TV$)."
        ),
        label="tab:pareto_tradeoff",
    )


if __name__ == "__main__":
    run_experiment()
