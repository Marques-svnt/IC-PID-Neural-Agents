# -*- coding: utf-8 -*-
"""Script de benchmark comparativo de sintonia clássica de PID no reator CSTR não-linear.

Gera as figuras de publicação e tabelas em LaTeX para o Artigo 1:
- Ziegler-Nichols (ZN)
- Cohen-Coon (CC)
- Skogestad SIMC
- Otimização Numérica com Restrição de Robustez (ITAE com Ms <= 1.6)

Adere estritamente aos padrões editoriais IEEE Transactions (300 DPI, fontes Type 42).
"""

# %% [1. Importações e Configuração de Caminhos]
import logging
import sys
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path para garantir importações relativas e absolutas
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

# Configuração de logging informativo no terminal
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("exp_01_benchmark")

# %% [2. Função Principal de Execução do Benchmark]
def run_benchmark() -> None:
    """Executa a campanha completa de benchmark determinístico, gerando figuras e tabelas LaTeX."""
    project_root = Path(__file__).resolve().parent.parent
    paper_dir = project_root / "Artigos_Rascunhos" / "Artigo_01_PID_Neural_Comparativo"
    figures_dir = paper_dir / "figures"
    tables_dir = paper_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Inicialização do modelo físico do reator CSTR e do integrador Runge-Kutta 4-5
    plant = CSTRPlant()
    integrator = NumericalIntegrator(method="RK45")
    q_j_ss = 100.0  # Vazão nominal de fluido refrigerante [L/min]
    ss = plant.find_steady_state(q_j=q_j_ss)

    logger.info("Estado estacionário nominal: T=%.2f K, C_A=%.4f mol/L", ss.T, ss.C_A)

    # %% [3. Identificação em Malha Aberta por Degrau (FOPTD)]
    # Aplica um degrau positivo de +10 L/min na camisa de resfriamento para identificar FOPTD
    logger.info("Executando degrau de identificação em malha aberta (+10 L/min)...")
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

    # Ajusta o modelo de primeira ordem com tempo morto: G(s) = K * exp(-theta*s) / (tau*s + 1)
    foptd, rmse = identify_foptd(
        t=id_res.t,
        y=id_res.states[:, 1],
        delta_u=delta_q,
        y0=ss.T,
    )
    logger.info(
        "FOPTD Identificado: K=%.4f K/(L/min), tau=%.3f min, theta=%.3f min (RMSE=%.4f K)",
        foptd.k_p,
        foptd.tau,
        foptd.theta,
        rmse,
    )

    # %% [4. Cálculo de Ganhos PID para Cada Metodologia Clássica]
    # Sintoniza os controladores para Ziegler-Nichols, Cohen-Coon, Skogestad SIMC e Otimização ITAE
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

    # %% [5. Simulação em Malha Fechada Determinística]
    # Avalia a resposta ao degrau de setpoint de -2.0 K sem ruído para isolar a dinâmica nominal
    sim = ClosedLoopSimulator(integrator=integrator)
    t_sim_span = (0.0, 10.0)
    target_T = ss.T - 2.0  # Degrau de setpoint de -2 K (resfriamento)

    def setpoint_profile(t: float) -> float:
        return target_T

    sim_results = {}
    table_rows = []

    # Limites físicos e dinâmicos do atuador da válvula
    limits = ActuatorLimits(u_min=0.0, u_max=300.0, max_slew_rate=80.0)

    for name, gains in controllers.items():
        logger.info("Simulando malha fechada para: %s", name)
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
            noise_std=0.0,  # Benchmark determinístico puro
            seed=42,
        )
        sim_results[name] = res

        # Cálculo da sensibilidade máxima de pico Ms (medida de robustez na frequência)
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

    # %% [6. Geração da Tabela LaTeX (Formato Booktabs)]
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
    logger.info("Tabela LaTeX salva com sucesso em: %s", table_file)

    # %% [7. Visualização Gráfica no Padrão IEEE (Coluna Única, 300 DPI)]
    setup_ieee_style(single_column=True)
    fig_w, fig_h = get_figure_dimensions(columns=1, height_override=3.6)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True, dpi=300)

    # Traçado do sinal de referência (setpoint)
    sample_res = next(iter(sim_results.values()))
    ax1.plot(
        sample_res.t,
        sample_res.setpoint,
        color=IEEE_PALETTE["Setpoint"],
        linestyle=IEEE_LINESTYLES["Setpoint"],
        label="Setpoint ($T_{sp}$)",
        linewidth=1.2,
    )

    # Traçado das respostas de temperatura e esforço de controle para cada controlador
    for name, res in sim_results.items():
        color = IEEE_PALETTE.get(name, "#333333")
        linestyle = IEEE_LINESTYLES.get(name, "-")
        ax1.plot(res.t, res.t_pv, label=name, color=color, linestyle=linestyle, linewidth=1.2)
        ax2.plot(res.t, res.u_applied, label=name, color=color, linestyle=linestyle, linewidth=1.2)

    # Ajustes estéticos e de headroom do Subplot 1 (Temperatura do Reator)
    # Limites estritos [394.0, 397.3] K posicionam a legenda no canto superior direito
    # perfeitamente acima da curva acomodada em 394.65 K, sem colisão visual.
    ax1.set_ylabel("Reactor Temp. $T$ (K)", fontsize=8.5)
    ax1.set_ylim(394.0, 397.3)
    ax1.legend(loc="upper right", frameon=True, fontsize=6.5, framealpha=0.92)
    ax1.set_title("Classical PID Benchmark: Setpoint Tracking", fontsize=9.0)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Limites físicos de saturação da válvula no Subplot 2 (Vazão de Fluido Refrigerante)
    ax2.axhline(
        limits.u_max,
        color=IEEE_PALETTE["Constraint"],
        linestyle=IEEE_LINESTYLES["Constraint"],
        label=r"Limits ($u_{\mathrm{min}}, u_{\mathrm{max}}$)",
        linewidth=1.0,
    )
    ax2.axhline(
        limits.u_min,
        color=IEEE_PALETTE["Constraint"],
        linestyle=IEEE_LINESTYLES["Constraint"],
        linewidth=1.0,
    )
    ax2.set_ylabel(r"Coolant Flow $q_j$ (L/min)", fontsize=8.5)
    ax2.set_xlabel("Time $t$ (min)", fontsize=8.5)
    # Limites [-15, 335] L/min e legenda compacta em 2 colunas no canto inferior direito
    # garantem que toda a legenda fique estritamente abaixo do patamar acomodado de 100-110 L/min.
    ax2.set_ylim(-15, 335)
    ax2.legend(
        loc="lower right",
        ncol=2,
        frameon=True,
        fontsize=6.3,
        framealpha=0.92,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_png = figures_dir / "fig1_classical_benchmark.png"
    fig_pdf = figures_dir / "fig1_classical_benchmark.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    logger.info("Figuras salvas com sucesso em: %s e %s", fig_png, fig_pdf)


# %% [8. Ponto de Entrada Principal]
if __name__ == "__main__":
    run_benchmark()
