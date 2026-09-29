# -*- coding: utf-8 -*-
"""Campanha abrangente de robustez, incerteza paramétrica e rejeição de distúrbios no CSTR.

Avalia os controladores PID clássicos (Ziegler-Nichols, Cohen-Coon, Skogestad SIMC
e ITAE Ótimo com restrição de sensibilidade máxima) através de quatro ensaios:
1. Rejeição de perturbações de carga não-mensuradas (choques térmicos e de
   concentração na alimentação).
2. Incrustação térmica (fouling) na camisa de resfriamento (degradação de UA de 100% até 60%).
3. Varredura de ruído de medição nos sensores e fronteira de esforço do atuador (IAE vs TV).
4. Rastreamento não-linear de múltiplos pontos de operação em larga escala.

Gera todas as figuras com resolução de 300 DPI e tabelas no padrão booktabs para o Artigo 1.
"""

# %% [1. Importações e Configuração de Caminhos]
import logging
import sys
from pathlib import Path
from typing import Any

# Adiciona a raiz do projeto ao sys.path para garantir importações relativas e absolutas
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
from src.evaluation.plot_styles import (
    IEEE_LINESTYLES,
    IEEE_PALETTE,
    get_figure_dimensions,
    setup_ieee_style,
)
from src.sim_core.actuators import ActuatorLimits
from src.sim_core.cstr_plant import CSTRPlant, CSTRState
from src.sim_core.integrator import NumericalIntegrator

# Configuração de logging estruturado
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("exp_02_robustness")

CONTROLLER_COLORS = IEEE_PALETTE


# %% [2. Identificação FOPTD e Sintonia dos Controladores PID]
def setup_controllers(
    plant: CSTRPlant, ss: CSTRState, q_j_ss: float
) -> dict[str, PIDGains]:
    """Identifica o modelo de primeira ordem com tempo morto e sintoniza as 4 estratégias PID.

    Passo a passo didático:
    1. Realiza um teste de degrau em malha aberta (+10 L/min na camisa de resfriamento).
    2. Identifica os parâmetros FOPTD (ganho K, constante de tempo tau, tempo morto theta).
    3. Calcula os parâmetros (Kp, Ti, Td) para Ziegler-Nichols, Cohen-Coon, Skogestad SIMC
       e sintoniza numericamente o ITAE restrito a Ms <= 1.6 via Nelder-Mead.

    Args:
        plant: Modelo físico do CSTR.
        ss: Estado estacionário nominal de operação.
        q_j_ss: Vazão de fluido refrigerante nominal [L/min].

    Returns:
        Dicionário mapeando o nome de cada controlador para seus respectivos ganhos PID.
    """
    logger.info("Executando degrau de identificação em malha aberta (+10 L/min)...")
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


# %% [3. Utilitário de Exportação de Tabelas em LaTeX]
def export_latex_table(
    data: list[dict[str, Any]],
    output_path: Path,
    caption: str,
    label: str,
) -> None:
    """Exporta registros tabulares para código LaTeX no padrão editorial booktabs.

    Args:
        data: Lista de dicionários representando as linhas da tabela.
        output_path: Caminho de destino para salvar o arquivo .tex.
        caption: Legenda descritiva da tabela.
        label: Rótulo de referência cruzada no LaTeX.
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_table, encoding="utf-8")
    logger.info("Tabela LaTeX salva com sucesso em: %s", output_path)


# %% [4. Ensaio 1: Rejeição de Perturbações de Carga Não-Mensuradas]
def run_load_disturbance_campaign(
    plant: CSTRPlant,
    ss: CSTRState,
    q_j_ss: float,
    controllers: dict[str, PIDGains],
    limits: ActuatorLimits,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    """Avalia o controle regulatório contra perturbações não-mensuradas na alimentação.

    Cronograma didático de perturbações:
    - t em [0, 2) min: Regime permanente nominal.
    - t em [2, 8) min: Choque térmico de alimentação Delta T_f = +5.0 K.
    - t em [8, 15] min: Choque cumulativo de concentração Delta C_Af = +0.2 mol/L (+20%).

    Args:
        plant: Modelo físico do reator CSTR.
        ss: Estado estacionário nominal.
        q_j_ss: Vazão base de resfriamento [L/min].
        controllers: Dicionário de controladores PID sintonizados.
        limits: Restrições físicas de saturação e taxa da válvula.
        figures_dir: Diretório de destino para figuras.
        tables_dir: Diretório de destino para tabelas.
    """
    logger.info("=== Executando Campanha de Rejeição de Perturbações de Carga ===")
    sim = ClosedLoopSimulator()
    t_span = (0.0, 15.0)
    dt = 0.02

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
            setpoint_func=lambda t: ss.T,  # Controle puramente regulatório em T nominal
            disturbance_func=disturbance_profile,
            noise_std=0.05,
            seed=42,
        )
        sim_results[name] = res

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

    # Geração dos gráficos IEEE em coluna única (3.5 in) com Legenda Unificada Superior
    setup_ieee_style(single_column=True)
    fig_w, fig_h = get_figure_dimensions(columns=1, height_override=4.2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True, dpi=300)

    # Sinal de referência nominal
    sample_res = next(iter(sim_results.values()))
    ax1.plot(
        sample_res.t,
        sample_res.setpoint,
        color=IEEE_PALETTE["Setpoint"],
        linestyle=IEEE_LINESTYLES["Setpoint"],
        label="Target $T_{sp}$",
        linewidth=1.2,
    )

    # Linhas verticais indicando a injeção dos distúrbios com anotações em posições desobstruídas
    ax1.axvline(2.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.text(2.1, 395.6, r"$\Delta T_f = +5$ K", fontsize=7.2, color="#444444")
    ax1.axvline(8.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.text(8.1, 395.6, r"$\Delta C_{Af} = +20\%$", fontsize=7.2, color="#444444")

    for name, res in sim_results.items():
        color = IEEE_PALETTE.get(name, "#333333")
        linestyle = IEEE_LINESTYLES.get(name, "-")
        ax1.plot(res.t, res.t_pv, label=name, color=color, linestyle=linestyle, linewidth=1.2)
        ax2.plot(res.t, res.u_applied, label=name, color=color, linestyle=linestyle, linewidth=1.2)

    # Subplot 1: Temperatura do reator com faixa [395.0, 404.0] K
    # Sem legenda interna: desobstrução completa dos transientes e picos de distúrbio
    ax1.set_ylabel("Reactor Temp. $T$ (K)", fontsize=8.5)
    ax1.set_title(r"Load Disturbance Rejection ($+5$ K & $+20\% C_{Af}$)", fontsize=9.0)
    ax1.set_ylim(395.0, 404.0)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Vazão de fluido refrigerante com limites físicos
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
    ax2.set_ylim(-15, 335)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Construção da Legenda Unificada Superior (fig.legend) estilo Figura 3
    # Extrai manipuladores de ax1 (referência + 4 PIDs) e ax2 (limites de saturação)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    handles_unified = h1 + [h2[-1]]
    labels_unified = l1 + [l2[-1]]

    fig.legend(
        handles_unified,
        labels_unified,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=True,
        fontsize=5.6,
        framealpha=0.92,
        columnspacing=0.4,
        handletextpad=0.25,
    )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.89])
    fig_png = figures_dir / "fig2_disturbance_rejection.png"
    fig_pdf = figures_dir / "fig2_disturbance_rejection.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    logger.info("Figuras salvas com sucesso em: %s e %s", fig_png, fig_pdf)


# %% [5. Ensaio 2: Incrustação Térmica na Camisa (Degradação de UA)]
def run_fouling_campaign(
    plant: CSTRPlant,
    ss: CSTRState,
    q_j_ss: float,
    controllers: dict[str, PIDGains],
    limits: ActuatorLimits,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    """Avalia a estabilidade e a perda de desempenho com a degradação térmica de UA (incrustação).

    Passo a passo didático:
    1. Varia o coeficiente global de transferência térmica UA de 100% até 60% do valor de projeto.
    2. Aplica degrau de setpoint (-2 K) e analisa a lentidão e saturação da válvula.
    3. Constrói um painel 2x2 com LEGENDA COMPARTILHADA ÚNICA no topo da figura (fig.legend),
       eliminando 4 caixas repetitivas e desobstruindo integralmente todos os subplots.

    Args:
        plant: Modelo físico do CSTR.
        ss: Estado estacionário nominal.
        q_j_ss: Vazão base de resfriamento.
        controllers: Controladores PID avaliados.
        limits: Restrições físicas da válvula.
        figures_dir: Diretório de destino para figuras.
        tables_dir: Diretório de destino para tabelas.
    """
    logger.info("=== Executando Campanha de Incrustação Térmica (Degradação de UA) ===")
    sim = ClosedLoopSimulator()
    t_span = (0.0, 10.0)
    dt = 0.02
    target_T = ss.T - 2.0

    fouling_levels = [1.0, 0.9, 0.8, 0.7, 0.6]  # UA / UA_nominal
    table_rows = []

    # Armazena trajetórias de 100% UA (limpo) e 70% UA (incrustação severa) para o gráfico 2x2
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

    # Figura em Coluna Dupla (7.16 in) no formato 2x2 com Legenda Compartilhada Superior
    setup_ieee_style(single_column=False)
    fig_w, fig_h = get_figure_dimensions(columns=2, height_override=4.2)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(fig_w, fig_h), sharex=True, dpi=300)

    sample_res = next(iter(plot_data["1.0"].values()))

    # Painel (a): Temperatura T com Camisa Limpa (100% UA)
    ax1.plot(
        sample_res.t,
        sample_res.setpoint,
        color=IEEE_PALETTE["Setpoint"],
        linestyle=IEEE_LINESTYLES["Setpoint"],
        label="Target $T_{sp}$",
        linewidth=1.2,
    )
    for name, res in plot_data["1.0"].items():
        color = IEEE_PALETTE.get(name, "#333333")
        linestyle = IEEE_LINESTYLES.get(name, "-")
        ax1.plot(res.t, res.t_pv, label=name, color=color, linestyle=linestyle, linewidth=1.2)
    ax1.set_title(r"(a) Clean Jacket ($100\% UA$): Temperature $T$", fontsize=9.0)
    ax1.set_ylabel("Reactor Temp. $T$ (K)", fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Painel (b): Temperatura T com Camisa Incrustada (70% UA)
    ax2.plot(
        sample_res.t,
        sample_res.setpoint,
        color=IEEE_PALETTE["Setpoint"],
        linestyle=IEEE_LINESTYLES["Setpoint"],
        label="Target $T_{sp}$",
        linewidth=1.2,
    )
    for name, res in plot_data["0.7"].items():
        color = IEEE_PALETTE.get(name, "#333333")
        linestyle = IEEE_LINESTYLES.get(name, "-")
        ax2.plot(res.t, res.t_pv, label=name, color=color, linestyle=linestyle, linewidth=1.2)
    ax2.set_title(r"(b) Fouled Jacket ($70\% UA$): Temperature $T$", fontsize=9.0)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Painel (c): Vazão de Fluido q_j com Camisa Limpa (100% UA)
    for name, res in plot_data["1.0"].items():
        color = IEEE_PALETTE.get(name, "#333333")
        linestyle = IEEE_LINESTYLES.get(name, "-")
        ax3.plot(res.t, res.u_applied, label=name, color=color, linestyle=linestyle, linewidth=1.2)
    ax3.axhline(
        limits.u_max,
        color=IEEE_PALETTE["Constraint"],
        linestyle=IEEE_LINESTYLES["Constraint"],
        label=r"Limits ($u_{\mathrm{min}}, u_{\mathrm{max}}$)",
        linewidth=1.0,
    )
    ax3.set_title(r"(c) Clean Jacket ($100\% UA$): Coolant $q_j$", fontsize=9.0)
    ax3.set_ylabel(r"Coolant Flow $q_j$ (L/min)", fontsize=8.5)
    ax3.set_xlabel("Time $t$ (min)", fontsize=8.5)
    ax3.set_ylim(-15, 335)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # Painel (d): Vazão de Fluido q_j com Camisa Incrustada (70% UA)
    for name, res in plot_data["0.7"].items():
        color = IEEE_PALETTE.get(name, "#333333")
        linestyle = IEEE_LINESTYLES.get(name, "-")
        ax4.plot(res.t, res.u_applied, label=name, color=color, linestyle=linestyle, linewidth=1.2)
    ax4.axhline(
        limits.u_max,
        color=IEEE_PALETTE["Constraint"],
        linestyle=IEEE_LINESTYLES["Constraint"],
        label=r"Limits ($u_{\mathrm{min}}, u_{\mathrm{max}}$)",
        linewidth=1.0,
    )
    ax4.set_title(r"(d) Fouled Jacket ($70\% UA$): Coolant $q_j$", fontsize=9.0)
    ax4.set_xlabel("Time $t$ (min)", fontsize=8.5)
    ax4.set_ylim(-15, 335)
    ax4.grid(True, linestyle=":", alpha=0.6)

    # Construção da Legenda Unificada Superior (fig.legend)
    # Extrai os manipuladores de ax1 (referência + 4 PIDs) e ax3 (limites físicos)
    h1, l1 = ax1.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    handles_unified = h1 + [h3[-1]]
    labels_unified = l1 + [l3[-1]]

    fig.legend(
        handles_unified,
        labels_unified,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=6,
        frameon=True,
        fontsize=7.2,
        framealpha=0.92,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    # Ajusta a margem superior para acomodar perfeitamente a legenda sem sobrepor os títulos
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig_png = figures_dir / "fig3_fouling_robustness.png"
    fig_pdf = figures_dir / "fig3_fouling_robustness.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    logger.info("Figuras salvas com sucesso em: %s e %s", fig_png, fig_pdf)


# %% [6. Ensaio 3: Sensibilidade a Ruído de Medição e Desgaste do Atuador (IAE vs TV)]
def run_noise_and_pareto_campaign(
    plant: CSTRPlant,
    ss: CSTRState,
    q_j_ss: float,
    controllers: dict[str, PIDGains],
    limits: ActuatorLimits,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    """Quantifica a sensibilidade ao ruído estocástico de medição via Monte Carlo.

    Passo a passo didático:
    1. Varia o desvio padrão do ruído gaussiano sigma em [0.0, 0.1, 0.2, 0.5, 1.0] K.
    2. Executa simulações Monte Carlo com 5 sementes pseudo-aleatórias para cada nível de ruído.
    3. Calcula média e desvio padrão para IAE (erro de rastreamento) e TV (desgaste da válvula).
    4. Plota o diagrama de dispersão IAE vs TV com anotações claras e legenda desobstruída.

    Args:
        plant: Modelo físico do CSTR.
        ss: Estado estacionário nominal.
        q_j_ss: Vazão base de resfriamento.
        controllers: Controladores PID avaliados.
        limits: Restrições físicas da válvula.
        figures_dir: Diretório de destino para figuras.
        tables_dir: Diretório de destino para tabelas.
    """
    logger.info("=== Executando Campanha de Ruído nos Sensores e Chattering da Válvula ===")
    sim = ClosedLoopSimulator()
    t_span = (0.0, 10.0)
    dt = 0.02
    target_T = ss.T - 2.0

    noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0]  # Desvio padrão do ruído gaussiano [K]
    n_monte_carlo = 5

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

    # Configuração gráfica no padrão IEEE de coluna única
    setup_ieee_style(single_column=True)
    fig_w, fig_h = get_figure_dimensions(columns=1, height_override=3.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    markers = ["o", "s", "^", "D"]
    for (name, marker) in zip(controllers.keys(), markers):
        iaes = [np.mean(results[name][sig]["iae"]) for sig in noise_levels]
        tvs = [np.mean(results[name][sig]["tv"]) for sig in noise_levels]
        color = CONTROLLER_COLORS[name]
        ax.plot(
            tvs,
            iaes,
            marker=marker,
            linestyle="-",
            label=name,
            color=color,
            markersize=5.0,
            linewidth=1.2,
        )

    # Anotações limpas com setas indicando os extremos de ruído sem sobreposição de texto
    sample_iaes = [np.mean(results["Skogestad SIMC"][sig]["iae"]) for sig in noise_levels]
    sample_tvs = [np.mean(results["Skogestad SIMC"][sig]["tv"]) for sig in noise_levels]
    ax.annotate(
        r"$\sigma = 0.0$ K",
        xy=(sample_tvs[0], sample_iaes[0]),
        xytext=(35, 12),
        textcoords="offset points",
        fontsize=7.2,
        color="#333333",
        arrowprops=dict(arrowstyle="->", lw=0.7, color="#555555"),
    )
    ax.annotate(
        r"$\sigma = 1.0$ K (high noise)",
        xy=(sample_tvs[-1], sample_iaes[-1]),
        xytext=(-85, 20),
        textcoords="offset points",
        fontsize=7.2,
        color="#333333",
        arrowprops=dict(arrowstyle="->", lw=0.7, color="#555555"),
    )

    ax.set_xlabel(r"Actuator Total Variation $\mathrm{TV}$ (L/min) [Wear / Effort]", fontsize=8.5)
    ax.set_ylabel(r"Tracking Error $\mathrm{IAE}$ (K$\cdot$min)", fontsize=8.5)
    ax.set_title(r"Noise Sensitivity: Tracking Quality vs Actuator Wear", fontsize=9.0)
    # A legenda em 'upper left' situa-se na região ampla e vazia (TV em [0, 400], IAE em [60, 100])
    ax.legend(
        loc="upper left", bbox_to_anchor=(0.04, 0.95), frameon=True, fontsize=6.8, framealpha=0.92
    )
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_png = figures_dir / "fig_noise_sensitivity.png"
    fig_pdf = figures_dir / "fig_noise_sensitivity.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    logger.info("Figuras salvas com sucesso em: %s e %s", fig_png, fig_pdf)


# %% [7. Ensaio 4: Rastreamento Não-Linear em Múltiplos Pontos de Operação]
def run_multistep_campaign(
    plant: CSTRPlant,
    ss: CSTRState,
    q_j_ss: float,
    controllers: dict[str, PIDGains],
    limits: ActuatorLimits,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    """Avalia o rastreamento em ampla faixa cobrindo regimes exotérmicos distintos.

    Cronograma didático de transição operacional:
    - 0 a 4 min: Ponto nominal estável (396.65 K).
    - 4 a 10 min: Degrau para 390.0 K (regime de menor taxa reacional).
    - 10 a 17 min: Degrau para 402.0 K (regime fortemente exotérmico, sensível a runaway térmico).
    - 17 a 24 min: Retorno ao ponto nominal estável (396.65 K).

    Args:
        plant: Modelo físico do CSTR.
        ss: Estado estacionário nominal.
        q_j_ss: Vazão base de resfriamento.
        controllers: Controladores PID avaliados.
        limits: Restrições físicas da válvula.
        figures_dir: Diretório de destino para figuras.
        tables_dir: Diretório de destino para tabelas.
    """
    logger.info("=== Executando Campanha de Rastreamento em Múltiplos Pontos de Operação ===")
    sim = ClosedLoopSimulator()
    t_span = (0.0, 24.0)
    dt = 0.02

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

    # Geração dos gráficos IEEE em coluna única (3.5 in)
    setup_ieee_style(single_column=True)
    fig_w, fig_h = get_figure_dimensions(columns=1, height_override=4.2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True, dpi=300)

    sample_res = next(iter(sim_results.values()))
    ax1.plot(
        sample_res.t,
        sample_res.setpoint,
        color=IEEE_PALETTE["Setpoint"],
        linestyle=IEEE_LINESTYLES["Setpoint"],
        label=r"Target $T_{sp}$",
        linewidth=1.2,
    )

    for name, res in sim_results.items():
        color = IEEE_PALETTE.get(name, "#333333")
        linestyle = IEEE_LINESTYLES.get(name, "-")
        ax1.plot(res.t, res.t_pv, label=name, color=color, linestyle=linestyle, linewidth=1.2)
        ax2.plot(res.t, res.u_applied, label=name, color=color, linestyle=linestyle, linewidth=1.2)

    # Subplot 1: Temperatura do reator
    # Sem legenda interna: desobstrução completa de todos os patamares operacionais
    ax1.set_ylabel(r"Reactor Temp. $T$ (K)", fontsize=8.5)
    ax1.set_title("Wide-Range Multi-Operating Point Tracking", fontsize=9.0)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Vazão de fluido refrigerante com limites físicos
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
    ax2.set_xlabel(r"Time $t$ (min)", fontsize=8.5)
    ax2.set_ylim(-15, 345)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Construção da Legenda Unificada Superior (fig.legend) estilo Figura 3
    # Extrai manipuladores de ax1 (referência + 4 PIDs) e ax2 (limites de saturação)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    handles_unified = h1 + [h2[-1]]
    labels_unified = l1 + [l2[-1]]

    fig.legend(
        handles_unified,
        labels_unified,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=True,
        fontsize=5.6,
        framealpha=0.92,
        columnspacing=0.4,
        handletextpad=0.25,
    )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.89])
    fig_png = figures_dir / "fig5_multistep_tracking.png"
    fig_pdf = figures_dir / "fig5_multistep_tracking.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    logger.info("Figuras salvas com sucesso em: %s e %s", fig_png, fig_pdf)


# %% [8. Orquestrador Geral da Campanha]
def run_campaign() -> None:
    """Orquestra a execução sequencial de todas as campanhas de robustez do Artigo 1."""
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

    logger.info("Iniciando campanha experimental abrangente de robustez...")
    run_load_disturbance_campaign(plant, ss, q_j_ss, controllers, limits, figures_dir, tables_dir)
    run_fouling_campaign(plant, ss, q_j_ss, controllers, limits, figures_dir, tables_dir)
    run_noise_and_pareto_campaign(plant, ss, q_j_ss, controllers, limits, figures_dir, tables_dir)
    run_multistep_campaign(plant, ss, q_j_ss, controllers, limits, figures_dir, tables_dir)
    logger.info("Campanha de robustez concluída com êxito!")


# %% [9. Ponto de Entrada Principal]
if __name__ == "__main__":
    run_campaign()
