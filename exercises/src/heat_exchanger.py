from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeatExchangerParameters:
    """Operational and physical parameters for the heat exchanger system.
    """

    t_sp: float = 80.0
    kp: float = 2.0
    u0: float = 33.3
    kv: float = 0.01
    lambda_v: float = 2.0e6
    cp: float = 4000.0
    t_in: float = 30.0
    t_amb: float = 25.0
    ul: float = 20.0
    c_heat_capacity: float = 1.0e5
    m_in_nominal: float = 3.0
    m_in_disturbed: float = 4.0
    disturbance_time_min: float = 10.0


class HeatExchangerAnalyzer:
    """Analyzer for heat exchanger control loop and dynamic thermal energy balances."""

    def __init__(self, params: Optional[HeatExchangerParameters] = None) -> None:
        """Initializes the HeatExchangerAnalyzer with physical parameters.
        """
        self.params = params if params is not None else HeatExchangerParameters()
        logger.info("Initialized HeatExchangerAnalyzer with parameters: %s", self.params)

    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Loads experimental temperature data from a CSV file.
        """
        if not file_path.exists():
            logger.error("Data file not found at: %s", file_path)
            raise FileNotFoundError(f"Data file not found at: {file_path}")

        df = pd.read_csv(file_path)
        required_columns = {"tempo", "Tpv"}
        if not required_columns.issubset(df.columns):
            logger.error("Missing required columns in %s. Found: %s", file_path, df.columns.tolist())
            raise ValueError(f"CSV file must contain columns: {required_columns}")

        logger.info("Successfully loaded %d records from %s", len(df), file_path)
        return df

    def compute_simulation(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Computes control actions, thermal powers, and dynamic derivatives.
        """
        df = df_raw.copy()

        logger.info("Starting progressive step-by-step physical calculations.")

        # 1. Process stream flow rate with disturbance at t = disturbance_time_min
        df["m_in"] = np.where(
            df["tempo"] < self.params.disturbance_time_min,
            self.params.m_in_nominal,
            self.params.m_in_disturbed,
        )

        # 2. Control error: e_k = T_SP - T_PV,k
        df["e"] = self.params.t_sp - df["Tpv"]

        # 3. Proportional control action: u_k = K_p * e_k
        df["u"] = self.params.kp * df["e"]

        # 4. Total valve opening with physical saturation: 0 <= u_v <= 100%
        raw_uv = self.params.u0 + df["u"]
        df["uv"] = np.clip(raw_uv, 0.0, 100.0)

        # 5. Steam mass flow rate: m_v,k = K_v * u_v,k (kg/min)
        df["m_v"] = self.params.kv * df["uv"]

        # 6. Thermal power provided by condensing steam: Q_v = (m_v / 60) * lambda_v (W)
        df["Q_v"] = (df["m_v"] / 60.0) * self.params.lambda_v

        # 7. Thermal power transferred to heat process stream:
        # Q_processo = (m_in / 60) * c_p * (T_PV - T_in) (W)
        df["Q_processo"] = (df["m_in"] / 60.0) * self.params.cp * (df["Tpv"] - self.params.t_in)

        # 8. Thermal losses to the environment: Q_perdas = U_L * (T_PV - T_amb) (W)
        df["Q_perdas"] = self.params.ul * (df["Tpv"] - self.params.t_amb)

        # 9. Net thermal power: Q_liq = Q_v - Q_processo - Q_perdas (W)
        df["Q_liq"] = df["Q_v"] - df["Q_processo"] - df["Q_perdas"]

        # 10. Rate of temperature change: dT/dt = Q_liq / C (°C/s and °C/min)
        df["dT_dt_s"] = df["Q_liq"] / self.params.c_heat_capacity
        df["dT_dt_min"] = 60.0 * df["dT_dt_s"]

        logger.info("Completed simulation calculations successfully.")
        return df

    def generate_plots(self, df: pd.DataFrame, output_dir: Path) -> Tuple[Path, Path, Path, Path]:
        """Generates publication-quality charts for process analysis.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # 1. Temperature vs Time
        fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
        ax1.plot(df["tempo"], df["Tpv"], "o-", color="#1f77b4", linewidth=2, label=r"$T_{PV}$ (Temperatura Medida)")
        ax1.axhline(self.params.t_sp, color="#d62728", linestyle="--", linewidth=1.8, label=f"$T_{{SP}} = {self.params.t_sp:.1f}^\\circ$C")
        ax1.axvline(self.params.disturbance_time_min, color="#7f7f7f", linestyle=":", linewidth=1.5, label="Perturbação de Carga ($t = 10$ min)")
        ax1.set_title("Dinâmica de Temperatura do Trocador de Calor com Controlador P", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Tempo (min)", fontsize=11)
        ax1.set_ylabel("Temperatura (°C)", fontsize=11)
        ax1.set_xticks(np.arange(0, df["tempo"].max() + 1, 2))
        ax1.legend(loc="lower right", frameon=True)
        ax1.grid(True, linestyle="--", alpha=0.6)
        path1 = output_dir / "grafico_1_temperatura_vs_tempo.png"
        fig1.tight_layout()
        fig1.savefig(path1)
        plt.close(fig1)

        # 2. Rate of Temperature Change dT/dt vs Time
        fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=300)
        ax2.plot(df["tempo"], df["dT_dt_min"], "s-", color="#2ca02c", linewidth=2, label=r"$\frac{dT}{dt}$ (°C/min)")
        ax2.axhline(0.0, color="black", linestyle="-", linewidth=1.0, alpha=0.7)
        ax2.axvline(self.params.disturbance_time_min, color="#7f7f7f", linestyle=":", linewidth=1.5, label="Perturbação ($t = 10$ min)")
        ax2.fill_between(df["tempo"], 0, df["dT_dt_min"], where=(df["dT_dt_min"] > 0), color="#2ca02c", alpha=0.15, label="Aquecimento ($Q_{liq} > 0$)")
        ax2.fill_between(df["tempo"], 0, df["dT_dt_min"], where=(df["dT_dt_min"] < 0), color="#d62728", alpha=0.15, label="Resfriamento ($Q_{liq} < 0$)")
        ax2.set_title("Taxa de Variação da Temperatura no Trocador de Calor", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Tempo (min)", fontsize=11)
        ax2.set_ylabel(r"$\frac{dT}{dt}$ (°C/min)", fontsize=11)
        ax2.set_xticks(np.arange(0, df["tempo"].max() + 1, 2))
        ax2.legend(loc="upper right", frameon=True)
        ax2.grid(True, linestyle="--", alpha=0.6)
        path2 = output_dir / "grafico_2_taxa_variacao_temperatura.png"
        fig2.tight_layout()
        fig2.savefig(path2)
        plt.close(fig2)

        # 3. Thermal Power Balance vs Time
        fig3, ax3 = plt.subplots(figsize=(8, 5), dpi=300)
        ax3.plot(df["tempo"], df["Q_v"], "o-", color="#d62728", linewidth=2, label=r"$\dot{Q}_v$ (Vapor Fornecido)")
        ax3.plot(df["tempo"], df["Q_processo"], "^-", color="#1f77b4", linewidth=2, label=r"$\dot{Q}_{processo}$ (Fluido Aquecido)")
        ax3.plot(df["tempo"], df["Q_perdas"], "v-", color="#ff7f0e", linewidth=1.8, label=r"$\dot{Q}_{perdas}$ (Perdas Ambiente)")
        ax3.axvline(self.params.disturbance_time_min, color="#7f7f7f", linestyle=":", linewidth=1.5, label="Perturbação ($t = 10$ min)")
        ax3.set_title("Balanço Dinâmico de Potências Térmicas no Trocador", fontsize=13, fontweight="bold")
        ax3.set_xlabel("Tempo (min)", fontsize=11)
        ax3.set_ylabel("Potência Térmica (W)", fontsize=11)
        ax3.set_xticks(np.arange(0, df["tempo"].max() + 1, 2))
        ax3.legend(loc="center right", frameon=True)
        ax3.grid(True, linestyle="--", alpha=0.6)
        path3 = output_dir / "grafico_3_balanco_termico.png"
        fig3.tight_layout()
        fig3.savefig(path3)
        plt.close(fig3)

        # 4. Consolidated Presentation Dashboard
        fig4, axes = plt.subplots(3, 1, figsize=(10, 12), dpi=300, sharex=True)

        # Panel 1: Temperature & Setpoint
        axes[0].plot(df["tempo"], df["Tpv"], "o-", color="#1f77b4", linewidth=2, label=r"$T_{PV}$ (Medida)")
        axes[0].axhline(self.params.t_sp, color="#d62728", linestyle="--", linewidth=1.5, label=f"$T_{{SP}} = {self.params.t_sp}^\\circ$C")
        axes[0].axvline(self.params.disturbance_time_min, color="gray", linestyle=":", linewidth=1.5, label="Perturbação")
        axes[0].set_ylabel("Temperatura (°C)", fontsize=11)
        axes[0].set_title("Controle Proporcional de Temperatura — Painel Geral de Operação", fontsize=14, fontweight="bold")
        axes[0].legend(loc="lower right")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Panel 2: Valve Opening & Control Action
        axes[1].plot(df["tempo"], df["uv"], "s-", color="#9467bd", linewidth=2, label=r"Abertura Válvula $u_v$ (%)")
        axes[1].plot(df["tempo"], df["u"], "x--", color="#8c564b", linewidth=1.5, label=r"Ação P $u$ (%)")
        axes[1].axvline(self.params.disturbance_time_min, color="gray", linestyle=":", linewidth=1.5)
        axes[1].set_ylabel("Abertura (%)", fontsize=11)
        axes[1].legend(loc="upper right")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        # Panel 3: Heat Powers
        axes[2].plot(df["tempo"], df["Q_v"], "o-", color="#d62728", linewidth=1.8, label=r"$\dot{Q}_v$ (Vapor)")
        axes[2].plot(df["tempo"], df["Q_processo"], "^-", color="#1f77b4", linewidth=1.8, label=r"$\dot{Q}_{processo}$ (Processo)")
        axes[2].plot(df["tempo"], df["Q_liq"], "d-", color="#2ca02c", linewidth=1.8, label=r"$\dot{Q}_{liq}$ (Líquida)")
        axes[2].axhline(0, color="black", linestyle="-", linewidth=0.8)
        axes[2].axvline(self.params.disturbance_time_min, color="gray", linestyle=":", linewidth=1.5)
        axes[2].set_xlabel("Tempo (min)", fontsize=11)
        axes[2].set_ylabel("Potência (W)", fontsize=11)
        axes[2].set_xticks(np.arange(0, df["tempo"].max() + 1, 2))
        axes[2].legend(loc="center right")
        axes[2].grid(True, linestyle="--", alpha=0.5)

        path4 = output_dir / "painel_completo_controle_trocador.png"
        fig4.tight_layout()
        fig4.savefig(path4)
        plt.close(fig4)

        logger.info("All plots saved successfully in %s", output_dir)
        return path1, path2, path3, path4


def run_pipeline() -> pd.DataFrame:
    """Executes the complete data processing, computation, and plotting pipeline.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "raw" / "dados_temperatura_trocador.csv"
    plots_dir = base_dir / "plots"

    analyzer = HeatExchangerAnalyzer()
    df_raw = analyzer.load_data(data_path)
    df_result = analyzer.compute_simulation(df_raw)

    output_csv = base_dir / "data" / "processed" / "resultados_trocador_calculados.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(output_csv, index=False)
    logger.info("Saved calculated results table to %s", output_csv)

    analyzer.generate_plots(df_result, plots_dir)

    return df_result


if __name__ == "__main__":
    df = run_pipeline()
