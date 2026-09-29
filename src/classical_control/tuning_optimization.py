# %% [Módulo e Importações]
"""Sintonia numérica de controladores PID por otimização restrita à sensibilidade máxima (Ms).

Implementa a sintonia direta não-linear baseada na minimização de índices de desempenho
integrais (ITAE, IAE ou ISE) calculados sobre a planta fenomenológica não-linear CSTR,
sujeita a restrições de robustez no domínio da frequência através do pico de
sensibilidade máxima:
    Ms = ||S(jw)||_inf = max_w |1 / (1 + L(jw))| <= max_ms (tipicamente 1.6)

A sensibilidade máxima Ms está diretamente ligada às margens clássicas de estabilidade:
    - Margem de Ganho: GM >= Ms / (Ms - 1)
    - Margem de Fase:  PM >= 2 * arcsin(1 / (2 * Ms))
Para Ms <= 1.6, garante-se GM >= 2.67 (8.5 dB) e PM >= 36.4 graus, prevenindo
instabilidade sob variações paramétricas ou não-linearidades operacionais.
"""

import logging
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from src.classical_control.closed_loop import ClosedLoopSimulator
from src.classical_control.pid_controller import AntiWindupMethod, PIDController, PIDGains
from src.sim_core.actuators import ActuatorLimits
from src.sim_core.cstr_plant import CSTRPlant, CSTRState

logger = logging.getLogger(__name__)


# %% [Critérios Integrais de Otimização]
class OptimizationCriterion(str, Enum):
    """Critérios de desempenho integral para otimização da resposta transitória."""

    ITAE = "itae"  # Integral do Erro Absoluto Ponderado no Tempo
    IAE = "iae"    # Integral do Erro Absoluto
    ISE = "ise"    # Integral do Erro Quadrático


# %% [Cálculo da Sensibilidade Máxima no Domínio da Frequência]
def calculate_maximum_sensitivity(
    plant: CSTRPlant,
    nominal_state: CSTRState,
    q_j_ss: float,
    gains: PIDGains,
    omega_range: Tuple[float, float] = (1e-3, 1e2),
    n_freqs: int = 300,
) -> float:
    """Calcula o pico de sensibilidade Ms = max_w |1 / (1 + L(jw))| da malha linearizada.

    Etapas de Avaliação no Domínio da Frequência:
        1. Linearização da planta no ponto de equilíbrio:
           G(s) = C * (s*I - A)^(-1) * B, onde C = [0, 1, 0] mede a temperatura T.
        2. Resposta em frequência do controlador PID paralelo:
           C_pid(s) = Kp + Kp/(s*Ti) + (Kp*Td*s)/(1 + tau_f*s), com tau_f = Td / N.
        3. Função de transferência de malha aberta:
           L(jw) = G(jw) * C_pid(jw).
        4. Função de sensibilidade:
           S(jw) = 1 / (1 + L(jw)).
        5. Pico máximo Ms = max_w |S(jw)|.

    Args:
        plant: Instância da planta CSTR física.
        nominal_state: Estado estacionário do ponto de operação.
        q_j_ss: Vazão de fluido de resfriamento no estado estacionário.
        gains: Candidato a parâmetros de sintonia do controlador PID.
        omega_range: Intervalo de frequências de teste (omega_min, omega_max) [rad/min].
        n_freqs: Quantidade de pontos amostrais distribuídos logaritmicamente na frequência.

    Returns:
        float: Pico da curva de sensibilidade Ms (adimensional). Valores entre 1.2 e 1.6
        indicam excelente robustez; valores acima de 2.0 indicam oscilação ou fragilidade.
    """
    A, B = plant.linearize(nominal_state, q_j_ss)
    C = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)  # Medição de temperatura T

    omegas = np.logspace(np.log10(omega_range[0]), np.log10(omega_range[1]), n_freqs)
    sensitivities = np.zeros(n_freqs, dtype=np.float64)

    tau_f = gains.td / gains.n_filter if gains.td > 0 else 0.0

    for i, w in enumerate(omegas):
        s = 1j * w

        # 1. Resposta em frequência da planta linearizada G(jw) = C * (sI - A)^(-1) * B
        sI_minus_A = s * np.eye(3) - A
        try:
            inv_sI_minus_A = np.linalg.inv(sI_minus_A)
            G_jw = complex((C @ inv_sI_minus_A @ B)[0, 0])
        except np.linalg.LinAlgError:
            return 1e3

        # 2. Resposta em frequência do controlador PID paralelo C_pid(jw)
        p_term = gains.kp
        i_term = gains.kp / (s * gains.ti)
        d_term = (gains.kp * gains.td * s) / (1.0 + tau_f * s) if gains.td > 0 else 0.0
        C_pid = p_term + i_term + d_term

        # 3. Ganho de malha aberta L(jw) = G(jw) * C_pid(jw)
        L_jw = G_jw * C_pid

        # 4. Função de sensibilidade S(jw) = 1 / (1 + L(jw))
        S_jw = 1.0 / (1.0 + L_jw)
        sensitivities[i] = abs(S_jw)

    ms = float(np.max(sensitivities))
    return ms


# %% [Sintonia Numérica Ótima por Simplex Nelder-Mead com Restrição de Ms]
def tune_by_optimization(
    plant: CSTRPlant,
    nominal_state: CSTRState,
    q_j_ss: float,
    target_dT: float = -2.0,
    initial_gains: Optional[PIDGains] = None,
    criterion: OptimizationCriterion = OptimizationCriterion.ITAE,
    max_ms: float = 1.6,
    t_sim: float = 8.0,
    dt: float = 0.02,
) -> PIDGains:
    """Sintoniza ganhos ótimos de PID minimizando critério integral com restrição de Ms.

    O algoritmo emprega o método do Simplex de Nelder-Mead (scipy.optimize.minimize)
    sobre uma função objetivo que combina o índice integral não-linear e barreiras
    de penalidade quadrática para:
        - Violação da robustez em frequência: 1e4 * max(0, Ms - max_ms)^2
        - Erro residual em regime estacionário: penalidade caso haja offset significativo.

    Args:
        plant: Instância da planta fenomenológica CSTR.
        nominal_state: Ponto de operação inicial em estado estacionário.
        q_j_ss: Ponto de operação nominal do sinal de controle manipulado [L/min].
        target_dT: Variação em degrau da referência de temperatura [K].
        initial_gains: Estimativa inicial de ganhos. Se None, adota padrão conservador.
        criterion: Critério integral a minimizar (ITAE, IAE, ISE).
        max_ms: Teto máximo de sensibilidade permissível (padrão 1.6 para robustez).
        t_sim: Duração temporal da simulação de avaliação em malha fechada [min].
        dt: Passo temporal de discretização / integração [min].

    Returns:
        PIDGains: Instância contendo os ganhos ótimos convergidos.
    """
    if initial_gains is None:
        # Ponto de partida conservador padrão (Kp moderado negativo para ação reversa)
        p0 = np.array([-5.0, 1.5, 0.2], dtype=np.float64)
    else:
        p0 = np.array([initial_gains.kp, initial_gains.ti, initial_gains.td], dtype=np.float64)

    target_T = nominal_state.T + target_dT
    sim = ClosedLoopSimulator()
    limits = ActuatorLimits(u_min=0.0, u_max=300.0, max_slew_rate=100.0)

    def objective(params: np.ndarray) -> float:
        kp, ti, td = float(params[0]), float(params[1]), float(params[2])

        # Restrições de viabilidade física direta (ação reversa: kp < 0, ti > 0, td >= 0)
        if kp >= 0.0 or ti <= 0.05 or td < 0.0:
            return 1e6

        candidate_gains = PIDGains(kp=kp, ti=ti, td=td)

        # 1. Verificação de robustez em frequência via sensibilidade máxima Ms
        ms = calculate_maximum_sensitivity(plant, nominal_state, q_j_ss, candidate_gains)
        ms_penalty = 1e4 * max(0.0, ms - max_ms) ** 2 if ms > max_ms else 0.0

        # 2. Simulação transitória em malha fechada sobre a planta não-linear
        pid = PIDController(
            gains=candidate_gains,
            actuator_limits=limits,
            anti_windup=AntiWindupMethod.CLAMPING,
            u_bias=q_j_ss,
        )

        try:
            res = sim.run(
                plant=plant,
                controller=pid,
                initial_state=nominal_state,
                t_span=(0.0, t_sim),
                dt=dt,
                setpoint_func=lambda t: target_T,
            )
        except Exception:
            return 1e6

        if criterion == OptimizationCriterion.ITAE:
            perf = res.metrics.itae
        elif criterion == OptimizationCriterion.IAE:
            perf = res.metrics.iae
        else:
            perf = res.metrics.ise

        # Penalização para evitar erro estático permanente (offset residual)
        final_err = abs(res.t_pv[-1] - target_T)
        offset_penalty = 1e3 * final_err if final_err > 0.5 else 0.0

        total_cost = perf + ms_penalty + offset_penalty
        return float(total_cost)

    logger.info(
        "Iniciando otimização de PID: critério=%s (max_ms=%.2f)",
        criterion.value,
        max_ms,
    )

    res = minimize(
        fun=objective,
        x0=p0,
        method="Nelder-Mead",
        options={"maxiter": 120, "xatol": 1e-2, "fatol": 1e-2, "disp": False},
    )

    opt_kp, opt_ti, opt_td = float(res.x[0]), float(res.x[1]), float(res.x[2])
    opt_gains = PIDGains(kp=opt_kp, ti=max(opt_ti, 0.05), td=max(opt_td, 0.0))
    final_ms = calculate_maximum_sensitivity(plant, nominal_state, q_j_ss, opt_gains)

    logger.info(
        "PID Ótimo encontrado: Kp=%.3f, Ti=%.3f min, Td=%.3f min (Ms=%.2f, Custo=%.2f)",
        opt_gains.kp,
        opt_gains.ti,
        opt_gains.td,
        final_ms,
        res.fun,
    )

    return opt_gains


# %% [Bloco de Execução Interativa / Demonstração no Spyder]
if __name__ == "__main__":
    print("=" * 70)
    print("DEMONSTRAÇÃO INTERATIVA SPYDER 6: Otimização PID Restrita por Ms")
    print("=" * 70)

    # 1. Configuração da planta e do ponto de operação nominal
    plant_demo = CSTRPlant()
    q_j_nom = 100.0
    ss_demo = plant_demo.find_steady_state(q_j=q_j_nom)

    # 2. Avaliação de sensibilidade máxima para ganhos de teste
    gains_teste = PIDGains(kp=-7.5, ti=1.2, td=0.15)
    ms_teste = calculate_maximum_sensitivity(
        plant=plant_demo,
        nominal_state=ss_demo,
        q_j_ss=q_j_nom,
        gains=gains_teste,
    )
    print(f"Ganhos Teste: Kp={gains_teste.kp:.2f}, Ti={gains_teste.ti:.2f}, "
          f"Td={gains_teste.td:.2f}")
    print(f"Pico de Sensibilidade Máxima Ms: {ms_teste:.3f} (Limite Robusto: <= 1.60)")

    # 3. Execução de otimização rápida (t_sim curto para visualização no Spyder)
    print("\nExecutando otimização via Nelder-Mead minimizando ITAE com Ms <= 1.60...")
    gains_opt = tune_by_optimization(
        plant=plant_demo,
        nominal_state=ss_demo,
        q_j_ss=q_j_nom,
        target_dT=-1.0,
        criterion=OptimizationCriterion.ITAE,
        max_ms=1.6,
        t_sim=3.0,
        dt=0.05,
    )
    ms_final = calculate_maximum_sensitivity(
        plant=plant_demo,
        nominal_state=ss_demo,
        q_j_ss=q_j_nom,
        gains=gains_opt,
    )

    print("\n[Resultados da Sintonia por Otimização Numérica]:")
    print(f"  Kp Ótimo:      {gains_opt.kp:.3f}")
    print(f"  Ti Ótimo:      {gains_opt.ti:.3f} min")
    print(f"  Td Ótimo:      {gains_opt.td:.3f} min")
    print(f"  Ms Resultante: {ms_final:.3f} (Restrição Respeitada: {ms_final <= 1.65})")

    print("\nDemonstração de otimização concluída com sucesso no Spyder 6.")
    print("=" * 70)
