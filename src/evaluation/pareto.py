# %% [Módulo e Importações]
"""Módulo de análise de fronteira de Pareto para sintonia bi-objetivo de controladores.

Fornece utilitários analíticos desacoplados de simulação para:
    1. Filtragem de soluções não-dominadas (Pareto-ótimas) no espaço bi-objetivo (IAE vs TV).
    2. Restrição de robustez em frequência: exclusão de soluções com pico Ms > ms_threshold.
    3. Normalização e escalarização convexa de objetivos:
       J(lambda) = lambda * (IAE / IAE_ref) + (1 - lambda) * (TV / TV_ref).
    4. Detecção analítica do ponto de joelho (knee point) pela menor distância euclidiana
       ao ponto utópico ideal no espaço bi-objetivo normalizado.
    5. Consolidação estatística da fronteira para geração de tabelas e síntese de resultados.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# %% [Estruturas de Dados de Pareto]
@dataclass(frozen=True)
class ParetoPoint:
    """Representa um ponto de projeto avaliado no espaço de objetivos IAE-TV.

    Attributes:
        iae: Integral do Erro Absoluto [K·min] (objetivo de rastreamento).
        tv: Variação Total da ação de controle [L/min] (objetivo de desgaste da válvula).
        ms: Norma H-infinito da função de sensibilidade ||S||_inf (indicador de robustez).
        label: Identificador textual descritivo (ex.: nome da regra de sintonia).
        metadata: Dicionário opcional para dados adicionais (ganhos Kp, Ti, Td, pesos lambda).
    """

    iae: float
    tv: float
    ms: float
    label: str = ""
    metadata: dict = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class ParetoFrontierSummary:
    """Estatísticas consolidadas e resumo quantitativo da fronteira de Pareto.

    Attributes:
        n_total: Quantidade total de pontos de projeto avaliados.
        n_dominated: Quantidade de pontos estritamente dominados (subótimos).
        n_pareto: Quantidade de soluções não-dominadas na fronteira ótima.
        n_infeasible: Quantidade de candidatos descartados por violarem Ms <= ms_threshold.
        iae_min: Menor IAE observado na fronteira factível.
        iae_max: Maior IAE observado na fronteira factível.
        tv_min: Menor TV observado na fronteira factível.
        tv_max: Maior TV observado na fronteira factível.
        knee_point: Ponto de joelho com melhor compromisso trade-off em relação à utopia.
    """

    n_total: int
    n_dominated: int
    n_pareto: int
    n_infeasible: int
    iae_min: float
    iae_max: float
    tv_min: float
    tv_max: float
    knee_point: Optional[ParetoPoint]


# %% [Filtragem e Dominância de Pareto]
def is_dominated(point: ParetoPoint, candidates: Sequence[ParetoPoint]) -> bool:
    """Verifica se um determinado ponto de projeto é dominado por algum outro candidato.

    Critério de Dominância de Pareto (Minimização Conjunta de IAE e TV):
        Um ponto p é dominado por q se, e somente se:
            (IAE_q <= IAE_p e TV_q <= TV_p) E (IAE_q < IAE_p ou TV_q < TV_p)

    Args:
        point: Candidato em avaliação.
        candidates: Conjunto total de candidatos contra os quais a comparação é efetuada.

    Returns:
        bool: True se o ponto for estritamente dominado por ao menos um candidato;
            False caso contrário.
    """
    for other in candidates:
        if other is point:
            continue
        # other domina point se for não-pior em ambos e estritamente melhor em pelo menos um
        if (other.iae <= point.iae and other.tv <= point.tv) and (
            other.iae < point.iae or other.tv < point.tv
        ):
            return True
    return False


def filter_pareto_front(
    points: Sequence[ParetoPoint],
    ms_threshold: float = np.inf,
) -> List[ParetoPoint]:
    """Extrai o conjunto não-dominado de Pareto respeitando a restrição de robustez Ms.

    Etapas de Filtragem:
        1. Descarte de candidatos inviáveis onde ms > ms_threshold (violação de robustez).
        2. Teste de dominância de Pareto par a par entre os candidatos factíveis remanescentes.
        3. Ordenação final da fronteira em ordem ascendente de esforço de controle (TV).

    Args:
        points: Sequência contendo todos os pontos de projeto avaliados.
        ms_threshold: Teto máximo tolerável para o pico de sensibilidade Ms.
            Padrão np.inf (sem descarte por robustez).

    Returns:
        List[ParetoPoint]: Lista ordenada dos pontos Pareto-ótimos não-dominados.
    """
    feasible = [p for p in points if p.ms <= ms_threshold]
    n_infeasible = len(points) - len(feasible)
    if n_infeasible:
        logger.info(
            "Excluídos %d pontos inviáveis (Ms > %.2f) da análise de Pareto.",
            n_infeasible,
            ms_threshold,
        )

    pareto_front = [p for p in feasible if not is_dominated(p, feasible)]
    pareto_front.sort(key=lambda p: p.tv)

    logger.info(
        "Fronteira de Pareto: %d/%d pontos factíveis são não-dominados.",
        len(pareto_front),
        len(feasible),
    )
    return pareto_front


# %% [Normalização e Escalarização de Objetivos]
def normalize_objectives(
    points: Sequence[ParetoPoint],
    iae_ref: Optional[float] = None,
    tv_ref: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Normaliza as métricas IAE e TV para a escala adimensional [0, 1].

    Args:
        points: Sequência de pontos de projeto a serem normalizados.
        iae_ref: Valor de referência para normalização do IAE. Se None, adota max(IAE).
        tv_ref: Valor de referência para normalização do TV. Se None, adota max(TV).

    Returns:
        List[Tuple[float, float]]: Lista de pares (iae_normalizado, tv_normalizado).

    Raises:
        ValueError: Caso a sequência de pontos esteja vazia ou referências não sejam positivas.
    """
    if not points:
        raise ValueError("Cannot normalize an empty set of points.")

    iae_values = np.array([p.iae for p in points])
    tv_values = np.array([p.tv for p in points])

    iae_ref = iae_ref if iae_ref is not None else float(np.max(iae_values))
    tv_ref = tv_ref if tv_ref is not None else float(np.max(tv_values))

    if iae_ref <= 0 or tv_ref <= 0:
        raise ValueError(
            f"Reference values must be positive. Got iae_ref={iae_ref}, tv_ref={tv_ref}."
        )

    return [(float(p.iae / iae_ref), float(p.tv / tv_ref)) for p in points]


def scalarized_objective(
    iae: float,
    tv: float,
    lam: float,
    iae_ref: float,
    tv_ref: float,
) -> float:
    """Calcula o custo escalarizado bi-objetivo: lam*(IAE/IAE0) + (1-lam)*(TV/TV0).

    Args:
        iae: Valor bruto de IAE obtido [K·min].
        tv: Valor bruto de TV obtido [L/min].
        lam: Fator de ponderação no intervalo [0, 1].
            lam = 1 prioriza unicamente rastreamento rápido (IAE);
            lam = 0 prioriza unicamente preservação da válvula (TV).
        iae_ref: Valor de normalização de referência para IAE.
        tv_ref: Valor de normalização de referência para TV.

    Returns:
        float: Valor do custo escalar agregado.

    Raises:
        ValueError: Se lam estiver fora do intervalo unitário [0, 1].
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"Trade-off weight lam must be in [0, 1]; got {lam}.")
    return float(lam * (iae / iae_ref) + (1.0 - lam) * (tv / tv_ref))


# %% [Detecção de Ponto de Joelho (Knee Point)]
def find_knee_point(pareto_front: Sequence[ParetoPoint]) -> Optional[ParetoPoint]:
    """Identifica o ponto de joelho (knee point) da fronteira de Pareto.

    O ponto de joelho representa a solução de melhor compromisso operacional entre
    rastreamento agressivo e suavidade da atuação. É determinado localizando
    a solução não-dominada cuja distância euclidiana normalizada ao ponto utópico
    ideal (min_IAE, min_TV) seja mínima.

    Args:
        pareto_front: Sequência de soluções Pareto-ótimas não-dominadas.

    Returns:
        Optional[ParetoPoint]: O ponto de joelho ótimo, ou None se a fronteira for vazia.
    """
    if not pareto_front:
        logger.warning("Fronteira de Pareto vazia — impossível determinar o ponto de joelho.")
        return None

    iae_vals = np.array([p.iae for p in pareto_front])
    tv_vals = np.array([p.tv for p in pareto_front])

    # Normalização min-max para a faixa [0, 1]
    iae_range = iae_vals.max() - iae_vals.min() or 1.0
    tv_range = tv_vals.max() - tv_vals.min() or 1.0

    iae_norm = (iae_vals - iae_vals.min()) / iae_range
    tv_norm = (tv_vals - tv_vals.min()) / tv_range

    # Distância Euclidiana ao ponto utópico (0, 0) no espaço adimensional
    distances = np.sqrt(iae_norm**2 + tv_norm**2)
    knee_idx = int(np.argmin(distances))

    logger.info(
        "Ponto de joelho detectado: IAE=%.3f, TV=%.1f, Ms=%.3f (label='%s')",
        pareto_front[knee_idx].iae,
        pareto_front[knee_idx].tv,
        pareto_front[knee_idx].ms,
        pareto_front[knee_idx].label,
    )
    return pareto_front[knee_idx]


# %% [Resumo Estatístico da Fronteira de Pareto]
def summarize_pareto_front(
    candidates: Sequence[ParetoPoint],
    ms_threshold: float = 1.6,
) -> ParetoFrontierSummary:
    """Gera um resumo estatístico abrangente da análise de fronteira de Pareto.

    Filtra os candidatos pela restrição de robustez em frequência (Ms <= ms_threshold),
    isola as soluções não-dominadas, identifica os limites e o ponto de joelho.

    Args:
        candidates: Conjunto completo de candidatos avaliados (viáveis, dominados e inviáveis).
        ms_threshold: Restrição de sensibilidade máxima (padrão 1.6 conforme diretrizes SIMC).

    Returns:
        ParetoFrontierSummary: Estrutura consolidada com contagens, limites e ponto de joelho.
    """
    feasible = [p for p in candidates if p.ms <= ms_threshold]
    n_infeasible = len(candidates) - len(feasible)

    pareto_front = filter_pareto_front(candidates, ms_threshold=ms_threshold)
    n_pareto = len(pareto_front)
    n_dominated = len(feasible) - n_pareto

    if not pareto_front:
        logger.warning(
            "Nenhum ponto Pareto-ótimo factível encontrado com Ms <= %.2f.", ms_threshold
        )
        return ParetoFrontierSummary(
            n_total=len(candidates),
            n_dominated=n_dominated,
            n_pareto=0,
            n_infeasible=n_infeasible,
            iae_min=float("nan"),
            iae_max=float("nan"),
            tv_min=float("nan"),
            tv_max=float("nan"),
            knee_point=None,
        )

    iae_values = [p.iae for p in pareto_front]
    tv_values = [p.tv for p in pareto_front]
    knee = find_knee_point(pareto_front)

    summary = ParetoFrontierSummary(
        n_total=len(candidates),
        n_dominated=n_dominated,
        n_pareto=n_pareto,
        n_infeasible=n_infeasible,
        iae_min=min(iae_values),
        iae_max=max(iae_values),
        tv_min=min(tv_values),
        tv_max=max(tv_values),
        knee_point=knee,
    )

    logger.info(
        "Resumo Pareto: total=%d, pareto=%d, dominados=%d, inviáveis=%d | "
        "IAE em [%.3f, %.3f], TV em [%.1f, %.1f]",
        summary.n_total,
        summary.n_pareto,
        summary.n_dominated,
        summary.n_infeasible,
        summary.iae_min,
        summary.iae_max,
        summary.tv_min,
        summary.tv_max,
    )
    return summary


# %% [Bloco de Execução Interativa / Demonstração no Spyder]
if __name__ == "__main__":
    print("=" * 70)
    print("DEMONSTRAÇÃO INTERATIVA SPYDER 6: Fronteira de Pareto Multi-Objetivo")
    print("=" * 70)

    # Conjunto de candidatos simulando diferentes sintonias de PID na malha do CSTR
    pontos_candidatos = [
        ParetoPoint(iae=3.5, tv=45.0,  ms=1.35, label="SIMC Muito Conservador"),
        ParetoPoint(iae=2.4, tv=65.0,  ms=1.45, label="SIMC Padrão (tc=theta)"),
        ParetoPoint(iae=1.8, tv=95.0,  ms=1.58, label="Otimização ITAE (Ms <= 1.6)"),
        ParetoPoint(iae=1.4, tv=180.0, ms=1.85, label="Ziegler-Nichols (Inviável Ms>1.6)"),
        ParetoPoint(iae=2.7, tv=85.0,  ms=1.50, label="Candidato Subótimo Dominado"),
        ParetoPoint(iae=1.6, tv=130.0, ms=1.59, label="Otimização IAE Agressiva"),
    ]

    print(f"Avaliando {len(pontos_candidatos)} candidatos sob restrição Ms <= 1.60...")
    resumo = summarize_pareto_front(pontos_candidatos, ms_threshold=1.60)

    print("\n[Resumo Consolidado da Fronteira de Pareto]:")
    print(f"  Total de Candidatos:   {resumo.n_total}")
    print(f"  Inviáveis (Ms > 1.60): {resumo.n_infeasible}")
    print(f"  Dominados (Subótimos): {resumo.n_dominated}")
    print(f"  Soluções Pareto-Ótimas: {resumo.n_pareto}")
    print(f"  Faixa de IAE Factível: [{resumo.iae_min:.3f}, {resumo.iae_max:.3f}] K·min")
    print(f"  Faixa de TV Factível:  [{resumo.tv_min:.1f}, {resumo.tv_max:.1f}] L/min")

    if resumo.knee_point:
        kp = resumo.knee_point
        print("\n[Ponto de Joelho Ótimo Detectado (Trade-off Balanceado)]:")
        print(f"  Identificador: '{kp.label}'")
        print(f"  IAE:           {kp.iae:.3f} K·min")
        print(f"  TV:            {kp.tv:.1f} L/min")
        print(f"  Sensibilidade: {kp.ms:.2f}")

    print("\nDemonstração de análise de Pareto concluída com sucesso no Spyder 6.")
    print("=" * 70)
