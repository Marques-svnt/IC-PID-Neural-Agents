"""Pareto frontier analysis utilities for multi-objective PID tuning trade-off evaluation.

Provides pure-function tools to:
- Filter non-dominated (Pareto-optimal) solutions from a set of (IAE, TV) points.
- Compute normalized scalarized objectives for bi-objective optimization.
- Detect the robustness-constrained boundary (Ms <= threshold).
- Generate summary statistics of a Pareto front (knee point, extremes).

These utilities are deliberately decoupled from simulation or plotting logic so
they can be re-used across Artigo 1 (classical PID), Artigo 2 (neural PID), and
Artigo 3 (agentic supervisory control) without modification.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParetoPoint:
    """A single evaluated design point on the IAE–TV objective space.

    Attributes:
        iae: Integral Absolute Error [K·min] — tracking accuracy objective.
        tv: Total Variation of control action [L/min] — actuator wear objective.
        ms: Maximum sensitivity H-infinity norm ||S||_inf — robustness indicator.
        label: Optional human-readable identifier (e.g., tuning rule name).
        metadata: Arbitrary extra data (e.g., gains, tau_c, lambda weight).
    """

    iae: float
    tv: float
    ms: float
    label: str = ""
    metadata: dict = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class ParetoFrontierSummary:
    """Summary statistics computed from a Pareto-optimal front.

    Attributes:
        n_total: Total number of candidate points evaluated.
        n_dominated: Number of dominated (non-optimal) points.
        n_pareto: Number of non-dominated (Pareto-optimal) points.
        n_infeasible: Points violating the Ms robustness constraint.
        iae_min: Minimum IAE on the feasible Pareto front.
        iae_max: Maximum IAE on the feasible Pareto front.
        tv_min: Minimum TV on the feasible Pareto front.
        tv_max: Maximum TV on the feasible Pareto front.
        knee_point: The Pareto point closest to the utopia point (normalized distance).
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


# ---------------------------------------------------------------------------
# Core Pareto filtering
# ---------------------------------------------------------------------------


def is_dominated(point: ParetoPoint, candidates: Sequence[ParetoPoint]) -> bool:
    """Returns True if *point* is dominated by at least one candidate.

    A point p is dominated by q if q is no worse in all objectives AND strictly
    better in at least one. Both IAE and TV are to be minimized.

    Args:
        point: The candidate to evaluate.
        candidates: The full set of candidates to compare against.

    Returns:
        True if *point* is Pareto-dominated; False otherwise.
    """
    for other in candidates:
        if other is point:
            continue
        # other dominates point if: other <= point in all objectives AND < in at least one
        if (other.iae <= point.iae and other.tv <= point.tv) and (
            other.iae < point.iae or other.tv < point.tv
        ):
            return True
    return False


def filter_pareto_front(
    points: Sequence[ParetoPoint],
    ms_threshold: float = np.inf,
) -> List[ParetoPoint]:
    """Extracts the non-dominated Pareto front from a set of design points.

    Optionally filters out points that violate the H-infinity robustness
    constraint Ms <= ms_threshold before computing dominance.

    Args:
        points: All evaluated design points.
        ms_threshold: Maximum allowed peak sensitivity. Points with
            ``ms > ms_threshold`` are excluded from dominance analysis.
            Default is ``np.inf`` (no filtering).

    Returns:
        Sorted list of non-dominated ParetoPoint objects (ascending TV).

    Example:
        >>> candidates = [ParetoPoint(iae=2.0, tv=50.0, ms=1.3),
        ...               ParetoPoint(iae=1.5, tv=80.0, ms=1.5),
        ...               ParetoPoint(iae=2.5, tv=60.0, ms=1.4)]
        >>> front = filter_pareto_front(candidates, ms_threshold=1.6)
        >>> len(front)
        2
    """
    feasible = [p for p in points if p.ms <= ms_threshold]
    n_infeasible = len(points) - len(feasible)
    if n_infeasible:
        logger.info(
            "Excluded %d infeasible points (Ms > %.2f) from Pareto analysis.",
            n_infeasible,
            ms_threshold,
        )

    pareto_front = [p for p in feasible if not is_dominated(p, feasible)]
    pareto_front.sort(key=lambda p: p.tv)

    logger.info(
        "Pareto front: %d/%d feasible points are non-dominated.",
        len(pareto_front),
        len(feasible),
    )
    return pareto_front


# ---------------------------------------------------------------------------
# Normalization and scalarization
# ---------------------------------------------------------------------------


def normalize_objectives(
    points: Sequence[ParetoPoint],
    iae_ref: Optional[float] = None,
    tv_ref: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Normalizes IAE and TV to [0, 1] using reference (utopia) values.

    Args:
        points: Design points to normalize.
        iae_ref: Reference IAE for normalization. If None, uses ``max(IAE)``.
        tv_ref: Reference TV for normalization. If None, uses ``max(TV)``.

    Returns:
        List of (iae_norm, tv_norm) tuples corresponding to each input point.

    Raises:
        ValueError: If *points* is empty.
    """
    if not points:
        raise ValueError("Cannot normalize an empty set of points.")

    iae_values = np.array([p.iae for p in points])
    tv_values = np.array([p.tv for p in points])

    iae_ref = iae_ref if iae_ref is not None else float(np.max(iae_values))
    tv_ref = tv_ref if tv_ref is not None else float(np.max(tv_values))

    if iae_ref <= 0 or tv_ref <= 0:
        raise ValueError(f"Reference values must be positive. Got iae_ref={iae_ref}, tv_ref={tv_ref}.")

    return [(float(p.iae / iae_ref), float(p.tv / tv_ref)) for p in points]


def scalarized_objective(
    iae: float,
    tv: float,
    lam: float,
    iae_ref: float,
    tv_ref: float,
) -> float:
    """Computes the bi-objective scalarized cost: lam*(IAE/IAE0) + (1-lam)*(TV/TV0).

    Args:
        iae: Raw IAE value [K·min].
        tv: Raw TV value [L/min].
        lam: Trade-off weight in [0, 1]. lam=1 → pure IAE, lam=0 → pure TV.
        iae_ref: Normalizing IAE reference (e.g., baseline SIMC or ITAE value).
        tv_ref: Normalizing TV reference.

    Returns:
        Scalar cost value.

    Raises:
        ValueError: If *lam* is outside [0, 1].
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"Trade-off weight lam must be in [0, 1]; got {lam}.")
    return float(lam * (iae / iae_ref) + (1.0 - lam) * (tv / tv_ref))


# ---------------------------------------------------------------------------
# Knee point detection
# ---------------------------------------------------------------------------


def find_knee_point(pareto_front: Sequence[ParetoPoint]) -> Optional[ParetoPoint]:
    """Identifies the knee point of a Pareto front via minimum normalized distance to utopia.

    The utopia point is defined as (min_IAE, min_TV) across the front. The knee
    point is the non-dominated solution with the smallest Euclidean distance to
    this ideal — representing the best balance between both objectives.

    Args:
        pareto_front: A sequence of non-dominated ParetoPoint objects.

    Returns:
        The ParetoPoint closest to the utopia point, or None if front is empty.
    """
    if not pareto_front:
        logger.warning("Empty Pareto front — cannot find knee point.")
        return None

    iae_vals = np.array([p.iae for p in pareto_front])
    tv_vals = np.array([p.tv for p in pareto_front])

    # Normalize to [0, 1]
    iae_range = iae_vals.max() - iae_vals.min() or 1.0
    tv_range = tv_vals.max() - tv_vals.min() or 1.0

    iae_norm = (iae_vals - iae_vals.min()) / iae_range
    tv_norm = (tv_vals - tv_vals.min()) / tv_range

    # Distance to utopia (0, 0) in normalized space
    distances = np.sqrt(iae_norm**2 + tv_norm**2)
    knee_idx = int(np.argmin(distances))

    logger.info(
        "Knee point: IAE=%.3f, TV=%.1f, Ms=%.3f (label='%s')",
        pareto_front[knee_idx].iae,
        pareto_front[knee_idx].tv,
        pareto_front[knee_idx].ms,
        pareto_front[knee_idx].label,
    )
    return pareto_front[knee_idx]


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def summarize_pareto_front(
    candidates: Sequence[ParetoPoint],
    ms_threshold: float = 1.6,
) -> ParetoFrontierSummary:
    """Computes summary statistics for a Pareto frontier analysis.

    Filters the feasible Pareto front, then computes bounds, counts, and
    identifies the knee point.

    Args:
        candidates: Full set of evaluated design points (dominated + non-dominated).
        ms_threshold: H-infinity robustness constraint. Default 1.6 per SIMC guidelines.

    Returns:
        ParetoFrontierSummary with all computed statistics.
    """
    feasible = [p for p in candidates if p.ms <= ms_threshold]
    n_infeasible = len(candidates) - len(feasible)

    pareto_front = filter_pareto_front(candidates, ms_threshold=ms_threshold)
    n_pareto = len(pareto_front)
    n_dominated = len(feasible) - n_pareto

    if not pareto_front:
        logger.warning("No feasible Pareto-optimal points found with Ms <= %.2f.", ms_threshold)
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
        "Pareto summary: total=%d, pareto=%d, dominated=%d, infeasible=%d | "
        "IAE in [%.3f, %.3f], TV in [%.1f, %.1f]",
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
