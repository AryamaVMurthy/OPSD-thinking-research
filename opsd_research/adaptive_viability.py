"""Exact small-sample Beta posterior utilities for adaptive GRAF routing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence


@dataclass(frozen=True)
class ActionEvidence:
    action_id: str
    successes: int
    trials: int

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must be nonempty")
        if (
            self.trials < 0
            or self.successes < 0
            or self.successes > self.trials
        ):
            raise ValueError("action evidence counts are inconsistent")


def beta_posterior_mean(
    successes: int, trials: int, *, prior: int = 1
) -> float:
    """Mean of Beta(successes+prior, failures+prior)."""
    _validate_counts(successes, trials, prior)
    return (successes + prior) / (trials + 2 * prior)


def _validate_counts(successes: int, trials: int, prior: int) -> None:
    if (
        not isinstance(successes, int)
        or not isinstance(trials, int)
        or not isinstance(prior, int)
        or prior < 1
        or trials < 0
        or successes < 0
        or successes > trials
    ):
        raise ValueError(
            "Beta evidence requires integer 0 <= successes <= trials "
            "and a positive integer prior"
        )


def _integer_beta_cdf(value: float, alpha: int, beta: int) -> float:
    """Regularized incomplete beta for positive integer parameters.

    For integer alpha and beta this is a binomial tail, which is exact enough
    for the two-to-four-sample routing regime and avoids a SciPy dependency.
    """
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    order = alpha + beta - 1
    complement = 1.0 - value
    return sum(
        math.comb(order, index)
        * value**index
        * complement ** (order - index)
        for index in range(alpha, order + 1)
    )


def _integer_beta_quantile(probability: float, alpha: int, beta: int) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must be in [0, 1]")
    if probability == 0.0:
        return 0.0
    if probability == 1.0:
        return 1.0
    lower, upper = 0.0, 1.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if _integer_beta_cdf(midpoint, alpha, beta) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def beta_credible_interval(
    successes: int,
    trials: int,
    *,
    credible_level: float = 0.9,
    prior: int = 1,
) -> tuple[float, float]:
    """Equal-tailed exact interval for an integer-parameter Beta posterior."""
    _validate_counts(successes, trials, prior)
    if not 0.0 < credible_level < 1.0:
        raise ValueError("credible_level must be in (0, 1)")
    tail = 0.5 * (1.0 - credible_level)
    alpha = successes + prior
    beta = trials - successes + prior
    return (
        _integer_beta_quantile(tail, alpha, beta),
        _integer_beta_quantile(1.0 - tail, alpha, beta),
    )


def next_uncertain_action(
    evidence: Sequence[ActionEvidence],
    *,
    max_trials: int,
    credible_level: float = 0.9,
    prior: int = 1,
) -> str | None:
    """Choose one overlapping contender, or stop after credible separation."""
    if len(evidence) < 2:
        return None
    if max_trials < 1:
        raise ValueError("max_trials must be positive")
    intervals = {
        item.action_id: beta_credible_interval(
            item.successes,
            item.trials,
            credible_level=credible_level,
            prior=prior,
        )
        for item in evidence
    }
    means = {
        item.action_id: beta_posterior_mean(
            item.successes, item.trials, prior=prior
        )
        for item in evidence
    }
    best = max(evidence, key=lambda item: (means[item.action_id], item.action_id))
    best_lower = intervals[best.action_id][0]
    other_upper = max(
        intervals[item.action_id][1]
        for item in evidence
        if item.action_id != best.action_id
    )
    if best_lower > other_upper:
        return None
    contenders = [
        item
        for item in evidence
        if item.trials < max_trials
        and intervals[item.action_id][1] >= best_lower
    ]
    if not contenders:
        return None
    selected = max(
        contenders,
        key=lambda item: (
            intervals[item.action_id][1] - intervals[item.action_id][0],
            -item.trials,
            item.action_id,
        ),
    )
    return selected.action_id
