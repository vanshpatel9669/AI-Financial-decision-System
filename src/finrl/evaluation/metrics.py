"""Portfolio performance metrics computed from real backtest trajectories.

Every metric here is a pure function over data a backtest actually
produced — nothing here is a target value or a claim, only a measurement.
Metric definitions are stated explicitly and are deliberately simple,
standard formulas — not tuned to reproduce any particular number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_ANNUALIZATION_FACTOR = 252


@dataclass(frozen=True)
class EpisodeResult:
    """One backtest episode's trajectory."""

    portfolio_values: list[float]
    returns: list[float]
    actions: list[int]


def cumulative_return(episode: EpisodeResult) -> float:
    if not episode.portfolio_values:
        return 0.0
    return episode.portfolio_values[-1] / episode.portfolio_values[0] - 1.0


def max_drawdown(episode: EpisodeResult) -> float:
    values = np.array(episode.portfolio_values)
    if len(values) == 0:
        return 0.0
    running_peak = np.maximum.accumulate(values)
    drawdowns = 1.0 - values / running_peak
    return float(np.max(drawdowns))


def sharpe_ratio(episode: EpisodeResult) -> float:
    """Annualized, assuming a zero risk-free rate (a stated simplification)."""
    returns = np.array(episode.returns)
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(_ANNUALIZATION_FACTOR))


def action_distribution(episode: EpisodeResult) -> dict[int, float]:
    """Fraction of steps the policy chose each action — reveals a policy
    that collapsed to always picking one action, which raw returns alone
    wouldn't necessarily surface.
    """
    if not episode.actions:
        return {}
    actions = np.array(episode.actions)
    return {int(a): float(np.mean(actions == a)) for a in np.unique(actions)}


@dataclass(frozen=True)
class AggregateResult:
    """Summary across many held-out episodes (different market seeds)."""

    episode_count: int
    mean_cumulative_return: float
    std_cumulative_return: float
    mean_max_drawdown: float
    mean_sharpe_ratio: float
    # Defined explicitly, not left implicit: the fraction of held-out
    # episodes where the policy did not lose money. This is the metric this
    # project calls "policy stability" — a specific, reproducible
    # definition, not a number chosen to match any external claim.
    stability_win_rate: float


def aggregate(episodes: list[EpisodeResult]) -> AggregateResult:
    if not episodes:
        raise ValueError("Cannot aggregate zero episodes.")

    cumulative_returns = np.array([cumulative_return(e) for e in episodes])
    drawdowns = np.array([max_drawdown(e) for e in episodes])
    sharpes = np.array([sharpe_ratio(e) for e in episodes])

    return AggregateResult(
        episode_count=len(episodes),
        mean_cumulative_return=float(np.mean(cumulative_returns)),
        std_cumulative_return=float(np.std(cumulative_returns)),
        mean_max_drawdown=float(np.mean(drawdowns)),
        mean_sharpe_ratio=float(np.mean(sharpes)),
        stability_win_rate=float(np.mean(cumulative_returns >= 0)),
    )
