import pytest

from finrl.evaluation.metrics import (
    EpisodeResult,
    action_distribution,
    aggregate,
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
)


def _episode(values, returns=None, actions=None):
    return EpisodeResult(
        portfolio_values=values,
        returns=returns if returns is not None else [0.0] * (len(values) - 1),
        actions=actions if actions is not None else [0] * (len(values) - 1),
    )


def test_cumulative_return_known_values():
    ep = _episode([1.0, 1.1, 1.21])
    assert cumulative_return(ep) == pytest.approx(0.21)


def test_cumulative_return_loss():
    ep = _episode([1.0, 0.9, 0.8])
    assert cumulative_return(ep) == pytest.approx(-0.2)


def test_max_drawdown_known_values():
    # Peak at 1.2, trough at 0.6 -> 50% drawdown from that peak.
    ep = _episode([1.0, 1.2, 0.6, 0.9])
    assert max_drawdown(ep) == pytest.approx(0.5)


def test_max_drawdown_monotonic_increase_is_zero():
    ep = _episode([1.0, 1.1, 1.2, 1.3])
    assert max_drawdown(ep) == pytest.approx(0.0)


def test_sharpe_ratio_zero_variance_is_zero():
    ep = _episode([1.0, 1.0, 1.0], returns=[0.0, 0.0])
    assert sharpe_ratio(ep) == 0.0


def test_sharpe_ratio_single_return_is_zero():
    ep = _episode([1.0, 1.05], returns=[0.05])
    assert sharpe_ratio(ep) == 0.0


def test_action_distribution_fractions_sum_to_one():
    ep = _episode([1.0, 1.0, 1.0, 1.0], actions=[0, 0, 1, 2])
    dist = action_distribution(ep)
    assert dist[0] == pytest.approx(0.5)
    assert dist[1] == pytest.approx(0.25)
    assert dist[2] == pytest.approx(0.25)
    assert sum(dist.values()) == pytest.approx(1.0)


def test_action_distribution_empty_episode():
    ep = _episode([1.0], actions=[])
    assert action_distribution(ep) == {}


def test_aggregate_rejects_empty_list():
    with pytest.raises(ValueError):
        aggregate([])


def test_aggregate_stability_win_rate():
    winner = _episode([1.0, 1.1])
    loser = _episode([1.0, 0.9])
    result = aggregate([winner, loser])
    assert result.episode_count == 2
    assert result.stability_win_rate == pytest.approx(0.5)
    assert result.mean_cumulative_return == pytest.approx(0.0)
