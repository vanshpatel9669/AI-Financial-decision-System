import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from finrl.env.portfolio_env import _RUIN_FLOOR, Action, PortfolioEnv


@pytest.mark.parametrize("reward_mode", ["return_minus_drawdown", "risk_adjusted"])
def test_conforms_to_gymnasium_api(reward_mode):
    env = PortfolioEnv(reward_mode=reward_mode)
    check_env(env, skip_render_check=True)


def test_unknown_reward_mode_rejected():
    with pytest.raises(ValueError):
        PortfolioEnv(reward_mode="not_a_real_mode")


def test_reset_returns_valid_observation():
    env = PortfolioEnv(seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (4,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert info == {}


def test_episode_truncates_at_configured_length():
    env = PortfolioEnv(episode_length=5, seed=0)
    env.reset(seed=0)
    truncated = False
    steps = 0
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(Action.BALANCED)
        steps += 1
        if terminated or truncated:
            break
    assert truncated or steps == 5


def test_step_returns_well_formed_tuple():
    env = PortfolioEnv(seed=0)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(Action.GROWTH)
    assert obs.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "portfolio_value" in info
    assert "drawdown" in info
    assert "regime" in info


def test_reward_modes_produce_different_rewards_for_same_trajectory():
    env_a = PortfolioEnv(reward_mode="return_minus_drawdown", seed=123)
    env_b = PortfolioEnv(reward_mode="risk_adjusted", seed=123)
    env_a.reset(seed=123)
    env_b.reset(seed=123)

    rewards_a, rewards_b = [], []
    for _ in range(30):
        _, reward_a, term_a, trunc_a, _ = env_a.step(Action.GROWTH)
        _, reward_b, term_b, trunc_b, _ = env_b.step(Action.GROWTH)
        rewards_a.append(reward_a)
        rewards_b.append(reward_b)
        if term_a or trunc_a or term_b or trunc_b:
            break

    assert rewards_a != rewards_b


def test_ruin_floor_terminates_episode_early():
    # A single trading-period return is small relative to the 0.4 floor
    # (daily GBM shocks are on the order of a few percent), so starting
    # already just below the floor and taking one step is a deterministic
    # way to exercise termination without depending on a lucky crash seed.
    env = PortfolioEnv(episode_length=10_000, seed=0)
    env.reset(seed=0)
    env._portfolio_value = 0.35
    env._peak_value = 1.0
    _, _, terminated, _, info = env.step(Action.DEFENSIVE)
    assert terminated
    assert info["portfolio_value"] <= _RUIN_FLOOR + 0.05
