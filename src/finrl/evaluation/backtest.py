"""Runs a trained policy over held-out market scenarios (seeds the policy
never trained on) and collects real trajectories for evaluation.metrics.

Observations are normalized with the SAME running statistics the policy was
trained under (via the saved VecNormalize state) before being passed to
model.predict() — a policy trained on normalized inputs and evaluated on raw
ones is silently evaluating a different function than the one it learned to
optimize. See agents/factory.py's docstring for how this was found.
"""

from __future__ import annotations

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecNormalize

from finrl.env.portfolio_env import PortfolioEnv, RewardMode
from finrl.evaluation.metrics import EpisodeResult


def run_episode(
    model: BaseAlgorithm, vec_normalize: VecNormalize, reward_mode: RewardMode, seed: int
) -> EpisodeResult:
    env = PortfolioEnv(reward_mode=reward_mode, seed=seed)
    obs, _info = env.reset(seed=seed)

    portfolio_values = [1.0]
    returns: list[float] = []
    actions: list[int] = []

    terminated = truncated = False
    while not (terminated or truncated):
        normalized_obs = vec_normalize.normalize_obs(obs.reshape(1, -1))
        action, _state = model.predict(normalized_obs, deterministic=True)
        action_int = int(action[0]) if hasattr(action, "__len__") else int(action)
        obs, _reward, terminated, truncated, info = env.step(action_int)
        actions.append(action_int)
        portfolio_values.append(info["portfolio_value"])
        returns.append(portfolio_values[-1] / portfolio_values[-2] - 1.0)

    return EpisodeResult(portfolio_values=portfolio_values, returns=returns, actions=actions)


def run_backtest(
    model: BaseAlgorithm, vec_normalize: VecNormalize, reward_mode: RewardMode, seeds: list[int]
) -> list[EpisodeResult]:
    """`seeds` should be disjoint from whatever seed(s) were used for
    training — held-out evaluation, not train-set performance.
    """
    return [run_episode(model, vec_normalize, reward_mode, seed) for seed in seeds]
