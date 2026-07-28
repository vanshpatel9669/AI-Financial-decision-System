"""Builds PPO/DQN agents (stable-baselines3, PyTorch-backed) over
PortfolioEnv. One factory function per algorithm so scripts/train.py and
scripts/run_benchmark.py share the exact same hyperparameters — a benchmark
comparing two algorithms is only meaningful if nothing else differs between
them.

Observations are normalized (VecNormalize) — found empirically, not assumed:
without it, both PPO and DQN converged to a policy that always picked the
same action regardless of state (verified via action-distribution logging,
not inferred from return alone). volatility/momentum/drawdown/cash_ratio
have very different natural scales, and unnormalized inputs to a small MLP
made the state signal too weak to learn from before the policy settled into
a low-variance local optimum. Normalization must be applied identically at
train and serve time — VecNormalize's running statistics are saved
alongside the model (see scripts/train.py) and reloaded at inference (see
evaluation/backtest.py, api/dependencies.py) rather than only living inside
the training process.
"""

from __future__ import annotations

from typing import Literal

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from finrl.env.portfolio_env import PortfolioEnv, RewardMode

AlgorithmName = Literal["ppo", "dqn"]


def make_vec_env(reward_mode: RewardMode, seed: int, training: bool = True) -> VecNormalize:
    """norm_reward is always False: evaluation code needs the environment's
    real reward-mode-specific reward to be comparable, and letting
    VecNormalize rescale it would make cross-run reward comparisons
    meaningless. Only observations are normalized.
    """
    venv = DummyVecEnv([lambda: PortfolioEnv(reward_mode=reward_mode, seed=seed)])
    return VecNormalize(venv, training=training, norm_obs=True, norm_reward=False)


def build_agent(
    algorithm: AlgorithmName,
    reward_mode: RewardMode,
    seed: int = 0,
    verbose: int = 0,
) -> tuple[BaseAlgorithm, VecNormalize]:
    env = make_vec_env(reward_mode, seed, training=True)

    if algorithm == "ppo":
        model: BaseAlgorithm = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=verbose,
            n_steps=256,
            batch_size=64,
            learning_rate=3e-4,
            ent_coef=0.01,
        )
    elif algorithm == "dqn":
        model = DQN(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=verbose,
            learning_rate=1e-3,
            buffer_size=20_000,
            learning_starts=500,
            train_freq=4,
            target_update_interval=500,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}. Supported: ppo, dqn.")

    return model, env
