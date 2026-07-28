"""FastAPI dependency-injection wiring — model loading is the one place
that knows about stable-baselines3 internals; routes.py only sees the
typed schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast

import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from finrl.config import Settings, get_settings
from finrl.env.portfolio_env import Action, PortfolioEnv

_MODEL_CLASSES: dict[str, type[BaseAlgorithm]] = {"ppo": PPO, "dqn": DQN}


@dataclass
class LoadedPolicy:
    model: BaseAlgorithm
    vec_normalize: VecNormalize
    algorithm: str
    reward_mode: str

    def predict(self, state: dict[str, float]) -> tuple[Action, list[float]]:
        obs = np.array(
            [state["volatility"], state["momentum"], state["drawdown"], state["cash_ratio"]],
            dtype=np.float32,
        ).reshape(1, -1)
        # normalize_obs can return dict[str, ndarray] for Dict observation
        # spaces; PortfolioEnv uses a flat Box space, so this is always
        # ndarray here.
        normalized_obs = cast(np.ndarray, self.vec_normalize.normalize_obs(obs))
        action, _ = self.model.predict(normalized_obs, deterministic=True)
        scores = self._per_action_scores(normalized_obs)
        return Action(int(action[0])), scores

    def _per_action_scores(self, normalized_obs: np.ndarray) -> list[float]:
        """DQN exposes Q-values per action directly. PPO's discrete policy
        exposes action probabilities via its distribution. Different
        underlying quantities (Q-value vs. probability) — both are real
        model outputs, neither is fabricated, but they are NOT
        interchangeable, which is why the response labels them
        "policy_scores" generically and the docs spell out the distinction.
        """
        obs_tensor = torch.as_tensor(normalized_obs, dtype=torch.float32)
        with torch.no_grad():
            if isinstance(self.model, DQN):
                q_values = self.model.q_net(obs_tensor)
                return [float(v) for v in q_values[0]]
            distribution = self.model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs
            return [float(v) for v in probs[0]]


@lru_cache
def get_cached_settings() -> Settings:
    return get_settings()


@lru_cache
def get_policy() -> LoadedPolicy:
    settings = get_cached_settings()
    model_dir = Path(settings.model_dir)
    algorithm = settings.algorithm
    reward_mode = settings.reward_mode

    model_path = model_dir / f"{algorithm}_{reward_mode}.zip"
    vecnorm_path = model_dir / f"{algorithm}_{reward_mode}_vecnormalize.pkl"
    if not model_path.exists() or not vecnorm_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path} (and its VecNormalize stats at "
            f"{vecnorm_path}). Run `python scripts/train.py --algorithm {algorithm} "
            f"--reward-mode {reward_mode}` first."
        )

    model_class = _MODEL_CLASSES[algorithm]
    model = model_class.load(str(model_path))

    dummy_venv = DummyVecEnv([lambda: PortfolioEnv(reward_mode=reward_mode)])
    vec_normalize = VecNormalize.load(str(vecnorm_path), dummy_venv)
    vec_normalize.training = False

    return LoadedPolicy(
        model=model, vec_normalize=vec_normalize, algorithm=algorithm, reward_mode=reward_mode
    )
