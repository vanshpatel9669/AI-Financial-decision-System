"""Gymnasium environment for discrete-allocation portfolio decisions over a
simulated multi-regime market.

Each episode is a sequence of steps (a "trading period"), not a single
stateless lookup: the agent's allocation choice changes exposure to
whatever the market does *next*, which is a real sequential-decision
problem — unlike the previous system's per-call reward function, which
computed a reward for a state that never evolved.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from finrl.env.market_simulator import MarketSimulator

_VOLATILITY_WINDOW = 20
_MOMENTUM_WINDOW = 10
_EPISODE_LENGTH = 60
_RUIN_FLOOR = 0.4  # episode ends early if portfolio value drops below 40% of start
_ANNUALIZATION_FACTOR = 252


class Action(IntEnum):
    DEFENSIVE = 0
    BALANCED = 1
    GROWTH = 2


# Fraction of the portfolio allocated to the risky asset under each action;
# the remainder sits in cash, which is assumed to earn zero return (a
# deliberately simple, stated assumption — see docs/architecture.md).
_RISK_EXPOSURE = {
    Action.DEFENSIVE: 0.20,
    Action.BALANCED: 0.55,
    Action.GROWTH: 0.90,
}

RewardMode = str  # "return_minus_drawdown" | "risk_adjusted"


class PortfolioEnv(gym.Env):
    """Observation: [volatility, momentum, drawdown, cash_ratio] — same
    feature names the original system used, now computed from a real
    simulated trajectory rather than sampled independently each call.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_mode: RewardMode = "return_minus_drawdown",
        episode_length: int = _EPISODE_LENGTH,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if reward_mode not in ("return_minus_drawdown", "risk_adjusted"):
            raise ValueError(f"Unknown reward_mode: {reward_mode!r}")
        self.reward_mode = reward_mode
        self.episode_length = episode_length

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )

        self._market = MarketSimulator(seed=seed)
        self._portfolio_value = 1.0
        self._peak_value = 1.0
        self._step_count = 0
        self._returns_log: list[float] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._market.reset(seed=seed)
        self._portfolio_value = 1.0
        self._peak_value = 1.0
        self._step_count = 0
        self._returns_log = []
        return self._observation(cash_ratio=1.0 - _RISK_EXPOSURE[Action.BALANCED]), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        exposure = _RISK_EXPOSURE[Action(action)]
        market_return = self._market.step()
        portfolio_return = exposure * market_return  # cash portion earns 0

        self._portfolio_value *= 1.0 + portfolio_return
        self._peak_value = max(self._peak_value, self._portfolio_value)
        self._returns_log.append(portfolio_return)
        self._step_count += 1

        drawdown = 1.0 - self._portfolio_value / self._peak_value
        reward = self._compute_reward(portfolio_return, drawdown)

        terminated = self._portfolio_value <= _RUIN_FLOOR
        truncated = self._step_count >= self.episode_length
        info = {
            "portfolio_value": self._portfolio_value,
            "drawdown": drawdown,
            "regime": int(self._market.regime),
        }
        return self._observation(cash_ratio=1.0 - exposure), reward, terminated, truncated, info

    def _compute_reward(self, portfolio_return: float, drawdown: float) -> float:
        if self.reward_mode == "return_minus_drawdown":
            return portfolio_return - 0.5 * drawdown
        # risk_adjusted: reward the return relative to how volatile recent
        # returns have been — a genuinely different optimization target
        # (penalizes volatility directly, not just realized drawdown).
        recent = np.array(self._returns_log[-_VOLATILITY_WINDOW:])
        vol = float(np.std(recent)) if len(recent) >= 2 else 0.0
        return portfolio_return / (vol + 1e-4)

    def _observation(self, cash_ratio: float) -> np.ndarray:
        trailing = self._market.trailing_returns(_VOLATILITY_WINDOW)
        if len(trailing) >= 2:
            volatility = float(np.std(trailing)) * np.sqrt(_ANNUALIZATION_FACTOR)
        else:
            volatility = 0.0
        momentum_window = self._market.trailing_returns(_MOMENTUM_WINDOW)
        momentum = float(np.sum(momentum_window)) if len(momentum_window) > 0 else 0.0
        drawdown = 1.0 - self._portfolio_value / self._peak_value

        return np.array(
            [
                np.clip(volatility, 0.0, 2.0),
                np.clip(momentum, -1.0, 1.0),
                np.clip(drawdown, 0.0, 1.0),
                np.clip(cash_ratio, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
