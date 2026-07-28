"""Synthetic multi-regime market simulator.

Not real market data — clearly synthetic, and documented as such everywhere
this is referenced. What makes it a genuine simulation rather than the
previous system's stateless per-call reward lookup: price evolves over a
sequence of steps via geometric Brownian motion, and which regime governs
that GBM's drift/volatility switches over time via a Markov transition
matrix. An allocation decision made now affects exposure to a regime shift
that hasn't happened yet — a real sequential-decision structure, not an
i.i.d. contextual bandit dressed up as one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Regime(IntEnum):
    CALM_BULL = 0
    CHOPPY = 1
    CRASH = 2


@dataclass(frozen=True)
class RegimeParams:
    """Annualized drift/volatility for the risky asset under this regime."""

    drift: float
    volatility: float


REGIME_PARAMS: dict[Regime, RegimeParams] = {
    Regime.CALM_BULL: RegimeParams(drift=0.12, volatility=0.10),
    Regime.CHOPPY: RegimeParams(drift=0.02, volatility=0.22),
    Regime.CRASH: RegimeParams(drift=-0.35, volatility=0.45),
}

# Row i = probability of transitioning FROM regime i TO each regime next step.
# Regimes are sticky (diagonal dominant) — real markets don't flip wildly
# step to step — with crash being both rare to enter and quick to exit.
_TRANSITION_MATRIX = np.array(
    [
        [0.94, 0.05, 0.01],  # from CALM_BULL
        [0.08, 0.85, 0.07],  # from CHOPPY
        [0.05, 0.35, 0.60],  # from CRASH
    ]
)

_TRADING_PERIODS_PER_YEAR = 252
_DT = 1.0 / _TRADING_PERIODS_PER_YEAR


class MarketSimulator:
    """Steps a regime-switching GBM price process one period at a time."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self.regime = Regime.CALM_BULL
        self.price = 100.0
        self._price_history: list[float] = [self.price]

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.regime = Regime.CALM_BULL
        self.price = 100.0
        self._price_history = [self.price]

    def step(self) -> float:
        """Advances one trading period. Returns the risky asset's simple
        return over that period (e.g. 0.01 == +1%).
        """
        self.regime = Regime(self._rng.choice(len(Regime), p=_TRANSITION_MATRIX[self.regime]))
        params = REGIME_PARAMS[self.regime]
        shock = self._rng.normal()
        drift_term = (params.drift - 0.5 * params.volatility**2) * _DT
        diffusion_term = params.volatility * np.sqrt(_DT) * shock
        log_return = drift_term + diffusion_term
        new_price = self.price * np.exp(log_return)
        simple_return = (new_price - self.price) / self.price
        self.price = new_price
        self._price_history.append(new_price)
        return float(simple_return)

    def trailing_returns(self, window: int) -> np.ndarray:
        prices = np.array(self._price_history[-(window + 1) :])
        if len(prices) < 2:
            return np.array([])
        return prices[1:] / prices[:-1] - 1.0
