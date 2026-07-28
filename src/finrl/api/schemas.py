"""API request/response schemas — real Pydantic validation, replacing the
original system's `request.get_json(force=True)` (accepted any JSON,
including missing/malformed fields, and would raise an unhandled KeyError
rather than a clean 422).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MarketStateRequest(BaseModel):
    volatility: float = Field(ge=0.0, le=2.0, description="Annualized trailing volatility")
    momentum: float = Field(ge=-1.0, le=1.0, description="Trailing cumulative return")
    drawdown: float = Field(ge=0.0, le=1.0, description="Current drawdown from peak")
    cash_ratio: float = Field(ge=0.0, le=1.0, description="Current fraction held in cash")


class PredictionResponse(BaseModel):
    recommended_action: str
    action_index: int
    policy_scores: list[float] = Field(
        description="Raw policy network output per action (not probabilities for PPO/DQN "
        "in general — see docs/architecture.md)."
    )
    algorithm: str
    reward_mode: str
    disclaimer: str = "Synthetic-data research demo. Not financial advice."
