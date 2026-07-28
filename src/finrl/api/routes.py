"""HTTP routes — thin, all logic lives in dependencies.LoadedPolicy."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from finrl.api.dependencies import LoadedPolicy, get_policy
from finrl.api.schemas import MarketStateRequest, PredictionResponse
from finrl.env.portfolio_env import Action
from finrl.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)

PolicyDep = Annotated[LoadedPolicy, Depends(get_policy)]


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(policy: PolicyDep) -> dict:
    return {"status": "ready", "algorithm": policy.algorithm, "reward_mode": policy.reward_mode}


@router.post("/v1/predict", response_model=PredictionResponse)
async def predict(payload: MarketStateRequest, policy: PolicyDep) -> PredictionResponse:
    try:
        action, scores = policy.predict(payload.model_dump())
    except Exception as exc:  # noqa: BLE001 — surface as a clean 500, not a raw traceback
        logger.error("prediction_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    logger.info(
        "prediction_served",
        recommended_action=action.name.lower(),
        algorithm=policy.algorithm,
        reward_mode=policy.reward_mode,
    )
    return PredictionResponse(
        recommended_action=action.name.lower(),
        action_index=int(action),
        policy_scores=scores,
        algorithm=policy.algorithm,
        reward_mode=policy.reward_mode,
    )


@router.get("/v1/actions")
async def list_actions() -> dict:
    return {a.name.lower(): int(a) for a in Action}
