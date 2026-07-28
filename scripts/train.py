#!/usr/bin/env python3
"""Trains one PPO or DQN agent and saves it to models/{algorithm}_{reward_mode}.zip,
plus its observation-normalization statistics to
models/{algorithm}_{reward_mode}_vecnormalize.pkl — both are required to
reproduce the trained policy's actual behavior; see finrl/agents/factory.py.

Usage:
  python scripts/train.py --algorithm ppo --reward-mode return_minus_drawdown
  python scripts/train.py --algorithm dqn --reward-mode risk_adjusted --timesteps 50000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finrl.agents.factory import build_agent  # noqa: E402
from finrl.logging_config import configure_logging, get_logger  # noqa: E402

DEFAULT_TIMESTEPS = 100_000


def train(algorithm: str, reward_mode: str, timesteps: int, seed: int, model_dir: Path) -> Path:
    configure_logging()
    logger = get_logger(__name__)

    logger.info(
        "training_started", algorithm=algorithm, reward_mode=reward_mode, timesteps=timesteps
    )
    start = time.monotonic()

    model, vec_env = build_agent(algorithm, reward_mode, seed=seed, verbose=0)  # type: ignore[arg-type]
    model.learn(total_timesteps=timesteps, progress_bar=False)

    elapsed_s = time.monotonic() - start
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{algorithm}_{reward_mode}.zip"
    vecnorm_path = model_dir / f"{algorithm}_{reward_mode}_vecnormalize.pkl"
    model.save(str(model_path))
    vec_env.save(str(vecnorm_path))

    logger.info(
        "training_completed",
        model_path=str(model_path),
        vecnorm_path=str(vecnorm_path),
        elapsed_s=round(elapsed_s, 1),
    )
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], required=True)
    parser.add_argument(
        "--reward-mode", choices=["return_minus_drawdown", "risk_adjusted"], required=True
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    train(args.algorithm, args.reward_mode, args.timesteps, args.seed, args.model_dir)


if __name__ == "__main__":
    main()
