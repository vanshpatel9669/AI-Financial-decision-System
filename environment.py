import numpy as np

ACTIONS = ["defensive", "balanced", "growth"]

class MarketScenarioEnvironment:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def sample_state(self):
        return {
            "volatility": float(self.rng.uniform(0.05, 0.55)),
            "momentum": float(self.rng.uniform(-0.20, 0.25)),
            "drawdown": float(self.rng.uniform(0.00, 0.35)),
            "cash_ratio": float(self.rng.uniform(0.05, 0.60)),
        }

    def reward(self, state, action_index):
        risk_penalty = state["volatility"] + state["drawdown"]
        momentum = state["momentum"]
        if ACTIONS[action_index] == "defensive":
            return 0.04 - risk_penalty * 0.18 + state["cash_ratio"] * 0.05
        if ACTIONS[action_index] == "balanced":
            return 0.06 + momentum * 0.35 - risk_penalty * 0.10
        return 0.08 + momentum * 0.65 - risk_penalty * 0.22
