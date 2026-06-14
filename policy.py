import json
from pathlib import Path
import numpy as np
from environment import ACTIONS

FEATURES = ["volatility", "momentum", "drawdown", "cash_ratio"]
MODEL_PATH = Path("outputs/policy.json")

class LinearPolicy:
    def __init__(self, weights=None):
        self.weights = np.array(weights if weights is not None else np.zeros((len(ACTIONS), len(FEATURES) + 1)), dtype=float)

    def _x(self, state):
        return np.array([1.0] + [float(state[f]) for f in FEATURES])

    def predict_scores(self, state):
        return self.weights @ self._x(state)

    def predict(self, state):
        idx = int(np.argmax(self.predict_scores(state)))
        return ACTIONS[idx]

    def save(self):
        MODEL_PATH.parent.mkdir(exist_ok=True)
        MODEL_PATH.write_text(json.dumps({"weights": self.weights.tolist()}), encoding="utf-8")

    @classmethod
    def load(cls):
        payload = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        return cls(payload["weights"])
