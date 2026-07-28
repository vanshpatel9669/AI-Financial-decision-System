"""API tests run against the real committed flagship model (models/ppo_
return_minus_drawdown.zip) — not a mock, so a broken train/save/load/predict
pipeline shows up here rather than only in manual smoke testing.
"""

from fastapi.testclient import TestClient

from finrl.api.main import app

client = TestClient(app)


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_loads_the_real_model():
    response = client.get("/readyz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["algorithm"] == "ppo"
    assert body["reward_mode"] == "return_minus_drawdown"


def test_list_actions():
    response = client.get("/v1/actions")
    assert response.status_code == 200
    assert response.json() == {"defensive": 0, "balanced": 1, "growth": 2}


def test_predict_returns_a_valid_action():
    response = client.post(
        "/v1/predict",
        json={"volatility": 0.15, "momentum": 0.02, "drawdown": 0.05, "cash_ratio": 0.3},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["recommended_action"] in {"defensive", "balanced", "growth"}
    assert body["action_index"] in {0, 1, 2}
    assert len(body["policy_scores"]) == 3
    assert body["algorithm"] == "ppo"
    assert "disclaimer" in body


def test_predict_rejects_out_of_range_volatility():
    response = client.post(
        "/v1/predict",
        json={"volatility": -1.0, "momentum": 0.02, "drawdown": 0.05, "cash_ratio": 0.3},
    )
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    response = client.post(
        "/v1/predict",
        json={"volatility": 0.15, "momentum": 0.02, "drawdown": 0.05},
    )
    assert response.status_code == 422


def test_metrics_endpoint_exposed():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "python_info" in response.text or "process_" in response.text
