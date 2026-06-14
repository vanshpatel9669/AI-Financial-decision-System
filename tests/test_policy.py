from policy import LinearPolicy

def test_policy_returns_action():
    p = LinearPolicy()
    action = p.predict({"volatility":0.2,"momentum":0.1,"drawdown":0.05,"cash_ratio":0.2})
    assert action in {"defensive", "balanced", "growth"}
