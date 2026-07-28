from finrl.env.market_simulator import MarketSimulator, Regime


def test_same_seed_is_reproducible():
    a = MarketSimulator(seed=42)
    b = MarketSimulator(seed=42)
    returns_a = [a.step() for _ in range(50)]
    returns_b = [b.step() for _ in range(50)]
    assert returns_a == returns_b


def test_different_seeds_diverge():
    a = MarketSimulator(seed=1)
    b = MarketSimulator(seed=2)
    returns_a = [a.step() for _ in range(50)]
    returns_b = [b.step() for _ in range(50)]
    assert returns_a != returns_b


def test_regime_stays_within_valid_range():
    sim = MarketSimulator(seed=7)
    for _ in range(200):
        sim.step()
        assert sim.regime in (Regime.CALM_BULL, Regime.CHOPPY, Regime.CRASH)


def test_reset_restores_initial_state():
    sim = MarketSimulator(seed=3)
    for _ in range(10):
        sim.step()
    sim.reset(seed=3)
    assert sim.price == 100.0
    assert sim.regime == Regime.CALM_BULL


def test_trailing_returns_length_and_consistency():
    sim = MarketSimulator(seed=5)
    for _ in range(30):
        sim.step()
    trailing = sim.trailing_returns(10)
    assert len(trailing) == 10

    # Empty history (only the seed price) yields no trailing returns.
    empty_sim = MarketSimulator(seed=5)
    assert len(empty_sim.trailing_returns(10)) == 0
