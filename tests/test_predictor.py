from energy_monitor.predictor import advisory_level


LEVELS = {"watch": 0.8, "moderate": 0.9, "high": 1.0}


def test_advisory_boundaries():
    assert advisory_level(79, 100, LEVELS)[0] == "NORMAL"
    assert advisory_level(80, 100, LEVELS)[0] == "WATCH"
    assert advisory_level(90, 100, LEVELS)[0] == "MODERATE"
    assert advisory_level(100, 100, LEVELS)[0] == "HIGH"
