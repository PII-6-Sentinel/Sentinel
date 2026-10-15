"""Testes do Risk Engine (src/risk/engine.py)."""

import dataclasses

import numpy as np
import pytest

from src.risk.engine import (
    ACTION_APPROVE,
    ACTION_BLOCK,
    ACTION_REVIEW,
    LEVEL_HIGH,
    LEVEL_LOW,
    LEVEL_MEDIUM,
    DecisionBands,
    EngineConfig,
    RiskEngine,
    combine_scores,
    train_engine,
)
from src.risk.evaluation import split_by_customer
from src.risk.ml_score import MLScorer


@pytest.fixture(scope="module")
def split(risk_table):
    return split_by_customer(risk_table)


@pytest.fixture(scope="module")
def engine(split):
    return train_engine(split.train)


def normal_features(**overrides):
    base = {
        "amount_deviation": 0.0,
        "distance_from_home_km": 4.0,
        "location_distance": 0.16,
        "unusual_hour": 0.0,
        "new_device": False,
        "device_untrusted": False,
        "implied_speed_kmh": 3.0,
        "is_impossible_travel": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# faixas de decisão — fronteiras
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score, level",
    [(0, LEVEL_LOW), (30, LEVEL_LOW), (31, LEVEL_MEDIUM), (70, LEVEL_MEDIUM), (71, LEVEL_HIGH), (100, LEVEL_HIGH)],
)
def test_default_band_boundaries(score, level):
    assert DecisionBands().level(score) == level


def test_actions_follow_levels():
    from src.risk.engine import ACTION_BY_LEVEL

    assert ACTION_BY_LEVEL == {LEVEL_LOW: ACTION_APPROVE, LEVEL_MEDIUM: ACTION_REVIEW, LEVEL_HIGH: ACTION_BLOCK}


def test_custom_bands_boundaries():
    bands = DecisionBands(review_threshold=54, block_threshold=62)
    assert bands.level(54) == LEVEL_LOW
    assert bands.level(55) == LEVEL_MEDIUM
    assert bands.level(62) == LEVEL_MEDIUM
    assert bands.level(63) == LEVEL_HIGH
    assert bands.min_score(LEVEL_MEDIUM) == 55 and bands.min_score(LEVEL_HIGH) == 63


@pytest.mark.parametrize("review, block", [(70, 30), (50, 50), (-1, 50), (30, 100)])
def test_invalid_bands_are_rejected(review, block):
    with pytest.raises(ValueError):
        DecisionBands(review_threshold=review, block_threshold=block)


# ---------------------------------------------------------------------------
# combinação e piso
# ---------------------------------------------------------------------------


def test_combination_is_the_weighted_average_rounded_half_up():
    cfg = EngineConfig()  # 50/50
    assert combine_scores([40.0], [21.0], [False], cfg).tolist() == [31]  # 30,5 -> 31
    assert combine_scores([60.0], [0.0], [False], cfg).tolist() == [30]
    cfg_ml = EngineConfig(ml_weight=0.8, behavior_weight=0.2)
    assert combine_scores([100.0], [0.0], [False], cfg_ml).tolist() == [80]


def test_impossible_travel_floor_forces_at_least_medium():
    cfg = EngineConfig()
    floored = combine_scores([0.0], [0.0], [True], cfg)[0]
    assert floored == cfg.bands.min_score(LEVEL_MEDIUM) == 31
    assert cfg.bands.level(int(floored)) == LEVEL_MEDIUM


def test_floor_never_lowers_a_score():
    cfg = EngineConfig()
    assert combine_scores([100.0], [100.0], [True], cfg)[0] == 100


def test_floor_can_be_disabled_or_raised_to_high():
    assert combine_scores([0.0], [0.0], [True], EngineConfig(impossible_travel_floor=None))[0] == 0
    assert combine_scores([0.0], [0.0], [True], EngineConfig(impossible_travel_floor=LEVEL_HIGH))[0] == 71


@pytest.mark.parametrize(
    "kwargs",
    [{"ml_weight": 0.6, "behavior_weight": 0.6}, {"ml_weight": -0.5, "behavior_weight": 1.5}, {"impossible_travel_floor": "baixo"}],
)
def test_invalid_engine_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        EngineConfig(**kwargs)


def test_engine_requires_a_trained_model():
    with pytest.raises(ValueError):
        RiskEngine(MLScorer())


# ---------------------------------------------------------------------------
# engine treinado
# ---------------------------------------------------------------------------


def test_engine_is_deterministic(split):
    row = split.test[split.test["is_fraud"]].iloc[0].to_dict()
    a = train_engine(split.train).assess(row)
    b = train_engine(split.train).assess(row)
    assert a == b


def test_same_input_same_output(engine):
    f = normal_features(amount_deviation=7.0)
    assert engine.assess(f) == engine.assess(f)


def test_result_fields_are_consistent(engine):
    result = engine.assess(normal_features(new_device=True, device_untrusted=True))
    assert 0 <= result.risk_score <= 100
    assert 0 <= result.ml_score <= 100 and 0 <= result.behavior_score <= 100
    assert result.level == engine.config.bands.level(result.risk_score)
    assert result.reasons  # dispositivo novo e não confiável sempre geram motivo


def test_normal_transaction_is_approved_without_reasons(engine):
    result = engine.assess(normal_features())
    assert result.level == LEVEL_LOW
    assert result.action == ACTION_APPROVE
    assert result.reasons == []


def test_impossible_travel_is_never_approved(engine):
    """Mesmo com todos os outros sinais normais."""
    result = engine.assess(normal_features(implied_speed_kmh=1500.0, is_impossible_travel=True))
    assert result.level in (LEVEL_MEDIUM, LEVEL_HIGH)
    assert any("Viagem impossível" in r for r in result.reasons)


def test_reasons_are_ordered_by_contribution(engine):
    result = engine.assess(
        normal_features(distance_from_home_km=2500.0, location_distance=0.79, unusual_hour=1 / 12)
    )
    # localização (até 25 pts) pesa mais que horário (~7 pts)
    loc = next(i for i, r in enumerate(result.reasons) if r.startswith("Localização"))
    hour = next(i for i, r in enumerate(result.reasons) if r.startswith("Horário"))
    assert loc < hour


def test_assess_and_assess_frame_agree(engine, split):
    sample = split.test.sample(200, random_state=0)
    frame = engine.assess_frame(sample)
    for row, (_, out) in zip(sample.to_dict("records"), frame.iterrows()):
        single = engine.assess(row)
        assert single.risk_score == out["risk_score"]
        assert single.level == out["level"]
        assert single.action == out["action"]


def test_with_bands_keeps_model_and_weights(engine):
    other = engine.with_bands(DecisionBands(50, 60))
    assert other.ml_model is engine.ml_model
    assert other.config.ml_weight == engine.config.ml_weight
    assert other.config.bands == DecisionBands(50, 60)


def test_bands_change_only_the_decision_not_the_score(engine, split):
    sample = split.test.head(300)
    a = engine.assess_frame(sample)
    b = engine.with_bands(DecisionBands(10, 20)).assess_frame(sample)
    # o piso de viagem impossível depende das faixas; fora dele o score é idêntico
    mask = ~sample["is_impossible_travel"].to_numpy()
    assert np.array_equal(a["risk_score"].to_numpy()[mask], b["risk_score"].to_numpy()[mask])


def test_config_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        EngineConfig().ml_weight = 1.0  # type: ignore[misc]
