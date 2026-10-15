"""Testes do score comportamental por regras (src/risk/behavior_score.py).

Cada regra é testada isoladamente com features montadas à mão — sem banco.
"""

import dataclasses

import pytest

from src.risk.behavior_score import (
    BehaviorConfig,
    behavior_score,
    ramp,
    rule_amount,
    rule_hour,
    rule_impossible_travel,
    rule_location,
    rule_new_device,
    rule_untrusted_device,
)

CFG = BehaviorConfig()


def features(**overrides):
    """Uma transação perfeitamente normal; cada teste altera só o que importa."""
    base = {
        "amount_deviation": 0.0,  # exatamente na média
        "distance_from_home_km": 4.0,  # compra no bairro
        "location_distance": 0.16,
        "unusual_hour": 0.0,
        "new_device": False,
        "new_city": False,
        "device_untrusted": False,
        "implied_speed_kmh": 3.0,
        "is_impossible_travel": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# rampa
# ---------------------------------------------------------------------------


def test_ramp_is_zero_up_to_start_then_base_then_max():
    assert ramp(1.0, 2.0, 6.0, 5.0, 25.0) == 0.0
    assert ramp(2.0, 2.0, 6.0, 5.0, 25.0) == 0.0  # start é inclusivo para o zero
    assert ramp(2.0001, 2.0, 6.0, 5.0, 25.0) == pytest.approx(5.0, abs=0.01)
    assert ramp(4.0, 2.0, 6.0, 5.0, 25.0) == pytest.approx(15.0)
    assert ramp(6.0, 2.0, 6.0, 5.0, 25.0) == pytest.approx(25.0)
    assert ramp(50.0, 2.0, 6.0, 5.0, 25.0) == pytest.approx(25.0)  # limitado ao máximo


# ---------------------------------------------------------------------------
# uma transação normal não dispara nada
# ---------------------------------------------------------------------------


def test_normal_transaction_scores_zero_without_reasons():
    result = behavior_score(features())
    assert result.score == 0.0
    assert result.hits == []


# ---------------------------------------------------------------------------
# regras isoladas
# ---------------------------------------------------------------------------


def test_location_rule():
    assert rule_location(features(distance_from_home_km=50.0), CFG) is None
    assert rule_location(features(distance_from_home_km=0.0), CFG) is None

    far = rule_location(features(distance_from_home_km=2727.4), CFG)
    assert far.points == pytest.approx(CFG.location_max_points)
    assert far.reason == "Localização incomum: 2.727 km da residência"

    mid = rule_location(features(distance_from_home_km=450.0), CFG)
    assert CFG.location_base_points < mid.points < CFG.location_max_points


def test_amount_rule_is_relative_to_the_customer_mean():
    assert rule_amount(features(amount_deviation=0.9), CFG) is None  # 1,9x: compra grande plausível
    assert rule_amount(features(amount_deviation=-0.6), CFG) is None  # bem abaixo da média não é risco

    hit = rule_amount(features(amount_deviation=7.94), CFG)  # 8,94x a média
    assert hit.points == pytest.approx(CFG.amount_max_points)
    assert hit.reason == "Valor 8,9x acima da média do cliente"

    moderate = rule_amount(features(amount_deviation=2.5), CFG)  # 3,5x
    assert CFG.amount_base_points < moderate.points < CFG.amount_max_points


def test_hour_rule():
    assert rule_hour(features(unusual_hour=0.0), CFG) is None

    hit = rule_hour(features(unusual_hour=2.5 / 12), CFG)  # 2,5 h fora
    assert CFG.hour_base_points < hit.points < CFG.hour_max_points
    assert hit.reason == "Horário fora da janela habitual (2,5 h fora)"

    far = rule_hour(features(unusual_hour=6 / 12), CFG)
    assert far.points == pytest.approx(CFG.hour_max_points)


def test_new_device_rule():
    assert rule_new_device(features(new_device=False), CFG) is None
    hit = rule_new_device(features(new_device=True), CFG)
    assert hit.points == CFG.new_device_points
    assert hit.reason == "Dispositivo nunca usado antes por este cliente"


def test_untrusted_device_rule():
    assert rule_untrusted_device(features(device_untrusted=False), CFG) is None
    hit = rule_untrusted_device(features(device_untrusted=True), CFG)
    assert hit.points == CFG.untrusted_device_points
    assert hit.reason == "Dispositivo não confiável"


def test_impossible_travel_rule():
    assert rule_impossible_travel(features(implied_speed_kmh=950.0, is_impossible_travel=False), CFG) is None
    hit = rule_impossible_travel(features(implied_speed_kmh=1900.4, is_impossible_travel=True), CFG)
    assert hit.points == CFG.impossible_travel_points
    assert hit.reason == "Viagem impossível: 1.900 km/h entre duas compras"


# ---------------------------------------------------------------------------
# soma, limite e ordenação
# ---------------------------------------------------------------------------


def test_score_is_the_sum_of_the_rules_that_fired():
    result = behavior_score(features(new_device=True, device_untrusted=True))
    assert result.score == pytest.approx(CFG.new_device_points + CFG.untrusted_device_points)
    assert {h.rule for h in result.hits} == {"dispositivo_novo", "dispositivo_nao_confiavel"}


def test_score_is_capped_at_100():
    everything = features(
        amount_deviation=14.0,
        distance_from_home_km=5000.0,
        unusual_hour=0.5,
        new_device=True,
        device_untrusted=True,
        implied_speed_kmh=3000.0,
        is_impossible_travel=True,
    )
    result = behavior_score(everything)
    assert sum(h.points for h in result.hits) > 100
    assert result.score == 100.0


def test_hits_are_sorted_by_points_descending():
    result = behavior_score(
        features(unusual_hour=1 / 12, distance_from_home_km=3000.0, implied_speed_kmh=2000.0, is_impossible_travel=True)
    )
    points = [h.points for h in result.hits]
    assert points == sorted(points, reverse=True)
    assert result.hits[0].rule == "viagem_impossivel"


def test_weights_come_only_from_the_config_object():
    """Mudar um peso na config muda o score — nenhum valor está fixo no código."""
    custom = dataclasses.replace(CFG, new_device_points=33.0, untrusted_device_points=0.0)
    result = behavior_score(features(new_device=True, device_untrusted=True), custom)
    assert result.score == pytest.approx(33.0)


def test_config_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CFG.new_device_points = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# regra de cidade nova
# ---------------------------------------------------------------------------


def test_new_city_rule():
    from src.risk.behavior_score import rule_new_city

    assert rule_new_city(features(new_city=False), CFG) is None

    hit = rule_new_city(features(new_city=True, merchant_city="Paris"), CFG)
    assert hit.rule == "cidade_nova"
    assert hit.points == CFG.new_city_points
    assert hit.reason == "Primeira compra do cliente nesta cidade (Paris)"


def test_new_city_rule_without_city_name_still_explains():
    from src.risk.behavior_score import rule_new_city

    hit = rule_new_city(features(new_city=True), CFG)
    assert hit.reason == "Primeira compra do cliente nesta cidade"


def test_far_purchase_in_a_new_city_outscores_a_trip_to_a_known_city():
    """A separação que motivou a regra: mesma distância, cidade inédita vs conhecida."""
    trip = features(distance_from_home_km=2500.0)
    anomaly = features(distance_from_home_km=2500.0, new_city=True)
    assert behavior_score(anomaly).score - behavior_score(trip).score == pytest.approx(CFG.new_city_points)
