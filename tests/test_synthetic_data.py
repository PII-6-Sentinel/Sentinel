"""Testes do gerador de dados sintéticos (src/synthetic/).

Nunca tocam em data/synthetic/sentinel.db — cada teste gera seu próprio
dataset em memória (`generate_dataset`) ou usa um SQLite `:memory:`
(`populate_database`), então a suíte roda igual com ou sem o banco real
já gerado na máquina.
"""

import random
from collections import defaultdict

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.synthetic.generator import (
    DEFAULT_N_CUSTOMERS,
    HISTORY_DAYS,
    MAX_PLAUSIBLE_SPEED_KMH,
    MIN_GAP_DAYS_BETWEEN_STAYS,
    build_itinerary,
    generate_dataset,
    populate_database,
)
from src.synthetic.geo_reference import CITIES, distance_km
from src.synthetic.models import BehaviorProfile, Customer, FraudScenario, Transaction

# ---------------------------------------------------------------------------
# Volume — usa os parâmetros DEFAULT (os mesmos do script), porque o
# requisito é sobre o dataset "de verdade" que scripts/generate_mock_data.py
# produz, não sobre um recorte pequeno de teste.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def default_dataset():
    return generate_dataset()  # seed e n_customers default (42, 130)


def test_customer_count_within_expected_range(default_dataset):
    assert 100 <= len(default_dataset.customers) <= 150
    assert len(default_dataset.customers) == DEFAULT_N_CUSTOMERS


def test_transaction_count_within_expected_range(default_dataset):
    assert 5_000 <= len(default_dataset.transactions) <= 10_000


def test_fraud_rate_within_expected_range(default_dataset):
    fraud = [t for t in default_dataset.transactions if t.is_fraud]
    rate = len(fraud) / len(default_dataset.transactions)
    # meta é ~3%; faixa com folga para não quebrar por arredondamento entre
    # os 6 cenários (ver _target_fraud_counts).
    assert 0.02 <= rate <= 0.04


def test_every_fraud_transaction_has_a_scenario(default_dataset):
    valid_scenarios = {s.value for s in FraudScenario}
    for t in default_dataset.transactions:
        if t.is_fraud:
            assert t.fraud_scenario in valid_scenarios
        else:
            assert t.fraud_scenario is None


def test_all_six_scenarios_are_represented_and_roughly_balanced(default_dataset):
    fraud = [t for t in default_dataset.transactions if t.is_fraud]
    counts = {s.value: sum(1 for t in fraud if t.fraud_scenario == s.value) for s in FraudScenario}

    assert len(FraudScenario) == 6
    assert all(c > 0 for c in counts.values())
    even_share = len(fraud) / len(FraudScenario)
    assert all(abs(c - even_share) <= 0.2 * even_share for c in counts.values()), counts


# ---------------------------------------------------------------------------
# Sem distância negativa
# ---------------------------------------------------------------------------


def test_no_negative_distance_from_home(default_dataset):
    for t in default_dataset.transactions:
        assert t.distance_from_home_km >= 0.0


# ---------------------------------------------------------------------------
# "Viagem impossível" viola de fato uma velocidade máxima plausível
# ---------------------------------------------------------------------------


def _speed_kmh(prev: Transaction, curr: Transaction) -> float:
    dist = distance_km(prev.lat, prev.lon, curr.lat, curr.lon)
    delta_hours = (curr.timestamp - prev.timestamp).total_seconds() / 3600.0
    assert delta_hours > 0, "timestamps fora de ordem — quebra a premissa do teste"
    return dist / delta_hours


def test_impossible_travel_scenario_violates_max_plausible_speed(default_dataset):
    by_customer = defaultdict(list)
    for t in default_dataset.transactions:
        by_customer[id(t.customer)].append(t)

    checked = 0
    for txs in by_customer.values():
        txs.sort(key=lambda t: t.timestamp)
        for prev, curr in zip(txs, txs[1:]):
            if curr.fraud_scenario == FraudScenario.VIAGEM_IMPOSSIVEL.value:
                speed = _speed_kmh(prev, curr)
                assert speed > MAX_PLAUSIBLE_SPEED_KMH, (
                    f"Transação 'viagem_impossivel' implica {speed:.0f} km/h, "
                    f"esperado > {MAX_PLAUSIBLE_SPEED_KMH} km/h"
                )
                checked += 1

    assert checked > 0, "nenhuma transação de viagem_impossivel encontrada para validar"


# ---------------------------------------------------------------------------
# Determinismo — mesma seed, mesmo resultado
# ---------------------------------------------------------------------------


def test_generation_is_deterministic():
    d1 = generate_dataset(seed=42, n_customers=30)
    d2 = generate_dataset(seed=42, n_customers=30)

    assert [c.name for c in d1.customers] == [c.name for c in d2.customers]
    assert [c.behavior_profile for c in d1.customers] == [c.behavior_profile for c in d2.customers]
    assert len(d1.transactions) == len(d2.transactions)
    assert [round(t.amount, 2) for t in d1.transactions] == [round(t.amount, 2) for t in d2.transactions]
    assert [t.timestamp for t in d1.transactions] == [t.timestamp for t in d2.transactions]


def test_different_seeds_produce_different_datasets():
    d1 = generate_dataset(seed=42, n_customers=30)
    d2 = generate_dataset(seed=7, n_customers=30)
    assert [c.name for c in d1.customers] != [c.name for c in d2.customers]


# ---------------------------------------------------------------------------
# Persistência (SQLite em memória — nunca o arquivo real)
# ---------------------------------------------------------------------------


def test_populate_database_persists_and_is_queryable():
    engine = create_engine("sqlite:///:memory:")
    summary = populate_database(engine, seed=42, n_customers=30)

    with Session(engine) as session:
        n_customers_db = session.query(Customer).count()
        n_tx_db = session.query(Transaction).count()
        n_fraud_db = session.query(Transaction).filter(Transaction.is_fraud.is_(True)).count()

    assert n_customers_db == 30
    assert n_tx_db == summary["n_transactions"]
    assert n_fraud_db == summary["n_fraud"]


def test_summary_has_expected_shape():
    dataset = generate_dataset(seed=42, n_customers=30)
    from src.synthetic.generator import summarize

    summary = summarize(dataset)
    assert set(summary["profile_counts"].keys()) <= {
        "fixo", "viajante", "alto_gasto", "novo_cliente",
    }
    assert summary["n_customers"] == 30
    assert 1 <= len(summary["example_frauds"]) <= len(FraudScenario)
    for ex in summary["example_frauds"]:
        assert ex["scenario"] in {s.value for s in FraudScenario}


# ---------------------------------------------------------------------------
# Viajantes: itinerário em blocos contínuos, sem "teletransporte"
# ---------------------------------------------------------------------------


def _traveler_setup(seed=1):
    rng = random.Random(seed)
    home = CITIES[0]
    others = [c for c in CITIES if c.city != home.city]
    return rng, home, rng.sample(others, k=3)


def test_itinerary_stays_are_contiguous_blocks_separated_by_a_gap():
    rng, home, travel = _traveler_setup()
    stays = build_itinerary(rng, home, travel)

    assert stays[0].first_day == 0
    assert stays[0].city == home
    for prev, nxt in zip(stays, stays[1:]):
        assert prev.first_day <= prev.last_day
        # exatamente MIN_GAP_DAYS dias inteiros vazios entre uma estadia e a próxima
        assert nxt.first_day == prev.last_day + 1 + MIN_GAP_DAYS_BETWEEN_STAYS
        assert nxt.city.city != prev.city.city  # "viajar" = mudar de cidade
    assert stays[-1].last_day <= HISTORY_DAYS - 1


def test_itinerary_only_visits_known_cities_and_stays_days_to_weeks():
    for seed in range(20):
        rng, home, travel = _traveler_setup(seed)
        known = {home.city} | {c.city for c in travel}
        stays = build_itinerary(rng, home, travel)

        assert {s.city.city for s in stays} <= known
        for s in stays[:-1]:  # a última pode ser truncada pelo fim do histórico
            length = s.last_day - s.first_day + 1
            bounds = (7, 28) if s.city.city == home.city else (3, 14)
            assert bounds[0] <= length <= bounds[1]


def test_itinerary_is_deterministic():
    rng1, home, travel = _traveler_setup(5)
    rng2, _, _ = _traveler_setup(5)
    assert build_itinerary(rng1, home, travel) == build_itinerary(rng2, home, travel)


def _legit_streams(dataset, profile):
    by_customer = defaultdict(list)
    for t in dataset.transactions:
        if not t.is_fraud and t.customer.behavior_profile == profile:
            by_customer[id(t.customer)].append(t)
    for txs in by_customer.values():
        txs.sort(key=lambda t: t.timestamp)
    return list(by_customer.values())


def test_legitimate_traveler_never_teleports_between_cities(default_dataset):
    """Nenhum par de transações NORMAIS consecutivas de um viajante implica
    uma velocidade acima do limiar — era ~5,9% dos pares antes do itinerário.
    """
    streams = _legit_streams(default_dataset, BehaviorProfile.VIAJANTE)
    assert streams

    pairs = 0
    for txs in streams:
        for prev, curr in zip(txs, txs[1:]):
            assert _speed_kmh(prev, curr) <= MAX_PLAUSIBLE_SPEED_KMH
            pairs += 1
    assert pairs > 500


def test_travelers_actually_change_cities(default_dataset):
    """A correção não pode ter "resolvido" o teletransporte deixando os
    viajantes parados em casa: a maioria precisa visitar >1 cidade.
    """
    streams = _legit_streams(default_dataset, BehaviorProfile.VIAJANTE)
    multi_city = sum(1 for txs in streams if len({t.merchant.city for t in txs}) > 1)
    assert multi_city / len(streams) >= 0.8


def test_traveler_city_changes_leave_a_realistic_time_gap(default_dataset):
    for txs in _legit_streams(default_dataset, BehaviorProfile.VIAJANTE):
        for prev, curr in zip(txs, txs[1:]):
            if prev.merchant.city != curr.merchant.city:
                assert (curr.timestamp - prev.timestamp).total_seconds() >= 24 * 3600


def test_normal_transactions_never_share_a_minute_within_a_customer(default_dataset):
    seen = defaultdict(set)
    for t in default_dataset.transactions:
        if not t.is_fraud:
            key = id(t.customer)
            assert t.timestamp not in seen[key]
            seen[key].add(t.timestamp)


# ---------------------------------------------------------------------------
# Cenário 6: dispositivo_novo
# ---------------------------------------------------------------------------


def _new_device_frauds(dataset):
    return [t for t in dataset.transactions if t.fraud_scenario == FraudScenario.DISPOSITIVO_NOVO.value]


def test_new_device_scenario_uses_an_untrusted_device(default_dataset):
    frauds = _new_device_frauds(default_dataset)
    assert frauds
    assert all(t.device.is_trusted is False for t in frauds)


def test_new_device_scenario_device_never_appears_anywhere_else(default_dataset):
    """O dispositivo aparece em UMA transação só — a própria fraude — e,
    portanto, nunca antes no histórico do cliente.
    """
    usage = defaultdict(int)
    for t in default_dataset.transactions:
        usage[id(t.device)] += 1

    for t in _new_device_frauds(default_dataset):
        assert usage[id(t.device)] == 1
        assert t.device in default_dataset.devices
        assert t.device.customer is t.customer
        assert t.device.first_seen_at == t.timestamp


def test_other_scenarios_and_normal_history_never_use_the_new_devices(default_dataset):
    new_devices = {id(t.device) for t in _new_device_frauds(default_dataset)}
    for t in default_dataset.transactions:
        if t.fraud_scenario != FraudScenario.DISPOSITIVO_NOVO.value:
            assert id(t.device) not in new_devices


def test_new_device_scenario_keeps_city_amount_and_hour_normal(default_dataset):
    """Só o dispositivo destoa — o resto tem que parecer normal."""
    for t in _new_device_frauds(default_dataset):
        c = t.customer
        assert c.typical_hour_start <= t.timestamp.hour <= c.typical_hour_end
        assert t.amount < 6 * c.avg_amount_baseline
