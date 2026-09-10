"""Testes do gerador de dados sintéticos (src/synthetic/).

Nunca tocam em data/synthetic/sentinel.db — cada teste gera seu próprio
dataset em memória (`generate_dataset`) ou usa um SQLite `:memory:`
(`populate_database`), então a suíte roda igual com ou sem o banco real
já gerado na máquina.
"""

from collections import defaultdict

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.synthetic.generator import (
    DEFAULT_N_CUSTOMERS,
    MAX_PLAUSIBLE_SPEED_KMH,
    generate_dataset,
    populate_database,
)
from src.synthetic.geo_reference import distance_km
from src.synthetic.models import Customer, FraudScenario, Transaction

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
    # os 5 cenários (ver _target_fraud_counts).
    assert 0.02 <= rate <= 0.04


def test_every_fraud_transaction_has_a_scenario(default_dataset):
    valid_scenarios = {s.value for s in FraudScenario}
    for t in default_dataset.transactions:
        if t.is_fraud:
            assert t.fraud_scenario in valid_scenarios
        else:
            assert t.fraud_scenario is None


def test_all_five_scenarios_are_represented(default_dataset):
    scenarios_present = {t.fraud_scenario for t in default_dataset.transactions if t.is_fraud}
    assert scenarios_present == {s.value for s in FraudScenario}


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
    assert 1 <= len(summary["example_frauds"]) <= 5
    for ex in summary["example_frauds"]:
        assert ex["scenario"] in {s.value for s in FraudScenario}
