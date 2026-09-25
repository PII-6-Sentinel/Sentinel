"""Testes das features comportamentais (src/synthetic/features.py e
feature_pipeline.py).

Funções puras são testadas com objetos simples (`SimpleNamespace`), sem
banco. O pipeline é testado sobre um SQLite `:memory:` populado pelo
gerador — nunca sobre data/synthetic/sentinel.db.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from src.synthetic.feature_pipeline import FEATURE_COLUMNS, TABLE_COLUMNS, build_feature_table
from src.synthetic.features import (
    MAX_REFERENCE_DISTANCE_KM,
    amount_deviation,
    implied_travel_speed,
    location_distance,
    new_device_flag,
    unusual_hour,
)
from src.synthetic.generator import MAX_PLAUSIBLE_SPEED_KMH, populate_database
from src.synthetic.models import Transaction

SAO_PAULO = (-23.5505, -46.6333)
RIO = (-22.9068, -43.1729)  # ~360 km de São Paulo


def customer(baseline=100.0, start=8, end=20):
    return SimpleNamespace(avg_amount_baseline=baseline, typical_hour_start=start, typical_hour_end=end)


def tx(amount=100.0, hour=12, minute=0, distance=0.0, device_id=1, latlon=SAO_PAULO, day=10):
    return SimpleNamespace(
        amount=amount,
        timestamp=datetime(2026, 8, day, hour, minute),
        distance_from_home_km=distance,
        device_id=device_id,
        lat=latlon[0],
        lon=latlon[1],
    )


# ---------------------------------------------------------------------------
# amount_deviation
# ---------------------------------------------------------------------------


def test_amount_deviation_is_zero_at_the_customer_mean():
    assert amount_deviation(tx(amount=100.0), customer(baseline=100.0)) == pytest.approx(0.0)


def test_amount_deviation_is_relative_to_each_customers_own_baseline():
    same_amount = tx(amount=3000.0)
    high_spender = amount_deviation(same_amount, customer(baseline=3000.0))
    fixed_profile = amount_deviation(same_amount, customer(baseline=100.0))

    assert high_spender == pytest.approx(0.0)
    assert fixed_profile == pytest.approx(29.0)  # 30x a média = +2900%


def test_amount_deviation_is_signed():
    assert amount_deviation(tx(amount=200.0), customer(baseline=100.0)) == pytest.approx(1.0)
    assert amount_deviation(tx(amount=50.0), customer(baseline=100.0)) == pytest.approx(-0.5)


@pytest.mark.parametrize("bad_baseline", [0.0, -10.0])
def test_amount_deviation_rejects_non_positive_baseline(bad_baseline):
    with pytest.raises(ValueError):
        amount_deviation(tx(), customer(baseline=bad_baseline))


# ---------------------------------------------------------------------------
# location_distance
# ---------------------------------------------------------------------------


def test_location_distance_bounds():
    assert location_distance(tx(distance=0.0)) == pytest.approx(0.0)
    assert location_distance(tx(distance=MAX_REFERENCE_DISTANCE_KM)) == pytest.approx(1.0)
    assert location_distance(tx(distance=50_000.0)) == 1.0  # acima da referência: limitado a 1


def test_location_distance_is_monotonic():
    values = [location_distance(tx(distance=d)) for d in (0, 1, 10, 100, 1000, 10000)]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_location_distance_log_scale_separates_short_distances():
    """Em escala log, 2 km vs 8 km (mesma cidade) pesa mais do que 2000 vs
    2006 km (mesma viagem longa) — em escala linear seria o contrário.
    """
    short_gap = location_distance(tx(distance=8.0)) - location_distance(tx(distance=2.0))
    long_gap = location_distance(tx(distance=2006.0)) - location_distance(tx(distance=2000.0))
    assert short_gap > 10 * long_gap


# ---------------------------------------------------------------------------
# unusual_hour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hour, minute", [(8, 0), (12, 30), (20, 0)])
def test_unusual_hour_is_zero_inside_the_window(hour, minute):
    assert unusual_hour(tx(hour=hour, minute=minute), customer(start=8, end=20)) == 0.0


def test_unusual_hour_grows_with_distance_from_the_window():
    c = customer(start=8, end=20)
    one_hour_out = unusual_hour(tx(hour=21), c)
    four_hours_out = unusual_hour(tx(hour=0), c)  # 0h está a 4h de 20h
    assert 0 < one_hour_out < four_hours_out
    assert one_hour_out == pytest.approx(1 / 12)
    assert four_hours_out == pytest.approx(4 / 12)


def test_unusual_hour_is_circular():
    """1h da manhã está a 2h de uma janela que termina às 23h — não a 22h."""
    c = customer(start=6, end=23)
    assert unusual_hour(tx(hour=1), c) == pytest.approx(2 / 12)


def test_unusual_hour_handles_windows_that_wrap_midnight():
    night_shift = customer(start=22, end=6)
    assert unusual_hour(tx(hour=23), night_shift) == 0.0
    assert unusual_hour(tx(hour=3), night_shift) == 0.0
    assert unusual_hour(tx(hour=12), night_shift) > 0.0


def test_unusual_hour_never_exceeds_one():
    c = customer(start=0, end=1)
    assert max(unusual_hour(tx(hour=h), c) for h in range(24)) <= 1.0


# ---------------------------------------------------------------------------
# new_device_flag
# ---------------------------------------------------------------------------


def test_new_device_flag():
    assert new_device_flag(tx(device_id=7), seen_device_ids={1, 2}) is True
    assert new_device_flag(tx(device_id=2), seen_device_ids={1, 2}) is False


def test_new_device_flag_with_no_history_is_true():
    assert new_device_flag(tx(device_id=1), seen_device_ids=set()) is True


# ---------------------------------------------------------------------------
# implied_travel_speed
# ---------------------------------------------------------------------------


def test_travel_speed_without_previous_transaction_is_explicit_none():
    result = implied_travel_speed(tx(), None)
    assert result.speed_kmh is None
    assert result.is_impossible_travel is False


def test_travel_speed_plausible_trip():
    earlier = tx(hour=8, latlon=SAO_PAULO)
    later = tx(hour=14, latlon=RIO)  # ~360 km em 6h = ~60 km/h
    result = implied_travel_speed(later, earlier)

    assert result.speed_kmh == pytest.approx(60, rel=0.1)
    assert result.is_impossible_travel is False


def test_travel_speed_impossible_trip():
    earlier = tx(hour=12, minute=0, latlon=SAO_PAULO)
    later = tx(hour=12, minute=10, latlon=RIO)  # ~360 km em 10 min = ~2.100 km/h
    result = implied_travel_speed(later, earlier)

    assert result.speed_kmh > MAX_PLAUSIBLE_SPEED_KMH
    assert result.is_impossible_travel is True


def test_travel_speed_same_place_is_zero():
    result = implied_travel_speed(tx(hour=13), tx(hour=12))
    assert result.speed_kmh == pytest.approx(0.0, abs=1e-6)
    assert result.is_impossible_travel is False


def test_travel_speed_same_minute_does_not_divide_by_zero():
    earlier = tx(hour=12, minute=0, latlon=SAO_PAULO)
    same_minute_far = tx(hour=12, minute=0, latlon=RIO)
    same_minute_same_city = tx(hour=12, minute=0, latlon=SAO_PAULO)

    far = implied_travel_speed(same_minute_far, earlier)
    near = implied_travel_speed(same_minute_same_city, earlier)

    assert far.speed_kmh is not None and far.speed_kmh != float("inf")
    assert far.is_impossible_travel is True
    assert near.is_impossible_travel is False


def test_travel_speed_rejects_out_of_order_transactions():
    earlier = tx(hour=12)
    later = tx(hour=12)
    later.timestamp = earlier.timestamp + timedelta(hours=1)
    with pytest.raises(ValueError):
        implied_travel_speed(earlier, later)  # "anterior" é posterior à atual


# ---------------------------------------------------------------------------
# build_feature_table — sobre um SQLite em memória populado pelo gerador
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def populated():
    engine = create_engine("sqlite:///:memory:")
    summary = populate_database(engine, seed=42, n_customers=130)
    with Session(engine) as session:
        table = build_feature_table(session)
    return engine, summary, table


def test_table_has_one_row_per_transaction_and_expected_columns(populated):
    _, summary, table = populated
    assert list(table.columns) == TABLE_COLUMNS
    assert len(table) == summary["n_transactions"]
    assert table["transaction_id"].is_unique


def test_no_unexpected_nan_in_feature_columns(populated):
    _, _, table = populated
    assert not table[FEATURE_COLUMNS].isna().any().any()
    assert not table[["transaction_id", "is_fraud"]].isna().any().any()


def test_fraud_scenario_is_empty_only_for_normal_transactions(populated):
    _, _, table = populated
    assert table.loc[~table["is_fraud"], "fraud_scenario"].isna().all()
    assert table.loc[table["is_fraud"], "fraud_scenario"].notna().all()


def test_first_transaction_of_each_customer_has_explicit_zero_speed(populated):
    engine, _, table = populated
    with Session(engine) as session:
        ordered = session.execute(
            select(Transaction.id, Transaction.customer_id).order_by(
                Transaction.customer_id, Transaction.timestamp, Transaction.id
            )
        ).all()

    first_ids, seen = [], set()
    for transaction_id, customer_id in ordered:
        if customer_id not in seen:
            seen.add(customer_id)
            first_ids.append(transaction_id)

    firsts = table[table["transaction_id"].isin(first_ids)]
    assert len(firsts) == len(seen)
    assert (firsts["implied_speed_kmh"] == 0.0).all()
    assert not firsts["is_impossible_travel"].any()
    assert firsts["new_device"].all()  # sem histórico, todo dispositivo é "novo"


def test_impossible_travel_is_true_for_nearly_all_viagem_impossivel_transactions(populated):
    _, _, table = populated
    viagem = table[table["fraud_scenario"] == "viagem_impossivel"]

    assert len(viagem) > 0
    assert viagem["is_impossible_travel"].mean() >= 0.95
    assert (viagem.loc[viagem["is_impossible_travel"], "implied_speed_kmh"] > MAX_PLAUSIBLE_SPEED_KMH).all()


def test_impossible_travel_is_rare_among_normal_transactions(populated):
    _, _, table = populated
    normal_rate = table.loc[~table["is_fraud"], "is_impossible_travel"].mean()
    viagem_rate = table.loc[table["fraud_scenario"] == "viagem_impossivel", "is_impossible_travel"].mean()
    assert normal_rate < 0.05
    assert viagem_rate > 10 * normal_rate


def test_feature_ranges(populated):
    _, _, table = populated
    assert table["location_distance"].between(0.0, 1.0).all()
    assert table["unusual_hour"].between(0.0, 1.0).all()
    assert (table["implied_speed_kmh"] >= 0).all()


def test_features_reflect_the_scenario_that_generated_each_fraud(populated):
    """Cada cenário deixa a assinatura esperada na feature correspondente."""
    _, _, table = populated
    normal = table[~table["is_fraud"]]

    valor = table[table["fraud_scenario"] == "valor_atipico"]
    assert (valor["amount_deviation"] >= 5.0).all()  # gerado com 6x-15x a média
    assert normal["amount_deviation"].abs().mean() < 1.0

    horario = table[table["fraud_scenario"] == "horario_atipico"]
    assert (horario["unusual_hour"] > 0).all()

    geo = table[table["fraud_scenario"] == "anomalia_geografica"]
    assert geo["location_distance"].mean() > 2 * normal["location_distance"].median()
