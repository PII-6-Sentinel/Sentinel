"""Testes das features comportamentais (src/synthetic/features.py e
feature_pipeline.py).

Funções puras são testadas com objetos simples (`SimpleNamespace`), sem
banco. O pipeline é testado sobre um SQLite `:memory:` populado pelo
gerador — nunca sobre data/synthetic/sentinel.db.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from src.synthetic.feature_pipeline import FEATURE_COLUMNS, TABLE_COLUMNS, build_feature_table
from src.synthetic.features import (
    MAX_REFERENCE_DISTANCE_KM,
    MIN_HISTORY_FOR_HABITS,
    HabitProfile,
    amount_deviation,
    habit_features,
    implied_travel_speed,
    location_distance,
    new_city_flag,
    new_device_flag,
    unusual_hour,
)
from src.synthetic.generator import MAX_PLAUSIBLE_SPEED_KMH, populate_database
from src.synthetic.models import BehaviorProfile, Customer, Merchant, Transaction

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


def test_new_device_flag_with_no_history_is_false():
    """Primeira transação do cliente: sem histórico, não há "dispositivo novo"."""
    assert new_device_flag(tx(device_id=1), seen_device_ids=set()) is False


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
    assert not firsts["new_device"].any()  # sem histórico, não há "dispositivo novo"


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
    assert normal_rate < 0.01  # era ~1,7% antes do itinerário dos viajantes
    assert viagem_rate > 50 * normal_rate


def test_impossible_travel_does_not_flag_legitimate_travelers(populated):
    """Regressão do problema original: 5,9% dos viajantes legítimos eram
    marcados. Com itinerário em blocos, devem ficar em ~0%.
    """
    engine, _, table = populated
    with Session(engine) as session:
        profile = {
            tid: p
            for tid, p in session.execute(
                select(Transaction.id, Customer.behavior_profile).join(
                    Customer, Customer.id == Transaction.customer_id
                )
            )
        }
    legit_travelers = table[
        (~table["is_fraud"]) & (table["transaction_id"].map(profile) == BehaviorProfile.VIAJANTE)
    ]

    assert len(legit_travelers) > 1000
    assert legit_travelers["is_impossible_travel"].mean() < 0.005


def test_new_device_separates_dispositivo_novo_from_normal_transactions(populated):
    _, _, table = populated
    novo = table[table["fraud_scenario"] == "dispositivo_novo"]
    normal = table[~table["is_fraud"]]

    assert len(novo) > 0
    assert novo["new_device"].mean() >= 0.95
    assert normal["new_device"].mean() < 0.02
    assert novo["new_device"].mean() > 20 * normal["new_device"].mean()


def test_new_device_is_now_informative_about_fraud_overall(populated):
    _, _, table = populated
    by_class = table.groupby("is_fraud")["new_device"].mean()
    assert by_class[True] > 10 * by_class[False]


def test_feature_ranges(populated):
    _, _, table = populated
    assert table["location_distance"].between(0.0, 1.0).all()
    assert table["unusual_hour"].between(0.0, 1.0).all()
    assert (table["implied_speed_kmh"] >= 0).all()


def test_features_reflect_the_scenario_that_generated_each_fraud(populated):
    """Cada cenário deixa a assinatura esperada na feature correspondente."""
    _, _, table = populated
    normal = table[~table["is_fraud"]]

    # Gerado com 6x-15x a média do GERADOR, mas comparado com a média do
    # HISTÓRICO, que compras grandes legítimas anteriores podem inflar. Sem
    # histórico suficiente (início do cliente), a feature fica em 0.
    valor = table[table["fraud_scenario"] == "valor_atipico"]
    known = valor["habit_avg_amount"].notna()
    assert known.mean() > 0.8
    assert (valor.loc[known, "amount_deviation"] >= 2.0).all()
    assert valor.loc[known, "amount_deviation"].median() >= 5.0
    assert normal["amount_deviation"].abs().mean() < 1.0

    # Com histórico suficiente, todo horário atípico sai da janela estimada.
    horario = table[table["fraud_scenario"] == "horario_atipico"]
    known = horario["habit_avg_amount"].notna()
    assert known.mean() > 0.8
    assert (horario.loc[known, "unusual_hour"] > 0).all()

    geo = table[table["fraud_scenario"] == "anomalia_geografica"]
    assert geo["location_distance"].mean() > 2 * normal["location_distance"].median()


# ---------------------------------------------------------------------------
# new_city_flag (função pura)
# ---------------------------------------------------------------------------


def home(city="São Paulo"):
    return SimpleNamespace(home_city=city)


def test_new_city_flag_without_history_is_false():
    """Primeira transação do cliente: mesma decisão de new_device."""
    assert new_city_flag("Paris", home(), seen_cities=set()) is False


def test_new_city_flag_for_a_city_never_seen_before():
    assert new_city_flag("Paris", home(), seen_cities={"São Paulo", "Rio de Janeiro"}) is True


def test_new_city_flag_for_a_repeated_city_is_false():
    assert new_city_flag("Rio de Janeiro", home(), seen_cities={"São Paulo", "Rio de Janeiro"}) is False


def test_home_city_is_always_known():
    """Mesmo sem nenhuma compra anterior na cidade natal."""
    assert new_city_flag("São Paulo", home("São Paulo"), seen_cities={"Rio de Janeiro"}) is False


# ---------------------------------------------------------------------------
# new_city no pipeline: cronológico por cliente, sem vazamento
# ---------------------------------------------------------------------------


def _expected_new_city(session):
    """Recalcula new_city do zero, transação a transação, olhando SÓ o
    passado de cada cliente — a referência contra a qual o pipeline é comparado.
    """
    rows = session.execute(
        select(Transaction.id, Transaction.customer_id, Merchant.city, Customer.home_city)
        .join(Merchant, Merchant.id == Transaction.merchant_id)
        .join(Customer, Customer.id == Transaction.customer_id)
        .order_by(Transaction.customer_id, Transaction.timestamp, Transaction.id)
    ).all()
    seen, expected = {}, {}
    for tid, cid, city, home_city in rows:
        past = seen.setdefault(cid, set())
        expected[tid] = bool(past) and city != home_city and city not in past
        past.add(city)
    return expected


def test_pipeline_new_city_matches_a_strictly_chronological_recomputation(populated):
    engine, _, table = populated
    with Session(engine) as session:
        expected = _expected_new_city(session)
    assert table["new_city"].tolist() == table["transaction_id"].map(expected).tolist()


def test_pipeline_new_city_is_false_on_each_customers_first_transaction(populated):
    _, _, table = populated
    with Session(populated[0]) as session:
        firsts = session.execute(
            select(Transaction.customer_id, Transaction.id).order_by(
                Transaction.customer_id, Transaction.timestamp, Transaction.id
            )
        ).all()
    first_ids = {}
    for cid, tid in firsts:
        first_ids.setdefault(cid, tid)
    assert not table[table["transaction_id"].isin(first_ids.values())]["new_city"].any()


def test_new_city_has_no_temporal_leakage():
    """Acrescentar transações NO FUTURO não muda o new_city do passado; a
    transação futura numa cidade inédita é marcada, e uma segunda nessa
    mesma cidade já não é.
    """
    engine = create_engine("sqlite:///:memory:")
    populate_database(engine, seed=3, n_customers=12)

    with Session(engine) as session:
        before = build_feature_table(session).set_index("transaction_id")["new_city"]

        customer = session.scalars(select(Customer)).first()
        last = max(customer.transactions, key=lambda t: t.timestamp)
        visited = {t.merchant.city for t in customer.transactions} | {customer.home_city}
        far_merchant = session.scalars(select(Merchant).where(Merchant.city.not_in(visited))).first()
        assert far_merchant is not None

        def add(ts):
            tx = Transaction(
                customer_id=customer.id, card_id=last.card_id, merchant_id=far_merchant.id,
                device_id=last.device_id, timestamp=ts, amount=last.amount,
                lat=far_merchant.lat, lon=far_merchant.lon, distance_from_home_km=0.0,
                is_fraud=False, fraud_scenario=None,
            )
            session.add(tx)
            session.flush()
            return tx.id

        first_new = add(last.timestamp + timedelta(days=2))
        second_same = add(last.timestamp + timedelta(days=3))
        after = build_feature_table(session).set_index("transaction_id")["new_city"]

    assert after.loc[before.index].equals(before)  # o passado não mudou
    assert bool(after.loc[first_new])  # cidade inédita no futuro: marcada
    assert not after.loc[second_same]


def test_new_city_flags_geographic_anomalies_far_more_than_normal_transactions(populated):
    _, _, table = populated
    geo = table[table["fraud_scenario"] == "anomalia_geografica"]
    normal = table[~table["is_fraud"]]
    assert geo["new_city"].mean() >= 0.8
    assert normal["new_city"].mean() < 0.02


# ---------------------------------------------------------------------------
# HabitProfile: hábitos estimados só com o histórico
# ---------------------------------------------------------------------------


def test_habit_profile_estimates_mean_and_hour_window_from_past_purchases():
    history = [tx(amount=a, hour=h) for a, h in [(100, 9), (120, 10), (80, 12), (100, 14), (100, 18)]]
    profile = HabitProfile(history)
    assert profile.is_known and profile.n_transactions == 5
    assert profile.avg_amount_baseline == pytest.approx(100.0)
    # quantis 5% e 95% dos horários (9, 10, 12, 14, 18), interpolação linear
    assert profile.typical_hour_start == pytest.approx(9.2)
    assert profile.typical_hour_end == pytest.approx(17.2)


def test_habit_features_are_neutral_without_enough_history():
    short = HabitProfile([tx(amount=10.0, hour=9)] * (MIN_HISTORY_FOR_HABITS - 1))
    assert not short.is_known
    assert habit_features(tx(amount=10_000.0, hour=3), short) == (0.0, 0.0)


def test_habit_features_compare_with_the_estimated_habit():
    profile = HabitProfile([tx(amount=100.0, hour=h) for h in (10, 11, 12, 13, 14)])
    deviation, hour_gap = habit_features(tx(amount=300.0, hour=22), profile)
    assert deviation == pytest.approx(2.0)
    assert hour_gap == pytest.approx((22 - profile.typical_hour_end) / 12)


def test_pipeline_never_reads_the_generator_parameters(populated):
    """Os parâmetros com que o gerador sorteou as compras (`Customer`) não
    podem influenciar as features: trocá-los no banco não muda a tabela."""
    engine, _, table = populated
    with Session(engine) as session:
        for c in session.scalars(select(Customer)):
            c.avg_amount_baseline *= 7.0
            c.typical_hour_start, c.typical_hour_end = 0, 1
        session.flush()
        changed = build_feature_table(session)
        session.rollback()
    pd.testing.assert_frame_equal(changed, table)


def test_pipeline_habits_use_only_earlier_transactions(populated):
    """O hábito de cada linha é recalculável só com as compras ANTERIORES
    do cliente (nunca a própria nem as futuras)."""
    engine, _, table = populated
    with Session(engine) as session:
        txs = session.scalars(select(Transaction).order_by(Transaction.customer_id, Transaction.timestamp, Transaction.id)).all()
        by_id = table.set_index("transaction_id")
        for cid in sorted({t.customer_id for t in txs})[:10]:
            history = [t for t in txs if t.customer_id == cid]
            for i, t in enumerate(history):
                expected = HabitProfile(history[:i])
                row = by_id.loc[t.id]
                if expected.is_known:
                    assert row["habit_avg_amount"] == pytest.approx(expected.avg_amount_baseline)
                else:
                    assert pd.isna(row["habit_avg_amount"])
