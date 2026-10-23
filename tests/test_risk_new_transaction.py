"""Testes da pontuação de transação nova (src/risk/new_transaction.py).

O teste central confere, para transações que JÁ estão no banco, que as
features calculadas a partir dos dados brutos + histórico são as mesmas
de `build_risk_table` — ou seja, uma transação nova é pontuada exatamente
como as da avaliação.
"""

from collections import Counter
from datetime import timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.risk.engine import LEVEL_LOW, train_engine
from src.risk.evaluation import split_by_customer
from src.risk.new_transaction import compute_transaction_features, score_new_transaction
from src.synthetic.models import Customer, Device, Transaction

EXACT = ["amount_deviation", "unusual_hour", "new_device", "new_city", "is_impossible_travel", "device_untrusted", "merchant_city"]
# a distância salva no banco é arredondada a 2 casas; a calculada aqui não
APPROX = {"distance_from_home_km": 0.01, "location_distance": 1e-3, "implied_speed_kmh": 1e-6}


@pytest.fixture(scope="module")
def engine(risk_table):
    return train_engine(split_by_customer(risk_table).train)


def _raw(tx):
    return dict(
        customer_id=tx.customer_id,
        amount=tx.amount,
        lat=tx.lat,
        lon=tx.lon,
        timestamp=tx.timestamp,
        device_id=tx.device_id,
        channel=tx.channel,
        payment_method=tx.payment_method,
    )


def test_features_match_build_risk_table_for_existing_transactions(synthetic_engine, risk_table):
    rows = risk_table.set_index("transaction_id")
    with Session(synthetic_engine) as session:
        txs = session.scalars(select(Transaction)).all()
        same_minute = Counter((t.customer_id, t.timestamp) for t in txs)
        # mesmo cliente, mesmo minuto: a tabela desempata por id, uma
        # transação nova não tem id (documentado em new_transaction.py)
        comparable = [t for t in txs if same_minute[(t.customer_id, t.timestamp)] == 1]
        assert len(comparable) > 0.95 * len(txs)

        for tx in comparable:
            got = compute_transaction_features(session, **_raw(tx))
            expected = rows.loc[tx.id]
            for col in EXACT:
                assert got[col] == expected[col], (tx.id, col)
            for col, tol in APPROX.items():
                assert got[col] == pytest.approx(expected[col], abs=tol), (tx.id, col)


def test_first_transaction_of_a_customer_has_no_previous(synthetic_engine):
    with Session(synthetic_engine) as session:
        customer = session.scalars(select(Customer)).first()
        first = min(customer.transactions, key=lambda t: t.timestamp)
        before_everything = dict(_raw(first), timestamp=first.timestamp - timedelta(days=365))
        f = compute_transaction_features(session, **before_everything)
    assert f["implied_speed_kmh"] == 0.0
    assert f["is_impossible_travel"] is False
    assert f["new_device"] is False  # sem histórico, não há "dispositivo novo"


def test_unknown_device_is_new_and_untrusted(synthetic_engine):
    with Session(synthetic_engine) as session:
        tx = session.scalars(select(Transaction).order_by(Transaction.id.desc())).first()
        f = compute_transaction_features(
            session, **dict(_raw(tx), timestamp=tx.timestamp + timedelta(days=1), device_id=999_999)
        )
    assert f["new_device"] is True
    assert f["device_untrusted"] is True


def test_another_customers_device_is_untrusted(synthetic_engine):
    with Session(synthetic_engine) as session:
        tx = session.scalars(select(Transaction)).first()
        other = session.scalars(select(Device).where(Device.customer_id != tx.customer_id)).first()
        f = compute_transaction_features(
            session, **dict(_raw(tx), timestamp=tx.timestamp + timedelta(days=1), device_id=other.id)
        )
    assert f["device_untrusted"] is True


@pytest.mark.parametrize("bad", [{"customer_id": 10**9}, {"amount": 0.0}, {"amount": -5.0}])
def test_invalid_input_is_rejected(synthetic_engine, bad):
    with Session(synthetic_engine) as session:
        tx = session.scalars(select(Transaction)).first()
        with pytest.raises(ValueError):
            compute_transaction_features(session, **dict(_raw(tx), **bad))


def test_scoring_a_new_transaction_does_not_write_to_the_database(synthetic_engine, engine):
    with Session(synthetic_engine) as session:
        before = session.query(Transaction).count()
        tx = session.scalars(select(Transaction)).first()
        score_new_transaction(session, engine, **dict(_raw(tx), timestamp=tx.timestamp + timedelta(days=1)))
        assert session.query(Transaction).count() == before


def test_new_transaction_score_matches_the_engine_on_the_table_row(synthetic_engine, engine, risk_table):
    """Mesma transação, dois caminhos (dados brutos vs tabela): mesmo resultado."""
    rows = risk_table.set_index("transaction_id")
    with Session(synthetic_engine) as session:
        txs = session.scalars(select(Transaction)).all()
        same_minute = Counter((t.customer_id, t.timestamp) for t in txs)
        frauds = [t for t in txs if t.is_fraud and same_minute[(t.customer_id, t.timestamp)] == 1]
        assert len(frauds) >= 30
        for tx in frauds[:30]:
            from_raw = score_new_transaction(session, engine, **_raw(tx))
            from_table = engine.assess(rows.loc[tx.id].to_dict())
            assert from_raw.risk_score == from_table.risk_score
            assert from_raw.level == from_table.level


def test_ordinary_purchase_at_home_is_low_risk(synthetic_engine, engine):
    """Compra no bairro, valor da média, horário habitual, aparelho de sempre."""
    with Session(synthetic_engine) as session:
        customer = session.scalars(select(Customer).where(Customer.behavior_profile == "fixo")).first()
        last = max(customer.transactions, key=lambda t: t.timestamp)
        used = {t.device_id for t in customer.transactions}
        trusted = next(d for d in customer.devices if d.is_trusted and d.id in used)
        noon = (last.timestamp + timedelta(days=1)).replace(hour=customer.typical_hour_start + 1, minute=0)
        result = score_new_transaction(
            session,
            engine,
            customer_id=customer.id,
            amount=customer.avg_amount_baseline,
            lat=customer.home_lat,
            lon=customer.home_lon,
            timestamp=noon,
            device_id=trusted.id,
        )
    assert result.level == LEVEL_LOW
