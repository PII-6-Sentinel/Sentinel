"""Testes da tabela de risco (src/risk/dataset.py)."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.risk.dataset import RISK_TABLE_COLUMNS
from src.synthetic.feature_pipeline import FEATURE_COLUMNS, build_feature_table
from src.synthetic.models import Device, FraudScenario, Transaction


def test_columns_and_one_row_per_transaction(synthetic_engine, risk_table):
    with Session(synthetic_engine) as session:
        n_tx = len(session.scalars(select(Transaction.id)).all())
    assert list(risk_table.columns) == RISK_TABLE_COLUMNS
    assert len(risk_table) == n_tx
    assert risk_table["transaction_id"].is_unique
    assert not risk_table.drop(columns=["fraud_scenario"]).isna().any().any()


def test_feature_columns_are_exactly_build_feature_table(synthetic_engine, risk_table):
    """A tabela de risco não recalcula nem altera as features existentes."""
    with Session(synthetic_engine) as session:
        features = build_feature_table(session)
    assert features["transaction_id"].tolist() == risk_table["transaction_id"].tolist()
    for col in FEATURE_COLUMNS:
        assert features[col].tolist() == risk_table[col].tolist()


def test_device_untrusted_mirrors_device_is_trusted(synthetic_engine, risk_table):
    with Session(synthetic_engine) as session:
        trusted = dict(
            session.execute(
                select(Transaction.id, Device.is_trusted).join(Device, Device.id == Transaction.device_id)
            ).all()
        )
    expected = risk_table["transaction_id"].map(lambda t: not trusted[t])
    assert (risk_table["device_untrusted"] == expected).all()


def test_new_device_fraud_always_uses_an_untrusted_device(risk_table):
    novo = risk_table[risk_table["fraud_scenario"] == FraudScenario.DISPOSITIVO_NOVO.value]
    assert len(novo) > 0
    assert novo["device_untrusted"].all()
