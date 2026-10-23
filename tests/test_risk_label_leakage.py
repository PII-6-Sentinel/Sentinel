"""O ML não vê o rótulo.

O rótulo (`is_fraud`, e o `fraud_scenario` que o acompanha) só pode ser o
ALVO do treino. Se ele vazasse para a matriz de entrada — direto, ou por uma
feature calculada a partir dele —, o modelo "acertaria" lendo a resposta.
Três garantias:

1. a matriz de entrada só tem `ML_FEATURES`, e nenhuma delas é o rótulo;
2. o score de uma transação não muda quando o rótulo dela muda (nem
   quando as colunas do rótulo somem);
3. as features calculadas a partir do banco são idênticas se os rótulos do
   banco forem embaralhados — nenhuma feature é derivada do rótulo.
"""

import numpy as np
import pytest
from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session

from src.risk.dataset import RISK_FEATURES, build_risk_table
from src.risk.ml_score import ML_FEATURES, MLScorer, ml_feature_matrix
from src.synthetic.feature_pipeline import FEATURE_COLUMNS, build_feature_table
from src.synthetic.generator import populate_database
from src.synthetic.models import Transaction

LABEL_COLUMNS = {"is_fraud", "fraud_scenario"}


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("leakage") / "sentinel.db"
    engine = create_engine(f"sqlite:///{path}")
    populate_database(engine, n_customers=60)
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def table(db):
    with Session(db) as session:
        return build_risk_table(session)


@pytest.fixture(scope="module")
def scorer(table):
    return MLScorer().fit(table)


def test_no_label_column_is_a_feature():
    assert not LABEL_COLUMNS & set(ML_FEATURES)
    assert not LABEL_COLUMNS & set(RISK_FEATURES)
    assert not LABEL_COLUMNS & set(FEATURE_COLUMNS)


def test_model_input_has_only_the_declared_features(table):
    assert list(ml_feature_matrix(table).columns) == ML_FEATURES


def test_score_ignores_the_label(table, scorer):
    original = scorer.score(table)

    flipped = table.copy()
    flipped["is_fraud"] = ~flipped["is_fraud"].astype(bool)
    flipped["fraud_scenario"] = "qualquer_coisa"
    np.testing.assert_array_equal(scorer.score(flipped), original)

    without = table.drop(columns=list(LABEL_COLUMNS))
    np.testing.assert_array_equal(scorer.score(without), original)


def test_features_do_not_depend_on_the_label(db, table):
    """Inverte o rótulo de TODAS as transações no banco: as features não mudam."""
    with Session(db) as session:
        before = build_feature_table(session)[["transaction_id", *FEATURE_COLUMNS]]
        session.execute(update(Transaction).values(is_fraud=~Transaction.is_fraud, fraud_scenario=None))
        session.flush()
        after = build_feature_table(session)[["transaction_id", *FEATURE_COLUMNS]]
        session.rollback()  # o banco do módulo continua intacto
    assert after.equals(before)


def test_training_needs_the_label_only_as_target(table):
    """Sem `is_fraud` não há alvo: o treino falha, em vez de treinar sem rótulo."""
    with pytest.raises(KeyError):
        MLScorer().fit(table.drop(columns=["is_fraud"]))
