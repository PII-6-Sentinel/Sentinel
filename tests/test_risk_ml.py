"""Testes do score de ML e da camada de avaliação (split, métricas,
calibração) do Risk Engine.
"""

import numpy as np
import pandas as pd
import pytest

from src.risk.evaluation import (
    calibrate_block_threshold,
    calibrate_review_threshold,
    classification_metrics,
    recall_by_scenario,
    split_by_customer,
)
from src.risk.ml_score import ML_FEATURES, MLScorer, ml_feature_matrix


@pytest.fixture(scope="module")
def split(risk_table):
    return split_by_customer(risk_table)


# ---------------------------------------------------------------------------
# split por cliente
# ---------------------------------------------------------------------------


def test_split_has_no_customer_in_more_than_one_set(split):
    train, val, test = (set(d["customer_id"]) for d in (split.train, split.validation, split.test))
    assert not train & val
    assert not train & test
    assert not val & test


def test_split_covers_every_transaction_exactly_once(split, risk_table):
    ids = pd.concat([split.train, split.validation, split.test])["transaction_id"]
    assert ids.is_unique
    assert set(ids) == set(risk_table["transaction_id"])


def test_split_fractions_are_over_customers(split, risk_table):
    n = risk_table["customer_id"].nunique()
    assert split.test["customer_id"].nunique() == pytest.approx(0.30 * n, abs=1)
    assert split.validation["customer_id"].nunique() == pytest.approx(0.20 * n, abs=1)


def test_every_set_has_frauds(split):
    for part in (split.train, split.validation, split.test):
        assert part["is_fraud"].sum() > 0


def test_split_is_deterministic_and_seed_dependent(risk_table):
    a = split_by_customer(risk_table, seed=42)
    b = split_by_customer(risk_table, seed=42)
    c = split_by_customer(risk_table, seed=7)
    assert a.test["transaction_id"].tolist() == b.test["transaction_id"].tolist()
    assert set(a.test["customer_id"]) != set(c.test["customer_id"])


@pytest.mark.parametrize("test_size, val_size", [(0.0, 0.2), (0.5, 0.5), (0.3, 1.0)])
def test_split_rejects_invalid_fractions(risk_table, test_size, val_size):
    with pytest.raises(ValueError):
        split_by_customer(risk_table, test_size=test_size, validation_size=val_size)


# ---------------------------------------------------------------------------
# modelo de ML
# ---------------------------------------------------------------------------


def test_feature_matrix_columns_and_log_speed(risk_table):
    X = ml_feature_matrix(risk_table)
    assert list(X.columns) == ML_FEATURES
    assert not X.isna().any().any()
    assert np.allclose(X["log_speed_kmh"], np.log1p(risk_table["implied_speed_kmh"]))


def test_ml_score_is_between_0_and_100(split):
    scores = MLScorer().fit(split.train).score(split.test)
    assert len(scores) == len(split.test)
    assert scores.min() >= 0 and scores.max() <= 100


def test_ml_training_is_deterministic(split):
    a = MLScorer().fit(split.train).score(split.test)
    b = MLScorer().fit(split.train).score(split.test)
    assert np.array_equal(a, b)


def test_ml_ranks_frauds_above_normal_on_unseen_customers(split):
    """Sanidade, não prova de desempenho real (rótulos sintéticos)."""
    scores = MLScorer().fit(split.train).score(split.test)
    m = classification_metrics(split.test["is_fraud"], scores, scores > 50)
    assert m["pr_auc"] > 0.5
    assert m["roc_auc"] > 0.9


def test_ml_refuses_to_score_before_fit(split):
    with pytest.raises(RuntimeError):
        MLScorer().score(split.test)


def test_ml_refuses_training_data_without_both_classes(split):
    with pytest.raises(ValueError):
        MLScorer().fit(split.train[~split.train["is_fraud"]])


def test_coefficients_cover_every_feature(split):
    coefs = MLScorer().fit(split.train).coefficients()
    assert list(coefs) == ML_FEATURES


# ---------------------------------------------------------------------------
# métricas
# ---------------------------------------------------------------------------


def test_classification_metrics_on_a_hand_made_case():
    y = [0, 0, 0, 1, 1]
    score = [10, 20, 80, 70, 90]
    m = classification_metrics(y, score, [s > 50 for s in score])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 0, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(0.8)
    assert set(m) >= {"pr_auc", "roc_auc", "precision", "recall", "f1"}


def test_recall_by_scenario_counts():
    table = pd.DataFrame(
        {
            "is_fraud": [True, True, True, False],
            "fraud_scenario": ["a", "a", "b", None],
        }
    )
    out = recall_by_scenario(table, [True, False, True, True])
    assert out.loc["a", "detected"] == 1 and out.loc["a", "total"] == 2
    assert out.loc["b", "recall"] == pytest.approx(1.0)
    assert len(out) == 2  # transações normais ficam de fora


# ---------------------------------------------------------------------------
# calibração de limiares
# ---------------------------------------------------------------------------


def test_review_threshold_is_the_highest_that_keeps_target_recall():
    y = np.array([1] * 10 + [0] * 10)
    score = np.array([95, 90, 85, 80, 75, 70, 65, 60, 55, 20] + [10] * 10)
    t = calibrate_review_threshold(y, score, target_recall=0.9)
    assert t == 54  # > 54 pega 9/10; > 55 pegaria só 8/10
    assert (score[y == 1] > t).mean() >= 0.9


def test_review_threshold_counts_forced_flags():
    y = np.array([1, 1, 0])
    score = np.array([90, 5, 1])
    forced = np.array([False, True, False])  # regra dura sinaliza a 2ª fraude
    assert calibrate_review_threshold(y, score, target_recall=1.0, forced=forced) == 89


def test_review_threshold_falls_back_to_default_when_unreachable():
    assert calibrate_review_threshold([1, 0], [0, 0], target_recall=0.9, default=30) == 30


def test_block_threshold_is_the_lowest_that_reaches_target_precision():
    y = np.array([1] * 6 + [0] * 4)
    score = np.array([95, 92, 90, 88, 86, 84, 83, 60, 50, 40])
    t = calibrate_block_threshold(y, score, review_threshold=30, target_precision=0.9, min_flagged=5)
    assert t == 83  # > 83: 6 fraudes de 6 bloqueios; > 82 incluiria um legítimo (6/7)


def test_block_threshold_is_always_above_the_review_threshold():
    y = np.array([1, 0, 0])
    score = np.array([10, 9, 8])
    assert calibrate_block_threshold(y, score, review_threshold=40, default=35) == 41
