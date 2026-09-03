import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src.models import (
    ModelResult,
    StatisticalBaseline,
    apply_threshold,
    calibrate_threshold_by_contamination,
    evaluate_at_threshold,
    evaluate_model,
    fraud_score,
    train_all_models,
)
from src.preprocessing import (
    scale_amount_time,
    split_features_target,
    train_test_split_stratified,
)


# ---------------------------------------------------------------------------
# 1. AUC-PR é calculada
# ---------------------------------------------------------------------------


def test_evaluate_model_computes_auc_pr_without_dropping_existing_metrics(synthetic_df):
    X, y = split_features_target(synthetic_df)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, random_state=0)
    X_train, X_test, _ = scale_amount_time(X_train, X_test)

    baseline = StatisticalBaseline().fit(X_train)
    threshold = calibrate_threshold_by_contamination(
        fraud_score(baseline, X_train), y_train.mean()
    )
    result = evaluate_model(baseline, X_test, y_test, threshold=threshold)

    expected_auc_pr = average_precision_score(y_test, fraud_score(baseline, X_test))
    assert result.auc_pr == pytest.approx(expected_auc_pr)
    assert 0.0 <= result.auc_pr <= 1.0

    # Nenhuma métrica pré-existente pode ter sido removida.
    for attr in ("precision", "recall", "f1", "auc_roc", "confusion_matrix", "y_pred"):
        assert hasattr(result, attr)
    assert result.confusion_matrix.shape == (2, 2)


# ---------------------------------------------------------------------------
# 2. Score é separado de classificação
# ---------------------------------------------------------------------------


def test_fraud_score_returns_continuous_values_not_labels(synthetic_df):
    X, y = split_features_target(synthetic_df)
    X_train, X_test, y_train, _ = train_test_split_stratified(X, y, random_state=0)
    X_train, X_test, _ = scale_amount_time(X_train, X_test)

    baseline = StatisticalBaseline().fit(X_train)
    scores = fraud_score(baseline, X_test)

    # Um escore contínuo não pode ser só {0, 1} — senão já seria uma decisão.
    assert scores.dtype.kind == "f"
    assert len(set(np.round(scores, 6))) > 2


def test_apply_threshold_is_independent_of_any_model():
    # apply_threshold não recebe nem precisa de um modelo — é uma função pura.
    y_score = np.array([0.1, 0.4, 0.6, 0.9])
    y_pred = apply_threshold(y_score, threshold=0.5)
    assert list(y_pred) == [0, 0, 1, 1]
    assert y_pred.dtype.kind in ("i", "u")


# ---------------------------------------------------------------------------
# 3. Diferentes thresholds produzem diferentes métricas
# ---------------------------------------------------------------------------


def test_different_thresholds_produce_different_metrics():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    low = evaluate_at_threshold(y_true, y_score, threshold=0.5)
    high = evaluate_at_threshold(y_true, y_score, threshold=0.95)

    assert low["recall"] == pytest.approx(1.0)
    assert high["recall"] == pytest.approx(0.0)
    assert not np.array_equal(low["confusion_matrix"], high["confusion_matrix"])


def test_evaluate_model_same_scores_different_thresholds(synthetic_df):
    """O mesmo modelo/escore, avaliado em dois thresholds, dá resultados
    diferentes de precision/recall/f1 — sem retreinar nada entre as duas
    chamadas (prova de que threshold não está mais escondido dentro do
    treino/predict do modelo).
    """
    X, y = split_features_target(synthetic_df)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, random_state=0)
    X_train, X_test, _ = scale_amount_time(X_train, X_test)

    logreg = train_all_models(X_train, y_train)["Regressão Logística"]
    score_train = fraud_score(logreg, X_train)

    permissive = calibrate_threshold_by_contamination(score_train, contamination=0.30)
    strict = calibrate_threshold_by_contamination(score_train, contamination=0.02)
    assert permissive < strict

    result_permissive = evaluate_model(logreg, X_test, y_test, threshold=permissive)
    result_strict = evaluate_model(logreg, X_test, y_test, threshold=strict)

    # Mesmo AUC-ROC/AUC-PR (não dependem do threshold, é o mesmo y_score)...
    assert result_permissive.auc_roc == pytest.approx(result_strict.auc_roc)
    # ...mas recall tem que ser maior (ou igual) no corte mais permissivo.
    assert result_permissive.recall >= result_strict.recall


# ---------------------------------------------------------------------------
# 4. Threshold inválido é tratado corretamente
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_threshold", [None, float("nan")])
def test_apply_threshold_rejects_invalid_threshold(bad_threshold):
    with pytest.raises(ValueError):
        apply_threshold(np.array([0.1, 0.9]), bad_threshold)


@pytest.mark.parametrize("bad_contamination", [0, 1, -0.1, 1.5, float("nan")])
def test_calibrate_threshold_rejects_invalid_contamination(bad_contamination):
    with pytest.raises(ValueError):
        calibrate_threshold_by_contamination(np.array([0.1, 0.5, 0.9]), bad_contamination)


def test_evaluate_model_requires_explicit_threshold(synthetic_df):
    """threshold não pode ter default escondido — omiti-lo é erro."""
    X, y = split_features_target(synthetic_df)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, random_state=0)
    X_train, X_test, _ = scale_amount_time(X_train, X_test)
    baseline = StatisticalBaseline().fit(X_train)

    with pytest.raises(TypeError):
        evaluate_model(baseline, X_test, y_test)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 5. (vazamento do scaler é coberto em tests/test_preprocessing.py)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. Os modelos continuam funcionando (smoke test de ponta a ponta)
# ---------------------------------------------------------------------------


def test_full_pipeline_smoke(synthetic_df):
    X, y = split_features_target(synthetic_df)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, random_state=0)
    X_train, X_test, _ = scale_amount_time(X_train, X_test)

    fraud_rate = float(y_train.mean())
    models = train_all_models(X_train, y_train)
    assert set(models.keys()) == {
        "Baseline Estatístico",
        "Regressão Logística",
        "Isolation Forest",
    }

    for name, model in models.items():
        threshold = calibrate_threshold_by_contamination(
            fraud_score(model, X_train), fraud_rate
        )
        result = evaluate_model(model, X_test, y_test, threshold=threshold)

        assert isinstance(result, ModelResult)
        assert len(result.y_pred) == len(y_test)
        assert set(np.unique(result.y_pred)).issubset({0, 1})
        assert 0.0 <= result.auc_roc <= 1.0
        assert 0.0 <= result.auc_pr <= 1.0
        # Neste dataset sintético (classes bem separadas), espera-se
        # detecção melhor que aleatória para os três modelos.
        assert result.auc_roc > 0.6, f"{name} com AUC-ROC muito baixo: {result.auc_roc}"
