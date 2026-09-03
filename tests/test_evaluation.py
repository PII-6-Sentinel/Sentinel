import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

import src.evaluation as evaluation_module
from src.evaluation import (
    aggregate_cv_results,
    calculate_metrics,
    cross_validate_models,
    threshold_sweep,
)
from src.models import apply_threshold, evaluate_at_threshold
from src.preprocessing import split_features_target, train_test_split_stratified

MODEL_NAMES = {"Baseline Estatístico", "Regressão Logística", "Isolation Forest"}


# ---------------------------------------------------------------------------
# 1. StratifiedKFold produz exatamente 5 folds
# ---------------------------------------------------------------------------


def test_cross_validate_produces_five_folds_per_model(synthetic_df):
    X, y = split_features_target(synthetic_df)
    X_train, _X_test, y_train, _y_test = train_test_split_stratified(X, y)

    results = cross_validate_models(X_train, y_train)

    for name, r in results.items():
        assert len(r.metrics_by_fold) == 5, f"{name} não tem 5 folds"


# ---------------------------------------------------------------------------
# 2. Todos os folds mantêm representação das duas classes
# ---------------------------------------------------------------------------


def test_all_folds_have_both_classes_represented(synthetic_df):
    X, y = split_features_target(synthetic_df)
    X_train, _X_test, y_train, _y_test = train_test_split_stratified(X, y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_train_pos, fold_val_pos in skf.split(X_train, y_train):
        y_fold_train = y_train.iloc[fold_train_pos]
        y_fold_val = y_train.iloc[fold_val_pos]
        assert set(y_fold_train.unique()) == {0, 1}
        assert set(y_fold_val.unique()) == {0, 1}


# ---------------------------------------------------------------------------
# 3 e 4. Threshold calibrado exclusivamente no fold train; validation nunca
# participa do cálculo do threshold
# ---------------------------------------------------------------------------


def test_threshold_calibrated_only_with_fold_train_scores(synthetic_df, monkeypatch):
    X, y = split_features_target(synthetic_df)
    X_train, _X_test, y_train, _y_test = train_test_split_stratified(X, y)

    # Replica os mesmos folds que cross_validate_models vai gerar
    # internamente (mesmos n_splits/random_state), só para saber de
    # antemão o tamanho esperado do treino de cada fold.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    expected_fold_train_sizes = [
        len(fold_train_pos) for fold_train_pos, _ in skf.split(X_train, y_train)
    ]

    calls: list[int] = []
    original = evaluation_module.calibrate_threshold_by_contamination

    def spy(y_score, contamination):
        # O que importa aqui é o TAMANHO do array de score recebido: se o
        # validation do fold tivesse vazado para dentro do cálculo do
        # threshold, esse tamanho seria maior que o do treino do fold.
        calls.append(len(y_score))
        return original(y_score, contamination)

    monkeypatch.setattr(evaluation_module, "calibrate_threshold_by_contamination", spy)

    cross_validate_models(X_train, y_train)

    # 5 folds x 3 modelos = 15 chamadas de calibração de threshold.
    assert len(calls) == 15

    # Chamadas agrupadas em blocos de 3 (um por modelo), na ordem dos folds.
    for fold_idx, expected_size in enumerate(expected_fold_train_sizes):
        calls_for_fold = calls[fold_idx * 3 : (fold_idx + 1) * 3]
        assert calls_for_fold == [expected_size] * 3, (
            f"Fold {fold_idx + 1}: esperava threshold calibrado com "
            f"{expected_size} scores (tamanho do fold train), recebeu {calls_for_fold} "
            "— sinal de que o validation do fold vazou para o cálculo do threshold."
        )


# ---------------------------------------------------------------------------
# 5. Métricas são calculadas corretamente
# ---------------------------------------------------------------------------


def test_calculate_metrics_on_perfectly_separable_case():
    y_true = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    metrics = calculate_metrics(y_true, y_score, threshold=0.5)

    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["fpr"] == pytest.approx(0.0)
    assert metrics["tpr"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["mcc"] == pytest.approx(1.0)
    assert metrics["auc_roc"] == pytest.approx(1.0)
    assert metrics["auc_pr"] == pytest.approx(1.0)
    assert metrics["threshold"] == 0.5


def test_calculate_metrics_on_worst_case_threshold():
    # threshold acima de todos os escores -> nada é sinalizado como fraude.
    y_true = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    metrics = calculate_metrics(y_true, y_score, threshold=0.95)

    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["tpr"] == pytest.approx(0.0)
    assert metrics["fpr"] == pytest.approx(0.0)
    assert metrics["specificity"] == pytest.approx(1.0)
    # AUC-ROC/AUC-PR não dependem do threshold — continuam perfeitas.
    assert metrics["auc_roc"] == pytest.approx(1.0)
    assert metrics["auc_pr"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. Resultado agregado possui mean e std
# ---------------------------------------------------------------------------


def test_aggregate_cv_results_has_mean_and_std():
    fold_records = [
        {"fold": 1, "precision": 0.5, "recall": 0.6},
        {"fold": 2, "precision": 0.7, "recall": 0.8},
    ]

    result = aggregate_cv_results("Modelo Fake", fold_records)

    assert result.model_name == "Modelo Fake"
    assert result.metrics_by_fold == fold_records
    assert result.mean_metrics["precision"] == pytest.approx(0.6)
    assert result.mean_metrics["recall"] == pytest.approx(0.7)
    assert result.std_metrics["precision"] == pytest.approx(np.std([0.5, 0.7], ddof=1))
    # "fold" é metadado, não métrica — não deve entrar na agregação.
    assert "fold" not in result.mean_metrics
    assert "fold" not in result.std_metrics


# ---------------------------------------------------------------------------
# 7. CV retorna resultado para os três modelos
# ---------------------------------------------------------------------------


def test_cross_validate_returns_all_three_models(synthetic_df):
    X, y = split_features_target(synthetic_df)
    X_train, _X_test, y_train, _y_test = train_test_split_stratified(X, y)

    results = cross_validate_models(X_train, y_train)

    assert set(results.keys()) == MODEL_NAMES


# ---------------------------------------------------------------------------
# 8. Test set continua separado da CV
# ---------------------------------------------------------------------------


def test_cross_validate_models_has_no_test_set_parameter():
    params = set(inspect.signature(cross_validate_models).parameters)
    assert params == {"X_train", "y_train", "n_splits", "random_state"}
    assert not any("test" in p.lower() for p in params)


# ---------------------------------------------------------------------------
# Análise de Threshold — threshold_sweep() e uso de calculate_metrics()
# para simular um threshold sem retreinar nada.
# ---------------------------------------------------------------------------

_SIMPLE_Y_TRUE = [0, 0, 0, 0, 1, 1, 1, 1]
_SIMPLE_Y_SCORE = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])


def test_lower_threshold_flags_at_least_as_many_positives():
    y_pred_low = apply_threshold(_SIMPLE_Y_SCORE, threshold=0.05)  # abaixo de tudo
    y_pred_mid = apply_threshold(_SIMPLE_Y_SCORE, threshold=0.5)
    assert y_pred_low.sum() >= y_pred_mid.sum()


def test_higher_threshold_flags_at_most_as_many_positives():
    y_pred_mid = apply_threshold(_SIMPLE_Y_SCORE, threshold=0.5)
    y_pred_high = apply_threshold(_SIMPLE_Y_SCORE, threshold=0.95)  # acima de tudo
    assert y_pred_high.sum() <= y_pred_mid.sum()


def test_calculate_metrics_agrees_with_evaluate_at_threshold():
    """calculate_metrics() não pode divergir de evaluate_at_threshold() para
    as métricas que as duas calculam — calculate_metrics reusa
    evaluate_at_threshold internamente, este teste protege essa garantia.
    """
    threshold = 0.5
    direct = evaluate_at_threshold(_SIMPLE_Y_TRUE, _SIMPLE_Y_SCORE, threshold)
    via_calculate = calculate_metrics(_SIMPLE_Y_TRUE, _SIMPLE_Y_SCORE, threshold)

    assert via_calculate["precision"] == pytest.approx(direct["precision"])
    assert via_calculate["recall"] == pytest.approx(direct["recall"])
    assert via_calculate["f1"] == pytest.approx(direct["f1"])
    assert np.array_equal(via_calculate["confusion_matrix"], direct["confusion_matrix"])
    assert np.array_equal(via_calculate["y_pred"], direct["y_pred"])


def test_auc_roc_unchanged_across_thresholds():
    m_low = calculate_metrics(_SIMPLE_Y_TRUE, _SIMPLE_Y_SCORE, threshold=0.15)
    m_high = calculate_metrics(_SIMPLE_Y_TRUE, _SIMPLE_Y_SCORE, threshold=0.85)
    assert m_low["auc_roc"] == pytest.approx(m_high["auc_roc"])


def test_auc_pr_unchanged_across_thresholds():
    m_low = calculate_metrics(_SIMPLE_Y_TRUE, _SIMPLE_Y_SCORE, threshold=0.15)
    m_high = calculate_metrics(_SIMPLE_Y_TRUE, _SIMPLE_Y_SCORE, threshold=0.85)
    assert m_low["auc_pr"] == pytest.approx(m_high["auc_pr"])


def test_precision_recall_f1_vary_across_threshold_sweep(synthetic_pipeline):
    result = synthetic_pipeline["results"]["Regressão Logística"]
    y_test = synthetic_pipeline["y_test"]

    sweep = threshold_sweep(y_test, result.y_score, n_points=50)

    assert len(sweep) == 50
    # Precisão/recall/f1 têm que MUDAR ao longo do sweep — se fossem
    # constantes, o gráfico "métrica x threshold" não teria utilidade.
    assert sweep["precision"].nunique() > 1
    assert sweep["recall"].nunique() > 1
    # threshold do sweep respeita a escala observada de y_score (nunca
    # assume 0-1 "cegamente" — ver docstring de threshold_sweep).
    assert sweep["threshold"].min() == pytest.approx(result.y_score.min())
    assert sweep["threshold"].max() == pytest.approx(result.y_score.max())


def test_confusion_matrix_changes_with_threshold():
    cm_low = calculate_metrics(_SIMPLE_Y_TRUE, _SIMPLE_Y_SCORE, threshold=0.15)[
        "confusion_matrix"
    ]
    cm_high = calculate_metrics(_SIMPLE_Y_TRUE, _SIMPLE_Y_SCORE, threshold=0.95)[
        "confusion_matrix"
    ]
    assert not np.array_equal(cm_low, cm_high)


def test_threshold_outside_observed_score_range_is_handled():
    """Threshold abaixo do mínimo (sinaliza tudo) ou acima do máximo
    (sinaliza nada) do escore observado precisa continuar produzindo
    métricas válidas, sem erro — o usuário pode arrastar o slider até a
    ponta, e as pontas são exatamente threshold == min/max do escore.
    """
    below_min = calculate_metrics(_SIMPLE_Y_TRUE, _SIMPLE_Y_SCORE, threshold=-100.0)
    assert below_min["recall"] == pytest.approx(1.0)  # tudo sinalizado
    assert below_min["fpr"] == pytest.approx(1.0)

    above_max = calculate_metrics(_SIMPLE_Y_TRUE, _SIMPLE_Y_SCORE, threshold=100.0)
    assert above_max["recall"] == pytest.approx(0.0)  # nada sinalizado
    assert above_max["fpr"] == pytest.approx(0.0)


def test_threshold_sweep_and_calculate_metrics_work_for_all_three_models(
    synthetic_pipeline,
):
    y_test = synthetic_pipeline["y_test"]
    for name, result in synthetic_pipeline["results"].items():
        sweep = threshold_sweep(y_test, result.y_score, n_points=20)
        assert len(sweep) == 20

        mid_threshold = float(np.median(result.y_score))
        metrics = calculate_metrics(y_test, result.y_score, mid_threshold)
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert metrics["confusion_matrix"].shape == (2, 2)


def test_manual_threshold_simulation_does_not_change_official_threshold(
    synthetic_pipeline,
):
    """Simular um threshold manual (como o slider da página de Análise de
    Threshold faz) não pode alterar o threshold oficial calibrado pelo
    pipeline, nem o y_score armazenado no resultado — a simulação é
    somente leitura.
    """
    model_name = "Regressão Logística"
    result = synthetic_pipeline["results"][model_name]
    official_threshold = synthetic_pipeline["thresholds"][model_name]
    y_score_before = result.y_score.copy()

    # Simula o usuário mexendo no slider para vários valores manuais,
    # bem diferentes do threshold oficial.
    for manual_threshold in (
        float(result.y_score.min()),
        official_threshold / 2,
        float(result.y_score.max()),
    ):
        calculate_metrics(synthetic_pipeline["y_test"], result.y_score, manual_threshold)

    assert synthetic_pipeline["thresholds"][model_name] == official_threshold
    assert result.threshold == official_threshold  # o do ModelResult também
    assert np.array_equal(result.y_score, y_score_before)
