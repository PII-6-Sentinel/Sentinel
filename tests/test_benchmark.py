"""Avaliação temporal, volume de alertas e limiar por custo (src/benchmark.py),
sobre o dataset sintético no formato do Kaggle (`synthetic_df`, conftest).
"""

import numpy as np
import pandas as pd
import pytest

from src.benchmark import (
    alert_volume,
    compare_splits,
    cost_curve,
    cost_optimal_threshold,
    model_scores_temporal,
    temporal_split,
)


def test_temporal_split_trains_on_the_past_and_tests_on_the_future(synthetic_df):
    X_train, X_test, y_train, y_test = temporal_split(synthetic_df, test_size=0.25)
    assert X_train["Time"].max() <= X_test["Time"].min()
    assert len(X_test) == 100 and len(X_train) == 300
    assert len(y_train) + len(y_test) == len(synthetic_df)
    assert "Class" not in X_train.columns


def test_temporal_split_rejects_invalid_size(synthetic_df):
    with pytest.raises(ValueError):
        temporal_split(synthetic_df, test_size=1.0)


def test_compare_splits_reports_both_splits_for_every_model(synthetic_df):
    table = compare_splits(synthetic_df)
    assert set(table["split"]) == {"aleatório", "temporal"}
    assert table.groupby("split")["modelo"].nunique().tolist() == [3, 3]
    assert table["pr_auc"].between(0, 1).all() and table["roc_auc"].between(0, 1).all()


def _toy():
    y = pd.Series([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    amount = pd.Series([100.0, 10, 300.0, 10, 10, 10, 10, 10, 10, 10])
    times = pd.Series(np.linspace(0, 2 * 24 * 3600, 10))  # dois dias
    return y, score, amount, times


def test_alert_volume_counts_alerts_and_what_they_catch():
    y, score, amount, times = _toy()
    table = alert_volume(y, score, amount, times, alert_rates=(0.1, 0.3))
    first, second = table.iloc[0], table.iloc[1]
    assert first["alertas"] == 1 and first["fraudes_pegas"] == 1 and first["precisao"] == 1.0
    assert second["alertas"] == 3 and second["fraudes_pegas"] == 2 and second["recall"] == 1.0
    assert first["alertas_por_dia"] == pytest.approx(0.5)  # 1 alerta em 2 dias
    assert first["valor_fraude_coberto"] == pytest.approx(100 / 400)
    assert second["valor_fraude_coberto"] == pytest.approx(1.0)


def test_alert_volume_grows_with_the_budget(synthetic_df):
    y_test, scores, amount, times = model_scores_temporal(synthetic_df)
    table = alert_volume(y_test, scores["Regressão Logística"], amount, times)
    assert table["alertas"].is_monotonic_increasing
    assert table["fraudes_pegas"].is_monotonic_increasing


def test_cost_curve_adds_review_and_missed_fraud_costs():
    y, score, amount, _ = _toy()
    curve = cost_curve(y, score, amount, review_cost=5.0)
    no_alert = curve.iloc[-1]
    assert no_alert["alertas"] == 0 and no_alert["custo_total"] == pytest.approx(400.0)
    for row in curve.itertuples():
        assert row.custo_total == pytest.approx(row.alertas * 5.0 + row.custo_fraude_perdida)


def test_cost_optimal_threshold_alerts_until_the_frauds_pay_off():
    y, score, amount, _ = _toy()
    best = cost_optimal_threshold(cost_curve(y, score, amount, review_cost=5.0))
    # alertar as 3 primeiras custa 15 e pega as duas fraudes (R$ 400)
    assert best["alertas"] == 3 and best["fn"] == 0 and best["custo_total"] == pytest.approx(15.0)


def test_expensive_reviews_mean_fewer_alerts():
    y, score, amount, _ = _toy()
    cheap = cost_optimal_threshold(cost_curve(y, score, amount, review_cost=1.0))
    pricey = cost_optimal_threshold(cost_curve(y, score, amount, review_cost=150.0))
    assert pricey["alertas"] < cheap["alertas"]


def test_cost_curve_rejects_negative_review_cost():
    y, score, amount, _ = _toy()
    with pytest.raises(ValueError):
        cost_curve(y, score, amount, review_cost=-1)


def test_benchmark_script_runs_end_to_end(synthetic_df, tmp_path, capsys):
    import importlib.util
    from pathlib import Path

    csv = tmp_path / "creditcard.csv"
    synthetic_df.to_csv(csv, index=False)
    script = Path(__file__).resolve().parent.parent / "scripts" / "run_kaggle_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_kaggle_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.main(["--custo-revisao", "10"], path=csv) == 0
    out = capsys.readouterr().out
    assert "temporal" in out and "Volume de alertas" in out and "Limiar por custo" in out
    assert module.main([], path=tmp_path / "nao_existe.csv") == 1
