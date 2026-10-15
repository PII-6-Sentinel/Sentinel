"""Testes da avaliação comparativa (src/risk/report.py) e do script
scripts/run_risk_engine.py.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.risk.engine import LEVEL_HIGH, LEVEL_LOW, LEVEL_MEDIUM, EngineConfig
from src.risk.report import (
    APPROACH_COMBINED,
    APPROACH_COMBINED_5050,
    APPROACH_ML,
    APPROACH_RULES,
    MIN_RULES_WEIGHT,
    choose_rules_weight,
    evaluate_risk_engine,
    pick_examples,
)

ROOT = Path(__file__).resolve().parent.parent
REAL_DB = ROOT / "data" / "synthetic" / "sentinel.db"


@pytest.fixture(scope="module")
def ev(risk_table):
    return evaluate_risk_engine(risk_table)


def test_comparison_has_rules_ml_chosen_weight_and_5050(ev):
    assert ev.comparison["abordagem"].tolist() == [
        APPROACH_RULES,
        APPROACH_ML,
        APPROACH_COMBINED,
        APPROACH_COMBINED_5050,
    ]
    for col in ("pr_auc", "roc_auc", "precision", "recall", "f1"):
        assert ev.comparison[col].between(0, 1).all()


def test_metrics_are_computed_on_the_test_customers_only(ev):
    n_test_frauds = int(ev.split.test["is_fraud"].sum())
    for _, row in ev.comparison.iterrows():
        assert row["tp"] + row["fn"] == n_test_frauds
        assert row["tp"] + row["fp"] + row["fn"] + row["tn"] == len(ev.split.test)


def test_calibrated_bands_are_valid_and_come_from_validation(ev):
    bands = ev.engine.config.bands
    assert 0 <= bands.review_threshold < bands.block_threshold < 100
    assert bands.review_threshold == ev.review_thresholds[APPROACH_COMBINED]
    assert bands.block_threshold == ev.block_thresholds[APPROACH_COMBINED]
    assert ev.review_thresholds[APPROACH_COMBINED_5050] < ev.block_thresholds[APPROACH_COMBINED_5050]


def test_weight_sweep_covers_0_to_100_percent_rules_in_steps_of_10(ev):
    assert ev.weight_sweep["peso_regras"].tolist() == pytest.approx([k / 10 for k in range(11)])
    assert (ev.weight_sweep["peso_ml"] + ev.weight_sweep["peso_regras"]).round(10).eq(1.0).all()
    assert ev.weight_sweep["elegivel"].tolist() == [w >= MIN_RULES_WEIGHT - 1e-9 for w in ev.weight_sweep["peso_regras"]]


def _sweep(pr_aucs):
    rules = [k / 10 for k in range(11)]
    return pd.DataFrame(
        {
            "peso_regras": rules,
            "peso_ml": [round(1 - w, 10) for w in rules],
            "pr_auc": pr_aucs,
            "roc_auc": [0.9] * 11,
            "elegivel": [w >= MIN_RULES_WEIGHT - 1e-9 for w in rules],
        }
    )


def test_choose_rules_weight_respects_the_minimum_even_if_pure_ml_is_best():
    # ML puro (0% regras) é o melhor da grade, mas está fora da restrição
    aucs = [0.99, 0.95, 0.90, 0.80, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]
    assert choose_rules_weight(_sweep(aucs)) == pytest.approx(0.4)


def test_choose_rules_weight_breaks_ties_towards_more_rules():
    aucs = [0.5] * 3 + [0.80, 0.80, 0.70] + [0.6] * 5
    assert choose_rules_weight(_sweep(aucs)) == pytest.approx(0.4)


def test_engine_uses_the_weight_chosen_on_validation(ev):
    chosen = choose_rules_weight(ev.weight_sweep)
    assert ev.engine.config.behavior_weight == pytest.approx(chosen)
    assert ev.engine.config.ml_weight == pytest.approx(1 - chosen)
    assert chosen >= MIN_RULES_WEIGHT


def test_explicit_config_overrides_the_weight_choice(risk_table):
    fixed = evaluate_risk_engine(risk_table, config=EngineConfig(ml_weight=0.5, behavior_weight=0.5))
    assert fixed.engine.config.behavior_weight == pytest.approx(0.5)


def test_scenario_recall_covers_every_fraud_in_the_test_set(ev):
    totals = ev.scenario_recall[(APPROACH_COMBINED, "total")]
    assert totals.sum() == int(ev.split.test["is_fraud"].sum())


def test_level_distribution_adds_up(ev):
    dist = ev.level_distribution
    assert dist.index.tolist() == [LEVEL_LOW, LEVEL_MEDIUM, LEVEL_HIGH]
    assert dist["transacoes"].sum() == len(ev.split.test)
    assert dist["fraudes"].sum() == int(ev.split.test["is_fraud"].sum())
    assert dist["pct_transacoes"].sum() == pytest.approx(1.0)


def test_stability_covers_the_four_approaches(ev):
    assert ev.stability["abordagem"].tolist() == [
        APPROACH_RULES, APPROACH_ML, APPROACH_COMBINED, APPROACH_COMBINED_5050,
    ]


def test_stability_uses_five_customer_folds(ev):
    assert ev.stability["folds"].eq(5).all()
    assert ev.stability["pr_auc_media"].between(0, 1).all()


def test_report_is_deterministic(risk_table, ev):
    again = evaluate_risk_engine(risk_table)
    assert again.comparison.equals(ev.comparison)
    assert again.engine.config.bands == ev.engine.config.bands


def test_pick_examples_returns_one_example_per_level(ev):
    examples = pick_examples(ev.test_scores)
    assert set(examples) <= {LEVEL_LOW, LEVEL_MEDIUM, LEVEL_HIGH}
    for level, row in examples.items():
        assert row["level"] == level
        assert ev.engine.assess(row.to_dict()).level == level


@pytest.mark.skipif(not REAL_DB.exists(), reason="data/synthetic/sentinel.db não gerado")
def test_script_runs_end_to_end():
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_risk_engine.py")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    for section in ("1. Split por cliente", "4. Compara", "5. Recall por cen", "6. Distribui", "8. Exemplos"):
        assert section in out.stdout
