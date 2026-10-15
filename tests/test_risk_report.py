"""Testes da avaliação comparativa (src/risk/report.py) e do script
scripts/run_risk_engine.py.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from src.risk.engine import LEVEL_HIGH, LEVEL_LOW, LEVEL_MEDIUM
from src.risk.report import (
    APPROACH_COMBINED,
    APPROACH_COMBINED_DEFAULT,
    APPROACH_ML,
    APPROACH_RULES,
    evaluate_risk_engine,
    pick_examples,
)

ROOT = Path(__file__).resolve().parent.parent
REAL_DB = ROOT / "data" / "synthetic" / "sentinel.db"


@pytest.fixture(scope="module")
def ev(risk_table):
    return evaluate_risk_engine(risk_table)


def test_comparison_has_the_three_approaches_plus_default_bands(ev):
    assert ev.comparison["abordagem"].tolist() == [
        APPROACH_RULES,
        APPROACH_ML,
        APPROACH_COMBINED,
        APPROACH_COMBINED_DEFAULT,
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


def test_weight_sweep_covers_pure_rules_to_pure_ml(ev):
    assert ev.weight_sweep["peso_ml"].tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert (ev.weight_sweep["peso_ml"] + ev.weight_sweep["peso_regras"]).eq(1.0).all()


def test_scenario_recall_covers_every_fraud_in_the_test_set(ev):
    totals = ev.scenario_recall[(APPROACH_COMBINED, "total")]
    assert totals.sum() == int(ev.split.test["is_fraud"].sum())


def test_level_distribution_adds_up(ev):
    dist = ev.level_distribution
    assert dist.index.tolist() == [LEVEL_LOW, LEVEL_MEDIUM, LEVEL_HIGH]
    assert dist["transacoes"].sum() == len(ev.split.test)
    assert dist["fraudes"].sum() == int(ev.split.test["is_fraud"].sum())
    assert dist["pct_transacoes"].sum() == pytest.approx(1.0)


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
