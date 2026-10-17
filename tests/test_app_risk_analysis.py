"""Testes da camada de análise da Central de Transações (app/risk_analysis.py).

Sem Streamlit: tudo roda sobre um banco sintético em arquivo temporário. O
ponto central é a EQUIVALÊNCIA com o Risk Engine — a camada do app executa
a análise etapa a etapa (para exibir tempos), mas não pode produzir nada
diferente de `evaluate_risk_engine` / `RiskEngine.assess_frame`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# app/ não é um pacote importável sob `app.` (ver app/app.py): os módulos são
# irmãos, importados pelo nome. Os testes fazem o mesmo.
APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import risk_analysis as ra  # noqa: E402
from src.risk.report import MIN_RULES_WEIGHT, evaluate_risk_engine  # noqa: E402


@pytest.fixture(scope="module")
def events():
    return []


@pytest.fixture(scope="module")
def result(synthetic_db_file, events):
    return ra.run_analysis(synthetic_db_file, progress=lambda label, seconds: events.append((label, seconds)))


@pytest.fixture(scope="module")
def official(result):
    """A avaliação oficial do motor (src/risk/report.py), sobre a mesma tabela."""
    return evaluate_risk_engine(result.features.reset_index(drop=True))


# ----------------------------------------------------------------------
# etapas e cronometragem
# ----------------------------------------------------------------------


def test_steps_are_the_six_visible_stages_in_order(result):
    assert [s.label for s in result.steps] == ra.STEP_LABELS
    assert ra.STEP_LABELS == [
        "Carregando transações do banco",
        "Calculando features comportamentais",
        "Aplicando o modelo de ML",
        "Aplicando as regras",
        "Combinando no Risk Score",
        "Classificando por nível",
    ]
    assert all(s.seconds >= 0 for s in result.steps)
    assert result.total_seconds == pytest.approx(sum(s.seconds for s in result.steps))


def test_progress_callback_announces_start_then_finish_of_each_step(events):
    expected = []
    for label in ra.STEP_LABELS:
        expected += [(label, None), (label, "done")]
    got = [(label, None if seconds is None else "done") for label, seconds in events]
    assert got == expected
    assert all(s >= 0 for _, s in events if s is not None)


def test_missing_database_raises_a_specific_error(tmp_path):
    with pytest.raises(ra.DatabaseNotFound):
        ra.run_analysis(tmp_path / "nao_existe.db")


def test_db_path_can_be_overridden_by_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(ra.DB_ENV_VAR, str(tmp_path / "x.db"))
    assert ra.resolve_db_path() == tmp_path / "x.db"
    monkeypatch.delenv(ra.DB_ENV_VAR)
    assert ra.resolve_db_path() == ra.DEFAULT_DB_PATH


# ----------------------------------------------------------------------
# equivalência com o motor
# ----------------------------------------------------------------------


def test_weight_and_bands_are_the_ones_the_official_evaluation_chooses(result, official):
    cfg = official.engine.config
    assert result.rules_weight == pytest.approx(cfg.behavior_weight)
    assert result.ml_weight == pytest.approx(cfg.ml_weight)
    assert result.rules_weight >= MIN_RULES_WEIGHT - 1e-9
    assert (result.review_threshold, result.block_threshold) == (cfg.bands.review_threshold, cfg.bands.block_threshold)


def test_scores_levels_and_actions_equal_the_engine_assess_frame(result):
    reference = result.engine.assess_frame(result.features.reset_index(drop=True)).set_index("transaction_id")
    frame = result.frame.set_index("transaction_id").loc[reference.index]
    for column in ("raw_score", "risk_score", "level", "action"):
        assert frame[column].tolist() == reference[column].tolist(), column
    assert frame["ml_score"].to_numpy() == pytest.approx(reference["ml_score"].round(1).to_numpy())
    assert frame["behavior_score"].to_numpy() == pytest.approx(reference["behavior_score"].round(1).to_numpy())


def test_test_customers_get_the_same_decisions_as_the_official_report(result, official):
    mine = result.frame[result.frame["conjunto"] == ra.SPLIT_TEST].set_index("transaction_id")
    ref = official.test_scores.set_index("transaction_id")
    assert set(mine.index) == set(ref.index)
    ref = ref.loc[mine.index]
    assert mine["raw_score"].tolist() == ref["raw_score"].tolist()
    assert mine["level"].tolist() == ref["level"].tolist()


# ----------------------------------------------------------------------
# a tabela de resultados
# ----------------------------------------------------------------------


def test_one_row_per_transaction_in_the_database(result, synthetic_db_file):
    import sqlite3

    with sqlite3.connect(synthetic_db_file) as conn:
        n = conn.execute("select count(*) from transactions").fetchone()[0]
    assert len(result.frame) == n
    assert result.frame["transaction_id"].is_unique


def test_frame_is_sorted_by_risk_score_descending(result):
    scores = result.frame["risk_score"].to_numpy()
    assert (np.diff(scores) <= 0).all()


def test_each_customer_belongs_to_exactly_one_split(result):
    per_customer = result.frame.groupby("customer_id")["conjunto"].nunique()
    assert (per_customer == 1).all()
    assert set(result.frame["conjunto"]) == {ra.SPLIT_TRAIN, ra.SPLIT_VALIDATION, ra.SPLIT_TEST}
    assert sum(result.customers_by_split.values()) == result.frame["customer_id"].nunique()


def test_frame_has_what_the_table_and_the_panel_need(result):
    needed = {
        "transaction_id", "customer", "profile", "amount", "city", "timestamp", "risk_score", "action", "level",
        "ml_score", "behavior_score", "home_city", "home_lat", "home_lon", "tx_lat", "tx_lon",
        "distance_from_home_km", "new_device", "device_untrusted", "device_type", "is_fraud", "conjunto",
    }
    assert needed <= set(result.frame.columns)
    assert result.frame["profile"].isin({"fixo", "viajante", "alto_gasto", "novo_cliente"}).all()


def test_analysis_is_deterministic(synthetic_db_file, result):
    again = ra.run_analysis(synthetic_db_file)
    pd.testing.assert_frame_equal(again.frame, result.frame)


# ----------------------------------------------------------------------
# resumo
# ----------------------------------------------------------------------


def test_summary_counts_add_up_to_the_total(result):
    s = ra.summarize(result.frame)
    assert s["total"] == len(result.frame)
    for key in ("por_nivel", "por_acao"):
        assert s[key]["quantidade"].sum() == s["total"]
        assert s[key]["percentual"].sum() == pytest.approx(1.0)
    assert s["por_nivel"].index.tolist() == ra.LEVELS
    assert s["por_acao"].index.tolist() == ra.ACTIONS
    # nível e ação são a mesma decisão: baixo=aprovar, médio=revisar, alto=bloquear
    assert s["por_nivel"]["quantidade"].tolist() == s["por_acao"]["quantidade"].tolist()


# ----------------------------------------------------------------------
# filtros
# ----------------------------------------------------------------------


def test_no_filter_returns_everything(result):
    assert len(ra.apply_filters(result.frame)) == len(result.frame)
    assert len(ra.apply_filters(result.frame, levels=[], actions=[], customers=[], profiles=[], splits=[])) == len(result.frame)


def test_level_and_action_filters(result):
    high = ra.apply_filters(result.frame, levels=["alto"])
    assert len(high) > 0 and (high["level"] == "alto").all()
    blocked = ra.apply_filters(result.frame, actions=["bloquear"])
    assert set(blocked["transaction_id"]) == set(high["transaction_id"])
    two = ra.apply_filters(result.frame, levels=["médio", "alto"])
    assert set(two["level"]) == {"médio", "alto"}


def test_customer_and_profile_filters(result):
    customer = int(result.frame["customer_id"].iloc[0])
    one = ra.apply_filters(result.frame, customers=[customer])
    assert (one["customer_id"] == customer).all() and len(one) > 0
    traveler = ra.apply_filters(result.frame, profiles=["viajante"])
    assert len(traveler) > 0 and (traveler["profile"] == "viajante").all()


def test_amount_range_is_inclusive(result):
    lo, hi = 100.0, 500.0
    band = ra.apply_filters(result.frame, amount_range=(lo, hi))
    assert band["amount"].between(lo, hi).all()
    assert len(band) == int(result.frame["amount"].between(lo, hi).sum())
    exact = float(result.frame["amount"].iloc[0])
    assert len(ra.apply_filters(result.frame, amount_range=(exact, exact))) >= 1


def test_filters_combine_with_and(result):
    both = ra.apply_filters(result.frame, levels=["alto"], profiles=["fixo"], splits=[ra.SPLIT_TEST])
    assert (both["level"] == "alto").all() and (both["profile"] == "fixo").all() and (both["conjunto"] == ra.SPLIT_TEST).all()
    assert len(both) < len(ra.apply_filters(result.frame, levels=["alto"]))


# ----------------------------------------------------------------------
# investigação
# ----------------------------------------------------------------------


def test_investigation_agrees_with_the_table_on_a_sample_of_every_level(result):
    sample = []
    for level in ra.LEVELS:
        sample += result.frame[result.frame["level"] == level]["transaction_id"].head(25).tolist()
    sample += result.frame["transaction_id"].iloc[::97].tolist()
    assert {result.frame.set_index("transaction_id").loc[t, "level"] for t in sample} == set(ra.LEVELS)

    table = result.frame.set_index("transaction_id")
    for tx_id in sample:
        a = ra.investigate(result, tx_id).assessment
        row = table.loc[tx_id]
        assert (a.risk_score, a.raw_score, a.level, a.action) == (
            row["risk_score"], row["raw_score"], row["level"], row["action"],
        ), tx_id
        assert a.ml_score == pytest.approx(row["ml_score"])
        assert a.behavior_score == pytest.approx(row["behavior_score"])


def test_flagged_transactions_come_with_readable_reasons(result):
    flagged = result.frame[result.frame["level"] != "baixo"]["transaction_id"].head(40)
    for tx_id in flagged:
        reasons = ra.investigate(result, int(tx_id)).assessment.reasons
        assert reasons and all(isinstance(r, str) and r for r in reasons)


def test_investigating_an_unknown_transaction_is_an_error(result):
    with pytest.raises(KeyError):
        ra.investigate(result, -1)


# ----------------------------------------------------------------------
# eficiência: só nos clientes de teste
# ----------------------------------------------------------------------


def test_efficiency_is_computed_on_test_customers_only(result):
    eff = ra.efficiency(result)
    test = result.frame[result.frame["conjunto"] == ra.SPLIT_TEST]
    m = eff["metrics"]
    assert eff["transactions"] == len(test)
    assert eff["customers"] == result.customers_by_split[ra.SPLIT_TEST]
    assert eff["frauds"] == int(test["is_fraud"].sum())
    assert eff["legit"] == len(test) - eff["frauds"]
    assert m["tp"] + m["fn"] == eff["frauds"]
    assert m["fp"] + m["tn"] == eff["legit"]
    assert eff["by_action"].to_numpy().sum() == len(test)


def test_efficiency_ignores_training_and_validation_rows(result):
    """Se as linhas de treino/validação influenciassem a métrica, estragar os
    rótulos delas mudaria o resultado. Não muda.
    """
    baseline = ra.efficiency(result)["metrics"]
    tampered = result.frame.copy()
    outside = tampered["conjunto"] != ra.SPLIT_TEST
    tampered.loc[outside, "is_fraud"] = ~tampered.loc[outside, "is_fraud"]
    tampered.loc[outside, "raw_score"] = 0
    tampered.loc[outside, "level"] = "baixo"
    other = ra.AnalysisResult(**{**result.__dict__, "frame": tampered})
    assert ra.efficiency(other)["metrics"] == baseline


def test_efficiency_matches_the_official_test_report(result, official):
    m = ra.efficiency(result)["metrics"]
    ref = official.comparison.set_index("abordagem").loc["Combinado (motor)"]
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (ref["tp"], ref["fp"], ref["fn"], ref["tn"])
    assert m["pr_auc"] == pytest.approx(ref["pr_auc"])
    assert m["precision"] == pytest.approx(ref["precision"])
    assert m["recall"] == pytest.approx(ref["recall"])


def test_efficiency_scenario_table_matches_the_official_recall(result, official):
    mine = ra.efficiency(result)["by_scenario"]
    ref = official.scenario_recall["Combinado (motor)"]
    assert mine["detected"].to_dict() == ref["detected"].to_dict()
    assert mine["total"].to_dict() == ref["total"].to_dict()
