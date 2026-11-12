"""Testes da decisão do analista (app/decisions.py) e do seu uso na Central
de Transações (app/views/transactions.py), com o `AppTest` do Streamlit.

O motor recomenda (aprovar/revisar/bloquear); o analista decide (aprovada/
bloqueada). A decisão é gravada na tabela `analyst_decisions`, sem alterar
as transações nem o rótulo da simulação.

Cada teste usa uma CÓPIA do banco temporário (`synthetic_db_file` é
compartilhado pela sessão de testes; gravar decisões nele vazaria estado
entre testes) — nunca o data/synthetic/sentinel.db real.
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, inspect, select
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import decisions as dc  # noqa: E402
import risk_analysis as ra  # noqa: E402

from src.synthetic.models import Transaction  # noqa: E402

APP = str(APP_DIR / "app.py")
PAGE = "Central de Transações"


@pytest.fixture
def db_copy(synthetic_db_file, tmp_path) -> Path:
    path = tmp_path / "sentinel_decisions.db"
    shutil.copy(synthetic_db_file, path)
    return path


def _frame(rows):
    """Frame mínimo no formato da análise: (id, ação, score interno)."""
    return pd.DataFrame(rows, columns=["transaction_id", "action", "raw_score"])


# ----------------------------------------------------------------------
# camada de dados (sem Streamlit)
# ----------------------------------------------------------------------


def test_table_is_created_on_demand_in_an_existing_database(db_copy):
    engine = create_engine(f"sqlite:///{db_copy}")
    with engine.connect() as conn:
        conn.exec_driver_sql("DROP TABLE IF EXISTS analyst_decisions")
    assert not inspect(engine).has_table("analyst_decisions")
    engine.dispose()

    decisions = dc.load_decisions(db_copy)
    assert decisions.empty

    engine = create_engine(f"sqlite:///{db_copy}")
    assert inspect(engine).has_table("analyst_decisions")
    engine.dispose()


def test_record_replace_and_undo(db_copy):
    when = datetime(2026, 11, 12, 14, 30)
    dc.record_decision(db_copy, 10, dc.DECISION_BLOCKED, "revisar", 55, note="  cliente não reconhece  ", now=when)
    decisions = dc.load_decisions(db_copy)
    assert decisions.loc[10, "decision"] == dc.DECISION_BLOCKED
    assert decisions.loc[10, "recommended_action"] == "revisar"
    assert decisions.loc[10, "risk_score"] == 55
    assert decisions.loc[10, "note"] == "cliente não reconhece"
    assert pd.Timestamp(decisions.loc[10, "decided_at"]) == pd.Timestamp(when)

    # decidir de novo substitui (uma decisão vigente por transação)
    dc.record_decision(db_copy, 10, dc.DECISION_APPROVED, "revisar", 55, note="   ")
    decisions = dc.load_decisions(db_copy)
    assert len(decisions) == 1
    assert decisions.loc[10, "decision"] == dc.DECISION_APPROVED
    assert decisions.loc[10, "note"] is None

    dc.undo_decision(db_copy, 10)
    assert dc.load_decisions(db_copy).empty
    dc.undo_decision(db_copy, 10)  # desfazer o que não existe não quebra


def test_invalid_decision_is_rejected(db_copy):
    with pytest.raises(ValueError):
        dc.record_decision(db_copy, 10, "revisar", "revisar", 55)


def test_deciding_never_touches_the_transactions_or_the_simulation_label(db_copy):
    engine = create_engine(f"sqlite:///{db_copy}")

    def snapshot():
        with Session(engine) as session:
            count = session.scalar(select(func.count()).select_from(Transaction))
            frauds = session.scalar(select(func.count()).select_from(Transaction).where(Transaction.is_fraud))
            first = session.get(Transaction, 1)
            return count, frauds, first.amount, first.is_fraud

    before = snapshot()
    dc.record_decision(db_copy, 1, dc.DECISION_BLOCKED, "aprovar", 3)
    assert snapshot() == before
    engine.dispose()


def test_status_comes_from_the_recommendation_until_the_analyst_decides():
    frame = _frame([(1, "aprovar", 10), (2, "revisar", 75), (3, "bloquear", 90), (4, "revisar", 76), (5, "aprovar", 5)])
    decisions = pd.DataFrame({"decision": [dc.DECISION_APPROVED, dc.DECISION_BLOCKED]}, index=[4, 5])
    out = dc.with_status(frame, decisions)
    assert out["status"].tolist() == [
        dc.STATUS_AUTO_APPROVED,
        dc.STATUS_PENDING_REVIEW,
        dc.STATUS_PREVENTIVE_BLOCK,
        dc.STATUS_ANALYST_APPROVED,
        dc.STATUS_ANALYST_BLOCKED,
    ]
    assert out["decision"].tolist() == [None, None, None, dc.DECISION_APPROVED, dc.DECISION_BLOCKED]
    assert "status" not in frame.columns  # não altera a análise guardada na sessão

    no_decisions = dc.with_status(frame, pd.DataFrame(columns=["decision"]))
    assert no_decisions["decision"].isna().all()


def test_queue_holds_what_the_engine_did_not_approve_highest_risk_first():
    frame = dc.with_status(
        _frame([(1, "aprovar", 10), (2, "revisar", 75), (3, "bloquear", 90), (4, "revisar", 80)]),
        pd.DataFrame({"decision": [dc.DECISION_APPROVED]}, index=[4]),
    )
    assert dc.review_queue(frame)["transaction_id"].tolist() == [3, 2]
    assert dc.next_in_queue(frame) == 3
    assert dc.next_in_queue(frame, exclude=3) == 2

    counts = dc.status_counts(frame)
    assert list(counts) == dc.STATUSES
    assert counts[dc.STATUS_PENDING_REVIEW] == counts[dc.STATUS_PREVENTIVE_BLOCK] == 1
    assert counts[dc.STATUS_ANALYST_APPROVED] == 1 and counts[dc.STATUS_ANALYST_BLOCKED] == 0

    everything_decided = dc.with_status(frame[["transaction_id", "action", "raw_score"]], pd.DataFrame(
        {"decision": [dc.DECISION_BLOCKED] * 3}, index=[2, 3, 4]
    ))
    assert dc.next_in_queue(everything_decided) is None


# ----------------------------------------------------------------------
# na Central de Transações
# ----------------------------------------------------------------------


@pytest.fixture
def analysis_calls(monkeypatch):
    calls = {"n": 0}
    original = ra.build_risk_table

    def counting(session):
        calls["n"] += 1
        return original(session)

    monkeypatch.setattr(ra, "build_risk_table", counting)
    return calls


@pytest.fixture
def central(monkeypatch, db_copy, analysis_calls) -> AppTest:
    """Central com a análise executada e o filtro "Conjunto" limpo."""
    monkeypatch.setenv(ra.DB_ENV_VAR, str(db_copy))
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio(key="nav_product").set_value(PAGE).run()
    next(b for b in at.button if b.label == "Executar análise de risco").click().run()
    at.multiselect(key="tx_f_split").set_value([]).run()
    assert not at.exception
    return at


def _status_frame(at: AppTest, db_path: Path) -> pd.DataFrame:
    return dc.with_status(at.session_state["risk_analysis"].frame, dc.load_decisions(db_path))


def _cards(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown if "metric-card" in m.value)


def _panel_id(at: AppTest) -> int | None:
    titles = [s.value for s in at.subheader if s.value.startswith("Investigação da transação #")]
    return int(titles[0].rsplit("#", 1)[1]) if titles else None


def test_queue_section_counts_what_waits_for_the_analyst(central, db_copy):
    frame = _status_frame(central, db_copy)
    pending = int((frame["action"] != "aprovar").sum())
    assert any(s.value == "Fila do analista" for s in central.subheader)
    cards = _cards(central)
    assert "Na fila" in cards and f">{pending:,}<".replace(",", ".") in cards
    assert "Aprovadas pelo analista" in cards and "Bloqueadas pelo analista" in cards
    assert set(central.dataframe[0].value["Status"]) <= set(dc.STATUSES)


def test_open_next_brings_the_highest_risk_pending_transaction(central, db_copy):
    central.button(key="tx_queue_next").click().run()
    expected = dc.next_in_queue(_status_frame(central, db_copy))
    assert _panel_id(central) == expected
    assert any("Decisão do analista" in m.value for m in central.markdown)


def test_approving_records_the_decision_and_updates_the_status_without_rerunning(central, db_copy, analysis_calls):
    central.button(key="tx_queue_next").click().run()
    tx_id = _panel_id(central)
    central.text_input(key=f"tx_note_{tx_id}").set_value("cliente confirmou por telefone")
    central.button(key="tx_approve").click().run()
    assert not central.exception

    decisions = dc.load_decisions(db_copy)
    assert decisions.loc[tx_id, "decision"] == dc.DECISION_APPROVED
    assert decisions.loc[tx_id, "note"] == "cliente confirmou por telefone"
    row = central.session_state["risk_analysis"].frame.set_index("transaction_id").loc[tx_id]
    assert decisions.loc[tx_id, "recommended_action"] == row["action"]
    assert decisions.loc[tx_id, "risk_score"] == row["risk_score"]

    assert _panel_id(central) == tx_id  # o painel continua na transação decidida
    table = central.dataframe[0].value.set_index("ID")
    assert table.loc[tx_id, "Status"] == dc.STATUS_ANALYST_APPROVED
    assert any(f"Transação #{tx_id} aprovada" in t.value for t in central.toast)
    assert central.button(key="tx_approve").disabled and not central.button(key="tx_undo").disabled
    assert analysis_calls["n"] == 1


def test_blocking_then_undoing_returns_to_the_engine_recommendation(central, db_copy):
    central.button(key="tx_queue_next").click().run()
    tx_id = _panel_id(central)
    original = central.dataframe[0].value.set_index("ID").loc[tx_id, "Status"]

    central.button(key="tx_block").click().run()
    assert dc.load_decisions(db_copy).loc[tx_id, "decision"] == dc.DECISION_BLOCKED
    assert central.dataframe[0].value.set_index("ID").loc[tx_id, "Status"] == dc.STATUS_ANALYST_BLOCKED

    central.button(key="tx_undo").click().run()
    assert tx_id not in dc.load_decisions(db_copy).index
    assert central.dataframe[0].value.set_index("ID").loc[tx_id, "Status"] == original
    assert central.button(key="tx_undo").disabled


def test_deciding_moves_the_queue_forward(central, db_copy):
    central.button(key="tx_queue_next").click().run()
    first = _panel_id(central)
    central.button(key="tx_block").click().run()
    central.button(key="tx_panel_next").click().run()
    second = _panel_id(central)
    assert second != first
    assert second == dc.next_in_queue(_status_frame(central, db_copy))


def test_show_queue_in_table_filters_the_pending_transactions_of_every_customer(central, db_copy):
    central.multiselect(key="tx_f_split").set_value(["teste"]).run()
    central.button(key="tx_queue_table").click().run()
    table = central.dataframe[0].value
    frame = _status_frame(central, db_copy)
    assert len(table) == int(frame["status"].isin(dc.QUEUE_STATUSES).sum()) > 0
    assert set(table["Status"]) <= set(dc.QUEUE_STATUSES)
    assert len(set(table["Conjunto"])) > 1  # a fila não se limita aos clientes de teste


def test_decisions_survive_clearing_and_rerunning_the_analysis(central, db_copy):
    central.button(key="tx_queue_next").click().run()
    tx_id = _panel_id(central)
    central.button(key="tx_approve").click().run()

    next(b for b in central.button if b.label == "Limpar resultado").click().run()
    next(b for b in central.button if b.label == "Executar análise de risco").click().run()
    central.multiselect(key="tx_f_split").set_value([]).run()
    assert central.dataframe[0].value.set_index("ID").loc[tx_id, "Status"] == dc.STATUS_ANALYST_APPROVED
