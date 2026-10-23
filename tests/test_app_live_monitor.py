"""Testes do Monitor ao Vivo: a camada `app/live_monitor.py` (sem Streamlit)
e a página `app/views/live_monitor.py` (com o `AppTest`).

Os pontos que importam:

- o veredito de cada transação do fluxo é o mesmo de `score_new_transaction`
  para a mesma entrada (o monitor não decide nada);
- a proporção de cenários de fraude respeita o controle (0% = só compras
  normais; 100% = só cenários de fraude);
- o resumo bate com as transações; nada é gravado no banco.
"""

import random
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import live_monitor as lm  # noqa: E402
import risk_analysis as ra  # noqa: E402
from src.risk.new_transaction import score_new_transaction  # noqa: E402
from src.synthetic.models import Transaction  # noqa: E402

APP = str(APP_DIR / "app.py")


@pytest.fixture(scope="module")
def db(synthetic_db_file):
    engine = create_engine(f"sqlite:///{synthetic_db_file}")
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def risk_engine(synthetic_db_file):
    return ra.run_analysis(synthetic_db_file).engine


def _stream(db, risk_engine, n: int, fraud_rate: float, seed: int = 7) -> list[lm.LiveEvent]:
    rng = random.Random(seed)
    with Session(db) as s:
        ids = lm.customer_ids(s)
        return [lm.next_event(s, risk_engine, rng, i + 1, fraud_rate, ids) for i in range(n)]


def test_verdict_is_the_same_as_score_new_transaction(db, risk_engine):
    events = _stream(db, risk_engine, 25, 0.5)
    with Session(db) as s:
        for e in events:
            expected = score_new_transaction(
                s,
                risk_engine,
                customer_id=e.customer_id,
                amount=e.amount,
                lat=e.lat,
                lon=e.lon,
                timestamp=e.timestamp,
                device_id=None if e.device_id == lm.NEW_DEVICE else e.device_id,
                city=e.city,
            )
            assert e.assessment == expected


def test_fraud_rate_zero_only_generates_normal_purchases(db, risk_engine):
    events = _stream(db, risk_engine, 20, 0.0)
    assert all(not e.injected_fraud for e in events)


def test_fraud_rate_one_only_generates_fraud_scenarios(db, risk_engine):
    events = _stream(db, risk_engine, 20, 1.0)
    assert all(e.injected_fraud for e in events)
    assert all(e.scenario in lm.FRAUD_SCENARIOS for e in events)


def test_transactions_come_after_the_customer_history(db, risk_engine):
    events = _stream(db, risk_engine, 20, 0.5)
    with Session(db) as s:
        for e in events:
            last = s.scalar(select(func.max(Transaction.timestamp)).where(Transaction.customer_id == e.customer_id))
            assert last is None or e.timestamp > last


def test_summary_matches_the_events(db, risk_engine):
    events = _stream(db, risk_engine, 30, 0.4)
    s = lm.summarize(events)
    assert s["total"] == 30
    assert s["approved"] + s["review"] + s["blocked"] == 30
    assert s["injected"] + s["normal"] == 30
    assert s["caught"] == sum(e.injected_fraud and e.flagged for e in events)
    assert s["false_alarms"] == sum(not e.injected_fraud and e.flagged for e in events)
    assert s["amount_held"] == pytest.approx(sum(e.amount for e in events if e.flagged))
    assert lm.summarize([])["detection_rate"] is None


def test_frame_is_newest_first(db, risk_engine):
    events = _stream(db, risk_engine, 5, 0.5)
    frame = lm.events_frame(events)
    assert frame["#"].tolist() == [5, 4, 3, 2, 1]


def test_reason_key_drops_numbers():
    assert lm._reason_key("Valor 3,2x acima da média do cliente") == "Valor acima da média do cliente"
    assert lm._reason_key("Viagem impossível: 1.234 km/h entre duas compras") == "Viagem impossível"
    assert lm._reason_key("Localização incomum: 2.100 km da residência") == "Localização incomum"


def test_stream_writes_nothing_to_the_database(db, risk_engine):
    with Session(db) as s:
        before = s.scalar(select(func.count()).select_from(Transaction))
    _stream(db, risk_engine, 10, 0.5)
    with Session(db) as s:
        assert s.scalar(select(func.count()).select_from(Transaction)) == before


# ----------------------------------------------------------------------
# página
# ----------------------------------------------------------------------


@pytest.fixture
def page(synthetic_db_file, monkeypatch):
    monkeypatch.setenv(ra.DB_ENV_VAR, str(synthetic_db_file))
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio(key="nav_product").set_value("Monitor ao Vivo").run()
    return at


def test_page_opens_empty_and_stopped(page):
    assert not page.exception
    assert page.toggle(key="live_running").value is False
    assert not page.session_state["live_events"] if "live_events" in page.session_state else True


def test_burst_button_fills_the_dashboard(page):
    next(b for b in page.button if b.label == "Gerar 10 agora").click().run()
    assert not page.exception
    assert len(page.session_state["live_events"]) == 10
    assert any(s.value == "Última transação" for s in page.subheader)


def test_clear_button_empties_the_stream(page):
    next(b for b in page.button if b.label == "Gerar 10 agora").click().run()
    next(b for b in page.button if b.label == "Limpar fluxo").click().run()
    assert not page.exception
    assert "live_events" not in page.session_state
