"""Testes do Monitor ao Vivo (Etapa 13): o gerador ao vivo
(`src/synthetic/live_generator.py`), a camada `app/live_monitor.py` (sem
Streamlit) e a página `app/views/live_monitor.py` (com o `AppTest`).

Os pontos que importam:

- cada transação gerada traz o rótulo real e segue a lógica do cenário;
- o motor não vê o rótulo: o veredito é o mesmo de `score_new_transaction`
  para os mesmos dados brutos;
- o veredito binário ("Fraudulenta" = não aprovada) e a matriz de confusão
  batem com as transações; nada é gravado no banco.
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
from src.synthetic.generator import CONTACTLESS_LIMIT_BRL, MAX_PLAUSIBLE_SPEED_KMH  # noqa: E402
from src.synthetic.geo_reference import distance_km  # noqa: E402
from src.synthetic.live_generator import generate_live_transaction  # noqa: E402
from src.synthetic.models import Channel, Customer, FraudScenario, Transaction  # noqa: E402

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


def _generated(db, n: int, fraud_rate: float, seed: int = 11):
    rng = random.Random(seed)
    with Session(db) as s:
        ids = lm.customer_ids(s)
        out = []
        for _ in range(n):
            tx = generate_live_transaction(s, rng, rng.choice(ids), fraud_rate)
            last = s.scalars(
                select(Transaction).where(Transaction.customer_id == tx.customer_id).order_by(Transaction.timestamp.desc())
            ).first()
            out.append((tx, last, s.get(Customer, tx.customer_id)))
        return out


# ----------------------------------------------------------------------
# gerador ao vivo
# ----------------------------------------------------------------------


def test_fraud_rate_controls_the_label(db):
    assert all(not tx.is_fraud and tx.fraud_scenario is None for tx, _, _ in _generated(db, 30, 0.0))
    assert all(tx.is_fraud and tx.fraud_scenario for tx, _, _ in _generated(db, 30, 1.0))


def test_every_transaction_comes_after_the_customer_history(db):
    for tx, last, _ in _generated(db, 60, 0.5):
        assert last is None or tx.timestamp > last.timestamp


def test_normal_purchases_are_not_impossible_travel(db):
    for tx, last, _ in _generated(db, 60, 0.0):
        hours = (tx.timestamp - last.timestamp).total_seconds() / 3600
        assert distance_km(last.lat, last.lon, tx.lat, tx.lon) / hours <= MAX_PLAUSIBLE_SPEED_KMH


def test_scenarios_follow_their_logic(db):
    for tx, last, customer in _generated(db, 120, 1.0):
        if tx.fraud_scenario == FraudScenario.VIAGEM_IMPOSSIVEL.value:
            hours = max((tx.timestamp - last.timestamp).total_seconds() / 3600, 1 / 60)
            assert distance_km(last.lat, last.lon, tx.lat, tx.lon) / hours > MAX_PLAUSIBLE_SPEED_KMH
        elif tx.fraud_scenario == FraudScenario.VALOR_ATIPICO.value:
            assert tx.amount >= 5.9 * customer.avg_amount_baseline
        elif tx.fraud_scenario == FraudScenario.DISPOSITIVO_NOVO.value:
            assert tx.device_id is None and tx.channel == Channel.ONLINE.value


def test_channel_and_payment_are_consistent(db):
    for tx, _, _ in _generated(db, 80, 0.5):
        if tx.channel == Channel.ONLINE.value:
            assert tx.payment_method is None
        elif tx.payment_method == "aproximacao":
            assert tx.amount <= CONTACTLESS_LIMIT_BRL


# ----------------------------------------------------------------------
# camada do monitor
# ----------------------------------------------------------------------


def test_verdict_is_the_same_as_score_new_transaction(db, risk_engine):
    """O motor recebe só os dados brutos: mesma entrada, mesmo resultado."""
    events = _stream(db, risk_engine, 25, 0.5)
    rng = random.Random(7)
    with Session(db) as s:
        ids = lm.customer_ids(s)
        for e in events:
            tx = generate_live_transaction(s, rng, rng.choice(ids), 0.5)
            expected = score_new_transaction(
                s,
                risk_engine,
                customer_id=tx.customer_id,
                amount=tx.amount,
                lat=tx.lat,
                lon=tx.lon,
                timestamp=tx.timestamp,
                device_id=tx.device_id,
                city=tx.city,
                channel=tx.channel,
                payment_method=tx.payment_method,
            )
            assert e.assessment == expected
            assert e.is_fraud == tx.is_fraud


def test_binary_verdict_and_correctness(db, risk_engine):
    for e in _stream(db, risk_engine, 30, 0.5):
        assert e.verdict == (lm.VERDICT_FRAUD if e.assessment.action != "aprovar" else lm.VERDICT_LEGIT)
        assert e.label == (lm.LABEL_FRAUD if e.is_fraud else lm.LABEL_LEGIT)
        assert e.correct == ((e.verdict == lm.VERDICT_FRAUD) == e.is_fraud)


def test_summary_matches_the_events(db, risk_engine):
    events = _stream(db, risk_engine, 40, 0.4)
    s = lm.summarize(events)
    assert s["tp"] + s["fp"] + s["fn"] + s["tn"] == 40
    assert s["approved"] + s["review"] + s["blocked"] == 40
    assert s["frauds"] == sum(e.is_fraud for e in events)
    assert s["flagged"] == s["review"] + s["blocked"]
    assert s["accuracy"] == pytest.approx(sum(e.correct for e in events) / 40)
    assert s["amount_held"] == pytest.approx(sum(e.amount for e in events if e.flagged))
    assert lm.summarize([])["accuracy"] is None


def test_frame_is_newest_first_and_has_verdict_and_label(db, risk_engine):
    frame = lm.events_frame(_stream(db, risk_engine, 5, 0.5))
    assert frame["#"].tolist() == [5, 4, 3, 2, 1]
    assert {"Veredito", "Rótulo real", "Acerto", "Canal"} <= set(frame.columns)


def test_reason_key_drops_numbers():
    assert lm._reason_key("Valor 3,2x acima da média do cliente") == "Valor acima da média do cliente"
    assert lm._reason_key("Viagem impossível: 1.234 km/h entre duas compras") == "Viagem impossível"
    assert lm._reason_key("Compra online, canal raro para o cliente (5% das compras anteriores)") == (
        "Compra online, canal raro para o cliente"
    )


def test_decision_time_is_measured(db, risk_engine):
    assert all(e.decision_ms > 0 for e in _stream(db, risk_engine, 5, 0.5))


def test_presentation_seed_repeats_the_same_stream(db, risk_engine):
    def picks():
        return [(e.customer_id, e.scenario, e.amount, e.timestamp) for e in _stream(db, risk_engine, 15, 0.4, lm.PRESENTATION_SEED)]

    assert picks() == picks()


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


def _burst(at: AppTest) -> AppTest:
    return next(b for b in at.button if b.label == "Gerar 10 agora").click().run()


def test_page_opens_empty_and_stopped(page):
    assert not page.exception
    assert page.toggle(key="live_running").value is False
    assert "live_events" not in page.session_state


def test_burst_shows_verdict_next_to_the_real_label(page):
    _burst(page)
    assert not page.exception
    assert len(page.session_state["live_events"]) == 10
    assert any(s.value == "Última transação" for s in page.subheader)
    html = " ".join(m.value for m in page.markdown)
    assert "Veredito do Sentinel" in html and "Rótulo real" in html


def test_clear_button_empties_the_stream(page):
    _burst(page)
    next(b for b in page.button if b.label == "Limpar fluxo").click().run()
    assert not page.exception
    assert "live_events" not in page.session_state


def test_presentation_mode_restarts_the_same_sequence(page):
    page.toggle(key="live_presentation").set_value(True).run()
    first = [(e.customer_id, e.scenario) for e in _burst(page).session_state["live_events"]]
    next(b for b in page.button if b.label == "Limpar fluxo").click().run()
    assert [(e.customer_id, e.scenario) for e in _burst(page).session_state["live_events"]] == first


def test_fraudulent_verdicts_raise_an_alert(page):
    page.slider(key="live_fraud_rate").set_value(100).run()
    _burst(page)
    flagged = [e for e in page.session_state["live_events"] if e.flagged]
    toasts = [t.value for t in page.toast]
    assert len(toasts) == min(len(flagged), 3)
    assert all("fraudulenta" in t for t in toasts)
    # curto: com 3 linhas o Streamlit mostra um "view more" que não acrescenta nada
    assert all(len(t) <= 80 for t in toasts)
