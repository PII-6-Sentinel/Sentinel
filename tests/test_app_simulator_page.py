"""Testes da página Simulador (app/views/simulator.py), com o `AppTest`.

Requisito central: NADA pré-calculado. A página abre sem veredito; os botões
de cenário só preenchem o formulário; o resultado só existe depois do clique
em "Analisar transação". Para provar isso contamos quantas vezes o motor é
treinado (a etapa de features só roda dentro de `run_analysis`).

O banco é um arquivo temporário (fixture `synthetic_db_file`) apontado por
`SENTINEL_DB_PATH` — nunca o data/synthetic/sentinel.db real.
"""

import json
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import risk_analysis as ra  # noqa: E402
import ui  # noqa: E402
import simulator as sim  # noqa: E402
from src.risk.new_transaction import score_new_transaction  # noqa: E402

APP = str(APP_DIR / "app.py")
PAGE = "Simulador"
RUN = "Analisar transação"
SCENARIO_LABEL = dict(sim.SCENARIOS)
VERDICT, HISTORY, ENGINE = "sim_verdict", "sim_history", "sim_engine"
SIMULATION_NOTICE = ui.PROTOTYPE_NOTICE


def _button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def _texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
    parts += [w.value for w in at.warning] + [s.value for s in at.subheader] + [e.value for e in at.error]
    return "\n".join(parts)


def _form(at: AppTest) -> sim.FormValues:
    return sim.FormValues(
        customer_id=at.selectbox(key="sim_customer").value,
        city=at.selectbox(key="sim_city").value,
        amount=at.number_input(key="sim_amount").value,
        timestamp=datetime.combine(at.date_input(key="sim_date").value, at.time_input(key="sim_time").value),
        device_id=at.selectbox(key="sim_device").value,
        channel=at.selectbox(key="sim_channel").value[0],
        payment_method=at.selectbox(key="sim_channel").value[1],
    )


@pytest.fixture
def analysis_calls(monkeypatch):
    """Conta quantas vezes o motor é treinado de verdade (`run_analysis`)."""
    calls = {"n": 0}
    original = ra.build_risk_table

    def counting(session):
        calls["n"] += 1
        return original(session)

    monkeypatch.setattr(ra, "build_risk_table", counting)
    return calls


@pytest.fixture
def db(synthetic_db_file):
    engine = create_engine(f"sqlite:///{synthetic_db_file}")
    yield engine
    engine.dispose()


@pytest.fixture
def page(monkeypatch, synthetic_db_file, analysis_calls) -> AppTest:
    monkeypatch.setenv(ra.DB_ENV_VAR, str(synthetic_db_file))
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio(key="nav_product").set_value(PAGE).run()
    assert not at.exception
    return at


def _customer_ids(db) -> list[int]:
    """Ids na ordem do seletor (por nome). `set_value(id)` em vez de
    `select_index`: o AppTest grava o rótulo formatado quando há `format_func`.
    """
    with Session(db) as s:
        return sim.list_customers(s)["customer_id"].astype(int).tolist()


def _ctx_of(db, customer_id: int) -> sim.CustomerContext:
    with Session(db) as s:
        return sim.load_context(s, customer_id)


# ----------------------------------------------------------------------
# estado vazio: só o formulário
# ----------------------------------------------------------------------


def test_page_is_in_the_main_navigation_next_to_the_other_pages(page):
    assert "Simulador" in page.sidebar.radio(key="nav_product").options


def test_page_opens_with_the_form_and_no_verdict(page, analysis_calls):
    assert [s.label for s in page.selectbox] == [
        "Cliente (nome — cidade natal)",
        "Cidade da compra",
        "Dispositivo",
        "Canal e pagamento",
    ]
    assert len(page.number_input) == 1 and len(page.date_input) == 1 and len(page.time_input) == 1
    assert RUN in [b.label for b in page.button]
    assert "Nenhuma transação analisada" in _texts(page)
    assert VERDICT not in page.session_state and HISTORY not in page.session_state
    assert analysis_calls["n"] == 0


def test_empty_state_shows_no_result_elements(page):
    assert len(page.status) == 0
    assert len(page.get("plotly_chart")) == 0
    assert len(page.dataframe) == 0
    subheaders = [s.value for s in page.subheader]
    assert "Veredito do Sentinel" not in subheaders
    assert "Nenhuma transação analisada nesta sessão ainda." in _texts(page)
    assert SIMULATION_NOTICE in _texts(page)


def test_customer_selector_shows_name_and_home_city(page, db):
    with Session(db) as s:
        customers = sim.list_customers(s)
    row = customers.iloc[0]
    options = page.selectbox(key="sim_customer").options
    assert len(options) == 130
    assert f"{row['name']} (#{row['customer_id']}) — {row['home_city']}" in options[0]


def test_city_selector_uses_the_reference_cities(page):
    from src.synthetic.geo_reference import CITIES

    assert set(page.selectbox(key="sim_city").options) == {c.city for c in CITIES}


def test_device_selector_has_the_known_devices_plus_the_unrecognized_option(page, db):
    ctx = _ctx_of(db, page.selectbox(key="sim_customer").value)
    options = page.selectbox(key="sim_device").options
    assert len(options) == len(ctx.devices) + 1
    assert "Novo dispositivo, não reconhecido" in options[-1]
    assert all("compras" in o for o in options[:-1])


def test_missing_database_shows_guidance(monkeypatch, tmp_path):
    monkeypatch.setenv(ra.DB_ENV_VAR, str(tmp_path / "nao_existe.db"))
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio(key="nav_product").set_value(PAGE).run()
    assert not at.exception
    assert any("generate_mock_data.py" in e.value for e in at.error)
    assert RUN not in [b.label for b in at.button]


# ----------------------------------------------------------------------
# cenários: só preenchem o formulário
# ----------------------------------------------------------------------


@pytest.mark.parametrize("key", [k for k, _ in sim.SCENARIOS])
def test_scenario_button_fills_the_form_without_running_the_analysis(page, db, analysis_calls, key):
    customer = page.selectbox(key="sim_customer").value
    expected = sim.build_scenarios(_ctx_of(db, customer))[key]

    _button(page, SCENARIO_LABEL[key]).click().run()

    assert not page.exception
    assert _form(page) == expected  # preencheu com os valores calculados do histórico real
    assert VERDICT not in page.session_state and HISTORY not in page.session_state
    assert len(page.status) == 0 and len(page.get("plotly_chart")) == 0
    assert analysis_calls["n"] == 0  # nem treinou o motor, nem analisou


def test_scenarios_follow_the_selected_customer(page, db):
    page.selectbox(key="sim_customer").set_value(_customer_ids(db)[5]).run()
    cid = page.selectbox(key="sim_customer").value
    _button(page, SCENARIO_LABEL[sim.SCENARIO_IMPOSSIBLE_TRAVEL]).click().run()
    assert _form(page) == sim.build_scenarios(_ctx_of(db, cid))[sim.SCENARIO_IMPOSSIBLE_TRAVEL]
    assert _form(page).customer_id == cid


def test_changing_the_customer_resets_the_form_to_that_customers_normal_purchase(page, db):
    first = page.selectbox(key="sim_customer").value
    _button(page, SCENARIO_LABEL[sim.SCENARIO_ATYPICAL_AMOUNT]).click().run()
    page.selectbox(key="sim_customer").set_value(_customer_ids(db)[9]).run()
    cid = page.selectbox(key="sim_customer").value
    assert cid != first
    assert _form(page) == sim.build_scenarios(_ctx_of(db, cid))[sim.SCENARIO_NORMAL]


# ----------------------------------------------------------------------
# validação do horário
# ----------------------------------------------------------------------


def test_time_not_after_the_last_transaction_warns_and_blocks_the_analysis(page, db, analysis_calls):
    ctx = _ctx_of(db, page.selectbox(key="sim_customer").value)
    before = ctx.last.timestamp - timedelta(days=2)
    page.date_input(key="sim_date").set_value(before.date()).run()
    page.time_input(key="sim_time").set_value(time(9, 0)).run()

    warnings = [w.value for w in page.warning]
    assert any("posterior à última transação" in w and f"{ctx.last.timestamp:%d/%m/%Y %H:%M}" in w for w in warnings)
    assert _button(page, RUN).disabled
    assert VERDICT not in page.session_state and analysis_calls["n"] == 0


def test_time_equal_to_the_last_transaction_is_also_refused(page, db):
    ctx = _ctx_of(db, page.selectbox(key="sim_customer").value)
    last = ctx.last.timestamp
    page.date_input(key="sim_date").set_value(last.date()).run()
    page.time_input(key="sim_time").set_value(time(last.hour, last.minute)).run()
    # o formulário trabalha em minutos: o minuto cheio da última compra nunca é POSTERIOR a ela
    assert any("posterior à última transação" in w.value for w in page.warning)
    assert _button(page, RUN).disabled


def test_a_valid_time_clears_the_warning_and_enables_the_button(page, db):
    ctx = _ctx_of(db, page.selectbox(key="sim_customer").value)
    later = ctx.last.timestamp + timedelta(days=1)
    page.date_input(key="sim_date").set_value(later.date()).run()
    assert not any("posterior à última transação" in w.value for w in page.warning)
    assert not _button(page, RUN).disabled


# ----------------------------------------------------------------------
# o resultado só existe depois do clique
# ----------------------------------------------------------------------


def _analyze(page, key=None):
    if key:
        _button(page, SCENARIO_LABEL[key]).click().run()
    _button(page, RUN).click().run()
    assert not page.exception
    return page


def test_click_produces_the_verdict_with_live_reasoning(page, analysis_calls):
    _analyze(page, sim.SCENARIO_NEW_CITY)
    assert "Veredito do Sentinel" in [s.value for s in page.subheader]
    status = page.status[0]
    assert status.state == "complete"
    lines = [m.value for m in status.markdown]
    text = "\n".join(lines)
    for label in (
        "Treinando o motor de risco", "Recuperando o histórico do cliente", "Calculando as features da transação",
        "Distância da residência", "Velocidade implícita desde a última compra", "Horário em relação à janela habitual",
        "Dispositivo", "Cidade", "Aplicando o modelo de ML", "Aplicando as regras", "Combinando no Risk Score",
        "Classificando por nível",
    ):
        assert f"**{label}" in text, label
    assert "NUNCA visitada" in text  # valor real do cenário, não texto fixo
    assert analysis_calls["n"] == 1  # sem motor na sessão: treinou como uma das etapas


def test_reasoning_lines_escape_the_currency_sign(page):
    """Duas linhas "R$ x ... R$ y" viram LaTeX no markdown se o $ não for escapado."""
    _analyze(page, sim.SCENARIO_NORMAL)
    lines = [m.value for m in page.status[0].markdown]
    value_line = next(line for line in lines if "Valor em relação à média" in line)
    escaped = "R" + chr(92) + "$"
    assert value_line.count(escaped) == 2 and "R$" not in value_line.replace(escaped, "")


def test_verdict_has_gauge_with_the_30_and_70_bands(page):
    _analyze(page, sim.SCENARIO_ATYPICAL_AMOUNT)
    result = page.session_state[VERDICT]["result"]
    figures = [json.loads(el.proto.spec)["data"][0] for el in page.get("plotly_chart")]
    gauge = next(f for f in figures if f.get("type") == "indicator")
    assert gauge["value"] == result.assessment.risk_score
    assert [s["range"] for s in gauge["gauge"]["steps"]] == [[0, 30], [30, 70], [70, 100]]
    assert gauge["gauge"]["axis"]["tickvals"] == [0, 30, 70, 100]


def test_verdict_shows_level_action_scores_reasons_map_and_customer_history(page):
    _analyze(page, sim.SCENARIO_ATYPICAL_AMOUNT)
    entry = page.session_state[VERDICT]
    a, ctx = entry["result"].assessment, entry["result"].context
    banner = next(m.value for m in page.markdown if 'class="sx-verdict"' in m.value)
    assert ui.ACTION_LABEL[a.action] in banner and f"nível {a.level}" in banner
    cards = " ".join(m.value for m in page.markdown if "metric-card" in m.value)
    assert "Score do ML" in cards and "Score das regras" in cards
    assert f">{a.ml_score:.0f}<" in cards and f">{a.behavior_score:.0f}<" in cards
    text = _texts(page)
    for reason in a.reasons:
        assert reason in text
    assert "Histórico do cliente" in text and ctx.name in text and "Cidades conhecidas" in text
    assert len(page.get("plotly_chart")) == 2  # gauge + mapa


def test_map_has_residence_purchase_and_the_line_between_them(page):
    _analyze(page, sim.SCENARIO_NEW_CITY)
    result = page.session_state[VERDICT]["result"]
    figures = [json.loads(el.proto.spec)["data"] for el in page.get("plotly_chart")]
    traces = next(data for data in figures if data[0].get("mode") == "lines")
    assert [t["mode"] for t in traces] == ["lines", "markers", "markers"]
    assert traces[1]["lat"] == [result.context.home_lat] and traces[1]["lon"] == [result.context.home_lon]
    assert traces[2]["lat"] == [result.city.lat] and traces[2]["lon"] == [result.city.lon]


def test_impossible_travel_scenario_fires_the_impossible_travel_signal(page):
    _analyze(page, sim.SCENARIO_IMPOSSIBLE_TRAVEL)
    result = page.session_state[VERDICT]["result"]
    assert result.features["is_impossible_travel"] is True
    assert result.assessment.level != "baixo"
    text = "\n".join(m.value for m in page.status[0].markdown)
    assert "IMPOSSÍVEL" in text
    assert any("Viagem impossível" in r for r in result.assessment.reasons)


def test_combined_signals_scenario_shows_the_three_signals_in_the_verdict(page):
    _analyze(page, sim.SCENARIO_COMBINED)
    result = page.session_state[VERDICT]["result"]
    f = result.features
    assert f["new_device"] and f["unusual_hour"] > 0 and f["amount_deviation"] == pytest.approx(9.0, abs=0.01)
    text = _texts(page)
    for reason in result.assessment.reasons:
        assert reason in text
    for expected in ("Dispositivo nunca usado", "fora da janela habitual", "acima da média"):
        assert expected in text
    assert result.assessment.level != "baixo"


def test_page_verdict_equals_score_new_transaction(page, db):
    _analyze(page, sim.SCENARIO_NEW_CITY)
    form = _form(page)
    entry = page.session_state[VERDICT]
    engine = page.session_state[ENGINE]["engine"]
    city = sim.CITY_BY_NAME[form.city]
    with Session(db) as s:
        reference = score_new_transaction(
            s, engine, customer_id=form.customer_id, amount=form.amount, lat=city.lat, lon=city.lon,
            timestamp=form.timestamp, device_id=None if form.device_id == sim.NEW_DEVICE else form.device_id, city=form.city,
            channel=form.channel, payment_method=form.payment_method,
        )
    assert entry["result"].assessment == reference


def test_changing_the_form_after_a_verdict_does_not_rerun_the_analysis(page, analysis_calls):
    _analyze(page, sim.SCENARIO_NEW_CITY)
    page.number_input(key="sim_amount").set_value(77.0).run()
    assert analysis_calls["n"] == 1
    assert len(page.session_state[HISTORY]) == 1


def test_scenario_click_after_a_verdict_clears_it_until_the_next_analysis(page):
    _analyze(page, sim.SCENARIO_NEW_CITY)
    _button(page, SCENARIO_LABEL[sim.SCENARIO_NORMAL]).click().run()
    assert VERDICT not in page.session_state
    assert "Veredito do Sentinel" not in [s.value for s in page.subheader]
    assert len(page.session_state[HISTORY]) == 1  # o histórico da sessão permanece


# ----------------------------------------------------------------------
# motor: treinar uma vez ou reaproveitar da Central
# ----------------------------------------------------------------------


def test_engine_is_trained_once_and_reused_by_later_analyses(page, analysis_calls):
    _analyze(page, sim.SCENARIO_NORMAL)
    assert analysis_calls["n"] == 1
    _analyze(page, sim.SCENARIO_ATYPICAL_AMOUNT)
    assert analysis_calls["n"] == 1
    assert "Motor de risco" in "\n".join(m.value for m in page.status[0].markdown)


def test_engine_trained_by_the_central_is_reused(page, analysis_calls):
    page.sidebar.radio(key="nav_product").set_value("Central de Transações").run()
    _button(page, "Executar análise de risco").click().run()
    assert analysis_calls["n"] == 1

    page.sidebar.radio(key="nav_product").set_value(PAGE).run()
    _analyze(page, sim.SCENARIO_NEW_DEVICE)
    assert analysis_calls["n"] == 1  # não treinou de novo
    assert ENGINE not in page.session_state
    text = "\n".join(m.value for m in page.status[0].markdown)
    assert "reaproveitado da Central de Transações" in text and "Treinando o motor" not in text
    assert page.session_state[VERDICT]["result"].assessment == _central_assessment(page)


def _central_assessment(page):
    """O mesmo cenário avaliado diretamente com o motor da Central."""
    central = page.session_state["risk_analysis"]
    result = page.session_state[VERDICT]["result"]
    f = result.form
    with Session(create_engine(f"sqlite:///{ra.resolve_db_path()}")) as s:
        return sim.simulate(s, central.engine, f).assessment


# ----------------------------------------------------------------------
# histórico da sessão
# ----------------------------------------------------------------------


def test_session_history_lists_every_analysis_newest_first(page, db):
    _analyze(page, sim.SCENARIO_NORMAL)
    first = page.session_state[VERDICT]["result"]
    _analyze(page, sim.SCENARIO_NEW_CITY)
    second = page.session_state[VERDICT]["result"]

    table = page.dataframe[-1].value
    assert len(table) == 2
    for column in ("Cliente", "Valor", "Cidade", "Risk Score", "Nível", "Ação"):
        assert column in table.columns
    assert table["Cidade"].tolist() == [second.city.city, first.city.city]  # mais recente primeiro
    assert table["Risk Score"].tolist() == [second.assessment.risk_score, first.assessment.risk_score]
    assert table["Nível"].tolist() == [second.assessment.level, first.assessment.level]
    assert table["Ação"].tolist() == [second.assessment.action, first.assessment.action]
    assert table["Cliente"].tolist() == [second.context.name, first.context.name]


def test_clear_history_button(page):
    _analyze(page, sim.SCENARIO_NORMAL)
    _button(page, "Limpar histórico").click().run()
    assert HISTORY not in page.session_state
    assert "Nenhuma transação analisada nesta sessão ainda." in _texts(page)


def test_history_is_session_only_and_never_written_to_the_database(page, db):
    from sqlalchemy import func, select

    from src.synthetic.models import Transaction

    with Session(db) as s:
        before = s.scalar(select(func.count()).select_from(Transaction))
    _analyze(page, sim.SCENARIO_IMPOSSIBLE_TRAVEL)
    _analyze(page, sim.SCENARIO_NORMAL)
    with Session(db) as s:
        assert s.scalar(select(func.count()).select_from(Transaction)) == before
    assert len(page.session_state[HISTORY]) == 2


def test_free_form_input_is_analyzed_like_a_scenario(page, db):
    """O formulário vale por si: valores digitados, não só cenários."""
    ctx = _ctx_of(db, page.selectbox(key="sim_customer").value)
    later = ctx.last.timestamp + timedelta(days=2)
    page.number_input(key="sim_amount").set_value(4321.09).run()
    page.selectbox(key="sim_city").select("Manaus").run()
    page.date_input(key="sim_date").set_value(date(later.year, later.month, later.day)).run()
    page.time_input(key="sim_time").set_value(time(3, 30)).run()
    _button(page, RUN).click().run()
    assert not page.exception
    result = page.session_state[VERDICT]["result"]
    assert result.form.amount == 4321.09 and result.city.city == "Manaus"
    assert result.form.timestamp.hour == 3 and result.form.timestamp.minute == 30
