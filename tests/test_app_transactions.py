"""Testes da página Central de Transações (app/views/transactions.py), com o
`AppTest` do Streamlit — sem navegador.

O requisito central é que NADA apareça pré-calculado: a página abre vazia e
só o clique em "Executar análise de risco" produz resultados. Por isso há
testes que contam quantas vezes a análise de verdade roda (a etapa de
features, que só acontece dentro de `run_analysis`).

O banco usado é um arquivo temporário (fixture `synthetic_db_file`), apontado
por `SENTINEL_DB_PATH` — nunca o data/synthetic/sentinel.db real.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import risk_analysis as ra  # noqa: E402
import ui  # noqa: E402

APP = str(APP_DIR / "app.py")
RUN_LABEL = "Executar análise de risco"
CLEAR_LABEL = "Limpar resultado"
PAGE = "Central de Transações"
SIMULATION_NOTICE = "dados simulados; valida a lógica do sistema, não o desempenho em fraude real"


def _button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def _has_button(at: AppTest, label: str) -> bool:
    return any(b.label == label for b in at.button)


def _texts(at: AppTest) -> str:
    """Todo o texto de markdown/caption/info/warning/subheader da página."""
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
    parts += [w.value for w in at.warning] + [s.value for s in at.subheader] + [e.value for e in at.error]
    return "\n".join(parts)


@pytest.fixture
def analysis_calls(monkeypatch):
    """Conta quantas vezes a análise REAL roda (a etapa de features só roda
    dentro de `run_analysis`), sem alterar o comportamento.
    """
    calls = {"n": 0}
    original = ra.build_risk_table

    def counting(session):
        calls["n"] += 1
        return original(session)

    monkeypatch.setattr(ra, "build_risk_table", counting)
    return calls


@pytest.fixture
def page(monkeypatch, synthetic_db_file, analysis_calls) -> AppTest:
    """A Central de Transações recém-aberta (sem clicar em nada)."""
    monkeypatch.setenv(ra.DB_ENV_VAR, str(synthetic_db_file))
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio(key="nav_product").set_value(PAGE).run()
    assert not at.exception
    return at


@pytest.fixture
def analyzed(page) -> AppTest:
    """A mesma página, depois do clique em "Executar análise de risco", com o
    filtro "Conjunto" limpo (todas as linhas) — o padrão "teste" tem teste próprio.
    """
    _button(page, RUN_LABEL).click().run()
    assert not page.exception
    page.multiselect(key="tx_f_split").set_value([]).run()
    return page


def test_conjunto_filter_opens_on_the_test_customers(page):
    """Padrão = teste: a tabela abre só com clientes que o modelo nunca viu."""
    _button(page, RUN_LABEL).click().run()
    assert page.multiselect(key="tx_f_split").value == ["teste"]
    frame = page.session_state["risk_analysis"].frame
    table = page.dataframe[0].value
    assert len(table) == (frame["conjunto"] == "teste").sum() < len(frame)
    assert set(table["Conjunto"]) == {"teste"}


def _table(at: AppTest):
    return at.dataframe[0].value


# ----------------------------------------------------------------------
# estado vazio: nada pré-calculado
# ----------------------------------------------------------------------


def test_page_opens_empty_with_only_the_run_button(page, analysis_calls):
    assert _has_button(page, RUN_LABEL)
    assert not _has_button(page, CLEAR_LABEL)
    assert "Nenhuma análise executada" in _texts(page)
    assert "risk_analysis" not in page.session_state
    assert analysis_calls["n"] == 0


def test_empty_state_shows_no_results_at_all(page):
    assert len(page.dataframe) == 0
    assert len(page.status) == 0
    assert len(page.get("plotly_chart")) == 0
    assert len(page.multiselect) == 0 and len(page.slider) == 0
    subheaders = [s.value for s in page.subheader]
    assert subheaders == []
    text = _texts(page)
    for result_only in ("Resumo da análise", "Eficiência", "Fraudes capturadas", "Falsos positivos", "Nível alto"):
        assert result_only not in text


def test_empty_state_lists_the_steps_that_will_run(page):
    text = _texts(page)
    for label in ra.STEP_LABELS:
        assert label in text


def test_simulation_notice_is_visible_from_the_start(page):
    assert SIMULATION_NOTICE in _texts(page)


def test_opening_the_page_does_not_run_the_analysis(page, analysis_calls):
    page.run()  # reexecuções sem clique (como qualquer interação) continuam vazias
    assert analysis_calls["n"] == 0
    assert len(page.dataframe) == 0


def test_missing_database_shows_guidance_instead_of_the_button(monkeypatch, tmp_path):
    monkeypatch.setenv(ra.DB_ENV_VAR, str(tmp_path / "nao_existe.db"))
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio(key="nav_product").set_value(PAGE).run()
    assert not at.exception
    assert any("generate_mock_data.py" in e.value for e in at.error)
    assert not _has_button(at, RUN_LABEL)


# ----------------------------------------------------------------------
# depois do clique
# ----------------------------------------------------------------------


def test_click_runs_the_analysis_once_and_stores_the_result(analyzed, analysis_calls):
    assert analysis_calls["n"] == 1
    result = analyzed.session_state["risk_analysis"]
    assert isinstance(result, ra.AnalysisResult)
    assert _has_button(analyzed, CLEAR_LABEL)


def test_status_shows_every_step_with_its_time(analyzed):
    assert len(analyzed.status) == 1
    status = analyzed.status[0]
    assert status.state == "complete"
    assert status.label.startswith("Análise concluída em")
    lines = [m.value for m in status.markdown]
    assert len(lines) == len(ra.STEP_LABELS)
    for label, line in zip(ra.STEP_LABELS, lines):
        assert label in line
        assert line.rstrip().endswith(("ms", " s"))


def test_summary_cards_match_the_analysis(analyzed):
    result = analyzed.session_state["risk_analysis"]
    summary = ra.summarize(result.frame)
    cards = " ".join(m.value for m in analyzed.markdown if "metric-card" in m.value)
    assert "Transações analisadas" in cards
    assert f">{summary['total']:,}<".replace(",", ".") in cards
    for level in ra.LEVELS:
        assert f"Nível {level}" in cards
        assert f">{int(summary['por_nivel'].loc[level, 'quantidade']):,}<".replace(",", ".") in cards
    for action in ("Aprovar", "Revisar", "Bloquear"):
        assert f"Ação: {action}" in cards
    assert "%" in cards


def test_table_is_sorted_by_risk_score_and_has_the_requested_columns(analyzed):
    table = _table(analyzed)
    assert table.columns.tolist() == [
        "ID", "Cliente", "Valor", "Cidade", "Data/hora", "Risk Score", "Nível", "Ação", "Conjunto",
    ]
    assert len(table) == len(analyzed.session_state["risk_analysis"].frame)
    assert table["Risk Score"].is_monotonic_decreasing
    assert table["Risk Score"].iloc[0] == table["Risk Score"].max()


def test_every_filter_widget_is_there(analyzed):
    assert {m.key for m in analyzed.multiselect} == {
        "tx_f_level", "tx_f_action", "tx_f_profile", "tx_f_customer", "tx_f_split",
    }
    assert [s.key for s in analyzed.slider] == ["tx_f_amount"]


def test_efficiency_block_shows_the_simulation_notice_and_only_test_numbers(analyzed):
    result = analyzed.session_state["risk_analysis"]
    eff = ra.efficiency(result)
    m = eff["metrics"]
    text = _texts(analyzed)
    assert "Eficiência medida nesta análise" in text
    assert any(SIMULATION_NOTICE in w.value for w in analyzed.warning)
    assert f"{eff['customers']} clientes de TESTE" in text
    cards = " ".join(x.value for x in analyzed.markdown if "metric-card" in x.value)
    assert f"{m['tp']} de {eff['frauds']}" in cards
    assert f">{m['fp']}<" in cards  # falsos positivos
    assert f">{m['fn']}<" in cards  # fraudes perdidas
    assert len(analyzed.get("plotly_chart")) == 0  # sem transação aberta não há mapa; a matriz é HTML
    matrix = next(x.value for x in analyzed.markdown if 'class="sx-matrix"' in x.value)
    for value in (m["tn"], m["fp"], m["fn"], m["tp"]):
        assert f'<div class="n">{ui.fmt_int(value)}</div>' in matrix
    for label in ("Acertou (aprovou legítima)", "Falso positivo", "Fraude perdida", "Fraude capturada"):
        assert label in matrix


def test_clear_button_returns_to_the_empty_state(analyzed):
    _button(analyzed, CLEAR_LABEL).click().run()
    assert not analyzed.exception
    assert "risk_analysis" not in analyzed.session_state
    assert len(analyzed.dataframe) == 0
    assert "Nenhuma análise executada" in _texts(analyzed)


# ----------------------------------------------------------------------
# filtros (não reexecutam a análise)
# ----------------------------------------------------------------------


def _frame(at: AppTest):
    return at.session_state["risk_analysis"].frame


def test_level_filter_narrows_the_table_without_rerunning_the_analysis(analyzed, analysis_calls):
    expected = int((_frame(analyzed)["level"] == "alto").sum())
    analyzed.multiselect(key="tx_f_level").set_value(["alto"]).run()
    assert not analyzed.exception
    table = _table(analyzed)
    assert len(table) == expected > 0
    assert (table["Nível"] == "alto").all()
    assert f"Mostrando {expected} de" in _texts(analyzed)
    assert analysis_calls["n"] == 1


def test_action_filter(analyzed):
    expected = int((_frame(analyzed)["action"] == "revisar").sum())
    analyzed.multiselect(key="tx_f_action").set_value(["revisar"]).run()
    table = _table(analyzed)
    assert len(table) == expected > 0
    assert (table["Ação"] == "revisar").all()


def test_profile_filter(analyzed):
    frame = _frame(analyzed)
    expected = int((frame["profile"] == "viajante").sum())
    analyzed.multiselect(key="tx_f_profile").set_value(["viajante"]).run()
    assert len(_table(analyzed)) == expected > 0
    assert set(_table(analyzed)["ID"]) == set(frame[frame["profile"] == "viajante"]["transaction_id"])


def test_customer_filter(analyzed):
    frame = _frame(analyzed)
    customer = int(frame["customer_id"].iloc[0])
    analyzed.multiselect(key="tx_f_customer").set_value([customer]).run()
    table = _table(analyzed)
    assert len(table) == int((frame["customer_id"] == customer).sum())
    assert set(table["Cliente"]) == {frame[frame["customer_id"] == customer]["customer"].iloc[0]}


def test_split_filter_keeps_only_test_customers(analyzed):
    frame = _frame(analyzed)
    analyzed.multiselect(key="tx_f_split").set_value([ra.SPLIT_TEST]).run()
    table = _table(analyzed)
    assert len(table) == int((frame["conjunto"] == ra.SPLIT_TEST).sum())
    assert (table["Conjunto"] == ra.SPLIT_TEST).all()


def test_amount_range_filter(analyzed):
    frame = _frame(analyzed)
    analyzed.slider(key="tx_f_amount").set_value((100.0, 300.0)).run()
    table = _table(analyzed)
    assert len(table) == int(frame["amount"].between(100.0, 300.0).sum()) > 0
    assert table["Valor"].between(100.0, 300.0).all()


def test_filters_combine_and_empty_result_is_handled(analyzed, analysis_calls):
    analyzed.multiselect(key="tx_f_level").set_value(["alto"]).run()
    analyzed.multiselect(key="tx_f_profile").set_value(["fixo"]).run()
    table = _table(analyzed)
    assert (table["Nível"] == "alto").all()
    frame = _frame(analyzed)
    assert len(table) == int(((frame["level"] == "alto") & (frame["profile"] == "fixo")).sum())

    analyzed.slider(key="tx_f_amount").set_value((5.0, 5.0)).run()  # nenhuma transação alto de R$ 5
    assert not analyzed.exception
    assert len(_table(analyzed)) == 0
    assert analysis_calls["n"] == 1


# ----------------------------------------------------------------------
# painel de investigação
# ----------------------------------------------------------------------


def _pick(at: AppTest, level: str) -> int:
    frame = _frame(at)
    return int(frame[frame["level"] == level]["transaction_id"].iloc[0])


def test_no_panel_until_a_transaction_is_chosen(analyzed):
    assert "Selecione uma transação na tabela" in _texts(analyzed)
    assert not any("Investigação da transação" in s.value for s in analyzed.subheader)


def test_lookup_by_id_opens_the_panel_with_everything_the_investigator_needs(analyzed, analysis_calls):
    tx_id = _pick(analyzed, "alto")
    analyzed.text_input(key="tx_lookup").set_value(str(tx_id)).run()
    assert not analyzed.exception
    assert f"Investigação da transação #{tx_id}" in [s.value for s in analyzed.subheader]

    row = _frame(analyzed).set_index("transaction_id").loc[tx_id]
    text = _texts(analyzed)
    for expected in (
        row["customer"],
        row["city"],
        row["home_city"],
        "Distância da residência",
        "Dispositivo",
        "Data e horário",
        "Local habitual",
        "Motivos do score",
    ):
        assert expected in text, expected

    cards = " ".join(m.value for m in analyzed.markdown if "metric-card" in m.value)
    assert "Risk Score final" in cards and "Score do ML" in cards and "Score das regras" in cards
    assert f"nível {row['level']}" in cards
    assert len(analyzed.get("plotly_chart")) == 1  # o mapa (a matriz de confusão é HTML)
    assert analysis_calls["n"] == 1


def test_panel_lists_the_reasons_the_engine_gives(analyzed):
    tx_id = _pick(analyzed, "alto")
    analyzed.text_input(key="tx_lookup").set_value(str(tx_id)).run()
    reasons = ra.investigate(analyzed.session_state["risk_analysis"], tx_id).assessment.reasons
    assert reasons
    text = _texts(analyzed)
    for reason in reasons:
        assert reason in text


def test_panel_shows_the_amount_and_the_customer_average_in_the_fact_sheet(analyzed):
    """A ficha é HTML (o bloco não passa pelo parser de fórmulas), então o `R$`
    vai sem barra — escapar mostraria a barra na tela.
    """
    tx_id = _pick(analyzed, "alto")
    analyzed.text_input(key="tx_lookup").set_value(str(tx_id)).run()
    row = _frame(analyzed).set_index("transaction_id").loc[tx_id]
    sheet = next(m.value for m in analyzed.markdown if "Ficha da transação" in m.value)
    assert ui.fmt_brl(row["amount"]) in sheet
    assert f"média habitual do cliente: {ui.fmt_brl(row['avg_amount_baseline'])}" in sheet
    assert chr(92) + "$" not in sheet


def test_unknown_or_invalid_id_warns_instead_of_breaking(analyzed):
    for bad in ("999999999", "abc"):
        analyzed.text_input(key="tx_lookup").set_value(bad).run()
        assert not analyzed.exception
        assert any("Não encontrei a transação" in w.value for w in analyzed.warning)
        assert not any("Investigação da transação" in s.value for s in analyzed.subheader)


def test_map_has_residence_purchase_and_the_line_between_them(analyzed):
    tx_id = _pick(analyzed, "alto")
    analyzed.text_input(key="tx_lookup").set_value(str(tx_id)).run()
    figures = [json.loads(el.proto.spec)["data"] for el in analyzed.get("plotly_chart")]
    traces = next(data for data in figures if data[0].get("mode") == "lines")  # a outra é a matriz de confusão
    assert [t["mode"] for t in traces] == ["lines", "markers", "markers"]
    row = _frame(analyzed).set_index("transaction_id").loc[tx_id]
    assert traces[1]["lat"] == [row["home_lat"]] and traces[1]["lon"] == [row["home_lon"]]
    assert traces[2]["lat"] == [row["tx_lat"]] and traces[2]["lon"] == [row["tx_lon"]]
    assert traces[0]["lat"] == [row["home_lat"], row["tx_lat"]]
    assert "Residência" in traces[1]["name"] and "Compra" in traces[2]["name"]


# ----------------------------------------------------------------------
# seleção de linha da tabela (o AppTest não simula o clique: o `st.dataframe`
# é substituído por um dublê que devolve a seleção, e o resto roda de verdade)
# ----------------------------------------------------------------------


def _select_row(monkeypatch, position: int):
    original = st.dataframe

    def fake(*args, **kwargs):
        original(*args, **kwargs)
        return SimpleNamespace(selection=SimpleNamespace(rows=[position], columns=[], cells=[]))

    monkeypatch.setattr(st, "dataframe", fake)


def test_selecting_a_row_opens_the_panel_for_that_transaction(monkeypatch, analyzed):
    _select_row(monkeypatch, 3)
    analyzed.run()
    assert not analyzed.exception
    expected = int(_table(analyzed)["ID"].iloc[3])
    assert f"Investigação da transação #{expected}" in [s.value for s in analyzed.subheader]


def test_selection_follows_the_filtered_table_not_the_full_one(monkeypatch, analyzed):
    analyzed.multiselect(key="tx_f_level").set_value(["médio"]).run()
    _select_row(monkeypatch, 0)
    analyzed.run()
    first_medium = int(_table(analyzed)["ID"].iloc[0])
    assert (_frame(analyzed).set_index("transaction_id").loc[first_medium, "level"]) == "médio"
    assert f"Investigação da transação #{first_medium}" in [s.value for s in analyzed.subheader]


def test_a_stale_selection_beyond_the_filtered_rows_is_ignored(monkeypatch, analyzed):
    analyzed.multiselect(key="tx_f_level").set_value(["alto"]).run()
    _select_row(monkeypatch, 10_000)
    analyzed.run()
    assert not analyzed.exception
    assert not any("Investigação da transação" in s.value for s in analyzed.subheader)


def test_typed_id_wins_over_the_selected_row(monkeypatch, analyzed):
    _select_row(monkeypatch, 0)
    typed = _pick(analyzed, "médio")
    analyzed.text_input(key="tx_lookup").set_value(str(typed)).run()
    assert f"Investigação da transação #{typed}" in [s.value for s in analyzed.subheader]


# ----------------------------------------------------------------------
# formatação pt-BR
# ----------------------------------------------------------------------


def test_ptbr_formatting_helpers():
    import ui as view

    assert view.fmt_int(6912) == "6.912"
    assert view.fmt_pct(0.0371) == "3,7%"
    assert view.fmt_pct(0.3, 0) == "30%"
    assert view.fmt_brl(1234.5) == "R$ 1.234,50"
    assert view.md_brl(1234.5) == "R\\$ 1.234,50"
    assert view.fmt_km(2317.4) == "2.317 km"
    assert view.fmt_seconds(0.0642) == "64 ms"
    assert view.fmt_seconds(1.5) == "1,50 s"
