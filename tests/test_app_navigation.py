"""Testes da arquitetura de navegação do app (app/app.py).

Usam o `AppTest` do Streamlit — executam o script sem navegador. O app só
tem o produto: a seção científica (Kaggle) saiu do app na Etapa 13 e vive
no repositório (`src/`, `notebooks/`).
"""

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import risk_analysis as ra  # noqa: E402

APP = str(APP_DIR / "app.py")
SCIENTIFIC_PAGES = ["Análise Exploratória", "Comparação de Modelos", "Análise de Threshold", "Demo (dataset Kaggle)"]


def _run() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


def _button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def test_main_navigation_only_lists_operational_pages():
    at = _run()
    assert not at.exception
    assert at.sidebar.radio(key="nav_product").options == [
        "Visão Geral",
        "Central de Transações",
        "Simulador",
        "Monitor ao Vivo",
    ]


def test_scientific_pages_are_not_in_the_app():
    at = _run()
    options = at.sidebar.radio(key="nav_product").options
    for scientific_page in SCIENTIFIC_PAGES:
        assert scientific_page not in options


def test_no_entry_point_to_a_scientific_section():
    at = _run()
    assert not [b for b in at.button if "Metodologia Científica" in b.label]
    assert "section" not in at.session_state


def test_overview_states_the_product_proposal_and_where_the_science_lives():
    at = _run()
    text = " ".join(m.value for m in at.markdown) + " ".join(i.value for i in at.info)
    # o negrito cobre "Sentinel é uma plataforma de análise de risco transacional"
    assert "Sentinel é uma plataforma de análise de risco transacional" in text
    assert "instituições financeiras" in text
    assert "não decide sozinho" in text
    assert "fora do app" in text  # a validação científica fica no repositório


def test_overview_does_not_need_the_kaggle_dataset(monkeypatch):
    """A Visão Geral é a página inicial do produto — não pode depender do CSV."""
    import pandas as pd

    def boom(*args, **kwargs):
        raise AssertionError("a Visão Geral tentou carregar o dataset do Kaggle")

    monkeypatch.setattr(pd, "read_csv", boom)
    at = _run()
    assert not at.exception


def test_transactions_page_is_the_central_not_a_placeholder(monkeypatch, synthetic_db_file):
    """A Central de Transações abre no estado vazio, pronta para executar a
    análise — o detalhe do comportamento fica em tests/test_app_transactions.py.
    """
    monkeypatch.setenv("SENTINEL_DB_PATH", str(synthetic_db_file))
    at = _run()
    at.sidebar.radio(key="nav_product").set_value("Central de Transações").run()

    assert not at.exception
    assert not any("Em construção" in w.value for w in at.warning)
    assert "Executar análise de risco" in [b.label for b in at.button]
    assert any("Nenhuma análise executada" in m.value for m in at.markdown)


def test_transactions_page_opens_without_the_simulator_state(monkeypatch, synthetic_db_file):
    monkeypatch.setenv("SENTINEL_DB_PATH", str(synthetic_db_file))
    at = _run()
    at.sidebar.radio(key="nav_product").set_value("Simulador").run()

    assert not at.exception
    assert [s.value for s in at.subheader if s.value == "Veredito do Sentinel"] == []
    assert "Analisar transação" in [b.label for b in at.button]
    assert any("Nenhuma transação analisada" in m.value for m in at.markdown)


def test_overview_ctas_navigate_to_the_central_and_to_the_simulator(monkeypatch, synthetic_db_file):
    monkeypatch.setenv("SENTINEL_DB_PATH", str(synthetic_db_file))
    at = _run()
    _button(at, "Abrir Central de Transações").click().run()
    assert not at.exception
    assert at.sidebar.radio(key="nav_product").value == "Central de Transações"
    assert any("Nenhuma análise executada" in m.value for m in at.markdown)

    at = _run()
    _button(at, "Testar no Simulador").click().run()
    assert not at.exception
    assert at.sidebar.radio(key="nav_product").value == "Simulador"
    assert "Analisar transação" in [b.label for b in at.button]


def test_overview_lists_every_product_component_as_ready():
    at = _run()
    status = at.dataframe[0].value
    state = dict(zip(status["Componente"], status["Situação"]))
    assert state["Risk Engine"] == "Pronto"
    assert state["Central de Transações"] == "Pronto"
    assert state["Simulador ao vivo"] == "Pronto"
    assert "Em construção" not in state.values()



def test_outdated_database_shows_how_to_fix_it(monkeypatch, tmp_path):
    """Banco de antes da Etapa 12 (sem as colunas de canal): as páginas que
    usam o banco explicam como regenerar, em vez de quebrar com erro de SQL.
    """
    import sqlite3

    path = tmp_path / "antigo.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("CREATE TABLE transactions (id INTEGER PRIMARY KEY, amount REAL)")
    assert ra.database_problem(path) == ra.OUTDATED_DB_MESSAGE
    assert ra.database_problem(tmp_path / "nao_existe.db") == ra.MISSING_DB_MESSAGE

    monkeypatch.setenv(ra.DB_ENV_VAR, str(path))
    for page in ("Central de Transações", "Simulador", "Monitor ao Vivo"):
        at = _run()
        at.sidebar.radio(key="nav_product").set_value(page).run()
        assert not at.exception, page
        assert any("desatualizado" in e.value and "--force" in e.value for e in at.error), page
