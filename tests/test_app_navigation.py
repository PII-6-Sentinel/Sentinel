"""Testes da arquitetura de navegação do app (app/app.py).

Usam o `AppTest` do Streamlit — executam o script sem navegador. Só a
seção do produto é exercitada sem dados: as páginas científicas carregam o
dataset do Kaggle, então os testes que entram nelas só rodam se o CSV
existir na máquina (nunca é exigido para rodar a suíte).
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app" / "app.py")
KAGGLE_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "creditcard.csv"

SCIENTIFIC_ENTRY = "🔬 Ver Metodologia Científica (Kaggle) →"
SCIENTIFIC_PAGES = ["Análise Exploratória", "Comparação de Modelos", "Análise de Threshold", "Demo (dataset Kaggle)"]


def _run() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


def _button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def test_main_navigation_only_lists_operational_pages():
    at = _run()
    assert not at.exception
    assert at.sidebar.radio(key="nav_product").options == ["Visão Geral", "Central de Transações"]


def test_scientific_pages_are_not_in_the_main_navigation():
    at = _run()
    options = at.sidebar.radio(key="nav_product").options
    for scientific_page in SCIENTIFIC_PAGES:
        assert scientific_page not in options


def test_single_discreet_entry_point_to_the_scientific_section():
    at = _run()
    entries = [b for b in at.button if "Metodologia Científica" in b.label]
    assert [b.label for b in entries] == [SCIENTIFIC_ENTRY]


def test_overview_states_the_product_proposal_and_where_the_science_lives():
    at = _run()
    text = " ".join(m.value for m in at.markdown) + " ".join(i.value for i in at.info)
    assert "plataforma de análise de risco transacional para" in text
    assert "instituições financeiras" in text
    assert "Metodologia Científica (Kaggle)" in text


def test_overview_does_not_need_the_kaggle_dataset(monkeypatch):
    """A Visão Geral é a página inicial do produto — não pode depender do CSV."""
    import pandas as pd

    def boom(*args, **kwargs):
        raise AssertionError("a Visão Geral tentou carregar o dataset do Kaggle")

    monkeypatch.setattr(pd, "read_csv", boom)
    at = _run()
    assert not at.exception


def test_transactions_page_is_a_clear_placeholder():
    at = _run()
    at.sidebar.radio(key="nav_product").set_value("Central de Transações").run()

    assert not at.exception
    assert any("Em construção" in w.value and "Risk Engine" in w.value for w in at.warning)


def test_product_section_does_not_show_the_scientific_intro():
    at = _run()
    assert not any("validação científica dos modelos" in i.value for i in at.info)


@pytest.mark.skipif(not KAGGLE_CSV.exists(), reason="dataset do Kaggle não baixado")
def test_scientific_section_opens_with_role_header_and_can_go_back():
    at = _run()
    _button(at, SCIENTIFIC_ENTRY).click().run()

    assert not at.exception
    assert at.session_state["section"] == "scientific"
    assert any(
        "Esta seção documenta a validação científica dos modelos usando o dataset "
        "público Kaggle. Ela fundamenta metodologicamente o sistema, mas não é o "
        "produto em si." in i.value
        for i in at.info
    )
    assert at.sidebar.radio(key="nav_scientific").options == SCIENTIFIC_PAGES

    _button(at, "← Voltar ao produto").click().run()
    assert at.session_state["section"] == "product"
    assert not at.exception
