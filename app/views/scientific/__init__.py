"""Seção "Metodologia Científica (Kaggle)" — atrás de um único ponto de
entrada na sidebar, fora da navegação principal do produto.

Documenta a validação científica dos modelos (baseline estatístico,
Regressão Logística, Isolation Forest) sobre o dataset público do Kaggle.
É a base metodológica do Sentinel, não o produto: por isso não compete
por atenção com as páginas operacionais.
"""

import streamlit as st

from . import demo, eda, models_page, threshold_analysis

PAGES = {
    "Análise Exploratória": eda,
    "Comparação de Modelos": models_page,
    "Análise de Threshold": threshold_analysis,
    "Demo (dataset Kaggle)": demo,
}

INTRO = (
    "Esta seção documenta a validação científica dos modelos usando o "
    "dataset público Kaggle. Ela fundamenta metodologicamente o sistema, "
    "mas não é o produto em si."
)


def render_intro() -> None:
    st.info(INTRO, icon="🔬")
