"""Sentinel — app de apresentação do projeto (`streamlit run app/app.py`).

Camada de UI pura: toda a lógica de dados/modelos vive em src/ e é reusada
aqui via app/pipeline.py (que adiciona cache). Cada página é uma função
`render()` em app/views/, escolhida pela navegação na sidebar.
"""

import sys
from pathlib import Path

import streamlit as st

# Permite `import src...` (raiz do projeto) e imports diretos dos módulos
# irmãos deste arquivo (theme, pipeline, views). Usamos imports sem o
# prefixo `app.` de propósito: quando o Streamlit executa app/app.py
# diretamente, ele registra o script em sys.modules sob o nome "app",
# o que colidiria com o pacote app/ se tentássemos `import app.theme`.
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for path in (PROJECT_ROOT, APP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from theme import inject_css  # noqa: E402
from views import demo, eda, models_page, overview  # noqa: E402

st.set_page_config(
    page_title="Sentinel — Detecção de Anomalias",
    page_icon="🛡️",
    layout="wide",
)
inject_css()

PAGES = {
    "Visão Geral": overview,
    "Análise Exploratória": eda,
    "Comparação de Modelos": models_page,
    "Demo ao Vivo": demo,
}

with st.sidebar:
    st.markdown("## 🛡️ Sentinel")
    st.caption("Detecção de anomalias em transações")
    page_name = st.radio("Navegação", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.caption(
        "Projeto Integrador — TTI 304\n\n"
        "Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle/ULB)"
    )

PAGES[page_name].render()
