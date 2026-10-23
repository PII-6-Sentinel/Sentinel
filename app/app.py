"""Sentinel — app (`streamlit run app/app.py`).

Só o produto: as páginas operacionais, voltadas a um cliente
(banco/fintech), em `app/views/`. A validação científica dos modelos sobre
o dataset do Kaggle continua no repositório (`src/`, `notebooks/`), mas não
faz parte do app.

Camada de UI pura: toda a lógica de dados/modelos vive em src/. Cada página
é uma função `render()`, escolhida pela navegação da barra lateral.
"""

import sys
from pathlib import Path

import streamlit as st

# Permite `import src...` (raiz do projeto) e imports diretos dos módulos
# irmãos deste arquivo (theme, ui, views). Usamos imports sem o
# prefixo `app.` de propósito: quando o Streamlit executa app/app.py
# diretamente, ele registra o script em sys.modules sob o nome "app",
# o que colidiria com o pacote app/ se tentássemos `import app.theme`.
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for path in (PROJECT_ROOT, APP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from theme import ASSETS_DIR, inject_css  # noqa: E402
from ui import brand_html  # noqa: E402
from views import overview, simulator, transactions  # noqa: E402

st.set_page_config(
    page_title="Sentinel — Análise de Risco Transacional",
    page_icon=str(ASSETS_DIR / "sentinel-icon.svg"),
    layout="wide",
)
inject_css()

PRODUCT_PAGES = {
    "Visão Geral": overview,
    "Central de Transações": transactions,
    "Simulador": simulator,
}

with st.sidebar:
    st.markdown(brand_html(), unsafe_allow_html=True)
    page_name = st.radio(
        "Navegação", list(PRODUCT_PAGES.keys()), key="nav_product", label_visibility="collapsed"
    )
    with st.container(key="sidebar-bottom"):
        st.markdown('<div class="sx-foot">Projeto Integrador — TTI 304</div>', unsafe_allow_html=True)

PRODUCT_PAGES[page_name].render()
