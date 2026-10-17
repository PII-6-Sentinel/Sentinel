"""Sentinel — app (`streamlit run app/app.py`).

Duas seções, com pesos diferentes de propósito:

- **Produto** (navegação principal): as páginas operacionais, voltadas a um
  cliente (banco/fintech). Em `app/views/`.
- **Metodologia Científica (Kaggle)**: a validação dos modelos, base
  metodológica do sistema mas não o produto. Em `app/views/scientific/`,
  atrás de um único ponto de entrada no rodapé da sidebar.

Camada de UI pura: toda a lógica de dados/modelos vive em src/ e é reusada
aqui via app/pipeline.py (que adiciona cache). Cada página é uma função
`render()`, escolhida pela navegação da seção ativa.
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

from theme import ASSETS_DIR, inject_css  # noqa: E402
from ui import brand_html  # noqa: E402
from views import overview, scientific, simulator, transactions  # noqa: E402

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

PRODUCT, SCIENTIFIC = "product", "scientific"
st.session_state.setdefault("section", PRODUCT)

SCIENTIFIC_ENTRY = "🔬 Ver Metodologia Científica (Kaggle) →"


def _go(section: str) -> None:
    st.session_state["section"] = section


section = st.session_state["section"]

with st.sidebar:
    st.markdown(brand_html(), unsafe_allow_html=True)

    if section == PRODUCT:
        page_name = st.radio(
            "Navegação", list(PRODUCT_PAGES.keys()), key="nav_product", label_visibility="collapsed"
        )
        # a seção científica fica atrás de um único botão discreto, na base
        with st.container(key="sidebar-bottom"):
            st.button(SCIENTIFIC_ENTRY, on_click=_go, args=(SCIENTIFIC,))
            st.markdown('<div class="sx-foot">Projeto Integrador — TTI 304</div>', unsafe_allow_html=True)
    else:
        st.button("← Voltar ao produto", on_click=_go, args=(PRODUCT,))
        st.caption("🔬 Metodologia Científica (Kaggle)")
        page_name = st.radio(
            "Navegação científica",
            list(scientific.PAGES.keys()),
            key="nav_scientific",
            label_visibility="collapsed",
        )
        with st.container(key="sidebar-bottom"):
            st.caption(
                "Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle/ULB)"
            )
            st.markdown('<div class="sx-foot">Projeto Integrador — TTI 304</div>', unsafe_allow_html=True)

if section == PRODUCT:
    PRODUCT_PAGES[page_name].render()
else:
    scientific.render_intro()
    scientific.PAGES[page_name].render()
