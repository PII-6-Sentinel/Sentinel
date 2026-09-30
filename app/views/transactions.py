"""Página 'Central de Transações' — placeholder da camada operacional.

A central vai consumir o domínio sintético (clientes, cartões, dispositivos,
transações) e o Risk Engine. Nenhum dos dois é consumido pelo app ainda.
"""

import streamlit as st


def render():
    st.title("Central de Transações")
    st.warning(
        "**Em construção — Risk Engine nas próximas etapas do projeto.**",
        icon="🚧",
    )
    st.markdown(
        "Esta página será a porta de entrada operacional do Sentinel: a "
        "lista de transações de clientes, com o risco atribuído a cada uma. "
        "Ela depende do Risk Engine, que ainda não foi construído — por "
        "isso não há dados nem funcionalidades aqui por enquanto."
    )
    st.caption(
        "Enquanto isso, a validação científica dos modelos está disponível "
        "em **🔬 Ver Metodologia Científica (Kaggle)**, no rodapé da barra lateral."
    )
