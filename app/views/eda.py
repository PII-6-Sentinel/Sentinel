"""Página 'Análise Exploratória' — versão interativa dos gráficos de
notebooks/01_exploracao.ipynb, usando Plotly em vez de matplotlib/seaborn.
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline import load_data
from theme import CLASS_COLORS, CORAL, ELECTRIC_BLUE, apply_plotly_theme

V_COLUMNS = [f"V{i}" for i in range(1, 29)]


def _label(series):
    return series.map({0: "Legítima", 1: "Fraude"})


def render():
    st.title("Análise Exploratória")
    st.caption(
        "Mesmos dados e conclusões de `notebooks/01_exploracao.ipynb`, "
        "em gráficos interativos."
    )

    df = load_data()
    class_label = _label(df["Class"])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição de classes")
        log_scale = st.checkbox("Escala logarítmica (eixo Y)", value=True)
        counts = class_label.value_counts().reindex(["Legítima", "Fraude"])
        fig = go.Figure(
            go.Bar(
                x=counts.index,
                y=counts.values,
                marker_color=[CLASS_COLORS[c] for c in counts.index],
                text=counts.values,
                textposition="outside",
            )
        )
        fig.update_layout(yaxis_type="log" if log_scale else "linear", height=380)
        st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)
        st.caption(
            f"{counts['Fraude']} fraudes em {counts.sum():,} transações "
            f"({counts['Fraude'] / counts.sum() * 100:.3f}%)."
            .replace(",", ".")
        )

    with col2:
        st.subheader("Amount por classe")
        log_amount = st.checkbox("Escala logarítmica (Amount)", value=True)
        fig = px.box(
            df.assign(Classe=class_label),
            x="Classe",
            y="Amount",
            color="Classe",
            color_discrete_map=CLASS_COLORS,
            category_orders={"Classe": ["Legítima", "Fraude"]},
            log_y=log_amount,
        )
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)
        st.caption(
            "Fraudes tendem a se concentrar em valores mais baixos de "
            "transação — vale confirmar isso olhando as medianas por classe."
        )

    st.markdown("")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Correlação com a classe (fraude)")
        corr_with_class = (
            df[V_COLUMNS + ["Amount", "Time"]]
            .corrwith(df["Class"])
            .sort_values()
        )
        colors = [CORAL if v > 0 else ELECTRIC_BLUE for v in corr_with_class.values]
        fig = go.Figure(
            go.Bar(
                x=corr_with_class.values,
                y=corr_with_class.index,
                orientation="h",
                marker_color=colors,
            )
        )
        fig.update_layout(height=520)
        st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)
        st.caption(
            "Correlação linear de cada feature com `Class`. Como V1..V28 vêm "
            "de um PCA, nenhuma é diretamente interpretável isoladamente — "
            "o gráfico serve para identificar quais componentes mais "
            "'carregam' o sinal de fraude."
        )

    with col4:
        st.subheader("Correlação entre features (V1..V28)")
        corr_matrix = df[V_COLUMNS].corr()
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale=["#0A192F", ELECTRIC_BLUE, CORAL],
            zmin=-1,
            zmax=1,
            aspect="auto",
        )
        fig.update_layout(height=520)
        st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)
        st.caption(
            "Baixa correlação entre V1..V28 é esperada: PCA gera componentes "
            "ortogonais (não-correlacionadas) por construção."
        )
