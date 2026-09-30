"""Página 'Demo ao Vivo' — classifica uma transação real do dataset,
selecionada pelo usuário, com o modelo escolhido.
"""

import plotly.graph_objects as go
import streamlit as st

from pipeline import get_pipeline
from theme import CORAL, ELECTRIC_BLUE, NAVY_LIGHT, apply_plotly_theme
from src.models import percentile_rank

V_COLUMNS = [f"V{i}" for i in range(1, 29)]


def render():
    st.title("Demo ao Vivo")
    st.caption(
        "Selecione uma transação real do conjunto de teste (nunca vista pelos "
        "modelos durante o treino) e veja a classificação na hora."
    )

    with st.spinner("Preparando modelos..."):
        pipeline = get_pipeline()

    models = pipeline["models"]
    results = pipeline["results"]
    X_test = pipeline["X_test"]
    y_test = pipeline["y_test"]

    col_a, col_b = st.columns([1, 1])
    with col_a:
        model_name = st.selectbox("Modelo", list(models.keys()))
    with col_b:
        class_filter = st.radio(
            "Filtrar transações por classe real",
            ["Apenas fraudes", "Apenas legítimas", "Todas"],
            horizontal=True,
            help="Só ~0,17% das transações são fraude — filtrar facilita achar um exemplo interessante.",
        )

    if class_filter == "Apenas fraudes":
        candidate_idx = y_test[y_test == 1].index
    elif class_filter == "Apenas legítimas":
        candidate_idx = y_test[y_test == 0].index
    else:
        candidate_idx = y_test.index

    selected_idx = st.selectbox(
        f"Transação (índice — {len(candidate_idx)} disponíveis)",
        candidate_idx,
    )

    result = results[model_name]
    pos = X_test.index.get_loc(selected_idx)
    y_pred = int(result.y_pred[pos])
    score = float(result.y_score[pos])
    risk_percentile = percentile_rank(score, result.y_score)
    actual = int(y_test.loc[selected_idx])

    row = X_test.loc[selected_idx]

    st.markdown("")
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Transação selecionada")
        st.markdown(
            f'<div class="metric-card"><div class="value">R$ {row["Amount"]:.2f}</div>'
            '<div class="label">Amount</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(
            f'<div class="metric-card"><div class="value">{row["Time"] / 3600:.1f}h</div>'
            '<div class="label">Time (desde a 1ª transação do dataset)</div></div>',
            unsafe_allow_html=True,
        )
        with st.expander("Ver todas as features (V1..V28)"):
            st.dataframe(row[V_COLUMNS].to_frame("valor"), use_container_width=True)

    with right:
        st.subheader("Classificação do modelo")
        if y_pred == 1:
            st.markdown(
                '<div class="result-badge fraud">⚠️ FRAUDE DETECTADA</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-badge normal">✅ TRANSAÇÃO NORMAL</div>',
                unsafe_allow_html=True,
            )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_percentile,
                number={"suffix": "%"},
                title={"text": "Score de risco (percentil no conjunto de teste)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": CORAL if y_pred == 1 else ELECTRIC_BLUE},
                    "bgcolor": NAVY_LIGHT,
                    "steps": [
                        {"range": [0, 60], "color": NAVY_LIGHT},
                        {"range": [60, 100], "color": "#3A2233"},
                    ],
                },
            )
        )
        fig.update_layout(height=300)
        st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

        actual_label = "Fraude" if actual == 1 else "Legítima"
        if actual == y_pred:
            st.success(f"Confere com o rótulo real do dataset (Classe real: **{actual_label}**).")
        else:
            st.warning(
                f"O modelo divergiu do rótulo real (Classe real: **{actual_label}**) — "
                "nenhum modelo é perfeito; bom ponto para discutir na apresentação."
            )

    st.caption(
        f"Escore bruto do modelo: {score:.4f}  •  "
        "escala não comparável entre modelos, por isso o percentil acima."
    )
