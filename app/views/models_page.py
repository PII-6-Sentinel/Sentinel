"""Página 'Comparação de Modelos' — baseline estatístico vs. Regressão
Logística vs. Isolation Forest, lado a lado.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline import get_pipeline
from theme import CORAL, ELECTRIC_BLUE, MUTED, apply_plotly_theme

MODEL_COLORS = {
    "Baseline Estatístico": MUTED,
    "Regressão Logística": ELECTRIC_BLUE,
    "Isolation Forest": CORAL,
}


def render():
    st.title("Comparação de Modelos")
    st.caption(
        "Métricas calculadas sobre o mesmo conjunto de teste (20% dos dados, "
        "nunca usado no treino/calibração de nenhum modelo). Acurácia é "
        "deliberadamente omitida — com ~0,17% de fraudes, ela não distingue "
        "um modelo bom de um modelo que só chuta 'legítima' sempre."
    )

    with st.spinner("Preparando dados e treinando modelos (só na primeira execução)..."):
        pipeline = get_pipeline()
    results = pipeline["results"]

    st.subheader("Resumo")
    summary = pd.DataFrame(
        {
            name: {
                "Precisão": r.precision,
                "Recall": r.recall,
                "F1-score": r.f1,
                "AUC-ROC": r.auc_roc,
            }
            for name, r in results.items()
        }
    ).T
    st.dataframe(
        summary.style.format("{:.3f}").background_gradient(
            cmap="Blues", vmin=0, vmax=1
        ),
        use_container_width=True,
    )

    st.markdown("")
    st.subheader("Matriz de confusão por modelo")
    cols = st.columns(len(results))
    for col, (name, r) in zip(cols, results.items()):
        with col:
            st.markdown(f"**{name}**")
            fig = px.imshow(
                r.confusion_matrix,
                text_auto=True,
                x=["Prev.: Legítima", "Prev.: Fraude"],
                y=["Real: Legítima", "Real: Fraude"],
                color_continuous_scale=["#0A192F", ELECTRIC_BLUE],
            )
            fig.update_layout(height=340, coloraxis_showscale=False)
            st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)
            tn, fp, fn, tp = r.confusion_matrix.ravel()
            st.caption(f"Fraudes detectadas: {tp}/{tp + fn}  •  Falsos positivos: {fp}")

    st.markdown("")
    st.subheader("Curvas ROC")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Aleatório",
            line=dict(color=MUTED, dash="dash"),
        )
    )
    for name, r in results.items():
        fig.add_trace(
            go.Scatter(
                x=r.fpr, y=r.tpr, mode="lines",
                name=f"{name} (AUC={r.auc_roc:.3f})",
                line=dict(color=MODEL_COLORS.get(name, ELECTRIC_BLUE), width=2.5),
            )
        )
    fig.update_layout(
        height=480,
        xaxis_title="Taxa de Falsos Positivos",
        yaxis_title="Taxa de Verdadeiros Positivos (Recall)",
    )
    st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    st.info(
        "**Como ler:** Precisão alta = poucas transações legítimas erradamente "
        "sinalizadas como fraude. Recall alto = poucas fraudes passando "
        "despercebidas. Em detecção de fraude, o custo de deixar passar uma "
        "fraude (falso negativo) costuma ser maior que o de investigar uma "
        "transação legítima por engano (falso positivo) — por isso recall "
        "e F1-score costumam pesar mais que precisão isolada nesse domínio."
    )
