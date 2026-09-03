"""Página 'Análise de Threshold' — simula como mudar o limiar de decisão
altera a classificação de um modelo JÁ treinado, sem retreinar nada.

Camada de UI pura: toda a matemática (varredura de threshold, métricas a
um threshold) vem de `src/evaluation.py`, reusada via `app/pipeline.py`
(que adiciona cache). Esta página não recalcula nada por conta própria.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline import get_pipeline, get_threshold_sweep
from theme import CORAL, ELECTRIC_BLUE, MUTED, apply_plotly_theme
from src.evaluation import calculate_metrics


def render():
    st.title("Análise de Threshold")
    st.caption(
        "Simulação: veja como mudar o limiar de decisão altera a "
        "classificação de um modelo já treinado — sem retreinar nada e "
        "sem rodar validação cruzada de novo. O slider abaixo é só uma "
        "simulação: não altera o modelo, não altera o threshold oficial "
        "do pipeline, e nada aqui é salvo como configuração."
    )

    with st.spinner("Carregando resultados do pipeline (já cacheados)..."):
        pipeline = get_pipeline()
    results = pipeline["results"]
    official_thresholds = pipeline["thresholds"]
    y_test = pipeline["y_test"]

    model_name = st.selectbox("Modelo", list(results.keys()))
    result = results[model_name]
    y_score = result.y_score  # escala nativa deste modelo — ver nota abaixo
    official_threshold = official_thresholds[model_name]

    score_min, score_max = float(np.min(y_score)), float(np.max(y_score))
    st.caption(
        f"O escore deste modelo varia entre **{score_min:.4f}** e "
        f"**{score_max:.4f}** no conjunto de teste. Cada modelo tem sua "
        "própria escala de escore (probabilidade, distância, escore de "
        "isolamento) — o slider abaixo usa a escala nativa do modelo "
        "selecionado, não uma escala fixa de 0 a 1."
    )

    step = (score_max - score_min) / 200 if score_max > score_min else 0.01
    threshold = st.slider(
        "Limiar de decisão (threshold)",
        min_value=score_min,
        max_value=score_max,
        value=float(np.clip(official_threshold, score_min, score_max)),
        step=step,
        format="%.4f",
        key=f"threshold_slider_{model_name}",
    )

    # Única chamada de avaliação por interação — nenhum modelo é
    # retreinado, nenhum dado é recarregado (ver src/evaluation.py).
    metrics = calculate_metrics(y_test, y_score, threshold)

    st.markdown("")
    st.subheader("Métricas dependentes do threshold")
    st.caption(
        "Mudam conforme você move o slider acima — refletem o ponto de "
        "operação escolhido, não a qualidade geral do modelo."
    )
    row1 = st.columns(3)
    row1[0].metric("Precisão", f"{metrics['precision'] * 100:.1f}%")
    row1[1].metric("Recall", f"{metrics['recall'] * 100:.1f}%")
    row1[2].metric("F1-score", f"{metrics['f1'] * 100:.1f}%")

    row2 = st.columns(4)
    row2[0].metric("FPR", f"{metrics['fpr'] * 100:.1f}%")
    row2[1].metric("Specificity", f"{metrics['specificity'] * 100:.1f}%")
    row2[2].metric("Balanced Accuracy", f"{metrics['balanced_accuracy'] * 100:.1f}%")
    row2[3].metric("MCC", f"{metrics['mcc']:.3f}")

    st.markdown("")
    st.subheader("Métricas independentes do threshold")
    st.caption(
        "AUC-ROC e AUC-PR usam o ranking completo do escore — **não mudam** "
        "conforme o slider; ficam aqui como referência fixa da qualidade "
        "geral do modelo, para comparar com o ponto de operação escolhido."
    )
    row3 = st.columns(2)
    row3[0].metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
    row3[1].metric("AUC-PR", f"{metrics['auc_pr']:.3f}")

    st.markdown("")
    st.subheader("Matriz de confusão neste threshold")
    fig = px.imshow(
        metrics["confusion_matrix"],
        text_auto=True,
        x=["Prev.: Normal", "Prev.: Potencialmente suspeita"],
        y=["Real: Normal", "Real: Fraude"],
        color_continuous_scale=["#0A192F", ELECTRIC_BLUE],
    )
    fig.update_layout(height=340, coloraxis_showscale=False)
    st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    st.markdown("")
    st.subheader("Precisão / Recall / F1 por threshold")
    with st.spinner("Calculando varredura de threshold (só na primeira vez por modelo)..."):
        sweep = get_threshold_sweep(model_name)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sweep["threshold"], y=sweep["precision"], mode="lines",
            name="Precisão", line=dict(color=ELECTRIC_BLUE, width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sweep["threshold"], y=sweep["recall"], mode="lines",
            name="Recall", line=dict(color=CORAL, width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sweep["threshold"], y=sweep["f1"], mode="lines",
            name="F1-score", line=dict(color=MUTED, width=2.5),
        )
    )
    fig.add_vline(
        x=threshold, line=dict(color=ELECTRIC_BLUE, dash="dot", width=2),
        annotation_text="Selecionado", annotation_position="top",
    )
    fig.add_vline(
        x=official_threshold, line=dict(color=MUTED, dash="dash", width=2),
        annotation_text="Oficial do pipeline", annotation_position="bottom",
    )
    fig.update_layout(
        height=420,
        xaxis_title="Threshold",
        yaxis_title="Métrica",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    st.markdown("")
    st.info(
        f"**Threshold oficial calibrado pelo pipeline para {model_name}:** "
        f"{official_threshold:.4f} — calibrado pela prevalência de fraude "
        "observada no treino (ver página 'Comparação de Modelos'). Mover o "
        "slider acima não altera esse valor nem os resultados oficiais."
    )
    if abs(threshold - official_threshold) < 1e-9:
        st.caption("O slider está no mesmo valor do threshold oficial.")
    elif threshold < official_threshold:
        st.caption(
            "Você está simulando um threshold **mais permissivo** que o "
            "oficial — tende a sinalizar mais transações como suspeitas."
        )
    else:
        st.caption(
            "Você está simulando um threshold **mais conservador** que o "
            "oficial — tende a sinalizar menos transações como suspeitas."
        )

    st.markdown("")
    with st.expander("Como interpretar"):
        st.markdown(
            "Um threshold **menor** tende a aumentar a sensibilidade do "
            "detector — identifica mais transações potencialmente "
            "suspeitas, mas também tende a aumentar falsos positivos. Um "
            "threshold **maior** tende a produzir decisões mais "
            "conservadoras — menos sinalizações, mas com mais risco de "
            "deixar passar uma transação anômala sem revisão.\n\n"
            "O Sentinel classifica cada transação como **Normal** ou "
            "**Potencialmente suspeita** — nunca como fraude confirmada. "
            "O escore de cada modelo não é uma probabilidade de fraude "
            "calibrada; é um ranking de risco relativo, específico da "
            "escala de cada modelo, usado aqui para avaliação de risco "
            "transacional, não como veredito."
        )
