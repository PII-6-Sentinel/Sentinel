"""Página 'Comparação de Modelos' — baseline estatístico vs. Regressão
Logística vs. Isolation Forest, lado a lado.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline import get_cv_results, get_pipeline
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
    fraud_rate = pipeline["fraud_rate"]

    st.subheader("Resumo")
    summary = pd.DataFrame(
        {
            name: {
                "Precisão": r.precision,
                "Recall": r.recall,
                "F1-score": r.f1,
                "AUC-ROC": r.auc_roc,
                "AUC-PR": r.auc_pr,
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
    st.caption(
        "AUC-ROC costuma parecer otimista demais em datasets tão desbalanceados "
        "quanto este (~0,17% de fraude) — AUC-PR (average precision) é mais "
        "sensível a falsos positivos e por isso mais informativa aqui."
    )

    with st.expander("Como o threshold de cada modelo foi escolhido"):
        st.markdown(
            "Precisão, recall, F1 e a matriz de confusão abaixo dependem de um "
            "**threshold** (ponto de corte sobre o escore contínuo de cada "
            "modelo). AUC-ROC e AUC-PR não — usam o ranking completo do escore.\n\n"
            f"Aqui, o threshold de cada modelo foi calibrado para sinalizar "
            f"aproximadamente a mesma proporção de transações que a taxa de "
            f"fraude observada no **treino** (~{fraud_rate * 100:.3f}%) — uma "
            "escolha didática para tornar os três modelos comparáveis nesta "
            "etapa, não uma prática de produção (em produção o threshold "
            "seria escolhido por custo de negócio, não pela taxa real de "
            "fraude, que não estaria disponível com essa precisão)."
        )
        st.dataframe(
            pd.DataFrame(
                {name: {"Threshold": r.threshold} for name, r in results.items()}
            ).T.style.format("{:.4f}"),
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
    col_roc, col_pr = st.columns(2)

    with col_roc:
        st.subheader("Curva ROC")
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
            height=460,
            xaxis_title="Taxa de Falsos Positivos",
            yaxis_title="Taxa de Verdadeiros Positivos (Recall)",
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(apply_plotly_theme(fig), use_container_width=True)

    with col_pr:
        st.subheader("Curva Precision-Recall")
        fig = go.Figure()
        fig.add_hline(
            y=fraud_rate, line=dict(color=MUTED, dash="dash"),
            annotation_text="Aleatório (taxa de fraude)", annotation_position="top left",
        )
        for name, r in results.items():
            fig.add_trace(
                go.Scatter(
                    x=r.pr_recall, y=r.pr_precision, mode="lines",
                    name=f"{name} (AUC-PR={r.auc_pr:.3f})",
                    line=dict(color=MODEL_COLORS.get(name, ELECTRIC_BLUE), width=2.5),
                )
            )
        fig.update_layout(
            height=460,
            xaxis_title="Recall",
            yaxis_title="Precisão",
            legend=dict(orientation="h", y=-0.25),
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

    st.markdown("")
    with st.expander("Validação cruzada"):
        st.caption(
            "5-fold Stratified Cross-Validation realizada exclusivamente no "
            "conjunto de treinamento — o conjunto de teste usado acima nunca "
            "entra nesta validação. Mede o quão **estável** cada modelo é "
            "entre diferentes recortes do treino; não substitui a avaliação "
            "final no teste (o restante desta página), que continua vindo de "
            "uma única avaliação, feita uma vez só."
        )

        with st.spinner("Rodando validação cruzada (só na primeira execução)..."):
            cv_results = get_cv_results()

        cv_table = pd.DataFrame(
            {
                name: {
                    "Precisão": f"{r.mean_metrics['precision']:.3f} ± {r.std_metrics['precision']:.3f}",
                    "Recall": f"{r.mean_metrics['recall']:.3f} ± {r.std_metrics['recall']:.3f}",
                    "F1-score": f"{r.mean_metrics['f1']:.3f} ± {r.std_metrics['f1']:.3f}",
                    "AUC-ROC": f"{r.mean_metrics['auc_roc']:.3f} ± {r.std_metrics['auc_roc']:.3f}",
                    "AUC-PR": f"{r.mean_metrics['auc_pr']:.3f} ± {r.std_metrics['auc_pr']:.3f}",
                }
                for name, r in cv_results.items()
            }
        ).T
        st.dataframe(cv_table, use_container_width=True)

        st.caption(
            "O threshold de cada fold é calibrado pela prevalência de fraude "
            "observada no treino DAQUELE fold (nunca no validation do fold). "
            "Esse threshold representa um ponto de operação baseado na "
            "prevalência observada no conjunto de treinamento e não deve ser "
            "interpretado como conhecimento disponível em produção — não é "
            "uma probabilidade de fraude."
        )

        if st.checkbox("Ver métricas por fold"):
            fold_rows = [
                {
                    "Modelo": name,
                    "Fold": fold["fold"],
                    "Precisão": fold["precision"],
                    "Recall": fold["recall"],
                    "F1-score": fold["f1"],
                    "AUC-ROC": fold["auc_roc"],
                    "AUC-PR": fold["auc_pr"],
                    "Threshold": fold["threshold"],
                }
                for name, r in cv_results.items()
                for fold in r.metrics_by_fold
            ]
            st.dataframe(
                pd.DataFrame(fold_rows).style.format(
                    {
                        "Precisão": "{:.3f}",
                        "Recall": "{:.3f}",
                        "F1-score": "{:.3f}",
                        "AUC-ROC": "{:.3f}",
                        "AUC-PR": "{:.3f}",
                        "Threshold": "{:.4f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
