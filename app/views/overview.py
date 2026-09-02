"""Página 'Visão Geral' — objetivo do projeto e resumo do planejamento (SWOT/GUT)."""

import streamlit as st

from pipeline import load_data


def render():
    st.markdown(
        '<div class="sentinel-header"><span class="icon">🛡️</span>'
        "<h1>Sentinel</h1></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sentinel-tagline">Vigilância estatística e de Machine '
        "Learning sobre transações financeiras</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        **Sentinel** é um Projeto Integrador acadêmico (TTI 304 — Gerenciamento
        de Projetos em TI) que constrói e compara, com rigor metodológico,
        diferentes abordagens para detectar transações fraudulentas:

        - um **baseline estatístico** (regra de limiar sobre um escore de anomalia);
        - um modelo supervisionado de **Machine Learning** (Regressão Logística);
        - um modelo **não supervisionado** (Isolation Forest).

        O objetivo não é só "ter um modelo que funciona", mas demonstrar
        entendimento do processo de avaliação — matriz de confusão, precisão,
        recall, F1-score e AUC-ROC — em um cenário de classes extremamente
        desbalanceadas.
        """
    )

    df = load_data()
    n_total = len(df)
    n_fraud = int(df["Class"].sum())
    fraud_pct = n_fraud / n_total * 100
    hours_covered = df["Time"].max() / 3600

    st.subheader("O dataset em números")
    cols = st.columns(4)
    metrics = [
        ("Transações totais", f"{n_total:,}".replace(",", ".")),
        ("Fraudes confirmadas", f"{n_fraud}"),
        ("Proporção de fraude", f"{fraud_pct:.3f}%"),
        ("Período coberto", f"~{hours_covered:.0f}h"),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.markdown(
            f'<div class="metric-card"><div class="value">{value}</div>'
            f'<div class="label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.subheader("Planejamento: Análise SWOT")
    st.caption(
        "Resumo — análise completa em `docs/SWOT_GUT.md`."
    )

    swot_cols = st.columns(2)
    swot_content = [
        ("Forças", [
            "Dataset público, bem documentado, benchmark acadêmico consolidado",
            "Problema bem delimitado, com métricas de sucesso claras na literatura",
        ], False),
        ("Fraquezas", [
            "Pouca experiência prévia da equipe com validação estatística rigorosa",
            "Pouca experiência com técnicas de balanceamento de classes",
        ], True),
        ("Oportunidades", [
            "Boa base para aprendizado didático de avaliação de modelos",
            "Comparação estatística vs. ML gera material rico para o relatório",
        ], False),
        ("Ameaças", [
            "Dataset fortemente desbalanceado (~0,17% de fraudes) — maior ameaça",
            "Risco de vazamento de dados (data leakage) se balanceamento/escala forem aplicados antes do split",
        ], True),
    ]
    for i, (title, items, is_risk) in enumerate(swot_content):
        css_class = "swot-card risk" if is_risk else "swot-card"
        items_html = "".join(f"<li>{item}</li>" for item in items)
        swot_cols[i % 2].markdown(
            f'<div class="{css_class}"><h4>{title}</h4><ul>{items_html}</ul></div>',
            unsafe_allow_html=True,
        )
        if i % 2 == 1:
            st.markdown("")

    st.markdown("")
    st.subheader("Priorização de riscos (matriz GUT)")
    st.dataframe(
        {
            "Risco": [
                "Falta de rigor estatístico na avaliação",
                "Desbalanceamento de classes tratado incorretamente",
                "Prazo de semestre apertado",
                "Overfitting por validação mal feita",
            ],
            "Gravidade": [5, 5, 4, 4],
            "Urgência": [5, 5, 4, 3],
            "Tendência": [4, 3, 4, 3],
            "GUT": [100, 75, 64, 36],
            "Prioridade": ["Crítica", "Alta", "Alta", "Média"],
        },
        hide_index=True,
        use_container_width=True,
    )
