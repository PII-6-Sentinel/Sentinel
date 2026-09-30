"""Página 'Visão Geral' — a proposta de produto do Sentinel.

Não carrega nenhum dataset: é texto e estrutura, então abre instantaneamente
e funciona mesmo sem o CSV do Kaggle baixado.
"""

import streamlit as st


def render():
    st.markdown(
        '<div class="sentinel-header"><span class="icon">🛡️</span>'
        "<h1>Sentinel</h1></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sentinel-tagline">Análise de risco transacional para '
        "instituições financeiras</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        **Sentinel é uma plataforma de análise de risco transacional para
        instituições financeiras** (bancos, fintechs). A proposta é examinar
        cada transação no contexto de quem a fez — o histórico do cliente,
        o cartão, o dispositivo, o local e o horário — e apontar as que se
        afastam do comportamento esperado, para que uma equipe de risco
        decida o que investigar.

        O sistema **não decide sozinho** se uma transação é fraude: ele
        classifica o risco de cada uma e deixa a decisão com quem opera.
        """
    )

    st.subheader("Sinais que a plataforma considera")
    cols = st.columns(4)
    signals = [
        ("Valor", "quanto a transação destoa do gasto habitual daquele cliente"),
        ("Localização", "distância de casa e velocidade implícita entre transações consecutivas"),
        ("Horário", "quão fora da janela habitual do cliente a transação ocorreu"),
        ("Dispositivo", "se o dispositivo já apareceu antes no histórico do cliente"),
    ]
    for col, (title, text) in zip(cols, signals):
        col.markdown(
            f'<div class="swot-card"><h4>{title}</h4><p style="color:#8AA0C4;margin:0">{text}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.subheader("Estado do projeto")
    st.dataframe(
        {
            "Componente": [
                "Domínio sintético (clientes, cartões, dispositivos, transações)",
                "Features comportamentais",
                "Validação científica dos modelos (dataset Kaggle)",
                "Risk Engine",
                "Central de Transações",
            ],
            "Situação": ["Pronto", "Pronto", "Pronto", "Em construção", "Em construção"],
        },
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "**Validação científica:** a base metodológica do sistema — "
        "comparação de modelos, métricas, validação cruzada e análise de "
        "threshold sobre o dataset público do Kaggle — fica em "
        "**🔬 Ver Metodologia Científica (Kaggle)**, no rodapé da barra lateral.",
        icon="🔬",
    )

    with st.expander("Planejamento do projeto (SWOT / GUT)"):
        st.markdown(
            "Projeto Integrador — TTI 304 (Gerenciamento de Projetos em TI). "
            "Análise completa em `docs/SWOT_GUT.md`."
        )
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
