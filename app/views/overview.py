"""Página 'Visão Geral' — a proposta de produto do Sentinel.

Não carrega nenhum dataset: é texto e estrutura, então abre instantaneamente.

Os dois botões do topo navegam de verdade: escolhem a página no mesmo
`st.radio` da barra lateral (chave `nav_product`).
"""

import streamlit as st
from ui import grid, layer_card, numbered_list, signal_card, step_card

NAV_KEY = "nav_product"
CENTRAL, SIMULATOR = "Central de Transações", "Simulador"


def _go(page: str) -> None:
    st.session_state[NAV_KEY] = page


def _hero() -> None:
    st.markdown(
        '<div class="sx-hero"><h1 class="sx-display">Análise de risco transacional para instituições financeiras.</h1>'
        "<p><strong>Sentinel é uma plataforma de análise de risco transacional</strong> para instituições "
        "financeiras (bancos, fintechs). A proposta é examinar cada transação no contexto de quem a fez — o "
        "histórico do cliente, o cartão, o dispositivo, o local e o horário — e apontar as que se afastam do "
        "comportamento esperado, para que uma equipe de risco decida o que investigar.</p>"
        "<p>O sistema <strong>não decide sozinho</strong> se uma transação é fraude: ele classifica o risco de "
        "cada uma e deixa a decisão com quem opera.</p></div>",
        unsafe_allow_html=True,
    )
    with st.container(key="ov-cta"):
        st.button("Abrir Central de Transações", type="primary", key="ov_open_central", on_click=_go, args=(CENTRAL,))
        st.button("Testar no Simulador", key="ov_open_simulator", on_click=_go, args=(SIMULATOR,))


def _how_it_works() -> None:
    st.subheader("Como funciona")
    st.markdown("Três passos, da transação à decisão.")
    st.markdown(
        grid(
            [
                step_card(
                    1,
                    "Contexto",
                    "Cada transação é comparada ao histórico do próprio cliente: gasto habitual, cidades, janela "
                    "de horário e dispositivos.",
                ),
                step_card(
                    2,
                    "Análise",
                    "Um modelo de machine learning e um conjunto de regras medem o quanto a transação se afasta "
                    "do comportamento esperado.",
                ),
                step_card(
                    3,
                    "Decisão",
                    "O resultado vira um Risk Score de 0 a 100, com nível (baixo, médio, alto) e ação sugerida: "
                    "aprovar, revisar ou bloquear.",
                ),
            ],
            3,
        ),
        unsafe_allow_html=True,
    )


def _signals() -> None:
    st.subheader("Sinais que a plataforma considera")
    st.markdown("Cada sinal é medido contra o comportamento habitual do próprio cliente.")
    signals = [
        ("dollar", "Valor", "quanto a transação destoa do gasto habitual daquele cliente"),
        ("pin", "Localização", "distância de casa e velocidade implícita entre transações consecutivas"),
        ("clock", "Horário", "quão fora da janela habitual do cliente a transação ocorreu"),
        ("phone", "Dispositivo", "se o dispositivo já apareceu antes no histórico do cliente"),
    ]
    st.markdown(grid([signal_card(*s) for s in signals], 4), unsafe_allow_html=True)


def _architecture() -> None:
    st.subheader("Arquitetura em duas camadas")
    st.markdown("Uma camada que valida o método e outra que o coloca para operar.")
    science = layer_card(
        "Camada 1 · Científica",
        "Valida o método antes de usá-lo",
        "<p>Comparação de modelos, métricas, validação cruzada e análise de threshold sobre um dataset público "
        "do Kaggle. É a base que fundamenta as escolhas do motor de risco.</p>"
        '<p style="margin-top:16px">Fica no repositório do projeto (<b style="color:var(--text)">notebooks/</b> '
        "e <b style=\"color:var(--text)\">src/</b>), fora do app: aqui aparece só o produto.</p>",
    )
    operational = layer_card(
        "Camada 2 · Operacional",
        "Analisa transações ao vivo",
        numbered_list(
            [
                ("Domínio sintético", "clientes, cartões, dispositivos e transações"),
                ("Features comportamentais", "valor, distância, velocidade, horário, dispositivo e canal"),
                ("Risk Engine", "modelo de ML (70%) + regras (30%) → Risk Score"),
                ("Central, Simulador e Monitor ao Vivo", "em lote, transação a transação e em tempo real"),
            ]
        ),
    )
    st.markdown(grid([science, operational], 2), unsafe_allow_html=True)


def _status_and_planning() -> None:
    st.markdown("")
    with st.expander("Estado do projeto"):
        st.dataframe(
            {
                "Componente": [
                    "Domínio sintético (clientes, cartões, dispositivos, transações)",
                    "Features comportamentais",
                    "Validação científica dos modelos (dataset Kaggle, no repositório)",
                    "Risk Engine",
                    "Central de Transações",
                    "Simulador ao vivo",
                    "Monitor ao Vivo",
                ],
                "Situação": ["Pronto", "Pronto", "Pronto", "Pronto", "Pronto", "Pronto", "Pronto"],
            },
            hide_index=True,
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
        )


def render():
    _hero()
    _how_it_works()
    _signals()
    _architecture()
    _status_and_planning()
