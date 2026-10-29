"""Página 'Visão Geral' — a proposta de produto do Sentinel.

Não carrega nenhum dataset: é texto e estrutura, então abre instantaneamente.

Os dois botões do topo navegam de verdade: escolhem a página no mesmo
`st.radio` da barra lateral (chave `nav_product`).
"""

import streamlit as st
from theme import RISK_HIGH
from ui import card, esc, grid, layer_card, numbered_list, signal_card, step_card

NAV_KEY = "nav_product"
CENTRAL, SIMULATOR = "Central de Transações", "Simulador"


def _go(page: str) -> None:
    st.session_state[NAV_KEY] = page


# Uma transação real da Central (seed 42, conjunto de teste: o modelo nunca
# viu este cliente), com o que o motor respondeu. Fixa aqui para a página
# abrir na hora, sem rodar a análise; confira com a transação #2269 na Central.
DEMO_TRANSACTION = {
    "id": 2269,
    "customer": "Fernanda Carvalho · mora em Madri",
    "purchase": "R$ 181,64 num posto em Belo Horizonte",
    "when": "20/06/2026 às 20:10",
    "score": 99,
    "action": "Bloquear",
    "reasons": [
        "Viagem impossível: 1.390 km/h entre duas compras",
        "Localização incomum: 7.874 km da residência",
        "Primeira compra do cliente nesta cidade",
        "Horário fora da janela habitual (3,7 h fora)",
    ],
}


def _demo_card() -> str:
    d = DEMO_TRANSACTION
    reasons = "".join(f"<li>{esc(r)}</li>" for r in d["reasons"])
    return (
        f'<div class="sx-demo" style="--lvl:{RISK_HIGH}">'
        f'<div class="sx-demo-head"><span class="sx-demo-live">Exemplo real</span>'
        f'<span>Transação #{d["id"]}</span></div>'
        f'<div class="sx-demo-who">{esc(d["customer"])}</div>'
        f'<div class="sx-demo-what">{esc(d["purchase"])}<br>{esc(d["when"])}</div>'
        f'<div class="sx-demo-verdict"><div class="sx-demo-score">{d["score"]}<small>Risk Score</small></div>'
        f'<div class="sx-demo-action">{esc(d["action"])}<small>nível alto · revisão urgente</small></div></div>'
        f'<ul class="sx-demo-reasons">{reasons}</ul>'
        f'<div class="sx-demo-foot">Decidido em milissegundos, com os motivos à vista.</div></div>'
    )


def _hero() -> None:
    text, demo = st.columns([1.35, 1], gap="large")
    with text:
        _hero_text()
    with demo:
        st.markdown(_demo_card(), unsafe_allow_html=True)


def _hero_text() -> None:
    st.markdown(
        '<div class="sx-hero"><h1 class="sx-display">Análise de risco transacional para instituições financeiras.</h1>'
        "<p>O Sentinel examina cada transação no contexto de quem a fez — histórico do cliente, dispositivo, "
        "local e horário — e aponta as que se afastam do comportamento esperado. Ele <strong>não decide "
        "sozinho</strong>: classifica o risco e deixa a decisão com a equipe de risco.</p>"
        "<p><strong>Protótipo demonstrativo.</strong> O motor do app roda sobre dados sintéticos: as métricas "
        "mostradas aqui validam a lógica do sistema, não o desempenho contra fraude real. Afirmar desempenho "
        "exigiria dados reais rotulados de uma instituição — o mais perto disso é o teste com fraude real abaixo.</p></div>",
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


# Números de `python scripts/run_kaggle_benchmark.py` com o creditcard.csv
# real (docs/VALIDACAO.md, seção 2). Fixos aqui: o app não carrega o Kaggle.
KAGGLE_RESULTS = [
    ("Transações reais", "284.807", "492 fraudes confirmadas (0,17%) — cartões europeus, Kaggle/ULB"),
    ("PR-AUC do Gradient Boosting", "0,86", "Regressão Logística 0,72 · um chute daria 0,002"),
    ("Alertas que eram fraude", "91%", "alertando só 0,1% das transações (179 por dia), teste no futuro"),
    ("Prejuízo evitado", "59%", "custo de revisão + fraudes perdidas, contra não alertar nada"),
]


def _real_fraud() -> None:
    st.subheader("Testado com fraude real")
    st.markdown(
        "Antes do motor, os modelos de ML foram validados num dataset público de fraudes reais: treino no "
        "passado, teste no futuro, como em produção. Ninguém escreveu as regras dessas fraudes — o modelo as "
        "aprendeu sozinho."
    )
    st.markdown(grid([card(*item, size="sm") for item in KAGGLE_RESULTS], 4), unsafe_allow_html=True)
    st.caption(
        "Fonte: `scripts/run_kaggle_benchmark.py` no repositório (detalhes em `docs/VALIDACAO.md`). Os números "
        "validam as técnicas; o dataset não tem os sinais de cliente que o motor do app usa."
    )


def _architecture() -> None:
    st.subheader("Arquitetura em duas camadas")
    st.markdown("Uma camada que valida o método e outra que o coloca para operar.")
    science = layer_card(
        "Camada 1 · Científica",
        "Valida o método antes de usá-lo",
        "<p>Comparação de modelos, métricas, validação cruzada, split temporal e limiar por custo sobre fraudes "
        "reais do Kaggle (seção acima). É a base que fundamenta as escolhas do motor de risco.</p>"
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
                ("Risk Engine", "modelo de ML + regras, com o peso calibrado na validação → Risk Score"),
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
    _real_fraud()
    _architecture()
    _status_and_planning()
