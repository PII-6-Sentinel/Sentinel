"""Paleta de cores e identidade visual do Sentinel.

Mantido separado das páginas para que todo gráfico Plotly e todo elemento
de UI customizado (badges, cards) puxe as mesmas cores — a paleta principal
(azul-marinho/azul-elétrico) já vem do tema nativo do Streamlit em
`.streamlit/config.toml`; aqui replicamos os mesmos valores em Python para
uso nos gráficos, e definimos o coral/laranja de destaque para anomalias.
"""

import streamlit as st

NAVY = "#0A192F"
NAVY_LIGHT = "#112B4C"
NAVY_LIGHTER = "#1B3A63"
ELECTRIC_BLUE = "#2E8BFF"
CORAL = "#FF6B4A"
TEXT = "#E6EDF7"
MUTED = "#8AA0C4"

# Cor consistente para "legítima" (azul-elétrico) vs "fraude" (coral) em
# todos os gráficos do app — reforça a identidade visual do projeto.
CLASS_COLORS = {"Legítima": ELECTRIC_BLUE, "Fraude": CORAL}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=NAVY,
    plot_bgcolor=NAVY,
    font=dict(color=TEXT, family="sans-serif"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=48, b=40, l=40, r=24),
)


def apply_plotly_theme(fig):
    """Aplica o tema padrão a uma figura Plotly (cores de fundo, fonte, margens)."""
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor=NAVY_LIGHTER, zerolinecolor=NAVY_LIGHTER)
    fig.update_yaxes(gridcolor=NAVY_LIGHTER, zerolinecolor=NAVY_LIGHTER)
    return fig


def inject_css() -> None:
    """CSS complementar para elementos que o tema nativo do Streamlit não cobre
    (badges de resultado, cards do SWOT, cabeçalho com o conceito de vigilância).
    """
    st.markdown(
        f"""
        <style>
        .sentinel-header {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.2rem;
        }}
        .sentinel-header .icon {{
            font-size: 2.2rem;
        }}
        .sentinel-tagline {{
            color: {MUTED};
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }}
        .metric-card {{
            background: {NAVY_LIGHT};
            border: 1px solid {NAVY_LIGHTER};
            border-radius: 10px;
            padding: 1rem 1.2rem;
            height: 100%;
        }}
        .metric-card .value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: {ELECTRIC_BLUE};
        }}
        .metric-card .label {{
            color: {MUTED};
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .swot-card {{
            background: {NAVY_LIGHT};
            border-left: 4px solid {ELECTRIC_BLUE};
            border-radius: 6px;
            padding: 0.9rem 1.1rem;
            height: 100%;
        }}
        .swot-card.risk {{
            border-left-color: {CORAL};
        }}
        .swot-card h4 {{
            margin: 0 0 0.5rem 0;
            color: {TEXT};
        }}
        .swot-card ul {{
            margin: 0;
            padding-left: 1.1rem;
            color: {MUTED};
        }}
        .result-badge {{
            border-radius: 10px;
            padding: 1.1rem 1.4rem;
            font-size: 1.4rem;
            font-weight: 700;
            text-align: center;
            margin: 0.6rem 0 1rem 0;
        }}
        .result-badge.fraud {{
            background: rgba(255, 107, 74, 0.15);
            border: 1px solid {CORAL};
            color: {CORAL};
        }}
        .result-badge.normal {{
            background: rgba(46, 139, 255, 0.12);
            border: 1px solid {ELECTRIC_BLUE};
            color: {ELECTRIC_BLUE};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
