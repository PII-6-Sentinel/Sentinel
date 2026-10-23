"""Identidade visual do Sentinel — tema escuro, estilo Apple.

Aqui ficam os TOKENS de design (a fonte da verdade, vinda do protótipo
"Sentinel — Redesign Dark") e a injeção do CSS global. O CSS em si vive em
`app/assets/theme.css` e só usa variáveis (`var(--...)`) geradas a partir
destas constantes, então trocar um token aqui troca em todo lugar.

Os tokens também servem aos gráficos Plotly (`apply_plotly_theme`) e aos
componentes HTML de `app/ui.py`.

`.streamlit/config.toml` repete as cores principais para o tema escuro valer
mesmo com o sistema operacional em modo claro e para o que o CSS não alcança
(o canvas do `st.dataframe`).
"""

from pathlib import Path

import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# ----------------------------------------------------------------------
# cores
# ----------------------------------------------------------------------
BG = "#000000"  # fundo da página e da barra lateral
SURFACE = "#1C1C1E"  # cartões e tabelas
SURFACE_RAISED = "#2C2C2E"  # campos, chips, selects
BORDER = "rgba(255,255,255,0.08)"  # borda de cartões e divisores
DIVIDER = "rgba(255,255,255,0.06)"  # linhas de tabela e de listas

TEXT = "#F5F5F7"  # texto principal
TEXT_SECONDARY = "#86868B"  # rótulos e legendas
# Só para o que não carrega informação (placeholders, numeração decorativa):
# sobre #000 dá ~3,4:1, abaixo dos 4,5:1 exigidos para texto.
TEXT_TERTIARY = "#636366"

ACCENT = "#3D8BFF"  # destaque, gráficos, risco baixo
ACCENT_STRONG = "#1F6FEB"  # botão primário (texto branco, 4,6:1)
RISK_MEDIUM = "#FFB020"  # nível médio
RISK_HIGH = "#FF6B57"  # nível alto / anomalia
LOGO_RING = "#9DB8F0"  # anel do ícone sobre fundo escuro

ACCENT_TINT = "rgba(61,139,255,0.14)"  # fundo do selo "baixo"
ACCENT_TINT_SELECTED = "rgba(61,139,255,0.10)"  # linha selecionada
MEDIUM_TINT = "rgba(255,176,32,0.14)"
HIGH_TINT = "rgba(255,107,87,0.14)"
WARNING_TINT = "rgba(255,176,32,0.10)"  # bloco do aviso de dados simulados

# Nível de risco -> cor (selo = cor a 100% sobre fundo da mesma cor a 14%).
LEVEL_COLORS = {"baixo": ACCENT, "médio": RISK_MEDIUM, "alto": RISK_HIGH}
LEVEL_TINTS = {"baixo": ACCENT_TINT, "médio": MEDIUM_TINT, "alto": HIGH_TINT}

# ----------------------------------------------------------------------
# tipografia, raios e espaçamento
# ----------------------------------------------------------------------
FONT_STACK = 'Inter, -apple-system, "SF Pro Display", "Helvetica Neue", sans-serif'
FONT_IMPORT = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"

RADIUS_CARD = 20  # cartões e painéis
RADIUS_CELL = 16  # células da matriz e avisos
RADIUS_FIELD = 12  # campos e itens de menu
RADIUS_PILL = 999  # botões, chips e selos

SPACE_PAGE_MARGIN = 64  # margem lateral da página
SPACE_SECTION = 80  # distância entre seções

# ----------------------------------------------------------------------
# Plotly: um único template escuro
# ----------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, family=FONT_STACK, size=13),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SECONDARY)),
    margin=dict(t=48, b=40, l=40, r=24),
    colorway=[ACCENT, RISK_HIGH, RISK_MEDIUM, LOGO_RING, TEXT_SECONDARY],
)


def apply_plotly_theme(fig):
    """Aplica o template escuro a uma figura Plotly: fundos transparentes (o
    cartão por trás dá a cor), grade a 6% de branco, fonte Inter.
    """
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor=DIVIDER, zerolinecolor=DIVIDER, linecolor=DIVIDER, tickfont=dict(color=TEXT_SECONDARY))
    fig.update_yaxes(gridcolor=DIVIDER, zerolinecolor=DIVIDER, linecolor=DIVIDER, tickfont=dict(color=TEXT_SECONDARY))
    return fig


# ----------------------------------------------------------------------
# CSS global
# ----------------------------------------------------------------------
def css_variables() -> str:
    """Bloco `:root` com os tokens, para o CSS usar `var(--...)`."""
    values = {
        "bg": BG,
        "surface": SURFACE,
        "surface-raised": SURFACE_RAISED,
        "border": BORDER,
        "divider": DIVIDER,
        "text": TEXT,
        "text-secondary": TEXT_SECONDARY,
        "text-tertiary": TEXT_TERTIARY,
        "accent": ACCENT,
        "accent-strong": ACCENT_STRONG,
        "risk-medium": RISK_MEDIUM,
        "risk-high": RISK_HIGH,
        "logo-ring": LOGO_RING,
        "accent-tint": ACCENT_TINT,
        "accent-tint-selected": ACCENT_TINT_SELECTED,
        "medium-tint": MEDIUM_TINT,
        "high-tint": HIGH_TINT,
        "warning-tint": WARNING_TINT,
        "radius-card": f"{RADIUS_CARD}px",
        "radius-cell": f"{RADIUS_CELL}px",
        "radius-field": f"{RADIUS_FIELD}px",
        "radius-pill": f"{RADIUS_PILL}px",
        "page-margin": f"{SPACE_PAGE_MARGIN}px",
        "section-gap": f"{SPACE_SECTION}px",
        "font": FONT_STACK,
    }
    body = "\n".join(f"  --{name}: {value};" for name, value in values.items())
    return f":root {{\n{body}\n}}"


def build_css() -> str:
    """Todo o CSS do app: fonte (com fallback — funciona offline), variáveis e
    o arquivo `assets/theme.css`.
    """
    stylesheet = (ASSETS_DIR / "theme.css").read_text(encoding="utf-8")
    return f"@import url('{FONT_IMPORT}');\n{css_variables()}\n{stylesheet}"


def inject_css() -> None:
    """Injeta o CSS global (chamado uma vez por execução, em app/app.py)."""
    st.markdown(f"<style>{build_css()}</style>", unsafe_allow_html=True)
