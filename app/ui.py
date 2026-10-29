"""Peças de interface compartilhadas pelas páginas do produto: formatação
pt-BR, componentes HTML (cartões, KPIs, selos, estados vazios, matriz...) e o
mapa de localização. Só apresentação — nenhuma lógica de risco aqui.

Os componentes devolvem HTML numa única linha (sem linhas em branco nem
recuo de 4 espaços, que o markdown do Streamlit trataria como bloco de
código) para ir em `st.markdown(..., unsafe_allow_html=True)`. O visual vem
de `app/assets/theme.css`; as cores, de `app/theme.py`.
"""

import base64
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from theme import (
    ACCENT,
    ASSETS_DIR,
    DIVIDER,
    LEVEL_COLORS,
    LEVEL_TINTS,
    RISK_HIGH,
    SURFACE,
    SURFACE_RAISED,
    TEXT,
    TEXT_SECONDARY,
    apply_plotly_theme,
)

# Gráficos sem a barra de ferramentas do Plotly (câmera, zoom, ...): numa
# apresentação ela só polui; o hover continua funcionando.
CHART_CONFIG = {"displayModeBar": False}

LEVEL_CSS = {"baixo": "baixo", "médio": "medio", "alto": "alto"}
ACTION_LABEL = {"aprovar": "Aprovar", "revisar": "Revisar", "bloquear": "Bloquear"}


# ----------------------------------------------------------------------
# formatação (pt-BR)
# ----------------------------------------------------------------------


def fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def fmt_pct(fraction: float, decimals: int = 1) -> str:
    return f"{fraction * 100:.{decimals}f}".replace(".", ",") + "%"


def fmt_brl(value: float) -> str:
    return "R$ " + f"{value:,.2f}".replace(",", "#").replace(".", ",").replace("#", ".")


def md_brl(value: float) -> str:
    """`fmt_brl` para texto markdown: o `$` precisa de barra, senão o Streamlit
    trata o trecho entre dois `$` como fórmula LaTeX. (Dentro de HTML não é
    preciso: o bloco HTML não passa pelo parser de fórmulas.)
    """
    return fmt_brl(value).replace("$", "\\$")


def fmt_km(value: float) -> str:
    return fmt_int(round(value)) + " km"


def fmt_seconds(seconds: float) -> str:
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.2f} s".replace(".", ",")


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def esc(text) -> str:
    """Escapa texto dinâmico para entrar em HTML."""
    return escape(str(text), quote=False)


# ----------------------------------------------------------------------
# ícones (SVG próprio, traço 1,6, herdam a cor do texto)
# ----------------------------------------------------------------------

_SVG = '<svg class="{cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{body}</svg>'
_ICON_BODIES = {
    "dollar": '<circle cx="12" cy="12" r="9"/><path d="M14.6 9.4c-.6-.8-1.5-1.2-2.6-1.2-1.4 0-2.5.8-2.5 1.9s1 1.6 2.5 1.9 2.5.8 2.5 1.9-1.1 1.9-2.5 1.9c-1.1 0-2-.4-2.6-1.2M12 6.4v1.8M12 15.8v1.8"/>',
    "pin": '<path d="M12 21s6-5.6 6-10.5a6 6 0 1 0-12 0C6 15.4 12 21 12 21z"/><circle cx="12" cy="10.5" r="2.2"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "phone": '<rect x="7" y="3" width="10" height="18" rx="2.5"/><path d="M11 18h2"/>',
    "empty": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/>',
    "warning": '<path d="M12 3.5 2.5 20h19L12 3.5z"/><path d="M12 10v4.5M12 17.2v.1"/>',
}


def icon(name: str, cls: str = "sx-signal-icon") -> str:
    return _SVG.format(cls=cls, body=_ICON_BODIES[name])


def brand_html() -> str:
    """Logo da barra lateral: ícone (app/assets/sentinel-icon.svg) + nome + legenda."""
    svg = (ASSETS_DIR / "sentinel-icon.svg").read_bytes()
    src = "data:image/svg+xml;base64," + base64.b64encode(svg).decode()
    return (
        '<div class="sx-brand">'
        f'<img src="{src}" alt="Sentinel">'
        '<div><div class="name">Sentinel</div><div class="tag">Análise de risco transacional</div></div>'
        "</div>"
    )


# ----------------------------------------------------------------------
# componentes
# ----------------------------------------------------------------------


def level_badge(level: str) -> str:
    """Selo de nível: cor a 100% sobre a mesma cor a 14%, sempre com texto."""
    return f'<span class="level-badge {LEVEL_CSS[level]}">nível {esc(level)}</span>'


def card(
    label: str,
    value: str,
    sub: str = "",
    accent: str | None = None,
    value_color: str | None = None,
    size: str = "",
    dot: bool = True,
) -> str:
    """Cartão de KPI: rótulo (com um ponto da cor do nível), número grande e
    legenda. `accent` colore o ponto; `value_color` colore o número;
    `size` = "sm" (44 px, eficiência) ou "xs" (32 px).
    """
    dot_style = f' style="background:{accent}"' if accent else ""
    dot_html = f'<span class="dot"{dot_style}></span>' if dot else ""
    value_style = f' style="color: {value_color};"' if value_color else ""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card {size}"><div class="label">{dot_html}{label}</div>'
        f'<div class="value"{value_style}>{value}</div>{sub_html}</div>'
    )


# Ressalva única do protótipo: o motor roda sobre dados SINTÉTICOS, cujos
# rótulos foram gerados com os mesmos sinais que ele usa. Toda métrica de
# desempenho mostrada no app vem acompanhada dela (`metrics_caveat`).
PROTOTYPE_NOTICE = (
    "protótipo demonstrativo com dados sintéticos; as métricas validam a lógica do sistema, "
    "não o desempenho contra fraude real"
)


def metrics_caveat() -> None:
    """Ressalva curta, logo abaixo de um bloco de métricas."""
    st.caption(
        ":material/science: **Protótipo demonstrativo.** Métricas calculadas sobre dados sintéticos, gerados com "
        "os mesmos sinais que o motor usa: validam a lógica do sistema, não provam desempenho contra fraude real."
    )


def simulation_notice(text: str) -> None:
    """Aviso de que os dados são simulados (bloco em tinta âmbar a 10%)."""
    st.warning(f"**Aviso:** {text}.", icon=":material/warning:")


def empty_state(icon_name: str, title: str, text: str) -> str:
    """Estado vazio: ícone, título e texto. O botão (um widget do Streamlit) e
    os chips (`chips_html`) vêm depois, no mesmo cartão.
    """
    return f'<div class="sx-empty">{icon(icon_name, "icon")}<h4>{esc(title)}</h4><p>{esc(text)}</p></div>'


def chips_html(chips: list[tuple[str, str]]) -> str:
    """Chips do pipeline: (rótulo curto, nome completo da etapa como dica)."""
    items = "".join(f'<span title="{esc(tip)}">{esc(label)}</span>' for label, tip in chips)
    return f'<div class="sx-chips">{items}</div>'


def step_card(number: int, title: str, text: str) -> str:
    return (
        f'<div class="sx-card pad24"><div class="sx-step-num">{number}</div>'
        f"<h4>{esc(title)}</h4><p>{esc(text)}</p></div>"
    )


def signal_card(icon_name: str, title: str, text: str) -> str:
    return f'<div class="sx-card pad24">{icon(icon_name)}<h4>{esc(title)}</h4><p>{esc(text)}</p></div>'


def layer_card(eyebrow: str, title: str, body_html: str) -> str:
    return f'<div class="sx-card"><div class="sx-eyebrow">{esc(eyebrow)}</div><h4>{esc(title)}</h4>{body_html}</div>'


def numbered_list(items: list[tuple[str, str]]) -> str:
    """Lista numerada com divisores: (termo em destaque, descrição)."""
    rows = "".join(
        f'<li><span class="n">{i}</span><span><b>{esc(term)}</b> {esc(desc)}</span></li>'
        for i, (term, desc) in enumerate(items, start=1)
    )
    return f'<ul class="sx-list">{rows}</ul>'


def grid(cards: list[str], columns: int) -> str:
    return f'<div class="sx-grid g{columns}">' + "".join(cards) + "</div>"


def reasons_list(reasons: list[str], color: str = RISK_HIGH) -> str:
    """Motivos com um marcador na cor do nível."""
    if not reasons:
        return "<p>Nenhum sinal relevante: a transação está dentro do padrão do cliente.</p>"
    rows = "".join(f'<li style="--dot:{color}">{esc(r)}</li>' for r in reasons)
    return f'<ul class="sx-list sx-reasons" style="--dot:{color}">{rows}</ul>'


def facts_grid(pairs: list[tuple[str, str]]) -> str:
    """Ficha chave/valor em duas colunas."""
    cells = "".join(f'<div><div class="k">{esc(k)}</div><div class="v">{v}</div></div>' for k, v in pairs)
    return f'<div class="sx-facts">{cells}</div>'


def panel(title: str, body_html: str) -> str:
    return f'<div class="sx-card pad24"><div class="sx-panel-title">{esc(title)}</div>{body_html}</div>'


def kv_table(rows: list[tuple[str, str]]) -> str:
    """Tabela de duas colunas (rótulo cinza, valor em destaque), com divisores."""
    body = "".join(
        f'<tr><td class="dim" style="text-align:left;width:38%">{esc(k)}</td><td style="text-align:left">{v}</td></tr>'
        for k, v in rows
    )
    return f'<table class="sx-table">{body}</table>'


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table class="sx-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def bar_rows(rows: list[tuple[str, str, float, str]]) -> str:
    """Linhas com mini-barra: (nome, "7 de 11", fração 0-1, "63,6%")."""
    items = "".join(
        '<div class="row"><div class="top">'
        f'<span>{esc(name)}</span><span class="r">{esc(detail)}<b>{esc(pct)}</b></span></div>'
        f'<div class="sx-bar"><i style="width:{max(0.0, min(1.0, frac)) * 100:.1f}%;background:{ACCENT}"></i></div></div>'
        for name, detail, frac, pct in rows
    )
    return f'<div class="sx-bars">{items}</div>'


def score_badge(score: int, level: str, label: str = "Risk Score") -> str:
    """Número grande na cor do nível, com uma mini-barra (0-100) embaixo."""
    color = LEVEL_COLORS[level]
    return (
        f'<div class="sx-score-wrap"><div class="k" style="font-size:13px;line-height:20px;color:{TEXT_SECONDARY}">{esc(label)}</div>'
        f'<div class="sx-score"><span class="num" style="color:{color}">{score}</span></div>'
        f'<div class="sx-bar" style="margin-top:8px"><i style="width:{max(0, min(100, score))}%;background:{color}"></i></div></div>'
    )


def confusion_matrix_html(tn: int, fp: int, fn: int, tp: int) -> str:
    """Matriz de confusão: células de raio 16, azul = acerto, coral = erro."""
    cells = [
        ("Legítima", tn, "Acertou (aprovou legítima)", "ok", fp, "Falso positivo", "err"),
        ("Fraude", fn, "Fraude perdida", "err", tp, "Fraude capturada", "ok"),
    ]
    rows = ""
    for row_label, left, left_text, left_kind, right, right_text, right_kind in cells:
        rows += (
            f'<div class="rh">{row_label}</div>'
            f'<div class="cell {left_kind}"><div class="n">{fmt_int(left)}</div><div class="t">{left_text}</div></div>'
            f'<div class="cell {right_kind}"><div class="n">{fmt_int(right)}</div><div class="t">{right_text}</div></div>'
        )
    return (
        '<div class="sx-matrix"><div></div><div class="hd">Aprovou</div><div class="hd">Sinalizou (revisar/bloquear)</div>'
        + rows
        + "</div>"
        '<div style="text-align:center;font-size:13px;line-height:20px;color:' + TEXT_SECONDARY + ';margin-top:12px">'
        "Decisão do sistema × rótulo da simulação</div>"
    )


# ----------------------------------------------------------------------
# mapa: residência (azul), compra (coral) e a linha entre elas
# ----------------------------------------------------------------------


def location_map(row) -> go.Figure:
    """Residência do cliente, local da compra e a linha entre eles."""
    home = (float(row["home_lat"]), float(row["home_lon"]))
    shop = (float(row["tx_lat"]), float(row["tx_lon"]))

    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lat=[home[0], shop[0]],
            lon=[home[1], shop[1]],
            mode="lines",
            line=dict(width=2, color=RISK_HIGH, dash="dash"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lat=[home[0]],
            lon=[home[1]],
            mode="markers",
            marker=dict(size=13, color=ACCENT, line=dict(width=2, color=TEXT)),
            name=f"Residência — {row['home_city']}",
            hovertemplate=f"Residência<br>{row['home_city']}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lat=[shop[0]],
            lon=[shop[1]],
            mode="markers",
            marker=dict(size=13, color=RISK_HIGH, symbol="diamond", line=dict(width=2, color=TEXT)),
            name=f"Compra — {row['city']}",
            hovertemplate=f"Compra<br>{row['city']}<extra></extra>",
        )
    )

    # enquadra os dois pontos com folga; abre um mínimo para a mesma cidade
    # não virar um zoom absurdo
    lats, lons = [home[0], shop[0]], [home[1], shop[1]]

    def span(values, minimum):
        mid, half = (min(values) + max(values)) / 2, max((max(values) - min(values)) / 2 * 1.5, minimum)
        return mid - half, mid + half

    lat0, lat1 = span(lats, 4.0)
    lon0, lon1 = span(lons, 6.0)
    # o geo mantém a proporção do recorte; alarga o eixo mais curto até a
    # proporção do contêiner (~1,4:1) para o mapa ocupar a área disponível
    lat_mid, lon_mid = (lat0 + lat1) / 2, (lon0 + lon1) / 2
    half_lat, half_lon = (lat1 - lat0) / 2, (lon1 - lon0) / 2
    half_lon = max(half_lon, half_lat * 1.4)
    half_lat = max(half_lat, half_lon / 1.4)
    lat0, lat1 = lat_mid - half_lat, lat_mid + half_lat
    lon0, lon1 = lon_mid - half_lon, lon_mid + half_lon
    fig.update_geos(
        lataxis_range=[max(lat0, -85), min(lat1, 85)],
        lonaxis_range=[max(lon0, -180), min(lon1, 180)],
        showland=True,
        landcolor=SURFACE_RAISED,
        showocean=True,
        oceancolor="#0A0A0B",
        showcountries=True,
        countrycolor=SURFACE,
        coastlinecolor=SURFACE,
        showlakes=False,
        showframe=False,
        bgcolor="rgba(0,0,0,0)",
        lonaxis=dict(showgrid=True, gridcolor=DIVIDER, dtick=30),
        lataxis=dict(showgrid=True, gridcolor=DIVIDER, dtick=30),
    )
    apply_plotly_theme(fig)
    fig.update_layout(
        height=330,
        margin=dict(t=4, b=4, l=0, r=0),
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="left", x=0, font=dict(color=TEXT_SECONDARY, size=13)),
    )
    return fig


def style_levels(display):
    """Colore a tabela pelo nível (vetorizado): o Risk Score ganha a cor do
    nível em negrito; a coluna Nível, a cor sobre o fundo da mesma cor a 14%
    (aproximação do selo — o st.dataframe não desenha cantos arredondados).
    """
    styles = pd.DataFrame("", index=display.index, columns=display.columns)
    levels = display["Nível"]
    styles["Risk Score"] = levels.map(lambda lv: f"color: {LEVEL_COLORS[lv]}; font-weight: 700")
    styles["Nível"] = levels.map(
        lambda lv: f"background-color: {LEVEL_TINTS[lv]}; color: {LEVEL_COLORS[lv]}; font-weight: 700"
    )
    return styles

