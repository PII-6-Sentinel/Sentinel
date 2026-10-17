"""Peças de interface compartilhadas pelas páginas do produto (Central de
Transações e Simulador): formatação pt-BR, cartões de métrica e o mapa de
localização. Só apresentação — nenhuma lógica de risco aqui.
"""

import pandas as pd
import plotly.graph_objects as go
from theme import CORAL, ELECTRIC_BLUE, LEVEL_COLORS, NAVY, NAVY_LIGHT, NAVY_LIGHTER, TEXT, apply_plotly_theme


LEVEL_CSS = {"baixo": "baixo", "médio": "medio", "alto": "alto"}
ACTION_LABEL = {"aprovar": "Aprovar", "revisar": "Revisar", "bloquear": "Bloquear"}


def fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def fmt_pct(fraction: float, decimals: int = 1) -> str:
    return f"{fraction * 100:.{decimals}f}".replace(".", ",") + "%"


def fmt_brl(value: float) -> str:
    return "R$ " + f"{value:,.2f}".replace(",", "#").replace(".", ",").replace("#", ".")


def md_brl(value: float) -> str:
    """`fmt_brl` para texto markdown: o `$` precisa de barra, senão o Streamlit
    trata o trecho entre dois `$` como fórmula LaTeX.
    """
    return fmt_brl(value).replace("$", "\\$")


def fmt_km(value: float) -> str:
    return fmt_int(round(value)) + " km"


def fmt_seconds(seconds: float) -> str:
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.2f} s".replace(".", ",")


def card(label: str, value: str, sub: str = "", accent: str | None = None) -> str:
    style = f' style="border-left: 4px solid {accent};"' if accent else ""
    value_style = f' style="color: {accent};"' if accent else ""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card"{style}><div class="label">{label}</div>'
        f'<div class="value"{value_style}>{value}</div>{sub_html}</div>'
    )


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


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
            line=dict(width=2, color=CORAL, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lat=[home[0]],
            lon=[home[1]],
            mode="markers",
            marker=dict(size=13, color=ELECTRIC_BLUE, line=dict(width=2, color=TEXT)),
            name=f"Residência — {row['home_city']}",
            hovertemplate=f"Residência<br>{row['home_city']}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lat=[shop[0]],
            lon=[shop[1]],
            mode="markers",
            marker=dict(size=13, color=CORAL, symbol="diamond", line=dict(width=2, color=TEXT)),
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
        landcolor=NAVY_LIGHT,
        showocean=True,
        oceancolor=NAVY,
        showcountries=True,
        countrycolor=NAVY_LIGHTER,
        coastlinecolor=NAVY_LIGHTER,
        showframe=False,
        bgcolor=NAVY,
    )
    apply_plotly_theme(fig)
    fig.update_layout(
        height=380,
        margin=dict(t=10, b=10, l=0, r=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="left", x=0),
    )
    return fig


def style_levels(display):
    """Colore Risk Score e Nível pela cor do nível (vetorizado)."""
    styles = pd.DataFrame("", index=display.index, columns=display.columns)
    for column in ("Risk Score", "Nível"):
        styles[column] = display["Nível"].map(
            lambda lv: f"background-color: {rgba(LEVEL_COLORS[lv], 0.22)}; color: {LEVEL_COLORS[lv]}; font-weight: 700"
        )
    return styles
