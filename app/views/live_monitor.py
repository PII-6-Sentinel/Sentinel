"""Página 'Monitor ao Vivo' — fluxo de transações em tempo real.

Com a transmissão ligada, a cada intervalo chega uma transação nova de um
cliente sorteado do banco sintético (compra normal ou um dos cenários de
fraude do Simulador, na proporção escolhida) e o Sentinel a avalia na hora:
Risk Score, nível, ação e motivos. O painel mostra os indicadores do fluxo,
a última transação em destaque, o mapa das compras, a linha do tempo dos
scores, os sinais que mais dispararam e o feed completo.

O ciclo ao vivo é um `st.fragment(run_every=...)`: só o painel é redesenhado
a cada transação, não a página inteira. O motor de risco é reaproveitado da
Central de Transações ou do Simulador se já tiver sido treinado nesta
sessão; senão, é treinado uma vez ao abrir a página.

Toda a lógica vive em `app/live_monitor.py` (que usa `src/`); aqui só há
orquestração e exibição. Nada é gravado no banco: o fluxo vive só em
`st.session_state`.
"""

import random

import plotly.graph_objects as go
import streamlit as st
from live_monitor import (
    MAX_EVENTS,
    PRESENTATION_SEED,
    LiveEvent,
    customer_ids,
    events_frame,
    next_event,
    reason_counts,
    scenario_detection,
    summarize,
)
from risk_analysis import resolve_db_path, run_analysis
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from theme import (
    ACCENT,
    DIVIDER,
    LEVEL_COLORS,
    RISK_HIGH,
    RISK_MEDIUM,
    SURFACE,
    SURFACE_RAISED,
    TEXT,
    TEXT_SECONDARY,
    apply_plotly_theme,
)
from ui import (
    ACTION_LABEL,
    card,
    empty_state,
    esc,
    facts_grid,
    fmt_brl,
    fmt_int,
    fmt_km,
    fmt_pct,
    level_badge,
    panel,
    reasons_list,
    score_badge,
    simulation_notice,
    style_levels,
)

from src.risk.engine import ACTION_BLOCK, ACTION_DESCRIPTION

CENTRAL_KEY = "risk_analysis"  # resultado da Central de Transações (ver views/transactions.py)
ENGINE_KEY = "sim_engine"  # mesmo formato do Simulador: {"engine", "note"}
EVENTS_KEY = "live_events"
SEQ_KEY = "live_seq"
RNG_KEY = "live_rng"
K_RUNNING, K_INTERVAL, K_FRAUD = "live_running", "live_interval", "live_fraud_rate"
K_PRESENTATION = "live_presentation"
MAX_TOASTS = 3  # alertas de bloqueio por rodada (o "Gerar 10" não vira uma enxurrada)

SIMULATION_NOTICE = (
    "fluxo de transações simuladas a partir do banco sintético; a coluna Cenário é o rótulo da simulação, "
    "não fraude real"
)
FEED_ROWS = 50


# ----------------------------------------------------------------------
# motor e geração do fluxo
# ----------------------------------------------------------------------


def _engine():
    """Reaproveita o motor já treinado na sessão (Central ou Simulador); senão treina."""
    central = st.session_state.get(CENTRAL_KEY)
    if central is not None:
        return central.engine
    held = st.session_state.get(ENGINE_KEY)
    if held is None:
        with st.spinner("Treinando o motor de risco (só na primeira vez nesta sessão)..."):
            result = run_analysis(resolve_db_path())
        cfg = result.engine.config
        note = (
            f"motor treinado pelo Monitor ao Vivo: regras {fmt_pct(cfg.behavior_weight, 0)} / "
            f"ML {fmt_pct(cfg.ml_weight, 0)}, faixas internas {cfg.bands.review_threshold}/{cfg.bands.block_threshold}"
        )
        held = {"engine": result.engine, "note": note}
        st.session_state[ENGINE_KEY] = held
    return held["engine"]


def _generate(n: int = 1) -> None:
    """Gera e avalia `n` transações novas e as acrescenta ao fluxo."""
    engine = _engine()
    if RNG_KEY not in st.session_state:
        presentation = st.session_state.get(K_PRESENTATION, False)
        st.session_state[RNG_KEY] = random.Random(PRESENTATION_SEED if presentation else None)
    rng = st.session_state[RNG_KEY]
    events: list[LiveEvent] = st.session_state.setdefault(EVENTS_KEY, [])
    start = len(events)
    fraud_rate = st.session_state.get(K_FRAUD, 20) / 100
    sa_engine = create_engine(f"sqlite:///{resolve_db_path()}")
    try:
        with Session(sa_engine) as session:
            ids = customer_ids(session)
            for _ in range(n):
                seq = st.session_state.get(SEQ_KEY, 0) + 1
                st.session_state[SEQ_KEY] = seq
                events.append(next_event(session, engine, rng, seq, fraud_rate, ids))
    finally:
        sa_engine.dispose()
    blocked = [e for e in events[start:] if e.assessment.action == ACTION_BLOCK]
    for e in blocked[-MAX_TOASTS:]:
        st.toast(
            f"**Bloqueada:** #{e.seq} · {e.customer_name} · {fmt_brl(e.amount)} em {e.city} "
            f"(Risk Score {e.assessment.risk_score})".replace("$", "\\$"),
            icon=":material/block:",
        )
    del events[:-MAX_EVENTS]


def _clear() -> None:
    """Zera o fluxo. O sorteio recomeça: no modo apresentação, do início da
    mesma sequência.
    """
    for key in (EVENTS_KEY, SEQ_KEY, RNG_KEY):
        st.session_state.pop(key, None)


# ----------------------------------------------------------------------
# componentes do painel
# ----------------------------------------------------------------------


def _fmt_ms(ms: float) -> str:
    return f"{ms:.0f} ms" if ms >= 10 else f"{ms:.1f} ms".replace(".", ",")


def _kpis(s: dict) -> None:
    detection = "—" if s["detection_rate"] is None else fmt_pct(s["detection_rate"], 0)
    false_alarm = "—" if s["false_alarm_rate"] is None else fmt_pct(s["false_alarm_rate"], 0)
    items = [
        card(
            "Transações",
            fmt_int(s["total"]),
            f"{fmt_brl(s['amount_total'])} no total · decisão em {_fmt_ms(s['avg_decision_ms'])} em média",
            size="xs",
            dot=False,
        ),
        card("Aprovadas", fmt_int(s["approved"]), "nível baixo", LEVEL_COLORS["baixo"], size="xs"),
        card("Em revisão", fmt_int(s["review"]), "nível médio", LEVEL_COLORS["médio"], size="xs"),
        card("Bloqueadas", fmt_int(s["blocked"]), "nível alto", LEVEL_COLORS["alto"], size="xs"),
        card(
            "Fraudes capturadas",
            detection,
            f"{s['caught']} de {s['injected']} injetadas · {false_alarm} de alarme falso",
            size="xs",
            dot=False,
        ),
        card("Valor retido", fmt_brl(s["amount_held"]), "em revisão ou bloqueado", RISK_HIGH, size="xs"),
    ]
    # duas linhas de três: valores em R$ não cabem em seis colunas
    for row in (items[:3], items[3:]):
        for col, html in zip(st.columns(3, gap="small"), row):
            col.markdown(html, unsafe_allow_html=True)


def _latest(e: LiveEvent) -> None:
    a, f = e.assessment, e.features
    color = LEVEL_COLORS[a.level]
    speed = f["implied_speed_kmh"]
    facts = [
        ("Cliente", f"{esc(e.customer_name)} (#{e.customer_id}) — mora em {esc(e.home_city)}"),
        ("Valor", f"{fmt_brl(e.amount)} ({f'{f["amount_deviation"] + 1:.1f}'.replace('.', ',')}x a média)"),
        ("Local", f"{esc(e.city)} · {fmt_km(f['distance_from_home_km'])} de casa"),
        ("Horário", f"{e.timestamp:%d/%m/%Y %H:%M}" + (" · fora da janela habitual" if f["unusual_hour"] > 0 else "")),
        ("Dispositivo", esc(e.device_label)),
        (
            "Velocidade desde a última compra",
            f"{fmt_int(speed)} km/h" + (" · IMPOSSÍVEL" if f["is_impossible_travel"] else ""),
        ),
        ("Cidade", "nunca visitada" if f["new_city"] else "conhecida"),
        ("Cenário simulado", esc(e.scenario_label)),
    ]
    left, mid, right = st.columns([0.8, 1.6, 1.4], gap="medium")
    with left:
        st.markdown(
            panel(
                f"Transação #{e.seq} · {e.received_at:%H:%M:%S}",
                score_badge(a.risk_score, a.level)
                + f'<div style="margin-top:16px">{level_badge(a.level)}</div>'
                + f'<div style="margin-top:12px;font-size:28px;font-weight:700;color:{color}">{ACTION_LABEL[a.action]}</div>'
                + f'<div style="color:{TEXT_SECONDARY};font-size:13px">{esc(ACTION_DESCRIPTION[a.action])}</div>'
                + f'<div style="color:{TEXT_SECONDARY};font-size:13px;margin-top:8px">ML {a.ml_score:.0f} · regras {a.behavior_score:.0f}</div>'
                + f'<div style="color:{TEXT_SECONDARY};font-size:13px">decidido em {_fmt_ms(e.decision_ms)}</div>',
            ),
            unsafe_allow_html=True,
        )
    mid.markdown(panel("Dados da transação", facts_grid(facts)), unsafe_allow_html=True)
    right.markdown(panel("Motivos", reasons_list(a.reasons, color)), unsafe_allow_html=True)


def _map(frame) -> go.Figure:
    """Todas as compras do fluxo, coloridas pelo nível; o tamanho segue o valor."""
    fig = go.Figure()
    # mais antigas primeiro, para as recentes ficarem por cima
    data = frame.iloc[::-1]
    max_amount = max(float(data["Valor"].max()), 1.0)
    for level in ("baixo", "médio", "alto"):
        part = data[data["Nível"] == level]
        if part.empty:
            continue
        fig.add_trace(
            go.Scattergeo(
                lat=part["lat"],
                lon=part["lon"],
                mode="markers",
                name=f"nível {level}",
                marker=dict(
                    size=8 + 22 * (part["Valor"] / max_amount) ** 0.5,
                    color=LEVEL_COLORS[level],
                    opacity=0.75,
                    line=dict(width=1, color=TEXT),
                ),
                customdata=part[["#", "Cliente", "Cidade", "Valor", "Risk Score", "Ação"]].values,
                hovertemplate="#%{customdata[0]} · %{customdata[1]}<br>%{customdata[2]} · R$ %{customdata[3]:,.2f}"
                "<br>Risk Score %{customdata[4]} → %{customdata[5]}<extra></extra>",
            )
        )
    fig.update_geos(
        lataxis_range=[-40, 55],
        lonaxis_range=[-95, 15],
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
    )
    apply_plotly_theme(fig)
    fig.update_layout(
        height=380,
        margin=dict(t=4, b=4, l=0, r=0),
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="left", x=0, font=dict(color=TEXT_SECONDARY, size=13)),
    )
    return fig


def _timeline(frame) -> go.Figure:
    """Risk Score de cada transação, em ordem de chegada, com as faixas 30 e 70."""
    data = frame.iloc[::-1]
    fig = go.Figure(
        go.Scatter(
            x=data["#"],
            y=data["Risk Score"],
            mode="lines+markers",
            line=dict(color=DIVIDER, width=1),
            marker=dict(size=9, color=[LEVEL_COLORS[lv] for lv in data["Nível"]]),
            customdata=data[["Cliente", "Cidade", "Cenário"]].values,
            hovertemplate="#%{x} · %{customdata[0]}<br>%{customdata[1]} · %{customdata[2]}"
            "<br>Risk Score %{y}<extra></extra>",
        )
    )
    fig.add_hline(y=30, line=dict(color=RISK_MEDIUM, dash="dot", width=1))
    fig.add_hline(y=70, line=dict(color=RISK_HIGH, dash="dot", width=1))
    apply_plotly_theme(fig)
    fig.update_layout(height=380, margin=dict(t=16, b=40, l=40, r=16), showlegend=False)
    fig.update_yaxes(range=[0, 102], title=None)
    fig.update_xaxes(title=dict(text="transação (ordem de chegada)", font=dict(color=TEXT_SECONDARY, size=12)))
    return fig


def _hbar(labels, values, color: str, text=None) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=color),
            text=text,
            textposition="auto",
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    apply_plotly_theme(fig)
    fig.update_layout(height=max(220, 44 * len(labels) + 60), margin=dict(t=8, b=32, l=8, r=16), showlegend=False)
    fig.update_yaxes(automargin=True)
    return fig


def _signals(events: list[LiveEvent]) -> None:
    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(key="sx-panel-live-reasons"):
            st.markdown('<div class="sx-panel-title">Sinais que mais dispararam (transações sinalizadas)</div>', unsafe_allow_html=True)
            counts = reason_counts(events)
            if counts.empty:
                st.caption("Nenhuma transação sinalizada ainda.")
            else:
                st.plotly_chart(_hbar(counts.index, counts.values, RISK_HIGH), key="live_reasons", theme=None)
    with right:
        with st.container(key="sx-panel-live-scenarios"):
            st.markdown('<div class="sx-panel-title">Sinalizadas por cenário simulado</div>', unsafe_allow_html=True)
            table = scenario_detection(events)
            text = [f"{r.Sinalizadas} de {r.Total} ({fmt_pct(r.Taxa, 0)})" for r in table.itertuples()]
            st.plotly_chart(_hbar(table["Cenário"], table["Sinalizadas"], ACCENT, text), key="live_scenarios", theme=None)
            st.caption("Em \"Compra normal\", cada sinalizada é um alarme falso; nos demais, uma fraude capturada.")


def _feed(frame) -> None:
    st.subheader(f"Feed de transações (últimas {min(FEED_ROWS, len(frame))})")
    table = frame.head(FEED_ROWS).drop(columns=["lat", "lon", "fraude_injetada"])
    table["Ação"] = table["Ação"].map(ACTION_LABEL)
    styled = table.style.format(
        {
            "Chegada": lambda t: t.strftime("%H:%M:%S"),
            "Data/hora": lambda t: t.strftime("%d/%m/%Y %H:%M"),
            "Valor": fmt_brl,
            "Distância (km)": lambda v: fmt_int(v),
            "Decisão (ms)": _fmt_ms,
        }
    ).apply(style_levels, axis=None)
    st.dataframe(styled, hide_index=True, key="live_feed")


# ----------------------------------------------------------------------
# página
# ----------------------------------------------------------------------


def _dashboard(running: bool) -> None:
    """Corpo ao vivo: com a transmissão ligada, gera uma transação por execução."""
    if running:
        _generate()
    events: list[LiveEvent] = st.session_state.get(EVENTS_KEY, [])
    if not events:
        with st.container(key="sx-panel-live-empty"):
            st.markdown(
                empty_state(
                    "empty",
                    "Nenhuma transação no fluxo",
                    "Ligue a transmissão para receber transações em tempo real, ou gere algumas manualmente.",
                ),
                unsafe_allow_html=True,
            )
        return

    frame = events_frame(events)
    _kpis(summarize(events))
    st.subheader("Última transação")
    _latest(events[-1])

    map_col, line_col = st.columns(2, gap="medium")
    with map_col:
        with st.container(key="sx-panel-live-map"):
            st.markdown('<div class="sx-panel-title">Onde as compras aconteceram</div>', unsafe_allow_html=True)
            st.plotly_chart(_map(frame), key="live_map", theme=None)
    with line_col:
        with st.container(key="sx-panel-live-timeline"):
            st.markdown('<div class="sx-panel-title">Risk Score ao longo do fluxo</div>', unsafe_allow_html=True)
            st.plotly_chart(_timeline(frame), key="live_timeline", theme=None)

    _signals(events)
    _feed(frame)


def render():
    st.title("Monitor ao Vivo")
    st.markdown(
        "Transações chegando em tempo real: cada uma é avaliada pelo Sentinel na hora — valor, localização, "
        "horário, dispositivo, cidade e velocidade desde a última compra — e recebe Risk Score, nível e ação."
    )
    simulation_notice(SIMULATION_NOTICE)

    if not resolve_db_path().exists():
        st.error(
            "**Banco sintético não encontrado.** Gere-o com `python scripts/generate_mock_data.py` "
            "(é determinístico: seed fixa) e recarregue a página.",
            icon=":material/database_off:",
        )
        return

    # duas linhas: em telas estreitas (projetor) cinco colunas cortavam os botões
    with st.container(key="live-controls"):
        c1, c2, c3, c4 = st.columns(4, vertical_alignment="center")
        running = c1.toggle("Transmitir ao vivo", key=K_RUNNING)
        c2.toggle(
            "Modo apresentação",
            key=K_PRESENTATION,
            on_change=_clear,
            help="Sequência fixa de transações: com os mesmos controles, a demonstração sai sempre igual. "
            "Ligar ou desligar recomeça o fluxo.",
        )
        if c3.button("Gerar 10 agora", key="live_burst", disabled=running, use_container_width=True):
            _generate(10)
        c4.button("Limpar fluxo", key="live_clear", on_click=_clear, use_container_width=True)
        s1, s2 = st.columns(2)
        interval = s1.select_slider(
            "Intervalo entre transações", options=[0.5, 1.0, 2.0, 3.0, 5.0], value=2.0, key=K_INTERVAL,
            format_func=lambda s: f"{s:g} s".replace(".", ","),
        )  # fmt: skip
        s2.slider("Cenários de fraude no fluxo (%)", 0, 100, 20, step=5, key=K_FRAUD)

    # só o painel é redesenhado a cada transação; desligado, não há ciclo
    st.fragment(_dashboard, run_every=interval if running else None)(running)
