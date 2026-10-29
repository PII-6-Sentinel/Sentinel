"""Página 'Monitor ao Vivo' — fluxo de transações em tempo real.

Com a transmissão ligada, a cada intervalo chega uma transação nova de um
cliente sorteado do banco sintético, gerada pelo gerador ao vivo (compra
normal ou um dos 6 cenários de fraude, na proporção escolhida). O Sentinel
a avalia na hora e dá o veredito — "Fraudulenta" ou "Legítima" —, mostrado
ao lado do rótulo real com que a transação foi gerada (que o motor não vê).

O painel mostra os indicadores do fluxo (matriz de confusão ao vivo,
acurácia, precisão, recall, valor retido), a última transação em destaque,
o mapa das compras, a linha do tempo dos scores, os sinais que mais
dispararam e o feed completo.

O ciclo ao vivo é um `st.fragment(run_every=...)`: só o painel é redesenhado
a cada transação, não a página inteira. O motor de risco é reaproveitado da
Central de Transações ou do Simulador se já tiver sido treinado nesta
sessão; senão, é treinado uma vez ao abrir a página.

Toda a lógica vive em `app/live_monitor.py` (que usa `src/`); aqui só há
orquestração e exibição. Nada é gravado no banco: o fluxo vive só em
`st.session_state`.
"""

import math
import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from live_monitor import (
    MAX_EVENTS,
    PRESENTATION_SEED,
    VERDICT_FRAUD,
    LiveEvent,
    customer_ids,
    events_frame,
    next_event,
    reason_counts,
    scenario_detection,
    summarize,
)
from risk_analysis import database_problem, resolve_db_path, run_analysis
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from theme import (
    ACCENT,
    DIVIDER,
    HIGH_TINT,
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
    CHART_CONFIG,
    ACTION_LABEL,
    PROTOTYPE_NOTICE,
    card,
    confusion_matrix_html,
    empty_state,
    esc,
    facts_grid,
    fmt_brl,
    fmt_int,
    fmt_km,
    fmt_pct,
    level_badge,
    metrics_caveat,
    panel,
    reasons_list,
    score_badge,
    simulation_notice,
    style_levels,
)

CENTRAL_KEY = "risk_analysis"  # resultado da Central de Transações (ver views/transactions.py)
ENGINE_KEY = "sim_engine"  # mesmo formato do Simulador: {"engine", "note"}
EVENTS_KEY = "live_events"
SEQ_KEY = "live_seq"
RNG_KEY = "live_rng"
K_RUNNING, K_INTERVAL, K_FRAUD = "live_running", "live_interval", "live_fraud_rate"
K_PRESENTATION = "live_presentation"
MAX_TOASTS = 3  # alertas por rodada (o "Gerar 10" não vira uma enxurrada)

SIMULATION_NOTICE = (
    f"{PROTOTYPE_NOTICE}. O fluxo vem do gerador do domínio sintético, e o \"rótulo real\" é o do gerador, "
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
    flagged = [e for e in events[start:] if e.flagged]
    for e in flagged[-MAX_TOASTS:]:
        mark = "" if e.correct else " · alarme falso"
        # curto de propósito: com 3 linhas ou mais o Streamlit corta o texto e
        # mostra um "view more" que não acrescenta nada (o detalhe está no feed)
        st.toast(
            f"**#{e.seq} {VERDICT_FRAUD.lower()}** · {ACTION_LABEL[e.assessment.action].lower()} · "
            f"{fmt_brl(e.amount)}{mark}".replace("$", "\\$"),
            icon=":material/gpp_bad:",
            duration="long",
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


def _pct(value: float | None) -> str:
    return "—" if value is None else fmt_pct(value, 0)


def _kpis(s: dict) -> None:
    items = [
        card(
            "Transações",
            fmt_int(s["total"]),
            f"{fmt_brl(s['amount_total'])} no total · decisão em {_fmt_ms(s['avg_decision_ms'])} em média",
            size="xs",
            dot=False,
        ),
        card(
            "Veredito: fraudulentas",
            fmt_int(s["flagged"]),
            f"{s['blocked']} bloqueadas · {s['review']} em revisão",
            RISK_HIGH,
            size="xs",
        ),
        card("Veredito: legítimas", fmt_int(s["total"] - s["flagged"]), "aprovadas automaticamente", ACCENT, size="xs"),
        card(
            "Acurácia",
            _pct(s["accuracy"]),
            f"veredito igual ao rótulo real em {s['tp'] + s['tn']} de {s['total']}",
            size="xs",
            dot=False,
        ),
        card(
            "Fraudes capturadas (recall)",
            _pct(s["recall"]),
            f"{s['tp']} de {s['frauds']} fraudes reais · precisão {_pct(s['precision'])}",
            size="xs",
            dot=False,
        ),
        card(
            "Valor retido",
            fmt_brl(s["amount_held"]),
            f"{fmt_brl(s['fraud_amount_held'])} era fraude de verdade",
            RISK_HIGH,
            size="xs",
        ),
    ]
    for row in (items[:3], items[3:]):
        for col, html in zip(st.columns(3, gap="small"), row):
            col.markdown(html, unsafe_allow_html=True)
    metrics_caveat()


def _verdict_html(e: LiveEvent) -> str:
    """Veredito do motor ao lado do rótulo real, com o selo de acerto/erro."""
    color = RISK_HIGH if e.flagged else ACCENT
    label_color = RISK_HIGH if e.is_fraud else ACCENT
    mark = (
        f'<span style="color:{ACCENT};font-weight:700">✓ acertou</span>'
        if e.correct
        else f'<span style="background:{HIGH_TINT};color:{RISK_HIGH};font-weight:700;padding:2px 8px;border-radius:999px">✗ errou</span>'
    )
    scenario = "" if not e.is_fraud else f'<div style="color:{TEXT_SECONDARY};font-size:13px">{esc(e.scenario_label)}</div>'
    return (
        '<div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start">'
        f'<div><div style="color:{TEXT_SECONDARY};font-size:13px">Veredito do Sentinel</div>'
        f'<div style="font-size:30px;font-weight:700;color:{color}">{e.verdict}</div></div>'
        f'<div><div style="color:{TEXT_SECONDARY};font-size:13px">Rótulo real</div>'
        f'<div style="font-size:30px;font-weight:700;color:{label_color}">{e.label}</div>{scenario}</div>'
        "</div>"
        f'<div style="margin-top:8px">{mark}</div>'
    )


def _latest(e: LiveEvent) -> None:
    a, f = e.assessment, e.features
    color = LEVEL_COLORS[a.level]
    speed = f["implied_speed_kmh"]
    ratio = f"{f['amount_deviation'] + 1:.1f}".replace(".", ",")
    facts = [
        ("Cliente", f"{esc(e.customer_name)} (#{e.customer_id}) — mora em {esc(e.home_city)}"),
        ("Valor", f"{fmt_brl(e.amount)} ({ratio}x a média)"),
        ("Local", f"{esc(e.city)} · {fmt_km(f['distance_from_home_km'])} de casa"),
        ("Canal", esc(e.channel_label) + (" · canal raro para o cliente" if f["channel_rarity"] >= 0.7 else "")),
        ("Horário", f"{e.timestamp:%d/%m/%Y %H:%M}" + (" · fora da janela habitual" if f["unusual_hour"] > 0 else "")),
        ("Dispositivo", esc(e.device_label)),
        (
            "Velocidade desde a última compra",
            f"{fmt_int(speed)} km/h" + (" · IMPOSSÍVEL" if f["is_impossible_travel"] else ""),
        ),
        ("Cidade", "nunca visitada" if f["new_city"] else "conhecida"),
    ]
    left, mid, right = st.columns([1.1, 1.5, 1.2], gap="medium")
    with left:
        st.markdown(
            panel(
                f"Transação #{e.seq} · {e.received_at:%H:%M:%S}",
                _verdict_html(e)
                + f'<div style="margin-top:20px">{score_badge(a.risk_score, a.level)}</div>'
                + f'<div style="margin-top:12px">{level_badge(a.level)} '
                f'<span style="color:{color};font-weight:700;margin-left:6px">{ACTION_LABEL[a.action]}</span></div>'
                + f'<div style="color:{TEXT_SECONDARY};font-size:13px;margin-top:8px">ML {a.ml_score:.0f} · regras '
                f"{a.behavior_score:.0f} · decidido em {_fmt_ms(e.decision_ms)}</div>",
            ),
            unsafe_allow_html=True,
        )
    mid.markdown(panel("Dados da transação", facts_grid(facts)), unsafe_allow_html=True)
    right.markdown(panel("Motivos", reasons_list(a.reasons, color)), unsafe_allow_html=True)


def _map(frame) -> go.Figure:
    """Todas as compras do fluxo, coloridas pelo veredito; o tamanho segue o valor."""
    fig = go.Figure()
    data = frame.iloc[::-1]  # mais antigas primeiro, para as recentes ficarem por cima
    max_amount = max(float(data["Valor"].max()), 1.0)
    for verdict, color in (("Legítima", ACCENT), (VERDICT_FRAUD, RISK_HIGH)):
        part = data[data["Veredito"] == verdict]
        if part.empty:
            continue
        fig.add_trace(
            go.Scattergeo(
                lat=part["lat"],
                lon=part["lon"],
                mode="markers",
                name=f"veredito: {verdict.lower()}",
                marker=dict(
                    size=8 + 22 * (part["Valor"] / max_amount) ** 0.5,
                    color=color,
                    opacity=0.75,
                    line=dict(width=1, color=TEXT),
                ),
                customdata=part[["#", "Cliente", "Cidade", "Valor", "Risk Score", "Rótulo real"]].values,
                hovertemplate="#%{customdata[0]} · %{customdata[1]}<br>%{customdata[2]} · R$ %{customdata[3]:,.2f}"
                "<br>Risk Score %{customdata[4]} · rótulo real: %{customdata[5]}<extra></extra>",
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
    """Risk Score de cada transação, em ordem de chegada, com as faixas 30 e 70.
    Fraudes reais em losango: um losango azul (abaixo de 30) é uma fraude que escapou.
    """
    data = frame.iloc[::-1]
    fig = go.Figure(
        go.Scatter(
            x=data["#"],
            y=data["Risk Score"],
            mode="lines+markers",
            line=dict(color=DIVIDER, width=1),
            marker=dict(
                size=[12 if r == "Fraude" else 8 for r in data["Rótulo real"]],
                symbol=["diamond" if r == "Fraude" else "circle" for r in data["Rótulo real"]],
                color=[LEVEL_COLORS[lv] for lv in data["Nível"]],
            ),
            customdata=data[["Cliente", "Cidade", "Rótulo real", "Veredito"]].values,
            hovertemplate="#%{x} · %{customdata[0]} · %{customdata[1]}<br>Risk Score %{y} → %{customdata[3]}"
            "<br>rótulo real: %{customdata[2]}<extra></extra>",
        )
    )
    fig.add_hline(y=30, line=dict(color=RISK_MEDIUM, dash="dot", width=1))
    fig.add_hline(y=70, line=dict(color=RISK_HIGH, dash="dot", width=1))
    apply_plotly_theme(fig)
    fig.update_layout(height=380, margin=dict(t=16, b=40, l=40, r=16), showlegend=False)
    fig.update_yaxes(range=[0, 102], title=None)
    fig.update_xaxes(title=dict(text="transação (ordem de chegada) · ◆ = fraude real", font=dict(color=TEXT_SECONDARY, size=12)))
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
    # Barras de espessura fixa: com 1 ou 2 categorias, a altura mínima
    # antiga virava um bloco que ocupava o painel inteiro.
    fig.update_layout(height=60 + 40 * len(labels), bargap=0.35, margin=dict(t=8, b=32, l=8, r=16), showlegend=False)
    fig.update_yaxes(automargin=True)
    # Contagens: ticks inteiros (com "d" sozinho, 0,5 virava "1" repetido).
    fig.update_xaxes(rangemode="tozero", tickformat="d", dtick=max(1, math.ceil(max(values, default=1) / 5)))
    return fig


def _quality(s: dict, events: list[LiveEvent]) -> None:
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(
            panel("Veredito × rótulo real (matriz de confusão ao vivo)", confusion_matrix_html(s["tn"], s["fp"], s["fn"], s["tp"])),
            unsafe_allow_html=True,
        )
    with right:
        with st.container(key="sx-panel-live-scenarios"):
            st.markdown('<div class="sx-panel-title">Chamadas de fraudulentas, por cenário real</div>', unsafe_allow_html=True)
            table = scenario_detection(events)
            text = [f"{r.Sinalizadas} de {r.Total} ({fmt_pct(r.Taxa, 0)})" for r in table.itertuples()]
            st.plotly_chart(_hbar(table["Cenário"], table["Sinalizadas"], ACCENT, text), key="live_scenarios", theme=None, config=CHART_CONFIG)
            st.caption('Em "Compra normal", cada uma é um alarme falso; nos demais, uma fraude capturada.')


def _reasons(events: list[LiveEvent]) -> None:
    with st.container(key="sx-panel-live-reasons"):
        st.markdown('<div class="sx-panel-title">Sinais que mais dispararam (transações fraudulentas no veredito)</div>', unsafe_allow_html=True)
        counts = reason_counts(events)
        if counts.empty:
            st.caption("Nenhuma transação chamada de fraudulenta ainda.")
        else:
            st.plotly_chart(_hbar(counts.index, counts.values, RISK_HIGH), key="live_reasons", theme=None, config=CHART_CONFIG)


def _style_feed(display: pd.DataFrame) -> pd.DataFrame:
    styles = style_levels(display)
    styles["Veredito"] = display["Veredito"].map(
        lambda v: f"color: {RISK_HIGH if v == VERDICT_FRAUD else ACCENT}; font-weight: 700"
    )
    styles["Rótulo real"] = display["Rótulo real"].map(
        lambda v: f"color: {RISK_HIGH if v == 'Fraude' else ACCENT}"
    )
    styles["Acerto"] = display["Acerto"].map(
        lambda v: f"color: {ACCENT}; font-weight: 700" if v == "✓" else f"background-color: {HIGH_TINT}; color: {RISK_HIGH}; font-weight: 700"
    )
    return styles


def _feed(frame) -> None:
    st.subheader(f"Feed de transações (últimas {min(FEED_ROWS, len(frame))})")
    table = frame.head(FEED_ROWS).drop(columns=["lat", "lon"])
    table["Ação"] = table["Ação"].map(ACTION_LABEL)
    styled = table.style.format(
        {
            "Chegada": lambda t: t.strftime("%H:%M:%S"),
            "Data/hora": lambda t: t.strftime("%d/%m/%Y %H:%M"),
            "Valor": fmt_brl,
            "Distância (km)": lambda v: fmt_int(v),
            "Decisão (ms)": _fmt_ms,
        }
    ).apply(_style_feed, axis=None)
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
    summary = summarize(events)
    _kpis(summary)
    st.subheader("Última transação")
    _latest(events[-1])

    map_col, line_col = st.columns(2, gap="medium")
    with map_col:
        with st.container(key="sx-panel-live-map"):
            st.markdown('<div class="sx-panel-title">Onde as compras aconteceram</div>', unsafe_allow_html=True)
            st.plotly_chart(_map(frame), key="live_map", theme=None, config=CHART_CONFIG)
    with line_col:
        with st.container(key="sx-panel-live-timeline"):
            st.markdown('<div class="sx-panel-title">Risk Score ao longo do fluxo</div>', unsafe_allow_html=True)
            st.plotly_chart(_timeline(frame), key="live_timeline", theme=None, config=CHART_CONFIG)

    _quality(summary, events)
    _reasons(events)
    _feed(frame)


def render():
    st.title("Monitor ao Vivo")
    st.markdown(
        "Transações chegando em tempo real. Cada uma é avaliada pelo Sentinel na hora — valor, localização, "
        "horário, dispositivo, canal (online ou presencial), forma de pagamento (aproximação ou senha) e "
        "velocidade desde a última compra — e recebe o veredito **Fraudulenta** ou **Legítima**, mostrado ao "
        "lado do **rótulo real** da transação, que o motor não vê."
    )
    simulation_notice(SIMULATION_NOTICE)

    problem = database_problem(resolve_db_path())
    if problem:
        st.error(problem, icon=":material/database_off:")
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
        if c3.button("Gerar 10 agora", key="live_burst", disabled=running, width="stretch"):
            _generate(10)
        c4.button("Limpar fluxo", key="live_clear", on_click=_clear, width="stretch")
        s1, s2 = st.columns(2)
        interval = s1.select_slider(
            "Intervalo entre transações", options=[0.5, 1.0, 2.0, 3.0, 5.0], value=2.0, key=K_INTERVAL,
            format_func=lambda s: f"{s:g} s".replace(".", ","),
        )  # fmt: skip
        s2.slider("Fraudes no fluxo (%)", 0, 100, 20, step=5, key=K_FRAUD)

    # só o painel é redesenhado a cada transação; desligado, não há ciclo
    st.fragment(_dashboard, run_every=interval if running else None)(running)
