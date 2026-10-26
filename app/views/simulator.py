"""Página 'Simulador' — descreva uma transação e veja o Sentinel analisá-la.

A página abre SEM veredito: só o formulário. O resultado aparece depois do
clique em "Analisar transação", com o raciocínio passo a passo (valores
reais, calculados na hora a partir do histórico do cliente no banco).

Os botões de cenário só PREENCHEM o formulário (com valores calculados do
histórico real do cliente escolhido); quem decide é sempre o motor, ao
clicar em Analisar.

O motor de risco é reaproveitado do `st.session_state` se a Central de
Transações já o treinou; senão, é treinado como uma das etapas.

Toda a lógica vive em `app/simulator.py` (que usa `src/`); aqui só há
orquestração e exibição (componentes visuais em `app/ui.py`). Nada é gravado
no banco: o histórico da sessão vive só em `st.session_state`.
"""

from datetime import date, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from risk_analysis import DatabaseNotFound, database_problem, resolve_db_path, run_analysis
from simulator import (
    CHANNEL_OPTIONS,
    CITY_NAMES,
    NEW_DEVICE,
    SCENARIO_HELP,
    SCENARIO_NORMAL,
    SCENARIOS,
    CustomerContext,
    FormValues,
    InvalidSimulation,
    SimulationResult,
    build_scenarios,
    history_row,
    list_customers,
    load_context,
    simulate,
    validate_timestamp,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from theme import ACCENT, LEVEL_COLORS, RISK_HIGH, RISK_MEDIUM, TEXT, TEXT_SECONDARY, apply_plotly_theme
from ui import (
    ACTION_LABEL,
    PROTOTYPE_NOTICE,
    card,
    empty_state,
    esc,
    fmt_brl,
    fmt_pct,
    fmt_seconds,
    html_table,
    kv_table,
    level_badge,
    location_map,
    md_brl,
    metrics_caveat,
    panel,
    reasons_list,
    style_levels,
)

from src.risk.engine import ACTION_DESCRIPTION

CENTRAL_KEY = "risk_analysis"  # resultado da Central de Transações (ver views/transactions.py)
ENGINE_KEY = "sim_engine"
VERDICT_KEY = "sim_verdict"
HISTORY_KEY = "sim_history"
FORM_CUSTOMER_KEY = "sim_form_customer"  # cliente para o qual o formulário foi preenchido
ACTIVE_SCENARIO_KEY = "sim_active_scenario"  # último cenário que preencheu o formulário (destaque do chip)

K_CUSTOMER, K_AMOUNT, K_CITY, K_DATE, K_TIME, K_DEVICE, K_CHANNEL = (
    "sim_customer",
    "sim_amount",
    "sim_city",
    "sim_date",
    "sim_time",
    "sim_device",
    "sim_channel",
)
RUN_LABEL = "Analisar transação"
SIMULATION_NOTICE = PROTOTYPE_NOTICE


# ----------------------------------------------------------------------
# estado do formulário
# ----------------------------------------------------------------------


def _apply_values(form: FormValues, set_customer: bool = True, scenario: str | None = None) -> None:
    """Preenche os campos do formulário. Só preenche: não analisa nada, e o
    veredito anterior (que já não descreve o formulário) some.

    `set_customer=False` quando o seletor de cliente já foi desenhado nesta
    execução (o Streamlit não deixa reatribuir a chave de um widget já criado).
    """
    if set_customer:
        st.session_state[K_CUSTOMER] = form.customer_id
    st.session_state[K_AMOUNT] = float(form.amount)
    st.session_state[K_CITY] = form.city
    st.session_state[K_DATE] = form.timestamp.date()
    st.session_state[K_TIME] = form.timestamp.time().replace(second=0, microsecond=0)
    st.session_state[K_DEVICE] = form.device_id
    st.session_state[K_CHANNEL] = (form.channel, form.payment_method)
    st.session_state[FORM_CUSTOMER_KEY] = form.customer_id
    st.session_state[ACTIVE_SCENARIO_KEY] = scenario
    st.session_state.pop(VERDICT_KEY, None)


def _device_label(ctx: CustomerContext):
    by_id = {d.id: d for d in ctx.devices}

    def label(device_id: int) -> str:
        if device_id == NEW_DEVICE:
            return "Novo dispositivo, não reconhecido"
        d = by_id[device_id]
        trust = "confiável" if d.is_trusted else "não confiável"
        return f"{d.device_type} #{d.id} — {trust}, {d.uses} compras"

    return label


# ----------------------------------------------------------------------
# resumo do cliente
# ----------------------------------------------------------------------


def _context_caption(ctx: CustomerContext) -> str:
    last = (
        f"última compra em {ctx.last.timestamp:%d/%m/%Y %H:%M} ({ctx.last.city})" if ctx.last else "sem compras anteriores"
    )
    return (
        f"Perfil **{ctx.profile}** · {ctx.n_transactions} transações · valor médio {md_brl(ctx.avg_amount_baseline)} · "
        f"janela habitual {ctx.window} · {last}"
    )


def _customer_panel(ctx: CustomerContext) -> str:
    cities = ", ".join(f"{esc(c)} ({n})" for c, n in sorted(ctx.city_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    trusted = sum(d.is_trusted for d in ctx.devices)
    observed = f" (média observada: {fmt_brl(ctx.observed_avg_amount)})" if ctx.observed_avg_amount else ""
    body = kv_table(
        [
            ("Cliente", f"{esc(ctx.name)} (#{ctx.customer_id}) — perfil {esc(ctx.profile)}"),
            ("Cidade natal", esc(ctx.home_city)),
            ("Transações", f"{ctx.n_transactions}; valor médio do cliente {fmt_brl(ctx.avg_amount_baseline)}{observed}"),
            ("Janela habitual", ctx.window),
            ("Cidades conhecidas", cities or "nenhuma"),
            ("Dispositivos", f"{len(ctx.devices)} ({trusted} confiáveis)"),
        ]
    )
    if len(ctx.recent):
        rows = [
            [
                str(r["ID"]),
                pd.Timestamp(r["Data/hora"]).strftime("%d/%m/%Y %H:%M"),
                fmt_brl(r["Valor"]),
                esc(r["Cidade"]),
                esc(r["Dispositivo"]),
            ]
            for _, r in ctx.recent.iterrows()
        ]
        body += (
            '<div style="margin-top:20px;font-size:13px;line-height:20px;color:var(--text-secondary);font-weight:500">'
            "Últimas compras</div>" + html_table(["ID", "Data/hora", "Valor", "Cidade", "Dispositivo"], rows)
        )
    return panel("Histórico do cliente", body)


# ----------------------------------------------------------------------
# motor de risco (reaproveitado ou treinado como etapa)
# ----------------------------------------------------------------------


def _engine_for_session(write):
    """(engine, nota). Reaproveita o motor da Central ou o já treinado aqui;
    se não houver nenhum, treina agora — como uma etapa visível.
    """
    central = st.session_state.get(CENTRAL_KEY)
    if central is not None:
        cfg = central.engine.config
        note = (
            f"motor reaproveitado da Central de Transações (já treinado nesta sessão): regras "
            f"{fmt_pct(cfg.behavior_weight, 0)} / ML {fmt_pct(cfg.ml_weight, 0)}, faixas internas "
            f"{cfg.bands.review_threshold}/{cfg.bands.block_threshold}"
        )
        write("Motor de risco", note, None)
        return central.engine, note

    held = st.session_state.get(ENGINE_KEY)
    if held is not None:
        write("Motor de risco", held["note"], None)
        return held["engine"], held["note"]

    label = "Treinando o motor de risco (nenhum motor treinado nesta sessão)"
    write(label, None, None)
    result = run_analysis(resolve_db_path())
    cfg = result.engine.config
    note = (
        f"ML treinado em {result.customers_by_split['treino']} clientes de treino; peso escolhido na validação: regras "
        f"{fmt_pct(cfg.behavior_weight, 0)} / ML {fmt_pct(cfg.ml_weight, 0)}; faixas internas calibradas "
        f"{cfg.bands.review_threshold}/{cfg.bands.block_threshold}"
    )
    write(label, note, result.total_seconds)
    st.session_state[ENGINE_KEY] = {"engine": result.engine, "note": note}
    return result.engine, note


# ----------------------------------------------------------------------
# execução ao vivo
# ----------------------------------------------------------------------


def _line(label: str, detail: str | None, seconds: float | None) -> str:
    icon = ":material/check_circle:" if seconds is not None else ":material/chevron_right:"
    timing = f" ({fmt_seconds(seconds)})" if seconds is not None else ""
    # "R$ x ... R$ y" na mesma linha viraria fórmula LaTeX no markdown do Streamlit
    return f"{icon} **{label}** — {detail.replace('$', chr(92) + '$')}{timing}"


def _run_simulation(session: Session, form: FormValues) -> None:
    """Analisa dentro de um `st.status`, mostrando cada etapa com valores reais."""
    lines: list[str] = []
    with st.status("Analisando a transação...", expanded=True) as status:

        def write(label: str, detail: str | None, seconds: float | None) -> None:
            if detail is None:
                status.update(label=f"{label}...")
                return
            line = _line(label, detail, seconds)
            lines.append(line)
            st.write(line)

        try:
            engine, engine_note = _engine_for_session(write)
            result = simulate(session, engine, form, progress=write)
        except DatabaseNotFound:
            status.update(label="Banco sintético não encontrado", state="error")
            return
        except (InvalidSimulation, ValueError) as exc:
            status.update(label="Não foi possível analisar", state="error")
            st.write(str(exc))
            return
        status.update(label="Raciocínio do Sentinel — análise concluída", state="complete", expanded=True)

    st.session_state[VERDICT_KEY] = {
        "result": result,
        "lines": lines,
        "engine_note": engine_note,
        "config": engine.config,
    }
    st.session_state.setdefault(HISTORY_KEY, []).append(history_row(result))


# ----------------------------------------------------------------------
# veredito
# ----------------------------------------------------------------------


def _gauge(score: int, level: str) -> go.Figure:
    """Risk Score de 0 a 100 com as faixas 30 e 70 sempre visíveis e um
    ponteiro (marcador branco) na posição do score.
    """
    color = LEVEL_COLORS[level]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(font=dict(size=64, color=color)),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickvals=[0, 30, 70, 100],
                    tickcolor=TEXT_SECONDARY,
                    tickfont=dict(color=TEXT_SECONDARY, size=13),
                ),
                bar=dict(color="rgba(0,0,0,0)", thickness=0),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 30], color=ACCENT),
                    dict(range=[30, 70], color=RISK_MEDIUM),
                    dict(range=[70, 100], color=RISK_HIGH),
                ],
                threshold=dict(line=dict(color=TEXT, width=5), thickness=0.9, value=score),
            ),
        )
    )
    apply_plotly_theme(fig)
    fig.update_layout(height=320, margin=dict(t=30, b=84, l=40, r=40))
    fig.add_annotation(
        text="Risk Score",
        x=0.5,
        y=-0.03,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color=TEXT_SECONDARY, size=14),
    )
    legend = "   ".join(
        f'<span style="color:{c}">●</span> {name}'
        for c, name in ((ACCENT, "baixo 0–30"), (RISK_MEDIUM, "médio 31–70"), (RISK_HIGH, "alto 71–100"))
    )
    fig.add_annotation(
        text=legend, x=0.5, y=-0.17, xref="paper", yref="paper", showarrow=False, font=dict(color=TEXT_SECONDARY, size=13)
    )
    return fig


def _render_log(entry: dict) -> None:
    result: SimulationResult = entry["result"]
    total = sum(s.seconds for s in result.steps if s.seconds is not None)
    with st.status(
        f"Raciocínio do Sentinel — análise concluída ({fmt_seconds(total)} nas etapas de cálculo)",
        state="complete",
        expanded=False,
    ):
        for line in entry["lines"]:
            st.write(line)


def _render_verdict(entry: dict, just_ran: bool) -> None:
    result: SimulationResult = entry["result"]
    a, form, ctx = result.assessment, result.form, result.context
    color = LEVEL_COLORS[a.level]
    cfg = entry["config"]

    st.subheader("Veredito do Sentinel")
    st.markdown(
        f"{esc(ctx.name)} · {md_brl(form.amount)} em {esc(result.city.city)} · {form.timestamp:%d/%m/%Y %H:%M} · "
        f"dispositivo: {'novo, não reconhecido' if form.device_id == NEW_DEVICE else f'#{form.device_id}'} · "
        f"{CHANNEL_OPTIONS[(form.channel, form.payment_method)].lower()}"
    )
    if not just_ran:  # logo após o clique, o status ao vivo já está na tela
        _render_log(entry)

    left, right = st.columns([1.1, 1], gap="medium")
    with left:
        with st.container(key="sx-panel-gauge"):
            st.plotly_chart(_gauge(a.risk_score, a.level), key="sim_gauge", theme=None)
    with right:
        st.markdown(
            card(
                "Ação recomendada",
                ACTION_LABEL[a.action],
                f"{level_badge(a.level)}<br>{ACTION_DESCRIPTION[a.action].capitalize()}",
                color,
                value_color=color,
                size="sm",
            ),
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2, gap="small")
        c1.markdown(
            card("Score do ML", f"{a.ml_score:.0f}", f"peso {fmt_pct(cfg.ml_weight, 0)}", size="xs", dot=False),
            unsafe_allow_html=True,
        )
        c2.markdown(
            card("Score das regras", f"{a.behavior_score:.0f}", f"peso {fmt_pct(cfg.behavior_weight, 0)}", size="xs", dot=False),
            unsafe_allow_html=True,
        )
    st.caption(
        f"Score interno {a.raw_score} → Risk Score exibido {a.risk_score}. As faixas calibradas na validação "
        f"({cfg.bands.review_threshold} e {cfg.bands.block_threshold} no score interno) aparecem sempre como 30 e 70. "
        f"Motor: {entry['engine_note']}."
    )
    metrics_caveat()

    st.markdown(panel("Motivos", reasons_list(a.reasons, color)), unsafe_allow_html=True)

    map_col, hist_col = st.columns(2, gap="medium")
    with map_col:
        with st.container(key="sx-panel-map"):
            st.markdown('<div class="sx-panel-title">Residência e local da compra</div>', unsafe_allow_html=True)
            st.plotly_chart(location_map(result.map_row), key="sim_map", theme=None)
    with hist_col:
        st.markdown(_customer_panel(ctx), unsafe_allow_html=True)


# ----------------------------------------------------------------------
# histórico da sessão
# ----------------------------------------------------------------------


def _clear_history() -> None:
    st.session_state.pop(HISTORY_KEY, None)


def _render_history() -> None:
    st.subheader("Histórico da sessão")
    history = st.session_state.get(HISTORY_KEY, [])
    if not history:
        st.caption("Nenhuma transação analisada nesta sessão ainda.")
        return
    table = pd.DataFrame(history)
    table.insert(0, "#", range(1, len(table) + 1))
    table = table.iloc[::-1].reset_index(drop=True)
    styled = table.style.format(
        {"Valor": fmt_brl, "Data/hora": lambda t: pd.Timestamp(t).strftime("%d/%m/%Y %H:%M"), "ML": "{:.0f}", "Regras": "{:.0f}"}
    ).apply(style_levels, axis=None)
    st.dataframe(styled, hide_index=True, key="sim_history_table")
    st.caption("Guardado só nesta sessão do navegador — nada é gravado no banco.")
    st.button("Limpar histórico", key="sim_clear_history", on_click=_clear_history)


# ----------------------------------------------------------------------
# página
# ----------------------------------------------------------------------


def _active_chip_style() -> None:
    """Destaca o chip do último cenário que preencheu o formulário."""
    active = st.session_state.get(ACTIVE_SCENARIO_KEY)
    if active:
        st.markdown(
            f'<style>.st-key-sim_scenario_{active} [data-testid^="stBaseButton"] '
            "{background: var(--accent-tint); color: var(--accent); border-color: var(--accent);}</style>",
            unsafe_allow_html=True,
        )


def _render(session: Session) -> None:
    customers = list_customers(session)
    ids = customers["customer_id"].astype(int).tolist()
    labels = {int(r.customer_id): f"{r.name} (#{r.customer_id}) — {r.home_city}" for r in customers.itertuples()}

    st.session_state.setdefault(K_CUSTOMER, ids[0])
    st.subheader("1. Descreva a transação")
    with st.container(key="sim-form"):
        customer_id = st.selectbox(
            "Cliente (nome — cidade natal)", ids, format_func=labels.get, key=K_CUSTOMER, help="Clientes do banco sintético."
        )
        ctx = load_context(session, int(customer_id))
        scenarios = build_scenarios(ctx)

        # primeira abertura ou troca de cliente: formulário com a compra normal dele
        if st.session_state.get(FORM_CUSTOMER_KEY) != customer_id:
            _apply_values(scenarios[SCENARIO_NORMAL], set_customer=False, scenario=SCENARIO_NORMAL)
        st.caption(_context_caption(ctx))

        st.markdown('<div class="sx-card-title">Cenários de demonstração</div>', unsafe_allow_html=True)
        st.caption(
            "Preenchem o formulário com valores calculados do histórico deste cliente. "
            "Não analisam nada: o resultado só aparece ao clicar em **Analisar transação**."
        )
        with st.container(key="sim-chips"):
            for key, label in SCENARIOS:
                values = scenarios[key]
                st.button(
                    label,
                    key=f"sim_scenario_{key}",
                    on_click=_apply_values,
                    args=(values, True, key) if values else None,
                    disabled=values is None,
                    help=SCENARIO_HELP[key] if values else "Não há como montar este cenário para este cliente.",
                )
        _active_chip_style()

        # duas linhas de três: seis campos numa linha cortavam o texto dos seletores
        f1, f2, f3 = st.columns([1, 1.2, 1])
        f4, f5, f6 = st.columns([1, 1.8, 1.4])
        amount = f1.number_input("Valor (R$)", min_value=0.01, step=1.0, format="%.2f", key=K_AMOUNT)
        city = f2.selectbox("Cidade da compra", CITY_NAMES, key=K_CITY)
        day = f3.date_input(
            "Data", format="DD/MM/YYYY", min_value=date(2020, 1, 1), max_value=date(2035, 12, 31), key=K_DATE
        )
        clock = f4.time_input("Horário", step=60, key=K_TIME)
        device_ids = [d.id for d in ctx.devices] + [NEW_DEVICE]
        device = f5.selectbox("Dispositivo", device_ids, format_func=_device_label(ctx), key=K_DEVICE)
        channel, payment = f6.selectbox(
            "Canal e pagamento", list(CHANNEL_OPTIONS), format_func=CHANNEL_OPTIONS.get, key=K_CHANNEL
        )

        form = FormValues(
            int(customer_id), city, float(amount), datetime.combine(day, clock), int(device), channel, payment
        )
        error = validate_timestamp(ctx, form.timestamp)
        if error:
            st.warning(error, icon=":material/schedule:")

        run = st.button(RUN_LABEL, type="primary", key="sim_run", disabled=error is not None)

    just_ran = False
    if run and error is None:
        _run_simulation(session, form)
        just_ran = True

    entry = st.session_state.get(VERDICT_KEY)
    if entry is None:
        with st.container(key="sx-panel-sim-empty"):
            st.markdown(
                empty_state(
                    "empty",
                    "Nenhuma transação analisada",
                    "Descreva a transação acima (ou use um cenário de demonstração) e clique em Analisar transação. "
                    "O veredito e o raciocínio do Sentinel só aparecem depois do clique.",
                ),
                unsafe_allow_html=True,
            )
    else:
        _render_verdict(entry, just_ran)

    _render_history()


def render():
    st.title("Simulador")
    st.markdown(
        "Descreva uma transação e veja o Sentinel analisá-la ao vivo, passo a passo, a partir do histórico do cliente. "
        f"Atenção: {SIMULATION_NOTICE}."
    )
    db_path = resolve_db_path()
    problem = database_problem(db_path)
    if problem:
        st.error(problem, icon=":material/database_off:")
        return

    sa_engine = create_engine(f"sqlite:///{db_path}")
    try:
        with Session(sa_engine) as session:
            _render(session)
    finally:
        sa_engine.dispose()
