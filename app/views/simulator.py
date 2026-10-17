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
orquestração e exibição. Nada é gravado no banco: o histórico da sessão
vive só em `st.session_state`.
"""

from datetime import date, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from risk_analysis import DatabaseNotFound, resolve_db_path, run_analysis
from simulator import (
    CITY_NAMES,
    NEW_DEVICE,
    SCENARIO_HELP,
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
from theme import AMBER, CORAL, ELECTRIC_BLUE, LEVEL_COLORS, MUTED, NAVY, TEXT, apply_plotly_theme
from ui import (
    ACTION_LABEL,
    LEVEL_CSS,
    card,
    fmt_brl,
    fmt_pct,
    fmt_seconds,
    location_map,
    md_brl,
    rgba,
    style_levels,
)

from src.risk.engine import ACTION_DESCRIPTION

CENTRAL_KEY = "risk_analysis"  # resultado da Central de Transações (ver views/transactions.py)
ENGINE_KEY = "sim_engine"
VERDICT_KEY = "sim_verdict"
HISTORY_KEY = "sim_history"
FORM_CUSTOMER_KEY = "sim_form_customer"  # cliente para o qual o formulário foi preenchido

K_CUSTOMER, K_AMOUNT, K_CITY, K_DATE, K_TIME, K_DEVICE = (
    "sim_customer",
    "sim_amount",
    "sim_city",
    "sim_date",
    "sim_time",
    "sim_device",
)
RUN_LABEL = "Analisar transação"
SIMULATION_NOTICE = "dados simulados; valida a lógica do sistema, não o desempenho em fraude real"


# ----------------------------------------------------------------------
# estado do formulário
# ----------------------------------------------------------------------


def _apply_values(form: FormValues, set_customer: bool = True) -> None:
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
    st.session_state[FORM_CUSTOMER_KEY] = form.customer_id
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


def _render_customer_summary(ctx: CustomerContext) -> None:
    st.markdown("**Histórico do cliente**")
    cities = ", ".join(f"{c} ({n})" for c, n in sorted(ctx.city_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    trusted = sum(d.is_trusted for d in ctx.devices)
    observed = f" (média observada: {md_brl(ctx.observed_avg_amount)})" if ctx.observed_avg_amount else ""
    st.markdown(
        "\n".join(
            [
                f"- **Cliente:** {ctx.name} (#{ctx.customer_id}) — perfil {ctx.profile}",
                f"- **Cidade natal:** {ctx.home_city}",
                f"- **Transações:** {ctx.n_transactions}; valor médio do cliente {md_brl(ctx.avg_amount_baseline)}{observed}",
                f"- **Janela habitual:** {ctx.window}",
                f"- **Cidades conhecidas:** {cities or 'nenhuma'}",
                f"- **Dispositivos:** {len(ctx.devices)} ({trusted} confiáveis)",
            ]
        )
    )
    if len(ctx.recent):
        recent = ctx.recent.style.format(
            {"Valor": fmt_brl, "Data/hora": lambda t: pd.Timestamp(t).strftime("%d/%m/%Y %H:%M")}
        )
        st.caption("Últimas compras")
        st.dataframe(recent, hide_index=True)


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
    icon = "✅" if seconds is not None else "▫️"
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
        status.update(label="Análise concluída", state="complete", expanded=True)

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
    """Risk Score de 0 a 100 com as faixas 30 e 70 sempre visíveis."""
    color = LEVEL_COLORS[level]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(font=dict(size=56, color=color)),
            title=dict(text="Risk Score", font=dict(size=16, color=MUTED)),
            gauge=dict(
                axis=dict(range=[0, 100], tickvals=[0, 30, 70, 100], tickcolor=MUTED, tickfont=dict(color=TEXT)),
                bar=dict(color=color, thickness=0.3),
                bgcolor=NAVY,
                borderwidth=0,
                steps=[
                    dict(range=[0, 30], color=rgba(ELECTRIC_BLUE, 0.35)),
                    dict(range=[30, 70], color=rgba(AMBER, 0.35)),
                    dict(range=[70, 100], color=rgba(CORAL, 0.35)),
                ],
                threshold=dict(line=dict(color=TEXT, width=3), thickness=0.9, value=score),
            ),
        )
    )
    apply_plotly_theme(fig)
    fig.update_layout(height=320, margin=dict(t=60, b=70, l=30, r=30))
    fig.add_annotation(
        text="baixo 0–30 · médio 31–70 · alto 71–100",
        x=0.5,
        y=-0.16,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color=MUTED, size=13),
    )
    return fig


def _render_log(entry: dict, expanded: bool) -> None:
    result: SimulationResult = entry["result"]
    total = sum(s.seconds for s in result.steps if s.seconds is not None)
    with st.status(
        f"Raciocínio do Sentinel — análise concluída ({fmt_seconds(total)} nas etapas de cálculo)",
        state="complete",
        expanded=expanded,
    ):
        for line in entry["lines"]:
            st.write(line)


def _render_verdict(entry: dict, just_ran: bool) -> None:
    result: SimulationResult = entry["result"]
    a, form, ctx = result.assessment, result.form, result.context
    color = LEVEL_COLORS[a.level]

    st.subheader("Veredito do Sentinel")
    st.caption(
        f"{ctx.name} · {md_brl(form.amount)} em {result.city.city} · {form.timestamp:%d/%m/%Y %H:%M} · "
        f"dispositivo: {'novo, não reconhecido' if form.device_id == NEW_DEVICE else f'#{form.device_id}'}"
    )
    if not just_ran:  # logo após o clique, o status ao vivo já está na tela
        _render_log(entry, expanded=False)

    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(_gauge(a.risk_score, a.level), key="sim_gauge")
    with right:
        badge = f'<span class="level-badge {LEVEL_CSS[a.level]}">nível {a.level}</span>'
        st.markdown(
            card("Ação recomendada", ACTION_LABEL[a.action], f"{badge}<br>{ACTION_DESCRIPTION[a.action].capitalize()}", color),
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        cfg = entry["config"]
        c1.markdown(card("Score do ML", f"{a.ml_score:.0f}", f"peso {fmt_pct(cfg.ml_weight, 0)}"), unsafe_allow_html=True)
        c2.markdown(
            card("Score das regras", f"{a.behavior_score:.0f}", f"peso {fmt_pct(cfg.behavior_weight, 0)}"),
            unsafe_allow_html=True,
        )
    st.caption(
        f"Score interno {a.raw_score} → Risk Score exibido {a.risk_score}. As faixas calibradas na validação "
        f"({cfg.bands.review_threshold} e {cfg.bands.block_threshold} no score interno) aparecem sempre como 30 e 70. "
        f"Motor: {entry['engine_note']}."
    )

    st.markdown("**Motivos**")
    if a.reasons:
        st.markdown("\n".join(f"- {reason}" for reason in a.reasons))
    else:
        st.markdown("- Nenhum sinal relevante: a transação está dentro do padrão do cliente.")

    map_col, hist_col = st.columns([1, 1])
    with map_col:
        st.markdown("**Residência e local da compra**")
        st.plotly_chart(location_map(result.map_row), key="sim_map")
    with hist_col:
        _render_customer_summary(ctx)


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


def _render(session: Session) -> None:
    customers = list_customers(session)
    ids = customers["customer_id"].astype(int).tolist()
    labels = {
        int(r.customer_id): f"{r.name} (#{r.customer_id}) — {r.home_city}" for r in customers.itertuples()
    }

    st.session_state.setdefault(K_CUSTOMER, ids[0])
    st.markdown("**1. Descreva a transação**")
    customer_id = st.selectbox(
        "Cliente (nome — cidade natal)", ids, format_func=labels.get, key=K_CUSTOMER, help="Clientes do banco sintético."
    )
    ctx = load_context(session, int(customer_id))
    scenarios = build_scenarios(ctx)

    # primeira abertura ou troca de cliente: formulário com a compra normal dele
    if st.session_state.get(FORM_CUSTOMER_KEY) != customer_id:
        _apply_values(scenarios["normal"], set_customer=False)
    st.caption(_context_caption(ctx))

    st.markdown("**Cenários de demonstração**")
    st.caption(
        "Preenchem o formulário com valores calculados do histórico deste cliente. "
        "Não analisam nada: o resultado só aparece ao clicar em **Analisar transação**."
    )
    per_row = 4  # com 7 cenários numa linha só, os rótulos ficam truncados
    for start in range(0, len(SCENARIOS), per_row):
        cols = st.columns(per_row)
        for col, (key, label) in zip(cols, SCENARIOS[start : start + per_row]):
            values = scenarios[key]
            col.button(
                label,
                key=f"sim_scenario_{key}",
                on_click=_apply_values,
                args=(values,) if values else None,
                disabled=values is None,
                help=SCENARIO_HELP[key] if values else "Não há como montar este cenário para este cliente.",
            )

    f1, f2, f3, f4, f5 = st.columns([1, 1.2, 1, 1, 1.8])
    amount = f1.number_input("Valor (R$)", min_value=0.01, step=1.0, format="%.2f", key=K_AMOUNT)
    city = f2.selectbox("Cidade da compra", CITY_NAMES, key=K_CITY)
    day = f3.date_input("Data", format="DD/MM/YYYY", min_value=date(2020, 1, 1), max_value=date(2035, 12, 31), key=K_DATE)
    clock = f4.time_input("Horário", step=60, key=K_TIME)
    device_ids = [d.id for d in ctx.devices] + [NEW_DEVICE]
    device = f5.selectbox("Dispositivo", device_ids, format_func=_device_label(ctx), key=K_DEVICE)

    form = FormValues(int(customer_id), city, float(amount), datetime.combine(day, clock), int(device))
    error = validate_timestamp(ctx, form.timestamp)
    if error:
        st.warning(error, icon="🕒")

    run = st.button(RUN_LABEL, type="primary", key="sim_run", disabled=error is not None)
    just_ran = False
    if run and error is None:
        _run_simulation(session, form)
        just_ran = True

    entry = st.session_state.get(VERDICT_KEY)
    if entry is None:
        st.info(
            "**Nenhuma transação analisada.** Descreva a transação acima (ou use um cenário de demonstração) "
            f"e clique em **{RUN_LABEL}**. O veredito e o raciocínio do Sentinel só aparecem depois do clique.",
            icon="▶️",
        )
    else:
        _render_verdict(entry, just_ran)

    _render_history()


def render():
    st.title("Simulador")
    st.caption(
        "Descreva uma transação e veja o Sentinel analisá-la ao vivo, passo a passo, a partir do histórico do cliente. "
        f"Atenção: {SIMULATION_NOTICE}."
    )
    db_path = resolve_db_path()
    if not db_path.exists():
        st.error(
            "**Banco sintético não encontrado.** Gere-o com `python scripts/generate_mock_data.py` "
            "(é determinístico: seed fixa) e recarregue a página.",
            icon="🗄️",
        )
        return

    sa_engine = create_engine(f"sqlite:///{db_path}")
    try:
        with Session(sa_engine) as session:
            _render(session)
    finally:
        sa_engine.dispose()
