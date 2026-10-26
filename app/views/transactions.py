"""Página 'Central de Transações' — a tela principal da camada operacional.

A página abre VAZIA. Só depois do clique em "Executar análise de risco" o
sistema lê o banco sintético, calcula as features, aplica ML e regras,
combina no Risk Score e classifica — com cada etapa e seu tempo visíveis.
Nada é pré-calculado nem cacheado ao abrir a página (de propósito: é uma
decisão da equipe que o resultado seja produzido na hora, na frente de
quem avalia).

O resultado fica em `st.session_state`, então filtros e seleção de linha
reexecutam apenas o script (barato), nunca a análise.

Toda a lógica vive em `app/risk_analysis.py` (que usa `src/risk/`); aqui
só há orquestração e exibição (componentes visuais em `app/ui.py`).
"""

import pandas as pd
import streamlit as st
from risk_analysis import (
    ACTIONS,
    LEVELS,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    STEP_LABELS,
    AnalysisResult,
    DatabaseNotFound,
    apply_filters,
    database_problem,
    efficiency,
    investigate,
    resolve_db_path,
    run_analysis,
    summarize,
)
from simulator import CHANNEL_OPTIONS
from theme import LEVEL_COLORS
from ui import (
    ACTION_LABEL,
    PROTOTYPE_NOTICE,
    bar_rows,
    card,
    chips_html,
    confusion_matrix_html,
    empty_state,
    esc,
    facts_grid,
    fmt_brl,
    fmt_int,
    fmt_km,
    fmt_pct,
    fmt_seconds,
    html_table,
    level_badge,
    location_map,
    metrics_caveat,
    panel,
    reasons_list,
    simulation_notice,
    style_levels,
)

from src.risk.engine import ACTION_BY_LEVEL, ACTION_DESCRIPTION

RESULT_KEY = "risk_analysis"
RUN_FLAG = "tx_run_requested"
SIMULATION_NOTICE = PROTOTYPE_NOTICE

SCENARIO_LABEL = {
    "anomalia_geografica": "Anomalia geográfica",
    "combinacao_de_sinais": "Combinação de sinais",
    "dispositivo_novo": "Dispositivo novo",
    "horario_atipico": "Horário atípico",
    "valor_atipico": "Valor atípico",
    "viagem_impossivel": "Viagem impossível",
}

# Chips do estado vazio: rótulo curto; a dica (title) é o nome completo da etapa.
PIPELINE_CHIPS = list(
    zip(
        ["Carregar transações", "Features comportamentais", "Modelo de ML", "Regras", "Risk Score", "Níveis"],
        STEP_LABELS,
    )
)


# ----------------------------------------------------------------------
# execução da análise (ao vivo, com etapas e tempos)
# ----------------------------------------------------------------------


def _step_line(label: str, seconds: float) -> str:
    return f":material/check_circle: {label} — {fmt_seconds(seconds)}"


def _run_with_status() -> AnalysisResult | None:
    """Roda a análise dentro de um `st.status`, mostrando cada etapa e o
    tempo que ela levou. Guarda o resultado em `st.session_state`.
    """
    with st.status("Executando análise de risco...", expanded=True) as status:

        def on_progress(label: str, seconds: float | None) -> None:
            if seconds is None:
                status.update(label=f"{label}...")
            else:
                st.write(_step_line(label, seconds))

        try:
            result = run_analysis(resolve_db_path(), progress=on_progress)
        except DatabaseNotFound:
            status.update(label="Banco sintético não encontrado", state="error")
            st.write("Gere o banco com `python scripts/generate_mock_data.py` e tente novamente.")
            return None
        except Exception as exc:  # noqa: BLE001 — a página mostra o erro em vez de quebrar
            status.update(label="A análise falhou", state="error")
            st.write(f"{type(exc).__name__}: {exc}")
            return None
        # fica ABERTO logo após rodar, para as etapas e tempos ficarem à vista;
        # nas reexecuções seguintes (filtros, seleção) vira um registro recolhido
        status.update(label=f"Análise concluída em {fmt_seconds(result.total_seconds)}", state="complete", expanded=True)

    st.session_state[RESULT_KEY] = result
    return result


def _render_step_log(result: AnalysisResult) -> None:
    """Registro (recolhido) da última execução, nas reexecuções do script."""
    with st.status(f"Análise concluída em {fmt_seconds(result.total_seconds)}", state="complete", expanded=False):
        for step in result.steps:
            st.write(_step_line(step.label, step.seconds))


def _request_run() -> None:
    """Callback do botão: a análise só roda no clique (o flag vale uma execução)."""
    st.session_state[RUN_FLAG] = True


def _clear_result() -> None:
    st.session_state.pop(RESULT_KEY, None)


# ----------------------------------------------------------------------
# estado vazio
# ----------------------------------------------------------------------


def _render_empty_state() -> None:
    with st.container(key="tx-empty"):
        st.markdown(
            empty_state(
                "empty",
                "Nenhuma análise executada",
                "Ao clicar, o Sentinel carrega as transações do banco, calcula as features de cada cliente, aplica "
                "o modelo de ML e as regras e classifica cada transação por nível de risco. Os resultados só "
                "aparecem depois da execução.",
            ),
            unsafe_allow_html=True,
        )
        st.button("Executar análise de risco", type="primary", key="tx_run", on_click=_request_run)
        st.markdown(chips_html(PIPELINE_CHIPS), unsafe_allow_html=True)


# ----------------------------------------------------------------------
# resumo
# ----------------------------------------------------------------------


def _render_summary(result: AnalysisResult) -> None:
    summary = summarize(result.frame)
    by_level, by_action = summary["por_nivel"], summary["por_acao"]

    st.subheader("Resumo da análise")
    cols = st.columns(4)
    cols[0].markdown(card("Transações analisadas", fmt_int(summary["total"]), "todos os clientes"), unsafe_allow_html=True)
    for col, level in zip(cols[1:], LEVELS):
        row = by_level.loc[level]
        col.markdown(
            card(f"Nível {level}", fmt_int(row["quantidade"]), f"{fmt_pct(row['percentual'])} das transações", LEVEL_COLORS[level]),
            unsafe_allow_html=True,
        )

    cols = st.columns(4)
    flagged = by_action.loc[["revisar", "bloquear"]]
    cols[0].markdown(
        card("Sinalizadas (revisar + bloquear)", fmt_int(flagged["quantidade"].sum()), f"{fmt_pct(flagged['percentual'].sum())} das transações"),
        unsafe_allow_html=True,
    )
    for col, action in zip(cols[1:], ACTIONS):
        row = by_action.loc[action]
        level = next(lv for lv in LEVELS if ACTION_BY_LEVEL[lv] == action)
        col.markdown(
            card(f"Ação: {ACTION_LABEL[action]}", fmt_int(row["quantidade"]), f"{fmt_pct(row['percentual'])} das transações", LEVEL_COLORS[level]),
            unsafe_allow_html=True,
        )

    cfg = result.engine.config
    st.caption(
        f"Configuração desta execução: regras {fmt_pct(result.rules_weight, 0)} / ML {fmt_pct(result.ml_weight, 0)} "
        f"(peso escolhido na validação). Faixas calibradas na validação, no score interno: revisar acima de "
        f"{cfg.bands.review_threshold}, bloquear acima de {cfg.bands.block_threshold} — exibidas como 30 e 70. "
        f"Clientes: {result.customers_by_split[SPLIT_TRAIN]} de treino, "
        f"{result.customers_by_split[SPLIT_VALIDATION]} de validação, {result.customers_by_split[SPLIT_TEST]} de teste."
    )
    metrics_caveat()


# ----------------------------------------------------------------------
# tabela investigável
# ----------------------------------------------------------------------


def _render_filters(result: AnalysisResult):
    frame = result.frame
    customers = frame[["customer_id", "customer"]].drop_duplicates().sort_values("customer")
    labels = {int(r.customer_id): f"{r.customer} (#{r.customer_id})" for r in customers.itertuples()}
    lo, hi = float(frame["amount"].min()), float(frame["amount"].max())
    lo, hi = float(int(lo)), float(int(hi) + 1)

    with st.expander("Filtros"):
        c1, c2, c3 = st.columns(3)
        levels = c1.multiselect("Nível", LEVELS, key="tx_f_level", placeholder="Todos")
        actions = c2.multiselect("Ação", ACTIONS, key="tx_f_action", placeholder="Todas")
        profiles = c3.multiselect("Perfil do cliente", sorted(frame["profile"].unique()), key="tx_f_profile", placeholder="Todos")
        c4, c5, c6 = st.columns(3)
        picked = c4.multiselect("Cliente", list(labels), format_func=labels.get, key="tx_f_customer", placeholder="Todos")
        # padrão = teste: os scores dos clientes de treino são "dentro da
        # amostra" (o modelo aprendeu com eles), então abrir a tabela neles
        # mostraria um desempenho melhor que o real. Quem quiser, inclui os outros.
        st.session_state.setdefault("tx_f_split", [SPLIT_TEST])
        splits = c5.multiselect(
            "Conjunto",
            [SPLIT_TEST, SPLIT_VALIDATION, SPLIT_TRAIN],
            key="tx_f_split",
            placeholder="Todos",
            help="Abre em \"teste\": clientes que o modelo nunca viu, os únicos que entram na medição de eficiência. "
            "Limpe o filtro para ver todos.",
        )
        amount = c6.slider("Faixa de valor (R$)", lo, hi, (lo, hi), step=1.0, key="tx_f_amount")

    return apply_filters(
        frame,
        levels=levels,
        actions=actions,
        customers=picked,
        profiles=profiles,
        amount_range=amount,
        splits=splits,
    )


def _render_table(view):
    """Tabela ordenada por Risk Score decrescente; devolve a linha selecionada
    (posição em `view`) ou None.
    """
    display = pd.DataFrame(
        {
            "ID": view["transaction_id"].to_numpy(),
            "Cliente": view["customer"].to_numpy(),
            "Valor": view["amount"].to_numpy(),
            "Cidade": view["city"].to_numpy(),
            "Canal": [_channel_text(c, p) for c, p in zip(view["channel"], view["payment_method"])],
            "Data/hora": view["timestamp"].to_numpy(),
            "Risk Score": view["risk_score"].to_numpy(),
            "Nível": view["level"].to_numpy(),
            "Ação": view["action"].to_numpy(),
            "Conjunto": view["conjunto"].to_numpy(),
        }
    )
    styled = display.style.format(
        {"Valor": fmt_brl, "Data/hora": lambda t: pd.Timestamp(t).strftime("%d/%m/%Y %H:%M")}
    ).apply(style_levels, axis=None)

    event = st.dataframe(
        styled,
        hide_index=True,
        height=420,
        on_select="rerun",
        selection_mode="single-row",
        key="tx_table",
    )
    rows = event.selection.rows
    return rows[0] if rows and rows[0] < len(view) else None


# ----------------------------------------------------------------------
# painel de investigação
# ----------------------------------------------------------------------


def _channel_text(channel: str, payment_method) -> str:
    """'Online', 'Presencial · senha' ou 'Presencial · aproximação' (online não tem forma de pagamento)."""
    payment = None if channel == "online" or pd.isna(payment_method) else payment_method
    return CHANNEL_OPTIONS.get((channel, payment), channel)


def _device_text(row) -> str:
    novelty = "novo para este cliente (primeira vez visto)" if row["new_device"] else "já usado antes por este cliente"
    trust = "não confiável" if row["device_untrusted"] else "confiável"
    return f"{row['device_type']} — {novelty}; {trust}"


def _score_card(a, cfg) -> str:
    """Risk Score grande na cor do nível, nível, ação e os dois scores com seus pesos."""
    color = LEVEL_COLORS[a.level]
    return (
        '<div class="metric-card sx-inv">'
        '<div class="sx-inv-top"><div><div class="label">Risk Score final</div>'
        f'<div class="value sx-big" style="color: {color};">{a.risk_score}</div></div>'
        f'<div class="sx-inv-action">{level_badge(a.level)}'
        f'<div class="a">{ACTION_LABEL[a.action]}</div>'
        f'<div class="d">{esc(ACTION_DESCRIPTION[a.action])}</div></div></div>'
        '<div class="sx-inv-split">'
        f'<div><div class="label">Score do ML</div><div class="v">{a.ml_score:.0f}</div><div class="sub">peso {fmt_pct(cfg.ml_weight, 0)}</div></div>'
        f'<div><div class="label">Score das regras</div><div class="v">{a.behavior_score:.0f}</div><div class="sub">peso {fmt_pct(cfg.behavior_weight, 0)}</div></div>'
        "</div></div>"
    )


def _render_investigation(result: AnalysisResult, transaction_id: int) -> None:
    inv = investigate(result, transaction_id)
    row, a = inv.row, inv.assessment
    color = LEVEL_COLORS[a.level]
    cfg = result.engine.config

    st.subheader(f"Investigação da transação #{transaction_id}")
    st.markdown(f"Transação #{transaction_id} · {esc(row['customer'])}")
    st.caption(
        f"Score interno {a.raw_score} = {fmt_pct(cfg.ml_weight, 0)} × ML ({a.ml_score:.0f}) + "
        f"{fmt_pct(cfg.behavior_weight, 0)} × regras ({a.behavior_score:.0f}), com o piso de viagem impossível "
        f"quando aplicável. O Risk Score exibido é o score interno reescalado para que as faixas calibradas "
        f"({cfg.bands.review_threshold} e {cfg.bands.block_threshold}) apareçam como 30 e 70."
    )

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(_score_card(a, cfg), unsafe_allow_html=True)
        st.markdown(panel("Motivos do score", reasons_list(a.reasons, color)), unsafe_allow_html=True)
        truth = "fraude" if row["is_fraud"] else "legítima"
        scenario = f" — {SCENARIO_LABEL.get(row['fraud_scenario'], row['fraud_scenario'])}" if row["is_fraud"] else ""
        st.caption(f"Rótulo da simulação (não existe em produção real): {truth}{scenario}.")
    with right:
        with st.container(key="sx-panel-map"):
            st.markdown('<div class="sx-panel-title">Residência e local da compra</div>', unsafe_allow_html=True)
            st.plotly_chart(location_map(row), key="tx_map", theme=None)
        window = f"{int(row['typical_hour_start'])}h–{int(row['typical_hour_end'])}h"
        st.markdown(
            panel(
                "Ficha da transação",
                facts_grid(
                    [
                        ("Cliente", f"{esc(row['customer'])} (#{row['customer_id']}) — perfil {esc(row['profile'])}"),
                        (
                            "Valor",
                            f"{fmt_brl(row['amount'])}"
                            f'<div class="k">média habitual do cliente: {fmt_brl(row["avg_amount_baseline"])}</div>',
                        ),
                        ("Cidade da compra", f"{esc(row['city'])} — {esc(row['merchant'])}"),
                        (
                            "Data e horário",
                            f"{row['timestamp']:%d/%m/%Y} às {row['timestamp']:%H:%M}"
                            f'<div class="k">janela habitual: {window}</div>',
                        ),
                        ("Local habitual", f"{esc(row['home_city'])} (residência)"),
                        ("Distância da residência", fmt_km(row["distance_from_home_km"])),
                        ("Dispositivo", esc(_device_text(row))),
                        ("Canal e pagamento", esc(_channel_text(row["channel"], row["payment_method"]))),
                        ("Conjunto", esc(row["conjunto"])),
                    ]
                ),
            ),
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------
# eficiência medida ao vivo
# ----------------------------------------------------------------------


def _render_efficiency(result: AnalysisResult) -> None:
    eff = efficiency(result)
    m = eff["metrics"]
    frauds, legit = eff["frauds"], eff["legit"]

    st.subheader("Eficiência medida nesta análise")
    simulation_notice(SIMULATION_NOTICE)
    st.caption(
        f"Calculada agora, só sobre os {eff['customers']} clientes de TESTE "
        f"({fmt_int(eff['transactions'])} transações, {fmt_int(frauds)} fraudes): clientes que o modelo nunca viu no "
        f"treino e cujos dados não serviram para escolher peso nem faixas. Treino e validação ficam de fora — "
        f"métricas ali não seriam eficiência. \"Sinalizada\" = decisão revisar ou bloquear."
    )

    cols = st.columns(4)
    cols[0].markdown(
        card("Fraudes capturadas", f"{m['tp']} de {frauds}", f"{fmt_pct(m['recall'])} das fraudes sinalizadas", LEVEL_COLORS["baixo"], size="sm"),
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        card("Falsos positivos", fmt_int(m["fp"]), f"{fmt_pct(m['fp'] / legit if legit else 0)} das {fmt_int(legit)} legítimas", LEVEL_COLORS["alto"], size="sm"),
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        card("Fraudes perdidas", fmt_int(m["fn"]), "aprovadas automaticamente", LEVEL_COLORS["alto"], size="sm"),
        unsafe_allow_html=True,
    )
    cols[3].markdown(
        card("Precisão da sinalização", fmt_pct(m["precision"]), "das sinalizadas, eram fraude", size="sm"),
        unsafe_allow_html=True,
    )
    metrics_caveat()

    left, right = st.columns([1.1, 1], gap="medium")
    with left:
        st.markdown(
            panel("Matriz de confusão", confusion_matrix_html(m["tn"], m["fp"], m["fn"], m["tp"])),
            unsafe_allow_html=True,
        )
    with right:
        by_action = eff["by_action"]
        st.markdown(
            panel(
                "Decisão do sistema × rótulo da simulação",
                html_table(
                    ["Ação", "Legítimas", "Fraudes"],
                    [[ACTION_LABEL[a], fmt_int(r["legítima"]), fmt_int(r["fraude"])] for a, r in by_action.iterrows()],
                ),
            ),
            unsafe_allow_html=True,
        )
        scenario = eff["by_scenario"]
        st.markdown(
            panel(
                "Fraudes capturadas por cenário",
                bar_rows(
                    [
                        (
                            SCENARIO_LABEL.get(name, name.replace("_", " ")),
                            f"{int(r['detected'])} de {int(r['total'])}",
                            float(r["recall"]),
                            fmt_pct(r["recall"]),
                        )
                        for name, r in scenario.iterrows()
                    ]
                ),
            ),
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------
# página
# ----------------------------------------------------------------------


def _parse_lookup(text: str) -> int | None:
    text = (text or "").strip().lstrip("#")
    return int(text) if text.isdigit() else None


def render():
    st.title("Central de Transações")
    st.markdown(
        "Análise de risco das transações do banco sintético — executada ao vivo, sob demanda. "
        f"Atenção: {SIMULATION_NOTICE}."
    )

    problem = database_problem(resolve_db_path())
    if problem:
        st.error(problem, icon=":material/database_off:")
        return

    run = st.session_state.pop(RUN_FLAG, False)
    result = st.session_state.get(RESULT_KEY)
    if result is None and not run:
        _render_empty_state()
        return

    action_col, clear_col, _ = st.columns([0.3, 0.2, 0.5])
    action_col.button("Executar análise de risco", type="primary", key="tx_run", on_click=_request_run)
    clear_slot = clear_col.empty()  # preenchido depois: o botão só existe com resultado

    just_ran = False
    if run:
        result = _run_with_status()
        just_ran = result is not None
    if result is None:
        return  # a análise falhou: a explicação já está no status e o botão acima permite tentar de novo
    clear_slot.button("Limpar resultado", key="tx_clear", on_click=_clear_result)
    if not just_ran:
        _render_step_log(result)

    _render_summary(result)

    st.subheader("Transações")
    view = _render_filters(result).reset_index(drop=True)
    st.caption(
        f"Mostrando {fmt_int(len(view))} de {fmt_int(len(result.frame))} transações, ordenadas por Risk Score "
        "decrescente. **Selecione uma linha** para investigar."
    )
    selected_pos = _render_table(view)

    lookup = st.text_input("Ou investigue por ID da transação", key="tx_lookup", placeholder="ex.: 1976")
    transaction_id = None
    if lookup.strip():
        transaction_id = _parse_lookup(lookup)
        if transaction_id is None or transaction_id not in set(result.frame["transaction_id"]):
            st.warning(f"Não encontrei a transação \"{lookup.strip()}\" nesta análise.")
            transaction_id = None
        else:
            st.caption("Busca por ID ativa — limpe o campo para voltar a usar a linha selecionada na tabela.")
    elif selected_pos is not None:
        transaction_id = int(view.iloc[selected_pos]["transaction_id"])

    if transaction_id is not None:
        _render_investigation(result, transaction_id)
    else:
        st.info("Selecione uma transação na tabela para abrir o painel de investigação.", icon=":material/search:")

    _render_efficiency(result)

