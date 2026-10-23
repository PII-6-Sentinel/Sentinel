"""Testes da camada do Simulador (app/simulator.py), sem Streamlit.

Os pontos que importam:

- os cenários de demonstração são CALCULADOS do histórico do cliente (não
  fixos) e só preenchem o formulário — nenhum deles roda o motor;
- cada cenário realmente aciona o sinal que promete (ex.: "viagem
  impossível" produz `is_impossible_travel`), para TODOS os clientes;
- o veredito do simulador é o mesmo de `score_new_transaction`
  (src/risk/new_transaction.py) para a mesma entrada;
- o raciocínio mostrado traz os valores reais, e nada vai para o banco.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import risk_analysis as ra  # noqa: E402
import simulator as sim  # noqa: E402
from src.risk.new_transaction import score_new_transaction  # noqa: E402
from src.synthetic.generator import MAX_PLAUSIBLE_SPEED_KMH  # noqa: E402
from src.synthetic.models import Transaction  # noqa: E402


@pytest.fixture(scope="module")
def db(synthetic_db_file):
    engine = create_engine(f"sqlite:///{synthetic_db_file}")
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def risk_engine(synthetic_db_file):
    return ra.run_analysis(synthetic_db_file).engine


@pytest.fixture(scope="module")
def customer_ids(db):
    with Session(db) as s:
        return sim.list_customers(s)["customer_id"].astype(int).tolist()


@pytest.fixture(scope="module")
def sample_ids(customer_ids):
    return customer_ids[::6]  # ~22 clientes de todos os perfis, rápido o bastante


def _ctx(db, customer_id):
    with Session(db) as s:
        return sim.load_context(s, customer_id)


def _simulate(db, engine, form, progress=None):
    with Session(db) as s:
        return sim.simulate(s, engine, form, progress=progress)


# ----------------------------------------------------------------------
# clientes e contexto
# ----------------------------------------------------------------------


def test_customer_list_is_sorted_by_name_and_complete(db):
    with Session(db) as s:
        customers = sim.list_customers(s)
    assert customers.columns.tolist() == ["customer_id", "name", "home_city", "profile"]
    assert customers["name"].is_monotonic_increasing
    assert len(customers) == 130
    assert customers["home_city"].isin(sim.CITY_BY_NAME).all()  # toda cidade natal existe no seletor


def test_context_summarizes_the_real_history(db, customer_ids):
    cid = customer_ids[7]
    ctx = _ctx(db, cid)
    with Session(db) as s:
        n, last_ts = s.execute(
            select(func.count(), func.max(Transaction.timestamp)).where(Transaction.customer_id == cid)
        ).one()
    assert ctx.n_transactions == n
    assert ctx.last.timestamp == last_ts
    assert sum(ctx.city_counts.values()) == n
    assert sum(d.uses for d in ctx.devices) == n
    assert ctx.devices and len(ctx.recent) == min(5, n)
    assert ctx.recent["Data/hora"].iloc[0] == last_ts  # mais recente primeiro


# ----------------------------------------------------------------------
# cenários: calculados do histórico, só preenchem o formulário
# ----------------------------------------------------------------------


def test_scenarios_are_exactly_the_seven_requested_ones():
    assert [k for k, _ in sim.SCENARIOS] == [
        "normal", "viagem_impossivel", "valor_atipico", "horario_atipico", "dispositivo_novo", "cidade_nova",
        "combinacao_de_sinais",
    ]
    assert dict(sim.SCENARIOS)["combinacao_de_sinais"] == "Combinação de sinais"


def test_scenario_forms_carry_inputs_only_never_a_result():
    assert set(sim.FormValues.__dataclass_fields__) == {
        "customer_id",
        "city",
        "amount",
        "timestamp",
        "device_id",
        "channel",
        "payment_method",
    }


def test_every_scenario_is_valid_for_every_customer(db, customer_ids):
    for cid in customer_ids:
        ctx = _ctx(db, cid)
        scenarios = sim.build_scenarios(ctx)
        assert set(scenarios) == {k for k, _ in sim.SCENARIOS}
        for key, form in scenarios.items():
            assert form is not None, (cid, key)
            assert form.customer_id == cid
            assert form.city in sim.CITY_BY_NAME
            assert form.amount > 0
            assert sim.validate_timestamp(ctx, form.timestamp) is None, (cid, key)
            assert form.timestamp.second == 0  # resolução de minuto, como o banco


def test_scenarios_are_computed_from_the_chosen_customer_history(db, customer_ids):
    a, b = (sim.build_scenarios(_ctx(db, c)) for c in customer_ids[:2])
    for key, _ in sim.SCENARIOS:
        assert a[key] != b[key], key  # nada de valores fixos iguais para todos


def test_normal_scenario_mirrors_the_customers_habits(db, customer_ids):
    for cid in customer_ids[::10]:
        ctx = _ctx(db, cid)
        form = sim.build_scenarios(ctx)[sim.SCENARIO_NORMAL]
        assert form.city == ctx.home_city
        assert form.amount == round(ctx.avg_amount_baseline, 2)
        device = next(d for d in ctx.devices if d.id == form.device_id)
        assert device.uses > 0  # dispositivo conhecido, já usado
        assert ctx.typical_hour_start <= form.timestamp.hour <= ctx.typical_hour_end or ctx.typical_hour_start > ctx.typical_hour_end


def test_atypical_amount_is_about_ten_times_the_average(db, customer_ids):
    for cid in customer_ids[::10]:
        ctx = _ctx(db, cid)
        form = sim.build_scenarios(ctx)[sim.SCENARIO_ATYPICAL_AMOUNT]
        assert form.amount == pytest.approx(10 * ctx.avg_amount_baseline, abs=0.01)
        assert form.city == ctx.home_city


def test_new_device_scenario_uses_the_unrecognized_device_option(db, customer_ids):
    form = sim.build_scenarios(_ctx(db, customer_ids[3]))[sim.SCENARIO_NEW_DEVICE]
    assert form.device_id == sim.NEW_DEVICE


def test_combined_scenario_stacks_new_device_atypical_hour_and_amount_from_the_history(db, customer_ids):
    for cid in customer_ids:
        ctx = _ctx(db, cid)
        scenarios = sim.build_scenarios(ctx)
        combined = scenarios[sim.SCENARIO_COMBINED]
        assert combined.device_id == sim.NEW_DEVICE  # dispositivo novo (igual ao cenário isolado)
        assert combined.amount == scenarios[sim.SCENARIO_ATYPICAL_AMOUNT].amount  # ~10x a média
        assert combined.timestamp == scenarios[sim.SCENARIO_ATYPICAL_HOUR].timestamp  # horário oposto à janela
        assert combined.city == ctx.home_city  # sem outros sinais: só os três


def test_new_city_scenario_picks_a_city_the_customer_never_visited(db, customer_ids):
    for cid in customer_ids:
        ctx = _ctx(db, cid)
        form = sim.build_scenarios(ctx)[sim.SCENARIO_NEW_CITY]
        assert form.city not in ctx.city_counts and form.city != ctx.home_city


def test_impossible_travel_scenario_is_minutes_after_the_last_purchase_far_away(db, customer_ids):
    for cid in customer_ids[::10]:
        ctx = _ctx(db, cid)
        form = sim.build_scenarios(ctx)[sim.SCENARIO_IMPOSSIBLE_TRAVEL]
        gap = form.timestamp - ctx.last.timestamp
        assert timedelta(minutes=sim.IMPOSSIBLE_TRAVEL_MINUTES) <= gap < timedelta(minutes=sim.IMPOSSIBLE_TRAVEL_MINUTES + 1)
        assert form.city != ctx.last.city


def test_building_scenarios_does_not_touch_the_database(db, customer_ids):
    with Session(db) as s:
        before = s.scalar(select(func.count()).select_from(Transaction))
    for cid in customer_ids[:5]:
        sim.build_scenarios(_ctx(db, cid))
    with Session(db) as s:
        assert s.scalar(select(func.count()).select_from(Transaction)) == before


# ----------------------------------------------------------------------
# cada cenário aciona o sinal que promete (motor real, todos os clientes)
# ----------------------------------------------------------------------


def _run(db, engine, cid, key):
    ctx = _ctx(db, cid)
    return _simulate(db, engine, sim.build_scenarios(ctx)[key])


def test_impossible_travel_scenario_triggers_the_impossible_travel_signal(db, risk_engine, customer_ids):
    for cid in customer_ids:  # TODOS os clientes
        result = _run(db, risk_engine, cid, sim.SCENARIO_IMPOSSIBLE_TRAVEL)
        assert result.features["is_impossible_travel"] is True, cid
        assert result.features["implied_speed_kmh"] > MAX_PLAUSIBLE_SPEED_KMH
        assert result.assessment.level != "baixo"  # piso de viagem impossível
        assert any("Viagem impossível" in r for r in result.assessment.reasons)


def test_normal_scenario_triggers_no_signal_and_is_approved(db, risk_engine, sample_ids):
    for cid in sample_ids:
        r = _run(db, risk_engine, cid, sim.SCENARIO_NORMAL)
        f = r.features
        assert not f["is_impossible_travel"] and not f["new_device"] and not f["new_city"] and not f["device_untrusted"]
        assert f["unusual_hour"] == 0 and f["amount_deviation"] == pytest.approx(0, abs=1e-3)
        assert r.assessment.level == "baixo" and r.assessment.action == "aprovar", cid


def test_atypical_amount_scenario_triggers_the_amount_signal(db, risk_engine, sample_ids):
    for cid in sample_ids:
        r = _run(db, risk_engine, cid, sim.SCENARIO_ATYPICAL_AMOUNT)
        assert r.features["amount_deviation"] == pytest.approx(9.0, abs=0.01)
        assert any("acima da média" in reason for reason in r.assessment.reasons)


def test_atypical_hour_scenario_triggers_the_hour_signal(db, risk_engine, sample_ids):
    for cid in sample_ids:
        r = _run(db, risk_engine, cid, sim.SCENARIO_ATYPICAL_HOUR)
        assert r.features["unusual_hour"] > 0, cid
        assert any("fora da janela habitual" in reason for reason in r.assessment.reasons)


def test_new_device_scenario_triggers_the_new_and_untrusted_device_signals(db, risk_engine, sample_ids):
    for cid in sample_ids:
        r = _run(db, risk_engine, cid, sim.SCENARIO_NEW_DEVICE)
        assert r.features["new_device"] is True and r.features["device_untrusted"] is True
        reasons = " ".join(r.assessment.reasons)
        assert "Dispositivo nunca usado" in reasons and "Dispositivo não confiável" in reasons


def test_combined_scenario_triggers_all_three_signals_for_every_customer(db, risk_engine, customer_ids):
    for cid in customer_ids:  # TODOS os clientes
        r = _run(db, risk_engine, cid, sim.SCENARIO_COMBINED)
        f = r.features
        assert f["new_device"] is True and f["device_untrusted"] is True
        assert f["unusual_hour"] > 0
        assert f["amount_deviation"] == pytest.approx(9.0, abs=0.01)
        # e nenhum outro: cidade natal, sem viagem impossível
        assert not f["new_city"] and not f["is_impossible_travel"], cid
        reasons = " ".join(r.assessment.reasons)
        for expected in ("Dispositivo nunca usado", "Dispositivo não confiável", "fora da janela habitual", "acima da média"):
            assert expected in reasons, (cid, expected)
        assert r.assessment.level != "baixo"  # a combinação leva ao menos a revisão


def test_combined_scenario_scores_higher_than_each_of_its_parts(db, risk_engine, customer_ids):
    for cid in customer_ids[::10]:
        raw = {k: _run(db, risk_engine, cid, k).assessment.raw_score for k in (
            sim.SCENARIO_COMBINED, sim.SCENARIO_NEW_DEVICE, sim.SCENARIO_ATYPICAL_HOUR, sim.SCENARIO_ATYPICAL_AMOUNT)}
        assert raw[sim.SCENARIO_COMBINED] > max(v for k, v in raw.items() if k != sim.SCENARIO_COMBINED), cid


def test_new_city_scenario_triggers_the_new_city_signal(db, risk_engine, sample_ids):
    for cid in sample_ids:
        r = _run(db, risk_engine, cid, sim.SCENARIO_NEW_CITY)
        assert r.features["new_city"] is True and r.features["merchant_city"] == r.form.city
        assert any("Primeira compra do cliente nesta cidade" in reason for reason in r.assessment.reasons)


# ----------------------------------------------------------------------
# o veredito é o do motor: coincide com src/risk/new_transaction.py
# ----------------------------------------------------------------------


@pytest.mark.parametrize("key", [k for k, _ in sim.SCENARIOS])
def test_verdict_equals_score_new_transaction_for_the_same_input(db, risk_engine, customer_ids, key):
    for cid in customer_ids[::13]:
        form = sim.build_scenarios(_ctx(db, cid))[key]
        simulated = _simulate(db, risk_engine, form).assessment
        with Session(db) as s:
            reference = score_new_transaction(
                s,
                risk_engine,
                customer_id=form.customer_id,
                amount=form.amount,
                lat=sim.CITY_BY_NAME[form.city].lat,
                lon=sim.CITY_BY_NAME[form.city].lon,
                timestamp=form.timestamp,
                device_id=None if form.device_id == sim.NEW_DEVICE else form.device_id,
                city=form.city,
            )
        assert simulated == reference  # RiskResult inteiro: scores, nível, ação, motivos


def test_manual_form_input_also_matches(db, risk_engine, customer_ids):
    ctx = _ctx(db, customer_ids[4])
    form = sim.FormValues(ctx.customer_id, "Manaus", 1234.56, ctx.last.timestamp + timedelta(days=3, hours=2), ctx.devices[0].id)
    simulated = _simulate(db, risk_engine, form).assessment
    with Session(db) as s:
        reference = score_new_transaction(
            s, risk_engine, customer_id=ctx.customer_id, amount=1234.56, lat=sim.CITY_BY_NAME["Manaus"].lat,
            lon=sim.CITY_BY_NAME["Manaus"].lon, timestamp=form.timestamp, device_id=ctx.devices[0].id, city="Manaus",
        )
    assert simulated == reference


# ----------------------------------------------------------------------
# validação do horário
# ----------------------------------------------------------------------


def test_timestamp_must_be_after_the_customers_last_transaction(db, customer_ids):
    ctx = _ctx(db, customer_ids[2])
    last = ctx.last.timestamp
    for bad in (last, last - timedelta(minutes=1), last - timedelta(days=30)):
        message = sim.validate_timestamp(ctx, bad)
        assert message and "posterior à última transação" in message
        assert f"{last:%d/%m/%Y %H:%M}" in message and ctx.last.city in message
    assert sim.validate_timestamp(ctx, last + timedelta(minutes=1)) is None


def test_simulating_with_an_invalid_timestamp_is_refused(db, risk_engine, customer_ids):
    ctx = _ctx(db, customer_ids[2])
    form = sim.FormValues(ctx.customer_id, ctx.home_city, 50.0, ctx.last.timestamp, ctx.devices[0].id)
    with pytest.raises(sim.InvalidSimulation, match="posterior"):
        _simulate(db, risk_engine, form)


def test_unknown_city_and_invalid_amount_are_refused(db, risk_engine, customer_ids):
    ctx = _ctx(db, customer_ids[2])
    when = ctx.last.timestamp + timedelta(days=1)
    with pytest.raises(sim.InvalidSimulation):
        _simulate(db, risk_engine, sim.FormValues(ctx.customer_id, "Atlântida", 50.0, when, ctx.devices[0].id))
    with pytest.raises(ValueError):
        _simulate(db, risk_engine, sim.FormValues(ctx.customer_id, ctx.home_city, 0.0, when, ctx.devices[0].id))


# ----------------------------------------------------------------------
# raciocínio passo a passo: valores reais, nunca texto fixo
# ----------------------------------------------------------------------


def test_reasoning_steps_are_in_order_and_announced_to_the_callback(db, risk_engine, customer_ids):
    events = []
    form = sim.build_scenarios(_ctx(db, customer_ids[1]))[sim.SCENARIO_NORMAL]
    result = _simulate(db, risk_engine, form, progress=lambda label, detail, seconds: events.append((label, detail, seconds)))
    labels = [s.label for s in result.steps]
    assert labels == [
        "Recuperando o histórico do cliente",
        "Calculando as features da transação",
        "Distância da residência",
        "Velocidade implícita desde a última compra",
        "Horário em relação à janela habitual",
        "Valor em relação à média",
        "Dispositivo",
        "Cidade",
        "Canal e forma de pagamento",
        "Aplicando o modelo de ML",
        "Aplicando as regras",
        "Combinando no Risk Score",
        "Classificando por nível",
    ]
    timed = [s for s in result.steps if s.seconds is not None]
    assert [s.label for s in timed][:2] == labels[:2] and all(s.seconds >= 0 for s in timed)
    starts = [e for e in events if e[1] is None]
    assert [e[0] for e in starts] == [s.label for s in timed]  # cada etapa cronometrada avisa o início
    assert [e[0] for e in events if e[1] is not None] == labels  # e todas avisam o fim, na ordem


def test_reasoning_shows_the_real_values_of_the_history(db, risk_engine, customer_ids):
    ctx = _ctx(db, customer_ids[9])
    result = _simulate(db, risk_engine, sim.build_scenarios(ctx)[sim.SCENARIO_NEW_CITY])
    step = {s.label: s.detail for s in result.steps}
    history = step["Recuperando o histórico do cliente"]
    assert f"{ctx.n_transactions} transações anteriores" in history
    assert f"{ctx.last.timestamp:%d/%m/%Y %H:%M}" in history
    for city in ctx.city_counts:
        assert city in history
    km = result.features["distance_from_home_km"]
    assert f"{round(km):,}".replace(",", ".") + " km" in step["Distância da residência"]
    assert result.city.city in step["Cidade"] and "NUNCA visitada" in step["Cidade"]
    a = result.assessment
    assert f"{a.ml_score:.1f}".replace(".", ",") in step["Aplicando o modelo de ML"]
    assert f"score interno {a.raw_score}" in step["Combinando no Risk Score"]
    assert f"Risk Score exibido {a.risk_score}" in step["Combinando no Risk Score"]
    assert f"nível {a.level}" in step["Classificando por nível"] and a.action in step["Classificando por nível"]


def test_reasoning_reports_speed_and_minutes_for_the_impossible_trip(db, risk_engine, customer_ids):
    ctx = _ctx(db, customer_ids[5])
    result = _simulate(db, risk_engine, sim.build_scenarios(ctx)[sim.SCENARIO_IMPOSSIBLE_TRAVEL])
    detail = {s.label: s.detail for s in result.steps}["Velocidade implícita desde a última compra"]
    minutes = result.facts["minutes_since_last"]
    assert sim.IMPOSSIBLE_TRAVEL_MINUTES <= minutes < sim.IMPOSSIBLE_TRAVEL_MINUTES + 1
    assert f"{round(minutes)} min" in detail and "IMPOSSÍVEL" in detail
    assert f"{round(result.features['implied_speed_kmh']):,}".replace(",", ".") + " km/h" in detail
    assert ctx.last.city in detail


def test_reasoning_device_step_distinguishes_known_from_unrecognized(db, risk_engine, customer_ids):
    ctx = _ctx(db, customer_ids[5])
    scenarios = sim.build_scenarios(ctx)
    known = {s.label: s.detail for s in _simulate(db, risk_engine, scenarios[sim.SCENARIO_NORMAL]).steps}["Dispositivo"]
    new = {s.label: s.detail for s in _simulate(db, risk_engine, scenarios[sim.SCENARIO_NEW_DEVICE]).steps}["Dispositivo"]
    assert "reconhecido" in known and "compras anteriores" in known
    assert "não reconhecido" in new


def test_simulating_never_writes_to_the_database(db, risk_engine, customer_ids):
    with Session(db) as s:
        before = s.scalar(select(func.count()).select_from(Transaction))
    for key, _ in sim.SCENARIOS:
        _run(db, risk_engine, customer_ids[0], key)
    with Session(db) as s:
        assert s.scalar(select(func.count()).select_from(Transaction)) == before


def test_history_row_has_the_session_table_columns(db, risk_engine, customer_ids):
    result = _run(db, risk_engine, customer_ids[0], sim.SCENARIO_NORMAL)
    row = sim.history_row(result)
    assert {"Cliente", "Valor", "Cidade", "Risk Score", "Nível", "Ação"} <= set(row)
    assert row["Cliente"] == result.context.name and row["Risk Score"] == result.assessment.risk_score
    assert isinstance(row["Data/hora"], datetime)


def test_channel_reaches_the_features_and_the_reasoning(db, risk_engine, customer_ids):
    ctx = _ctx(db, customer_ids[1])
    normal = sim.build_scenarios(ctx)[sim.SCENARIO_NORMAL]
    online = sim.FormValues(**{**normal.__dict__, "channel": "online", "payment_method": None})
    result = _simulate(db, risk_engine, online)
    assert result.features["is_online"] is True
    step = next(s for s in result.steps if s.label == "Canal e forma de pagamento")
    assert step.detail.startswith("Online")
