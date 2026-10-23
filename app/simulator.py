"""Camada fina do Simulador ao vivo (sem Streamlit).

O simulador descreve UMA transação nova de um cliente do banco sintético e
mostra o raciocínio do Sentinel ao analisá-la. Tudo que importa vem de
`src/` sem alteração:

- `compute_transaction_features` (src/risk/new_transaction.py) calcula as
  features a partir do histórico do cliente no banco — as mesmas funções
  puras de `src/synthetic/features.py` que montam a tabela de treino;
- `RiskEngine.assess` combina ML e regras no Risk Score.

Este módulo só acrescenta o que a tela precisa e o `src/` não expõe:

- o CONTEXTO do cliente (dispositivos, cidades conhecidas, última compra);
- os CENÁRIOS de demonstração, calculados do histórico real do cliente
  escolhido (nunca valores fixos, nunca um resultado embutido);
- a VALIDAÇÃO do horário (a transação simulada precisa ser posterior à
  última do cliente — senão o "histórico" não faria sentido);
- o RACIOCÍNIO passo a passo, com os valores intermediários reais
  (distância, velocidade, horário, dispositivo, cidade...).

O veredito é exatamente `engine.assess(compute_transaction_features(...))`,
a mesma coisa que `score_new_transaction` faz (testado). Nada é gravado no
banco.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta
from typing import Callable

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.risk.behavior_score import BehaviorConfig, behavior_score
from src.risk.engine import RiskEngine, RiskResult, combine_scores
from src.risk.new_transaction import compute_transaction_features
from src.synthetic.generator import CONTACTLESS_LIMIT_BRL, MAX_PLAUSIBLE_SPEED_KMH
from src.synthetic.geo_reference import CITIES, CityRef, distance_km
from src.synthetic.models import Channel, Customer, Device, Merchant, PaymentMethod, Transaction

# Valor da opção "Novo dispositivo, não reconhecido" no seletor de dispositivo.
NEW_DEVICE = -1

# Canal e forma de pagamento (Etapa 12): (canal, forma de pagamento) -> rótulo.
CHANNEL_OPTIONS: dict[tuple[str, str | None], str] = {
    (Channel.PRESENCIAL.value, PaymentMethod.SENHA.value): "Presencial · senha",
    (Channel.PRESENCIAL.value, PaymentMethod.APROXIMACAO.value): "Presencial · aproximação",
    (Channel.ONLINE.value, None): "Online",
}

# Velocidade (km/h) usada para escolher horários plausíveis nos cenários que
# NÃO são de viagem impossível: bem abaixo dos 900 km/h do limiar.
PLAUSIBLE_TRAVEL_KMH = 500.0
# Viagem impossível: poucos minutos depois da última compra.
IMPOSSIBLE_TRAVEL_MINUTES = 20

SCENARIO_NORMAL = "normal"
SCENARIO_IMPOSSIBLE_TRAVEL = "viagem_impossivel"
SCENARIO_ATYPICAL_AMOUNT = "valor_atipico"
SCENARIO_ATYPICAL_HOUR = "horario_atipico"
SCENARIO_NEW_DEVICE = "dispositivo_novo"
SCENARIO_NEW_CITY = "cidade_nova"
SCENARIO_COMBINED = "combinacao_de_sinais"
SCENARIOS = [
    (SCENARIO_NORMAL, "Compra normal"),
    (SCENARIO_IMPOSSIBLE_TRAVEL, "Viagem impossível"),
    (SCENARIO_ATYPICAL_AMOUNT, "Valor atípico"),
    (SCENARIO_ATYPICAL_HOUR, "Horário atípico"),
    (SCENARIO_NEW_DEVICE, "Dispositivo novo"),
    (SCENARIO_NEW_CITY, "Cidade nova"),
    (SCENARIO_COMBINED, "Combinação de sinais"),
]

# O que cada cenário preenche (tooltip do botão).
SCENARIO_HELP = {
    SCENARIO_NORMAL: "Cidade natal, valor médio, horário habitual e dispositivo conhecido do cliente.",
    SCENARIO_IMPOSSIBLE_TRAVEL: "A cidade mais distante da última compra, poucos minutos depois dela.",
    SCENARIO_ATYPICAL_AMOUNT: "Cerca de 10x o valor médio do cliente, no resto igual à compra normal.",
    SCENARIO_ATYPICAL_HOUR: "Horário do lado oposto da janela habitual do cliente, no resto igual à compra normal.",
    SCENARIO_NEW_DEVICE: "Dispositivo novo, não reconhecido, numa compra online; no resto igual à compra normal.",
    SCENARIO_NEW_CITY: "Cidade que o cliente nunca visitou, a cerca de 2.000 km da residência.",
    SCENARIO_COMBINED: "Dispositivo novo (online), horário do lado oposto da janela habitual e cerca de 10x o valor médio, na cidade natal.",
}

CITY_BY_NAME: dict[str, CityRef] = {c.city: c for c in CITIES}
CITY_NAMES: list[str] = sorted(CITY_BY_NAME)


class InvalidSimulation(ValueError):
    """Entrada que não dá para simular (ex.: horário não posterior à última compra)."""


# ----------------------------------------------------------------------
# contexto do cliente
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceInfo:
    id: int
    device_type: str
    is_trusted: bool
    uses: int  # transações anteriores feitas com este dispositivo


@dataclass(frozen=True)
class LastPurchase:
    timestamp: datetime
    lat: float
    lon: float
    city: str
    amount: float


@dataclass
class CustomerContext:
    customer_id: int
    name: str
    profile: str
    home_city: str
    home_state: str | None
    home_lat: float
    home_lon: float
    avg_amount_baseline: float
    typical_hour_start: int
    typical_hour_end: int
    devices: list[DeviceInfo]
    n_transactions: int
    observed_avg_amount: float | None
    first_timestamp: datetime | None
    last: LastPurchase | None
    city_counts: dict[str, int]
    channel_counts: dict[tuple[str, str | None], int] = field(default_factory=dict)  # (canal, pagamento) -> compras
    recent: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def window(self) -> str:
        return f"{self.typical_hour_start}h–{self.typical_hour_end}h"


def list_customers(session: Session) -> pd.DataFrame:
    """Clientes para o seletor: id, nome, cidade natal e perfil (por nome)."""
    rows = session.execute(
        select(Customer.id, Customer.name, Customer.home_city, Customer.behavior_profile)
    ).all()
    frame = pd.DataFrame(
        [(r[0], r[1], r[2], getattr(r[3], "value", r[3])) for r in rows],
        columns=["customer_id", "name", "home_city", "profile"],
    )
    return frame.sort_values(["name", "customer_id"]).reset_index(drop=True)


def load_context(session: Session, customer_id: int) -> CustomerContext:
    """Histórico do cliente no banco, resumido para a tela e para os cenários."""
    customer = session.get(Customer, customer_id)
    if customer is None:
        raise InvalidSimulation(f"cliente {customer_id} não existe")

    uses = dict(
        session.execute(
            select(Transaction.device_id, func.count())
            .where(Transaction.customer_id == customer_id)
            .group_by(Transaction.device_id)
        ).all()
    )
    devices = [
        DeviceInfo(d.id, getattr(d.device_type, "value", d.device_type), bool(d.is_trusted), int(uses.get(d.id, 0)))
        for d in session.scalars(select(Device).where(Device.customer_id == customer_id).order_by(Device.id))
    ]

    rows = session.execute(
        select(Transaction, Merchant.city, Device.device_type)
        .join(Merchant, Merchant.id == Transaction.merchant_id)
        .join(Device, Device.id == Transaction.device_id)
        .where(Transaction.customer_id == customer_id)
        .order_by(Transaction.timestamp, Transaction.id)
    ).all()
    city_counts: dict[str, int] = {}
    channel_counts: dict[tuple[str, str | None], int] = {}
    for tx, city, _ in rows:
        city_counts[city] = city_counts.get(city, 0) + 1
        key = (tx.channel, tx.payment_method)
        channel_counts[key] = channel_counts.get(key, 0) + 1

    last = None
    recent = pd.DataFrame(columns=["ID", "Data/hora", "Valor", "Cidade", "Dispositivo"])
    if rows:
        tx, city, _ = rows[-1]
        last = LastPurchase(tx.timestamp, tx.lat, tx.lon, city, tx.amount)
        recent = pd.DataFrame(
            [
                (t.id, t.timestamp, t.amount, c, getattr(dt, "value", dt))
                for t, c, dt in reversed(rows[-5:])
            ],
            columns=["ID", "Data/hora", "Valor", "Cidade", "Dispositivo"],
        )

    return CustomerContext(
        customer_id=customer.id,
        name=customer.name,
        profile=getattr(customer.behavior_profile, "value", customer.behavior_profile),
        home_city=customer.home_city,
        home_state=customer.home_state,
        home_lat=customer.home_lat,
        home_lon=customer.home_lon,
        avg_amount_baseline=customer.avg_amount_baseline,
        typical_hour_start=customer.typical_hour_start,
        typical_hour_end=customer.typical_hour_end,
        devices=devices,
        n_transactions=len(rows),
        observed_avg_amount=(sum(t.amount for t, _, _ in rows) / len(rows)) if rows else None,
        first_timestamp=rows[0][0].timestamp if rows else None,
        last=last,
        city_counts=city_counts,
        channel_counts=channel_counts,
        recent=recent,
    )


# ----------------------------------------------------------------------
# formulário, validação e cenários
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FormValues:
    """O que o formulário descreve. `device_id == NEW_DEVICE` = novo dispositivo."""

    customer_id: int
    city: str
    amount: float
    timestamp: datetime
    device_id: int
    channel: str = Channel.PRESENCIAL.value
    payment_method: str | None = PaymentMethod.SENHA.value


def validate_timestamp(ctx: CustomerContext, timestamp: datetime) -> str | None:
    """Mensagem de erro se o horário não for POSTERIOR à última transação do
    cliente (None se estiver tudo certo). A transação simulada é o "próximo
    passo" do histórico — antes disso as features não fariam sentido.
    """
    if ctx.last is not None and timestamp <= ctx.last.timestamp:
        return (
            f"O horário informado ({timestamp:%d/%m/%Y %H:%M}) precisa ser posterior à última "
            f"transação do cliente ({ctx.last.timestamp:%d/%m/%Y %H:%M}, em {ctx.last.city})."
        )
    return None


def _habitual_minutes(ctx: CustomerContext) -> int:
    """Meio da janela habitual do cliente, em minutos do dia (circular)."""
    start, end = ctx.typical_hour_start, ctx.typical_hour_end
    mid = (start + end) / 2 if start <= end else ((start + end + 24) / 2) % 24
    return int(round(mid * 60)) % (24 * 60)


def _ceil_minute(moment: datetime) -> datetime:
    """Próximo minuto cheio (o formulário tem resolução de minuto; algumas
    transações do banco, como as fraudes, têm segundos).
    """
    floored = moment.replace(second=0, microsecond=0)
    return floored if floored == moment else floored + timedelta(minutes=1)


def _next_slot(ctx: CustomerContext, city: CityRef, minutes_of_day: int) -> datetime:
    """Primeiro instante com o horário do dia pedido que (1) vem depois da
    última compra e (2) deixa tempo para ir da última compra até `city` a
    `PLAUSIBLE_TRAVEL_KMH` — para o cenário não virar "viagem impossível" sem querer.
    """
    time_of_day = dtime(minutes_of_day // 60, minutes_of_day % 60)
    if ctx.last is None:
        return datetime.combine(datetime(2026, 9, 2).date(), time_of_day)
    km = distance_km(ctx.last.lat, ctx.last.lon, city.lat, city.lon)
    earliest = ctx.last.timestamp + timedelta(hours=km / PLAUSIBLE_TRAVEL_KMH)
    candidate = datetime.combine(earliest.date(), time_of_day)
    while candidate <= earliest:
        candidate += timedelta(days=1)
    return candidate


def known_device_id(ctx: CustomerContext) -> int:
    """Dispositivo "conhecido" para os cenários: o mais usado entre os
    confiáveis que já têm compras (senão, entre os usados; senão, o primeiro).
    """
    ranked = sorted(ctx.devices, key=lambda d: (not d.is_trusted, -d.uses, d.id))
    used = [d for d in ranked if d.uses > 0]
    chosen = (used or ranked)
    if not chosen:
        return NEW_DEVICE
    return chosen[0].id


def habitual_channel(ctx: CustomerContext) -> tuple[str, str | None]:
    """(canal, forma de pagamento) mais usado pelo cliente; sem histórico,
    presencial com senha.
    """
    if not ctx.channel_counts:
        return Channel.PRESENCIAL.value, PaymentMethod.SENHA.value
    return max(ctx.channel_counts, key=lambda k: (ctx.channel_counts[k], k[0], k[1] or ""))


def _payable(channel: tuple[str, str | None], amount: float) -> tuple[str, str | None]:
    """Aproximação só vale até o limite sem senha; acima dele, a compra presencial é com senha."""
    if channel == (Channel.PRESENCIAL.value, PaymentMethod.APROXIMACAO.value) and amount > CONTACTLESS_LIMIT_BRL:
        return Channel.PRESENCIAL.value, PaymentMethod.SENHA.value
    return channel


def build_scenarios(ctx: CustomerContext) -> dict[str, FormValues | None]:
    """Valores de formulário de cada cenário de demonstração, calculados do
    histórico REAL do cliente (inclusive o canal habitual). Um cenário que não dá para montar (ex.: não
    sobrou cidade nunca visitada) vem como None. Não roda análise nenhuma.
    """
    home = CITY_BY_NAME.get(ctx.home_city) or CityRef(ctx.home_city, ctx.home_state, "", ctx.home_lat, ctx.home_lon)
    usual = _habitual_minutes(ctx)
    avg = round(ctx.avg_amount_baseline, 2)
    device = known_device_id(ctx)

    # Canal: o habitual do cliente, para o cenário só mudar o que promete. Com
    # dispositivo novo, online — é uma compra por aplicativo/site, como no
    # gerador do banco (o fraudador não tem a senha do cartão).
    habitual = habitual_channel(ctx)

    def form(city: CityRef, amount: float, when: datetime, dev: int = device) -> FormValues:
        channel = (Channel.ONLINE.value, None) if dev == NEW_DEVICE else _payable(habitual, amount)
        return FormValues(ctx.customer_id, city.city, amount, when, dev, *channel)

    scenarios: dict[str, FormValues | None] = {
        SCENARIO_NORMAL: form(home, avg, _next_slot(ctx, home, usual)),
        SCENARIO_ATYPICAL_AMOUNT: form(home, round(10 * ctx.avg_amount_baseline, 2), _next_slot(ctx, home, usual)),
        SCENARIO_ATYPICAL_HOUR: form(home, avg, _next_slot(ctx, home, (usual + 12 * 60) % (24 * 60))),
        SCENARIO_NEW_DEVICE: form(home, avg, _next_slot(ctx, home, usual), NEW_DEVICE),
        # os três sinais "de conta" juntos: aparelho novo + horário oposto + ~10x a média
        SCENARIO_COMBINED: form(
            home, round(10 * ctx.avg_amount_baseline, 2), _next_slot(ctx, home, (usual + 12 * 60) % (24 * 60)), NEW_DEVICE
        ),
    }

    # viagem impossível: a cidade mais distante de ONDE o cliente comprou por
    # último, poucos minutos depois dela
    if ctx.last is None:
        scenarios[SCENARIO_IMPOSSIBLE_TRAVEL] = None
    else:
        far = max(CITIES, key=lambda c: distance_km(ctx.last.lat, ctx.last.lon, c.lat, c.lon))
        scenarios[SCENARIO_IMPOSSIBLE_TRAVEL] = form(
            far, avg, _ceil_minute(ctx.last.timestamp + timedelta(minutes=IMPOSSIBLE_TRAVEL_MINUTES))
        )

    # cidade nunca visitada: entre as que o cliente nunca usou (e que não são a
    # natal), a que fica mais perto de onde a regra de localização satura
    # (BehaviorConfig.location_full_km) — uma anomalia geográfica "típica"
    target = BehaviorConfig().location_full_km
    unvisited = [c for c in CITIES if c.city not in ctx.city_counts and c.city != ctx.home_city]
    if unvisited:
        pick = min(unvisited, key=lambda c: abs(distance_km(ctx.home_lat, ctx.home_lon, c.lat, c.lon) - target))
        scenarios[SCENARIO_NEW_CITY] = form(pick, avg, _next_slot(ctx, pick, usual))
    else:
        scenarios[SCENARIO_NEW_CITY] = None
    return scenarios


# ----------------------------------------------------------------------
# simulação com raciocínio passo a passo
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SimStep:
    label: str
    detail: str
    seconds: float | None  # None = leitura de um valor já calculado (sem tempo próprio)


@dataclass
class SimulationResult:
    form: FormValues
    context: CustomerContext
    city: CityRef
    features: dict
    assessment: RiskResult
    steps: list[SimStep]
    facts: dict

    @property
    def map_row(self) -> dict:
        """Campos que o mapa de localização (`ui.location_map`) lê."""
        return {
            "home_lat": self.context.home_lat,
            "home_lon": self.context.home_lon,
            "home_city": self.context.home_city,
            "tx_lat": self.city.lat,
            "tx_lon": self.city.lon,
            "city": self.city.city,
        }


# (rótulo, detalhe | None, segundos | None): início = (rótulo, None, None)
ProgressCallback = Callable[[str, "str | None", "float | None"], None]


def _fmt_int(value: float) -> str:
    return f"{round(value):,}".replace(",", ".")


def _fmt_dec(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def _brl(value: float) -> str:
    return "R$ " + f"{value:,.2f}".replace(",", "#").replace(".", ",").replace("#", ".")


def _cities_text(counts: dict[str, int], limit: int = 6) -> str:
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = ", ".join(f"{c} ({n})" for c, n in ranked[:limit])
    more = len(ranked) - limit
    return shown + (f" e mais {more}" if more > 0 else "")


def simulate(
    session: Session,
    engine: RiskEngine,
    form: FormValues,
    progress: ProgressCallback | None = None,
) -> SimulationResult:
    """Analisa a transação descrita em `form`, mostrando o raciocínio.

    `progress` é chamado com `(rótulo, None, None)` ao começar cada etapa
    cronometrada e com `(rótulo, detalhe, segundos)` ao terminar; as leituras
    de valores já calculados chegam com `segundos=None`.
    """
    steps: list[SimStep] = []

    def begin(label: str) -> float:
        if progress:
            progress(label, None, None)
        return time.perf_counter()

    def finish(label: str, detail: str, t0: float | None) -> None:
        seconds = None if t0 is None else time.perf_counter() - t0
        steps.append(SimStep(label, detail, seconds))
        if progress:
            progress(label, detail, seconds)

    # 1. histórico
    label = "Recuperando o histórico do cliente"
    t0 = begin(label)
    ctx = load_context(session, form.customer_id)
    error = validate_timestamp(ctx, form.timestamp)
    if error:
        raise InvalidSimulation(error)
    if form.city not in CITY_BY_NAME:
        raise InvalidSimulation(f"cidade desconhecida: {form.city!r}")
    city = CITY_BY_NAME[form.city]
    last_text = (
        f"última compra em {ctx.last.timestamp:%d/%m/%Y %H:%M} ({ctx.last.city})" if ctx.last else "sem compras anteriores"
    )
    finish(
        label,
        f"{ctx.n_transactions} transações anteriores · valor médio do cliente {_brl(ctx.avg_amount_baseline)} · "
        f"{len(ctx.city_counts)} cidades conhecidas: {_cities_text(ctx.city_counts)} · {last_text}",
        t0,
    )

    # 2. features (a função de src/, sem alteração)
    label = "Calculando as features da transação"
    t0 = begin(label)
    device_id = None if form.device_id == NEW_DEVICE else form.device_id
    features = compute_transaction_features(
        session,
        customer_id=form.customer_id,
        amount=form.amount,
        lat=city.lat,
        lon=city.lon,
        timestamp=form.timestamp,
        device_id=device_id,
        city=city.city,
        channel=form.channel,
        payment_method=form.payment_method,
    )
    finish(label, "distância, valor, horário, dispositivo, cidade, canal e velocidade implícita", t0)

    # leituras dos valores intermediários reais
    km_home = features["distance_from_home_km"]
    where = "cidade natal" if city.city == ctx.home_city else f"{ctx.home_city} → {city.city}"
    finish("Distância da residência", f"{_fmt_int(km_home)} km ({where})", None)

    facts: dict = {"km_home": km_home}
    if ctx.last is None:
        finish("Velocidade implícita desde a última compra", "sem compra anterior para comparar", None)
        facts.update(minutes_since_last=None, km_since_last=None, speed_kmh=None)
    else:
        minutes = (form.timestamp - ctx.last.timestamp).total_seconds() / 60.0
        km_last = distance_km(ctx.last.lat, ctx.last.lon, city.lat, city.lon)
        speed = features["implied_speed_kmh"]
        verdict = (
            f"IMPOSSÍVEL (acima de {_fmt_int(MAX_PLAUSIBLE_SPEED_KMH)} km/h)"
            if features["is_impossible_travel"]
            else f"plausível (limite {_fmt_int(MAX_PLAUSIBLE_SPEED_KMH)} km/h)"
        )
        finish(
            "Velocidade implícita desde a última compra",
            f"{_fmt_int(speed)} km/h — {_fmt_int(km_last)} km desde {ctx.last.city} em {_fmt_int(minutes)} min → {verdict}",
            None,
        )
        facts.update(minutes_since_last=minutes, km_since_last=km_last, speed_kmh=speed)

    hours_out = features["unusual_hour"] * 12.0
    if hours_out == 0:
        hour_text = f"{form.timestamp:%H:%M} — dentro da janela habitual ({ctx.window})"
    else:
        hour_text = f"{form.timestamp:%H:%M} — fora da janela habitual ({ctx.window}): {_fmt_dec(hours_out)} h fora"
    finish("Horário em relação à janela habitual", hour_text, None)
    facts["hours_outside"] = hours_out

    ratio = features["amount_deviation"] + 1.0
    finish(
        "Valor em relação à média",
        f"{_brl(form.amount)} = {_fmt_dec(ratio)}x a média do cliente ({_brl(ctx.avg_amount_baseline)})",
        None,
    )
    facts["amount_ratio"] = ratio

    device = next((d for d in ctx.devices if d.id == device_id), None)
    if device is None:
        device_text = "Novo dispositivo, não reconhecido — sem registro para este cliente (tratado como não confiável)"
    elif device.uses == 0:
        device_text = f"{device.device_type} #{device.id} — cadastrado, mas nunca usado em compras anteriores"
    else:
        trust = "confiável" if device.is_trusted else "NÃO confiável"
        device_text = f"{device.device_type} #{device.id} — reconhecido: {device.uses} compras anteriores, {trust}"
    finish("Dispositivo", device_text, None)

    seen = ctx.city_counts.get(city.city, 0)
    if city.city == ctx.home_city:
        city_text = f"{city.city} é a cidade natal (sempre conhecida)"
    elif seen:
        city_text = f"{city.city} já visitada ({seen} compras anteriores)"
    elif ctx.n_transactions == 0:
        city_text = f"{city.city}: cliente sem histórico para comparar"
    else:
        city_text = f"{city.city} NUNCA visitada — primeira compra do cliente nesta cidade"
    finish("Cidade", city_text, None)

    channel_label = CHANNEL_OPTIONS[(form.channel, form.payment_method if form.channel != Channel.ONLINE.value else None)]
    if ctx.n_transactions == 0:
        channel_text = f"{channel_label}: cliente sem histórico para comparar"
    else:
        share = (1.0 - features["channel_rarity"]) * 100
        channel_text = f"{channel_label} — {_fmt_int(share)}% das compras anteriores do cliente foram por este canal"
    finish("Canal e forma de pagamento", channel_text, None)
    facts["channel_rarity"] = features["channel_rarity"]

    # 3. ML e regras (cada uma isolada, para mostrar os dois scores e o tempo)
    label = "Aplicando o modelo de ML"
    t0 = begin(label)
    ml = float(engine.ml_model.score(pd.DataFrame([features]))[0])
    finish(label, f"score do ML {_fmt_dec(ml)}/100", t0)

    label = "Aplicando as regras"
    t0 = begin(label)
    behavior = behavior_score(features, engine.behavior_config)
    hits = "; ".join(f"{h.reason} (+{_fmt_int(h.points)})" for h in behavior.hits) or "nenhuma regra disparou"
    finish(label, f"score das regras {_fmt_dec(behavior.score)}/100 — {hits}", t0)

    # 4. combinação e classificação: o veredito é o do motor
    label = "Combinando no Risk Score"
    t0 = begin(label)
    assessment = engine.assess(features)
    cfg = engine.config
    pre_floor = int(combine_scores([ml], [behavior.score], [False], cfg)[0])
    floor_note = ""
    if bool(features["is_impossible_travel"]) and assessment.raw_score > pre_floor:
        floor_note = f"; piso de viagem impossível eleva de {pre_floor} para {assessment.raw_score}"
    finish(
        label,
        f"{_fmt_int(cfg.ml_weight * 100)}% × ML {_fmt_dec(assessment.ml_score)} + "
        f"{_fmt_int(cfg.behavior_weight * 100)}% × regras {_fmt_dec(assessment.behavior_score)} = "
        f"score interno {assessment.raw_score}{floor_note} → Risk Score exibido {assessment.risk_score}",
        t0,
    )

    label = "Classificando por nível"
    t0 = begin(label)
    bands = cfg.bands
    finish(
        label,
        f"score interno {assessment.raw_score} (faixas calibradas: revisar acima de {bands.review_threshold}, "
        f"bloquear acima de {bands.block_threshold}) → nível {assessment.level} → ação: {assessment.action}",
        t0,
    )

    return SimulationResult(
        form=form, context=ctx, city=city, features=features, assessment=assessment, steps=steps, facts=facts
    )


def history_row(result: SimulationResult) -> dict:
    """Uma linha da tabela do histórico da sessão."""
    a = result.assessment
    return {
        "Cliente": result.context.name,
        "Valor": result.form.amount,
        "Cidade": result.city.city,
        "Canal": CHANNEL_OPTIONS[(result.form.channel, result.form.payment_method if result.form.channel != Channel.ONLINE.value else None)],
        "Data/hora": result.form.timestamp,
        "Risk Score": a.risk_score,
        "Nível": a.level,
        "Ação": a.action,
        "ML": a.ml_score,
        "Regras": a.behavior_score,
    }
