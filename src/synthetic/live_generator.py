"""Gerador de transações AO VIVO — com o rótulo real de cada uma.

Gera uma transação nova por vez, para um cliente do banco sintético,
seguindo as MESMAS regras do gerador do banco (`generator.py`): a compra
normal respeita o perfil e os hábitos do cliente (cidades, valor, janela de
horário, dispositivos, canal), e a fraude é um dos 6 cenários, montado com a
mesma lógica dos injetores. Por isso a transação já nasce com o rótulo
(`is_fraud`, `fraud_scenario`) — o "rótulo real" que o dashboard ao vivo
compara com o veredito do motor.

O rótulo é usado só para GERAR a transação; quem avalia recebe apenas os
dados brutos (cliente, valor, local, horário, dispositivo, canal) e calcula
as features com `compute_transaction_features`, sem ver o rótulo.

A transação é posterior à última do cliente no banco (é o "próximo passo"
do histórico) e não é gravada: o histórico usado pelas features é sempre o
do banco.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.synthetic.generator import (
    CITY_BY_NAME,
    REFERENCE_END_DATE,
    _random_amount,
    _timestamp_on_day,
    pick_channel,
)
from src.synthetic.geo_reference import CITIES, CityRef, distance_km
from src.synthetic.models import BehaviorProfile, Customer, FraudScenario, Merchant, Transaction

# As compras normais caem de 1 a 3 dias depois da última do cliente: com
# ao menos um dia de folga, nenhuma distância vira "viagem impossível" sem
# querer (mesmo raciocínio de MIN_GAP_DAYS_BETWEEN_STAYS no gerador).
NORMAL_DAYS_AFTER = (1, 3)
# Viagem impossível: o mínimo de distância e a faixa de velocidade dos injetores.
IMPOSSIBLE_MIN_KM = 300.0
IMPOSSIBLE_SPEED_KMH = (1200.0, 2500.0)


@dataclass(frozen=True)
class LiveTransaction:
    """Dados brutos de uma transação nova + o rótulo com que foi gerada."""

    customer_id: int
    city: str
    lat: float
    lon: float
    amount: float
    timestamp: datetime
    device_id: int | None  # None = dispositivo novo, nunca visto
    channel: str
    payment_method: str | None
    is_fraud: bool
    fraud_scenario: str | None


@dataclass
class _Context:
    customer: Customer
    last: Transaction | None
    known_cities: set[str]
    devices: list[int]  # dispositivos das compras legítimas do cliente


def _context(session: Session, customer_id: int) -> _Context:
    customer = session.get(Customer, customer_id)
    if customer is None:
        raise ValueError(f"cliente {customer_id} não existe")
    rows = session.execute(
        select(Transaction, Merchant.city)
        .join(Merchant, Merchant.id == Transaction.merchant_id)
        .where(Transaction.customer_id == customer_id)
        .order_by(Transaction.timestamp, Transaction.id)
    ).all()
    legit = [(t, c) for t, c in rows if not t.is_fraud]
    return _Context(
        customer=customer,
        last=rows[-1][0] if rows else None,
        known_cities={customer.home_city} | {c for _, c in legit},
        devices=sorted({t.device_id for t, _ in legit}),
    )


def _base_day(ctx: _Context, rng: random.Random):
    start = ctx.last.timestamp if ctx.last else REFERENCE_END_DATE
    return (start + timedelta(days=rng.randint(*NORMAL_DAYS_AFTER))).date()


def _usual_city(ctx: _Context, rng: random.Random) -> CityRef:
    """Cliente fixo compra na cidade natal; viajante, numa das que conhece."""
    if ctx.customer.behavior_profile == BehaviorProfile.VIAJANTE:
        return CITY_BY_NAME[rng.choice(sorted(ctx.known_cities))]
    return CITY_BY_NAME[ctx.customer.home_city]


def _hour_outside_window(customer: Customer, rng: random.Random) -> int:
    excluded = set(range(max(0, customer.typical_hour_start - 1), min(24, customer.typical_hour_end + 2)))
    candidates = [h for h in range(24) if h not in excluded]
    return rng.choice(candidates) if candidates else (customer.typical_hour_start + 12) % 24


def _at(day, hour: int, minute: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute)


def generate_live_transaction(
    session: Session,
    rng: random.Random,
    customer_id: int,
    fraud_rate: float,
) -> LiveTransaction:
    """Uma transação nova do cliente: normal, ou (com probabilidade
    `fraud_rate`) um dos 6 cenários de fraude, sorteado por igual.
    """
    ctx = _context(session, customer_id)
    customer = ctx.customer
    device = rng.choice(ctx.devices) if ctx.devices else None
    scenario: FraudScenario | None = rng.choice(list(FraudScenario)) if rng.random() < fraud_rate else None

    city = _usual_city(ctx, rng)
    day = _base_day(ctx, rng)
    timestamp = _timestamp_on_day(rng, customer, day)
    amount = _random_amount(rng, customer)

    if scenario == FraudScenario.ANOMALIA_GEOGRAFICA:
        unknown = [c for c in CITIES if c.city not in ctx.known_cities]
        if unknown:
            city = rng.choice(unknown)
        else:  # conhece todas as cidades: vira a combinação de sinais
            scenario = FraudScenario.COMBINACAO_DE_SINAIS
    if scenario == FraudScenario.VIAGEM_IMPOSSIVEL:
        far = [c for c in CITIES if ctx.last and distance_km(ctx.last.lat, ctx.last.lon, c.lat, c.lon) >= IMPOSSIBLE_MIN_KM]
        if far:
            city = rng.choice(far)
            km = distance_km(ctx.last.lat, ctx.last.lon, city.lat, city.lon)
            timestamp = ctx.last.timestamp + timedelta(hours=km / rng.uniform(*IMPOSSIBLE_SPEED_KMH))
        else:  # sem compra anterior para comparar
            scenario = FraudScenario.VALOR_ATIPICO
    if scenario == FraudScenario.VALOR_ATIPICO:
        amount = customer.avg_amount_baseline * rng.uniform(6.0, 15.0)
    elif scenario == FraudScenario.HORARIO_ATIPICO:
        timestamp = _at(day, _hour_outside_window(customer, rng), rng.randint(0, 59))
    elif scenario == FraudScenario.COMBINACAO_DE_SINAIS:
        city = rng.choice([c for c in CITIES if c.city != customer.home_city])
        amount = customer.avg_amount_baseline * rng.uniform(2.5, 4.5)
        timestamp = _at(day, (customer.typical_hour_end + rng.randint(2, 4)) % 24, rng.randint(0, 59))
    elif scenario == FraudScenario.DISPOSITIVO_NOVO:
        device = None

    # estabelecimento sorteado na cidade (~5 km do centro, como no banco)
    lat = city.lat + rng.uniform(-0.05, 0.05)
    lon = city.lon + rng.uniform(-0.05, 0.05)
    amount = round(amount, 2)
    channel, method = pick_channel(rng, customer, amount, scenario.value if scenario else None)
    return LiveTransaction(
        customer_id=customer.id,
        city=city.city,
        lat=lat,
        lon=lon,
        amount=amount,
        timestamp=timestamp,
        device_id=device,
        channel=channel,
        payment_method=method,
        is_fraud=scenario is not None,
        fraud_scenario=scenario.value if scenario else None,
    )
