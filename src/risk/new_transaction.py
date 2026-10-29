"""Pontuação de uma transação NOVA (base para o simulador da Etapa 10).

Recebe dados brutos — cliente, valor, coordenadas, horário, dispositivo —,
calcula as features a partir do histórico daquele cliente no banco
(reaproveitando as funções puras de `src/synthetic/features.py`, as mesmas
usadas por `build_feature_table`) e chama o Risk Engine.

Nada é gravado no banco: a transação é só avaliada.

"Histórico" = transações do cliente com horário ESTRITAMENTE anterior ao da
transação nova. É a mesma definição de `build_feature_table` (cliente,
horário, id), exceto quando há outra transação do mesmo cliente no mesmo
minuto: a tabela desempata pelo id, e uma transação nova ainda não tem id
— aqui as do mesmo minuto não contam como anteriores.

Cidade: a tabela usa `Merchant.city`. Uma transação nova recebe só
coordenadas; sem `city` explícita, usa-se a cidade de referência mais
próxima (`nearest_city`). Os estabelecimentos do domínio ficam a ~5 km do
centro da sua cidade e as cidades de referência estão a dezenas de km umas
das outras, então a inferência coincide com `Merchant.city`.

Canal: sem `channel`, a transação é presencial com senha — o caso que não
acende nenhum sinal de canal sozinho. Compra online não tem forma de
pagamento presencial (`payment_method` é ignorado).
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from types import SimpleNamespace
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.risk.engine import RiskEngine, RiskResult
from src.synthetic.feature_pipeline import habit_context
from src.synthetic.features import (
    HabitProfile,
    channel_rarity,
    habit_features,
    implied_travel_speed,
    is_contactless,
    is_online,
    location_distance,
    new_city_flag,
    new_device_flag,
)
from src.synthetic.geo_reference import CITIES, distance_km
from src.synthetic.models import Channel, Customer, Device, Merchant, PaymentMethod, Transaction


def nearest_city(lat: float, lon: float) -> str:
    """Nome da cidade de referência mais próxima das coordenadas."""
    return min(CITIES, key=lambda c: distance_km(lat, lon, c.lat, c.lon)).city


def compute_transaction_features(
    session: Session,
    *,
    customer_id: int,
    amount: float,
    lat: float,
    lon: float,
    timestamp: datetime,
    device_id: int | None,
    city: str | None = None,
    channel: str = Channel.PRESENCIAL.value,
    payment_method: str | None = PaymentMethod.SENHA.value,
) -> dict[str, Any]:
    """Features de risco (as chaves de `src.risk.dataset.RISK_FEATURES`) de
    uma transação nova, a partir do histórico do cliente no banco.

    `city` = cidade do estabelecimento; se omitida, é inferida das
    coordenadas (`nearest_city`).

    Dispositivo desconhecido (`device_id` que não existe, que pertence a
    outro cliente, ou `None`) é tratado como NÃO confiável: não há registro
    de confiança para ele.
    """
    customer = session.get(Customer, customer_id)
    if customer is None:
        raise ValueError(f"cliente {customer_id} não existe")
    if amount <= 0:
        raise ValueError(f"valor precisa ser > 0, recebido {amount!r}")

    rows = session.execute(
        select(Transaction, Merchant.city)
        .join(Merchant, Merchant.id == Transaction.merchant_id)
        .where(Transaction.customer_id == customer_id, Transaction.timestamp < timestamp)
        .order_by(Transaction.timestamp, Transaction.id)
    ).all()
    history = [tx for tx, _ in rows]
    previous = history[-1] if history else None
    seen_devices = {t.device_id for t in history}
    seen_cities = {c for _, c in rows}
    seen_channels = Counter(t.channel for t in history)
    habit = HabitProfile(history)
    city = city or nearest_city(lat, lon)
    if channel not in {c.value for c in Channel}:
        raise ValueError(f"canal desconhecido: {channel!r}")
    if channel == Channel.ONLINE.value:
        payment_method = None

    km = distance_km(customer.home_lat, customer.home_lon, lat, lon)
    tx = SimpleNamespace(
        amount=amount,
        timestamp=timestamp,
        distance_from_home_km=km,
        device_id=device_id,
        lat=lat,
        lon=lon,
        channel=channel,
        payment_method=payment_method,
    )
    travel = implied_travel_speed(tx, previous)
    deviation, hour_gap = habit_features(tx, habit)

    device = session.get(Device, device_id) if device_id is not None else None
    trusted = device is not None and device.customer_id == customer_id and bool(device.is_trusted)

    return {
        "amount_deviation": deviation,
        "location_distance": location_distance(tx),
        "unusual_hour": hour_gap,
        "new_device": new_device_flag(tx, seen_devices),
        "new_city": new_city_flag(city, customer, seen_cities),
        # mesma convenção de build_feature_table: sem transação anterior, 0.0
        "implied_speed_kmh": 0.0 if travel.speed_kmh is None else travel.speed_kmh,
        "is_impossible_travel": travel.is_impossible_travel,
        "is_online": is_online(tx),
        "is_contactless": is_contactless(tx),
        "channel_rarity": channel_rarity(channel, seen_channels),
        "distance_from_home_km": km,
        "device_untrusted": not trusted,
        "merchant_city": city,
        **habit_context(habit),
    }


def score_new_transaction(
    session: Session,
    engine: RiskEngine,
    *,
    customer_id: int,
    amount: float,
    lat: float,
    lon: float,
    timestamp: datetime,
    device_id: int | None,
    city: str | None = None,
    channel: str = Channel.PRESENCIAL.value,
    payment_method: str | None = PaymentMethod.SENHA.value,
) -> RiskResult:
    """Calcula as features da transação nova e devolve a avaliação do engine."""
    features = compute_transaction_features(
        session,
        customer_id=customer_id,
        amount=amount,
        lat=lat,
        lon=lon,
        timestamp=timestamp,
        device_id=device_id,
        city=city,
        channel=channel,
        payment_method=payment_method,
    )
    return engine.assess(features)
