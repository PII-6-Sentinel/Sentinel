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
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.risk.engine import RiskEngine, RiskResult
from src.synthetic.features import (
    amount_deviation,
    implied_travel_speed,
    location_distance,
    new_device_flag,
    unusual_hour,
)
from src.synthetic.geo_reference import distance_km
from src.synthetic.models import Customer, Device, Transaction


def compute_transaction_features(
    session: Session,
    *,
    customer_id: int,
    amount: float,
    lat: float,
    lon: float,
    timestamp: datetime,
    device_id: int | None,
) -> dict[str, Any]:
    """Features de risco (as chaves de `src.risk.dataset.RISK_FEATURES`) de
    uma transação nova, a partir do histórico do cliente no banco.

    Dispositivo desconhecido (`device_id` que não existe, que pertence a
    outro cliente, ou `None`) é tratado como NÃO confiável: não há registro
    de confiança para ele.
    """
    customer = session.get(Customer, customer_id)
    if customer is None:
        raise ValueError(f"cliente {customer_id} não existe")
    if amount <= 0:
        raise ValueError(f"valor precisa ser > 0, recebido {amount!r}")

    history = session.scalars(
        select(Transaction)
        .where(Transaction.customer_id == customer_id, Transaction.timestamp < timestamp)
        .order_by(Transaction.timestamp, Transaction.id)
    ).all()
    previous = history[-1] if history else None
    seen_devices = {t.device_id for t in history}

    km = distance_km(customer.home_lat, customer.home_lon, lat, lon)
    tx = SimpleNamespace(
        amount=amount,
        timestamp=timestamp,
        distance_from_home_km=km,
        device_id=device_id,
        lat=lat,
        lon=lon,
    )
    travel = implied_travel_speed(tx, previous)

    device = session.get(Device, device_id) if device_id is not None else None
    trusted = device is not None and device.customer_id == customer_id and bool(device.is_trusted)

    return {
        "amount_deviation": amount_deviation(tx, customer),
        "location_distance": location_distance(tx),
        "unusual_hour": unusual_hour(tx, customer),
        "new_device": new_device_flag(tx, seen_devices),
        # mesma convenção de build_feature_table: sem transação anterior, 0.0
        "implied_speed_kmh": 0.0 if travel.speed_kmh is None else travel.speed_kmh,
        "is_impossible_travel": travel.is_impossible_travel,
        "distance_from_home_km": km,
        "device_untrusted": not trusted,
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
    )
    return engine.assess(features)
