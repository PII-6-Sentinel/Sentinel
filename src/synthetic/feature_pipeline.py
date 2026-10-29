"""Monta a tabela de features a partir do banco do domínio sintético.

Só orquestração: lê as transações do banco, percorre cada cliente em ordem
cronológica mantendo o "estado" que as features precisam (transação
anterior, dispositivos, cidades e canais já vistos, hábitos de valor e
horário) e chama as funções puras de `src/synthetic/features.py`. Nenhuma
lógica de risco aqui.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

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
from src.synthetic.models import Customer, Merchant, Transaction

FEATURE_COLUMNS = [
    "amount_deviation",
    "location_distance",
    "unusual_hour",
    "new_device",
    "new_city",
    "implied_speed_kmh",
    "is_impossible_travel",
    "is_online",
    "is_contactless",
    "channel_rarity",
]
# Hábito estimado no momento da transação (só com as compras anteriores):
# não é feature, é o contexto que explica `amount_deviation`/`unusual_hour`.
# Vazio enquanto o cliente não tem histórico suficiente.
HABIT_COLUMNS = ["habit_avg_amount", "habit_hour_start", "habit_hour_end"]
TABLE_COLUMNS = ["transaction_id", *FEATURE_COLUMNS, *HABIT_COLUMNS, "is_fraud", "fraud_scenario"]


def build_feature_table(session: Session) -> pd.DataFrame:
    """Uma linha por transação: `transaction_id`, as features, `is_fraud` e
    `fraud_scenario`.

    Transações são percorridas por (cliente, timestamp, id) — a ordem que
    dá sentido a "transação anterior" e "dispositivo já visto". O id
    desempata timestamps iguais (resolução de minuto).

    Primeira transação de um cliente não tem anterior: `implied_speed_kmh`
    vira 0.0 e `is_impossible_travel` False, explicitamente (a função de
    feature devolve None; a conversão para 0.0 acontece aqui, em um só
    lugar, para a tabela nunca ter NaN nas colunas de feature).

    Valor e horário são comparados com o hábito estimado das compras
    ANTERIORES (`HabitProfile`), nunca com os parâmetros do gerador
    guardados em `Customer`.

    `fraud_scenario` é None para transações normais e as `HABIT_COLUMNS`
    ficam vazias enquanto não há histórico suficiente — os únicos "vazios"
    esperados na tabela.
    """
    customers = {c.id: c for c in session.scalars(select(Customer))}
    merchant_city = dict(session.execute(select(Merchant.id, Merchant.city)).all())
    transactions = session.scalars(
        select(Transaction).order_by(Transaction.customer_id, Transaction.timestamp, Transaction.id)
    )

    previous: dict[int, Transaction] = {}
    seen_devices: dict[int, set[int]] = {}
    seen_cities: dict[int, set[str]] = {}
    seen_channels: dict[int, Counter] = {}
    habits: dict[int, HabitProfile] = {}
    rows = []

    for tx in transactions:
        cid = tx.customer_id
        customer = customers[cid]
        devices = seen_devices.setdefault(cid, set())
        cities = seen_cities.setdefault(cid, set())
        channels = seen_channels.setdefault(cid, Counter())
        habit = habits.setdefault(cid, HabitProfile())
        city = merchant_city[tx.merchant_id]
        travel = implied_travel_speed(tx, previous.get(cid))
        deviation, hour_gap = habit_features(tx, habit)

        rows.append(
            {
                "transaction_id": tx.id,
                "amount_deviation": deviation,
                "location_distance": location_distance(tx),
                "unusual_hour": hour_gap,
                "new_device": new_device_flag(tx, devices),
                "new_city": new_city_flag(city, customer, cities),
                "implied_speed_kmh": 0.0 if travel.speed_kmh is None else travel.speed_kmh,
                "is_impossible_travel": travel.is_impossible_travel,
                "is_online": is_online(tx),
                "is_contactless": is_contactless(tx),
                "channel_rarity": channel_rarity(tx.channel, channels),
                **habit_context(habit),
                "is_fraud": tx.is_fraud,
                "fraud_scenario": tx.fraud_scenario,
            }
        )

        devices.add(tx.device_id)
        cities.add(city)
        channels[tx.channel] += 1
        habit.add(tx)
        previous[cid] = tx

    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def habit_context(habit: HabitProfile) -> dict[str, float | None]:
    """As `HABIT_COLUMNS` de uma transação: o hábito com que ela foi
    comparada, ou vazio se ainda não havia histórico suficiente."""
    if not habit.is_known:
        return dict.fromkeys(HABIT_COLUMNS)
    return {
        "habit_avg_amount": habit.avg_amount_baseline,
        "habit_hour_start": habit.typical_hour_start,
        "habit_hour_end": habit.typical_hour_end,
    }
