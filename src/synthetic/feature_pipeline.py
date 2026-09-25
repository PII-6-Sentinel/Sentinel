"""Monta a tabela de features a partir do banco do domínio sintético.

Só orquestração: lê as transações do banco, percorre cada cliente em ordem
cronológica mantendo o "estado" que as features precisam (transação
anterior, dispositivos já vistos) e chama as funções puras de
`src/synthetic/features.py`. Nenhuma lógica de risco aqui.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.synthetic.features import (
    amount_deviation,
    implied_travel_speed,
    location_distance,
    new_device_flag,
    unusual_hour,
)
from src.synthetic.models import Customer, Transaction

FEATURE_COLUMNS = [
    "amount_deviation",
    "location_distance",
    "unusual_hour",
    "new_device",
    "implied_speed_kmh",
    "is_impossible_travel",
]
TABLE_COLUMNS = ["transaction_id", *FEATURE_COLUMNS, "is_fraud", "fraud_scenario"]


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

    `fraud_scenario` é None para transações normais — esse é o único
    "vazio" esperado na tabela.
    """
    customers = {c.id: c for c in session.scalars(select(Customer))}
    transactions = session.scalars(
        select(Transaction).order_by(Transaction.customer_id, Transaction.timestamp, Transaction.id)
    )

    previous: dict[int, Transaction] = {}
    seen_devices: dict[int, set[int]] = {}
    rows = []

    for tx in transactions:
        cid = tx.customer_id
        customer = customers[cid]
        devices = seen_devices.setdefault(cid, set())
        travel = implied_travel_speed(tx, previous.get(cid))

        rows.append(
            {
                "transaction_id": tx.id,
                "amount_deviation": amount_deviation(tx, customer),
                "location_distance": location_distance(tx),
                "unusual_hour": unusual_hour(tx, customer),
                "new_device": new_device_flag(tx, devices),
                "implied_speed_kmh": 0.0 if travel.speed_kmh is None else travel.speed_kmh,
                "is_impossible_travel": travel.is_impossible_travel,
                "is_fraud": tx.is_fraud,
                "fraud_scenario": tx.fraud_scenario,
            }
        )

        devices.add(tx.device_id)
        previous[cid] = tx

    return pd.DataFrame(rows, columns=TABLE_COLUMNS)
