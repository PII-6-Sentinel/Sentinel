"""Tabela de risco: as features comportamentais + o que as regras precisam.

Reaproveita `build_feature_table` (sem modificá-la) e acrescenta, por
transação, colunas que vêm direto do banco:

- `customer_id`: necessário para o split treino/teste POR CLIENTE;
- `distance_from_home_km`: distância bruta, para motivos legíveis
  ("2.727 km da residência") — `location_distance` é a mesma informação
  em escala log 0-1, boa para o modelo mas ruim para um texto;
- `device_untrusted`: `not Device.is_trusted` — sinal que existia no banco
  mas ainda não era feature;
- `merchant_city`: cidade do estabelecimento — não é feature (o modelo não
  a lê), só contexto para o motivo "Primeira compra do cliente nesta cidade".
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.synthetic.feature_pipeline import FEATURE_COLUMNS, build_feature_table
from src.synthetic.models import Device, Merchant, Transaction

# Features que o Risk Engine consome (regras e/ou modelo).
RISK_FEATURES = [*FEATURE_COLUMNS, "distance_from_home_km", "device_untrusted"]

# Colunas descritivas: aparecem nos motivos, nunca entram no modelo.
CONTEXT_COLUMNS = ["merchant_city"]

RISK_TABLE_COLUMNS = ["transaction_id", "customer_id", *RISK_FEATURES, *CONTEXT_COLUMNS, "is_fraud", "fraud_scenario"]


def build_risk_table(session: Session) -> pd.DataFrame:
    """Uma linha por transação, com `RISK_TABLE_COLUMNS`, na mesma ordem de
    `build_feature_table` (cliente, timestamp, id).
    """
    features = build_feature_table(session)

    extra = pd.DataFrame(
        session.execute(
            select(
                Transaction.id.label("transaction_id"),
                Transaction.customer_id,
                Transaction.distance_from_home_km,
                Device.is_trusted,
                Merchant.city,
            )
            .join(Device, Device.id == Transaction.device_id)
            .join(Merchant, Merchant.id == Transaction.merchant_id)
        ).all(),
        columns=["transaction_id", "customer_id", "distance_from_home_km", "is_trusted", "merchant_city"],
    )
    extra["device_untrusted"] = ~extra.pop("is_trusted").astype(bool)

    table = features.merge(extra, on="transaction_id", how="left", validate="one_to_one")
    if table[["customer_id", "distance_from_home_km", "device_untrusted"]].isna().any().any():
        raise ValueError("transação sem cliente/dispositivo correspondente no banco")

    return table[RISK_TABLE_COLUMNS].reset_index(drop=True)
