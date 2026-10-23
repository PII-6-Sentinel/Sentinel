"""Camada fina do Monitor ao Vivo (sem Streamlit).

O monitor gera um FLUXO de transações novas, uma de cada vez, para clientes
aleatórios do banco sintético, e passa cada uma pelo Risk Engine — como se
fossem compras chegando em tempo real numa operação antifraude.

Nada aqui decide o veredito. Cada transação é montada a partir dos cenários
do Simulador (`build_scenarios`, calculados do histórico REAL do cliente
sorteado) e avaliada com `engine.assess(compute_transaction_features(...))`,
a mesma chamada de `score_new_transaction`. O monitor só acrescenta:

- o SORTEIO: qual cliente, e se a transação é uma compra normal ou um dos
  cenários de fraude (com a proporção escolhida na tela);
- uma pequena VARIAÇÃO no valor e no horário, para o fluxo não repetir
  sempre os mesmos números (o horário só anda para frente, então a compra
  continua posterior à última do cliente e não vira viagem impossível sem
  querer);
- o RESUMO do fluxo (contagens por ação, fraudes injetadas capturadas,
  falsos alarmes, valor retido) para os indicadores da tela.

"Fraude injetada" é o rótulo da SIMULAÇÃO (o cenário sorteado), não uma
verdade sobre fraude real. Cada transação é avaliada contra o histórico do
cliente no banco; as do próprio fluxo não entram nesse histórico — nada é
gravado no banco.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import pandas as pd
from simulator import (
    CITY_BY_NAME,
    NEW_DEVICE,
    SCENARIO_IMPOSSIBLE_TRAVEL,
    SCENARIO_NORMAL,
    SCENARIOS,
    CustomerContext,
    FormValues,
    build_scenarios,
    load_context,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.risk.engine import ACTION_APPROVE, ACTION_BLOCK, ACTION_REVIEW, RiskEngine, RiskResult
from src.risk.new_transaction import compute_transaction_features
from src.synthetic.models import Customer

SCENARIO_LABEL = dict(SCENARIOS)
FRAUD_SCENARIOS = [key for key, _ in SCENARIOS if key != SCENARIO_NORMAL]

# Quantas transações o monitor guarda (as mais antigas saem do fluxo).
MAX_EVENTS = 500

# Variação aplicada ao valor do cenário: compra normal varia mais (é o dia a
# dia do cliente); nos cenários de fraude a variação é pequena para não
# descaracterizar o sinal (ex.: "~10x a média").
NORMAL_AMOUNT_RANGE = (0.6, 1.5)
FRAUD_AMOUNT_RANGE = (0.9, 1.15)
# Minutos acrescentados ao horário do cenário (só para frente).
MAX_EXTRA_MINUTES = 40


@dataclass(frozen=True)
class LiveEvent:
    """Uma transação do fluxo, já avaliada pelo motor."""

    seq: int
    received_at: datetime  # relógio de parede: quando chegou ao monitor
    customer_id: int
    customer_name: str
    home_city: str
    home_lat: float
    home_lon: float
    scenario: str
    city: str
    lat: float
    lon: float
    amount: float
    timestamp: datetime  # horário da transação no domínio sintético
    device_id: int  # NEW_DEVICE = dispositivo novo, não reconhecido
    device_label: str
    features: dict
    assessment: RiskResult

    @property
    def scenario_label(self) -> str:
        return SCENARIO_LABEL[self.scenario]

    @property
    def injected_fraud(self) -> bool:
        return self.scenario != SCENARIO_NORMAL

    @property
    def flagged(self) -> bool:
        """O motor NÃO aprovou automaticamente (revisar ou bloquear)."""
        return self.assessment.action != ACTION_APPROVE


def customer_ids(session: Session) -> list[int]:
    return sorted(session.scalars(select(Customer.id)))


def _device_label(ctx: CustomerContext, device_id: int) -> str:
    if device_id == NEW_DEVICE:
        return "novo, não reconhecido"
    d = next((d for d in ctx.devices if d.id == device_id), None)
    if d is None:
        return f"#{device_id}"
    return f"{d.device_type} #{d.id} ({'confiável' if d.is_trusted else 'não confiável'})"


def pick_form(ctx: CustomerContext, rng: random.Random, fraud_rate: float) -> tuple[str, FormValues]:
    """Sorteia o cenário (normal ou um de fraude que dê para montar para este
    cliente) e aplica a variação de valor e horário.
    """
    scenarios = build_scenarios(ctx)
    available = [key for key in FRAUD_SCENARIOS if scenarios.get(key) is not None]
    if available and rng.random() < fraud_rate:
        key = rng.choice(available)
        low, high = FRAUD_AMOUNT_RANGE
    else:
        key = SCENARIO_NORMAL
        low, high = NORMAL_AMOUNT_RANGE
    form = scenarios[key]
    amount = round(max(0.01, form.amount * rng.uniform(low, high)), 2)
    # viagem impossível depende de poucos minutos desde a última compra: não mexe
    extra = 0 if key == SCENARIO_IMPOSSIBLE_TRAVEL else rng.randint(0, MAX_EXTRA_MINUTES)
    return key, replace(form, amount=amount, timestamp=form.timestamp + timedelta(minutes=extra))


def next_event(
    session: Session,
    engine: RiskEngine,
    rng: random.Random,
    seq: int,
    fraud_rate: float,
    ids: list[int],
    now: datetime | None = None,
) -> LiveEvent:
    """Gera e avalia a próxima transação do fluxo."""
    if not ids:
        raise ValueError("o banco sintético não tem clientes")
    ctx = load_context(session, rng.choice(ids))
    scenario, form = pick_form(ctx, rng, fraud_rate)
    city = CITY_BY_NAME[form.city]
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
    )
    return LiveEvent(
        seq=seq,
        received_at=now or datetime.now(),
        customer_id=ctx.customer_id,
        customer_name=ctx.name,
        home_city=ctx.home_city,
        home_lat=ctx.home_lat,
        home_lon=ctx.home_lon,
        scenario=scenario,
        city=city.city,
        lat=city.lat,
        lon=city.lon,
        amount=form.amount,
        timestamp=form.timestamp,
        device_id=form.device_id,
        device_label=_device_label(ctx, form.device_id),
        features=features,
        assessment=engine.assess(features),
    )


# ----------------------------------------------------------------------
# resumo do fluxo
# ----------------------------------------------------------------------


def summarize(events: list[LiveEvent]) -> dict:
    """Indicadores do fluxo. "Capturada" = fraude injetada que o motor não
    aprovou; "falso alarme" = compra normal que o motor não aprovou.
    """
    actions = Counter(e.assessment.action for e in events)
    injected = [e for e in events if e.injected_fraud]
    normal = [e for e in events if not e.injected_fraud]
    caught = sum(e.flagged for e in injected)
    false_alarms = sum(e.flagged for e in normal)
    return {
        "total": len(events),
        "approved": actions.get(ACTION_APPROVE, 0),
        "review": actions.get(ACTION_REVIEW, 0),
        "blocked": actions.get(ACTION_BLOCK, 0),
        "injected": len(injected),
        "caught": caught,
        "detection_rate": caught / len(injected) if injected else None,
        "normal": len(normal),
        "false_alarms": false_alarms,
        "false_alarm_rate": false_alarms / len(normal) if normal else None,
        "amount_total": sum(e.amount for e in events),
        "amount_held": sum(e.amount for e in events if e.flagged),
        "avg_score": sum(e.assessment.risk_score for e in events) / len(events) if events else None,
    }


def events_frame(events: list[LiveEvent]) -> pd.DataFrame:
    """Uma linha por transação (mais recente primeiro), para tabela, mapa e gráficos."""
    columns = [
        "#", "Chegada", "Cliente", "Valor", "Cidade", "Data/hora", "Dispositivo", "Cenário",
        "Risk Score", "Nível", "Ação", "Distância (km)", "lat", "lon", "fraude_injetada",
    ]  # fmt: skip
    rows = [
        (
            e.seq,
            e.received_at,
            e.customer_name,
            e.amount,
            e.city,
            e.timestamp,
            e.device_label,
            e.scenario_label,
            e.assessment.risk_score,
            e.assessment.level,
            e.assessment.action,
            e.features["distance_from_home_km"],
            e.lat,
            e.lon,
            e.injected_fraud,
        )
        for e in reversed(events)
    ]
    return pd.DataFrame(rows, columns=columns)


def reason_counts(events: list[LiveEvent]) -> pd.Series:
    """Quantas transações SINALIZADAS citaram cada motivo."""
    counter: Counter = Counter()
    for e in events:
        if e.flagged:
            counter.update({_reason_key(r) for r in e.assessment.reasons})
    return pd.Series(counter, dtype=int).sort_values(ascending=True)


def _reason_key(reason: str) -> str:
    """Motivo sem os números (ex.: "Valor 3,2x acima da média do cliente" →
    "Valor acima da média do cliente"), para agrupar transações diferentes.
    """
    for sep in (":", "("):
        reason = reason.split(sep)[0]
    reason = re.sub(r"\d[\d.,]*x?", "", reason)
    return " ".join(reason.split()).rstrip(".")


def scenario_detection(events: list[LiveEvent]) -> pd.DataFrame:
    """Por cenário: total, quantas o motor sinalizou e a taxa."""
    rows = []
    for key, label in SCENARIOS:
        group = [e for e in events if e.scenario == key]
        if group:
            flagged = sum(e.flagged for e in group)
            rows.append((label, len(group), flagged, flagged / len(group)))
    return pd.DataFrame(rows, columns=["Cenário", "Total", "Sinalizadas", "Taxa"])
