"""Camada fina do Monitor ao Vivo (sem Streamlit).

O monitor gera um FLUXO de transações novas, uma de cada vez, para clientes
aleatórios do banco sintético, e passa cada uma pelo Risk Engine — como se
fossem compras chegando em tempo real numa operação antifraude.

Cada transação vem do gerador ao vivo (`src/synthetic/live_generator.py`),
que segue as mesmas regras do gerador do banco e por isso traz o RÓTULO
REAL (legítima, ou fraude de um dos 6 cenários). O motor recebe só os dados
brutos — `engine.assess(compute_transaction_features(...))`, a mesma chamada
de `score_new_transaction` — e nunca vê o rótulo. O monitor acrescenta:

- o VEREDITO binário: "Fraudulenta" quando o motor não aprova (revisar ou
  bloquear — o mesmo critério da avaliação, "sinalizada = não aprovada
  automaticamente"), "Legítima" quando aprova;
- a comparação veredito × rótulo real e o RESUMO do fluxo (matriz de
  confusão, acurácia, precisão, recall, valor retido) para o painel.

Nada é gravado no banco: cada transação é avaliada contra o histórico do
cliente no banco; as do próprio fluxo não entram nesse histórico.
"""

from __future__ import annotations

import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.risk.engine import ACTION_APPROVE, ACTION_BLOCK, ACTION_REVIEW, RiskEngine, RiskResult
from src.risk.new_transaction import compute_transaction_features
from src.synthetic.live_generator import generate_live_transaction
from src.synthetic.models import Channel, Customer, Device, FraudScenario, PaymentMethod

NORMAL = "normal"
SCENARIO_LABEL = {
    NORMAL: "Compra normal",
    FraudScenario.ANOMALIA_GEOGRAFICA.value: "Anomalia geográfica",
    FraudScenario.VIAGEM_IMPOSSIVEL.value: "Viagem impossível",
    FraudScenario.VALOR_ATIPICO.value: "Valor atípico",
    FraudScenario.HORARIO_ATIPICO.value: "Horário atípico",
    FraudScenario.COMBINACAO_DE_SINAIS.value: "Combinação de sinais",
    FraudScenario.DISPOSITIVO_NOVO.value: "Dispositivo novo",
}
VERDICT_FRAUD, VERDICT_LEGIT = "Fraudulenta", "Legítima"
LABEL_FRAUD, LABEL_LEGIT = "Fraude", "Legítima"

CHANNEL_LABEL = {
    (Channel.ONLINE.value, None): "Online",
    (Channel.PRESENCIAL.value, PaymentMethod.APROXIMACAO.value): "Presencial · aproximação",
    (Channel.PRESENCIAL.value, PaymentMethod.SENHA.value): "Presencial · senha",
}

# Semente do modo apresentação: com ela (e os mesmos controles), o fluxo
# sorteia sempre a mesma sequência de clientes e transações — dá para ensaiar.
PRESENTATION_SEED = 2026

# Quantas transações o monitor guarda (as mais antigas saem do fluxo).
MAX_EVENTS = 500


@dataclass(frozen=True)
class LiveEvent:
    """Uma transação do fluxo, já avaliada pelo motor, com o rótulo real."""

    seq: int
    received_at: datetime  # relógio de parede: quando chegou ao monitor
    customer_id: int
    customer_name: str
    home_city: str
    city: str
    lat: float
    lon: float
    amount: float
    timestamp: datetime  # horário da transação no domínio sintético
    device_label: str
    channel: str
    payment_method: str | None
    is_fraud: bool  # rótulo real (do gerador) — o motor não o vê
    scenario: str  # NORMAL ou o .value do FraudScenario
    features: dict
    assessment: RiskResult
    decision_ms: float  # cálculo das features + avaliação do motor

    @property
    def scenario_label(self) -> str:
        return SCENARIO_LABEL[self.scenario]

    @property
    def channel_label(self) -> str:
        return CHANNEL_LABEL[(self.channel, self.payment_method)]

    @property
    def flagged(self) -> bool:
        """O motor NÃO aprovou automaticamente (revisar ou bloquear)."""
        return self.assessment.action != ACTION_APPROVE

    @property
    def verdict(self) -> str:
        return VERDICT_FRAUD if self.flagged else VERDICT_LEGIT

    @property
    def label(self) -> str:
        return LABEL_FRAUD if self.is_fraud else LABEL_LEGIT

    @property
    def correct(self) -> bool:
        return self.flagged == self.is_fraud


def customer_ids(session: Session) -> list[int]:
    return sorted(session.scalars(select(Customer.id)))


def _device_label(session: Session, device_id: int | None) -> str:
    if device_id is None:
        return "novo, não reconhecido"
    d = session.get(Device, device_id)
    kind = getattr(d.device_type, "value", d.device_type)
    return f"{kind} #{d.id} ({'confiável' if d.is_trusted else 'não confiável'})"


def next_event(
    session: Session,
    engine: RiskEngine,
    rng: random.Random,
    seq: int,
    fraud_rate: float,
    ids: list[int],
    now: datetime | None = None,
) -> LiveEvent:
    """Gera a próxima transação do fluxo e a avalia (sem passar o rótulo)."""
    if not ids:
        raise ValueError("o banco sintético não tem clientes")
    tx = generate_live_transaction(session, rng, rng.choice(ids), fraud_rate)
    customer = session.get(Customer, tx.customer_id)
    t0 = time.perf_counter()
    features = compute_transaction_features(
        session,
        customer_id=tx.customer_id,
        amount=tx.amount,
        lat=tx.lat,
        lon=tx.lon,
        timestamp=tx.timestamp,
        device_id=tx.device_id,
        city=tx.city,
        channel=tx.channel,
        payment_method=tx.payment_method,
    )
    assessment = engine.assess(features)
    decision_ms = (time.perf_counter() - t0) * 1000
    return LiveEvent(
        seq=seq,
        received_at=now or datetime.now(),
        customer_id=customer.id,
        customer_name=customer.name,
        home_city=customer.home_city,
        city=tx.city,
        lat=tx.lat,
        lon=tx.lon,
        amount=tx.amount,
        timestamp=tx.timestamp,
        device_label=_device_label(session, tx.device_id),
        channel=tx.channel,
        payment_method=tx.payment_method,
        is_fraud=tx.is_fraud,
        scenario=tx.fraud_scenario or NORMAL,
        features=features,
        assessment=assessment,
        decision_ms=decision_ms,
    )


# ----------------------------------------------------------------------
# resumo do fluxo
# ----------------------------------------------------------------------


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def summarize(events: list[LiveEvent]) -> dict:
    """Indicadores do fluxo: veredito × rótulo real (matriz de confusão) e
    contagens por ação. Positivo = fraude.
    """
    actions = Counter(e.assessment.action for e in events)
    tp = sum(e.flagged and e.is_fraud for e in events)
    fp = sum(e.flagged and not e.is_fraud for e in events)
    fn = sum(not e.flagged and e.is_fraud for e in events)
    tn = sum(not e.flagged and not e.is_fraud for e in events)
    return {
        "total": len(events),
        "approved": actions.get(ACTION_APPROVE, 0),
        "review": actions.get(ACTION_REVIEW, 0),
        "blocked": actions.get(ACTION_BLOCK, 0),
        "flagged": tp + fp,
        "frauds": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": _ratio(tp + tn, len(events)),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "amount_total": sum(e.amount for e in events),
        "amount_held": sum(e.amount for e in events if e.flagged),
        "fraud_amount_held": sum(e.amount for e in events if e.flagged and e.is_fraud),
        "avg_decision_ms": sum(e.decision_ms for e in events) / len(events) if events else None,
    }


def events_frame(events: list[LiveEvent]) -> pd.DataFrame:
    """Uma linha por transação (mais recente primeiro), para tabela, mapa e gráficos."""
    columns = [
        "#", "Chegada", "Cliente", "Valor", "Cidade", "Canal", "Data/hora", "Dispositivo",
        "Veredito", "Rótulo real", "Acerto", "Cenário", "Risk Score", "Nível", "Ação",
        "Distância (km)", "Decisão (ms)", "lat", "lon",
    ]  # fmt: skip
    rows = [
        (
            e.seq,
            e.received_at,
            e.customer_name,
            e.amount,
            e.city,
            e.channel_label,
            e.timestamp,
            e.device_label,
            e.verdict,
            e.label,
            "✓" if e.correct else "✗",
            e.scenario_label,
            e.assessment.risk_score,
            e.assessment.level,
            e.assessment.action,
            e.features["distance_from_home_km"],
            e.decision_ms,
            e.lat,
            e.lon,
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
    reason = re.sub(r"\d[\d.,]*x?%?", "", reason)
    return " ".join(reason.split()).rstrip(".,")


def scenario_detection(events: list[LiveEvent]) -> pd.DataFrame:
    """Por cenário (rótulo real): total, quantas o motor chamou de fraudulentas e a taxa."""
    rows = []
    for key, label in SCENARIO_LABEL.items():
        group = [e for e in events if e.scenario == key]
        if group:
            flagged = sum(e.flagged for e in group)
            rows.append((label, len(group), flagged, flagged / len(group)))
    return pd.DataFrame(rows, columns=["Cenário", "Total", "Sinalizadas", "Taxa"])
