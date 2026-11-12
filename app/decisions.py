"""Decisão do analista: a etapa que fecha o ciclo da Central de Transações.

O motor RECOMENDA (aprovar, revisar ou bloquear); o analista DECIDE. Esta
camada guarda essa decisão na tabela `analyst_decisions` do banco sintético
(`src/synthetic/models.py`) e deriva o status de cada transação a partir da
recomendação e da decisão:

| Recomendação do motor | Sem decisão do analista          | Com decisão                        |
|-----------------------|----------------------------------|------------------------------------|
| aprovar               | aprovada automaticamente         | aprovada / bloqueada pelo analista |
| revisar               | aguardando revisão (fila)        | aprovada / bloqueada pelo analista |
| bloquear              | bloqueada preventivamente (fila) | aprovada / bloqueada pelo analista |

As transações e o rótulo da simulação nunca são alterados: a decisão vive
na tabela própria. A tabela é criada sob demanda (`ensure_decisions_table`),
então um banco gerado antes dela continua funcionando sem ser regenerado.

Sem Streamlit aqui: a página (`app/views/transactions.py`) só orquestra e
exibe; a lógica é testada em tests/test_app_decisions.py.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

from src.risk.engine import ACTION_APPROVE, ACTION_BLOCK, ACTION_REVIEW
from src.synthetic.models import AnalystDecision

DECISION_APPROVED, DECISION_BLOCKED = "aprovada", "bloqueada"
DECISIONS = [DECISION_APPROVED, DECISION_BLOCKED]

STATUS_AUTO_APPROVED = "Aprovada automaticamente"
STATUS_PENDING_REVIEW = "Aguardando revisão"
STATUS_PREVENTIVE_BLOCK = "Bloqueada preventivamente"
STATUS_ANALYST_APPROVED = "Aprovada pelo analista"
STATUS_ANALYST_BLOCKED = "Bloqueada pelo analista"
STATUSES = [
    STATUS_PENDING_REVIEW,
    STATUS_PREVENTIVE_BLOCK,
    STATUS_ANALYST_APPROVED,
    STATUS_ANALYST_BLOCKED,
    STATUS_AUTO_APPROVED,
]
# Precisam do analista: o motor não aprovou e ninguém decidiu ainda.
QUEUE_STATUSES = [STATUS_PENDING_REVIEW, STATUS_PREVENTIVE_BLOCK]

STATUS_BY_ACTION = {
    ACTION_APPROVE: STATUS_AUTO_APPROVED,
    ACTION_REVIEW: STATUS_PENDING_REVIEW,
    ACTION_BLOCK: STATUS_PREVENTIVE_BLOCK,
}
STATUS_BY_DECISION = {
    DECISION_APPROVED: STATUS_ANALYST_APPROVED,
    DECISION_BLOCKED: STATUS_ANALYST_BLOCKED,
}

DECISION_COLUMNS = ["decision", "recommended_action", "risk_score", "note", "decided_at"]


def _engine(db_path: str | Path):
    return create_engine(f"sqlite:///{db_path}")


def ensure_decisions_table(db_path: str | Path) -> None:
    """Cria a tabela de decisões se ela ainda não existir (não toca nas outras)."""
    engine = _engine(db_path)
    try:
        AnalystDecision.__table__.create(engine, checkfirst=True)
    finally:
        engine.dispose()


def load_decisions(db_path: str | Path) -> pd.DataFrame:
    """Decisões vigentes, indexadas por `transaction_id` (vazio se não houver)."""
    ensure_decisions_table(db_path)
    engine = _engine(db_path)
    try:
        with Session(engine) as session:
            rows = session.execute(
                select(AnalystDecision.transaction_id, *(getattr(AnalystDecision, c) for c in DECISION_COLUMNS))
            ).all()
    finally:
        engine.dispose()
    frame = pd.DataFrame(rows, columns=["transaction_id", *DECISION_COLUMNS])
    return frame.set_index("transaction_id")


def record_decision(
    db_path: str | Path,
    transaction_id: int,
    decision: str,
    recommended_action: str,
    risk_score: int,
    note: str | None = None,
    now: datetime | None = None,
) -> None:
    """Grava (ou substitui) a decisão do analista sobre uma transação."""
    if decision not in DECISIONS:
        raise ValueError(f"decisão inválida: {decision!r} (use {DECISIONS})")
    ensure_decisions_table(db_path)
    note = (note or "").strip() or None
    engine = _engine(db_path)
    try:
        with Session(engine) as session:
            row = session.scalar(select(AnalystDecision).where(AnalystDecision.transaction_id == transaction_id))
            if row is None:
                row = AnalystDecision(transaction_id=transaction_id)
                session.add(row)
            row.decision = decision
            row.recommended_action = recommended_action
            row.risk_score = int(risk_score)
            row.note = note
            row.decided_at = now or datetime.now()
            session.commit()
    finally:
        engine.dispose()


def undo_decision(db_path: str | Path, transaction_id: int) -> None:
    """Apaga a decisão: a transação volta ao status que o motor deu."""
    ensure_decisions_table(db_path)
    engine = _engine(db_path)
    try:
        with Session(engine) as session:
            session.execute(delete(AnalystDecision).where(AnalystDecision.transaction_id == transaction_id))
            session.commit()
    finally:
        engine.dispose()


def with_status(frame: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Cópia de `frame` com as colunas `decision` (None sem decisão) e `status`."""
    out = frame.copy()
    by_id = decisions["decision"].to_dict() if len(decisions) else {}
    decided = [by_id.get(int(t)) for t in out["transaction_id"]]
    # dtype object: sem ele o pandas infere texto e troca o None por NaN
    out["decision"] = pd.Series(decided, index=out.index, dtype=object)
    out["status"] = [
        STATUS_BY_DECISION[d] if d is not None else STATUS_BY_ACTION[a] for d, a in zip(decided, out["action"])
    ]
    return out


def review_queue(frame: pd.DataFrame) -> pd.DataFrame:
    """Transações que esperam o analista, da mais arriscada para a menos.
    `frame` precisa vir de `with_status` (a análise já vem ordenada por risco).
    """
    queue = frame[frame["status"].isin(QUEUE_STATUSES)]
    return queue.sort_values(["raw_score", "transaction_id"], ascending=[False, True])


def next_in_queue(frame: pd.DataFrame, exclude: int | None = None) -> int | None:
    """ID da próxima transação da fila (a de maior risco), pulando `exclude`."""
    queue = review_queue(frame)
    if exclude is not None:
        queue = queue[queue["transaction_id"] != exclude]
    return int(queue["transaction_id"].iloc[0]) if len(queue) else None


def status_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Quantidade de transações em cada status (todos os `STATUSES`, mesmo zerados)."""
    counts = frame["status"].value_counts()
    return {status: int(counts.get(status, 0)) for status in STATUSES}
