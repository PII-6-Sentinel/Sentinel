"""Camada fina entre a Central de Transações e o Risk Engine (src/risk/).

Sem Streamlit aqui: tudo é função pura sobre DataFrames, para ser testada
sem interface. A página (`app/views/transactions.py`) só orquestra e exibe.

Nada é pré-calculado nem cacheado: `run_analysis` lê o banco, calcula as
features, treina o ML, aplica as regras, escolhe o peso e calibra as faixas
a cada chamada, e avisa cada etapa (com o tempo gasto) por um callback.

O que é reaproveitado de `src/risk/` (sem alterá-lo):

- `build_risk_table`, `split_by_customer`, `MLScorer`, `behavior_score`;
- a REGRA de escolha do peso (`weight_sweep` + `choose_rules_weight`, na
  validação) e a calibração dos limiares (`calibrate_*_threshold`) — os
  mesmos critérios de `evaluate_risk_engine`, só que executados etapa a
  etapa para que cada uma possa ser exibida e cronometrada;
- `combine_scores` e `DecisionBands`, a ÚNICA implementação da combinação,
  do nível e da escala exibida.

A equivalência com `evaluate_risk_engine` e com `RiskEngine.assess_frame`
é garantida por testes (tests/test_app_risk_analysis.py).
"""

from __future__ import annotations

import enum
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from src.risk.behavior_score import behavior_score
from src.risk.dataset import build_risk_table
from src.risk.engine import (
    ACTION_APPROVE,
    ACTION_BLOCK,
    ACTION_BY_LEVEL,
    ACTION_REVIEW,
    LEVEL_HIGH,
    LEVEL_LOW,
    LEVEL_MEDIUM,
    DecisionBands,
    EngineConfig,
    RiskEngine,
    RiskResult,
    combine_scores,
)
from src.risk.evaluation import (
    DEFAULT_SEED,
    calibrate_block_threshold,
    calibrate_review_threshold,
    classification_metrics,
    recall_by_scenario,
    split_by_customer,
)
from src.risk.ml_score import MLScorer
from src.risk.report import (
    TARGET_BLOCK_PRECISION,
    TARGET_RECALL,
    choose_rules_weight,
    weight_sweep,
)
from src.synthetic.models import Customer, Device, Merchant, Transaction

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "synthetic" / "sentinel.db"
DB_ENV_VAR = "SENTINEL_DB_PATH"

LEVELS = [LEVEL_LOW, LEVEL_MEDIUM, LEVEL_HIGH]
ACTIONS = [ACTION_APPROVE, ACTION_REVIEW, ACTION_BLOCK]

SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST = "treino", "validação", "teste"

# Etapas na ordem em que acontecem — o rótulo é o que a interface exibe.
STEP_LOAD = "Carregando transações do banco"
STEP_FEATURES = "Calculando features comportamentais"
STEP_ML = "Aplicando o modelo de ML"
STEP_RULES = "Aplicando as regras"
STEP_COMBINE = "Combinando no Risk Score"
STEP_CLASSIFY = "Classificando por nível"
STEP_LABELS = [STEP_LOAD, STEP_FEATURES, STEP_ML, STEP_RULES, STEP_COMBINE, STEP_CLASSIFY]

# Chamado com (rótulo, None) ao começar a etapa e (rótulo, segundos) ao terminar.
ProgressCallback = Callable[[str, "float | None"], None]

# Colunas da tabela de risco que o painel de investigação também exibe.
_FEATURE_COLUMNS_FOR_PANEL = [
    "transaction_id",
    "new_device",
    "new_city",
    "device_untrusted",
    "is_impossible_travel",
    "implied_speed_kmh",
    "distance_from_home_km",
    "amount_deviation",
]


def resolve_db_path() -> Path:
    """Caminho do banco: `SENTINEL_DB_PATH` (usado nos testes) ou o padrão."""
    override = os.environ.get(DB_ENV_VAR)
    return Path(override) if override else DEFAULT_DB_PATH


@dataclass(frozen=True)
class Step:
    label: str
    seconds: float


@dataclass
class AnalysisResult:
    """Tudo que a página precisa depois de uma análise (guardado no
    `st.session_state`, para filtros e seleção não reexecutarem nada).
    """

    frame: pd.DataFrame  # uma linha por transação: dados + scores + decisão
    features: pd.DataFrame  # tabela de risco (entrada do motor), indexada por transaction_id
    engine: RiskEngine
    rules_weight: float
    ml_weight: float
    review_threshold: int  # faixas INTERNAS calibradas na validação
    block_threshold: int
    customers_by_split: dict[str, int]
    steps: list[Step] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return sum(s.seconds for s in self.steps)


class DatabaseNotFound(FileNotFoundError):
    """O banco sintético ainda não foi gerado."""


# ----------------------------------------------------------------------
# a análise
# ----------------------------------------------------------------------


def _load_transactions(session: Session) -> pd.DataFrame:
    """Dados descritivos de cada transação (o que a tabela e o painel exibem)."""
    statement = (
        select(
            Transaction.id.label("transaction_id"),
            Transaction.timestamp,
            Transaction.amount,
            Transaction.lat.label("tx_lat"),
            Transaction.lon.label("tx_lon"),
            Customer.id.label("customer_id"),
            Customer.name.label("customer"),
            Customer.behavior_profile.label("profile"),
            Customer.home_city,
            Customer.home_state,
            Customer.home_lat,
            Customer.home_lon,
            Customer.avg_amount_baseline,
            Customer.typical_hour_start,
            Customer.typical_hour_end,
            Merchant.name.label("merchant"),
            Merchant.city.label("city"),
            Device.device_type,
        )
        .join(Customer, Customer.id == Transaction.customer_id)
        .join(Merchant, Merchant.id == Transaction.merchant_id)
        .join(Device, Device.id == Transaction.device_id)
    )
    result = session.execute(statement)
    columns = list(result.keys())
    # Enum -> texto aqui, linha a linha: não depende de como o pandas infere
    # o tipo de uma coluna de enums (str-Enum).
    records = [
        {c: (v.value if isinstance(v, enum.Enum) else v) for c, v in zip(columns, row)}
        for row in result.all()
    ]
    if not records:
        raise ValueError("o banco não tem transações")
    return pd.DataFrame(records, columns=columns)


class _Timed:
    """Cronometra uma etapa e avisa o início e o fim ao callback."""

    def __init__(self, label: str, steps: list[Step], progress: ProgressCallback | None):
        self.label, self.steps, self.progress = label, steps, progress

    def __enter__(self):
        if self.progress:
            self.progress(self.label, None)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            seconds = time.perf_counter() - self._t0
            self.steps.append(Step(self.label, seconds))
            if self.progress:
                self.progress(self.label, seconds)
        return False


def run_analysis(
    db_path: str | Path | None = None,
    progress: ProgressCallback | None = None,
    seed: int = DEFAULT_SEED,
) -> AnalysisResult:
    """Executa a análise de risco completa sobre o banco, ao vivo.

    O ML é treinado só nos clientes de TREINO; peso e faixas são escolhidos
    nos clientes de VALIDAÇÃO; os clientes de TESTE servem de base para a
    medição de eficiência (`efficiency`). Todas as transações recebem Risk
    Score — a coluna `conjunto` diz a qual grupo cada cliente pertence.
    """
    db_path = Path(db_path) if db_path else resolve_db_path()
    if not db_path.exists():
        raise DatabaseNotFound(str(db_path))

    steps: list[Step] = []
    sa_engine = create_engine(f"sqlite:///{db_path}")
    try:
        with _Timed(STEP_LOAD, steps, progress):
            with Session(sa_engine) as session:
                info = _load_transactions(session)

        with _Timed(STEP_FEATURES, steps, progress):
            with Session(sa_engine) as session:
                table = build_risk_table(session)
    finally:
        sa_engine.dispose()

    with _Timed(STEP_ML, steps, progress):
        split = split_by_customer(table, seed=seed)
        ml_model = MLScorer(random_state=seed).fit(split.train)
        ml = ml_model.score(table)

    with _Timed(STEP_RULES, steps, progress):
        behavior = np.array([behavior_score(row).score for row in table.to_dict("records")])

    with _Timed(STEP_COMBINE, steps, progress):
        impossible = table["is_impossible_travel"].to_numpy(dtype=bool)
        in_val = table["customer_id"].isin(split.validation["customer_id"].unique()).to_numpy()
        y_val = table.loc[in_val, "is_fraud"].astype(int).to_numpy()

        # peso: a regra declarada, na validação (src/risk/report.py)
        rules_weight = choose_rules_weight(weight_sweep(y_val, ml[in_val], behavior[in_val]))
        config = replace(EngineConfig(), ml_weight=round(1 - rules_weight, 10), behavior_weight=rules_weight)

        # faixas: calibradas na validação, sem o piso (que sinaliza sempre a
        # viagem impossível, então ela conta como sinalizada na calibração)
        no_floor = replace(config, impossible_travel_floor=None)
        combined_val = combine_scores(ml[in_val], behavior[in_val], impossible[in_val], no_floor)
        forced = impossible[in_val] if config.impossible_travel_floor else None
        review = calibrate_review_threshold(y_val, combined_val, TARGET_RECALL, forced=forced)
        block = calibrate_block_threshold(y_val, combined_val, review, TARGET_BLOCK_PRECISION)
        engine = RiskEngine(ml_model, config).with_bands(DecisionBands(review, block))

        raw = combine_scores(ml, behavior, impossible, engine.config)
        display = np.array([engine.config.bands.to_display(int(s)) for s in raw])

    with _Timed(STEP_CLASSIFY, steps, progress):
        levels = [engine.config.bands.level(int(s)) for s in raw]
        scores = pd.DataFrame(
            {
                "transaction_id": table["transaction_id"].to_numpy(),
                "ml_score": np.round(ml, 1),
                "behavior_score": np.round(behavior, 1),
                "raw_score": raw,
                "risk_score": display,
                "level": levels,
                "action": [ACTION_BY_LEVEL[lv] for lv in levels],
                "is_fraud": table["is_fraud"].astype(bool).to_numpy(),
                "fraud_scenario": table["fraud_scenario"].to_numpy(),
            }
        )
        split_of = {
            **{c: SPLIT_TRAIN for c in split.train["customer_id"].unique()},
            **{c: SPLIT_VALIDATION for c in split.validation["customer_id"].unique()},
            **{c: SPLIT_TEST for c in split.test["customer_id"].unique()},
        }
        frame = info.merge(scores, on="transaction_id", how="inner", validate="one_to_one")
        frame["conjunto"] = frame["customer_id"].map(split_of)
        frame = frame.merge(table[_FEATURE_COLUMNS_FOR_PANEL], on="transaction_id", validate="one_to_one")
        frame = frame.sort_values(
            ["raw_score", "behavior_score", "transaction_id"], ascending=[False, False, True]
        ).reset_index(drop=True)

    return AnalysisResult(
        frame=frame,
        features=table.set_index("transaction_id", drop=False),
        engine=engine,
        rules_weight=rules_weight,
        ml_weight=engine.config.ml_weight,
        review_threshold=review,
        block_threshold=block,
        customers_by_split={
            SPLIT_TRAIN: int(split.train["customer_id"].nunique()),
            SPLIT_VALIDATION: int(split.validation["customer_id"].nunique()),
            SPLIT_TEST: int(split.test["customer_id"].nunique()),
        },
        steps=steps,
    )


# ----------------------------------------------------------------------
# resumo, filtros, investigação, eficiência
# ----------------------------------------------------------------------


def summarize(frame: pd.DataFrame) -> dict:
    """Total e distribuição (quantidade e %) por nível e por ação."""
    total = len(frame)

    def dist(column: str, order: list[str]) -> pd.DataFrame:
        counts = frame[column].value_counts().reindex(order, fill_value=0)
        return pd.DataFrame({"quantidade": counts.astype(int), "percentual": counts / total if total else 0.0})

    return {"total": total, "por_nivel": dist("level", LEVELS), "por_acao": dist("action", ACTIONS)}


def apply_filters(
    frame: pd.DataFrame,
    levels=None,
    actions=None,
    customers=None,
    profiles=None,
    amount_range: tuple[float, float] | None = None,
    splits=None,
) -> pd.DataFrame:
    """Filtros da tabela. `None` (ou lista vazia) em um critério = sem filtro."""
    mask = pd.Series(True, index=frame.index)
    for column, values in (
        ("level", levels),
        ("action", actions),
        ("customer_id", customers),
        ("profile", profiles),
        ("conjunto", splits),
    ):
        if values:
            mask &= frame[column].isin(list(values))
    if amount_range is not None:
        low, high = amount_range
        mask &= frame["amount"].between(low, high)
    return frame[mask]


@dataclass(frozen=True)
class Investigation:
    """Painel de uma transação: fatos do banco + o que o motor decidiu."""

    row: pd.Series
    assessment: RiskResult


def investigate(result: AnalysisResult, transaction_id: int) -> Investigation:
    """Reavalia UMA transação com o motor (para obter os motivos legíveis).
    O score bate com o da tabela porque a combinação é a mesma (testado).
    """
    match = result.frame[result.frame["transaction_id"] == transaction_id]
    if match.empty:
        raise KeyError(f"transação {transaction_id} não está na análise")
    features = result.features.loc[transaction_id].to_dict()
    return Investigation(row=match.iloc[0], assessment=result.engine.assess(features))


def efficiency(result: AnalysisResult) -> dict:
    """Decisão do sistema vs rótulo da simulação, SÓ nos clientes de teste.

    "Sinalizada" = nível médio ou alto (revisar ou bloquear). Nunca usa
    treino nem validação: o modelo foi ajustado em uma e as faixas na outra.
    """
    test = result.frame[result.frame["conjunto"] == SPLIT_TEST]
    flagged = (test["level"] != LEVEL_LOW).to_numpy()
    metrics = classification_metrics(test["is_fraud"], test["raw_score"], flagged)

    by_action = (
        test.assign(real=np.where(test["is_fraud"], "fraude", "legítima"))
        .groupby(["action", "real"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=ACTIONS, columns=["legítima", "fraude"], fill_value=0)
    )
    return {
        "customers": int(test["customer_id"].nunique()),
        "transactions": len(test),
        "frauds": int(test["is_fraud"].sum()),
        "legit": int((~test["is_fraud"]).sum()),
        "metrics": metrics,
        "by_action": by_action,
        "by_scenario": recall_by_scenario(test, flagged),
    }
