"""Avaliação comparativa do Risk Engine: só regras vs só ML vs combinado.

Fluxo (sem vazamento):
1. split por cliente: treino / validação / teste (`evaluation.split_by_customer`);
2. ML treinado só no TREINO;
3. na VALIDAÇÃO: varredura de pesos ML/regras (evidência para 50/50) e
   calibração dos limiares — cada abordagem com o mesmo critério;
4. no TESTE, uma vez: métricas, recall por cenário, distribuição por nível;
5. estabilidade: validação cruzada 5-fold por cliente sobre a tabela toda,
   porque o teste tem poucas fraudes por cenário.

Tudo isso valida a lógica do sistema sobre dados SIMULADOS. Os rótulos
foram gerados a partir das mesmas features (circularidade); nada aqui
prova desempenho em fraude real.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.risk.behavior_score import behavior_score
from src.risk.engine import (
    LEVEL_HIGH,
    LEVEL_LOW,
    LEVEL_MEDIUM,
    DecisionBands,
    EngineConfig,
    RiskEngine,
    combine_scores,
)
from src.risk.evaluation import (
    CustomerSplit,
    calibrate_block_threshold,
    calibrate_review_threshold,
    classification_metrics,
    recall_by_scenario,
    split_by_customer,
)
from src.risk.ml_score import MLScorer

DEFAULT_SEED = 42
TARGET_RECALL = 0.90  # meta da faixa de revisão: no máximo 10% das fraudes aprovadas
TARGET_BLOCK_PRECISION = 0.90  # meta do bloqueio: ao menos 90% do bloqueado é fraude
WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

APPROACH_RULES = "Só regras"
APPROACH_ML = "Só ML"
APPROACH_COMBINED = "Combinado (motor)"
APPROACH_COMBINED_DEFAULT = "Combinado, faixas padrão 30/70"


@dataclass
class EngineEvaluation:
    split: CustomerSplit
    engine: RiskEngine  # ML treinado no treino + faixas calibradas na validação
    weight_sweep: pd.DataFrame  # validação
    review_thresholds: dict[str, int]  # por abordagem, calibrados na validação
    comparison: pd.DataFrame  # teste
    scenario_recall: pd.DataFrame  # teste
    level_distribution: pd.DataFrame  # teste, faixas calibradas
    test_scores: pd.DataFrame  # teste: tabela de risco + scores e decisão
    stability: pd.DataFrame  # PR-AUC em 5-fold por cliente
    coefficients: dict[str, float]


def _behavior_scores(table: pd.DataFrame) -> np.ndarray:
    return np.array([behavior_score(row).score for row in table.to_dict("records")])


def _weight_sweep(y, ml, behavior) -> pd.DataFrame:
    rows = []
    for w in WEIGHT_GRID:
        combined = w * ml + (1 - w) * behavior
        rows.append(
            {
                "peso_ml": w,
                "peso_regras": 1 - w,
                "pr_auc": average_precision_score(y, combined),
                "roc_auc": roc_auc_score(y, combined),
            }
        )
    return pd.DataFrame(rows)


def _stability(table: pd.DataFrame, config: EngineConfig, seed: int, n_splits: int = 5) -> pd.DataFrame:
    """PR-AUC de cada abordagem em validação cruzada estratificada POR
    CLIENTE: cada fold usa clientes que o modelo nunca viu.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_all = table["is_fraud"].astype(int).to_numpy()
    behavior_all = _behavior_scores(table)
    scores = {APPROACH_RULES: [], APPROACH_ML: [], APPROACH_COMBINED: []}

    for train_idx, test_idx in cv.split(table, y_all, groups=table["customer_id"]):
        ml = MLScorer(random_state=seed).fit(table.iloc[train_idx]).score(table.iloc[test_idx])
        y = y_all[test_idx]
        beh = behavior_all[test_idx]
        combined = combine_scores(ml, beh, table["is_impossible_travel"].to_numpy()[test_idx], config)
        scores[APPROACH_RULES].append(average_precision_score(y, beh))
        scores[APPROACH_ML].append(average_precision_score(y, ml))
        scores[APPROACH_COMBINED].append(average_precision_score(y, combined))

    return pd.DataFrame(
        {
            "abordagem": list(scores),
            "pr_auc_media": [float(np.mean(v)) for v in scores.values()],
            "pr_auc_desvio": [float(np.std(v, ddof=1)) for v in scores.values()],
            "folds": [len(v) for v in scores.values()],
        }
    )


def evaluate_risk_engine(
    table: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    config: EngineConfig | None = None,
) -> EngineEvaluation:
    config = config or EngineConfig()
    split = split_by_customer(table, seed=seed)
    ml_model = MLScorer(random_state=seed).fit(split.train)

    # ---------------- validação: pesos e limiares ----------------
    val = split.validation
    y_val = val["is_fraud"].astype(int).to_numpy()
    ml_val = ml_model.score(val)
    beh_val = _behavior_scores(val)
    imp_val = val["is_impossible_travel"].to_numpy(dtype=bool)

    sweep = _weight_sweep(y_val, ml_val, beh_val)

    no_floor = EngineConfig(config.ml_weight, config.behavior_weight, config.bands, None, config.ml_reason_min_score)
    combined_val = combine_scores(ml_val, beh_val, imp_val, no_floor)

    review = {
        APPROACH_RULES: calibrate_review_threshold(y_val, beh_val, TARGET_RECALL),
        APPROACH_ML: calibrate_review_threshold(y_val, ml_val, TARGET_RECALL),
        # o motor sinaliza viagem impossível sempre (piso), então ela conta
        # como sinalizada em qualquer limiar durante a calibração
        APPROACH_COMBINED: calibrate_review_threshold(
            y_val, combined_val, TARGET_RECALL, forced=imp_val if config.impossible_travel_floor else None
        ),
    }
    block = calibrate_block_threshold(
        y_val, combined_val, review[APPROACH_COMBINED], TARGET_BLOCK_PRECISION
    )
    engine = RiskEngine(ml_model, config).with_bands(DecisionBands(review[APPROACH_COMBINED], block))

    # ---------------- teste: uma vez ----------------
    test = split.test
    y_test = test["is_fraud"].astype(int).to_numpy()
    out = engine.assess_frame(test)
    beh_test = out["behavior_score"].to_numpy()
    ml_test = out["ml_score"].to_numpy()
    default_engine = engine.with_bands(DecisionBands())
    out_default = default_engine.assess_frame(test)

    flagged = {
        APPROACH_RULES: beh_test > review[APPROACH_RULES],
        APPROACH_ML: ml_test > review[APPROACH_ML],
        APPROACH_COMBINED: (out["level"] != LEVEL_LOW).to_numpy(),
        APPROACH_COMBINED_DEFAULT: (out_default["level"] != LEVEL_LOW).to_numpy(),
    }
    ranking = {
        APPROACH_RULES: beh_test,
        APPROACH_ML: ml_test,
        # score interno: a escala exibida comprime trechos e criaria empates
        APPROACH_COMBINED: out["raw_score"].to_numpy(),
        APPROACH_COMBINED_DEFAULT: out_default["raw_score"].to_numpy(),
    }
    comparison = pd.DataFrame(
        [{"abordagem": a, **classification_metrics(y_test, ranking[a], flagged[a])} for a in flagged]
    )

    scenario = pd.concat(
        {a: recall_by_scenario(test, flagged[a]) for a in (APPROACH_RULES, APPROACH_ML, APPROACH_COMBINED)},
        axis=1,
    )

    test_scores = pd.concat([test.reset_index(drop=True), out.drop(columns="transaction_id")], axis=1)
    level_distribution = (
        test_scores.groupby("level")
        .agg(transacoes=("transaction_id", "size"), fraudes=("is_fraud", "sum"))
        .reindex([LEVEL_LOW, LEVEL_MEDIUM, LEVEL_HIGH], fill_value=0)
    )
    level_distribution["fraudes"] = level_distribution["fraudes"].astype(int)
    level_distribution["pct_transacoes"] = level_distribution["transacoes"] / len(test_scores)
    level_distribution["taxa_fraude"] = (
        level_distribution["fraudes"] / level_distribution["transacoes"].replace(0, np.nan)
    ).fillna(0.0)
    level_distribution["pct_das_fraudes"] = level_distribution["fraudes"] / max(1, int(y_test.sum()))

    return EngineEvaluation(
        split=split,
        engine=engine,
        weight_sweep=sweep,
        review_thresholds=review,
        comparison=comparison,
        scenario_recall=scenario,
        level_distribution=level_distribution,
        test_scores=test_scores,
        stability=_stability(table, config, seed),
        coefficients=ml_model.coefficients(),
    )


def pick_examples(test_scores: pd.DataFrame) -> dict[str, pd.Series]:
    """Um exemplo por nível, escolhido por critério fixo (não à mão):

    - baixo: a transação aprovada com MAIS sinais de regra (mostra um caso
      com algo incomum que ainda assim foi aprovado);
    - médio: a mais próxima do meio da faixa de revisão (score interno);
    - alto: a de maior Risk Score (desempate: maior score por regras).
    """
    examples: dict[str, pd.Series] = {}
    low = test_scores[test_scores["level"] == LEVEL_LOW]
    if len(low):
        examples[LEVEL_LOW] = low.sort_values(
            ["behavior_score", "transaction_id"], ascending=[False, True]
        ).iloc[0]
    med = test_scores[test_scores["level"] == LEVEL_MEDIUM]
    if len(med):
        mid = med["raw_score"].median()
        examples[LEVEL_MEDIUM] = med.assign(_d=(med["raw_score"] - mid).abs()).sort_values(
            ["_d", "transaction_id"]
        ).iloc[0]
    high = test_scores[test_scores["level"] == LEVEL_HIGH]
    if len(high):
        examples[LEVEL_HIGH] = high.sort_values(
            ["raw_score", "behavior_score", "transaction_id"], ascending=[False, False, True]
        ).iloc[0]
    return examples
