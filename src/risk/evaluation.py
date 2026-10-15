"""Avaliação do Risk Engine: split por cliente, métricas e calibração.

Rigor metodológico
------------------
- Split por CLIENTE (`GroupShuffleSplit`), nunca por transação: transações
  do mesmo cliente compartilham perfil (cidade, média de gasto, horário,
  dispositivos). Separar por transação deixaria o modelo "conhecer" o
  cliente no treino e reconhecê-lo no teste — vazamento.
- Três conjuntos disjuntos de clientes: TREINO (ajusta o modelo de ML),
  VALIDAÇÃO (calibra os limiares de decisão) e TESTE (só para o relatório
  final, tocado uma única vez).
- Seed fixa: o mesmo split e o mesmo modelo a cada execução.

Os resultados validam a lógica do sistema sobre dados SIMULADOS — os
rótulos foram gerados a partir das mesmas features que o motor usa
(circularidade). Não provam desempenho em fraude real.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

DEFAULT_SEED = 42


@dataclass(frozen=True)
class CustomerSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_by_customer(
    table: pd.DataFrame,
    test_size: float = 0.30,
    validation_size: float = 0.20,
    seed: int = DEFAULT_SEED,
) -> CustomerSplit:
    """Divide a tabela de risco em treino/validação/teste POR CLIENTE.

    `test_size` e `validation_size` são frações dos CLIENTES (não das
    transações). Nenhum `customer_id` aparece em mais de um conjunto.
    """
    if not 0 < test_size < 1 or not 0 < validation_size < 1 or test_size + validation_size >= 1:
        raise ValueError("test_size e validation_size precisam estar em (0, 1) e somar < 1")

    groups = table["customer_id"]
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    rest_idx, test_idx = next(outer.split(table, groups=groups))
    rest, test = table.iloc[rest_idx], table.iloc[test_idx]

    # validação como fração do TOTAL de clientes, tirada do que sobrou
    inner = GroupShuffleSplit(
        n_splits=1, test_size=validation_size / (1 - test_size), random_state=seed
    )
    train_idx, val_idx = next(inner.split(rest, groups=rest["customer_id"]))

    return CustomerSplit(
        train=rest.iloc[train_idx].reset_index(drop=True),
        validation=rest.iloc[val_idx].reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def classification_metrics(y_true, score, flagged) -> dict:
    """Métricas de ranking (PR-AUC, ROC-AUC, sobre o score contínuo) e de
    decisão (precisão, recall, F1 e matriz de confusão, sobre `flagged`).

    PR-AUC vem primeiro de propósito: com ~3% de fraude, ROC-AUC tende a
    parecer otimista (a mesma lição do benchmark do Kaggle).
    """
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(flagged).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def recall_by_scenario(table: pd.DataFrame, flagged) -> pd.DataFrame:
    """Para cada cenário de fraude: quantas foram sinalizadas, de quantas.

    Inclui as contagens (não só a fração): com poucas dezenas de fraudes por
    cenário no teste, "7/8" diz mais do que "88%".
    """
    df = table.assign(_flagged=np.asarray(flagged).astype(bool))
    fraud = df[df["is_fraud"].astype(bool)]
    out = fraud.groupby("fraud_scenario")["_flagged"].agg(detected="sum", total="size")
    out["recall"] = out["detected"] / out["total"]
    out["detected"] = out["detected"].astype(int)
    return out.sort_index()


def _flag(score: np.ndarray, threshold: int, forced: np.ndarray | None) -> np.ndarray:
    flagged = score > threshold
    if forced is not None:
        flagged = flagged | forced
    return flagged


def calibrate_review_threshold(
    y_true, score, target_recall: float = 0.90, forced=None, default: int = 30
) -> int:
    """Maior limiar inteiro t (0-99) tal que sinalizar `score > t` ainda
    pega pelo menos `target_recall` das fraudes.

    Critério de negócio: nenhuma faixa deve deixar passar mais de 10% das
    fraudes sem revisão humana; entre os limiares que garantem isso, o mais
    alto é o que manda menos clientes legítimos para revisão. `forced` marca
    transações sinalizadas independentemente do score (regras duras).
    Se nenhum limiar atinge a meta, devolve `default`.
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(score, dtype=float)
    f = None if forced is None else np.asarray(forced).astype(bool)
    if y.sum() == 0:
        return default
    best = None
    for t in range(0, 100):
        recall = (_flag(s, t, f) & y).sum() / y.sum()
        if recall >= target_recall:
            best = t
        else:
            break  # recall só cai à medida que t sobe
    return default if best is None else best


def calibrate_block_threshold(
    y_true,
    score,
    review_threshold: int,
    target_precision: float = 0.90,
    min_flagged: int = 5,
    default: int = 70,
) -> int:
    """Menor limiar inteiro t > `review_threshold` tal que bloquear
    `score > t` tenha precisão ≥ `target_precision` (com pelo menos
    `min_flagged` transações acima de t, para não calibrar sobre 1 caso).

    Critério de negócio: bloquear um cliente legítimo é caro (atrito,
    reclamação); só se bloqueia automaticamente quando quase tudo o que é
    bloqueado é fraude. Entre os limiares que garantem isso, o mais baixo é
    o que bloqueia mais fraudes. Se nenhum atinge a meta, devolve `default`
    (ou `review_threshold + 1`, o que for maior).
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(score, dtype=float)
    for t in range(review_threshold + 1, 100):
        flagged = s > t
        if flagged.sum() < min_flagged:
            break
        if (flagged & y).sum() / flagged.sum() >= target_precision:
            return t
    return max(default, review_threshold + 1)
