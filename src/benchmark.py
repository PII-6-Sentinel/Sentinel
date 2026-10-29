"""Avaliação mais próxima da operação real, sobre o benchmark público (Kaggle).

O split padrão do projeto (`get_train_test_split`) é ALEATÓRIO e
estratificado: bom para comparar modelos, mas otimista, porque o modelo
treina com transações do mesmo período em que é testado. Numa operação de
verdade o modelo é treinado com o passado e decide sobre o futuro. Este
módulo acrescenta três análises, sem alterar as existentes:

1. **Avaliação temporal** (`temporal_split`, `compare_splits`): treino com
   as transações mais antigas e teste com as mais recentes (coluna `Time`),
   comparada lado a lado com o split aleatório.
2. **Volume de alertas** (`alert_volume`): para cada orçamento de alertas
   (a fração de transações que a equipe consegue revisar), quantos alertas
   por dia o modelo gera, quantas fraudes eles pegam e quanto dinheiro de
   fraude cobrem.
3. **Limiar por custo** (`cost_curve`, `cost_optimal_threshold`): o limiar
   que minimiza o custo esperado = revisões × custo de revisão + valor das
   fraudes não detectadas.

Reusa `train_all_models`, `fraud_score` e `prepare_features` como estão.
`Time` só ORDENA o split temporal: o modelo recebe a hora do dia, não o
`Time` bruto (ver `time_of_day_features`).
Como o resto da avaliação no Kaggle, nenhum limiar é escolhido olhando o
teste: o limiar por custo é calculado no próprio conjunto em que é
aplicado só para DESCREVER a curva de custo; para decidir, use um conjunto
de validação (ver docs/VALIDACAO.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models import fraud_score, train_all_models
from src.preprocessing import get_train_test_split, prepare_features, split_features_target

TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"
SECONDS_PER_DAY = 24 * 3600

# Orçamentos de alerta: fração das transações que viram alerta (revisão).
DEFAULT_ALERT_RATES = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)

# Custo de revisar um alerta, na moeda do dataset (euro: cartões europeus):
# ordem de grandeza de alguns minutos de um analista. É uma premissa
# declarada, não um dado — passe o seu valor.
DEFAULT_REVIEW_COST = 5.0


def temporal_split(df: pd.DataFrame, test_size: float = 0.2):
    """(X_train, X_test, y_train, y_test) com o teste = as transações MAIS
    RECENTES (`Time`), sem embaralhar. Empates em `Time` ficam do mesmo lado
    pela ordem estável.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size precisa estar entre 0 e 1, recebido {test_size!r}")
    ordered = df.sort_values(TIME_COLUMN, kind="stable")
    cut = int(round(len(ordered) * (1 - test_size)))
    X, y = split_features_target(ordered)
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def _scores(X_train, X_test, y_train) -> dict[str, np.ndarray]:
    X_train_s, X_test_s, _ = prepare_features(X_train, X_test)
    models = train_all_models(X_train_s, y_train)
    return {name: fraud_score(model, X_test_s) for name, model in models.items()}


def compare_splits(df: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
    """PR-AUC e ROC-AUC de cada modelo no split aleatório (o do projeto) e
    no temporal, lado a lado. Uma queda no temporal indica que o resultado
    do split aleatório é otimista para uso em produção.
    """
    splits = {
        "aleatório": get_train_test_split(df, test_size=test_size),
        "temporal": temporal_split(df, test_size=test_size),
    }
    rows = []
    for split_name, (X_train, X_test, y_train, y_test) in splits.items():
        if y_test.nunique() < 2 or y_train.nunique() < 2:
            raise ValueError(f"split {split_name}: treino e teste precisam ter fraudes e legítimas")
        for model, score in _scores(X_train, X_test, y_train).items():
            rows.append(
                {
                    "modelo": model,
                    "split": split_name,
                    "pr_auc": float(average_precision_score(y_test, score)),
                    "roc_auc": float(roc_auc_score(y_test, score)),
                    "fraudes_teste": int(y_test.sum()),
                    "transacoes_teste": int(len(y_test)),
                }
            )
    return pd.DataFrame(rows)


def _period_days(times: pd.Series) -> float:
    span = float(times.max() - times.min())
    return max(span, 1.0) / SECONDS_PER_DAY


def alert_volume(
    y_true: pd.Series,
    y_score: np.ndarray,
    amount: pd.Series,
    times: pd.Series,
    alert_rates=DEFAULT_ALERT_RATES,
) -> pd.DataFrame:
    """Uma linha por orçamento de alertas: as `rate × n` transações de maior
    escore viram alerta. Mostra o volume (alertas por dia e por mil
    transações) e o que ele rende (fraudes pegas, precisão, recall e valor
    de fraude coberto).
    """
    y = np.asarray(y_true, dtype=int)
    value = np.asarray(amount, dtype=float)
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="stable")
    n, frauds, days = len(y), int(y.sum()), _period_days(times)
    fraud_value = float(value[y == 1].sum())

    rows = []
    for rate in alert_rates:
        k = max(1, int(round(rate * n)))
        top = order[:k]
        caught = int(y[top].sum())
        rows.append(
            {
                "orcamento": rate,
                "alertas": k,
                "alertas_por_dia": k / days,
                "alertas_por_mil": 1000 * k / n,
                "fraudes_pegas": caught,
                "precisao": caught / k,
                "recall": caught / frauds if frauds else float("nan"),
                "valor_fraude_coberto": float(value[top][y[top] == 1].sum()) / fraud_value if fraud_value else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def cost_curve(
    y_true: pd.Series,
    y_score: np.ndarray,
    amount: pd.Series,
    review_cost: float = DEFAULT_REVIEW_COST,
    n_points: int = 200,
) -> pd.DataFrame:
    """Custo esperado em cada limiar: cada alerta custa `review_cost` (uma
    revisão, seja fraude ou não) e cada fraude NÃO alertada custa o próprio
    valor (`Amount`), que vira prejuízo.

    Os limiares são quantis do escore observado (escala nativa de cada
    modelo, como em `threshold_sweep`), mais um acima do máximo (nenhum
    alerta).
    """
    if review_cost < 0:
        raise ValueError("review_cost não pode ser negativo")
    y = np.asarray(y_true, dtype=int)
    value = np.asarray(amount, dtype=float)
    score = np.asarray(y_score, dtype=float)
    thresholds = np.unique(np.quantile(score, np.linspace(0, 1, n_points)))
    thresholds = np.append(thresholds, np.nextafter(score.max(), np.inf))

    rows = []
    for t in thresholds:
        alert = score >= t
        missed = (y == 1) & ~alert
        alerts = int(alert.sum())
        rows.append(
            {
                "limiar": float(t),
                "alertas": alerts,
                "fp": int((alert & (y == 0)).sum()),
                "fn": int(missed.sum()),
                "custo_revisao": alerts * review_cost,
                "custo_fraude_perdida": float(value[missed].sum()),
            }
        )
    curve = pd.DataFrame(rows)
    curve["custo_total"] = curve["custo_revisao"] + curve["custo_fraude_perdida"]
    return curve


def cost_optimal_threshold(curve: pd.DataFrame) -> pd.Series:
    """Linha de menor custo total da `cost_curve` (empate: menos alertas)."""
    best = curve.sort_values(["custo_total", "alertas"], kind="stable")
    return best.iloc[0]


def model_scores_temporal(df: pd.DataFrame, test_size: float = 0.2):
    """Escores de cada modelo no teste TEMPORAL, com o que as análises de
    volume e custo precisam: (y_test, escores por modelo, Amount e Time
    originais do teste).
    """
    X_train, X_test, y_train, y_test = temporal_split(df, test_size=test_size)
    return y_test, _scores(X_train, X_test, y_train), X_test[AMOUNT_COLUMN], X_test[TIME_COLUMN]

