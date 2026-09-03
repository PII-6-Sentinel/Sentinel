"""Validação cruzada estratificada — avalia a ESTABILIDADE dos modelos,
sem contaminar o conjunto de teste final.

Este módulo é deliberadamente separado de `src/models.py`:

    - `models.py` treina UM modelo e avalia UMA vez, montando um `ModelResult`
      completo (com curvas ROC/PR inteiras) — pensado para o resultado final
      exibido no conjunto de teste.
    - `evaluation.py` (aqui) repete treino+avaliação 5 vezes (uma por fold),
      só com métricas escalares por fold — guardar 5 curvas completas por
      modelo não teria uso na tabela de "mean ± std", só custo de memória.

Nada aqui duplica lógica de `models.py`: reusa `train_all_models`,
`fraud_score`, `calibrate_threshold_by_contamination` e `evaluate_at_threshold`
tal como estão, e reusa `scale_amount_time` de `preprocessing.py` — a
mudança é só a ORDEM em que essas peças são chamadas (uma vez por fold, em
vez de uma vez no treino inteiro).

Regra de ouro que este módulo garante: o conjunto de TESTE final nunca é
passado a nenhuma função aqui. `cross_validate_models()` só aceita
`X_train`/`y_train` — não há parâmetro para dados de teste, então não há
como o teste vazar para dentro da validação cruzada por engano.

Sobre o threshold usado em cada fold (e, pelo mesmo motivo, no conjunto de
teste final): ele representa um ponto de operação calibrado pela
PREVALÊNCIA de fraude observada no conjunto de TREINO (ou, aqui, do
fold de treino) — não uma probabilidade de fraude, e não uma informação
disponível com essa precisão em um cenário de produção real. Ver a
docstring de `calibrate_threshold_by_contamination` em `models.py` para os
detalhes completos dessa ressalva.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.models import (
    calibrate_threshold_by_contamination,
    evaluate_at_threshold,
    fraud_score,
    train_all_models,
)
from src.preprocessing import scale_amount_time

DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 42

# Chaves de calculate_metrics() que NÃO entram na agregação mean/std:
# "fold" é metadado (índice do fold); "confusion_matrix" e "y_pred" são
# arrays (não escalares) — média/desvio-padrão elemento a elemento não
# faria sentido para a matriz, e "y_pred" sequer tem o mesmo tamanho em
# folds diferentes (cada fold de validação tem um número distinto de
# linhas), então tentar agregá-lo quebraria aggregate_cv_results().
_METRIC_KEYS_EXCLUDED_FROM_AGGREGATION = {"fold", "confusion_matrix", "y_pred"}


@dataclass
class CVResult:
    """Resultado da validação cruzada de UM modelo.

    `metrics_by_fold` guarda os valores BRUTOS (sem arredondar) de cada um
    dos 5 folds — arredondamento é responsabilidade de quem exibe o
    resultado (ex.: a página Streamlit), nunca deste módulo.
    """

    model_name: str
    metrics_by_fold: list[dict]
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]


def calculate_metrics(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict:
    """Métricas completas para UM conjunto de avaliação (um fold, o teste
    final, ou uma simulação de threshold na página de Análise de Threshold)
    a UM threshold.

    Reusa `evaluate_at_threshold` (models.py) para precision/recall/f1, a
    matriz de confusão e y_pred — não recalcula essas quatro à mão.
    Adiciona só o que `evaluate_at_threshold` não calcula: AUC-ROC, AUC-PR
    (independentes de threshold, usam o ranking completo do escore) e as
    métricas extras pedidas para a validação cruzada (FPR, TPR,
    specificity, balanced accuracy, MCC), todas derivadas da mesma matriz
    de confusão já calculada.

    `confusion_matrix` e `y_pred` ficam no dicionário de saída (úteis para
    quem quer exibir a matriz, como a página de Análise de Threshold), mas
    são EXCLUÍDOS da agregação mean/std feita por `aggregate_cv_results`
    (ver `_METRIC_KEYS_EXCLUDED_FROM_AGGREGATION`) — não são métricas
    escalares.
    """
    at_threshold = evaluate_at_threshold(y_true, y_score, threshold)
    tn, fp, fn, tp = at_threshold["confusion_matrix"].ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    tpr = at_threshold["recall"]  # TPR e recall são a mesma quantidade, por definição

    return {
        "threshold": threshold,
        "precision": at_threshold["precision"],
        "recall": at_threshold["recall"],
        "f1": at_threshold["f1"],
        "auc_roc": roc_auc_score(y_true, y_score),
        "auc_pr": average_precision_score(y_true, y_score),
        "fpr": fpr,
        "tpr": tpr,
        "specificity": 1.0 - fpr,
        "balanced_accuracy": balanced_accuracy_score(y_true, at_threshold["y_pred"]),
        "mcc": matthews_corrcoef(y_true, at_threshold["y_pred"]),
        "confusion_matrix": at_threshold["confusion_matrix"],
        "y_pred": at_threshold["y_pred"],
    }


def aggregate_cv_results(model_name: str, fold_records: list[dict]) -> CVResult:
    """Agrega os registros de cada fold em mean/std por métrica.

    `ddof=1` (desvio-padrão amostral, denominador n-1): com apenas 5 folds,
    tratamos os folds como uma AMOSTRA de possíveis splits, não a população
    inteira — é a convenção usual ao reportar "mean ± std" sobre um número
    pequeno de execuções independentes.
    """
    metric_keys = [
        k for k in fold_records[0] if k not in _METRIC_KEYS_EXCLUDED_FROM_AGGREGATION
    ]
    mean_metrics = {k: float(np.mean([r[k] for r in fold_records])) for k in metric_keys}
    std_metrics = {
        k: float(np.std([r[k] for r in fold_records], ddof=1)) for k in metric_keys
    }
    return CVResult(
        model_name=model_name,
        metrics_by_fold=fold_records,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
    )


def cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, CVResult]:
    """Validação cruzada estratificada dos três modelos, SÓ no treino.

    Não recebe X_test/y_test — de propósito: não há como o conjunto de
    teste vazar para dentro desta função, porque ela nunca o enxerga.

    Para cada fold (treino do fold vs. validação do fold):
        1. ajusta um StandardScaler NOVO, só com o treino do fold
           (`scale_amount_time`, mesma função usada no split externo —
           nunca reusa o scaler ajustado no X_train completo);
        2. treina os três modelos do zero nesse treino de fold
           (`train_all_models` — sem usar a taxa de fraude, igual ao
           treino "de verdade");
        3. calibra o threshold de cada modelo usando SÓ o escore do
           treino do fold (nunca do validation do fold);
        4. aplica esse threshold ao escore do validation do fold e calcula
           as métricas ali.

    O resultado é, para cada modelo, uma lista de 5 dicionários de métricas
    (um por fold) mais a média e o desvio-padrão — uma medida de quão
    ESTÁVEL cada modelo é entre diferentes recortes do treino, não uma
    substituição da avaliação no teste final (essa continua vindo de
    `src.models.evaluate_model`, chamada uma única vez, em `app/pipeline.py`).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_records: dict[str, list[dict]] = {}

    for fold_idx, (fold_train_pos, fold_val_pos) in enumerate(
        skf.split(X_train, y_train), start=1
    ):
        X_fold_train = X_train.iloc[fold_train_pos]
        X_fold_val = X_train.iloc[fold_val_pos]
        y_fold_train = y_train.iloc[fold_train_pos]
        y_fold_val = y_train.iloc[fold_val_pos]

        # Scaler novo, ajustado só com o treino DESTE fold — nunca com
        # X_train inteiro (isso vazaria estatísticas do validation deste
        # fold, que faz parte do X_train completo, para dentro do treino).
        X_fold_train_scaled, X_fold_val_scaled, _ = scale_amount_time(
            X_fold_train, X_fold_val
        )

        models = train_all_models(X_fold_train_scaled, y_fold_train)
        fold_fraud_rate = float(y_fold_train.mean())

        for name, model in models.items():
            score_fold_train = fraud_score(model, X_fold_train_scaled)
            threshold = calibrate_threshold_by_contamination(
                score_fold_train, fold_fraud_rate
            )

            score_fold_val = fraud_score(model, X_fold_val_scaled)
            metrics = calculate_metrics(y_fold_val, score_fold_val, threshold)
            metrics["fold"] = fold_idx

            fold_records.setdefault(name, []).append(metrics)

    return {
        name: aggregate_cv_results(name, records) for name, records in fold_records.items()
    }


def threshold_sweep(
    y_true: pd.Series, y_score: np.ndarray, n_points: int = 100
) -> pd.DataFrame:
    """Varre um intervalo de thresholds sobre um escore JÁ calculado,
    devolvendo precisão/recall/F1 em cada ponto — os dados prontos para o
    gráfico "métrica × threshold" da página de Análise de Threshold.

    Não retreina nada e não recalcula o escore: só chama
    `evaluate_at_threshold` repetidamente sobre o `y_score` recebido, cada
    chamada independente da anterior (nenhum estado compartilhado entre
    pontos do sweep).

    O intervalo do sweep é o intervalo OBSERVADO de `y_score` (mínimo ao
    máximo) — nunca assume que os escores estão entre 0 e 1. Isso importa
    porque os três modelos do projeto produzem escores em escalas bem
    diferentes: `fraud_score()` (models.py) já normaliza a DIREÇÃO do
    escore (maior = mais suspeito, igual para os três), mas não a escala
    — o Isolation Forest continua em unidades de escore de isolamento, o
    baseline em unidades de distância no espaço V1..V28, e só a Regressão
    Logística fica naturalmente entre 0 e 1 (probabilidade). Esta função
    não introduz nenhuma normalização paralela: o `threshold` de cada
    linha do resultado está na mesma escala nativa de `y_score`.
    """
    score_min = float(np.min(y_score))
    score_max = float(np.max(y_score))

    thresholds = np.linspace(score_min, score_max, n_points)

    rows = []
    for t in thresholds:
        at_t = evaluate_at_threshold(y_true, y_score, float(t))
        rows.append(
            {
                "threshold": float(t),
                "precision": at_t["precision"],
                "recall": at_t["recall"],
                "f1": at_t["f1"],
            }
        )

    return pd.DataFrame(rows)
