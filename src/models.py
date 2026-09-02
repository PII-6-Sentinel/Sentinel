"""Treinamento e avaliação das abordagens comparadas pelo projeto.

Centralizado aqui (em vez de dentro do app Streamlit) porque a mesma lógica
serve tanto a interface quanto notebooks futuros de modelagem — o app é só
uma camada de visualização sobre este módulo.

Três abordagens, cada uma tratando o desbalanceamento à sua maneira:
    - Baseline estatístico: regra de distância no espaço PCA, sem
      "treinar" nada no sentido de ML — só um limiar calibrado por percentil.
    - Regressão Logística: `class_weight="balanced"` penaliza mais os erros
      na classe minoritária (fraude), sem precisar reamostrar os dados.
    - Isolation Forest: não-supervisionado, usa `contamination` (proporção
      esperada de anomalias) para calibrar o limiar de decisão.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

V_COLUMNS = [f"V{i}" for i in range(1, 29)]


class StatisticalBaseline:
    """Regra estatística simples: distância euclidiana ao quadrado no
    espaço das componentes V1..V28, usada como escore de anomalia.

    V1..V28 são componentes de um PCA ajustado sobre o dataset (majoritariamente
    transações legítimas), com média ~0. Transações "típicas" tendem a ficar
    perto da origem nesse espaço; um escore alto indica desvio do padrão
    normal — um proxy simples e interpretável de anomalia, análogo a uma
    distância de Mahalanobis simplificada, sem depender de rótulos.

    O limiar é calibrado por percentil: escolhemos o corte que classifica
    como anomalia a mesma proporção de casos esperada de fraude
    (`contamination`), estimada a partir do treino.
    """

    def __init__(self, contamination: float):
        self.contamination = contamination
        self.threshold_: float | None = None

    def _score(self, X: pd.DataFrame) -> np.ndarray:
        return (X[V_COLUMNS] ** 2).sum(axis=1).to_numpy()

    def fit(self, X: pd.DataFrame, y=None) -> "StatisticalBaseline":
        scores = self._score(X)
        self.threshold_ = float(np.quantile(scores, 1 - self.contamination))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self._score(X) >= self.threshold_).astype(int)

    def decision_score(self, X: pd.DataFrame) -> np.ndarray:
        return self._score(X)


@dataclass
class ModelResult:
    """Resultado de avaliação de um modelo sobre o conjunto de teste."""

    name: str
    y_pred: np.ndarray
    y_score: np.ndarray
    confusion_matrix: np.ndarray
    precision: float
    recall: float
    f1: float
    auc_roc: float
    fpr: np.ndarray = field(repr=False)
    tpr: np.ndarray = field(repr=False)


def train_all_models(
    X_train: pd.DataFrame, y_train: pd.Series, contamination: float
) -> dict[str, object]:
    """Treina as três abordagens comparadas pelo projeto.

    `contamination` é a proporção de fraudes esperada (estimada do próprio
    treino), usada para calibrar o baseline estatístico e o Isolation Forest.
    """
    baseline = StatisticalBaseline(contamination=contamination).fit(X_train)

    logreg = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    ).fit(X_train, y_train)

    iso_forest = IsolationForest(
        contamination=contamination, n_estimators=200, random_state=42, n_jobs=-1
    ).fit(X_train)

    return {
        "Baseline Estatístico": baseline,
        "Regressão Logística": logreg,
        "Isolation Forest": iso_forest,
    }


def predict_labels(model, X: pd.DataFrame) -> np.ndarray:
    """Prediz 0 (legítima) / 1 (fraude), normalizando a convenção de cada modelo."""
    if isinstance(model, StatisticalBaseline):
        return model.predict(X)
    if isinstance(model, IsolationForest):
        # IsolationForest usa a convenção -1 (anomalia) / 1 (normal) — inverso
        # da convenção 0/1 usada pelo resto do projeto.
        raw = model.predict(X)
        return np.where(raw == -1, 1, 0)
    return model.predict(X)


def fraud_score(model, X: pd.DataFrame) -> np.ndarray:
    """Escore contínuo — quanto maior, mais 'parecido com fraude'.

    Usado para AUC-ROC/curva ROC e para o gauge de risco na demo, com a
    mesma direção (maior = mais suspeito) independente do modelo.
    """
    if isinstance(model, StatisticalBaseline):
        return model.decision_score(X)
    if isinstance(model, IsolationForest):
        # score_samples: quanto menor, mais anômalo. Invertido para manter
        # "maior = mais suspeito" consistente entre os três modelos.
        return -model.score_samples(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raise TypeError(f"Não sei extrair um escore de {type(model)}")


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> ModelResult:
    """Calcula matriz de confusão + métricas adequadas a classes desbalanceadas.

    Acurácia é deliberadamente omitida: com ~0,17% de fraudes, é uma métrica
    enganosa (ver docs/SWOT_GUT.md).
    """
    y_pred = predict_labels(model, X_test)
    y_score = fraud_score(model, X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)

    return ModelResult(
        name=type(model).__name__,
        y_pred=y_pred,
        y_score=y_score,
        confusion_matrix=confusion_matrix(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        auc_roc=roc_auc_score(y_test, y_score),
        fpr=fpr,
        tpr=tpr,
    )


def percentile_rank(value: float, distribution: np.ndarray) -> float:
    """Posição percentual (0-100) de `value` dentro de `distribution`.

    Usado para exibir o escore de um modelo em uma escala 0-100 comum,
    já que Regressão Logística (probabilidade), Isolation Forest (escore
    de isolamento) e o baseline (distância) não são naturalmente
    comparáveis em escala.
    """
    return float((distribution < value).mean() * 100)
