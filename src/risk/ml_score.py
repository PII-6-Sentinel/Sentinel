"""Score de Machine Learning — Regressão Logística treinada SÓ nos dados
sintéticos.

Por que Regressão Logística: é o modelo supervisionado mais simples que
ainda aprende combinações de sinais, tem coeficientes interpretáveis (dá
para mostrar o que o modelo aprendeu) e já foi validado com rigor no
benchmark do Kaggle. `class_weight='balanced'` compensa os ~3% de fraude.

O score é a probabilidade estimada × 100. Atenção: com `class_weight`
balanceado, essa "probabilidade" fica inflada para a classe minoritária —
ela ordena bem as transações por risco, mas NÃO é uma probabilidade
calibrada de fraude e não deve ser apresentada como tal.

LIMITAÇÃO METODOLÓGICA (registrada aqui de propósito)
-----------------------------------------------------
Os rótulos de fraude do domínio sintético foram GERADOS a partir de
cenários que manipulam justamente estas features (valor, local, horário,
dispositivo). O modelo pode "acertar" por circularidade: está aprendendo a
regra de geração, não a fraude do mundo real. A avaliação deste modelo
valida a LÓGICA do sistema sobre dados simulados; não prova desempenho em
fraude real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_SEED = 42

# Features de entrada do modelo, já transformadas (ver `ml_feature_matrix`).
ML_FEATURES = [
    "amount_deviation",
    "location_distance",
    "unusual_hour",
    "new_device",
    "new_city",
    "device_untrusted",
    "log_speed_kmh",
    "is_impossible_travel",
    "is_online",
    "is_contactless",
    "channel_rarity",
]


def ml_feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Converte a tabela de risco nas colunas que o modelo consome.

    A velocidade implícita vai para escala log (`log1p`): ela vai de ~0 a
    milhares de km/h, e em escala linear poucas transações extremas
    dominariam o ajuste. Booleanos viram 0/1. A distância entra só como
    `location_distance` (já em escala log 0-1); a distância bruta em km é
    usada pelas regras, para os motivos legíveis.
    """
    return pd.DataFrame(
        {
            "amount_deviation": features["amount_deviation"].astype(float),
            "location_distance": features["location_distance"].astype(float),
            "unusual_hour": features["unusual_hour"].astype(float),
            "new_device": features["new_device"].astype(float),
            "new_city": features["new_city"].astype(float),
            "device_untrusted": features["device_untrusted"].astype(float),
            "log_speed_kmh": np.log1p(features["implied_speed_kmh"].astype(float).clip(lower=0)),
            "is_impossible_travel": features["is_impossible_travel"].astype(float),
            "is_online": features["is_online"].astype(float),
            "is_contactless": features["is_contactless"].astype(float),
            "channel_rarity": features["channel_rarity"].astype(float),
        },
        columns=ML_FEATURES,
    )


class MLScorer:
    """Regressão Logística (com padronização) sobre as features comportamentais."""

    def __init__(self, random_state: int = DEFAULT_SEED, C: float = 1.0):
        self.random_state = random_state
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced", C=C, max_iter=1000, random_state=random_state
                    ),
                ),
            ]
        )
        self.is_fitted = False

    def fit(self, train: pd.DataFrame) -> "MLScorer":
        """Treina em `train` (tabela de risco com coluna `is_fraud`).

        Só deve receber o conjunto de TREINO — nunca validação nem teste.
        O rótulo (`is_fraud`) é só o alvo `y`: a matriz de entrada vem de
        `ml_feature_matrix`, que lê apenas `ML_FEATURES` — nem `is_fraud`
        nem `fraud_scenario` entram (testado em tests/test_risk_label_leakage.py).
        """
        y = train["is_fraud"].astype(int)
        if y.nunique() < 2:
            raise ValueError("o conjunto de treino precisa ter fraudes e não-fraudes")
        self.pipeline.fit(ml_feature_matrix(train), y)
        self.is_fitted = True
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Score 0-100 (probabilidade estimada × 100) para cada linha."""
        if not self.is_fitted:
            raise RuntimeError("MLScorer.score chamado antes de fit")
        return self.pipeline.predict_proba(ml_feature_matrix(features))[:, 1] * 100.0

    def coefficients(self) -> dict[str, float]:
        """Coeficientes sobre as features PADRONIZADAS — comparáveis entre si:
        quanto maior em módulo, mais a feature move o score.
        """
        coefs = self.pipeline.named_steps["model"].coef_[0]
        return dict(zip(ML_FEATURES, (float(c) for c in coefs)))
