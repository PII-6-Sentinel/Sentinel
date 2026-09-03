"""Fixtures compartilhadas entre os testes.

Os testes NUNCA usam data/raw/creditcard.csv — o dataset real não é
versionado e não deveria ser necessário para rodar a suíte (nem em CI, nem
na máquina de outra pessoa do time que ainda não baixou o CSV). Em vez
disso, geramos um dataset sintético pequeno com o mesmo formato de colunas
(Time, V1..V28, Amount, Class), com a classe minoritária deslocada em
algumas features para que os modelos consigam de fato separar as classes
— sem isso, os testes de "thresholds diferentes produzem métricas
diferentes" ficariam frágeis (dependeriam de ruído aleatório).
"""

import numpy as np
import pandas as pd
import pytest

V_COLUMNS = [f"V{i}" for i in range(1, 29)]


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_legit, n_fraud = 360, 40  # ~10% de fraude — desbalanceado, mas rápido de rodar

    legit = pd.DataFrame(
        rng.normal(loc=0.0, scale=1.0, size=(n_legit, len(V_COLUMNS))),
        columns=V_COLUMNS,
    )
    legit["Class"] = 0

    # Classe minoritária deslocada em V1/V2, para ser separável dos legítimos
    # (do contrário LogisticRegression/IsolationForest não aprenderiam nada
    # e os testes de threshold ficariam instáveis).
    fraud = pd.DataFrame(
        rng.normal(loc=0.0, scale=1.0, size=(n_fraud, len(V_COLUMNS))),
        columns=V_COLUMNS,
    )
    fraud["V1"] += 6.0
    fraud["V2"] += 4.0
    fraud["Class"] = 1

    df = pd.concat([legit, fraud], ignore_index=True)
    df["Time"] = rng.uniform(0, 172_800, size=len(df))
    df["Amount"] = rng.exponential(scale=50.0, size=len(df))
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Ordem de colunas igual ao dataset real (Time, V1..V28, Amount, Class).
    return df[["Time"] + V_COLUMNS + ["Amount", "Class"]]


@pytest.fixture
def synthetic_pipeline(synthetic_df):
    """Réplica, com dados sintéticos, do que `app/pipeline.py::get_pipeline()`
    faz — sem Streamlit. Reusada pelos testes de Análise de Threshold, que
    precisam de um modelo já treinado + escore de teste + threshold oficial
    para simular o slider sem repetir esse setup em cada teste.
    """
    from src.models import (
        calibrate_threshold_by_contamination,
        evaluate_model,
        fraud_score,
        train_all_models,
    )
    from src.preprocessing import get_train_test_split, scale_amount_time

    X_train, X_test, y_train, y_test = get_train_test_split(synthetic_df)
    X_train_scaled, X_test_scaled, _ = scale_amount_time(X_train, X_test)

    models = train_all_models(X_train_scaled, y_train)
    fraud_rate = float(y_train.mean())
    thresholds = {
        name: calibrate_threshold_by_contamination(
            fraud_score(model, X_train_scaled), fraud_rate
        )
        for name, model in models.items()
    }
    results = {
        name: evaluate_model(model, X_test_scaled, y_test, threshold=thresholds[name])
        for name, model in models.items()
    }

    return {"models": models, "results": results, "thresholds": thresholds, "y_test": y_test}
