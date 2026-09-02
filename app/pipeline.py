"""Camada de cache do Streamlit sobre a lógica de src/.

Todo o app compartilha os mesmos dados carregados e os mesmos modelos
treinados — sem esta camada, cada interação do usuário (trocar de página,
mexer num filtro) re-executaria o carregamento do CSV de 150MB e o
retreinamento dos três modelos, o que tornaria o app inutilizável.

`st.cache_data` é usado para o DataFrame bruto (dado "de valor", copiado a
cada chamada para proteger contra mutação acidental). `st.cache_resource` é
usado para o pipeline treinado (modelos + splits), pois evita cópias
desnecessárias de objetos grandes e é a forma recomendada pelo Streamlit
para cachear recursos como modelos treinados.
"""

import streamlit as st

from src.data_loader import load_raw_data
from src.models import evaluate_model, train_all_models
from src.preprocessing import (
    scale_amount_time,
    split_features_target,
    train_test_split_stratified,
)


@st.cache_data(show_spinner="Carregando dataset...")
def load_data():
    return load_raw_data()


@st.cache_resource(show_spinner="Treinando modelos (baseline, regressão logística, isolation forest)...")
def get_pipeline():
    df = load_data()

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    X_train_scaled, X_test_scaled, _scaler = scale_amount_time(X_train, X_test)

    # Proporção de fraude estimada no treino, usada para calibrar o baseline
    # estatístico e o Isolation Forest (ambos não-supervisionados/baseados em
    # limiar de contaminação, em vez de aprenderem a fronteira via rótulos).
    fraud_rate = float(y_train.mean())

    models = train_all_models(X_train_scaled, y_train, contamination=fraud_rate)
    results = {
        name: evaluate_model(model, X_test_scaled, y_test)
        for name, model in models.items()
    }

    return {
        "models": models,
        "results": results,
        "X_test": X_test,  # não escalado — para exibição legível na demo
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "fraud_rate": fraud_rate,
    }
