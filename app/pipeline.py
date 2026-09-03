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

import pandas as pd
import streamlit as st

from src.data_loader import load_raw_data
from src.evaluation import CVResult, cross_validate_models, threshold_sweep
from src.models import (
    calibrate_threshold_by_contamination,
    evaluate_model,
    fraud_score,
    train_all_models,
)
from src.preprocessing import get_train_test_split, scale_amount_time


@st.cache_data(show_spinner="Carregando dataset...")
def load_data():
    return load_raw_data()


@st.cache_resource(show_spinner="Treinando modelos (baseline, regressão logística, isolation forest)...")
def get_pipeline():
    df = load_data()

    # Fonte única do split treino/teste (ver src/preprocessing.py) —
    # get_cv_results() abaixo usa exatamente a mesma função, com os mesmos
    # parâmetros default, então os dois enxergam o mesmo X_train/y_train.
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    X_train_scaled, X_test_scaled, _scaler = scale_amount_time(X_train, X_test)

    # Treino não usa a taxa de fraude em nenhum momento (ver train_all_models).
    models = train_all_models(X_train_scaled, y_train)

    # Proporção de fraude estimada no TREINO, usada só para calibrar um
    # threshold default por modelo — nunca para treinar. Calibrado com o
    # escore do treino (nunca do teste), para não vazar o conjunto de
    # avaliação para dentro da escolha do ponto de operação. Ver a ressalva
    # completa em calibrate_threshold_by_contamination.
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

    return {
        "models": models,
        "results": results,
        "thresholds": thresholds,
        "X_test": X_test,  # não escalado — para exibição legível na demo
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "fraud_rate": fraud_rate,
    }


@st.cache_resource(
    show_spinner="Rodando validação cruzada estratificada (5 folds × 3 modelos)..."
)
def get_cv_results() -> dict[str, CVResult]:
    """Validação cruzada — resultado SEPARADO do de `get_pipeline()`.

    Chama `get_train_test_split(df)` — a MESMA função, com os mesmos
    parâmetros default, que `get_pipeline()` usa — para chegar em
    X_train/y_train. Determinístico (`random_state` fixo dentro da
    função), então os dois enxergam exatamente o mesmo split. O conjunto
    de teste (`_X_test`/`_y_test`) é descartado imediatamente aqui, sem
    nunca ser passado para `cross_validate_models` — a validação cruzada
    nunca tem acesso ao conjunto de teste final, nem por acidente (a
    própria assinatura de `cross_validate_models` não tem parâmetro de
    teste).

    Cacheado à parte de `get_pipeline()` de propósito: a validação cruzada
    treina 3 modelos × 5 folds (mais caro que o pipeline principal, que
    treina cada modelo uma única vez) e serve a um propósito diferente
    (estabilidade entre folds, não o resultado final) — mantê-la em cache
    próprio evita que trocar de página force recomputar as duas coisas.
    """
    df = load_data()
    X_train, _X_test, y_train, _y_test = get_train_test_split(df)
    return cross_validate_models(X_train, y_train)


@st.cache_data(show_spinner=False)
def get_threshold_sweep(model_name: str, n_points: int = 100) -> pd.DataFrame:
    """Varredura de threshold (precisão/recall/F1) para UM modelo, cacheada
    por nome de modelo.

    Não depende do valor do threshold escolhido no slider da página de
    Análise de Threshold — só do modelo selecionado — então não precisa
    ser recalculada a cada movimento do slider (só `calculate_metrics`,
    chamada diretamente pela página a cada interação, precisa rodar de
    novo — e é barata: nenhum modelo é retreinado, nenhum dado é
    recarregado).
    """
    pipeline = get_pipeline()
    result = pipeline["results"][model_name]
    return threshold_sweep(pipeline["y_test"], result.y_score, n_points=n_points)
