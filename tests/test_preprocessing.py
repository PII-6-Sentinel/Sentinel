import numpy as np
import pandas as pd

from src.preprocessing import (
    get_train_test_split,
    scale_amount_time,
    train_test_split_stratified,
)


def test_scaler_is_fit_only_on_train_data():
    """O StandardScaler não pode usar nenhuma estatística do teste.

    Construímos treino e teste com distribuições BEM diferentes de
    Time/Amount de propósito: se o scaler vazasse dados de teste (ex.: por
    usar fit_transform no teste, ou fit no conjunto combinado), a média
    aprendida mudaria e o teste abaixo falharia.
    """
    X_train = pd.DataFrame({"Time": [0.0, 100.0, 200.0], "Amount": [10.0, 20.0, 30.0]})
    X_test = pd.DataFrame({"Time": [10_000.0, 20_000.0], "Amount": [5_000.0, 6_000.0]})

    train_mean = X_train[["Time", "Amount"]].mean()
    train_std = X_train[["Time", "Amount"]].std(ddof=0)

    X_train_scaled, X_test_scaled, scaler = scale_amount_time(X_train, X_test)

    # O scaler só pode "conhecer" a média/desvio do treino.
    assert np.allclose(scaler.mean_, train_mean.values)
    assert np.allclose(scaler.scale_, train_std.values)

    # X_test_scaled precisa ser (X_test - média_do_treino) / desvio_do_treino,
    # calculado à mão só com estatísticas do treino — nunca do teste.
    expected_test_scaled = (X_test[["Time", "Amount"]] - train_mean) / train_std
    assert np.allclose(X_test_scaled[["Time", "Amount"]].values, expected_test_scaled.values)

    # E o resultado do teste não pode ter influenciado o treino escalado.
    expected_train_scaled = (X_train[["Time", "Amount"]] - train_mean) / train_std
    assert np.allclose(X_train_scaled[["Time", "Amount"]].values, expected_train_scaled.values)


def test_scaler_only_touches_time_and_amount(synthetic_df):
    X = synthetic_df.drop(columns=["Class"])
    other_cols = [c for c in X.columns if c not in ("Time", "Amount")]

    X_train, X_test, _, _ = train_test_split_stratified(
        X, synthetic_df["Class"], test_size=0.3, random_state=0
    )
    X_train_scaled, X_test_scaled, _ = scale_amount_time(X_train, X_test)

    # Colunas V1..V28 devem sair intocadas (mesmo valor, mesma ordem de linhas).
    assert np.allclose(X_train_scaled[other_cols].values, X_train[other_cols].values)
    assert np.allclose(X_test_scaled[other_cols].values, X_test[other_cols].values)


def test_stratified_split_preserves_class_proportion(synthetic_df):
    X = synthetic_df.drop(columns=["Class"])
    y = synthetic_df["Class"]

    _, _, y_train, y_test = train_test_split_stratified(X, y, test_size=0.25, random_state=0)

    overall_rate = y.mean()
    assert abs(y_train.mean() - overall_rate) < 0.02
    assert abs(y_test.mean() - overall_rate) < 0.02


# ---------------------------------------------------------------------------
# get_train_test_split — fonte única do split (usada por get_pipeline() e
# get_cv_results() em app/pipeline.py, antes duplicada entre os dois).
# ---------------------------------------------------------------------------


def test_get_train_test_split_is_deterministic(synthetic_df):
    """Mesma entrada, mesmos parâmetros -> exatamente o mesmo split, sempre.

    Esta é a garantia que permite CV e pipeline principal compartilharem o
    mesmo X_train/y_train sem precisar passar os dados de um lado para o
    outro: os dois chamam get_train_test_split(df) e confiam que o
    resultado é idêntico.
    """
    X_train_1, X_test_1, y_train_1, y_test_1 = get_train_test_split(synthetic_df)
    X_train_2, X_test_2, y_train_2, y_test_2 = get_train_test_split(synthetic_df)

    pd.testing.assert_frame_equal(X_train_1, X_train_2)
    pd.testing.assert_frame_equal(X_test_1, X_test_2)
    pd.testing.assert_series_equal(y_train_1, y_train_2)
    pd.testing.assert_series_equal(y_test_1, y_test_2)


def test_cv_and_pipeline_see_the_same_split(synthetic_df):
    """CV (`get_cv_results`) e pipeline final (`get_pipeline`) chamam
    get_train_test_split(df) com os mesmos parâmetros default — este teste
    prova que isso realmente resulta no mesmo X_train/y_train para os dois,
    simulando exatamente as duas chamadas que app/pipeline.py faz.
    """
    # Chamada "como get_pipeline() faz"
    X_train_pipeline, X_test_pipeline, y_train_pipeline, y_test_pipeline = (
        get_train_test_split(synthetic_df)
    )
    # Chamada "como get_cv_results() faz" (descarta o teste)
    X_train_cv, _X_test_cv, y_train_cv, _y_test_cv = get_train_test_split(synthetic_df)

    pd.testing.assert_frame_equal(X_train_pipeline, X_train_cv)
    pd.testing.assert_series_equal(y_train_pipeline, y_train_cv)


def test_get_train_test_split_keeps_stratification(synthetic_df):
    X_train, X_test, y_train, y_test = get_train_test_split(synthetic_df)

    overall_rate = synthetic_df["Class"].mean()
    assert abs(y_train.mean() - overall_rate) < 0.02
    assert abs(y_test.mean() - overall_rate) < 0.02
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_get_train_test_split_feeds_scaler_without_leakage(synthetic_df):
    """Fim a fim: o resultado de get_train_test_split, passado para
    scale_amount_time, continua sem vazar estatísticas de teste — prova
    que centralizar o split não quebrou essa garantia já testada em
    test_scaler_is_fit_only_on_train_data.
    """
    X_train, X_test, _y_train, _y_test = get_train_test_split(synthetic_df)
    train_mean = X_train[["Time", "Amount"]].mean()
    train_std = X_train[["Time", "Amount"]].std(ddof=0)

    _X_train_scaled, X_test_scaled, scaler = scale_amount_time(X_train, X_test)

    assert np.allclose(scaler.mean_, train_mean.values)
    expected_test_scaled = (X_test[["Time", "Amount"]] - train_mean) / train_std
    assert np.allclose(X_test_scaled[["Time", "Amount"]].values, expected_test_scaled.values)
