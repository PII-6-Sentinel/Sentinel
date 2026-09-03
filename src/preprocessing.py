"""Pipeline de pré-processamento reutilizável entre modelos.

Todas as abordagens comparadas no projeto (regras estatísticas, regressão
logística, isolation forest, ...) partem dos mesmos dados de treino/teste
e do mesmo critério de avaliação. Centralizar essa lógica aqui evita que
cada notebook de modelo implemente o split ou o balanceamento de um jeito
sutilmente diferente, o que invalidaria a comparação entre eles.

Ordem das operações (importante para não vazar dados de teste no treino):
    1. split treino/teste (estratificado, preserva a proporção de fraudes)
    2. escalonar Time/Amount (fit só no treino)
    3. balancear classes (SMOTE/undersampling) — SÓ no treino
"""

from typing import Literal

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "Class"
BalanceMethod = Literal["smote", "undersample", "none"]


def split_features_target(
    df: pd.DataFrame, target: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split treino/teste com `stratify=y`.

    Com apenas ~0,17% de fraudes, um split aleatório comum pode, por azar,
    deixar poucas (ou nenhuma) fraude no conjunto de teste. `stratify=y`
    garante que treino e teste mantenham a mesma proporção de classes do
    dataset original.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Fonte única de verdade do split treino/teste do projeto.

    Antes desta função, `app/pipeline.py` chamava `split_features_target` +
    `train_test_split_stratified` de forma duplicada em `get_pipeline()`
    (resultado final) e `get_cv_results()` (validação cruzada) — com o
    mesmo `random_state`, o resultado já saía idêntico, mas a lógica de
    montar o split estava escrita duas vezes. Agora os dois chamam só esta
    função.

    Deliberadamente NÃO escalona nem balanceia — só separa
    features/alvo e faz o split estratificado. Cada consumidor decide o
    que fazer depois de forma independente: o pipeline principal escala
    uma vez sobre o treino completo; a validação cruzada escala de novo,
    por fold, dentro de `cross_validate_models` — reusar um scaler daqui
    para os dois casos misturaria essas duas responsabilidades.
    """
    X, y = split_features_target(df)
    return train_test_split_stratified(
        X, y, test_size=test_size, random_state=random_state
    )


def scale_amount_time(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Padroniza `Time` e `Amount`; deixa `V1`...`V28` como estão.

    As colunas V1-V28 já são saída de um PCA (o dataset original já foi
    normalizado antes da transformação), então já estão em escala
    comparável entre si. `Time` e `Amount` são as únicas colunas originais
    e ficam em escalas muito diferentes uma da outra (e do resto), o que
    prejudica modelos sensíveis a escala como regressão logística.

    O `StandardScaler` é ajustado (`fit`) apenas no treino e só aplicado
    (`transform`) no teste, para não vazar estatísticas do conjunto de
    teste (média/desvio-padrão) para o treino.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()
    cols_to_scale = ["Time", "Amount"]

    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train, X_test, scaler


def balance_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: BalanceMethod = "smote",
    random_state: int = 42,
):
    """Balanceia as classes — aplicado SOMENTE no conjunto de treino.

    Nunca balancear o conjunto de teste: ele precisa continuar refletindo a
    proporção real de fraudes (~0,17%) para que as métricas calculadas
    sobre ele representem o desempenho esperado em produção.

    - "smote": gera exemplos sintéticos da classe minoritária (fraude),
      interpolando entre vizinhos próximos no espaço de features. Preserva
      todos os dados reais de transações legítimas.
    - "undersample": remove aleatoriamente exemplos da classe majoritária
      até igualar a minoritária. Mais rápido, mas descarta dados.
    - "none": retorna os dados originais (desbalanceados) — útil como
      referência para comparar com/sem balanceamento.
    """
    if method == "none":
        return X_train, y_train

    if method == "smote":
        sampler = SMOTE(random_state=random_state)
    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError(f"Método de balanceamento desconhecido: {method!r}")

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def build_train_test_data(
    df: pd.DataFrame,
    balance_method: BalanceMethod = "smote",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Executa o pipeline completo: split -> escalonamento -> balanceamento.

    Função de conveniência para notebooks de modelagem: uma chamada entrega
    X_train/X_test/y_train/y_test prontos para treinar qualquer classificador,
    já na ordem correta de operações.
    """
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_test, _ = scale_amount_time(X_train, X_test)
    X_train, y_train = balance_training_data(
        X_train, y_train, method=balance_method, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
