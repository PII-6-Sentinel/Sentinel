"""Treinamento e avaliação das abordagens comparadas pelo projeto.

Centralizado aqui (em vez de dentro do app Streamlit) porque a mesma lógica
serve tanto a interface quanto notebooks futuros de modelagem — o app é só
uma camada de visualização sobre este módulo.

O fluxo de avaliação segue sempre a mesma sequência, deliberadamente em
três passos separados (nunca escondidos dentro de um só):

    score (fraud_score)  →  threshold  →  decisão (apply_threshold)

Isso existe porque, na versão anterior deste módulo, cada modelo decidia
"fraude ou não" à sua própria maneira e escondida: a Regressão Logística
usava o `.predict()` padrão do sklearn (threshold de probabilidade = 0.5),
enquanto o baseline estatístico e o Isolation Forest usavam um corte de
percentil calibrado com a taxa REAL de fraude do treino. Isso tornava a
comparação entre os três inconsistente (pontos de operação diferentes) e
fazia dois dos três modelos usarem, de forma não-documentada, uma
informação (a taxa exata de fraude) que não estaria disponível com essa
precisão em produção.

Agora todo modelo expõe só um `fraud_score()` (contínuo, maior = mais
suspeito); a decisão 0/1 é sempre `apply_threshold(score, threshold)`, com
o threshold escolhido de forma explícita e visível — nunca implícito.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
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

    Esta classe é SÓ um escore — não decide "fraude ou não" e não conhece
    nenhuma taxa de contaminação. `fit()` existe apenas por uniformidade de
    interface com os modelos do scikit-learn (permite chamar `.fit(X_train)`
    igual para os três modelos); como o escore aqui é uma fórmula fixa, não
    há nada de fato "aprendido" com os dados. A escolha do threshold de
    decisão é responsabilidade de `calibrate_threshold_by_contamination` /
    `apply_threshold`, chamadas separadamente por quem for avaliar o modelo.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "StatisticalBaseline":
        return self

    def decision_score(self, X: pd.DataFrame) -> np.ndarray:
        return (X[V_COLUMNS] ** 2).sum(axis=1).to_numpy()


@dataclass
class ModelResult:
    """Resultado de avaliação de um modelo sobre o conjunto de teste, a um
    threshold específico. Métricas baseadas em ranking (AUC-ROC, AUC-PR e
    as curvas) não dependem do threshold; precision/recall/f1/matriz de
    confusão dependem — por isso `threshold` fica registrado no resultado.
    """

    name: str
    threshold: float
    y_pred: np.ndarray
    y_score: np.ndarray
    confusion_matrix: np.ndarray
    precision: float
    recall: float
    f1: float
    auc_roc: float
    auc_pr: float
    fpr: np.ndarray = field(repr=False)
    tpr: np.ndarray = field(repr=False)
    pr_precision: np.ndarray = field(repr=False)
    pr_recall: np.ndarray = field(repr=False)


def train_all_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, object]:
    """Treina as três abordagens comparadas pelo projeto.

    Nenhum modelo recebe a taxa real de fraude neste passo — treino e
    calibração de threshold são etapas deliberadamente separadas (ver
    `calibrate_threshold_by_contamination`). O Isolation Forest é criado
    com `contamination="auto"` (heurística fixa do próprio scikit-learn,
    não derivada de `y_train`); como este projeto nunca chama
    `IsolationForest.predict()`/`.decision_function()` (só `score_samples()`,
    via `fraud_score`), esse parâmetro não influencia nenhum resultado — é
    mantido apenas porque o construtor do sklearn o exige.
    """
    baseline = StatisticalBaseline().fit(X_train)

    logreg = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    ).fit(X_train, y_train)

    iso_forest = IsolationForest(
        contamination="auto", n_estimators=200, random_state=42, n_jobs=-1
    ).fit(X_train)

    return {
        "Baseline Estatístico": baseline,
        "Regressão Logística": logreg,
        "Isolation Forest": iso_forest,
    }


def fraud_score(model, X: pd.DataFrame) -> np.ndarray:
    """Escore contínuo — quanto maior, mais 'parecido com fraude'.

    Único lugar do projeto que sabe como extrair um escore de cada tipo de
    modelo. A partir daqui, todo o resto do pipeline (threshold, métricas,
    gauge da demo) trabalha só com esse escore normalizado — nunca com a
    API específica de cada modelo.
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


def apply_threshold(y_score: np.ndarray, threshold: float) -> np.ndarray:
    """Converte um escore contínuo em decisão binária (0/1).

    Esta é a ÚNICA função do projeto que transforma "score" em "decisão".
    Funciona igual para qualquer modelo, porque `fraud_score()` já garante
    que escore maior = mais suspeito, independente da origem do escore.

    Levanta ValueError para threshold ausente/NaN — silenciosamente aceitar
    um threshold inválido produziria uma comparação (`>=`) que ou levanta
    erro mais adiante de forma confusa, ou (pior, com NaN) retorna sempre
    False sem avisar ninguém.
    """
    if threshold is None or (isinstance(threshold, float) and np.isnan(threshold)):
        raise ValueError(f"Threshold inválido: {threshold!r} (precisa ser um número finito)")
    return (np.asarray(y_score) >= threshold).astype(int)


def calibrate_threshold_by_contamination(y_score: np.ndarray, contamination: float) -> float:
    """Escolhe um threshold como o percentil (1 - contamination) do escore.

    IMPORTANTE — leia antes de reusar: `contamination` aqui deve vir de uma
    taxa ESTIMADA a partir do conjunto de TREINO (ex.: `y_train.mean()`),
    nunca do conjunto de teste — calibrar com o teste vazaria informação de
    avaliação para dentro da escolha do ponto de operação.

    Isso também NÃO é um threshold de produção real: estamos usando a taxa
    exata de fraude do dataset rotulado só para gerar, nesta etapa
    acadêmica, um ponto de operação default comparável entre os três
    modelos. Em produção, essa taxa não estaria disponível com essa
    precisão — o threshold seria escolhido por critério de negócio (custo
    de investigar um falso positivo vs. custo de deixar passar uma fraude).
    Uma futura página de "análise de threshold" deve permitir escolher o
    corte de outras formas, reusando `apply_threshold`/`evaluate_at_threshold`
    diretamente sobre o mesmo `y_score`, sem depender desta função.
    """
    if not np.isfinite(contamination) or not (0 < contamination < 1):
        raise ValueError(
            f"contamination precisa ser um número em (0, 1), recebido {contamination!r}"
        )
    return float(np.quantile(y_score, 1 - contamination))


def evaluate_at_threshold(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict:
    """Precision/recall/F1/matriz de confusão para UM threshold específico.

    Reutilizável para recalcular essas métricas em quantos thresholds
    forem necessários sobre o MESMO escore já calculado — sem retreinar
    nenhum modelo e sem recalcular `fraud_score`. É a peça que uma futura
    página de "análise de threshold" (ainda não implementada) vai chamar
    repetidamente conforme o usuário mover um slider.
    """
    y_pred = apply_threshold(y_score, threshold)
    return {
        "threshold": threshold,
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> ModelResult:
    """Calcula matriz de confusão + métricas adequadas a classes desbalanceadas,
    para o `threshold` fornecido.

    `threshold` é obrigatório e sem valor default de propósito: esta função
    não escolhe um ponto de operação por conta própria — quem chama decide
    (normalmente via `calibrate_threshold_by_contamination`, mas pode ser
    qualquer outro critério).

    Acurácia é deliberadamente omitida: com ~0,17% de fraudes, é uma métrica
    enganosa (ver docs/SWOT_GUT.md). AUC-ROC e AUC-PR são calculadas sobre o
    ranking completo do escore — não dependem do threshold escolhido; as
    demais métricas, sim.
    """
    y_score = fraud_score(model, X_test)
    at_threshold = evaluate_at_threshold(y_test, y_score, threshold)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_score)

    return ModelResult(
        name=type(model).__name__,
        threshold=threshold,
        y_pred=at_threshold["y_pred"],
        y_score=y_score,
        confusion_matrix=at_threshold["confusion_matrix"],
        precision=at_threshold["precision"],
        recall=at_threshold["recall"],
        f1=at_threshold["f1"],
        auc_roc=roc_auc_score(y_test, y_score),
        auc_pr=average_precision_score(y_test, y_score),
        fpr=fpr,
        tpr=tpr,
        pr_precision=pr_precision,
        pr_recall=pr_recall,
    )


def percentile_rank(value: float, distribution: np.ndarray) -> float:
    """Posição percentual (0-100) de `value` dentro de `distribution`.

    Usado para exibir o escore de um modelo em uma escala 0-100 comum,
    já que Regressão Logística (probabilidade), Isolation Forest (escore
    de isolamento) e o baseline (distância) não são naturalmente
    comparáveis em escala.
    """
    return float((distribution < value).mean() * 100)
