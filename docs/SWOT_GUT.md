# Análise SWOT / GUT — Projeto Sentinel

> Projeto Integrador (TTI 304 — Gerenciamento de Projetos em TI)
> Tema: Detecção de anomalias em transações financeiras — comparação entre
> abordagens estatísticas e Machine Learning.

## Objetivo do projeto

Construir e comparar, de forma metodologicamente rigorosa, diferentes
abordagens de detecção de fraude em transações de cartão de crédito:

1. **Baseline estatístico** — regras simples (ex.: limiares em `Amount`,
   z-score, desvio em relação à distribuição por período de `Time`).
2. **Machine Learning supervisionado** — ex.: Regressão Logística, Random
   Forest.
3. **Machine Learning não supervisionado** — ex.: Isolation Forest, para
   detecção de anomalias sem depender fortemente do rótulo `Class`.

O critério de sucesso do PI não é apenas "ter um modelo que funciona", mas
**demonstrar entendimento do processo de avaliação** — matriz de confusão,
precisão/recall, F1-score, AUC-ROC/AUC-PR — em um cenário de classes
extremamente desbalanceadas.

## Dataset

- **Credit Card Fraud Detection** (Kaggle / ULB — Machine Learning Group)
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- ~284.807 transações, 492 fraudes (~0,17%)
- Features `V1`–`V28`: componentes anonimizadas via PCA (não interpretáveis
  individualmente por design, para preservar confidencialidade)
- `Time`: segundos decorridos desde a primeira transação do dataset
- `Amount`: valor da transação
- `Class`: 0 = legítima, 1 = fraude (variável alvo)

## Análise SWOT (resumo)

| | Interno | Externo |
|---|---|---|
| **Positivo** | Dataset público, bem documentado e amplamente usado como benchmark acadêmico | Boa disponibilidade de literatura e bibliotecas maduras (`scikit-learn`, `imbalanced-learn`) para o problema |
| **Negativo** | Pouca experiência prévia da equipe com validação estatística rigorosa | Prazo de semestre apertado para cobrir exploração, modelagem e comparação com rigor |

### Forças (Strengths)
- Dataset já pré-processado (PCA aplicado), reduzindo trabalho de engenharia de features.
- Problema bem delimitado e com métricas de sucesso claras na literatura.

### Fraquezas (Weaknesses)
- Equipe com pouca vivência prática em métricas além de acurácia (risco de
  interpretar mal um modelo "99,8% acurado" que na prática não detecta fraude
  nenhuma).
- Pouca experiência com técnicas de balanceamento de classes.

### Oportunidades (Opportunities)
- Boa base para aprendizado didático de todo o time em avaliação de modelos.
- Comparação estatística vs. ML gera material rico para a monografia/relatório.

### Ameaças (Threats)
- **Dataset fortemente desbalanceado (~0,17% de fraudes)** — maior ameaça
  identificada. Sem tratamento adequado (balanceamento + métricas corretas),
  qualquer modelo tende a "aprender" a sempre prever a classe majoritária e
  parecer bom em acurácia, mascarando desempenho ruim de fato.
- Risco de vazamento de dados (data leakage) entre treino/teste se o
  balanceamento (SMOTE) for aplicado antes do split, ou se normalização usar
  estatísticas do conjunto completo.

## Análise GUT (Gravidade, Urgência, Tendência)

| Risco | Gravidade (1-5) | Urgência (1-5) | Tendência (1-5) | GUT | Prioridade |
|---|---|---|---|---|---|
| Falta de rigor estatístico na avaliação (equipe pouco experiente) | 5 | 5 | 4 | 100 | **Crítica** |
| Desbalanceamento de classes tratado incorretamente | 5 | 5 | 3 | 75 | **Alta** |
| Prazo de semestre apertado | 4 | 4 | 4 | 64 | Alta |
| Overfitting em modelos ML por validação mal feita | 4 | 3 | 3 | 36 | Média |

## Decisões de mitigação já incorporadas ao projeto

1. **Pipeline de avaliação estruturado desde o início** — módulo dedicado em
   `src/`, testado com um baseline simples antes de qualquer modelo mais
   complexo (ver [`src/preprocessing.py`](../src/preprocessing.py)).
2. **Métricas adequadas a classes desbalanceadas por padrão**: matriz de
   confusão, precisão, recall, F1-score e AUC-ROC/AUC-PR — nunca acurácia
   isolada como critério de sucesso.
3. **Split estratificado** (`train_test_split(..., stratify=y)`) e
   balanceamento (SMOTE/undersampling via `imbalanced-learn`) aplicado
   **somente no conjunto de treino**, depois do split, para evitar
   vazamento de dados.
4. **MVP primeiro**: baseline estatístico simples funcionando de ponta a
   ponta antes de otimizar modelos de ML.
