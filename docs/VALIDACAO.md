# Validação — o que o projeto prova e o que ainda não prova

O Sentinel tem duas vias de evidência, com pesos diferentes. Este documento
diz o que cada uma sustenta e o que seria preciso para afirmar desempenho
contra fraude real.

## 1. Motor de risco (domínio sintético) — protótipo demonstrativo

O motor (`src/risk/`) e as telas do app rodam sobre um domínio **sintético**.
Os rótulos de fraude foram gerados a partir dos mesmos sinais que o motor
usa (valor, local, horário, dispositivo, canal): há **circularidade**. Por
isso as métricas do motor (Central de Transações, Monitor ao Vivo,
`scripts/run_risk_engine.py`, `docs/RISK_ENGINE.md`) **validam a lógica do
sistema** — features sem vazamento, combinação de ML e regras, calibração
por validação, decisões explicáveis — e **não provam desempenho contra
fraude real**.

No app e nos relatórios, essa ressalva aparece junto de cada bloco de
métricas (`ui.metrics_caveat`, `PROTOTYPE_NOTICE`).

A circularidade foi reduzida, não eliminada (Etapa 14, `docs/RISK_ENGINE.md`):
valor e horário são comparados com hábitos **estimados do histórico**, não
com os parâmetros do gerador; compras legítimas às vezes fogem do hábito
(horário, valor, celular novo); e metade das fraudes usa o canal habitual do
titular. Com isso, nenhum cenário depende mais de um único sinal que nunca
aparece em compras legítimas — exceto `dispositivo_novo`, que continua
sempre online, com aparelho novo e não confiável.

## 2. Benchmark público (Kaggle) — técnicas em dados reais

O dataset Credit Card Fraud Detection (Kaggle/ULB) tem fraudes **reais e
rotuladas**, mas de outro contexto: cartões europeus, dois dias de 2013,
features anonimizadas por PCA. Ele valida as **técnicas** (modelos,
métricas, limiares), não o motor do Sentinel, que usa sinais que esse
dataset não tem.

`scripts/run_kaggle_benchmark.py` (lógica em `src/benchmark.py`) acrescenta
ao benchmark três análises mais próximas da operação:

| Análise | O que responde |
|---|---|
| **Split aleatório × temporal** | Quanto o resultado cai quando o modelo treina no passado e é testado no futuro, como em produção. O split aleatório (o padrão do projeto) é otimista. |
| **Volume de alertas** | Para cada orçamento de alertas (0,1% a 5% das transações): alertas por dia, fraudes pegas, precisão, recall e fração do valor de fraude coberta. Um limiar só é usável se o volume couber na equipe de revisão. |
| **Limiar por custo** | O limiar que minimiza revisões × custo de revisão + valor das fraudes não detectadas. O custo de revisão é uma premissa (`--custo-revisao`). |

```bash
python scripts/run_kaggle_benchmark.py --custo-revisao 5 --teste 0.2
```

Os quatro modelos (baseline estatístico, Regressão Logística, Isolation
Forest e Gradient Boosting) recebem a **hora do dia**, não o `Time` bruto:
`Time` conta segundos desde o início da coleta e, no split temporal, todo o
teste ficaria fora do intervalo visto no treino. Assim a queda do aleatório
para o temporal mede a mudança nos dados, e não um artefato dessa coluna.
Valores (`Amount`, custos) estão na moeda do dataset, o euro.

Requer `data/raw/creditcard.csv` (README, seção 3). No relatório, o limiar
de menor custo é calculado no próprio teste só para **descrever** a curva de
custo; para **escolher** um limiar, calcule-o num período de validação
anterior ao teste.

### Resultados com o dataset real

Rodado em 29/10/2026 com `data/raw/creditcard.csv` (284.807 transações,
492 fraudes), parâmetros padrão (`--custo-revisao 5 --teste 0.2`):

| Modelo | PR-AUC aleatório | PR-AUC temporal | ROC-AUC temporal | Recall com 0,1% de alertas (temporal) | Custo no limiar ótimo |
|---|---|---|---|---|---|
| Baseline Estatístico | 0,169 | 0,048 | 0,953 | 0% | € 7.709 (100%) |
| Regressão Logística | 0,717 | 0,767 | 0,986 | 69% (52/75) | € 3.802 (49%) |
| Isolation Forest | 0,185 | 0,045 | 0,956 | 0% | € 6.014 (78%) |
| **Gradient Boosting** | **0,864** | **0,784** | 0,981 | 69% (52/75) | **€ 3.179 (41%)** |

Custo em % do prejuízo sem nenhum alerta (€ 7.729). Com 0,1% das
transações alertadas (179 por dia), Regressão Logística e Gradient Boosting
acertam 91% dos alertas.

**Como ler:**

- **Os modelos supervisionados aprendem a fraude real.** Regressão
  Logística e Gradient Boosting ficam muito acima do acaso (PR-AUC do acaso
  = 0,0017, a taxa de fraude). O Gradient Boosting é o melhor em PR-AUC e
  em custo: com o mesmo número de alertas (287) deixa passar 12 fraudes,
  contra 11 da Regressão Logística, mas pega as de maior valor.
- **Os não supervisionados não servem aqui.** Baseline e Isolation Forest
  procuram "o que é raro", e fraude neste dataset não é só raridade: no
  menor orçamento de alertas, não pegam nenhuma.
- **A Regressão Logística não caiu no temporal** (0,717 → 0,767). Com dois
  dias de dados e 75 fraudes no teste temporal, a diferença entre os splits
  está dentro do ruído; o Gradient Boosting caiu de 0,864 para 0,784.
- **O limiar de custo foi achado no próprio teste** (ver acima): os custos
  descrevem a curva, não um resultado de operação.

**Correção do Gradient Boosting.** Na primeira execução, com
`learning_rate` 0,1 e sem regularização, o treino divergiu: a perda subia a
cada árvore, o early stopping parou na 11ª e o ROC-AUC ficou em 0,48 (pior
que o acaso). Com 0,17% de fraudes, folhas com 1 ou 2 fraudes recebiam
valores extremos. A configuração atual (`learning_rate` 0,05,
`l2_regularization` 1, folhas com ao menos 50 transações, 300 árvores) foi
escolhida numa divisão interna do treino (PR-AUC de validação 0,54 → 0,82),
sem usar o teste.

## 3. Requisitos para qualquer alegação de desempenho antifraude

Nenhum número deste projeto deve ser apresentado como "o Sentinel detecta X%
das fraudes". Para fazer essa afirmação, a validação precisa de:

1. **Dados reais rotulados** da instituição: transações com o desfecho
   confirmado (contestação, chargeback, confirmação do cliente), respeitando
   o atraso entre a transação e a confirmação do rótulo.
2. **Separação por período**: treino com um período, escolha de pesos e
   limiares com o período seguinte (validação) e medição final com um
   período ainda posterior (teste), nunca visto. Clientes e cartões não
   podem vazar entre os períodos de forma que antecipe o futuro.
3. **Limiares avaliados por custo**: o custo de um falso positivo (revisão,
   atrito com o cliente, compra perdida) e o de uma fraude não detectada
   (valor perdido, custos operacionais), escolhidos com a área de negócio e
   aplicados na validação — não um limiar fixo nem escolhido no teste.
4. **Volume de alertas compatível** com a capacidade da equipe de revisão
   (alertas por dia), reportado junto das métricas.
5. **Métricas com incerteza**: PR-AUC, recall e precisão no orçamento de
   alertas, valor de fraude coberto e custo total, com variação entre
   períodos (ou intervalos por bootstrap), não só um número.

Até lá, o Sentinel é apresentado como **protótipo demonstrativo**: a
arquitetura e a lógica funcionam de ponta a ponta; o desempenho real é o
próximo passo, com dados de uma instituição parceira.
