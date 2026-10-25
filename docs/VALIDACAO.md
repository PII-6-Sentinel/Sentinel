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

Requer `data/raw/creditcard.csv` (README, seção 3). No relatório, o limiar
de menor custo é calculado no próprio teste só para **descrever** a curva de
custo; para **escolher** um limiar, calcule-o num período de validação
anterior ao teste.

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
