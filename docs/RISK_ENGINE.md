# Risk Engine — arquitetura, decisões e resultados

> Etapas 7-8 do Sentinel. Complementa
> [`docs/DOMINIO_SINTETICO.md`](DOMINIO_SINTETICO.md) (de onde vêm os dados e
> as features) e [`docs/SWOT_GUT.md`](SWOT_GUT.md) (planejamento).

**Aviso que vale para todo este documento:** o motor opera sobre o domínio
**sintético**. Os rótulos de fraude foram gerados por cenários que manipulam
exatamente as features que o motor usa. Os resultados abaixo validam a
**lógica** do sistema sobre dados simulados — **não provam desempenho em
fraude real**.

## O que o motor faz

Recebe uma transação (suas features) e devolve um `RiskResult`:

| Campo | Significado |
|---|---|
| `risk_score` | 0-100, inteiro |
| `ml_score` | 0-100, score do modelo de Regressão Logística |
| `behavior_score` | 0-100, score das regras explicáveis |
| `level` | `baixo` / `médio` / `alto` |
| `action` | `aprovar` / `revisar` / `bloquear` |
| `reasons` | motivos legíveis, ordenados pela contribuição ao score final |

Exemplo real (transação de teste, rótulo `combinacao_de_sinais`):

```
Risk Score 93/100 (ML 100.0, regras 85.8) -> bloquear / revisão urgente
  - Modelo estatístico: padrão semelhante a fraudes conhecidas (score ML 100/100)
  - Viagem impossível: 1.390 km/h entre duas compras
  - Localização incomum: 7.874 km da residência
  - Horário fora da janela habitual (3,2 h fora)
  - Valor 2,6x acima da média do cliente
```

## Arquitetura (`src/risk/`)

```
banco sintético ──► dataset.build_risk_table ──┬─► behavior_score (regras) ─┐
                    (features + km, dispositivo)│                            ├─► engine.RiskEngine ─► RiskResult
                                                └─► ml_score (Reg. Logística)┘
transação nova ──► new_transaction.compute_transaction_features ──► engine.assess
avaliação ──► evaluation (split por cliente, métricas, calibração) ──► report ──► scripts/run_risk_engine.py
```

| Módulo | Responsabilidade |
|---|---|
| `dataset.py` | `build_risk_table`: `build_feature_table` (inalterada) + `customer_id`, distância em km e `device_untrusted` (`not Device.is_trusted`) |
| `behavior_score.py` | 6 regras, pesos/limiares num único `BehaviorConfig` |
| `ml_score.py` | `MLScorer`: padronização + Regressão Logística |
| `engine.py` | `RiskEngine`, `EngineConfig`, `DecisionBands`, `RiskResult`; `combine_scores` é a única implementação da combinação |
| `new_transaction.py` | features de uma transação nova a partir do histórico do cliente no banco |
| `evaluation.py` | split por cliente, métricas, calibração de limiares |
| `report.py` | avaliação comparativa completa (usada pelo script) |

Nada em `src/risk/` usa a camada Kaggle: aqueles modelos operam em V1–V28
(PCA) e não conseguem pontuar transações sintéticas.

## 1. Score por regras — pesos e justificativas

Sinais contínuos usam uma **rampa**: abaixo do início a regra não dispara;
acima, vale o mínimo e cresce linearmente até o máximo. Assim o score é
contínuo (permite ordenar e calibrar), mas cada ponto continua explicável.

| Regra | Dispara quando | Pontos | Por quê |
|---|---|---|---|
| Viagem impossível | velocidade implícita > 900 km/h | 40 | Única evidência fisicamente incompatível com um único titular honesto |
| Localização | > 100 km da residência (rampa log até 2.000 km) | 5 → 25 | Forte, mas com explicação legítima comum (viagens) |
| Valor | > 2x a média do cliente (rampa até 6x) | 5 → 25 | Forte, mas compras grandes planejadas existem; pesa o mesmo que localização |
| Dispositivo novo | aparelho nunca usado pelo cliente | 20 | Típico de tomada de conta, mas pessoas trocam de celular |
| Dispositivo não confiável | `Device.is_trusted = False` | 15 | Somado ao anterior (35), aproxima-se de uma viagem impossível |
| Horário | fora da janela habitual (rampa até 4 h fora) | 5 → 15 | O sinal mais fraco: janelas de horário são estimativas difusas |

Soma limitada a 100. Os pesos foram escolhidos por raciocínio de domínio,
**não ajustados aos dados sintéticos** — ajustá-los repetiria a
circularidade dos rótulos. Todos ficam em `BehaviorConfig`.

## 2. Score de ML

- Regressão Logística com `class_weight='balanced'` (compensa ~3% de
  fraude), padronização das features, seed fixa (42).
- Features: `amount_deviation`, `location_distance`, `unusual_hour`,
  `new_device`, `device_untrusted`, `log1p(implied_speed_kmh)`,
  `is_impossible_travel`.
- Score = probabilidade × 100. Com `class_weight` balanceado essa
  "probabilidade" fica inflada: ela **ordena** bem, mas **não é uma
  probabilidade calibrada de fraude** e não deve ser apresentada como tal.

O que o modelo aprendeu (coeficientes sobre features padronizadas):

| Feature | Coeficiente |
|---|---|
| `unusual_hour` | +2,32 |
| `location_distance` | +1,61 |
| `amount_deviation` | +1,57 |
| `log_speed_kmh` | +1,03 |
| `new_device` | +0,80 |
| `device_untrusted` | +0,51 |
| `is_impossible_travel` | +0,10 |

`unusual_hour` é o maior coeficiente porque, no gerador, nenhuma transação
normal sai da janela de horário: para o modelo, horário é um separador
perfeito. É um artefato dos dados, não conhecimento sobre fraude.
`is_impossible_travel` fica perto de zero porque a velocidade em escala log
já carrega a mesma informação.

## 3. Combinação, piso e faixas

**Score final** = 0,5 × ML + 0,5 × regras, arredondado (configurável em
`EngineConfig`). Pesos iguais não privilegiam nenhum dos dois a priori: as
regras são explicáveis e independem dos rótulos; o ML aprende combinações,
mas a partir de rótulos circulares. Com 50/50, metade de todo score vem de
motivos legíveis. Na varredura de pesos na **validação**, 50/50 teve o
maior PR-AUC (0,894 contra 0,883 do ML puro). No teste, porém, o combinado
perde para o ML — ver seção 5.

**Piso por regra dura:** viagem impossível força o nível a ser ao menos
**médio** (revisão), nunca é aprovada automaticamente. Não força **alto**
(bloqueio) porque a transação marcada pode ser a legítima: se a compra
anterior é que foi a fraude (num lugar distante), a compra seguinte do
cliente real é que parece impossível. Bloquear puniria o cliente; uma
revisão humana resolve.

**Faixas:** padrão 0-30 baixo → aprovar; 31-70 médio → revisar; 71-100 alto
→ bloquear. No relatório, são **calibradas na validação**:

- revisão: o maior limiar que ainda sinaliza ≥ 90% das fraudes (no máximo
  10% das fraudes aprovadas sem revisão humana);
- bloqueio: o menor limiar com precisão ≥ 90% (só se bloqueia
  automaticamente quando quase tudo o que é bloqueado é fraude).

Resultado da calibração: **baixo 0-54, médio 55-61, alto 62-100**. A faixa
média fica estreita porque, nestes dados, os scores das fraudes e das
transações normais se separam bem — há pouca "zona cinzenta".

## 4. Metodologia de avaliação

- **Split por cliente** (`GroupShuffleSplit`), nunca por transação:
  transações do mesmo cliente compartilham perfil; separá-las deixaria o
  modelo reconhecer no teste um cliente visto no treino.
- Três conjuntos de clientes, sem sobreposição: treino (64 clientes, 3.311
  transações, 111 fraudes) ajusta o ML; validação (27 / 1.345 / 40) escolhe
  pesos e limiares; teste (39 / 2.256 / 56) é tocado uma única vez.
- Seed fixa (42) em tudo.
- Métricas: PR-AUC (principal, pelo desbalanceamento), ROC-AUC, precisão,
  recall, F1, matriz de confusão e recall por cenário com contagens.
- **Estabilidade:** como o teste tem só 56 fraudes (6 a 14 por cenário),
  o relatório repete o PR-AUC em validação cruzada 5-fold estratificada por
  cliente sobre a tabela inteira.

Para reproduzir: `python scripts/run_risk_engine.py`.

## 5. Resultados (teste, 39 clientes nunca vistos)

"Sinalizada" = não aprovada automaticamente (nível médio ou alto).

| Abordagem | PR-AUC | ROC-AUC | Precisão | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| Só regras | 0,606 | 0,967 | 0,166 | 0,911 | 0,281 | 51 | 256 | 5 |
| Só ML | **0,924** | 0,996 | **0,689** | 0,911 | **0,785** | 51 | 23 | 5 |
| Combinado (motor) | 0,842 | 0,994 | 0,662 | 0,911 | 0,767 | 51 | 26 | 5 |
| Combinado, faixas padrão 30/70 | 0,844 | 0,994 | 0,357 | 0,982 | 0,524 | 55 | 99 | 1 |

Estabilidade (PR-AUC, 5-fold por cliente): regras 0,682 ± 0,074; ML
0,938 ± 0,020; combinado 0,887 ± 0,049.

**O combinado NÃO supera o ML sozinho** — nem no teste nem na validação
cruzada. Ele supera muito as regras sozinhas. A diferença para o ML não é
artefato do arredondamento (sem arredondar e sem piso, o combinado fica em
0,850). Investigando a causa:

- **Horário (principal):** sem as fraudes `horario_atipico`, o combinado
  (0,883) praticamente empata com o ML (0,892). O ML dá ~100 a essas
  fraudes porque explora o artefato do horário; as regras, de propósito,
  tratam horário como sinal fraco (15 pontos) e puxam o combinado para
  baixo justamente nelas.
- **Viajantes:** 24 dos 26 falsos positivos do motor são clientes
  `viajante` legítimos longe de casa. A regra de localização dá a eles os
  mesmos 25 pontos que dá a uma `anomalia_geografica`.

Ou seja: boa parte da vantagem do ML sozinho vem de um artefato dos dados
simulados, que não existiria em dados reais. Por isso o motor mantém 50/50
como padrão; a decisão de dar mais peso ao ML fica registrada como
trade-off, não escondida.

Recall por cenário (teste, sinalizadas / total):

| Cenário | Só regras | Só ML | Combinado |
|---|---|---|---|
| `anomalia_geografica` | 10/11 | 6/11 | **6/11** |
| `combinacao_de_sinais` | 14/14 | 14/14 | 14/14 |
| `dispositivo_novo` | 7/7 | 7/7 | 7/7 |
| `horario_atipico` | 7/11 | 11/11 | 11/11 |
| `valor_atipico` | 7/7 | 7/7 | 7/7 |
| `viagem_impossivel` | 6/6 | 6/6 | 6/6 |

Distribuição por nível (teste, faixas calibradas):

| Nível | Ação | Transações | % | Fraudes | Taxa de fraude no nível | % das fraudes |
|---|---|---|---|---|---|---|
| baixo | aprovar | 2.179 | 96,6% | 5 | 0,2% | 8,9% |
| médio | revisar | 33 | 1,5% | 15 | 45,5% | 26,8% |
| alto | bloquear | 44 | 2,0% | 36 | 81,8% | 64,3% |

As 5 fraudes aprovadas são todas `anomalia_geografica`. A precisão do
bloqueio no teste (81,8%) ficou abaixo da meta de 90% usada na calibração:
com 40 fraudes na validação, o limiar calibrado não generaliza perfeitamente.

## 6. Transação nova

`src/risk/new_transaction.py`: `score_new_transaction(session, engine,
customer_id=…, amount=…, lat=…, lon=…, timestamp=…, device_id=…)`. Calcula
as features a partir do histórico do cliente (transações estritamente
anteriores), reaproveitando as funções puras de `src/synthetic/features.py`,
e chama o engine. Não grava nada. Dispositivo desconhecido conta como não
confiável. Testado: para todas as transações do banco (exceto as que
empatam no mesmo minuto com outra do mesmo cliente), as features batem com
`build_risk_table`.

## 7. Limitações (para defender na banca)

1. **Circularidade dos rótulos.** As fraudes foram geradas manipulando as
   mesmas features que o motor lê; os resultados validam a lógica, não o
   desempenho em fraude real.
2. **`anomalia_geografica` é o ponto cego.** Com as features atuais, uma
   compra em cidade nunca visitada é indistinguível de uma viagem legítima
   de um cliente `viajante`: 5 das 11 escaparam no teste, e 24 dos 26
   falsos positivos são viajantes. A correção natural é uma feature de
   novidade relativa ao histórico ("o cliente já esteve nesta cidade?").
3. **Artefato do horário.** Transações normais nunca saem da janela do
   cliente no gerador, então o ML trata horário como separador perfeito
   (maior coeficiente). Em dados reais esse sinal seria bem mais fraco.
4. **O combinado não supera o ML sozinho** (seção 5), em boa parte por
   causa do item 3.
5. **Amostra pequena.** O teste tem 56 fraudes (6 a 14 por cenário):
   recall por cenário varia muito de um sorteio para outro; a validação
   cruzada mostra a ordem de grandeza da incerteza.
6. **Limiares calibrados com 40 fraudes** não generalizam perfeitamente
   (precisão do bloqueio: 81,8% no teste contra a meta de 90%).
7. **`ml_score` não é probabilidade calibrada** (`class_weight`
   balanceado infla as probabilidades da classe minoritária).
8. **`device_untrusted` é perfeito para `dispositivo_novo`** porque o
   gerador sempre cria esses dispositivos como não confiáveis — outro caso
   de circularidade.
