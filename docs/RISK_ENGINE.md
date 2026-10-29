# Risk Engine — arquitetura, decisões e resultados

> Etapas 7-8 (motor) e 7.5 (novidade de cidade) do Sentinel. Complementa
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
| `risk_score` | 0-100, inteiro, **na escala exibida**: revisa acima de 30, bloqueia acima de 70 (ver seção 3) |
| `raw_score` | 0-100, score interno, comparado com as faixas calibradas |
| `ml_score` | 0-100, score do modelo de Regressão Logística |
| `behavior_score` | 0-100, score das regras explicáveis |
| `level` | `baixo` / `médio` / `alto` |
| `action` | `aprovar` / `revisar` / `bloquear` |
| `reasons` | motivos legíveis, ordenados pela contribuição ao score final |

Exemplo real (transação de teste, rótulo `combinacao_de_sinais`, motor atual):

```
Risk Score 99/100 [interno 99] (ML 100.0, regras 99.2) -> bloquear / revisão urgente
  - Modelo estatístico: padrão semelhante a fraudes conhecidas (score ML 100/100)
  - Viagem impossível: 1.390 km/h entre duas compras
  - Localização incomum: 7.874 km da residência
  - Primeira compra do cliente nesta cidade (Belo Horizonte)
  - Horário fora da janela habitual (3,7 h fora)
```

> **Números atuais:** seção 6.2 (Etapa 14). As seções 2, 3, 5 e 7 registram
> a Etapa 12.

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
| `dataset.py` | `build_risk_table`: `build_feature_table` + `customer_id`, distância em km, `device_untrusted` (`not Device.is_trusted`) e a cidade do estabelecimento (só contexto para os motivos) |
| `behavior_score.py` | 7 regras, pesos/limiares num único `BehaviorConfig` |
| `ml_score.py` | `MLScorer`: padronização + Regressão Logística |
| `engine.py` | `RiskEngine`, `EngineConfig`, `DecisionBands` (com a escala de exibição), `RiskResult`; `combine_scores` é a única implementação da combinação |
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
| Cidade nova | 1ª compra numa cidade fora do histórico (e não a natal) | 20 | O que separa uma fraude geográfica de uma viagem: o viajante volta às mesmas cidades. Mas toda 1ª visita legítima também dispara, então vale o mesmo que dispositivo novo (outro sinal de "primeira vez") |
| Dispositivo novo | aparelho nunca usado pelo cliente | 20 | Típico de tomada de conta, mas pessoas trocam de celular |
| Dispositivo não confiável | `Device.is_trusted = False` | 15 | Somado ao anterior (35), aproxima-se de uma viagem impossível |
| Horário | fora da janela habitual (rampa até 4 h fora) | 5 → 15 | O sinal mais fraco: janelas de horário são estimativas difusas |

Soma limitada a 100. Os pesos foram escolhidos por raciocínio de domínio,
**não ajustados aos dados sintéticos** — ajustá-los repetiria a
circularidade dos rótulos. Todos ficam em `BehaviorConfig` (o peso da cidade
nova também: é ali que vivem os pesos de todas as regras; `EngineConfig`
guarda só os pesos da combinação ML/regras).

## 2. Score de ML

- Regressão Logística com `class_weight='balanced'` (compensa ~3% de
  fraude), padronização das features, seed fixa (42).
- Features: `amount_deviation`, `location_distance`, `unusual_hour`,
  `new_device`, `new_city`, `device_untrusted`, `log1p(implied_speed_kmh)`,
  `is_impossible_travel`.
- Score = probabilidade × 100. Com `class_weight` balanceado essa
  "probabilidade" fica inflada: ela **ordena** bem, mas **não é uma
  probabilidade calibrada de fraude** e não deve ser apresentada como tal.

O que o modelo aprendeu (coeficientes sobre features padronizadas; Etapa 12
— os atuais estão na seção 6.2):

| Feature | Coeficiente |
|---|---|
| `unusual_hour` | +2,51 |
| `amount_deviation` | +1,62 |
| `location_distance` | +1,32 |
| `new_city` | +0,78 |
| `new_device` | +0,77 |
| `device_untrusted` | +0,62 |
| `is_impossible_travel` | +0,60 |
| `log_speed_kmh` | −0,06 |

`unusual_hour` é o maior coeficiente porque, no gerador, nenhuma transação
normal sai da janela de horário: para o modelo, horário é um separador
perfeito. É um artefato dos dados, não conhecimento sobre fraude.
`log_speed_kmh` e `is_impossible_travel` carregam a mesma informação e
dividem o peso entre si (por isso um deles fica perto de zero).

## 3. Combinação, piso, faixas e escala de exibição

> Números da Etapa 12. Os atuais (Etapa 14) estão na seção 6.2.

**Score interno** = peso_ML × ML + peso_regras × regras, arredondado.
`EngineConfig` tem 50/50 como ponto de partida neutro; o motor avaliado usa
o peso escolhido na validação pela regra abaixo: **30% regras / 70% ML**.

**Regra de escolha do peso (declarada antes de olhar o teste):** maximizar o
PR-AUC na **validação**, numa grade de 0% a 100% de peso das regras com passo
de 10%, **com no mínimo 30% de peso nas regras**; em caso de empate, o maior
peso de regras. Implementada em `src/risk/report.py` (`choose_rules_weight`),
então o script reproduz a escolha em vez de ela ser feita à mão.

**Por que o mínimo de 30%:** explicabilidade é objetivo do produto — a
equipe de risco precisa entender cada decisão, e o cliente bloqueado precisa
de uma justificativa. Sem a restrição, a validação escolheria o ML puro
(PR-AUC 0,927), que nestes dados se apoia em grande parte num artefato do
gerador (horário fora da janela nunca ocorre em transação normal; ver
seção 2). Com ≥ 30% vindo das regras, uma parte substancial de todo score
continua saindo de motivos legíveis e de conhecimento de domínio que não
depende dos rótulos sintéticos. O piso de 30% foi fixado por esse critério de
produto, não ajustado aos dados.

Curva de PR-AUC na validação (score combinado contínuo):

| Regras | ML | PR-AUC | |
|---|---|---|---|
| 0% | 100% | 0,927 | fora da restrição |
| 10% | 90% | 0,909 | fora da restrição |
| 20% | 80% | 0,872 | fora da restrição |
| **30%** | **70%** | **0,837** | **escolhido** |
| 40% | 60% | 0,817 | |
| 50% | 50% | 0,803 | ponto de partida anterior |
| 60% | 40% | 0,793 | |
| 70% | 30% | 0,784 | |
| 80% | 20% | 0,775 | |
| 90% | 10% | 0,727 | |
| 100% | 0% | 0,651 | |

A curva cai de forma monotônica com o peso das regras, então quem decide é
a restrição: o peso escolhido é o mínimo permitido de regras.

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

Resultado da calibração (score interno, peso 30/70): **baixo 0-73, médio
74-81, alto 82-100**. Com 50/50 seria 0-56, 57-71, 72-100 (e, antes da
feature de cidade, 0-54, 55-61, 62-100). Os limiares mudam com o peso
porque o score combinado muda de distribuição; por isso são sempre
recalibrados para o peso em uso.

**Escala de exibição:** para quem opera, o `risk_score` exibido remapeia os
limiares calibrados para **30 e 70** por uma transformação monotônica por
partes (`DecisionBands.to_display`): [0, 73] → [0, 30], (73, 81] → [31, 70],
(81, 100] → [71, 100]. Nível e ação continuam decididos pelo score interno;
a transformação só muda o número exibido. Isso é testado exaustivamente:
para centenas de combinações de faixas e todo score interno de 0 a 100, o
nível lido na escala exibida é idêntico ao do score interno. As métricas de
ranking do relatório usam o score interno (a escala exibida comprime
trechos e criaria empates artificiais).

## 4. Metodologia de avaliação

- **Split por cliente** (`GroupShuffleSplit`), nunca por transação:
  transações do mesmo cliente compartilham perfil; separá-las deixaria o
  modelo reconhecer no teste um cliente visto no treino.
- Três conjuntos de clientes, sem sobreposição: treino (64 clientes, 3.311
  transações, 111 fraudes) ajusta o ML; validação (27 / 1.345 / 40) escolhe
  o peso (pela regra da seção 3) e calibra os limiares; teste
  (39 / 2.256 / 56) é tocado uma única vez, para o relatório.
- Seed fixa (42) em tudo.
- Métricas: PR-AUC (principal, pelo desbalanceamento), ROC-AUC, precisão,
  recall, F1, matriz de confusão e recall por cenário com contagens.
- **Estabilidade:** como o teste tem só 56 fraudes (6 a 14 por cenário),
  o relatório repete o PR-AUC em validação cruzada 5-fold estratificada por
  cliente sobre a tabela inteira.

Para reproduzir: `python scripts/run_risk_engine.py`.

## 5. Resultados (teste, 39 clientes nunca vistos)

> Números da Etapa 12. Os atuais (Etapa 14) estão na seção 6.2.

> **Protótipo demonstrativo.** Os números desta seção e da 6.1 vêm do domínio
> sintético, cujos rótulos foram gerados com os mesmos sinais que o motor usa:
> validam a lógica do sistema, não o desempenho contra fraude real. Ver
> [`VALIDACAO.md`](VALIDACAO.md).

"Sinalizada" = não aprovada automaticamente (nível médio ou alto). Cada
abordagem tem seus limiares calibrados na validação com o mesmo critério.

| Abordagem | PR-AUC | ROC-AUC | Precisão | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| Só regras | 0,641 | 0,972 | 0,169 | 0,929 | 0,286 | 52 | 256 | 4 |
| Só ML | **0,929** | 0,997 | **0,731** | 0,875 | **0,797** | 49 | 18 | 7 |
| **Combinado (motor, 30% regras / 70% ML)** | 0,804 | 0,994 | 0,676 | 0,893 | 0,769 | 50 | 24 | 6 |
| Combinado 50/50 | 0,787 | 0,993 | 0,634 | 0,929 | 0,754 | 52 | 30 | 4 |

Estabilidade (PR-AUC, 5-fold por cliente): regras 0,688 ± 0,065; ML
0,953 ± 0,016; **motor 0,864 ± 0,040**; 50/50 0,845 ± 0,043.

**Peso escolhido vs 50/50:** o motor melhora no ranking (PR-AUC 0,787 →
0,804; validação cruzada 0,845 → 0,864), em precisão (0,634 → 0,676) e em
F1 (0,754 → 0,769), com menos falsos positivos (30 → 24). O custo é recall:
**2 fraudes a menos** (52 → 50 de 56), ambas `anomalia_geografica`
(9 → 7 de 11), e a precisão do bloqueio cai (89,7% → 83,9%, abaixo da meta
de 90% usada na validação). Um exemplo concreto do custo, tirado do próprio
relatório: uma compra de R$ 103,83 em Natal, de um cliente `fixo` de São
Paulo, é fraude `anomalia_geografica`; com ML 83 e regras 45, o score interno
fica em 72 — logo abaixo do limiar de revisão (73) — e ela é aprovada. A
50/50, teria ido para revisão.

**O combinado continua NÃO superando o ML sozinho** — nem no teste, nem na
validação cruzada, nem na validação. Isso é esperado e foi aceito
deliberadamente pela restrição de 30%: a diferença para o ML se explica por
dois grupos de transações. Sem as fraudes `horario_atipico` e sem as
primeiras visitas legítimas a uma cidade, o combinado 50/50 (0,974) empata
com o ML (0,972).

- **Horário:** o ML dá ~100 às fraudes `horario_atipico` porque explora o
  artefato do horário; as regras, de propósito, tratam horário como sinal
  fraco (15 pontos) e puxam o combinado para baixo justamente nelas.
- **Primeiras visitas legítimas:** com estas features, a primeira visita
  de um viajante a uma cidade é indistinguível de uma fraude geográfica (as
  duas ficam com 45 pontos nas regras e ~90 no ML).

Recall por cenário (teste, sinalizadas / total):

| Cenário | Só regras | Só ML | Motor (30/70) | 50/50 |
|---|---|---|---|---|
| `anomalia_geografica` | 11/11 | 4/11 | 7/11 | 9/11 |
| `combinacao_de_sinais` | 14/14 | 14/14 | 14/14 | 14/14 |
| `dispositivo_novo` | 7/7 | 7/7 | 7/7 | 7/7 |
| `horario_atipico` | 7/11 | 11/11 | 9/11 | 9/11 |
| `valor_atipico` | 7/7 | 7/7 | 7/7 | 7/7 |
| `viagem_impossivel` | 6/6 | 6/6 | 6/6 | 6/6 |

Distribuição por nível (teste, motor com faixas calibradas):

| Nível | Ação | Transações | % | Fraudes | Taxa de fraude no nível | % das fraudes |
|---|---|---|---|---|---|---|
| baixo | aprovar | 2.182 | 96,7% | 6 | 0,3% | 10,7% |
| médio | revisar | 43 | 1,9% | 24 | 55,8% | 42,9% |
| alto | bloquear | 31 | 1,4% | 26 | 83,9% | 46,4% |

## 6. Etapa 7.5 — novidade de cidade (`new_city`)

**Motivação:** a primeira avaliação mostrou um ponto cego geográfico (5 de
11 fraudes `anomalia_geografica` escaparam; 24 dos 26 falsos positivos eram
viajantes), porque nenhuma feature sabia se o cliente já tinha estado na
cidade da compra.

**Feature:** `new_city_flag` (`src/synthetic/features.py`), calculada em
ordem cronológica por cliente em `build_feature_table`, com a mesma lógica
de `new_device`: a cidade do estabelecimento já apareceu no histórico
ANTERIOR do cliente? A cidade natal é sempre conhecida; sem histórico
anterior, `False`. Entra no ML e numa regra nova de 20 pontos.

**Antes vs depois (teste)** — números com o combinado a 50/50, o peso em
uso nesta etapa (a regra de escolha de pesos da seção 3 veio depois):

| Abordagem | PR-AUC | Recall | Precisão | F1 | FP | FP de viajantes |
|---|---|---|---|---|---|---|
| Só regras | 0,606 → 0,641 | 0,911 → 0,929 | 0,166 → 0,169 | 0,281 → 0,286 | 256 → 256 | 215 → 215 |
| Só ML | 0,924 → **0,929** | 0,911 → 0,875 | 0,689 → 0,731 | 0,785 → 0,797 | 23 → 18 | 22 → 18 |
| Combinado (motor) | 0,842 → **0,787** | 0,911 → 0,929 | 0,662 → 0,634 | 0,767 → 0,754 | 26 → 30 | 24 → 28 |

Validação cruzada (PR-AUC): regras 0,682 → 0,688; ML 0,938 → 0,953;
combinado 0,887 → **0,845**.

`anomalia_geografica`: regras 10/11 → 11/11; ML 6/11 → **4/11**; combinado
6/11 → **9/11**.

**O que melhorou:**

- O problema que motivou a feature foi resolvido: falsos positivos de
  viajantes em cidades **já conhecidas** caíram de 14 para 4 no combinado.
- O combinado pega mais fraudes geográficas (6 → 9 de 11) e mais fraudes no
  total (51 → 52), e o bloqueio ficou mais preciso (81,8% → 89,7%).
- O ML sozinho melhorou no ranking (PR-AUC 0,924 → 0,929; validação
  cruzada 0,938 → 0,953) com menos falsos positivos (23 → 18).

**O que piorou (dito sem rodeios):**

- **O motor combinado piorou no ranking** (PR-AUC 0,842 → 0,787; validação
  cruzada 0,887 → 0,845) e passou a ter mais falsos positivos (26 → 30;
  de viajantes, 24 → 28). A regra soma 20 pontos a TODA primeira visita
  legítima, levando todas elas para cima do limiar: os falsos positivos de
  viajantes em primeira visita subiram de 10 para 24. A natureza do falso
  positivo mudou — de "viagem a cidade conhecida" para "primeira visita a
  uma cidade", o caso em que um banco real também costuma pedir
  confirmação —, mas o número subiu.
- O combinado perdeu 2 fraudes `horario_atipico` (11 → 9 de 11): o limiar
  de revisão calibrado subiu (54 → 56).
- O ML sozinho, no limiar calibrado, pega **menos** fraudes geográficas
  (6 → 4 de 11) e menos fraudes no total (51 → 49).

**Circularidade parcial do ganho:** o gerador cria `anomalia_geografica`
sorteando, por construção, uma cidade fora das cidades normais do cliente
(`src/synthetic/generator.py`, `inject_geo_anomaly`) — ou seja, uma cidade
que, na maioria dos casos, ele nunca visitou. A feature `new_city` mede
quase exatamente a regra que gerou o rótulo (32 das 35 fraudes geográficas
da base têm `new_city = True`). Parte do ganho em `anomalia_geografica`
é, portanto, circular: em dados reais, fraudes geográficas também
acontecem em cidades que o cliente já visitou.

## 6.1 Etapa 12 — canal e forma de pagamento

**Dados:** cada transação tem `channel` (`online` ou `presencial`) e, se
presencial, `payment_method` (`aproximacao` ou `senha`). Cada cliente tem
um hábito de canal (`online_share`, `contactless_share`). Normais seguem o
hábito; acima de R$ 200 a compra presencial sempre exige senha. Fraudes
seguem a lógica do golpe: 25% são cartão clonado com senha capturada
(presencial com senha); as demais são online (acima de R$ 200, metade
abaixo disso) ou por aproximação; "dispositivo novo" é sempre online. O
canal é sorteado por um gerador próprio, então as transações das etapas
anteriores não mudam.

**Features:** `is_online`, `is_contactless` e `channel_rarity` (1 − fração
das compras ANTERIORES do cliente no mesmo canal; 0 sem histórico). Entram
no ML e em duas regras: canal raro (5 a 15 pontos a partir de 0,7) e
aproximação com valor ≥ 1,5x a média (10 pontos).

**O ML não vê o rótulo:** `tests/test_risk_label_leakage.py` garante que a
matriz de entrada só tem `ML_FEATURES` (sem `is_fraud`/`fraud_scenario`), que
o score não muda se o rótulo for invertido ou removido, e que as features
são idênticas com os rótulos do banco invertidos.

**Resultado (teste, mesma metodologia):**

| | antes | depois |
|---|---|---|
| Motor: fraudes capturadas | 50/56 (89%) | 53/56 (95%) |
| Motor: falsos positivos | 24 | 23 |
| Motor: PR-AUC | 0,804 | 0,825 |
| Só regras: PR-AUC | 0,641 | 0,678 |
| Só ML: PR-AUC | 0,929 | 0,929 |

O ganho vem quase todo de `horario_atipico` (9 → 10 de 11) e de
`anomalia_geografica` (7 → 9 de 11). Mesma ressalva de circularidade: o
canal das fraudes foi gerado com uma regra, e o modelo a aprende. No ML,
`channel_rarity` tem coeficiente negativo por colinearidade com
`is_online` — o efeito conjunto de "online raro" continua positivo.

## 6.2 Etapa 14 — hábitos pelo histórico e ruído legítimo (números atuais)

> **Os números desta seção são os do motor atual.** As seções 2, 3, 5 e 7
> registram a Etapa 12, antes desta mudança.

**O que mudou** (detalhes em [`DOMINIO_SINTETICO.md`](DOMINIO_SINTETICO.md)):

1. `amount_deviation` e `unusual_hour` comparam a transação com a média e a
   janela de horário **estimadas das compras anteriores** (`HabitProfile`),
   e não com os parâmetros que o gerador usou (`Customer.avg_amount_baseline`,
   `typical_hour_start/end`). Um sistema real só teria o histórico.
2. **Ruído legítimo** no gerador: compras normais às vezes fora do horário
   (4%), bem acima da média (3%), e 25% dos clientes trocam de celular.
3. Metade das fraudes (fora `dispositivo_novo`) usa o **canal habitual** do
   titular.
4. A Central de Transações mostra **o que o modelo de ML aprendeu** (os
   coeficientes) e o hábito estimado com que cada transação foi comparada.

**Por quê:** antes, várias features separavam fraude de legítima sozinhas
(nenhuma compra normal saía da janela de horário, e o hábito era o
"verdadeiro" do gerador). O ML acertava lendo o gerador, não aprendendo.

**Antes × depois (teste, 39 clientes nunca vistos, seed 42):**

| | Etapa 12 | Etapa 14 |
|---|---|---|
| Peso escolhido na validação | 30% regras / 70% ML | **70% regras / 30% ML** |
| Faixas internas (revisar / bloquear acima de) | 73 / 82 | 18 / 79 |
| Só regras: PR-AUC | 0,678 | 0,629 |
| Só ML: PR-AUC | 0,929 | 0,754 |
| Motor: PR-AUC | 0,825 | 0,694 |
| Combinado 50/50: PR-AUC | 0,803 | 0,709 |
| Motor: fraudes capturadas | 53/56 | **55/56** |
| Motor: falsos positivos | 23 | **258** |
| Motor: precisão | 0,697 | 0,176 |
| Validação cruzada (PR-AUC): ML | 0,950 ± 0,012 | 0,720 ± 0,032 |
| Validação cruzada (PR-AUC): motor | 0,887 ± 0,031 | 0,720 ± 0,050 |

Distribuição por nível (teste, motor atual):

| Nível | Ação | Transações | % | Fraudes | Taxa de fraude no nível | % das fraudes |
|---|---|---|---|---|---|---|
| baixo | aprovar | 1.943 | 86,1% | 1 | 0,1% | 1,8% |
| médio | revisar | 299 | 13,3% | 41 | 13,7% | 73,2% |
| alto | bloquear | 14 | 0,6% | 14 | 100,0% | 25,0% |

Recall por cenário (teste): todas as abordagens pegam 100% de
`combinacao_de_sinais`, `dispositivo_novo`, `valor_atipico` e
`viagem_impossivel`. Em `horario_atipico`, só regras pega **0/11**, o ML
11/11 e o motor 10/11; em `anomalia_geografica`, regras e motor 11/11, ML
10/11.

O que o ML aprendeu (coeficientes sobre features padronizadas):
`unusual_hour` +0,96, `amount_deviation` +0,95, `is_online` +0,91,
`new_city` +0,74, `channel_rarity` −0,73, `new_device` +0,60,
`location_distance` +0,46, `log_speed_kmh` +0,28, `device_untrusted` +0,27,
`is_impossible_travel` +0,26, `is_contactless` +0,06. Antes, `unusual_hour`
dominava (+2,7, mais que o dobro do segundo); agora horário e valor pesam o
mesmo, e o canal entra logo atrás.

**Como ler:**

- **O problema ficou mais difícil, e os números caíram para um patamar mais
  honesto.** O PR-AUC do ML caiu de 0,93 para 0,75. A queda é o efeito
  esperado de tirar os atalhos do gerador, não uma piora do motor.
- **O motor captura mais fraudes, ao custo de muitas revisões.** A meta da
  calibração (revisar ≥ 90% das fraudes na validação) agora exige um limiar
  de revisão baixo (18): 13% das transações vão para revisão, contra 2%
  antes. Num banco real, isso decide o tamanho da equipe; o limiar deveria
  ser escolhido por custo (ver `VALIDACAO.md`).
- **O peso escolhido mudou para 70% regras.** A curva de PR-AUC na validação
  ficou quase plana entre 30% e 80% de regras (0,619 a 0,627), e a regra de
  desempate favorece as regras. Nenhum peso se destaca; com 40 fraudes na
  validação, a escolha é frágil (no teste, o 50/50 teria ficado um pouco
  melhor).
- **O ML sozinho continua melhor que o motor no ranking**, mas a diferença
  caiu (0,754 contra 0,694; na validação cruzada, empate em 0,720).
- **Regras não pegam horário atípico** (0/11): horário vale no máximo
  15 pontos, de propósito, e 17% das compras legítimas com histórico já saem
  um pouco da janela estimada. O ML combina o horário com outros sinais e
  pega todas.
- **Gradient Boosting no lugar da Regressão Logística** foi testado fora do
  motor: PR-AUC 0,734 / 0,787 / 0,791 contra 0,754 / 0,679 / 0,617 da
  Regressão Logística nas seeds 42 / 7 / 123. Ele tende a ser melhor, mas
  varia muito com tão poucas fraudes, e perderia os coeficientes legíveis.
  O motor mantém a Regressão Logística; a troca fica como decisão do time.

**Cenários do Simulador** (todos os 130 clientes, motor atual, faixas
internas 18 e 79):

| Cenário | Score interno (mín–máx) | Risk Score exibido | Resultado nos 130 clientes |
|---|---|---|---|
| Compra normal | 0–4 | 0–6 | 130 baixo (aprovar) |
| Viagem impossível | 69–100 | 64–100 (mediana 85) | 125 alto, 5 médio |
| Valor atípico | 47 | 50 | 130 médio (revisar) |
| Horário atípico | 31–40 | 39–45 | 130 médio (revisar) |
| Dispositivo novo | 52–63 | 53–60 | 130 médio (revisar) |
| Cidade nova | 53–61 | 53–59 | **130 médio** (antes, 59 aprovados) |
| Combinação de sinais | 82–92 | 75–89 | 130 alto (bloquear) |

Os sinais isolados deixaram de ficar rente aos limiares (limitação 9): todos
revisam com folga de pelo menos 13 pontos internos, e a combinação bloqueia
nos 130 clientes.

## 7. Transação nova

`src/risk/new_transaction.py`: `score_new_transaction(session, engine,
customer_id=…, amount=…, lat=…, lon=…, timestamp=…, device_id=…, city=None)`.
Calcula as features a partir do histórico do cliente (transações
estritamente anteriores), reaproveitando as funções puras de
`src/synthetic/features.py`, e chama o engine. Não grava nada. Dispositivo
desconhecido conta como não confiável. Sem `city`, a cidade é a de
referência mais próxima das coordenadas. Testado: para todas as transações
do banco (exceto as que empatam no mesmo minuto com outra do mesmo
cliente), as features — inclusive `new_city` e a cidade inferida — batem
com `build_risk_table`.

### Resultados dos cenários de demonstração (Simulador)

> Números da Etapa 12. Os atuais (Etapa 14) estão na seção 6.2.

O Simulador (`app/views/simulator.py`) tem sete cenários que só preenchem o
formulário, com valores calculados do histórico do cliente escolhido; quem
classifica é o motor. Desde a Etapa 12, cada cenário usa o **canal habitual
do cliente** (o par canal/forma de pagamento mais frequente no histórico;
aproximação só até R$ 200), para mudar só o sinal que promete — exceto os
cenários com **dispositivo novo**, que são **online**, como no gerador do
banco. Rodando cada cenário para **todos os 130 clientes** com o motor atual
(30% regras / 70% ML; faixas internas 73 e 82):

| Cenário | O que preenche | Score interno (mín–máx) | Risk Score exibido | Resultado nos 130 clientes |
|---|---|---|---|---|
| Compra normal | cidade natal, valor médio, horário habitual, dispositivo e canal habituais | 0 | 0 | 130 baixo (aprovar) |
| Viagem impossível | cidade mais distante da última compra, 20 min depois | 74–100 | 35–100 (mediana 92) | 129 alto (bloquear), 1 médio |
| Valor atípico | 10x o valor médio | 77 | 48 | 130 médio (revisar) |
| Horário atípico | lado oposto da janela habitual | 74–75 | 35–39 | 130 médio (revisar) |
| Dispositivo novo | "Novo dispositivo, não reconhecido", online | 73–78 | 30–53 (mediana 39) | 125 médio, **5 baixo** |
| Cidade nova | cidade nunca visitada, ~2.000 km da residência | 52–83 | 21–72 (mediana 35) | **59 baixo**, 47 médio, 24 alto |
| Combinação de sinais | dispositivo novo (online) + horário do lado oposto da janela + 10x o valor médio | 93–97 | 89–95 | **130 alto (bloquear)** |

O que a equipe precisa saber antes da banca:

- **Cidade nova isolada é aprovada em 59 dos 130 clientes.** Antes da Etapa
  12 eram 2. Nos aprovados a compra é presencial com senha (o canal habitual
  deles), e o ML, que agora lê o canal, dá a ela menos risco (mediana 82,
  contra 97 nos revisados): nos dados, só 25% das fraudes usam a senha
  (cartão clonado). As regras dão 45 nos dois casos. É o comportamento real
  do modelo, e o mesmo dilema da limitação 2: uma primeira visita legítima e
  uma fraude geográfica com senha têm os mesmos sinais. Na avaliação com as
  fraudes geradas (seção 6.1), `anomalia_geografica` subiu de 7 para 9 de 11.
- **Dispositivo novo, sozinho, agora vai para revisão** (125 de 130; antes,
  130 aprovados). O cenário passou a ser online, como as fraudes desse tipo
  no banco; presencial com senha, o ML o aprovaria.
- **Horário atípico continua revisado por 1 a 2 pontos** (74–75 contra 73).
  É frágil: uma calibração um pouco diferente dos limiares o aprovaria.
- **Só a viagem impossível chega a "alto" entre os sinais isolados** (mais
  algumas cidades novas). Valor e horário, isolados, vão para revisão. O
  único caso "médio" de viagem impossível tem score interno 74 (revisa, não
  bloqueia).
- **A combinação de sinais chega a "alto" nos 130 clientes** (score interno
  93–97, limiar de bloqueio 82). Isoladamente, os mesmos sinais revisam ou
  aprovam: é a soma que muda a ação. Parte do 100 do ML vem do artefato do
  horário (limitação 3), então a folga não mede robustez em dados reais.

Esses números são o comportamento real do motor com dados sintéticos; não
foram ajustados para que os cenários "dêem certo".

## 8. Limitações (para defender na banca)

1. **Circularidade dos rótulos.** As fraudes foram geradas manipulando as
   mesmas features que o motor lê; os resultados validam a lógica, não o
   desempenho em fraude real. Vale também para `new_city`, que mede quase
   exatamente a regra que gerou `anomalia_geografica` (seção 6).
2. **Primeira visita legítima = fraude geográfica.** Com as features
   atuais, a primeira visita de um viajante a uma cidade é indistinguível
   de uma fraude geográfica; é a maior fonte de falsos positivos do motor
   (24 dos 30 no teste).
3. **Artefato do horário — reduzido na Etapa 14.** Até a Etapa 12,
   transações normais nunca saíam da janela do cliente e o ML tratava o
   horário como separador perfeito. Agora 4% das compras legítimas caem num
   horário qualquer e a janela é estimada do histórico; o horário deixou de
   dominar os coeficientes (seção 6.2).
4. **O combinado nem sempre supera o ML sozinho.** Na seed 42, o ML puro
   tem PR-AUC 0,754 e o motor 0,694; em outras seeds o motor fica à frente
   (item 6). A regra de escolha exige no mínimo 30% de regras, por
   explicabilidade: é uma escolha consciente de produto, não o que os
   números sozinhos recomendam.
5. **Amostra pequena.** O teste tem 56 fraudes (6 a 14 por cenário):
   recall por cenário varia muito de um sorteio para outro; a validação
   cruzada mostra a ordem de grandeza da incerteza.
6. **Peso e limiares mudam com a seed.** Calibrados com ~40 fraudes, eles
   não generalizam perfeitamente. Mesmo gerador, 5 seeds (teste):

   | seed | peso das regras | revisar acima de | bloquear acima de | PR-AUC regras | PR-AUC ML | PR-AUC motor | recall motor | FP motor |
   |---|---|---|---|---|---|---|---|---|
   | 42 | 70% | 18 | 79 | 0,629 | 0,754 | 0,694 | 98% | 258 |
   | 7 | 40% | 35 | 80 | 0,625 | 0,679 | 0,739 | 97% | 111 |
   | 123 | 60% | 29 | 79 | 0,662 | 0,617 | 0,746 | 85% | 86 |
   | 2024 | 70% | 22 | 65 | 0,556 | 0,710 | 0,655 | 95% | 174 |
   | 99 | 30% | 24 | 87 | 0,705 | 0,706 | 0,747 | 90% | 189 |

   O peso escolhido vai de 30% a 70% de regras, e o limiar de revisão de
   18 a 35. O que se mantém: o motor fica entre 0,65 e 0,75 de PR-AUC e
   supera as regras sozinhas em 4 das 5 seeds. Os números da seed 42 não
   são "o" resultado do motor, são um sorteio.
7. **`ml_score` não é probabilidade calibrada** (`class_weight`
   balanceado infla as probabilidades da classe minoritária).
8. **`dispositivo_novo` continua fácil de separar.** O gerador sempre cria
   esses dispositivos como não confiáveis e a compra é sempre online. Desde
   a Etapa 14 há clientes legítimos com celular novo e não confiável, mas a
   combinação "aparelho novo + não confiável + online" segue quase
   exclusiva da fraude — outro caso de circularidade.
9. **Folga dos sinais isolados depende da calibração.** Até a Etapa 12,
   dispositivo novo era aprovado por até 2 pontos e horário atípico revisado
   por 1. Com as faixas da Etapa 14 (18 e 79), todos os cenários isolados do
   Simulador têm folga (seção 6.2). Mas essas faixas vêm de ~40 fraudes e
   mudam com a seed (item 6): com o limiar de revisão 35 da seed 7, o
   horário atípico (31–40) voltaria a ficar rente.
10. **Muitas revisões (Etapa 14).** Com os dados mais realistas, revisar
    90% das fraudes exige mandar 13% das transações para revisão humana
    (seção 6.2). A meta de recall foi mantida; num banco real, o limiar
    seria escolhido pelo custo de revisão.
11. **Compra online usa a localização do estabelecimento.** Distância e
    velocidade de uma compra online vêm das coordenadas do lojista. Por isso
    uma fraude geográfica online "revela" a cidade do fraudador, o que não
    acontece de verdade (num banco real seria a geolocalização do IP, que é
    imprecisa e fácil de mascarar). É uma premissa do domínio sintético.
