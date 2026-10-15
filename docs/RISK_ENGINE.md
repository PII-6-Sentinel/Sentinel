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

Exemplo real (transação de teste, rótulo `combinacao_de_sinais`):

```
Risk Score 100/100 [interno 100] (ML 100.0, regras 100.0) -> bloquear / revisão urgente
  - Modelo estatístico: padrão semelhante a fraudes conhecidas (score ML 100/100)
  - Viagem impossível: 1.390 km/h entre duas compras
  - Localização incomum: 7.874 km da residência
  - Primeira compra do cliente nesta cidade (Belo Horizonte)
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

O que o modelo aprendeu (coeficientes sobre features padronizadas):

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

## 8. Limitações (para defender na banca)

1. **Circularidade dos rótulos.** As fraudes foram geradas manipulando as
   mesmas features que o motor lê; os resultados validam a lógica, não o
   desempenho em fraude real. Vale também para `new_city`, que mede quase
   exatamente a regra que gerou `anomalia_geografica` (seção 6).
2. **Primeira visita legítima = fraude geográfica.** Com as features
   atuais, a primeira visita de um viajante a uma cidade é indistinguível
   de uma fraude geográfica; é a maior fonte de falsos positivos do motor
   (24 dos 30 no teste).
3. **Artefato do horário.** Transações normais nunca saem da janela do
   cliente no gerador, então o ML trata horário como separador perfeito
   (maior coeficiente). Em dados reais esse sinal seria bem mais fraco.
4. **O combinado não supera o ML sozinho** no ranking (seção 5). A
   validação, sem restrição, escolheria o ML puro; o motor usa 30% regras /
   70% ML porque a regra de escolha exige no mínimo 30% de regras, por
   explicabilidade. É uma escolha consciente de produto, não o que os
   números sozinhos recomendam — e o peso escolhido aprova 2 fraudes
   geográficas que o 50/50 mandaria para revisão.
5. **Amostra pequena.** O teste tem 56 fraudes (6 a 14 por cenário):
   recall por cenário varia muito de um sorteio para outro; a validação
   cruzada mostra a ordem de grandeza da incerteza.
6. **Limiares calibrados com 40 fraudes** não generalizam perfeitamente.
7. **`ml_score` não é probabilidade calibrada** (`class_weight`
   balanceado infla as probabilidades da classe minoritária).
8. **`device_untrusted` é perfeito para `dispositivo_novo`** porque o
   gerador sempre cria esses dispositivos como não confiáveis — outro caso
   de circularidade.
