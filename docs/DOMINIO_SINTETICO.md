# Domínio Sintético — justificativa metodológica

> Complementa [`docs/SWOT_GUT.md`](SWOT_GUT.md). Enquanto aquele documento
> cobre o planejamento do projeto como um todo, este cobre especificamente
> a segunda via de dados do Sentinel: a simulação operacional.

## Por que dois datasets

O Sentinel passa a operar em **duas vias complementares**, não concorrentes:

| | Kaggle (Credit Card Fraud Detection) | Domínio sintético |
|---|---|---|
| Papel | Validação científica dos modelos | Simulação de um ambiente antifraude operacional |
| Origem | Dataset público, real, rotulado por especialistas | Gerado por regras explícitas, com seed fixa |
| Features | `V1`...`V28` anonimizadas via PCA — sem significado de negócio | Cliente, cartão, dispositivo, estabelecimento, geolocalização — com significado de negócio |
| Serve para | Comparar estatística vs. ML com rigor (métricas, validação cruzada) | Demonstrar como um sistema antifraude "enxergaria" clientes, transações e contexto (cidade, horário, dispositivo) |

O Kaggle prova que os modelos funcionam sobre um benchmark aceito pela
comunidade. Sozinho, porém, ele não deixa demonstrar nada que dependa de
**contexto de negócio** — não dá pra mostrar "essa transação é suspeita
porque é longe de casa e fora do horário do cliente", porque `V1`...`V28`
não significam isso para quem olha o dataset. O domínio sintético existe
para preencher exatamente essa lacuna, sem inventar dados de forma
arbitrária: cada campo e cada cenário de fraude tem uma regra de geração
documentada e reproduzível (este arquivo).

**Isto NÃO substitui o Kaggle.** As próximas etapas do projeto continuam
usando os dois: o Kaggle para a avaliação estatística rigorosa já
construída (score, threshold, AUC-PR, validação cruzada), e o domínio
sintético para a camada operacional (Risk Engine, Central de Transações)
que virá depois.

## Por que dados sintéticos, e não mais dados reais

- Não existe um segundo dataset público de fraude com metadados de
  cliente/cidade/dispositivo tão granulares quanto o que o projeto precisa
  para a próxima fase — datasets reais desse tipo são, com razão,
  confidenciais (dados financeiros pessoais).
- Gerar sinteticamente permite **controlar o que é fraude e por quê** —
  cada transação fraudulenta tem uma causa explícita e documentada (ver
  "Cenários de fraude" abaixo), o que facilita explicar o comportamento
  esperado de qualquer modelo/regra rodando sobre esses dados.
- Sem dados pessoais reais envolvidos: nomes são combinações aleatórias de
  nomes/sobrenomes comuns (não correspondem a pessoas reais), e as únicas
  coordenadas geográficas reais usadas são de **cidades** (não de endereços
  de pessoas) — ver `src/synthetic/geo_reference.py`.

## Como foi gerado — nada é "aleatório sem critério"

Tudo em `src/synthetic/generator.py` passa por uma única fonte de
aleatoriedade, `random.Random(seed=42)` — a mesma seed sempre produz o
mesmo dataset, byte a byte (testado em
[`tests/test_synthetic_data.py`](../tests/test_synthetic_data.py),
`test_generation_is_deterministic`). Isso permite reproduzir e auditar
qualquer transação específica: dado o mesmo seed e os mesmos parâmetros,
rodar `scripts/generate_mock_data.py` de novo gera exatamente o mesmo banco.

As coordenadas geográficas usadas (`src/synthetic/geo_reference.py`) são de
cidades **reais** (a maioria capitais brasileiras, mais algumas
internacionais), para que distâncias e velocidades calculadas
(`geopy.distance.geodesic`) sejam fisicamente corretas — sem isso, o
cenário de "viagem impossível" (que depende de uma velocidade implícita
km/h de verdade) não teria fundamento.

## Entidades

| Entidade | O que representa |
|---|---|
| `Customer` | Um cliente — cidade natal, perfil comportamental, valor médio de gasto de referência, janela de horário típico |
| `Card` | Um cartão vinculado ao cliente (débito/crédito) |
| `Device` | Um dispositivo usado pelo cliente (mobile/desktop/POS), com uma flag de confiança |
| `Merchant` | Um estabelecimento comercial, localizado em uma das cidades de referência |
| `Transaction` | Uma transação — liga cliente, cartão, dispositivo e estabelecimento, com valor, horário, localização e rótulo de fraude |

Ver [`src/synthetic/models.py`](../src/synthetic/models.py) para o schema
completo (SQLAlchemy).

## Perfis comportamentais

Cada cliente recebe um perfil na criação, que molda como sua atividade
"normal" é gerada — é o que dá sentido a "anomalia": uma transação só é
estranha em relação ao padrão do PRÓPRIO cliente, não a um limiar global.

| Perfil | % dos clientes | Comportamento normal |
|---|---|---|
| `fixo` | ~50% | Transações concentradas na cidade natal; valor e horário estáveis |
| `viajante` | ~25% | Além da cidade natal, conhece 2-3 outras cidades. Segue um **itinerário em blocos contínuos**: fica de 1 a 4 semanas em casa, viaja para uma cidade que conhece e fica de 3 a 14 dias, depois volta para casa ou vai para outra. Entre duas estadias há sempre 1 dia inteiro sem transações (o deslocamento). Transações nessas cidades são NORMAIS para este cliente, não anomalias |
| `alto_gasto` | ~15% | Mesmo padrão geográfico de um cliente fixo, mas com valor médio de referência bem mais alto (R$500–3.000 vs. R$50–300) |
| `novo_cliente` | ~10% | Poucas transações no histórico (8-15, contra 30-80 dos demais) — ainda não tem um padrão bem estabelecido |

## Cenários de fraude

Cada transação fraudulenta carrega o campo `fraud_scenario` explicando por
que foi marcada — nunca "fraude" sem justificativa. Meta: ~3% das
transações totais, dividida o mais igualmente possível entre os 6
cenários.

### 1. Anomalia geográfica (`anomalia_geografica`)
Transação em uma cidade que o cliente **nunca** frequenta — nem a cidade
natal, nem (para viajantes) o conjunto de cidades que costuma visitar.
Valor e horário continuam normais — só a geografia destoa.

### 2. Viagem impossível (`viagem_impossivel`)
Duas transações consecutivas do mesmo cliente, distantes o bastante e
próximas o bastante no tempo para implicar uma velocidade **acima de
900 km/h** (velocidade comercial de cruzeiro de avião) entre elas —
fisicamente incompatível com qualquer meio de transporte real disponível
ao público. A distância e o intervalo de tempo são escolhidos juntos, de
forma que a velocidade implícita sempre exceda o limiar por uma margem
confortável (testado explicitamente em
`test_impossible_travel_scenario_violates_max_plausible_speed`).

### 3. Valor atípico (`valor_atipico`)
Valor entre 6x e 15x a média histórica **daquele cliente específico**
(não um valor absoluto fixo — R$ 5.000 é normal para um cliente
`alto_gasto` e muito estranho para um `fixo` com baseline de R$ 80).
Cidade e horário permanecem normais.

### 4. Horário atípico (`horario_atipico`)
Transação fora da janela de horário típica **daquele cliente** (ex.: um
cliente que só transaciona entre 9h e 19h tendo uma transação às 3h da
manhã). Cidade e valor permanecem normais.

### 5. Combinação de sinais (`combinacao_de_sinais`)
Três sinais **moderados** ao mesmo tempo — cidade incomum (mas não
necessariamente nunca visitada), valor de 2,5x a 4,5x a média (bem menos
extremo que o cenário 3) e horário um pouco fora da janela típica (menos
extremo que o cenário 4). Nenhum sinal isolado seria extremo o bastante
para disparar os cenários de sinal único — é o cenário desenhado para ser mais
difícil de capturar com uma única regra estatística simples, motivando
diretamente por que o projeto compara regras com Machine Learning.

### 6. Dispositivo novo (`dispositivo_novo`)
Transação feita com um dispositivo que **nunca apareceu** no histórico do
cliente e que não é confiável (`Device.is_trusted = False`). O dispositivo
é criado junto com a transação (`first_seen_at` = horário dela) e nenhuma
outra transação o usa — então, em ordem cronológica, esta é a primeira e
única vez que ele aparece. Cidade, valor e horário permanecem normais (e,
para viajantes, coerentes com a cidade onde o cliente está naquele dia): só
o dispositivo destoa.

## Como o itinerário do viajante evita "teletransporte"

Na primeira versão do gerador, a cidade de cada transação normal de um
viajante era sorteada de forma independente — o cliente "aparecia" em
cidades a 1.000+ km de distância com poucas horas de intervalo, e 5,9% dos
viajantes legítimos eram marcados como viagem impossível pela feature
`is_impossible_travel`. Agora o cliente permanece numa cidade por um bloco
contínuo de dias e só então "viaja".

A garantia é por construção, não por sorte: entre a última transação de uma
estadia e a primeira da seguinte há sempre pelo menos 1 dia inteiro vazio.
Como transações só ocorrem entre 6h e 23h, a folga mínima é de ~25h — e até
a maior distância possível na Terra (~20.000 km) fica abaixo de ~800 km/h,
sempre abaixo do limiar de 900 km/h, para qualquer par de cidades. Os
cenários de fraude de sinal único (valor, horário, dispositivo) também
respeitam o itinerário: acontecem na cidade onde o cliente está naquele dia,
não na cidade natal por padrão.

Transações normais de um mesmo cliente também nunca dividem o mesmo minuto:
dois estabelecimentos da mesma cidade (a até ~15 km um do outro) no mesmo
minuto implicariam ~900 km/h — um falso positivo que não é comportamento
real.

## Limitações conhecidas (documentadas de propósito)

- As cidades que um `viajante` conhece são sorteadas na geração e o
  itinerário é uma cadeia simples (casa -> viagem -> casa ou outra
  viagem). Em um sistema real, esse padrão emergiria dos dados históricos do
  próprio cliente, com sazonalidade (férias, feriados) que aqui não existe.
- A distribuição de valores usa uma normal truncada (`random.gauss`, piso
  em R$ 5) por simplicidade; gastos reais tendem a ter cauda mais longa
  (distribuição log-normal), como já observado na análise exploratória do
  dataset do Kaggle (`notebooks/01_exploracao.ipynb`).
- Esta etapa gera e persiste os dados — nenhum modelo (estatístico ou ML)
  roda sobre este domínio ainda. Isso é intencional: o Risk Engine que vai
  consumir este banco é uma etapa futura separada.

## Camada de features comportamentais (Etapa 6)

`src/synthetic/features.py` (funções puras) e
`src/synthetic/feature_pipeline.py` (`build_feature_table(session)`)
transformam as transações do banco em uma tabela de features — uma linha
por transação, mais `is_fraud` e `fraud_scenario`. Nenhuma decisão de risco
acontece aqui; o Risk Engine (etapa futura) consome esta tabela em regras
e/ou modelos.

| Feature | Definição |
|---|---|
| `amount_deviation` | `(valor - média do cliente) / média do cliente`, com sinal. Relativa ao PRÓPRIO cliente (R$ 3.000 é ~0 para um `alto_gasto` e ~+30 para um `fixo` de média R$ 100) |
| `location_distance` | `distance_from_home_km` em escala log, normalizado para [0, 1] (referência fixa de 20.000 km) |
| `unusual_hour` | Distância circular (h) até a janela típica do cliente, dividida por 12; 0 dentro da janela |
| `new_device` | O dispositivo nunca apareceu antes no histórico do cliente — mas só quando JÁ existe histórico: na primeira transação do cliente é `False` ("nunca vi esse dispositivo porque nunca vi nada" não é o mesmo sinal que "dispositivo novo depois de um histórico estabelecido") |
| `implied_speed_kmh` / `is_impossible_travel` | km/h entre duas transações consecutivas do cliente (`geopy.geodesic`); impossível se > 900 km/h, o mesmo limiar do gerador. Primeira transação do cliente: 0.0 / False, explicitamente |

Duas decisões que valem registrar:

- `amount_deviation` usa diferença relativa e não z-score porque `Customer`
  guarda só a média; derivar um desvio-padrão das constantes do gerador
  vazaria o mecanismo de geração para dentro da feature.
- Timestamps têm resolução de minuto, então duas transações no mesmo
  minuto têm delta 0. `implied_travel_speed` aplica um piso de 1 minuto no
  delta (sem ele a velocidade seria infinita até para dois estabelecimentos
  a poucos km).

### O que as features mostram (banco com seed 42, 6.912 transações)

Médias por classe, depois das correções da Etapa 6.1 (entre parênteses, o
valor da primeira versão, antes delas):

| Feature | Média (normal) | Média (fraude) | Leitura |
|---|---|---|---|
| `amount_deviation` | 0,003 (-0,004) | 1,971 (2,240) | Separa bem (valor_atípico: 9,2; combinação: 2,5) |
| `location_distance` | 0,215 (0,214) | 0,500 (0,526) | Separa bem (geográfica/viagem/combinação: ~0,78) |
| `unusual_hour` | 0,000 (0,000) | 0,115 (0,128) | Separa, mas só em 2 dos 6 cenários |
| `new_device` | 0,006 (0,025) | 0,169 (0,025) | **Agora separa**: 100% de `dispositivo_novo`, 0,6% dos normais |
| `implied_speed_kmh` | 7,9 (140) | 501,8 (508) | Separa; viagem_impossível: 1.890 |
| `is_impossible_travel` | 0,1% (1,7%) | 21,7% (24,3%) | Pega 100% de `viagem_impossivel` (35 de 35) |

**As duas correções funcionaram:**

- `is_impossible_travel` em viajantes legítimos: 5,9% -> **0,0%** (0 de
  1.668 transações). Nas transações normais de todos os perfis: 1,7% ->
  0,13% (9 de 6.705).
- `new_device`: 2,5% vs. 2,5% (idêntico) -> 0,6% vs. 16,9%.

**Resíduos que valem saber:**

- As 9 transações normais ainda marcadas como viagem impossível vêm todas
  *logo depois de uma transação fraudulenta em local distante* (geográfica,
  combinação ou a própria viagem impossível): a feature compara com a
  transação imediatamente anterior, seja ela fraude ou não. Não é
  comportamento do cliente legítimo — é a cauda de uma fraude.
- Os 0,6% de `new_device` entre os normais são a primeira vez que um cliente
  com 2 dispositivos usa o segundo (sinal legítimo, e fraco). Uma fraude de
  outro cenário (1 de 35 `valor_atipico`) cai nesse mesmo caso.
- `is_impossible_travel` continua sendo específica: pega 100% de
  `viagem_impossivel`, mas só 17% de `anomalia_geografica` e 12% de
  `combinacao_de_sinais` (longe de casa, mas nem sempre pouco tempo depois
  da transação anterior). É a feature certa para *um* cenário, não um
  detector geral.
- `unusual_hour` é exatamente 0 para toda transação normal — artefato do
  gerador (transações normais nunca saem da janela do cliente). Em dados
  reais, clientes saem da própria janela de vez em quando, e a separação
  seria menos limpa.
- `Device.is_trusted` agora tem valor real (`False` em `dispositivo_novo`),
  mas ainda não é uma feature da tabela — candidata para o Risk Engine.
