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
| `viajante` | ~25% | Além da cidade natal, tem um conjunto pessoal de 2-3 cidades que visita com frequência — transações nessas cidades são NORMAIS para este cliente, não anomalias |
| `alto_gasto` | ~15% | Mesmo padrão geográfico de um cliente fixo, mas com valor médio de referência bem mais alto (R$500–3.000 vs. R$50–300) |
| `novo_cliente` | ~10% | Poucas transações no histórico (8-15, contra 30-80 dos demais) — ainda não tem um padrão bem estabelecido |

## Cenários de fraude

Cada transação fraudulenta carrega o campo `fraud_scenario` explicando por
que foi marcada — nunca "fraude" sem justificativa. Meta: ~3% das
transações totais, dividida o mais igualmente possível entre os 5
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
para disparar os outros 4 cenários — é o cenário desenhado para ser mais
difícil de capturar com uma única regra estatística simples, motivando
diretamente por que o projeto compara regras com Machine Learning.

## Limitações conhecidas (documentadas de propósito)

- O relacionamento entre cidades "frequentadas" por um cliente `viajante`
  e a lista de cidades candidatas para os cenários de fraude é uma
  aproximação — em um sistema real, esse padrão emergiria dos dados
  históricos do próprio cliente, não seria definido na geração.
- A distribuição de valores usa uma normal truncada (`random.gauss`, piso
  em R$ 5) por simplicidade; gastos reais tendem a ter cauda mais longa
  (distribuição log-normal), como já observado na análise exploratória do
  dataset do Kaggle (`notebooks/01_exploracao.ipynb`).
- Esta etapa gera e persiste os dados — nenhum modelo (estatístico ou ML)
  roda sobre este domínio ainda. Isso é intencional: o Risk Engine que vai
  consumir este banco é uma etapa futura separada.
