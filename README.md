# Sentinel

Projeto Integrador (TTI 304 — Gerenciamento de Projetos em TI) sobre
**detecção de anomalias em transações financeiras**, comparando abordagens
estatísticas e de Machine Learning com foco em rigor metodológico de
avaliação.

O projeto opera em **duas vias complementares**: o dataset público
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(Kaggle/ULB) — ~284.807 transações, das quais apenas 492 (~0,17%) são
fraudes, usado para validação científica rigorosa dos modelos — e um
**domínio sintético** (clientes, cartões, dispositivos, estabelecimentos),
gerado de forma determinística e persistido em SQLite, que simula um
ambiente antifraude operacional com contexto de negócio que o Kaggle não
tem. Veja [`docs/SWOT_GUT.md`](docs/SWOT_GUT.md) para a análise de risco do
projeto e [`docs/DOMINIO_SINTETICO.md`](docs/DOMINIO_SINTETICO.md) para a
justificativa metodológica completa da via sintética.

> **Protótipo demonstrativo.** O motor de risco e o app rodam sobre o domínio
> sintético, cujos rótulos de fraude foram gerados com os mesmos sinais que o
> motor usa: as métricas dele validam a **lógica** do sistema, não o
> desempenho contra fraude real. O que cada via prova, e o que seria preciso
> para afirmar desempenho (dados reais rotulados, separação por período,
> limiares por custo), está em [`docs/VALIDACAO.md`](docs/VALIDACAO.md).

## Estrutura do repositório

```
Sentinel/
├── app/                       # aplicação Streamlit (apresentação/demo)
│   ├── app.py                 # ponto de entrada: streamlit run app/app.py
│   ├── theme.py                # paleta de cores e CSS
│   ├── risk_analysis.py        # camada fina sobre src/risk/ para a Central (sem Streamlit, sem cache)
│   ├── simulator.py            # camada fina do Simulador: contexto do cliente, cenários, raciocínio (sem Streamlit)
│   ├── live_monitor.py         # camada fina do Monitor ao Vivo: fluxo, veredito × rótulo, resumo (sem Streamlit)
│   ├── ui.py                   # formatação pt-BR, cartões, mapa e estilo de tabela compartilhados
│   └── views/                  # uma página por módulo
│       ├── overview.py         # PRODUTO: Visão Geral (proposta do Sentinel)
│       ├── transactions.py     # PRODUTO: Central de Transações (análise de risco ao vivo)
│       ├── simulator.py        # PRODUTO: Simulador (descreva uma transação, veja a análise)
│       └── live_monitor.py     # PRODUTO: Monitor ao Vivo (fluxo em tempo real, veredito × rótulo real)
├── .streamlit/
│   └── config.toml            # tema de cores nativo do Streamlit
├── data/
│   ├── raw/          # dataset do Kaggle (creditcard.csv) — não versionado
│   ├── processed/     # dados intermediários gerados pelos notebooks — não versionado
│   └── synthetic/     # sentinel.db (gerado) — não versionado, reproduzível via seed
├── notebooks/
│   └── 01_exploracao.ipynb
├── src/               # código reutilizável (importado por notebooks E pelo app)
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py               # baseline estatístico, regressão logística, isolation forest, gradient boosting
│   ├── evaluation.py            # validação cruzada, varredura de threshold
│   └── synthetic/                # domínio sintético — independente do Kaggle
│       ├── models.py             # ORM SQLAlchemy (Customer, Card, Device, Merchant, Transaction)
│       ├── geo_reference.py      # cidades reais (lat/lon) + distância geodésica local
│       ├── generator.py          # gerador determinístico (perfis + 6 cenários de fraude)
│       ├── live_generator.py     # gerador ao vivo: uma transação nova por vez, com o rótulo real
│       ├── features.py           # features comportamentais (funções puras)
│       └── feature_pipeline.py   # build_feature_table(session): banco -> DataFrame
│   └── risk/                     # Risk Engine (só domínio sintético)
│       ├── dataset.py            # tabela de risco: features + km + confiança do dispositivo
│       ├── behavior_score.py     # score por regras explicáveis (pesos em BehaviorConfig)
│       ├── ml_score.py           # Regressão Logística treinada nos dados sintéticos
│       ├── engine.py             # RiskEngine: combina ML + regras -> score, nível, ação, motivos
│       ├── new_transaction.py    # pontua uma transação nova a partir do histórico do cliente
│       ├── evaluation.py         # split por cliente, métricas, calibração de limiares
│       └── report.py             # avaliação comparativa (regras vs ML vs combinado)
├── scripts/
│   ├── download_dataset.py   # download do Kaggle via API
│   ├── generate_mock_data.py # gera e persiste o domínio sintético
│   └── run_risk_engine.py    # relatório de avaliação do Risk Engine
├── reports/           # figuras e relatórios gerados
├── docs/
│   ├── SWOT_GUT.md
│   ├── DOMINIO_SINTETICO.md
│   └── RISK_ENGINE.md
├── requirements.txt
└── README.md
```

**Por que separar `src/` de `notebooks/` e de `app/`:** o projeto compara
várias abordagens (regras estatísticas, regressão logística, isolation
forest, gradient boosting). Se cada notebook (ou a aplicação) reimplementar o
carregamento de dados, o split treino/teste, o balanceamento de classes e
o treino dos modelos do zero, é fácil introduzir inconsistências sutis que
invalidam a comparação entre modelos. Colocando essa lógica em `src/` como
funções puras, tanto os notebooks quanto o app Streamlit partem exatamente
da mesma base — o app é só uma camada de visualização sobre `src/`.

## 1. Pré-requisitos

- **Python 3.12+** instalado e no PATH (as versões fixadas em `requirements.txt`
  não têm pacotes para o 3.11).
  No Windows, verifique com:
  ```bash
  python --version
  ```
  Se aparecer uma mensagem dizendo para instalar pela Microsoft Store, o
  Python real não está instalado — baixe o instalador oficial em
  [python.org/downloads](https://www.python.org/downloads/) e marque a opção
  **"Add python.exe to PATH"** durante a instalação.
- Uma conta gratuita no [Kaggle](https://www.kaggle.com/) (necessária para
  baixar o dataset).

## 2. Configurando o ambiente

Crie e ative um ambiente virtual isolado para o projeto (evita conflitos com
outras versões de bibliotecas instaladas na máquina):

```bash
python -m venv venv
```

Ativar o ambiente:

```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (cmd)
venv\Scripts\activate.bat

# Linux / macOS
source venv/bin/activate
```

Com o ambiente ativo (o prompt do terminal deve mostrar `(venv)` no início),
instale as dependências:

```bash
pip install -r requirements.txt
```

### Por que estas bibliotecas

| Biblioteca | Papel no projeto |
|---|---|
| `pandas` / `numpy` | Manipulação dos dados tabulares |
| `scikit-learn` | Modelos de ML, split treino/teste, métricas (matriz de confusão, AUC-ROC, F1) |
| `imbalanced-learn` | Técnicas de balanceamento de classes (SMOTE, undersampling) — essencial dado o desbalanceamento de ~0,17% |
| `matplotlib` / `seaborn` | Visualização no notebook de análise exploratória |
| `jupyter` / `notebook` | Ambiente dos notebooks |
| `kaggle` | Download automatizado do dataset via API (opcional, ver seção 3) |
| `streamlit` | Aplicação interativa de apresentação (ver seção 5) |
| `plotly` | Gráficos interativos usados na aplicação |
| `sqlalchemy` / `geopy` | Banco SQLite do domínio sintético e distâncias geográficas |

## 3. Obtendo o dataset

O arquivo `creditcard.csv` **não é versionado no Git** (tem ~150 MB e é
distribuído pelo Kaggle sob os termos deles, não nossos). Escolha uma das
duas opções abaixo — o resultado precisa ser o arquivo
`data/raw/creditcard.csv`.

### Opção A — Download manual (mais simples, exige login no navegador)

1. Acesse https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud e faça
   login (crie uma conta gratuita se não tiver).
2. Clique em **Download** (baixa um `.zip`).
3. Extraia o `.zip` — dentro dele está o `creditcard.csv`.
4. Mova o arquivo para `data/raw/creditcard.csv` dentro do repositório.

### Opção B — Script automatizado via API do Kaggle

Só compensa se você for baixar o dataset mais de uma vez (ex.: em máquinas
diferentes do time). Requer gerar um token de API uma vez:

1. Acesse https://www.kaggle.com/settings, seção **API**, clique em
   **Create New Token**. Isso baixa um arquivo `kaggle.json`.
2. Coloque esse arquivo em `C:\Users\<seu-usuário>\.kaggle\kaggle.json`
   (Windows) ou `~/.kaggle/kaggle.json` (Linux/macOS).
3. Com o ambiente virtual ativo, rode:
   ```bash
   python scripts/download_dataset.py
   ```

> **Atenção:** nunca faça commit do `kaggle.json` — ele é uma credencial
> pessoal. O `.gitignore` já bloqueia esse arquivo por segurança.

## 4. Rodando o notebook de exploração

Com o ambiente ativo e o dataset em `data/raw/creditcard.csv`:

```bash
jupyter notebook notebooks/01_exploracao.ipynb
```

Esse notebook carrega os dados, confirma o desbalanceamento de classes e
explora as distribuições de `Amount` e `Time` — é o ponto de partida antes
de qualquer modelagem.

## 5. Rodando a aplicação (Streamlit)

A aplicação é o que se roda **para a apresentação/banca**. Com o ambiente
ativo:

```bash
streamlit run app/app.py
```

Isso abre automaticamente uma aba no navegador em `http://localhost:8501`.

O app mostra **só o produto** (uma plataforma de análise de risco
transacional para uma instituição financeira) e não precisa do dataset do
Kaggle. A validação científica dos modelos sobre o Kaggle — base
metodológica do produto — continua no repositório (`notebooks/` e `src/`),
fora do app.

Páginas da barra lateral:

| Página | O que mostra |
|---|---|
| **Visão Geral** | A proposta do Sentinel, os sinais que considera, o estado do projeto e (recolhido) o planejamento SWOT/GUT |
| **Central de Transações** | A tela principal do produto: abre vazia e, ao clicar em "Executar análise de risco", processa o banco sintético ao vivo (ver abaixo) |
| **Simulador** | Descreva uma transação de um cliente e veja o Sentinel analisá-la ao vivo, passo a passo (ver abaixo) |
| **Monitor ao Vivo** | Fluxo de transações em tempo real: cada uma recebe o veredito "Fraudulenta" ou "Legítima", mostrado ao lado do rótulo real (ver abaixo) |

#### Monitor ao Vivo

Com **Transmitir ao vivo** ligado, chega uma transação nova a cada intervalo
(0,5 a 5 s), para um cliente sorteado do banco sintético. Ela vem do gerador
ao vivo (`src/synthetic/live_generator.py`), que segue as mesmas regras do
gerador do banco — compra normal, ou um dos 6 cenários de fraude na
proporção escolhida — e por isso traz o **rótulo real**. O motor recebe só
os dados brutos (nunca o rótulo) e dá o **veredito**: *Fraudulenta* quando
não aprova (revisar ou bloquear), *Legítima* quando aprova. O painel mostra
os dois lado a lado, com a matriz de confusão ao vivo, acurácia, recall,
precisão, valor retido, mapa, linha do tempo do Risk Score, sinais mais
frequentes e o feed. **Modo apresentação** fixa a sequência sorteada (para
ensaiar a demonstração); **Gerar 10 agora** preenche o painel sem esperar.
Nada é gravado no banco.

#### Central de Transações

Precisa do banco sintético (`python scripts/generate_mock_data.py`, seção
6); sem ele a página explica o que fazer em vez de quebrar. **Nada vem
pré-calculado**: ao abrir, a página está vazia (só o botão e a lista do que
vai acontecer) e não há `st.cache_*` rodando a análise. Só o clique em
**Executar análise de risco** processa:

1. um `st.status` mostra cada etapa e o tempo que ela levou — carregando
   transações do banco, calculando features comportamentais, aplicando o
   modelo de ML, aplicando as regras, combinando no Risk Score, classificando
   por nível;
2. o resultado vai para `st.session_state`, então filtros e seleção de linha
   reexecutam só a página, nunca a análise (o botão "Limpar resultado"
   volta ao estado vazio);
3. **resumo** com total, e quantidade e percentual por nível e por ação;
4. **tabela investigável**, ordenada por Risk Score decrescente (colorido
   por nível), com filtros de nível, ação, cliente, perfil, faixa de valor e
   conjunto (treino/validação/teste). Clicar numa linha — ou digitar o ID da
   transação — abre o **painel de investigação**: valor, cidade, horário,
   cliente, local habitual, distância da residência, dispositivo, score do ML
   e das regras separados, Risk Score final, nível, ação recomendada, os
   motivos em português e um mapa (residência, local da compra e a linha
   entre eles);
5. **eficiência medida ao vivo**: a decisão do sistema contra o rótulo da
   simulação (`is_fraud`) — fraudes capturadas, falsos positivos, matriz de
   confusão — calculada **só sobre os clientes de teste** (o mesmo split por
   cliente da avaliação do motor), sempre com o aviso "dados simulados;
   valida a lógica do sistema, não o desempenho em fraude real".

A análise treina o ML nos clientes de treino, escolhe o peso e calibra as
faixas nos de validação (as mesmas regras de `src/risk/report.py`) e aplica o
resultado a todas as transações; por isso a tabela mostra a coluna
"Conjunto", e as linhas de treino/validação nunca entram nas métricas. A
lógica vive em `app/risk_analysis.py` e é testada contra
`evaluate_risk_engine` e `RiskEngine.assess_frame` (mesmos pesos, faixas e
decisões); a página só orquestra e exibe.

#### Simulador

Também precisa do banco sintético. Descreve **uma transação nova** de um
cliente existente e mostra a análise de risco na hora. A página abre sem
veredito — só o formulário — e **o resultado só aparece depois do clique em
"Analisar transação"**:

1. **Formulário:** cliente (nome e cidade natal), valor, cidade da compra
   (as de `src/synthetic/geo_reference.py`), data e hora, e dispositivo (os
   conhecidos do cliente, mais "Novo dispositivo, não reconhecido"). O
   horário precisa ser **posterior à última transação do cliente**; senão a
   tela avisa e o botão fica desabilitado.
2. **Cenários de demonstração:** compra normal, viagem impossível, valor
   atípico (~10x a média), horário atípico, dispositivo novo, cidade nova e
   combinação de sinais (dispositivo novo + horário atípico + valor atípico).
   Os botões **só preenchem o formulário**, com valores calculados do
   histórico real do cliente escolhido (cidade natal, média, janela
   habitual, última compra...). Nenhum traz resultado embutido: o veredito
   vem sempre do motor, ao clicar em Analisar.
3. **Raciocínio ao vivo** (`st.status`): histórico recuperado (nº de
   transações, valor médio, cidades conhecidas), distância da residência,
   velocidade implícita desde a última compra (km/h e minutos), horário
   contra a janela habitual, valor contra a média, dispositivo e cidade
   reconhecidos ou não, score do ML, score das regras e a combinação no Risk
   Score — todos calculados na hora, nenhum texto fixo. Se não houver motor
   treinado na sessão ele é treinado como uma das etapas; se a Central de
   Transações já o treinou, é reaproveitado.
4. **Veredito:** Risk Score num gauge com as faixas 30 e 70, nível, ação,
   scores de ML e regras separados, motivos em português, mapa (residência,
   compra e a linha entre elas) e o resumo do histórico do cliente.
5. **Histórico da sessão:** tabela das transações analisadas, guardada só em
   `st.session_state` — nada é gravado no banco.

O veredito é exatamente o de `src/risk/new_transaction.py`
(`compute_transaction_features` + `RiskEngine.assess`): a camada
`app/simulator.py` só acrescenta o contexto do cliente, os cenários, a
validação do horário e o texto do raciocínio, e os testes garantem que o
resultado coincide com `score_new_transaction` para a mesma entrada.

**Os cenários não garantem um nível.** Quem decide é o motor, e há casos em
que uma anomalia fica abaixo do limiar de revisão (por exemplo, dispositivo
novo isolado). Isso é um comportamento do motor atual, não da página — ver
"Resultados dos cenários de demonstração" em `docs/RISK_ENGINE.md`.

### Score, threshold e decisão (`src/models.py`)

Todo modelo segue o mesmo fluxo, em três passos deliberadamente separados:

```
fraud_score(modelo, X)  →  threshold  →  apply_threshold(score, threshold)  →  y_pred
```

Nenhum modelo decide "fraude ou não" sozinho via `model.predict()` — isso
evitaria comparar os modelos de forma consistente, já que cada biblioteca tem
sua própria convenção interna de corte. Em vez disso:

- **`fraud_score()`** extrai um escore contínuo de qualquer modelo (maior =
  mais suspeito), escondendo a API específica de cada um (probabilidade da
  Regressão Logística e do Gradient Boosting, distância do baseline,
  escore de isolamento do Isolation Forest).
- **`calibrate_threshold_by_contamination()`** calcula um threshold default
  a partir da taxa de fraude do **treino** — só para gerar um ponto de
  operação comparável entre os modelos nesta etapa acadêmica; não é
  uma prática de produção (a taxa real de fraude não estaria disponível
  com essa precisão fora de um dataset já rotulado).
- **`apply_threshold()`** é a única função que transforma escore em decisão
  0/1 — pura, sem depender de modelo.
- **`evaluate_at_threshold()`** recalcula precisão/recall/F1/matriz de
  confusão para qualquer threshold sobre o mesmo escore, sem retreinar
  nada — a base para uma futura página de análise de threshold.

Treino e calibração de threshold são etapas separadas: nenhum modelo é
treinado usando a taxa real de fraude (o `IsolationForest` usa
`contamination="auto"`, a heurística padrão do scikit-learn).

**Gradient Boosting** (`HistGradientBoostingClassifier`) é o modelo que mais
aprende sozinho: árvores em sequência, cada uma corrigindo as anteriores,
que capturam interações entre features (ex.: "V14 baixo **e** valor alto")
que a Regressão Logística, linear, não representa. Um teste
(`tests/test_models.py`) mostra isso num padrão que só existe na combinação
de duas features.

**`Time` não entra nos modelos.** No dataset, `Time` são segundos desde a
primeira transação, não um horário. Como feature, ele só diria "quão tarde
na coleta" a compra ocorreu, e no split temporal todo o teste fica fora do
intervalo do treino (o modelo extrapola). `prepare_features`
([`src/preprocessing.py`](src/preprocessing.py)) troca `Time` pela hora do
dia em seno/cosseno e padroniza `Amount`; `Time` segue usado só para
ordenar o split temporal.

O balanceamento via `class_weight="balanced"` (Regressão Logística) prioriza
um app rápido de treinar/interagir. O SMOTE/undersampling implementado em
[`src/preprocessing.py`](src/preprocessing.py) continua disponível e é o
que o notebook de exploração usa como exemplo alternativo de tratamento de
desbalanceamento.

### Validação cruzada (`src/evaluation.py`)

O projeto reporta dois resultados **diferentes e não-intercambiáveis**:

- **Conjunto de teste final** (a maior parte da página "Comparação de
  Modelos"): cada modelo é treinado UMA vez no treino completo e avaliado
  UMA vez no teste, nunca visto antes — é o número que representa "o
  desempenho do modelo".
- **Validação cruzada** (seção separada, dentro de um expander): mede o
  quão **estável** cada modelo é entre diferentes recortes do treino — não
  substitui o teste final, complementa.

A validação cruzada usa `StratifiedKFold(n_splits=5, shuffle=True,
random_state=42)` — estratificado pelo mesmo motivo do split treino/teste
(com ~0,17% de fraude, um fold não-estratificado corre risco real de ficar
com poucas fraudes, ou nenhuma). Ela roda **inteiramente dentro do
conjunto de treino**: o conjunto de teste final nunca é passado para
`cross_validate_models()` — a função nem aceita esse parâmetro.

Dentro de cada um dos 5 folds, na ordem certa para não vazar dados:

1. um `StandardScaler` **novo** é ajustado só com o treino daquele fold
   (nunca com o treino completo, e nunca com o validation do fold);
2. os modelos são treinados do zero nesse treino de fold;
3. o threshold de cada modelo é calibrado usando só o escore do **treino
   do fold**;
4. esse threshold é aplicado ao **validation do fold** (nunca visto pelos
   passos 1-3) para calcular as métricas daquele fold.

Repetido 5 vezes, isso dá 5 medidas independentes de cada métrica por
modelo — reportadas como `média ± desvio-padrão` na aplicação (nunca só a
média sozinha, para não esconder o quanto os modelos variam entre folds).

**Sobre o threshold (nos folds e no teste final):** ele representa um
ponto de operação calibrado pela prevalência de fraude observada no
conjunto de treinamento (ou, em cada fold, no treino daquele fold) — **não
deve ser interpretado como conhecimento disponível em produção**, e não é
uma probabilidade de fraude. Em produção, essa prevalência não estaria
disponível com essa precisão; o threshold seria escolhido por critério de
negócio.

**Fonte única do split:** `get_pipeline()` (teste final) e
`get_cv_results()` (validação cruzada) chamam os dois a mesma função,
[`get_train_test_split()`](src/preprocessing.py) — antes cada um montava o
split separadamente (mesmo resultado, já que usavam o mesmo
`random_state`, mas com a lógica escrita duas vezes). Agora há um único
lugar que decide como o dataset é dividido.

### Análise de Threshold (`src/evaluation.py`)

Score e decisão são conceitos diferentes (ver "Score, threshold e decisão"
acima) — `threshold_sweep` mostra como precisão/recall/F1 mudam com o
limiar **sem retreinar nada**. (A página interativa que usava esta função
saiu do app junto com a seção científica, na Etapa 13; a função continua
em `src/` e testada.)

Alguns pontos de design importantes:

- **O slider é só uma simulação.** Mover o threshold na página nunca
  altera o modelo, nunca altera o threshold "oficial" calibrado pelo
  pipeline (o mesmo usado na página "Comparação de Modelos"), e nada é
  persistido — trocar de página ou reabrir o app volta ao ponto de partida.
- **A escala do slider é a escala nativa de cada modelo** — nunca 0 a 1
  "assumido cegamente". Regressão Logística produz probabilidade (0-1),
  mas o baseline estatístico (distância no espaço V1..V28) e o Isolation
  Forest (escore de isolamento) usam escalas completamente diferentes;
  `fraud_score()` (`src/models.py`) já garante que "maior = mais
  suspeito" para todos, mas não uniformiza a escala — e esta página não
  cria uma segunda normalização por cima disso, só usa a escala como ela é.
- **AUC-ROC e AUC-PR não mudam com o slider** — são calculadas sobre o
  ranking completo do escore, exibidas como referência fixa da qualidade
  do modelo, ao lado (não misturadas) das métricas que de fato dependem
  do threshold escolhido (precisão, recall, F1, FPR, TPR, specificity,
  balanced accuracy, MCC).
- **Nada é retreinado, nada de CV roda de novo.** A página parte do
  resultado já cacheado de `get_pipeline()`; a única coisa recalculada a
  cada movimento do slider é `calculate_metrics()` sobre o escore já
  pronto — barato, não envolve o modelo. O gráfico "métrica × threshold"
  usa uma varredura (`threshold_sweep()`) cacheada por modelo, para não
  recalcular 100 pontos a cada tick do slider.
- **Linguagem:** o Sentinel nunca afirma "fraude confirmada" — a decisão é
  sempre "Normal" ou "Potencialmente suspeita". O escore não é uma
  probabilidade de fraude calibrada; é um ranking de risco relativo,
  usado para avaliação de risco transacional, não como veredito.

## 6. Domínio sintético (via de simulação operacional)

Complementa o Kaggle com um domínio que tem contexto de negócio explícito
(cliente, cartão, dispositivo, estabelecimento, geolocalização), gerado de
forma determinística e persistido em SQLite via SQLAlchemy — nunca CSV
solto. Justificativa metodológica completa, entidades, perfis
comportamentais e os 6 cenários de fraude:
[`docs/DOMINIO_SINTETICO.md`](docs/DOMINIO_SINTETICO.md).

Para gerar (ou regenerar) o banco:

```bash
python scripts/generate_mock_data.py
```

> **Banco antigo:** a Etapa 12 acrescentou canal e forma de pagamento às
> transações. Um `sentinel.db` gerado antes disso não tem essas colunas —
> regenere com `python scripts/generate_mock_data.py --force`.

Cria `data/synthetic/sentinel.db` com ~130 clientes e alguns milhares de
transações (~3% marcadas como fraude, com o cenário que a justifica). Seed
fixa (`--seed`, default 42) — rodar de novo produz exatamente o mesmo
banco; use `--force` para sobrescrever um banco já existente, ou
`--n-customers` para gerar um volume diferente.

O Risk Engine (seção 7), a Central de Transações e o Simulador (seção 5)
consomem este banco.

## 7. Risk Engine

O núcleo do produto: recebe uma transação do domínio sintético e devolve um
**Risk Score 0-100**, um **nível** (baixo/médio/alto), uma **ação
recomendada** (aprovar/revisar/bloquear) e os **motivos** em português,
ordenados pela contribuição de cada um. Combina dois scores, com o peso
escolhido na validação por uma regra declarada (maior PR-AUC com no mínimo
30% de peso nas regras, por explicabilidade) — hoje 30% regras / 70% ML:

- **regras explicáveis** (viagem impossível, localização, valor, cidade
  nova, dispositivo novo, dispositivo não confiável, horário), com pesos
  justificados num único objeto de configuração;
- **Regressão Logística** treinada só nos dados sintéticos.

Viagem impossível nunca é aprovada automaticamente (piso de nível médio).
O Risk Score exibido usa uma escala em que os limiares calibrados aparecem
sempre como 30 e 70 (a decisão é a mesma do score interno).
Arquitetura, justificativa de cada peso, resultados e limitações:
[`docs/RISK_ENGINE.md`](docs/RISK_ENGINE.md).

Para rodar a avaliação completa (requer `data/synthetic/sentinel.db`):

```bash
python scripts/run_risk_engine.py
```

O relatório compara só regras, só ML e o combinado (PR-AUC, ROC-AUC,
precisão, recall, F1), mostra o recall por cenário de fraude, a
distribuição por nível/ação e 3 exemplos explicados. O treino/teste é
dividido **por cliente**; peso e limiares são escolhidos num conjunto de
validação separado, e o teste é usado uma única vez. Resultado principal, dito sem rodeios: o combinado
supera muito as regras sozinhas, mas **não supera o ML sozinho** no
ranking — a diferença vem de um artefato do gerador (horário) e das
primeiras visitas legítimas a uma cidade, indistinguíveis de fraude
geográfica com as features atuais (ver as seções 5 e 6 do doc).

Os números validam a lógica do sistema sobre dados **simulados**: os
rótulos foram gerados a partir das mesmas features que o motor usa, então
não provam desempenho em fraude real.

## 8. Metodologia de avaliação

Como a maior fraqueza identificada na análise SWOT/GUT é a pouca experiência
do time com validação estatística, o projeto segue algumas regras fixas
(implementadas em [`src/preprocessing.py`](src/preprocessing.py)):

- **Nunca usar acurácia isolada como métrica de sucesso.** Com 99,83% das
  transações sendo legítimas, um modelo que nunca detecta fraude nenhuma
  ainda "acerta" 99,83% das vezes. Métricas usadas: matriz de confusão,
  precisão, recall, F1-score, AUC-ROC e AUC-PR.
- **AUC-PR ao lado de AUC-ROC.** Em datasets tão desbalanceados quanto este
  (~0,17% de fraude), AUC-ROC tende a parecer otimista demais — a taxa de
  falsos positivos é normalizada pelo tamanho da classe majoritária, então
  um número grande de falsos positivos ainda resulta numa curva ROC
  "bonita". AUC-PR (average precision) é mais sensível a isso.
- **Threshold explícito, nunca escondido dentro do treino/predict** — ver
  "Score, threshold e decisão" acima.
- **Validação cruzada estratificada (5 folds), só no treino** — mede
  estabilidade além do ponto único do teste final. Ver "Validação cruzada"
  acima.
- **Split estratificado** (`stratify=y`) para preservar a proporção de
  fraudes em treino e teste.
- **Balanceamento (SMOTE/undersampling) aplicado só no treino**, depois do
  split — nunca no teste, para as métricas continuarem representando a
  proporção real de fraudes.
- **MVP antes de otimização:** um baseline estatístico simples (regras) roda
  de ponta a ponta antes de investir em ajuste fino de modelos de ML.
- **Avaliação temporal, volume de alertas e limiar por custo** no benchmark
  público ([`src/benchmark.py`](src/benchmark.py),
  `python scripts/run_kaggle_benchmark.py`): treino no passado e teste no
  futuro, comparado ao split aleatório; alertas por dia para cada orçamento
  de alertas; e o limiar que minimiza revisões + fraudes não detectadas. Ver
  [`docs/VALIDACAO.md`](docs/VALIDACAO.md).

## 9. Testes

```bash
pytest
```

Os testes também rodam sozinhos no GitHub (Actions), em cada push e em cada
pull request para a `main`, com Python 3.12 e as versões fixadas em
`requirements.txt` (`.github/workflows/tests.yml`). Um PR com teste
falhando aparece marcado em vermelho antes do merge.

1.213 testes (a maioria parametrizações da escala de exibição), nenhum depende de dados baixados/gerados manualmente:

- **Camada de avaliação (Kaggle)**: separação entre escore e decisão,
  AUC-PR, thresholds inválidos, validação cruzada (5 folds, threshold
  calibrado só no treino do fold, resultado agregado com mean/std), a
  varredura de threshold (`threshold_sweep`) e que o `StandardScaler`
  nunca usa dados de teste/validation para se ajustar — inclusive através
  da função de split centralizada (`get_train_test_split`). Rodam sobre um
  dataset sintético **no formato do Kaggle** gerado em `tests/conftest.py`
  (não confundir com o domínio sintético abaixo), não sobre
  `data/raw/creditcard.csv`.
- **Features comportamentais** (`tests/test_features.py`): cada feature com
  casos simples, primeira transação de cada cliente sem `NaN`, e
  `is_impossible_travel` verdadeiro em `viagem_impossivel`.
- **Risk Engine** (`tests/test_risk_*.py`): cada regra isolada,
  fronteiras das faixas (30/31, 70/71), piso de viagem impossível,
  escala de exibição que nunca muda a classificação, `new_city` sem
  vazamento temporal,
  determinismo, split sem cliente em dois conjuntos, calibração de
  limiares e transação nova coerente com `build_risk_table`. Usam um
  SQLite em memória populado pelo gerador.
- **Central de Transações** (`tests/test_app_risk_analysis.py`,
  `tests/test_app_transactions.py`): a camada de análise produz os mesmos
  pesos, faixas e decisões que `evaluate_risk_engine` e
  `RiskEngine.assess_frame`; a eficiência usa só clientes de teste (estragar
  os rótulos de treino/validação não a altera); filtros; painel coerente com
  a tabela. Com `AppTest`: a página abre vazia e **não executa a análise** ao
  abrir, os resultados só existem depois do clique, e filtros/seleção não a
  reexecutam. Usam um SQLite temporário em arquivo, nunca o banco real.
- **Simulador** (`tests/test_app_simulator.py`,
  `tests/test_app_simulator_page.py`): os cenários são calculados do
  histórico de cada um dos 130 clientes e só preenchem o formulário; cada
  cenário aciona o sinal que promete (viagem impossível produz
  `is_impossible_travel` para todos os clientes); o veredito coincide com
  `score_new_transaction`; validação do horário; o raciocínio traz valores
  reais; nada é gravado no banco. Com `AppTest`: a página abre sem veredito,
  o resultado só existe depois do clique, o motor é treinado uma vez (ou
  reaproveitado da Central) e o histórico da sessão acumula as análises.
- **Domínio sintético** (`tests/test_synthetic_data.py`): volume de
  clientes/transações dentro do esperado, taxa de fraude na faixa (~3%),
  nenhuma `distance_from_home_km` negativa, que toda transação
  `viagem_impossivel` realmente viola 900 km/h, geração determinística
  (mesma seed = mesmo dataset) e persistência em SQLite `:memory:` — nunca
  o `data/synthetic/sentinel.db` real.

## 10. Próximos passos

- [x] Implementar o baseline estatístico, Regressão Logística e Isolation Forest (`src/models.py`).
- [x] Gradient Boosting e hora do dia no lugar de `Time` bruto (`src/models.py`, `src/preprocessing.py`).
- [x] Aplicação interativa (Streamlit) para apresentação (`app/`).
- [x] AUC-PR e separação explícita entre score e threshold (`src/models.py`).
- [x] Validação cruzada estratificada, só no treino (`src/evaluation.py`).
- [x] Fonte única do split treino/teste (`get_train_test_split`, `src/preprocessing.py`).
- [x] Página de Análise de Threshold interativa (removida do app na Etapa 13; `threshold_sweep` continua em `src/evaluation.py`).
- [x] Domínio sintético gerado e persistido em SQLite (`src/synthetic/`, `docs/DOMINIO_SINTETICO.md`).
- [x] Testes unitários da camada de avaliação e do domínio sintético (`tests/`).
- [ ] Rodar `01_exploracao.ipynb` com o dataset real e preencher a seção de conclusões.
- [x] Features comportamentais sobre o domínio sintético (`src/synthetic/features.py`).
- [x] Risk Engine: regras + ML, score 0-100, nível, ação e motivos (`src/risk/`, `docs/RISK_ENGINE.md`).
- [x] Feature de novidade de cidade (`new_city`) — ver `docs/RISK_ENGINE.md`, seção 6.
- [x] Central de Transações: análise de risco ao vivo, tabela investigável, painel, eficiência no teste (`app/views/transactions.py`).
- [x] Simulador ao vivo: formulário, cenários de demonstração, raciocínio passo a passo, veredito e histórico da sessão (`app/views/simulator.py`).
- [x] Avaliação temporal, volume de alertas e limiar por custo no benchmark público (`src/benchmark.py`).
- [x] Dependências fixadas e testes automáticos no GitHub Actions (`.github/workflows/tests.yml`).
- [x] Rodar `scripts/run_kaggle_benchmark.py` com o dataset real e registrar os números (`docs/VALIDACAO.md`, seção 2).
- [ ] Validar com dados reais rotulados de uma instituição, separados por período, com limiares por custo (`docs/VALIDACAO.md`, seção 3).
- [ ] Ajustar hiperparâmetros e comparar com balanceamento via SMOTE (não só `class_weight`).
- [ ] Notebook comparativo formalizando as métricas das três abordagens (hoje só na aplicação).
