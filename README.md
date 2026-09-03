# Sentinel

Projeto Integrador (TTI 304 — Gerenciamento de Projetos em TI) sobre
**detecção de anomalias em transações financeiras**, comparando abordagens
estatísticas e de Machine Learning com foco em rigor metodológico de
avaliação.

O dataset usado é o [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(Kaggle/ULB): ~284.807 transações, das quais apenas 492 (~0,17%) são fraudes.
Esse desbalanceamento extremo é o maior risco técnico do projeto — veja
[`docs/SWOT_GUT.md`](docs/SWOT_GUT.md) para a análise completa e as decisões
de mitigação adotadas.

## Estrutura do repositório

```
Sentinel/
├── app/                       # aplicação Streamlit (apresentação/demo)
│   ├── app.py                 # ponto de entrada: streamlit run app/app.py
│   ├── theme.py                # paleta de cores e CSS
│   ├── pipeline.py             # cache de dados/modelos (st.cache_data/cache_resource)
│   └── views/                  # uma página por módulo
│       ├── overview.py         # Visão Geral
│       ├── eda.py              # Análise Exploratória
│       ├── models_page.py      # Comparação de Modelos
│       └── demo.py             # Demo ao Vivo
├── .streamlit/
│   └── config.toml            # tema de cores nativo do Streamlit
├── data/
│   ├── raw/          # dataset original (creditcard.csv) — não versionado
│   └── processed/     # dados intermediários gerados pelos notebooks — não versionado
├── notebooks/
│   └── 01_exploracao.ipynb
├── src/               # código reutilizável (importado por notebooks E pelo app)
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── models.py               # baseline estatístico, regressão logística, isolation forest
├── scripts/
│   └── download_dataset.py   # download automatizado via API do Kaggle
├── reports/           # figuras e relatórios gerados
├── docs/
│   └── SWOT_GUT.md
├── requirements.txt
└── README.md
```

**Por que separar `src/` de `notebooks/` e de `app/`:** o projeto compara
várias abordagens (regras estatísticas, regressão logística, isolation
forest, ...). Se cada notebook (ou a aplicação) reimplementar o
carregamento de dados, o split treino/teste, o balanceamento de classes e
o treino dos modelos do zero, é fácil introduzir inconsistências sutis que
invalidam a comparação entre modelos. Colocando essa lógica em `src/` como
funções puras, tanto os notebooks quanto o app Streamlit partem exatamente
da mesma base — o app é só uma camada de visualização sobre `src/`
(`app/pipeline.py` adiciona cache em cima das mesmas funções).

## 1. Pré-requisitos

- **Python 3.11+** instalado e no PATH.
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
| `matplotlib` / `seaborn` | Visualização na análise exploratória |
| `jupyter` / `notebook` | Ambiente dos notebooks |
| `kaggle` | Download automatizado do dataset via API (opcional, ver seção 3) |
| `streamlit` | Aplicação interativa de apresentação (ver seção 5) |
| `plotly` | Gráficos interativos usados na aplicação |

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

A aplicação é o que se roda **para a apresentação/banca** — um dashboard
interativo, em vez do professor ter que ler um notebook. Com o ambiente
ativo e o dataset em `data/raw/creditcard.csv`:

```bash
streamlit run app/app.py
```

Isso abre automaticamente uma aba no navegador em `http://localhost:8501`.
Na primeira vez que a página **Comparação de Modelos** (ou **Demo ao Vivo**,
ou **Análise de Threshold**) é aberta, o app treina os três modelos — leva
alguns segundos; depois disso fica em cache (`st.cache_resource`) e as
trocas de página/filtro são instantâneas, sem retreinar nada.

Cinco páginas, navegáveis pela barra lateral:

| Página | O que mostra |
|---|---|
| **Visão Geral** | Objetivo do projeto e resumo visual da análise SWOT/GUT |
| **Análise Exploratória** | Distribuição de classes, `Amount` por classe e correlações — versão interativa do notebook |
| **Comparação de Modelos** | Matriz de confusão, precisão, recall, F1, curvas ROC/PR no conjunto de teste, e uma seção separada de validação cruzada (estabilidade dos modelos) |
| **Análise de Threshold** | Simula, para qualquer um dos 3 modelos, como mudar o limiar de decisão altera precisão/recall/F1/matriz de confusão — sem retreinar nada |
| **Demo ao Vivo** | Escolha uma transação real do conjunto de teste e veja a classificação de qualquer um dos 3 modelos na hora, com um gauge de score de risco |

### Score, threshold e decisão (`src/models.py`)

Todo modelo segue o mesmo fluxo, em três passos deliberadamente separados:

```
fraud_score(modelo, X)  →  threshold  →  apply_threshold(score, threshold)  →  y_pred
```

Nenhum modelo decide "fraude ou não" sozinho via `model.predict()` — isso
evitaria comparar os três de forma consistente, já que cada biblioteca tem
sua própria convenção interna de corte. Em vez disso:

- **`fraud_score()`** extrai um escore contínuo de qualquer modelo (maior =
  mais suspeito), escondendo a API específica de cada um (probabilidade da
  Regressão Logística, distância do baseline, escore de isolamento do
  Isolation Forest).
- **`calibrate_threshold_by_contamination()`** calcula um threshold default
  a partir da taxa de fraude do **treino** — só para gerar um ponto de
  operação comparável entre os três modelos nesta etapa acadêmica; não é
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
2. os três modelos são treinados do zero nesse treino de fold;
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

### Análise de Threshold (`src/evaluation.py`, página "Análise de Threshold")

Score e decisão são conceitos diferentes (ver "Score, threshold e decisão"
acima) — esta página deixa isso literalmente manipulável: escolha um
modelo e mova um slider de threshold para ver, em tempo real, como
precisão/recall/F1/matriz de confusão mudam **sem retreinar nada**.

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
  suspeito" para os três, mas não uniformiza a escala — e esta página não
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

## 6. Metodologia de avaliação

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

## 7. Testes

```bash
pytest
```

Os testes (`tests/`) cobrem a camada de avaliação — separação entre escore
e decisão, AUC-PR, thresholds inválidos, validação cruzada (5 folds,
threshold calibrado só no treino do fold, resultado agregado com
mean/std), a varredura de threshold (`threshold_sweep`) e que o
`StandardScaler` nunca usa dados de teste/validation para se ajustar,
inclusive através da função de split centralizada
(`get_train_test_split`). Rodam sobre um dataset sintético pequeno (gerado
em `tests/conftest.py`), não sobre `data/raw/creditcard.csv` — não é
necessário ter o dataset baixado para rodar a suíte.

## 8. Próximos passos

- [x] Implementar o baseline estatístico, Regressão Logística e Isolation Forest (`src/models.py`).
- [x] Aplicação interativa (Streamlit) para apresentação (`app/`).
- [x] AUC-PR e separação explícita entre score e threshold (`src/models.py`).
- [x] Validação cruzada estratificada, só no treino (`src/evaluation.py`).
- [x] Fonte única do split treino/teste (`get_train_test_split`, `src/preprocessing.py`).
- [x] Página de Análise de Threshold interativa (`app/views/threshold_analysis.py`).
- [x] Testes unitários da camada de avaliação (`tests/`).
- [ ] Rodar `01_exploracao.ipynb` com o dataset real e preencher a seção de conclusões.
- [ ] Ajustar hiperparâmetros e comparar com balanceamento via SMOTE (não só `class_weight`).
- [ ] Notebook comparativo formalizando as métricas das três abordagens (hoje só na aplicação).
