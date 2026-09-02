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
Na primeira vez que a página **Comparação de Modelos** (ou **Demo ao Vivo**)
é aberta, o app treina os três modelos — leva alguns segundos; depois disso
fica em cache (`st.cache_resource`) e as trocas de página/filtro são
instantâneas, sem retreinar nada.

Quatro páginas, navegáveis pela barra lateral:

| Página | O que mostra |
|---|---|
| **Visão Geral** | Objetivo do projeto e resumo visual da análise SWOT/GUT |
| **Análise Exploratória** | Distribuição de classes, `Amount` por classe e correlações — versão interativa do notebook |
| **Comparação de Modelos** | Matriz de confusão, precisão, recall, F1 e curvas ROC dos 3 modelos lado a lado |
| **Demo ao Vivo** | Escolha uma transação real do conjunto de teste e veja a classificação de qualquer um dos 3 modelos na hora, com um gauge de score de risco |

### Como o desbalanceamento é tratado em cada modelo (`src/models.py`)

- **Baseline estatístico**: escore de anomalia = distância euclidiana ao
  quadrado no espaço `V1`..`V28` (uma transação "típica" fica perto da
  origem desse espaço); o limiar de corte é calibrado por percentil, sem
  usar rótulos.
- **Regressão Logística**: `class_weight="balanced"` — penaliza mais o erro
  na classe minoritária (fraude), sem precisar reamostrar os dados.
- **Isolation Forest**: não-supervisionado; `contamination` (proporção
  esperada de anomalias, estimada do treino) calibra o limiar de decisão.

Essa escolha prioriza um app rápido de treinar/interagir. O SMOTE/undersampling
implementado em [`src/preprocessing.py`](src/preprocessing.py) continua
disponível e é o que o notebook de exploração usa como exemplo alternativo
de tratamento de desbalanceamento.

## 6. Metodologia de avaliação

Como a maior fraqueza identificada na análise SWOT/GUT é a pouca experiência
do time com validação estatística, o projeto segue algumas regras fixas
(implementadas em [`src/preprocessing.py`](src/preprocessing.py)):

- **Nunca usar acurácia isolada como métrica de sucesso.** Com 99,83% das
  transações sendo legítimas, um modelo que nunca detecta fraude nenhuma
  ainda "acerta" 99,83% das vezes. Métricas usadas: matriz de confusão,
  precisão, recall, F1-score e AUC-ROC/AUC-PR.
- **Split estratificado** (`stratify=y`) para preservar a proporção de
  fraudes em treino e teste.
- **Balanceamento (SMOTE/undersampling) aplicado só no treino**, depois do
  split — nunca no teste, para as métricas continuarem representando a
  proporção real de fraudes.
- **MVP antes de otimização:** um baseline estatístico simples (regras) roda
  de ponta a ponta antes de investir em ajuste fino de modelos de ML.

## 7. Próximos passos

- [x] Implementar o baseline estatístico, Regressão Logística e Isolation Forest (`src/models.py`).
- [x] Aplicação interativa (Streamlit) para apresentação (`app/`).
- [ ] Rodar `01_exploracao.ipynb` com o dataset real e preencher a seção de conclusões.
- [ ] Ajustar hiperparâmetros e comparar com balanceamento via SMOTE (não só `class_weight`/`contamination`).
- [ ] Notebook comparativo formalizando as métricas das três abordagens (hoje só na aplicação).
