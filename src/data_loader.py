"""Carregamento do dataset Credit Card Fraud Detection.

Mantido separado do pré-processamento para que carregar os dados brutos
e transformá-los sejam etapas independentes e reutilizáveis — cada notebook
ou script de modelo importa só o que precisa.
"""

from pathlib import Path

import pandas as pd

# Caminho do dataset relativo à raiz do projeto, não ao diretório de execução,
# para que o loader funcione igual seja chamado de um notebook em notebooks/
# ou de um script em src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

EXPECTED_COLUMNS = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
)


def load_raw_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Carrega o dataset bruto `creditcard.csv`.

    Levanta um erro claro e acionável se o arquivo não existir, já que esse
    é o primeiro obstáculo que qualquer pessoa do time vai encontrar ao
    clonar o repositório (o CSV não é versionado — ver README).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em '{path}'.\n"
            "Baixe o creditcard.csv do Kaggle e coloque em data/raw/ "
            "(instruções no README.md, seção 'Obtendo o dataset')."
        )

    df = pd.read_csv(path)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Colunas esperadas ausentes no CSV: {sorted(missing)}. "
            "Verifique se o arquivo baixado é a versão original do dataset."
        )

    return df


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna contagem e proporção de cada classe (0 = legítima, 1 = fraude).

    Extraído como função própria porque é a primeira checagem que qualquer
    pessoa deve fazer neste dataset — confirmar o desbalanceamento antes de
    tirar qualquer outra conclusão.
    """
    counts = df["Class"].value_counts().rename({0: "legítima", 1: "fraude"})
    proportions = df["Class"].value_counts(normalize=True).rename(
        {0: "legítima", 1: "fraude"}
    )
    return pd.DataFrame({"contagem": counts, "proporção": proportions})
