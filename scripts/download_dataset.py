"""Baixa o dataset Credit Card Fraud Detection do Kaggle para data/raw/.

Requer credenciais da API do Kaggle configuradas (arquivo kaggle.json).
Se você preferir baixar manualmente pelo navegador, veja o README.md,
seção "Obtendo o dataset" — o resultado final é o mesmo: um arquivo
data/raw/creditcard.csv.

Uso:
    python scripts/download_dataset.py
"""

import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATASET = "mlg-ulb/creditcardfraud"


def main() -> None:
    target_csv = RAW_DIR / "creditcard.csv"
    if target_csv.exists():
        print(f"Dataset já existe em {target_csv}, nada a fazer.")
        return

    try:
        # Import feito dentro da função: a biblioteca 'kaggle' valida as
        # credenciais (~/.kaggle/kaggle.json) assim que é importada, então
        # um import no topo do arquivo quebraria o script inteiro mesmo
        # para quem só quer ver a mensagem de ajuda.
        from kaggle.api.kaggle_api_extended import KaggleApi
    except OSError:
        print(
            "Credenciais do Kaggle não encontradas.\n"
            "Configure kaggle.json (veja README.md, seção 'Obtendo o "
            "dataset') ou baixe o arquivo manualmente pelo navegador.",
            file=sys.stderr,
        )
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Baixando '{DATASET}' para {RAW_DIR} ...")
    api.dataset_download_files(DATASET, path=str(RAW_DIR), unzip=False)

    zip_path = RAW_DIR / "creditcardfraud.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()

    if target_csv.exists():
        print(f"Concluído: {target_csv}")
    else:
        print(
            "Download terminou mas creditcard.csv não foi encontrado — "
            "verifique o conteúdo de data/raw/ manualmente.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
