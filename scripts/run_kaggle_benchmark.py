"""Avaliação temporal, volume de alertas e limiar por custo no benchmark
público do Kaggle (Credit Card Fraud Detection).

Toda a lógica está em src/benchmark.py (testada); aqui é só orquestração e
saída. Ver docs/VALIDACAO.md para o que estes números provam e o que não.

Uso:
    python scripts/run_kaggle_benchmark.py
    python scripts/run_kaggle_benchmark.py --custo-revisao 10 --teste 0.3

Requer data/raw/creditcard.csv (ver README, seção 3).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# O console do Windows (cp1252) não representa alguns caracteres; troca-os
# por "?" em vez de quebrar a execução no meio do relatório.
sys.stdout.reconfigure(errors="replace")

from src.benchmark import (  # noqa: E402
    DEFAULT_REVIEW_COST,
    alert_volume,
    compare_splits,
    cost_curve,
    cost_optimal_threshold,
    model_scores_temporal,
)
from src.data_loader import RAW_DATA_PATH, load_raw_data  # noqa: E402

LINE = "=" * 78


def title(text: str) -> None:
    print(f"\n{LINE}\n{text}\n{LINE}")


def brl(value: float) -> str:
    return "R$ " + f"{value:,.2f}".replace(",", "#").replace(".", ",").replace("#", ".")


def main(argv=None, path=RAW_DATA_PATH) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teste", type=float, default=0.2, help="fração mais recente usada como teste (padrão 0,2)")
    parser.add_argument(
        "--custo-revisao", type=float, default=DEFAULT_REVIEW_COST,
        help=f"custo de revisar um alerta, em R$ (padrão {DEFAULT_REVIEW_COST:g}; premissa, ajuste ao seu caso)",
    )  # fmt: skip
    args = parser.parse_args(argv)

    try:
        df = load_raw_data(path)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    print(LINE)
    print("SENTINEL - benchmark público (Kaggle): avaliação temporal, alertas e custo")
    print(LINE)
    print(
        "Dados reais e rotulados, mas de outro contexto (cartões europeus, 2 dias de\n"
        "2013, features anonimizadas por PCA). Validam as TÉCNICAS, não o motor do\n"
        "Sentinel, que usa sinais que este dataset não tem."
    )

    title(f"1. Split aleatório × temporal (teste = {args.teste:.0%} das transações)")
    table = compare_splits(df, test_size=args.teste)
    print(f"  {'modelo':22s} {'split':10s} {'PR-AUC':>7s} {'ROC-AUC':>8s} {'fraudes no teste':>17s}")
    for r in table.itertuples():
        print(f"  {r.modelo:22s} {r.split:10s} {r.pr_auc:7.3f} {r.roc_auc:8.3f} {r.fraudes_teste:17d}")
    print("  Temporal = treino no passado, teste no futuro (como em produção). Uma queda\n"
          "  em relação ao aleatório mostra quanto o split aleatório é otimista.")  # fmt: skip

    y_test, scores, amount, times = model_scores_temporal(df, test_size=args.teste)

    title("2. Volume de alertas no teste temporal (orçamento = % das transações alertadas)")
    for model, score in scores.items():
        print(f"\n  {model}")
        print(f"  {'orçamento':>9s} {'alertas':>8s} {'por dia':>8s} {'fraudes':>8s} {'precisão':>9s} {'recall':>7s} {'valor coberto':>14s}")
        for r in alert_volume(y_test, score, amount, times).itertuples():
            print(
                f"  {r.orcamento:9.2%} {r.alertas:8d} {r.alertas_por_dia:8.0f} {r.fraudes_pegas:8d} "
                f"{r.precisao:9.1%} {r.recall:7.1%} {r.valor_fraude_coberto:14.1%}"
            )

    title(f"3. Limiar por custo (revisão = {brl(args.custo_revisao)}; fraude perdida = o próprio valor)")
    no_alert = float(amount[y_test == 1].sum())
    print(f"  Sem nenhum alerta, o prejuízo seria {brl(no_alert)} (todas as fraudes do teste).")
    for model, score in scores.items():
        best = cost_optimal_threshold(cost_curve(y_test, score, amount, review_cost=args.custo_revisao))
        print(
            f"  {model:22s} {int(best.alertas):6d} alertas · {int(best.fn):3d} fraudes perdidas · "
            f"custo {brl(best.custo_total)} ({best.custo_total / no_alert:.0%} do prejuízo sem alertas)"
        )
    print("  ATENÇÃO: o limiar ótimo foi achado no próprio teste, para DESCREVER a curva de\n"
          "  custo. Para decidir um limiar, calcule-o num período de validação anterior.")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
