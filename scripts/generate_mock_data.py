"""Gera o domínio sintético antifraude e popula data/synthetic/sentinel.db.

Toda a lógica de geração vive em src/synthetic/ (testável isoladamente,
sem depender deste script) - aqui é só orquestração: cria o banco, chama
o gerador, imprime um resumo legível.

Uso:
    python scripts/generate_mock_data.py
    python scripts/generate_mock_data.py --n-customers 150 --seed 42
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine  # noqa: E402

from src.synthetic.generator import DEFAULT_N_CUSTOMERS, DEFAULT_SEED, populate_database  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "synthetic" / "sentinel.db"


def _print_summary(summary: dict) -> None:
    print()
    print("=" * 60)
    print("Domínio sintético gerado")
    print("=" * 60)
    print(f"Clientes: {summary['n_customers']}")
    for profile, count in sorted(summary["profile_counts"].items()):
        pct = count / summary["n_customers"] * 100
        print(f"  - {profile:15s}: {count:4d} ({pct:.1f}%)")

    print(f"\nEstabelecimentos: {summary['n_merchants']}")

    print(f"\nTransações totais: {summary['n_transactions']}")
    print(f"Fraudes: {summary['n_fraud']} ({summary['fraud_rate'] * 100:.2f}%)")
    print("Por cenário:")
    for scenario, count in sorted(summary["fraud_by_scenario"].items()):
        print(f"  - {scenario:22s}: {count:4d}")

    print("\nExemplos de transações fraudulentas:")
    for ex in summary["example_frauds"]:
        print(f"\n  [{ex['scenario']}] cliente: {ex['customer']} ({ex['profile']})")
        print(f"    Valor: R$ {ex['amount']:.2f}  (média do cliente: R$ {ex['avg_amount_baseline']:.2f})")
        print(f"    Cidade natal: {ex['home_city']}  ->  transação em: {ex['merchant_city']}")
        print(f"    Distância de casa: {ex['distance_from_home_km']:.1f} km")
        print(f"    Data/hora: {ex['timestamp']}")

    print()
    print("=" * 60)
    print(f"Banco salvo em: {DB_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-customers", type=int, default=DEFAULT_N_CUSTOMERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--force", action="store_true",
        help="Sobrescreve o banco existente em vez de pedir confirmação.",
    )
    args = parser.parse_args()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists() and not args.force:
        print(f"{DB_PATH.relative_to(PROJECT_ROOT)} já existe. Use --force para sobrescrever.")
        sys.exit(1)
    if DB_PATH.exists():
        DB_PATH.unlink()

    engine = create_engine(f"sqlite:///{DB_PATH}")
    summary = populate_database(engine, seed=args.seed, n_customers=args.n_customers)
    _print_summary(summary)


if __name__ == "__main__":
    main()
