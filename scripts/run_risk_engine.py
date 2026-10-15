"""Roda a avaliação completa do Risk Engine sobre o domínio sintético e
imprime um relatório: só regras vs só ML vs combinado, recall por cenário,
distribuição por nível e 3 exemplos explicados.

Toda a lógica está em src/risk/ (testada); aqui é só orquestração e saída.

Uso:
    python scripts/run_risk_engine.py

Requer data/synthetic/sentinel.db (gere com scripts/generate_mock_data.py).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# O console do Windows (cp1252) não representa alguns caracteres; troca-os
# por "?" em vez de quebrar a execução no meio do relatório.
sys.stdout.reconfigure(errors="replace")

from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from src.risk.dataset import build_risk_table  # noqa: E402
from src.risk.engine import ACTION_BY_LEVEL, ACTION_DESCRIPTION  # noqa: E402
from src.risk.report import (  # noqa: E402
    APPROACH_COMBINED,
    APPROACH_COMBINED_5050,
    APPROACH_ML,
    APPROACH_RULES,
    MIN_RULES_WEIGHT,
    TARGET_BLOCK_PRECISION,
    TARGET_RECALL,
    evaluate_risk_engine,
    pick_examples,
)
from src.synthetic.models import Customer, Merchant, Transaction  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "synthetic" / "sentinel.db"
LINE = "=" * 78


def pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def title(text: str) -> None:
    print()
    print(LINE)
    print(text)
    print(LINE)


def main() -> None:
    if not DB_PATH.exists():
        print(f"{DB_PATH.relative_to(PROJECT_ROOT)} não existe. Rode antes: python scripts/generate_mock_data.py")
        sys.exit(1)

    engine_db = create_engine(f"sqlite:///{DB_PATH}")
    with Session(engine_db) as session:
        table = build_risk_table(session)
    ev = evaluate_risk_engine(table)
    bands = ev.engine.config.bands

    print(LINE)
    print("SENTINEL - Risk Engine: relatório de avaliação")
    print(LINE)
    print(
        "AVISO: dados SIMULADOS. Os rótulos de fraude foram gerados a partir das\n"
        "mesmas features que o motor usa (circularidade). Estes números validam a\n"
        "LÓGICA do sistema; não provam desempenho em fraude real."
    )

    # 1 ------------------------------------------------------------------
    title("1. Split por cliente (nenhum cliente em mais de um conjunto)")
    for name, part in (("treino", ev.split.train), ("validação", ev.split.validation), ("teste", ev.split.test)):
        print(
            f"  {name:10s} clientes={part['customer_id'].nunique():4d}  "
            f"transações={len(part):5d}  fraudes={int(part['is_fraud'].sum()):4d}"
        )
    ids = [set(p["customer_id"]) for p in (ev.split.train, ev.split.validation, ev.split.test)]
    overlap = len(ids[0] & ids[1]) + len(ids[0] & ids[2]) + len(ids[1] & ids[2])
    print(f"  clientes em mais de um conjunto: {overlap}")

    # 2 ------------------------------------------------------------------
    title("2. Escolha do peso ML/regras - curva de PR-AUC na VALIDAÇÃO")
    print(f"  regra: maior PR-AUC na validação com peso das regras >= {MIN_RULES_WEIGHT:.0%}")
    print("         (empate: maior peso de regras). Declarada antes de olhar o teste.")
    chosen = ev.engine.config.behavior_weight
    print(f"  {'regras':>7s} {'ML':>5s} {'PR-AUC':>8s} {'ROC-AUC':>8s}")
    for _, r in ev.weight_sweep.iterrows():
        notes = []
        if abs(r["peso_regras"] - chosen) < 1e-9:
            notes.append("<- escolhido")
        if abs(r["peso_regras"] - 0.5) < 1e-9:
            notes.append("(50/50)")
        if not r["elegivel"]:
            notes.append("fora da restrição")
        print(
            f"  {r['peso_regras']:7.0%} {r['peso_ml']:5.0%} {r['pr_auc']:8.3f} {r['roc_auc']:8.3f}  {' '.join(notes)}"
        )
    print(f"  motor: {chosen:.0%} regras / {ev.engine.config.ml_weight:.0%} ML")

    # 3 ------------------------------------------------------------------
    title("3. Limiares calibrados na VALIDAÇÃO (o teste não foi usado)")
    print(f"  critério de revisão : maior limiar com recall >= {TARGET_RECALL:.0%} das fraudes")
    print(f"  critério de bloqueio: menor limiar com precisão >= {TARGET_BLOCK_PRECISION:.0%}")
    print(f"  motor (combinado): baixo 0-{bands.review_threshold} | médio {bands.review_threshold + 1}-{bands.block_threshold} | alto {bands.block_threshold + 1}-100   (padrão: 0-30 | 31-70 | 71-100)")
    r50, b50 = ev.review_thresholds[APPROACH_COMBINED_5050], ev.block_thresholds[APPROACH_COMBINED_5050]
    print(f"  combinado 50/50  : baixo 0-{r50} | médio {r50 + 1}-{b50} | alto {b50 + 1}-100")
    print("  escala exibida: o Risk Score mostrado remapeia esses limiares para 30 e 70")
    print("  (mesmas decisões; ex.: score interno "
          f"{bands.review_threshold + 1} -> exibido {bands.to_display(bands.review_threshold + 1)})")
    print(f"  só regras: sinaliza se score > {ev.review_thresholds[APPROACH_RULES]}")
    print(f"  só ML    : sinaliza se score > {ev.review_thresholds[APPROACH_ML]}")

    # 4 ------------------------------------------------------------------
    title("4. Comparação no TESTE (sinalizada = não aprovada automaticamente)")
    print(f"  {'abordagem':32s} {'PR-AUC':>7s} {'ROC-AUC':>8s} {'Precisão':>9s} {'Recall':>7s} {'F1':>6s}   TP  FP  FN")
    for _, r in ev.comparison.iterrows():
        print(
            f"  {r['abordagem']:32s} {r['pr_auc']:7.3f} {r['roc_auc']:8.3f} {r['precision']:9.3f} "
            f"{r['recall']:7.3f} {r['f1']:6.3f}  {r['tp']:3d} {r['fp']:3d} {r['fn']:3d}"
        )
    comp = ev.comparison.set_index("abordagem")
    c, m, r_ = comp.loc[APPROACH_COMBINED], comp.loc[APPROACH_ML], comp.loc[APPROACH_RULES]
    c50 = comp.loc[APPROACH_COMBINED_5050]
    print()
    for metric, label in (("pr_auc", "PR-AUC"), ("f1", "F1")):
        best_part = max(m[metric], r_[metric])
        if c[metric] > best_part:
            print(f"  {label}: o motor ({c[metric]:.3f}) SUPERA a melhor parte isolada ({best_part:.3f}).")
        else:
            print(
                f"  {label}: o motor ({c[metric]:.3f}) NÃO supera a melhor parte isolada "
                f"({best_part:.3f}) - a combinação não melhora esta métrica no teste."
            )
        verb = "melhor" if c[metric] > c50[metric] else ("igual" if c[metric] == c50[metric] else "PIOR")
        print(f"  {label}: motor com o peso escolhido ({c[metric]:.3f}) {verb} que o 50/50 ({c50[metric]:.3f}).")

    print()
    print("  Estabilidade - PR-AUC em validação cruzada 5-fold por cliente (tabela inteira):")
    for _, s in ev.stability.iterrows():
        print(f"    {s['abordagem']:22s} {s['pr_auc_media']:.3f} +/- {s['pr_auc_desvio']:.3f}")

    # 5 ------------------------------------------------------------------
    title("5. Recall por cenário de fraude (TESTE) - sinalizadas / total")
    approaches = (APPROACH_RULES, APPROACH_ML, APPROACH_COMBINED, APPROACH_COMBINED_5050)
    print(f"  {'cenário':24s}" + "".join(f"{a:>22s}" for a in approaches))
    for scenario, row in ev.scenario_recall.iterrows():
        cells = "".join(
            f"{int(row[(a, 'detected')]):>8d}/{int(row[(a, 'total')]):<3d} ({row[(a, 'recall')] * 100:5.1f}%)"
            for a in approaches
        )
        print(f"  {scenario:24s}{cells}")

    # 6 ------------------------------------------------------------------
    title("6. Distribuição por nível e ação (TESTE, motor com limiares calibrados)")
    print(f"  {'nível':7s} {'ação':28s} {'transações':>11s} {'% do total':>11s} {'fraudes':>8s} {'taxa fraude':>12s} {'% das fraudes':>14s}")
    for level, r in ev.level_distribution.iterrows():
        action = ACTION_DESCRIPTION[ACTION_BY_LEVEL[level]]
        print(
            f"  {level:7s} {action:28s} {int(r['transacoes']):11d} {pct(r['pct_transacoes']):>11s} "
            f"{int(r['fraudes']):8d} {pct(r['taxa_fraude']):>12s} {pct(r['pct_das_fraudes']):>14s}"
        )

    # 7 ------------------------------------------------------------------
    title("7. O que o modelo de ML aprendeu (coeficientes sobre features padronizadas)")
    for feat, coef in sorted(ev.coefficients.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {feat:22s} {coef:+.2f}")

    # 8 ------------------------------------------------------------------
    title("8. Exemplos explicados (TESTE)")
    examples = pick_examples(ev.test_scores)
    with Session(engine_db) as session:
        for level, row in examples.items():
            tx_id = int(row["transaction_id"])
            tx, customer, merchant = session.execute(
                select(Transaction, Customer, Merchant)
                .join(Customer, Customer.id == Transaction.customer_id)
                .join(Merchant, Merchant.id == Transaction.merchant_id)
                .where(Transaction.id == tx_id)
            ).one()
            result = ev.engine.assess(row[[*row.index]].to_dict())
            truth = f"FRAUDE ({tx.fraud_scenario})" if tx.is_fraud else "legítima"
            print()
            print(f"  [{result.level.upper()}] transação #{tx_id} - rótulo real: {truth}")
            print(f"    cliente: {customer.name} ({customer.behavior_profile.value}, mora em {customer.home_city})")
            print(
                f"    R$ {tx.amount:,.2f} em {merchant.name} ({merchant.city}), {tx.timestamp:%d/%m/%Y %H:%M}"
                f" - média do cliente R$ {customer.avg_amount_baseline:,.2f}"
            )
            print(
                f"    Risk Score {result.risk_score}/100 [interno {result.raw_score}] "
                f"(ML {result.ml_score:.1f}, regras {result.behavior_score:.1f})"
                f" -> {ACTION_DESCRIPTION[result.action]}"
            )
            if result.reasons:
                for reason in result.reasons:
                    print(f"      - {reason}")
            else:
                print("      - nenhum sinal de risco")
    print()


if __name__ == "__main__":
    main()
