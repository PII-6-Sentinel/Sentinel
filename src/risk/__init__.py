"""Risk Engine do Sentinel — opera só sobre o domínio sintético.

- `dataset`: tabela de risco (features comportamentais + colunas que as
  regras precisam, como distância em km e confiança do dispositivo).
- `behavior_score`: score por regras explicáveis (0-100).
- `ml_score`: Regressão Logística treinada só nos dados sintéticos (0-100).
- `engine`: combina os dois num Risk Score final, nível e ação recomendada.
- `new_transaction`: calcula as features de uma transação nova a partir do
  histórico do cliente no banco e chama o engine.
- `evaluation`: split por cliente, calibração de limiares e métricas.

Não usa a camada Kaggle (`src/models.py` etc.): aqueles modelos operam em
V1–V28 (PCA) e não conseguem pontuar transações sintéticas.
"""
