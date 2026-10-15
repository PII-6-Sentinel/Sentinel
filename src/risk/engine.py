"""Risk Engine — combina o score de ML e o score por regras num Risk Score
final 0-100, um nível, uma ação recomendada e os motivos.

Combinação (por que 50/50 como ponto de partida)
------------------------------------------------
score = peso_ml × ml_score + peso_regras × behavior_score, arredondado.

- As regras são explicáveis por construção e codificam conhecimento de
  domínio que não depende dos rótulos sintéticos.
- O ML aprende COMBINAÇÕES de sinais a partir dos dados, mas os rótulos
  foram gerados a partir das mesmas features (circularidade) — confiar
  só nele seria confiar no próprio gerador.
- Pesos iguais não privilegiam nenhum dos dois a priori; por isso 50/50 é
  o padrão de `EngineConfig`. O motor AVALIADO usa o peso escolhido na
  validação por uma regra declarada (maior PR-AUC com no mínimo 30% de
  peso nas regras, por explicabilidade) — ver `src/risk/report.py` e
  docs/RISK_ENGINE.md. O teste nunca é usado para escolher pesos.

Piso por regra dura (por que "médio" e não "alto")
--------------------------------------------------
Viagem impossível é fisicamente incompatível com um único titular honesto,
então nunca deve ser aprovada automaticamente: ela força o nível a ser ao
menos "médio" (revisão). Não força "alto" (bloqueio) porque a transação
sinalizada pode ser a LEGÍTIMA — se a compra anterior é que foi a fraude
(num lugar distante), a compra seguinte do cliente real é que parece
impossível. Bloquear automaticamente puniria o cliente; uma revisão humana
resolve o caso. Se o score já for alto por conta própria, continua alto.

Faixas de decisão
-----------------
Padrão: 0-30 baixo → aprovar; 31-70 médio → revisar; 71-100 alto →
bloquear. São configuráveis (`DecisionBands`) e, no script de avaliação,
calibradas num conjunto de VALIDAÇÃO (ver `src/risk/evaluation.py`).

Escala de exibição
------------------
Com faixas calibradas (ex.: 56/71), o score interno (`raw_score`) é
remapeado para a escala exibida (`risk_score`) por uma transformação
monotônica por partes que leva os limiares calibrados para 30 e 70.
Assim, quem opera sempre lê "acima de 30 revisa, acima de 70 bloqueia".
Nível e ação continuam decididos pelo score interno — a transformação
só muda o número exibido, nunca a decisão (testado exaustivamente).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.risk.behavior_score import BehaviorConfig, behavior_score
from src.risk.ml_score import MLScorer

LEVEL_LOW, LEVEL_MEDIUM, LEVEL_HIGH = "baixo", "médio", "alto"
ACTION_APPROVE, ACTION_REVIEW, ACTION_BLOCK = "aprovar", "revisar", "bloquear"

ACTION_BY_LEVEL = {
    LEVEL_LOW: ACTION_APPROVE,
    LEVEL_MEDIUM: ACTION_REVIEW,
    LEVEL_HIGH: ACTION_BLOCK,
}

# Descrição legível de cada ação, para relatórios e interface.
ACTION_DESCRIPTION = {
    ACTION_APPROVE: "aprovar",
    ACTION_REVIEW: "revisar manualmente",
    ACTION_BLOCK: "bloquear / revisão urgente",
}


@dataclass(frozen=True)
class DecisionBands:
    """Faixas sobre o Risk Score inteiro 0-100.

    score ≤ review_threshold → baixo; ≤ block_threshold → médio; acima → alto.
    Padrão 30/70: 0-30 baixo, 31-70 médio, 71-100 alto.
    """

    review_threshold: int = 30
    block_threshold: int = 70

    def __post_init__(self):
        if not 0 <= self.review_threshold < self.block_threshold < 100:
            raise ValueError(
                f"faixas inválidas: precisa 0 ≤ revisão ({self.review_threshold}) "
                f"< bloqueio ({self.block_threshold}) < 100"
            )

    def level(self, score: int) -> str:
        if score <= self.review_threshold:
            return LEVEL_LOW
        if score <= self.block_threshold:
            return LEVEL_MEDIUM
        return LEVEL_HIGH

    def to_display(self, score: int) -> int:
        """Score interno → escala exibida, em que os limiares desta faixa
        viram 30 e 70 (os de `DecisionBands()`). Monotônica por partes:
        [0, revisão] → [0, 30]; (revisão, bloqueio] → [31, 70];
        (bloqueio, 100] → [71, 100]. Com as faixas padrão é a identidade.

        O arredondamento (piso no 1º trecho, teto nos demais) garante que
        o nível lido na escala exibida com as faixas padrão é sempre o
        mesmo nível do score interno com estas faixas.
        """
        r, b = self.review_threshold, self.block_threshold
        dr, db = DISPLAY_BANDS.review_threshold, DISPLAY_BANDS.block_threshold
        s = int(min(100, max(0, score)))
        if s <= r:
            return 0 if r == 0 else (s * dr) // r
        if s <= b:
            return dr + math.ceil((s - r) * (db - dr) / (b - r))
        return db + math.ceil((s - b) * (100 - db) / (100 - b))

    def min_score(self, level: str) -> int:
        """Menor score inteiro que pertence ao nível."""
        return {LEVEL_LOW: 0, LEVEL_MEDIUM: self.review_threshold + 1, LEVEL_HIGH: self.block_threshold + 1}[level]


# Escala em que o score é EXIBIDO: limiares sempre em 30 e 70.
DISPLAY_BANDS = DecisionBands()


@dataclass(frozen=True)
class EngineConfig:
    ml_weight: float = 0.5
    behavior_weight: float = 0.5
    bands: DecisionBands = field(default_factory=DecisionBands)
    # Nível mínimo quando há viagem impossível (None desliga o piso).
    impossible_travel_floor: str | None = LEVEL_MEDIUM
    # A partir de que score de ML o modelo vira um "motivo" exibido.
    ml_reason_min_score: float = 50.0

    def __post_init__(self):
        if self.ml_weight < 0 or self.behavior_weight < 0:
            raise ValueError("pesos não podem ser negativos")
        if not math.isclose(self.ml_weight + self.behavior_weight, 1.0):
            raise ValueError("ml_weight + behavior_weight precisa somar 1")
        if self.impossible_travel_floor not in (None, LEVEL_MEDIUM, LEVEL_HIGH):
            raise ValueError("impossible_travel_floor precisa ser None, 'médio' ou 'alto'")


@dataclass(frozen=True)
class RiskResult:
    risk_score: int  # escala exibida: revisa acima de 30, bloqueia acima de 70
    ml_score: float
    behavior_score: float
    level: str
    action: str
    reasons: list[str]
    raw_score: int  # score interno, comparado com as faixas calibradas


def _round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(values, dtype=float) + 0.5).astype(int)


def combine_scores(ml, behavior, impossible_travel, config: EngineConfig) -> np.ndarray:
    """Risk Score inteiro 0-100 — a ÚNICA implementação da combinação,
    usada tanto por `RiskEngine.assess` quanto por `RiskEngine.assess_frame`.
    """
    raw = config.ml_weight * np.asarray(ml, dtype=float) + config.behavior_weight * np.asarray(
        behavior, dtype=float
    )
    score = _round_half_up(np.clip(raw, 0.0, 100.0))
    if config.impossible_travel_floor is not None:
        floor = config.bands.min_score(config.impossible_travel_floor)
        score = np.where(np.asarray(impossible_travel, dtype=bool), np.maximum(score, floor), score)
    return score


class RiskEngine:
    """Recebe as features de uma transação e devolve um `RiskResult`."""

    def __init__(
        self,
        ml_model: MLScorer,
        config: EngineConfig | None = None,
        behavior_config: BehaviorConfig | None = None,
    ):
        if not ml_model.is_fitted:
            raise ValueError("o RiskEngine precisa de um MLScorer já treinado")
        self.ml_model = ml_model
        self.config = config or EngineConfig()
        self.behavior_config = behavior_config or BehaviorConfig()

    def with_bands(self, bands: DecisionBands) -> "RiskEngine":
        """Mesmo modelo e pesos, outras faixas de decisão (ex.: calibradas)."""
        config = EngineConfig(
            ml_weight=self.config.ml_weight,
            behavior_weight=self.config.behavior_weight,
            bands=bands,
            impossible_travel_floor=self.config.impossible_travel_floor,
            ml_reason_min_score=self.config.ml_reason_min_score,
        )
        return RiskEngine(self.ml_model, config, self.behavior_config)

    # ------------------------------------------------------------------
    # uma transação, com motivos
    # ------------------------------------------------------------------

    def assess(self, features: Mapping[str, Any]) -> RiskResult:
        ml = float(self.ml_model.score(pd.DataFrame([dict(features)]))[0])
        behavior = behavior_score(features, self.behavior_config)
        impossible = bool(features["is_impossible_travel"])

        score = int(combine_scores([ml], [behavior.score], [impossible], self.config)[0])
        level = self.config.bands.level(score)
        reasons = self._reasons(ml, behavior, impossible, score)

        return RiskResult(
            risk_score=self.config.bands.to_display(score),
            ml_score=round(ml, 1),
            behavior_score=round(behavior.score, 1),
            level=level,
            action=ACTION_BY_LEVEL[level],
            reasons=reasons,
            raw_score=score,
        )

    def _reasons(self, ml: float, behavior, impossible: bool, final_score: int) -> list[str]:
        """Motivos ordenados pela contribuição de cada um ao score final."""
        cfg = self.config
        weighted: list[tuple[float, str]] = []

        # Regras: contribuição = peso × pontos (proporcionalmente reduzidos se
        # a soma passou do limite de 100 — o que conta é o que entrou no score).
        total_points = sum(h.points for h in behavior.hits)
        scale = min(1.0, behavior.score / total_points) if total_points else 0.0
        for hit in behavior.hits:
            weighted.append((cfg.behavior_weight * hit.points * scale, hit.reason))

        if ml >= cfg.ml_reason_min_score:
            weighted.append(
                (
                    cfg.ml_weight * ml,
                    f"Modelo estatístico: padrão semelhante a fraudes conhecidas (score ML {ml:.0f}/100)",
                )
            )

        pre_floor = int(combine_scores([ml], [behavior.score], [False], cfg)[0])
        if impossible and final_score > pre_floor:
            weighted.append(
                (
                    float(final_score - pre_floor),
                    f"Regra obrigatória: viagem impossível exige ao menos nível {cfg.impossible_travel_floor}",
                )
            )

        weighted.sort(key=lambda item: item[0], reverse=True)
        return [reason for _, reason in weighted]

    # ------------------------------------------------------------------
    # muitas transações (avaliação), sem motivos
    # ------------------------------------------------------------------

    def assess_frame(self, table: pd.DataFrame) -> pd.DataFrame:
        """Avalia uma tabela de risco inteira. Mesma combinação de `assess`
        (via `combine_scores`), sem montar os textos dos motivos.
        """
        ml = self.ml_model.score(table)
        behavior = np.array(
            [behavior_score(row, self.behavior_config).score for row in table.to_dict("records")]
        )
        score = combine_scores(ml, behavior, table["is_impossible_travel"].to_numpy(), self.config)
        levels = [self.config.bands.level(int(s)) for s in score]
        return pd.DataFrame(
            {
                "transaction_id": table["transaction_id"].to_numpy(),
                "ml_score": ml,
                "behavior_score": behavior,
                "raw_score": score,
                "risk_score": [self.config.bands.to_display(int(s)) for s in score],
                "level": levels,
                "action": [ACTION_BY_LEVEL[lv] for lv in levels],
            }
        )


def train_engine(train: pd.DataFrame, config: EngineConfig | None = None, seed: int = 42) -> RiskEngine:
    """Treina o modelo de ML em `train` (só o conjunto de TREINO) e devolve
    um RiskEngine com a configuração dada.
    """
    return RiskEngine(MLScorer(random_state=seed).fit(train), config)
