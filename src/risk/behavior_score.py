"""Score comportamental por regras — explicável por construção.

Cada sinal de risco vira uma regra que, se disparar, contribui com pontos e
um motivo legível em português. O score é a soma dos pontos, limitada a 100.

Todos os pesos e limiares vivem em `BehaviorConfig` (um único lugar). As
regras recebem as features de UMA transação (um dict ou qualquer Mapping com
as chaves de `src.risk.dataset.RISK_FEATURES`) e não leem banco.

Justificativa dos pesos (pontos máximos de cada regra)
------------------------------------------------------
- Viagem impossível (40): é a única evidência fisicamente incompatível com
  um único titular honesto — ninguém percorre 1.900 km em 1 hora. O peso
  maior reflete que ela quase não tem explicação legítima.
- Localização (até 25) e valor (até 25): sinais fortes, mas com explicações
  legítimas comuns (viagens, compras grandes planejadas). Pesam bem menos
  que uma impossibilidade física, e o mesmo entre si: nenhum dos dois é,
  sozinho, mais suspeito que o outro.
- Cidade nova (20): a primeira compra do cliente numa cidade onde ele
  nunca esteve é o que separa uma fraude geográfica de uma viagem — um
  viajante legítimo volta às mesmas cidades. Mas toda primeira visita
  legítima também dispara, então vale o mesmo que dispositivo novo (outro
  sinal de "primeira vez"), menos que uma impossibilidade física. Somada
  à localização (até 25), uma compra longe numa cidade inédita chega a 45
  pontos; uma viagem para uma cidade já conhecida fica nos 25.
- Dispositivo novo (20) + não confiável (15): um aparelho nunca visto é
  típico de tomada de conta, mas pessoas trocam de celular; por isso vale
  menos que localização/valor. Se ele também não é confiável, a suspeita
  sobe (35 no total), aproximando-se de uma viagem impossível.
- Horário (até 15): o sinal mais fraco — horários habituais são uma
  estimativa difusa e qualquer pessoa compra fora deles de vez em quando.
- Canal raro (até 15): compra online de quem quase só compra presencial
  (ou o contrário). É a marca do golpe com dados do cartão, mas qualquer
  cliente às vezes muda de hábito — vale o mesmo que o horário, outro
  hábito difuso.
- Aproximação acima da média (10): aproximação dispensa a senha — o que
  um fraudador com o cartão físico precisa. Sozinha não diz nada (é como
  muita gente paga), então só pesa quando o valor também destoa da média
  do cliente, e pouco: menos que qualquer sinal de "primeira vez".

Os pesos foram escolhidos por esse raciocínio de domínio, NÃO ajustados aos
dados sintéticos — ajustá-los aos dados repetiria a circularidade dos
rótulos (ver docs/RISK_ENGINE.md).

Rampas
------
Sinais contínuos (distância, valor, horário) usam uma rampa: abaixo de
`start` a regra não dispara; acima, vale `base` e cresce linearmente até
`max` em `full`. Uma rampa (em vez de um degrau) mantém o score contínuo,
o que permite ordenar transações por risco e calibrar limiares.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class BehaviorConfig:
    """Todos os pesos e limiares das regras — o único lugar onde ficam."""

    # Localização: km da residência, rampa em escala log10. 100 km = fora da
    # região metropolitana (clientes que não viajam nunca passam de ~10 km);
    # 2.000 km = outra região do país ou outro país.
    location_start_km: float = 100.0
    location_full_km: float = 2_000.0
    location_base_points: float = 5.0
    location_max_points: float = 25.0

    # Valor: múltiplo da média do próprio cliente. 2x ainda é uma compra
    # grande plausível; 6x ou mais é atípico para qualquer perfil.
    amount_start_ratio: float = 2.0
    amount_full_ratio: float = 6.0
    amount_base_points: float = 5.0
    amount_max_points: float = 25.0

    # Horário: horas fora da janela habitual do cliente.
    hour_start_gap_h: float = 0.0
    hour_full_gap_h: float = 4.0
    hour_base_points: float = 5.0
    hour_max_points: float = 15.0

    # Cidade nunca vista no histórico do cliente (nem a cidade natal).
    new_city_points: float = 20.0

    # Dispositivo.
    new_device_points: float = 20.0
    untrusted_device_points: float = 15.0

    # Canal raro: `channel_rarity` (1 - fração das compras anteriores no
    # mesmo canal). 0,7 = canal usado em no máximo 30% das compras.
    channel_rarity_start: float = 0.7
    channel_rarity_full: float = 1.0
    channel_base_points: float = 5.0
    channel_max_points: float = 15.0

    # Aproximação (sem senha) com valor a partir de 1,5x a média do cliente.
    contactless_min_ratio: float = 1.5
    contactless_points: float = 10.0

    # Viagem impossível (velocidade implícita > 900 km/h, ver features.py).
    impossible_travel_points: float = 40.0

    max_score: float = 100.0


@dataclass(frozen=True)
class RuleHit:
    """Uma regra que disparou: identificador, pontos e motivo legível."""

    rule: str
    points: float
    reason: str


@dataclass(frozen=True)
class BehaviorResult:
    score: float
    hits: list[RuleHit] = field(default_factory=list)


def ramp(value: float, start: float, full: float, base: float, maximum: float) -> float:
    """0 até `start` (inclusive); depois `base` crescendo até `maximum` em `full`."""
    if value <= start:
        return 0.0
    fraction = min(1.0, (value - start) / (full - start))
    return base + (maximum - base) * fraction


def _fmt_int(value: float) -> str:
    """2727.4 -> '2.727' (separador de milhar brasileiro)."""
    return f"{round(value):,}".replace(",", ".")


def _fmt_dec(value: float) -> str:
    """8.94 -> '8,9'."""
    return f"{value:.1f}".replace(".", ",")


# --------------------------------------------------------------------------
# Regras — cada uma recebe as features e a config, devolve RuleHit ou None
# --------------------------------------------------------------------------


def rule_location(f: Mapping[str, Any], c: BehaviorConfig) -> RuleHit | None:
    km = float(f["distance_from_home_km"])
    if km <= 0:
        return None
    points = ramp(
        math.log10(km),
        math.log10(c.location_start_km),
        math.log10(c.location_full_km),
        c.location_base_points,
        c.location_max_points,
    )
    if points == 0:
        return None
    return RuleHit("localizacao", points, f"Localização incomum: {_fmt_int(km)} km da residência")


def rule_amount(f: Mapping[str, Any], c: BehaviorConfig) -> RuleHit | None:
    ratio = float(f["amount_deviation"]) + 1.0  # valor / média do cliente
    points = ramp(ratio, c.amount_start_ratio, c.amount_full_ratio, c.amount_base_points, c.amount_max_points)
    if points == 0:
        return None
    return RuleHit("valor", points, f"Valor {_fmt_dec(ratio)}x acima da média do cliente")


def rule_hour(f: Mapping[str, Any], c: BehaviorConfig) -> RuleHit | None:
    gap_h = float(f["unusual_hour"]) * 12.0  # unusual_hour = horas fora / 12
    points = ramp(gap_h, c.hour_start_gap_h, c.hour_full_gap_h, c.hour_base_points, c.hour_max_points)
    if points == 0:
        return None
    return RuleHit("horario", points, f"Horário fora da janela habitual ({_fmt_dec(gap_h)} h fora)")


def rule_new_city(f: Mapping[str, Any], c: BehaviorConfig) -> RuleHit | None:
    if not bool(f["new_city"]):
        return None
    city = f.get("merchant_city")
    where = f" ({city})" if city else ""
    return RuleHit("cidade_nova", c.new_city_points, f"Primeira compra do cliente nesta cidade{where}")


def rule_new_device(f: Mapping[str, Any], c: BehaviorConfig) -> RuleHit | None:
    if not bool(f["new_device"]):
        return None
    return RuleHit("dispositivo_novo", c.new_device_points, "Dispositivo nunca usado antes por este cliente")


def rule_untrusted_device(f: Mapping[str, Any], c: BehaviorConfig) -> RuleHit | None:
    if not bool(f["device_untrusted"]):
        return None
    return RuleHit("dispositivo_nao_confiavel", c.untrusted_device_points, "Dispositivo não confiável")


def rule_impossible_travel(f: Mapping[str, Any], c: BehaviorConfig) -> RuleHit | None:
    if not bool(f["is_impossible_travel"]):
        return None
    speed = float(f["implied_speed_kmh"])
    return RuleHit(
        "viagem_impossivel",
        c.impossible_travel_points,
        f"Viagem impossível: {_fmt_int(speed)} km/h entre duas compras",
    )


def rule_channel(f: Mapping[str, Any], c: BehaviorConfig) -> RuleHit | None:
    rarity = float(f["channel_rarity"])
    points = ramp(rarity, c.channel_rarity_start, c.channel_rarity_full, c.channel_base_points, c.channel_max_points)
    if points == 0:
        return None
    channel = "online" if bool(f["is_online"]) else "presencial"
    share = round((1.0 - rarity) * 100)
    return RuleHit("canal_raro", points, f"Compra {channel}, canal raro para o cliente ({share}% das compras anteriores)")


def rule_contactless(f: Mapping[str, Any], c: BehaviorConfig) -> RuleHit | None:
    ratio = float(f["amount_deviation"]) + 1.0
    if not bool(f["is_contactless"]) or ratio < c.contactless_min_ratio:
        return None
    return RuleHit(
        "aproximacao",
        c.contactless_points,
        f"Pagamento por aproximação, sem senha, {_fmt_dec(ratio)}x a média do cliente",
    )


RULES: tuple[Callable[[Mapping[str, Any], BehaviorConfig], RuleHit | None], ...] = (
    rule_impossible_travel,
    rule_location,
    rule_amount,
    rule_new_city,
    rule_new_device,
    rule_untrusted_device,
    rule_hour,
    rule_channel,
    rule_contactless,
)


def behavior_score(features: Mapping[str, Any], config: BehaviorConfig | None = None) -> BehaviorResult:
    """Aplica todas as regras e soma os pontos, limitando a `max_score`.

    `hits` vem ordenado do maior para o menor número de pontos.
    """
    config = config or BehaviorConfig()
    hits = [hit for rule in RULES if (hit := rule(features, config)) is not None]
    hits.sort(key=lambda h: h.points, reverse=True)
    total = sum(h.points for h in hits)
    return BehaviorResult(score=min(config.max_score, total), hits=hits)
