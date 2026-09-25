"""Features comportamentais do domínio sintético — funções PURAS.

Cada função recebe dados (uma transação, um cliente, um conjunto de
dispositivos já vistos, a transação anterior) e devolve um valor. Nenhuma
lê banco, nenhuma decide "isso é fraude ou não" — decisão de risco é
responsabilidade do Risk Engine (etapa futura), que vai consumir estas
features tanto em regras quanto em modelos.

Duck typing de propósito: as funções só usam atributos (`transaction.amount`,
`customer.avg_amount_baseline`, ...), então funcionam igual com objetos ORM
(`src/synthetic/models.py`) e com objetos simples nos testes, sem precisar
de banco para testar cada feature isoladamente.

Ver `src/synthetic/feature_pipeline.py` para a montagem da tabela completa
a partir do banco.
"""

from __future__ import annotations

import math
from typing import Any, Collection, NamedTuple

from src.synthetic.generator import MAX_PLAUSIBLE_SPEED_KMH
from src.synthetic.geo_reference import distance_km

# Meia circunferência da Terra (~20.015 km): maior distância possível entre
# dois pontos. Referência FIXA (não min-max do dataset), para que
# `location_distance` continue sendo uma função pura de uma única transação.
MAX_REFERENCE_DISTANCE_KM = 20_000.0

# Os timestamps do domínio têm resolução de minuto — duas transações no
# mesmo minuto têm delta 0 (ou quase). Sem um piso, dividir a distância por
# ~0 daria velocidade infinita mesmo para dois estabelecimentos a poucos km.
# Com o piso de 1 minuto, "mesmo minuto, mesma cidade" fica em centenas de
# km/h (plausível), e "mesmo minuto, cidades a 2.000 km" continua absurdo.
MIN_TIME_DELTA_HOURS = 1.0 / 60.0


class TravelSpeed(NamedTuple):
    """Resultado de `implied_travel_speed`.

    `speed_kmh` é None quando não há transação anterior (primeira do
    cliente) — o "não sei" fica explícito, em vez de virar 0 ou NaN aqui.
    """

    speed_kmh: float | None
    is_impossible_travel: bool


def amount_deviation(transaction: Any, customer: Any) -> float:
    """Diferença RELATIVA (com sinal) do valor da transação para a média
    histórica do próprio cliente: `(valor - média) / média`.

    0.0 = exatamente na média; +1.0 = o dobro da média; -0.5 = metade.
    R$ 3.000 dá ~0 para um `alto_gasto` com média de R$ 3.000, e ~+30 para
    um `fixo` com média de R$ 100 — o mesmo valor, sinais opostos.

    Diferença relativa (e não z-score) porque `Customer` guarda só a média
    (`avg_amount_baseline`), não o desvio-padrão — e inventar um desvio a
    partir das constantes do gerador vazaria o mecanismo de geração para
    dentro da feature.
    """
    baseline = customer.avg_amount_baseline
    if baseline <= 0:
        raise ValueError(f"avg_amount_baseline precisa ser > 0, recebido {baseline!r}")
    return (transaction.amount - baseline) / baseline


def location_distance(transaction: Any) -> float:
    """`distance_from_home_km` normalizado para [0, 1] em escala logarítmica:
    `log(1 + km) / log(1 + 20.000)`.

    Log porque quase toda distância é pequena (compras na própria cidade,
    0-10 km) e poucas são enormes (milhares de km) — em escala linear, a
    diferença entre 2 km e 8 km sumiria perto da diferença entre 0 e
    2.000 km. Aqui 5 km ≈ 0.18 e 2.000 km ≈ 0.77.
    """
    km = max(0.0, transaction.distance_from_home_km)
    return min(1.0, math.log1p(km) / math.log1p(MAX_REFERENCE_DISTANCE_KM))


def _circular_hour_gap(a: float, b: float) -> float:
    gap = abs(a - b)
    return min(gap, 24.0 - gap)


def unusual_hour(transaction: Any, customer: Any) -> float:
    """Quão fora da janela habitual do cliente está o horário, em [0, 1].

    0.0 = dentro de `[typical_hour_start, typical_hour_end]`. Fora dela,
    é a distância circular (em horas) até a borda mais próxima da janela,
    dividida por 12 (a maior distância possível num relógio de 24h) — então
    1.0 = exatamente do lado oposto do relógio. Circular porque 23h e 1h
    estão a 2 horas uma da outra, não a 22.
    """
    ts = transaction.timestamp
    hour = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    start, end = customer.typical_hour_start, customer.typical_hour_end

    inside = (start <= hour <= end) if start <= end else (hour >= start or hour <= end)
    if inside:
        return 0.0

    gap = min(_circular_hour_gap(hour, start), _circular_hour_gap(hour, end))
    return gap / 12.0


def new_device_flag(transaction: Any, seen_device_ids: Collection[int]) -> bool:
    """True se o dispositivo da transação NUNCA apareceu antes, no histórico
    daquele cliente.

    `seen_device_ids` = dispositivos dos quais o cliente já tem transação
    ANTERIOR a esta (quem monta a tabela é que mantém esse conjunto,
    cliente a cliente, em ordem cronológica — ver feature_pipeline). Sem
    histórico nenhum (conjunto vazio), qualquer dispositivo é "novo":
    é literalmente a primeira vez que aparece.
    """
    return transaction.device_id not in seen_device_ids


def implied_travel_speed(transaction: Any, previous_transaction: Any | None) -> TravelSpeed:
    """Velocidade implícita (km/h) entre duas transações CONSECUTIVAS do
    mesmo cliente, e se ela é fisicamente implausível.

    `is_impossible_travel` usa o mesmo limiar do gerador
    (`MAX_PLAUSIBLE_SPEED_KMH`, 900 km/h — velocidade de cruzeiro de avião
    comercial), então feature e cenário de fraude "viagem_impossivel" são
    medidos com a mesma régua.

    Sem transação anterior -> `TravelSpeed(None, False)`, de forma
    explícita. Se `previous_transaction` for posterior à atual (fora de
    ordem), levanta ValueError: a função assume ordem cronológica e não
    vai calcular em silêncio uma velocidade com sinal trocado.
    """
    if previous_transaction is None:
        return TravelSpeed(speed_kmh=None, is_impossible_travel=False)

    delta_hours = (transaction.timestamp - previous_transaction.timestamp).total_seconds() / 3600.0
    if delta_hours < 0:
        raise ValueError("previous_transaction é posterior à transação atual (fora de ordem)")

    km = distance_km(
        previous_transaction.lat, previous_transaction.lon, transaction.lat, transaction.lon
    )
    speed = km / max(delta_hours, MIN_TIME_DELTA_HOURS)
    return TravelSpeed(speed_kmh=speed, is_impossible_travel=speed > MAX_PLAUSIBLE_SPEED_KMH)
