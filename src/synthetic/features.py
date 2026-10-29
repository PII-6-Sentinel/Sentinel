"""Features comportamentais do domínio sintético — funções PURAS.

Cada função recebe dados (uma transação, um cliente, um conjunto de
dispositivos já vistos, a transação anterior) e devolve um valor. Nenhuma
lê banco, nenhuma decide "isso é fraude ou não" — decisão de risco é
responsabilidade do Risk Engine (etapa futura), que vai consumir estas
features tanto em regras quanto em modelos.

Duck typing de propósito: as funções só usam atributos (`transaction.amount`,
`profile.avg_amount_baseline`, ...), então funcionam igual com objetos ORM
(`src/synthetic/models.py`) e com objetos simples nos testes, sem precisar
de banco para testar cada feature isoladamente.

Hábitos vêm do HISTÓRICO, não do gerador
----------------------------------------
`amount_deviation` e `unusual_hour` comparam a transação com os hábitos do
cliente (valor médio, janela de horário). O `Customer` do banco guarda os
parâmetros que o GERADOR usou para sortear as compras — um sistema real não
teria esses números, só as compras passadas. Por isso quem monta as features
passa um `HabitProfile`, estimado só com as transações ANTERIORES do cliente,
e não o `Customer` (ver `habit_features`).

Ver `src/synthetic/feature_pipeline.py` para a montagem da tabela completa
a partir do banco.
"""

from __future__ import annotations

import math
from typing import Any, Collection, Mapping, NamedTuple, Sequence

from src.synthetic.generator import MAX_PLAUSIBLE_SPEED_KMH
from src.synthetic.geo_reference import distance_km
from src.synthetic.models import Channel, PaymentMethod

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


# Compras anteriores necessárias para estimar um hábito. Com menos, valor e
# horário não acendem nenhum sinal: sem passado suficiente não há hábito
# (mesma convenção de `new_device_flag` e `channel_rarity` sem histórico).
MIN_HISTORY_FOR_HABITS = 5

# Janela habitual = entre estes quantis dos horários anteriores. Quantis em
# vez de mínimo/máximo para que uma compra de madrugada esporádica (ou uma
# fraude já no histórico) não alargue a janela para sempre.
HABIT_HOUR_QUANTILES = (0.05, 0.95)


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Quantil com interpolação linear (o mesmo critério do `numpy.quantile`)."""
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


class HabitProfile:
    """Hábitos do cliente estimados SÓ com as compras já vistas.

    Quem monta as features chama `add` depois de calcular as features de
    cada transação, em ordem cronológica — o perfil nunca enxerga a própria
    transação nem o futuro. Expõe os mesmos atributos que `amount_deviation`
    e `unusual_hour` leem (`avg_amount_baseline`, `typical_hour_start`,
    `typical_hour_end`), agora como estimativas.

    A janela de horário é linear (início ≤ fim): estimada por quantis dos
    horários, não representa uma janela que cruza a meia-noite.
    """

    def __init__(self, transactions: Sequence[Any] = ()):
        self._amounts: list[float] = []
        self._hours: list[float] = []
        for transaction in transactions:
            self.add(transaction)

    def add(self, transaction: Any) -> None:
        ts = transaction.timestamp
        self._amounts.append(float(transaction.amount))
        self._hours.append(ts.hour + ts.minute / 60.0 + ts.second / 3600.0)

    @property
    def n_transactions(self) -> int:
        return len(self._amounts)

    @property
    def is_known(self) -> bool:
        """Há histórico suficiente para comparar uma compra com o hábito."""
        return self.n_transactions >= MIN_HISTORY_FOR_HABITS

    @property
    def avg_amount_baseline(self) -> float | None:
        if not self._amounts:
            return None
        return sum(self._amounts) / len(self._amounts)

    def _hour_quantile(self, q: float) -> float | None:
        return _quantile(sorted(self._hours), q) if self._hours else None

    @property
    def typical_hour_start(self) -> float | None:
        return self._hour_quantile(HABIT_HOUR_QUANTILES[0])

    @property
    def typical_hour_end(self) -> float | None:
        return self._hour_quantile(HABIT_HOUR_QUANTILES[1])


class TravelSpeed(NamedTuple):
    """Resultado de `implied_travel_speed`.

    `speed_kmh` é None quando não há transação anterior (primeira do
    cliente) — o "não sei" fica explícito, em vez de virar 0 ou NaN aqui.
    """

    speed_kmh: float | None
    is_impossible_travel: bool


def amount_deviation(transaction: Any, profile: Any) -> float:
    """Diferença RELATIVA (com sinal) do valor da transação para a média
    histórica do próprio cliente: `(valor - média) / média`. `profile` é
    um `HabitProfile` (ou qualquer objeto com `avg_amount_baseline`).

    0.0 = exatamente na média; +1.0 = o dobro da média; -0.5 = metade.
    R$ 3.000 dá ~0 para um `alto_gasto` com média de R$ 3.000, e ~+30 para
    um `fixo` com média de R$ 100 — o mesmo valor, sinais opostos.

    Diferença relativa (e não z-score) porque, com poucas compras no
    histórico, a média já é uma estimativa ruidosa; um desvio-padrão
    estimado do mesmo histórico seria ainda mais instável.
    """
    baseline = profile.avg_amount_baseline
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


def unusual_hour(transaction: Any, profile: Any) -> float:
    """Quão fora da janela habitual do cliente está o horário, em [0, 1].

    0.0 = dentro de `[typical_hour_start, typical_hour_end]`. Fora dela,
    é a distância circular (em horas) até a borda mais próxima da janela,
    dividida por 12 (a maior distância possível num relógio de 24h) — então
    1.0 = exatamente do lado oposto do relógio. Circular porque 23h e 1h
    estão a 2 horas uma da outra, não a 22.
    """
    ts = transaction.timestamp
    hour = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    start, end = profile.typical_hour_start, profile.typical_hour_end

    inside = (start <= hour <= end) if start <= end else (hour >= start or hour <= end)
    if inside:
        return 0.0

    gap = min(_circular_hour_gap(hour, start), _circular_hour_gap(hour, end))
    return gap / 12.0


def habit_features(transaction: Any, profile: HabitProfile) -> tuple[float, float]:
    """(`amount_deviation`, `unusual_hour`) da transação contra o hábito
    estimado. Sem histórico suficiente (`profile.is_known` falso), (0.0,
    0.0): não se sabe o que é habitual, então nada destoa.
    """
    if not profile.is_known:
        return 0.0, 0.0
    return amount_deviation(transaction, profile), unusual_hour(transaction, profile)


def new_device_flag(transaction: Any, seen_device_ids: Collection[int]) -> bool:
    """True se o dispositivo da transação NUNCA apareceu antes no histórico
    do cliente — MAS só quando já existe algum histórico.

    `seen_device_ids` = dispositivos dos quais o cliente já tem transação
    ANTERIOR a esta (quem monta a tabela mantém esse conjunto, cliente a
    cliente, em ordem cronológica — ver feature_pipeline).

    Sem histórico nenhum (conjunto vazio = primeira transação do cliente)
    devolve False: "nunca vi esse dispositivo porque nunca vi nada" não é o
    mesmo sinal que "dispositivo novo depois de um histórico estabelecido".
    Sem essa regra, a primeira transação de todo cliente seria marcada como
    dispositivo novo, um positivo sem nenhuma informação.
    """
    if not seen_device_ids:
        return False
    return transaction.device_id not in seen_device_ids


def new_city_flag(city: str, customer: Any, seen_cities: Collection[str]) -> bool:
    """True se `city` (cidade do estabelecimento da transação) nunca apareceu
    no histórico ANTERIOR do cliente e não é a cidade natal dele.

    Mesma lógica de `new_device_flag`: `seen_cities` = cidades das
    transações do cliente estritamente anteriores a esta (quem monta a
    tabela mantém esse conjunto em ordem cronológica — sem olhar o futuro).
    Sem histórico (primeira transação) devolve False. A cidade natal é
    sempre conhecida, mesmo que o cliente ainda não tenha comprado nela.
    """
    if not seen_cities:
        return False
    if city == customer.home_city:
        return False
    return city not in seen_cities


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


def is_online(transaction: Any) -> bool:
    """True se a compra foi feita online (cartão não presente)."""
    return transaction.channel == Channel.ONLINE.value


def is_contactless(transaction: Any) -> bool:
    """True se a compra presencial foi paga por aproximação (sem senha).
    Online é sempre False: não há forma de pagamento presencial.
    """
    return transaction.payment_method == PaymentMethod.APROXIMACAO.value


def channel_rarity(channel: str, seen_channels: Mapping[str, int]) -> float:
    """Quão raro é o canal da transação no histórico ANTERIOR do cliente,
    em [0, 1]: `1 - (compras anteriores neste canal / compras anteriores)`.

    0.0 = o cliente sempre comprou por este canal; 0.9 = só 10% das compras
    anteriores foram por ele; 1.0 = nunca. `seen_channels` conta as
    transações estritamente anteriores, por canal (quem monta a tabela
    mantém a contagem em ordem cronológica — sem olhar o futuro). Sem
    histórico devolve 0.0, como `new_device_flag`: sem passado não há hábito.
    """
    total = sum(seen_channels.values())
    if total == 0:
        return 0.0
    return 1.0 - seen_channels.get(channel, 0) / total
