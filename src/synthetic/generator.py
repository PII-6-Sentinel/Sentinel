"""Gerador determinístico do domínio sintético antifraude.

Nada aqui depende do Streamlit nem é acoplado a `scripts/generate_mock_data.py`
— fica em `src/` (não só no script) de propósito: os testes
(`tests/test_synthetic_data.py`) chamam `generate_dataset()`/`populate_database()`
diretamente, com um número pequeno de clientes, sobre um SQLite em memória —
sem isso, testar propriedades do gerador exigiria rodar o script inteiro e
ler um arquivo de banco a cada teste, lento e difícil de isolar.

Determinismo: toda a aleatoriedade passa por uma única instância
`random.Random(seed)`, nunca pelo módulo `random` global — dado o mesmo
`seed` e os mesmos parâmetros, `generate_dataset()` produz sempre exatamente
o mesmo dataset, seja chamado uma vez ou cem.

Ver `docs/DOMINIO_SINTETICO.md` para a justificativa metodológica completa
de cada perfil comportamental e cenário de fraude.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from src.synthetic.geo_reference import CITIES, CityRef, distance_km
from src.synthetic.models import (
    Base,
    BehaviorProfile,
    Card,
    CardStatus,
    CardType,
    Customer,
    Device,
    DeviceType,
    FraudScenario,
    Merchant,
    Transaction,
)

DEFAULT_SEED = 42
DEFAULT_N_CUSTOMERS = 130

# Ponto fixo no tempo usado como "hoje" da simulação — nunca `datetime.now()`.
# Reprodutibilidade de verdade significa que rodar o gerador duas vezes (hoje
# ou daqui a um ano) produz o MESMO dataset, byte a byte; ancorar em "agora"
# quebraria isso a cada execução em um dia diferente.
REFERENCE_END_DATE = datetime(2026, 9, 1)
HISTORY_DAYS = 90

FRAUD_RATE_TARGET = 0.03

# Velocidade comercial de avião — usada tanto para GERAR o cenário de
# "viagem impossível" (a transação injetada sempre excede isso, por
# construção) quanto para VALIDAR o cenário nos testes.
MAX_PLAUSIBLE_SPEED_KMH = 900.0

PROFILE_WEIGHTS: dict[BehaviorProfile, float] = {
    BehaviorProfile.FIXO: 0.50,
    BehaviorProfile.VIAJANTE: 0.25,
    BehaviorProfile.ALTO_GASTO: 0.15,
    BehaviorProfile.NOVO_CLIENTE: 0.10,
}

CITY_BY_NAME: dict[str, CityRef] = {c.city: c for c in CITIES}

MERCHANT_CATEGORIES = [
    "Supermercado", "Restaurante", "Posto de Combustível", "Farmácia",
    "Vestuário", "Eletrônicos", "E-commerce", "Hotel", "Companhia Aérea",
    "Livraria", "Pet Shop", "Academia", "Cafeteria", "Cinema",
]

FIRST_NAMES = [
    "Ana", "Bruno", "Carla", "Daniel", "Eduarda", "Felipe", "Gabriela",
    "Hugo", "Isabela", "João", "Larissa", "Marcelo", "Natália", "Otávio",
    "Patrícia", "Rafael", "Sabrina", "Thiago", "Vanessa", "Wesley",
    "Camila", "Diego", "Fernanda", "Gustavo", "Juliana", "Lucas",
    "Mariana", "Pedro", "Renata", "Vitor",
]
LAST_NAMES = [
    "Silva", "Souza", "Oliveira", "Santos", "Pereira", "Costa",
    "Rodrigues", "Almeida", "Nascimento", "Lima", "Araújo", "Fernandes",
    "Carvalho", "Gomes", "Martins", "Rocha", "Ribeiro", "Alves",
    "Monteiro", "Cardoso",
]


@dataclass
class _CustomerBundle:
    """Agrupa, durante a geração, tudo que pertence a um cliente — evita
    passar 4-5 parâmetros soltos por função e mantém as transações (normais
    + fraude injetada) sempre acessíveis para checar conflito de horário.
    """

    customer: Customer
    cards: list[Card]
    devices: list[Device]
    transactions: list[Transaction] = field(default_factory=list)
    traveler_cities: list[CityRef] = field(default_factory=list)


@dataclass
class GeneratedDataset:
    customers: list[Customer]
    cards: list[Card]
    devices: list[Device]
    merchants: list[Merchant]
    transactions: list[Transaction]


# ---------------------------------------------------------------------------
# Clientes, cartões, dispositivos
# ---------------------------------------------------------------------------


def _build_profile_sequence(rng: random.Random, n_customers: int) -> list[BehaviorProfile]:
    """~50/25/15/10% entre os 4 perfis (arredondado; o resto vai para
    'novo_cliente' para garantir soma exata), depois embaralhado.
    """
    n_fixo = round(n_customers * PROFILE_WEIGHTS[BehaviorProfile.FIXO])
    n_viajante = round(n_customers * PROFILE_WEIGHTS[BehaviorProfile.VIAJANTE])
    n_alto = round(n_customers * PROFILE_WEIGHTS[BehaviorProfile.ALTO_GASTO])
    n_novo = n_customers - n_fixo - n_viajante - n_alto

    sequence = (
        [BehaviorProfile.FIXO] * n_fixo
        + [BehaviorProfile.VIAJANTE] * n_viajante
        + [BehaviorProfile.ALTO_GASTO] * n_alto
        + [BehaviorProfile.NOVO_CLIENTE] * n_novo
    )
    rng.shuffle(sequence)
    return sequence


def _build_customer(rng: random.Random, profile: BehaviorProfile) -> Customer:
    home = rng.choice(CITIES)
    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"

    if profile == BehaviorProfile.ALTO_GASTO:
        baseline = rng.uniform(500.0, 3000.0)
    elif profile == BehaviorProfile.VIAJANTE:
        baseline = rng.uniform(80.0, 400.0)
    elif profile == BehaviorProfile.NOVO_CLIENTE:
        baseline = rng.uniform(40.0, 200.0)
    else:  # FIXO
        baseline = rng.uniform(50.0, 300.0)

    start_hour = rng.randint(6, 11)
    end_hour = min(23, start_hour + rng.randint(8, 13))

    return Customer(
        name=name,
        home_city=home.city,
        home_state=home.state,
        home_country=home.country,
        home_lat=home.lat,
        home_lon=home.lon,
        behavior_profile=profile,
        avg_amount_baseline=round(baseline, 2),
        typical_hour_start=start_hour,
        typical_hour_end=end_hour,
    )


def _build_cards_and_devices(
    rng: random.Random, customer: Customer, reference_end: datetime
) -> tuple[list[Card], list[Device]]:
    n_cards = rng.choices([1, 2], weights=[0.8, 0.2])[0]
    cards = []
    for _ in range(n_cards):
        card_type = rng.choices([CardType.CREDITO, CardType.DEBITO], weights=[0.6, 0.4])[0]
        issued_at = reference_end - timedelta(days=rng.randint(100, 1500))
        cards.append(
            Card(customer=customer, card_type=card_type, status=CardStatus.ATIVO, issued_at=issued_at)
        )

    n_devices = rng.choices([1, 2], weights=[0.7, 0.3])[0]
    devices = []
    for i in range(n_devices):
        device_type = rng.choices(
            [DeviceType.MOBILE, DeviceType.DESKTOP, DeviceType.POS], weights=[0.7, 0.2, 0.1]
        )[0]
        first_seen_at = reference_end - timedelta(days=rng.randint(30, 1500))
        is_trusted = True if i == 0 else rng.random() > 0.3
        devices.append(
            Device(
                customer=customer,
                device_type=device_type,
                first_seen_at=first_seen_at,
                is_trusted=is_trusted,
            )
        )
    return cards, devices


# ---------------------------------------------------------------------------
# Estabelecimentos
# ---------------------------------------------------------------------------


def _build_merchants(rng: random.Random) -> list[Merchant]:
    merchants = []
    for city in CITIES:
        n = rng.randint(3, 6)
        for i in range(n):
            category = rng.choice(MERCHANT_CATEGORIES)
            jitter_lat = rng.uniform(-0.05, 0.05)  # ~5km, espalha os estabelecimentos dentro da cidade
            jitter_lon = rng.uniform(-0.05, 0.05)
            merchants.append(
                Merchant(
                    name=f"{category} {city.city} #{i + 1}",
                    category=category,
                    city=city.city,
                    state=city.state,
                    country=city.country,
                    lat=round(city.lat + jitter_lat, 6),
                    lon=round(city.lon + jitter_lon, 6),
                )
            )
    return merchants


# ---------------------------------------------------------------------------
# Transações "normais"
# ---------------------------------------------------------------------------


def _random_timestamp(rng: random.Random, customer: Customer, reference_end: datetime) -> datetime:
    days_ago = rng.uniform(0.0, HISTORY_DAYS)
    base_day = (reference_end - timedelta(days=days_ago)).date()
    hour_float = rng.uniform(customer.typical_hour_start, customer.typical_hour_end)
    hour = int(hour_float)
    minute = int((hour_float - hour) * 60)
    return datetime(base_day.year, base_day.month, base_day.day, hour, minute)


def _random_amount(rng: random.Random, customer: Customer) -> float:
    return max(5.0, rng.gauss(customer.avg_amount_baseline, customer.avg_amount_baseline * 0.35))


def _pick_normal_city(rng: random.Random, bundle: _CustomerBundle) -> CityRef:
    customer = bundle.customer
    if (
        customer.behavior_profile == BehaviorProfile.VIAJANTE
        and bundle.traveler_cities
        and rng.random() < 0.35
    ):
        return rng.choice(bundle.traveler_cities)
    return CITY_BY_NAME[customer.home_city]


def _make_transaction(
    customer: Customer,
    card: Card,
    device: Device,
    merchant: Merchant,
    timestamp: datetime,
    amount: float,
    is_fraud: bool = False,
    fraud_scenario: str | None = None,
) -> Transaction:
    dist = distance_km(customer.home_lat, customer.home_lon, merchant.lat, merchant.lon)
    return Transaction(
        customer=customer,
        card=card,
        merchant=merchant,
        device=device,
        timestamp=timestamp,
        amount=round(amount, 2),
        lat=merchant.lat,
        lon=merchant.lon,
        distance_from_home_km=round(dist, 2),
        is_fraud=is_fraud,
        fraud_scenario=fraud_scenario,
    )


def _build_customer_bundles(
    rng: random.Random,
    n_customers: int,
    merchants_by_city: dict[str, list[Merchant]],
    reference_end: datetime,
) -> list[_CustomerBundle]:
    profiles = _build_profile_sequence(rng, n_customers)
    bundles = []

    for profile in profiles:
        customer = _build_customer(rng, profile)

        traveler_cities: list[CityRef] = []
        if profile == BehaviorProfile.VIAJANTE:
            others = [c for c in CITIES if c.city != customer.home_city]
            traveler_cities = rng.sample(others, k=min(len(others), rng.randint(2, 3)))

        cards, devices = _build_cards_and_devices(rng, customer, reference_end)
        bundle = _CustomerBundle(
            customer=customer, cards=cards, devices=devices, traveler_cities=traveler_cities
        )

        # Cliente novo tem pouco histórico por definição; os demais perfis
        # acumulam um histórico "normal" de tamanho comparável.
        n_normal = (
            rng.randint(8, 15) if profile == BehaviorProfile.NOVO_CLIENTE else rng.randint(30, 80)
        )
        for _ in range(n_normal):
            city_ref = _pick_normal_city(rng, bundle)
            merchant = rng.choice(merchants_by_city[city_ref.city])
            card = rng.choice(cards)
            device = rng.choice(devices)
            timestamp = _random_timestamp(rng, customer, reference_end)
            amount = _random_amount(rng, customer)
            bundle.transactions.append(
                _make_transaction(customer, card, device, merchant, timestamp, amount)
            )

        bundle.transactions.sort(key=lambda t: t.timestamp)
        bundles.append(bundle)

    return bundles


# ---------------------------------------------------------------------------
# Cenários de fraude (ver docs/DOMINIO_SINTETICO.md para a descrição de cada um)
# ---------------------------------------------------------------------------


def inject_geo_anomaly(
    rng: random.Random,
    bundle: _CustomerBundle,
    merchants_by_city: dict[str, list[Merchant]],
    reference_end: datetime,
) -> Transaction | None:
    """Transação numa cidade que o cliente NUNCA frequenta — nem a cidade
    natal, nem (para viajantes) o conjunto de cidades que costuma visitar.
    """
    customer = bundle.customer
    normal_cities = {customer.home_city} | {c.city for c in bundle.traveler_cities}
    candidates = [c for c in CITIES if c.city not in normal_cities]
    if not candidates:
        return None

    city_ref = rng.choice(candidates)
    merchant = rng.choice(merchants_by_city[city_ref.city])
    card = rng.choice(bundle.cards)
    device = rng.choice(bundle.devices)
    timestamp = _random_timestamp(rng, customer, reference_end)
    amount = _random_amount(rng, customer)
    return _make_transaction(
        customer, card, device, merchant, timestamp, amount,
        is_fraud=True, fraud_scenario=FraudScenario.ANOMALIA_GEOGRAFICA.value,
    )


def inject_impossible_travel(
    rng: random.Random,
    bundle: _CustomerBundle,
    merchants_by_city: dict[str, list[Merchant]],
    reference_end: datetime,
    max_attempts: int = 20,
) -> Transaction | None:
    """Duas transações consecutivas do mesmo cliente, longe o bastante e
    perto o bastante no tempo para implicar uma velocidade > 900km/h.

    A distância e o delta de tempo são escolhidos JUNTOS (delta_hours =
    distância / velocidade_alvo, com velocidade_alvo sempre > 900) — não é
    uma tentativa de "sorte" até bater o limiar, o limiar é garantido por
    construção. O que pode falhar e exigir nova tentativa é achar uma
    cidade longe o bastante (>= 300km, pra não ser uma diferença trivial de
    poucos km dentro da mesma cidade) cujo horário resultante não colida
    com outra transação já existente desse cliente.
    """
    customer = bundle.customer
    if not bundle.transactions:
        return None

    for _ in range(max_attempts):
        prev_tx = rng.choice(bundle.transactions)
        candidate = rng.choice(CITIES)
        dist = distance_km(prev_tx.lat, prev_tx.lon, candidate.lat, candidate.lon)
        if dist < 300.0:
            continue

        target_speed = rng.uniform(1200.0, 2500.0)  # sempre > MAX_PLAUSIBLE_SPEED_KMH, com folga
        delta_hours = dist / target_speed
        new_ts = prev_tx.timestamp + timedelta(hours=delta_hours)
        if new_ts > reference_end:
            continue

        conflict = any(prev_tx.timestamp < t.timestamp < new_ts for t in bundle.transactions)
        if conflict:
            continue

        merchant = rng.choice(merchants_by_city[candidate.city])
        card = rng.choice(bundle.cards)
        device = rng.choice(bundle.devices)
        amount = _random_amount(rng, customer)
        return _make_transaction(
            customer, card, device, merchant, new_ts, amount,
            is_fraud=True, fraud_scenario=FraudScenario.VIAGEM_IMPOSSIVEL.value,
        )

    return None


def inject_atypical_amount(
    rng: random.Random,
    bundle: _CustomerBundle,
    merchants_by_city: dict[str, list[Merchant]],
    reference_end: datetime,
) -> Transaction | None:
    """Valor muito acima da média do PRÓPRIO cliente — cidade e horário
    normais, só o valor destoa (relativo à baseline de cada cliente, não a
    um limiar fixo global).
    """
    customer = bundle.customer
    city_ref = CITY_BY_NAME[customer.home_city]
    merchant = rng.choice(merchants_by_city[city_ref.city])
    card = rng.choice(bundle.cards)
    device = rng.choice(bundle.devices)
    timestamp = _random_timestamp(rng, customer, reference_end)
    amount = customer.avg_amount_baseline * rng.uniform(6.0, 15.0)
    return _make_transaction(
        customer, card, device, merchant, timestamp, amount,
        is_fraud=True, fraud_scenario=FraudScenario.VALOR_ATIPICO.value,
    )


def inject_atypical_hour(
    rng: random.Random,
    bundle: _CustomerBundle,
    merchants_by_city: dict[str, list[Merchant]],
    reference_end: datetime,
) -> Transaction | None:
    """Horário claramente fora da janela típica DAQUELE cliente — cidade e
    valor normais.
    """
    customer = bundle.customer
    city_ref = CITY_BY_NAME[customer.home_city]
    merchant = rng.choice(merchants_by_city[city_ref.city])
    card = rng.choice(bundle.cards)
    device = rng.choice(bundle.devices)

    excluded = set(range(max(0, customer.typical_hour_start - 1), min(24, customer.typical_hour_end + 2)))
    candidate_hours = [h for h in range(24) if h not in excluded]
    hour = rng.choice(candidate_hours) if candidate_hours else (customer.typical_hour_start + 12) % 24
    minute = rng.randint(0, 59)
    days_ago = rng.uniform(0.0, HISTORY_DAYS)
    base_day = (reference_end - timedelta(days=days_ago)).date()
    timestamp = datetime(base_day.year, base_day.month, base_day.day, hour, minute)

    amount = _random_amount(rng, customer)
    return _make_transaction(
        customer, card, device, merchant, timestamp, amount,
        is_fraud=True, fraud_scenario=FraudScenario.HORARIO_ATIPICO.value,
    )


def inject_combined_signals(
    rng: random.Random,
    bundle: _CustomerBundle,
    merchants_by_city: dict[str, list[Merchant]],
    reference_end: datetime,
) -> Transaction | None:
    """Combina 3 sinais MODERADOS ao mesmo tempo (cidade incomum, valor
    algumas vezes acima da média, horário um pouco fora da janela) — nenhum
    extremo o bastante sozinho para disparar os outros 4 cenários, mas a
    combinação é suspeita. É o cenário mais difícil de pegar com uma regra
    única — motivação direta para comparar regras estatísticas com ML nas
    próximas etapas do projeto.
    """
    customer = bundle.customer
    candidates = [c for c in CITIES if c.city != customer.home_city]
    city_ref = rng.choice(candidates)
    merchant = rng.choice(merchants_by_city[city_ref.city])
    card = rng.choice(bundle.cards)
    device = rng.choice(bundle.devices)

    amount = customer.avg_amount_baseline * rng.uniform(2.5, 4.5)
    hour = (customer.typical_hour_end + rng.randint(2, 4)) % 24
    minute = rng.randint(0, 59)
    days_ago = rng.uniform(0.0, HISTORY_DAYS)
    base_day = (reference_end - timedelta(days=days_ago)).date()
    timestamp = datetime(base_day.year, base_day.month, base_day.day, hour, minute)

    return _make_transaction(
        customer, card, device, merchant, timestamp, amount,
        is_fraud=True, fraud_scenario=FraudScenario.COMBINACAO_DE_SINAIS.value,
    )


_INJECTORS = {
    FraudScenario.ANOMALIA_GEOGRAFICA: inject_geo_anomaly,
    FraudScenario.VIAGEM_IMPOSSIVEL: inject_impossible_travel,
    FraudScenario.VALOR_ATIPICO: inject_atypical_amount,
    FraudScenario.HORARIO_ATIPICO: inject_atypical_hour,
    FraudScenario.COMBINACAO_DE_SINAIS: inject_combined_signals,
}


def _target_fraud_counts(total_normal: int) -> dict[FraudScenario, int]:
    """Divide a meta de ~3% de fraude o mais igualmente possível entre os
    5 cenários (resto distribuído nos primeiros cenários da lista).
    """
    target_total = round(total_normal * FRAUD_RATE_TARGET / (1 - FRAUD_RATE_TARGET))
    scenarios = list(FraudScenario)
    base = target_total // len(scenarios)
    remainder = target_total - base * len(scenarios)
    counts = {s: base for s in scenarios}
    for s in scenarios[:remainder]:
        counts[s] += 1
    return counts


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------


def generate_dataset(
    seed: int = DEFAULT_SEED,
    n_customers: int = DEFAULT_N_CUSTOMERS,
    reference_end: datetime = REFERENCE_END_DATE,
) -> GeneratedDataset:
    """Gera o dataset completo em memória (nenhum I/O aqui — persistir é
    responsabilidade de `populate_database`). Determinístico: mesmo
    `seed`/`n_customers`/`reference_end` sempre produz o mesmo resultado.
    """
    rng = random.Random(seed)

    merchants = _build_merchants(rng)
    merchants_by_city: dict[str, list[Merchant]] = defaultdict(list)
    for m in merchants:
        merchants_by_city[m.city].append(m)

    bundles = _build_customer_bundles(rng, n_customers, merchants_by_city, reference_end)

    total_normal = sum(len(b.transactions) for b in bundles)
    scenario_counts = _target_fraud_counts(total_normal)

    for scenario, target_count in scenario_counts.items():
        inject_fn = _INJECTORS[scenario]
        placed = 0
        attempts = 0
        max_attempts = target_count * 20 + 30
        while placed < target_count and attempts < max_attempts:
            attempts += 1
            bundle = rng.choice(bundles)
            tx = inject_fn(rng, bundle, merchants_by_city, reference_end)
            if tx is None:
                continue
            bundle.transactions.append(tx)
            placed += 1

    for bundle in bundles:
        bundle.transactions.sort(key=lambda t: t.timestamp)

    return GeneratedDataset(
        customers=[b.customer for b in bundles],
        cards=[c for b in bundles for c in b.cards],
        devices=[d for b in bundles for d in b.devices],
        merchants=merchants,
        transactions=[t for b in bundles for t in b.transactions],
    )


def summarize(dataset: GeneratedDataset) -> dict:
    """Resumo pronto para exibição — usado tanto pelo script quanto,
    potencialmente, por testes.
    """
    profile_counts = Counter(c.behavior_profile.value for c in dataset.customers)
    fraud_txs = [t for t in dataset.transactions if t.is_fraud]
    total = len(dataset.transactions)
    fraud_by_scenario = Counter(t.fraud_scenario for t in fraud_txs)

    examples = []
    for scenario in FraudScenario:
        match = next((t for t in fraud_txs if t.fraud_scenario == scenario.value), None)
        if match is None:
            continue
        examples.append(
            {
                "scenario": match.fraud_scenario,
                "customer": match.customer.name,
                "profile": match.customer.behavior_profile.value,
                "amount": match.amount,
                "avg_amount_baseline": match.customer.avg_amount_baseline,
                "home_city": match.customer.home_city,
                "merchant_city": match.merchant.city,
                "distance_from_home_km": match.distance_from_home_km,
                "timestamp": match.timestamp,
            }
        )

    return {
        "n_customers": len(dataset.customers),
        "profile_counts": dict(profile_counts),
        "n_merchants": len(dataset.merchants),
        "n_transactions": total,
        "n_fraud": len(fraud_txs),
        "fraud_rate": (len(fraud_txs) / total) if total else 0.0,
        "fraud_by_scenario": dict(fraud_by_scenario),
        "example_frauds": examples,
    }


def populate_database(
    engine: Engine,
    seed: int = DEFAULT_SEED,
    n_customers: int = DEFAULT_N_CUSTOMERS,
    reference_end: datetime = REFERENCE_END_DATE,
) -> dict:
    """Gera o dataset e persiste no `engine` fornecido (cria as tabelas se
    não existirem). Devolve o mesmo resumo de `summarize()`.

    `expire_on_commit=False`: sem isso, os objetos ORM ficariam "expirados"
    após o commit, e montar o resumo (que lê atributos e relacionamentos,
    como `match.customer.name`) faria uma consulta nova ao banco para cada
    atributo — desnecessário, já que os objetos em memória continuam
    corretos logo após o commit.
    """
    dataset = generate_dataset(seed=seed, n_customers=n_customers, reference_end=reference_end)

    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as session:
        session.add_all(dataset.customers)
        session.add_all(dataset.merchants)
        session.add_all(dataset.cards)
        session.add_all(dataset.devices)
        session.add_all(dataset.transactions)
        session.commit()
        summary = summarize(dataset)

    return summary
