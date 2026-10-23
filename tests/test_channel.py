"""Canal (online / presencial) e forma de pagamento (aproximação / senha):
nos dados, nas features, nas regras e no ML (Etapa 12).
"""

from collections import Counter
from types import SimpleNamespace

import pytest

from src.risk.behavior_score import BehaviorConfig, behavior_score, rule_channel, rule_contactless
from src.risk.ml_score import ML_FEATURES
from src.synthetic.features import channel_rarity, is_contactless, is_online
from src.synthetic.generator import CONTACTLESS_LIMIT_BRL, generate_dataset
from src.synthetic.models import Channel, FraudScenario, PaymentMethod


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(n_customers=60)


# ----------------------------------------------------------------------
# dados
# ----------------------------------------------------------------------


def test_every_transaction_has_a_valid_channel_and_payment(dataset):
    for tx in dataset.transactions:
        assert tx.channel in {c.value for c in Channel}
        if tx.channel == Channel.ONLINE.value:
            assert tx.payment_method is None
        else:
            assert tx.payment_method in {p.value for p in PaymentMethod}


def test_contactless_never_above_the_limit(dataset):
    for tx in dataset.transactions:
        if tx.payment_method == PaymentMethod.APROXIMACAO.value:
            assert tx.amount <= CONTACTLESS_LIMIT_BRL


def test_new_device_fraud_is_always_online(dataset):
    frauds = [t for t in dataset.transactions if t.fraud_scenario == FraudScenario.DISPOSITIVO_NOVO.value]
    assert frauds
    assert all(t.channel == Channel.ONLINE.value for t in frauds)


def test_both_channels_and_methods_appear_in_normal_and_fraud(dataset):
    normal = Counter((t.channel, t.payment_method) for t in dataset.transactions if not t.is_fraud)
    assert len(normal) == 3  # online, presencial+senha, presencial+aproximação
    fraud = Counter(t.channel for t in dataset.transactions if t.is_fraud)
    assert set(fraud) == {Channel.ONLINE.value, Channel.PRESENCIAL.value}


def test_frauds_are_more_often_online_than_normal(dataset):
    def online_rate(txs):
        return sum(t.channel == Channel.ONLINE.value for t in txs) / len(txs)

    frauds = [t for t in dataset.transactions if t.is_fraud]
    normal = [t for t in dataset.transactions if not t.is_fraud]
    assert online_rate(frauds) > online_rate(normal)


def test_channels_keep_earlier_stages_unchanged():
    """O canal é sorteado por outro gerador: valores, locais e horários das
    etapas anteriores são os mesmos para a mesma semente.
    """
    a, b = generate_dataset(n_customers=20), generate_dataset(n_customers=20)
    assert [(t.amount, t.timestamp, t.lat, t.channel, t.payment_method) for t in a.transactions] == [
        (t.amount, t.timestamp, t.lat, t.channel, t.payment_method) for t in b.transactions
    ]


def test_customer_channel_habits_are_set(dataset):
    for c in dataset.customers:
        assert 0.0 < c.online_share < 1.0
        assert 0.0 < c.contactless_share < 1.0


# ----------------------------------------------------------------------
# features
# ----------------------------------------------------------------------


def test_is_online_and_is_contactless():
    online = SimpleNamespace(channel="online", payment_method=None)
    tap = SimpleNamespace(channel="presencial", payment_method="aproximacao")
    pin = SimpleNamespace(channel="presencial", payment_method="senha")
    assert is_online(online) and not is_contactless(online)
    assert not is_online(tap) and is_contactless(tap)
    assert not is_online(pin) and not is_contactless(pin)


def test_channel_rarity():
    assert channel_rarity("online", Counter()) == 0.0  # sem histórico, sem hábito
    assert channel_rarity("online", Counter(presencial=9, online=1)) == pytest.approx(0.9)
    assert channel_rarity("presencial", Counter(presencial=9, online=1)) == pytest.approx(0.1)
    assert channel_rarity("online", Counter(presencial=5)) == 1.0


# ----------------------------------------------------------------------
# regras e ML
# ----------------------------------------------------------------------


def _features(**overrides):
    base = {
        "amount_deviation": 0.0,
        "distance_from_home_km": 4.0,
        "location_distance": 0.16,
        "unusual_hour": 0.0,
        "new_device": False,
        "new_city": False,
        "device_untrusted": False,
        "implied_speed_kmh": 3.0,
        "is_impossible_travel": False,
        "is_online": False,
        "is_contactless": False,
        "channel_rarity": 0.0,
    }
    base.update(overrides)
    return base


def test_rare_online_purchase_fires_the_channel_rule():
    hit = rule_channel(_features(is_online=True, channel_rarity=0.95), BehaviorConfig())
    assert hit is not None and hit.rule == "canal_raro"
    assert "online" in hit.reason and "5%" in hit.reason


def test_habitual_channel_does_not_fire():
    assert rule_channel(_features(is_online=True, channel_rarity=0.4), BehaviorConfig()) is None


def test_contactless_only_counts_above_the_average():
    c = BehaviorConfig()
    assert rule_contactless(_features(is_contactless=True, amount_deviation=0.0), c) is None
    hit = rule_contactless(_features(is_contactless=True, amount_deviation=1.0), c)
    assert hit is not None and "aproximação" in hit.reason


def test_normal_presencial_purchase_still_scores_zero():
    assert behavior_score(_features()).score == 0


def test_ml_reads_the_channel_features():
    assert {"is_online", "is_contactless", "channel_rarity"} <= set(ML_FEATURES)
