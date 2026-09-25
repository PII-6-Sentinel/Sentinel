"""Modelos SQLAlchemy do domínio sintético — a "via operacional" do Sentinel.

Enquanto o dataset do Kaggle (Credit Card Fraud Detection) serve para
validação científica dos modelos sobre um benchmark público e rotulado,
este domínio simula um ambiente antifraude operacional: clientes, cartões,
dispositivos e estabelecimentos comerciais, com transações geradas de
forma determinística (ver `src/synthetic/generator.py`) e persistidas em
SQLite via este ORM — nunca CSV solto, para já nascer com a forma de um
sistema de verdade (consultável, com integridade referencial).

Ver `docs/DOMINIO_SINTETICO.md` para a justificativa metodológica completa
(por que dados sintéticos, como cada cenário de fraude foi desenhado).
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class BehaviorProfile(str, enum.Enum):
    """Perfil comportamental do cliente — molda como sua atividade "normal"
    é gerada (cidade, valor, horário) e serve de contexto para julgar o
    que é uma anomalia PARA AQUELE cliente especificamente.
    """

    FIXO = "fixo"
    VIAJANTE = "viajante"
    ALTO_GASTO = "alto_gasto"
    NOVO_CLIENTE = "novo_cliente"


class CardType(str, enum.Enum):
    DEBITO = "debito"
    CREDITO = "credito"


class CardStatus(str, enum.Enum):
    ATIVO = "ativo"
    BLOQUEADO = "bloqueado"


class DeviceType(str, enum.Enum):
    MOBILE = "mobile"
    DESKTOP = "desktop"
    POS = "pos"


class FraudScenario(str, enum.Enum):
    """Os 6 cenários de fraude injetados pelo gerador — ver
    docs/DOMINIO_SINTETICO.md para a descrição de cada um.
    """

    ANOMALIA_GEOGRAFICA = "anomalia_geografica"
    VIAGEM_IMPOSSIVEL = "viagem_impossivel"
    VALOR_ATIPICO = "valor_atipico"
    HORARIO_ATIPICO = "horario_atipico"
    COMBINACAO_DE_SINAIS = "combinacao_de_sinais"
    DISPOSITIVO_NOVO = "dispositivo_novo"


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    home_city: Mapped[str] = mapped_column(String(80))
    home_state: Mapped[str | None] = mapped_column(String(40), nullable=True)
    home_country: Mapped[str] = mapped_column(String(60))
    home_lat: Mapped[float] = mapped_column(Float)
    home_lon: Mapped[float] = mapped_column(Float)
    behavior_profile: Mapped[BehaviorProfile] = mapped_column(SAEnum(BehaviorProfile))
    avg_amount_baseline: Mapped[float] = mapped_column(Float)
    typical_hour_start: Mapped[int] = mapped_column(Integer)
    typical_hour_end: Mapped[int] = mapped_column(Integer)

    cards: Mapped[list["Card"]] = relationship(back_populates="customer")
    devices: Mapped[list["Device"]] = relationship(back_populates="customer")
    transactions: Mapped[list["Transaction"]] = relationship(back_populates="customer")


class Card(Base):
    __tablename__ = "cards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    card_type: Mapped[CardType] = mapped_column(SAEnum(CardType))
    status: Mapped[CardStatus] = mapped_column(SAEnum(CardStatus), default=CardStatus.ATIVO)
    issued_at: Mapped[datetime] = mapped_column(DateTime)

    customer: Mapped["Customer"] = relationship(back_populates="cards")


class Device(Base):
    __tablename__ = "devices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    device_type: Mapped[DeviceType] = mapped_column(SAEnum(DeviceType))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime)
    is_trusted: Mapped[bool] = mapped_column(Boolean, default=True)

    customer: Mapped["Customer"] = relationship(back_populates="devices")


class Merchant(Base):
    __tablename__ = "merchants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    category: Mapped[str] = mapped_column(String(60))
    city: Mapped[str] = mapped_column(String(80))
    state: Mapped[str | None] = mapped_column(String(40), nullable=True)
    country: Mapped[str] = mapped_column(String(60))
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    card_id: Mapped[int] = mapped_column(ForeignKey("cards.id"))
    merchant_id: Mapped[int] = mapped_column(ForeignKey("merchants.id"))
    device_id: Mapped[int] = mapped_column(ForeignKey("devices.id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    amount: Mapped[float] = mapped_column(Float)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    distance_from_home_km: Mapped[float] = mapped_column(Float)
    is_fraud: Mapped[bool] = mapped_column(Boolean, default=False)
    # String (não Enum) de propósito: nullable, guarda o .value de
    # FraudScenario para transações fraudulentas, e None para normais.
    fraud_scenario: Mapped[str | None] = mapped_column(String(40), nullable=True)

    customer: Mapped["Customer"] = relationship(back_populates="transactions")
    card: Mapped["Card"] = relationship()
    merchant: Mapped["Merchant"] = relationship()
    device: Mapped["Device"] = relationship()
