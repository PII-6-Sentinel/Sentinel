"""Referência geográfica estática usada pelo domínio sintético.

Coordenadas de cidades REAIS (latitude/longitude verdadeiras), mas usadas
aqui só como pontos de referência para gerar dados sintéticos — nenhuma
chamada de rede, nenhuma API de geocoding. Isso garante que o gerador
(`scripts/generate_mock_data.py`) seja 100% determinístico e reproduzível
offline, sem depender de serviço externo algum.

A lista cobre a maioria das capitais/grandes cidades do Brasil (de propósito
espalhadas por regiões diferentes, para que "distância entre cidades" tenha
variação real) mais algumas cidades internacionais, usadas nos cenários de
"cliente viajante" e de anomalia geográfica.
"""

from dataclasses import dataclass

from geopy.distance import geodesic


@dataclass(frozen=True)
class CityRef:
    city: str
    state: str | None
    country: str
    lat: float
    lon: float


CITIES: list[CityRef] = [
    # --- Brasil ---
    CityRef("São Paulo", "SP", "Brasil", -23.5505, -46.6333),
    CityRef("Rio de Janeiro", "RJ", "Brasil", -22.9068, -43.1729),
    CityRef("Belo Horizonte", "MG", "Brasil", -19.9167, -43.9345),
    CityRef("Curitiba", "PR", "Brasil", -25.4284, -49.2733),
    CityRef("Porto Alegre", "RS", "Brasil", -30.0346, -51.2177),
    CityRef("Salvador", "BA", "Brasil", -12.9777, -38.5016),
    CityRef("Recife", "PE", "Brasil", -8.0476, -34.8770),
    CityRef("Fortaleza", "CE", "Brasil", -3.7172, -38.5433),
    CityRef("Brasília", "DF", "Brasil", -15.7939, -47.8828),
    CityRef("Manaus", "AM", "Brasil", -3.1190, -60.0217),
    CityRef("Florianópolis", "SC", "Brasil", -27.5954, -48.5480),
    CityRef("Goiânia", "GO", "Brasil", -16.6869, -49.2648),
    CityRef("Belém", "PA", "Brasil", -1.4558, -48.4902),
    CityRef("Vitória", "ES", "Brasil", -20.3155, -40.3128),
    CityRef("Natal", "RN", "Brasil", -5.7945, -35.2110),
    CityRef("Campo Grande", "MS", "Brasil", -20.4697, -54.6201),
    CityRef("João Pessoa", "PB", "Brasil", -7.1195, -34.8450),
    CityRef("Maceió", "AL", "Brasil", -9.6498, -35.7089),
    CityRef("Teresina", "PI", "Brasil", -5.0892, -42.8019),
    CityRef("Cuiabá", "MT", "Brasil", -15.6014, -56.0979),
    CityRef("Londrina", "PR", "Brasil", -23.3045, -51.1696),
    CityRef("Ribeirão Preto", "SP", "Brasil", -21.1775, -47.8103),
    CityRef("Campinas", "SP", "Brasil", -22.9099, -47.0626),
    # --- Internacional ---
    CityRef("Lisboa", None, "Portugal", 38.7223, -9.1393),
    CityRef("Buenos Aires", None, "Argentina", -34.6037, -58.3816),
    CityRef("Miami", "FL", "Estados Unidos", 25.7617, -80.1918),
    CityRef("Madri", None, "Espanha", 40.4168, -3.7038),
    CityRef("Paris", None, "França", 48.8566, 2.3522),
]


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distância geodésica (grande círculo) entre dois pontos, em km.

    Usa `geopy.distance.geodesic` (fórmula de Vincenty/WGS-84) — cálculo
    puramente local, sem chamada de rede. Sempre >= 0 por construção
    (é uma distância física).
    """
    return geodesic((lat1, lon1), (lat2, lon2)).km
