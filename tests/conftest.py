import pytest

from src import ParametrosBoost


@pytest.fixture
def parametros_nominais() -> ParametrosBoost:
    """Benchmark nominal fornecido para o conversor boost."""
    return ParametrosBoost(
        vin=12.0,
        resistencia_carga=28.8,
        indutancia=470e-6,
        capacitancia=47e-6,
        resistencia_indutor=0.125,
        duty_cycle=0.50,
        frequencia_chaveamento=50e3,
    )
