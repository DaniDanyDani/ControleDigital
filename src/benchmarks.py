"""Parâmetros nominais usados como benchmark do projeto."""

from .modelos import ParametrosBoost

VIN_NOMINAL = 12.0
TENSAO_SAIDA_ALVO = 24.0
RESISTENCIA_CARGA_NOMINAL = 28.8
INDUTANCIA_NOMINAL = 470e-6
CAPACITANCIA_NOMINAL = 47e-6
RESISTENCIA_INDUTOR_NOMINAL = 0.125
FREQUENCIA_CHAVEAMENTO_NOMINAL = 50e3

# Em regime permanente, q = 1-D satisfaz:
# Vo*q^2 - Vin*q + rL*Vo/R = 0 (I_ext = 0).
_discriminante = VIN_NOMINAL**2 - (
    4.0 * TENSAO_SAIDA_ALVO**2 * RESISTENCIA_INDUTOR_NOMINAL
    / RESISTENCIA_CARGA_NOMINAL
)
DUTY_CYCLE_BENCHMARK = 1.0 - (
    VIN_NOMINAL + _discriminante**0.5
) / (2.0 * TENSAO_SAIDA_ALVO)

PARAMETROS_NOMINAIS = ParametrosBoost(
    vin=VIN_NOMINAL,
    resistencia_carga=RESISTENCIA_CARGA_NOMINAL,
    indutancia=INDUTANCIA_NOMINAL,
    capacitancia=CAPACITANCIA_NOMINAL,
    resistencia_indutor=RESISTENCIA_INDUTOR_NOMINAL,
    duty_cycle=DUTY_CYCLE_BENCHMARK,
    frequencia_chaveamento=FREQUENCIA_CHAVEAMENTO_NOMINAL,
)
