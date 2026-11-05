"""Equações diferenciais dos modelos do conversor boost."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

VetorEstados = NDArray[np.float64]


@dataclass(frozen=True)
class ParametrosBoost:
    """Parâmetros elétricos e de chaveamento da planta.

    ``corrente_carga_externa`` representa a corrente adicional ``I_ext``
    drenada da saída, além da corrente consumida pelo resistor ``R``.
    """

    vin: float
    indutancia: float
    capacitancia: float
    resistencia_carga: float
    resistencia_indutor: float
    corrente_carga_externa: float = 0.0
    duty_cycle: float = 0.5
    frequencia_chaveamento: float = 20_000.0

    def __post_init__(self) -> None:
        valores = {
            "vin": self.vin,
            "indutancia": self.indutancia,
            "capacitancia": self.capacitancia,
            "resistencia_carga": self.resistencia_carga,
            "resistencia_indutor": self.resistencia_indutor,
            "corrente_carga_externa": self.corrente_carga_externa,
            "duty_cycle": self.duty_cycle,
            "frequencia_chaveamento": self.frequencia_chaveamento,
        }
        for nome, valor in valores.items():
            if not math.isfinite(valor):
                raise ValueError(f"O parâmetro {nome} precisa ser finito.")

        for nome, valor in (
            ("indutancia", self.indutancia),
            ("capacitancia", self.capacitancia),
            ("resistencia_carga", self.resistencia_carga),
            ("frequencia_chaveamento", self.frequencia_chaveamento),
        ):
            if valor <= 0:
                raise ValueError(f"O parâmetro {nome} precisa ser maior que zero.")

        if self.resistencia_indutor < 0:
            raise ValueError("A resistência do indutor não pode ser negativa.")
        if not 0.0 <= self.duty_cycle <= 1.0:
            raise ValueError("O duty cycle precisa estar entre 0 e 1.")


class ModeloBoost(ABC):
    """Interface comum para modelos contínuos do conversor boost."""

    @abstractmethod
    def derivadas(
        self,
        tempo: float,
        estados: VetorEstados,
        parametros: ParametrosBoost,
    ) -> VetorEstados:
        """Retorna ``[di_L/dt, dv_o/dt]`` no instante informado."""

    def tempos_descontinuidade(
        self,
        inicio: float,
        fim: float,
        parametros: ParametrosBoost,
    ) -> NDArray[np.float64]:
        """Instantes internos em que a equação do modelo muda."""
        return np.empty(0, dtype=float)

    def derivadas_no_intervalo(
        self,
        tempo: float,
        estados: VetorEstados,
        parametros: ParametrosBoost,
        instante_referencia: float,
    ) -> VetorEstados:
        """Derivadas dentro de um trecho sem descontinuidades.

        Modelos contínuos podem usar a implementação padrão. Modelos
        chaveados podem fixar o modo usando o ponto médio do trecho, evitando
        que o integrador atravesse uma borda PWM durante um passo interno.
        """
        return self.derivadas(tempo, estados, parametros)


class ModeloBoostMedio(ModeloBoost):
    """Modelo médio com duty cycle constante."""

    def derivadas(
        self,
        tempo: float,
        estados: VetorEstados,
        parametros: ParametrosBoost,
    ) -> VetorEstados:
        del tempo  # O modelo médio não depende explicitamente do tempo.
        corrente_indutor, tensao_saida = estados
        p = parametros
        d = p.duty_cycle

        di_dt = (
            p.vin / p.indutancia
            - (1.0 - d) * tensao_saida / p.indutancia
            - p.resistencia_indutor * corrente_indutor / p.indutancia
        )
        dv_dt = (
            (1.0 - d) * corrente_indutor / p.capacitancia
            - tensao_saida / (p.resistencia_carga * p.capacitancia)
            - p.corrente_carga_externa / p.capacitancia
        )
        return np.asarray([di_dt, dv_dt], dtype=float)


class ModeloBoostChaveado(ModeloBoost):
    """Modelo com PWM ideal e equações distintas nos dois estados da chave."""

    @staticmethod
    def chave_ligada(tempo: float, parametros: ParametrosBoost) -> bool:
        periodo = 1.0 / parametros.frequencia_chaveamento
        fase = tempo % periodo
        return fase < parametros.duty_cycle * periodo

    def _derivadas_modo(
        self,
        estados: VetorEstados,
        parametros: ParametrosBoost,
        chave_ligada: bool,
    ) -> VetorEstados:
        corrente_indutor, tensao_saida = estados
        p = parametros

        if chave_ligada:
            di_dt = (p.vin - p.resistencia_indutor * corrente_indutor) / p.indutancia
            dv_dt = (
                -tensao_saida / p.resistencia_carga - p.corrente_carga_externa
            ) / p.capacitancia
        else:
            di_dt = (
                p.vin - tensao_saida - p.resistencia_indutor * corrente_indutor
            ) / p.indutancia
            dv_dt = (
                corrente_indutor
                - tensao_saida / p.resistencia_carga
                - p.corrente_carga_externa
            ) / p.capacitancia

        return np.asarray([di_dt, dv_dt], dtype=float)

    def derivadas(
        self,
        tempo: float,
        estados: VetorEstados,
        parametros: ParametrosBoost,
    ) -> VetorEstados:
        return self._derivadas_modo(
            estados,
            parametros,
            chave_ligada=self.chave_ligada(tempo, parametros),
        )

    def derivadas_no_intervalo(
        self,
        tempo: float,
        estados: VetorEstados,
        parametros: ParametrosBoost,
        instante_referencia: float,
    ) -> VetorEstados:
        return self._derivadas_modo(
            estados,
            parametros,
            chave_ligada=self.chave_ligada(instante_referencia, parametros),
        )

    def tempos_descontinuidade(
        self,
        inicio: float,
        fim: float,
        parametros: ParametrosBoost,
    ) -> NDArray[np.float64]:
        duty = parametros.duty_cycle
        if duty <= 0.0 or duty >= 1.0:
            return np.empty(0, dtype=float)

        periodo = 1.0 / parametros.frequencia_chaveamento
        tempo_ligado = duty * periodo
        primeiro_ciclo = math.floor(inicio / periodo)
        ultimo_ciclo = math.ceil(fim / periodo)
        ciclos = np.arange(primeiro_ciclo, ultimo_ciclo + 1, dtype=float)
        bordas = np.concatenate((ciclos * periodo, ciclos * periodo + tempo_ligado))
        return np.unique(bordas[(bordas > inicio) & (bordas < fim)])
