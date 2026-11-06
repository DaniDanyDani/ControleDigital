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


class ModeloBoostLinear(ModeloBoostMedio):
    """Modelo de pequenos sinais linearizado em torno de um ponto de operação.

    Os estados são desvios ``[delta_i_L, delta_v_o]``. ``duty_perturbacao``
    e ``vin_perturbacao`` são entradas incrementais, não valores absolutos.
    """

    def __init__(
        self,
        corrente_indutor_media: float,
        tensao_saida_media: float,
        duty_perturbacao: float = 0.0,
        vin_perturbacao: float = 0.0,
    ) -> None:
        self.corrente_indutor_media = corrente_indutor_media
        self.tensao_saida_media = tensao_saida_media
        self.duty_perturbacao = duty_perturbacao
        self.vin_perturbacao = vin_perturbacao

    def matrizes(self, parametros: ParametrosBoost) -> tuple[NDArray[np.float64], ...]:
        """Retorna ``A``, ``B_d`` e ``B_v`` do modelo de pequenos sinais.

        Os estados são desvios em torno de ``corrente_indutor_media`` e
        ``tensao_saida_media``; as entradas são ``delta_D`` e ``delta_Vin``.
        """
        p = parametros
        a = np.array(
            [
                [-p.resistencia_indutor / p.indutancia, (p.duty_cycle - 1.0) / p.indutancia],
                [(1.0 - p.duty_cycle) / p.capacitancia,
                 -1.0 / (p.resistencia_carga * p.capacitancia)],
            ],
            dtype=float,
        )
        b_d = np.array(
            [
                [self.tensao_saida_media / p.indutancia],
                [-self.corrente_indutor_media / p.capacitancia],
            ],
            dtype=float,
        )
        b_v = np.array([[1.0 / p.indutancia], [0.0]], dtype=float)
        return a, b_d, b_v

    def derivadas(
        self,
        tempo: float,
        estados: VetorEstados,
        parametros: ParametrosBoost,
    ) -> VetorEstados:
        del tempo
        delta_i, delta_v = estados
        p = parametros
        q = 1.0 - p.duty_cycle
        di_dt = (
            -p.resistencia_indutor * delta_i / p.indutancia
            - q * delta_v / p.indutancia
            + self.tensao_saida_media * self.duty_perturbacao / p.indutancia
            + self.vin_perturbacao / p.indutancia
        )
        dv_dt = (
            q * delta_i / p.capacitancia
            - delta_v / (p.resistencia_carga * p.capacitancia)
            - self.corrente_indutor_media * self.duty_perturbacao / p.capacitancia
        )
        return np.asarray([di_dt, dv_dt], dtype=float)


class ModeloBoostLinearMatrizes(ModeloBoostLinear):
    """Modelo linear calculado diretamente por ``A*x + B_d*d + B_v*vin``."""

    def derivadas(
        self,
        tempo: float,
        estados: VetorEstados,
        parametros: ParametrosBoost,
    ) -> VetorEstados:
        del tempo
        a, b_d, b_v = self.matrizes(parametros)
        return (
            a @ estados
            + b_d[:, 0] * self.duty_perturbacao
            + b_v[:, 0] * self.vin_perturbacao
        )


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
