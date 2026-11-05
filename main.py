"""Executa e compara os modelos médio e chaveado do conversor boost."""

import matplotlib.pyplot as plt
import numpy as np

from src import (
    ModeloBoostChaveado,
    ModeloBoostMedio,
    ParametrosBoost,
    simular,
)
from src.benchmarks import DUTY_CYCLE_BENCHMARK, PARAMETROS_NOMINAIS


def main() -> None:
    parametros = PARAMETROS_NOMINAIS
    duracao = 30e-3
    frequencia_amostragem = 1e6
    quantidade_amostras = int(duracao * frequencia_amostragem) + 1
    tempos = np.linspace(0.0, duracao, quantidade_amostras)
    estado_inicial = (0.0, 0.0)

    resultado_medio = simular(
        ModeloBoostMedio(), tempos, parametros, estado_inicial=estado_inicial
    )
    resultado_chaveado = simular(
        ModeloBoostChaveado(), tempos, parametros, estado_inicial=estado_inicial
    )

    fig, eixos = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    eixos[0].plot(tempos, resultado_medio[:, 0], label="Modelo médio", linewidth=2)
    eixos[0].plot(
        tempos,
        resultado_chaveado[:, 0],
        label="Modelo chaveado",
        linewidth=1,
        alpha=0.8,
    )
    eixos[0].set_ylabel("Corrente do indutor $i_L$ (A)")
    eixos[0].grid(True, alpha=0.3)
    eixos[0].legend()

    eixos[1].plot(tempos, resultado_medio[:, 1], label="Modelo médio", linewidth=2)
    eixos[1].plot(
        tempos,
        resultado_chaveado[:, 1],
        label="Modelo chaveado",
        linewidth=1,
        alpha=0.8,
    )
    eixos[1].set_xlabel("Tempo (s)")
    eixos[1].set_ylabel("Tensão de saída $v_o$ (V)")
    eixos[1].grid(True, alpha=0.3)
    eixos[1].legend()

    fig.suptitle("Conversor boost: modelo médio e modelo chaveado")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
