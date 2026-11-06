"""Executa e compara os modelos médio, chaveado e linear do boost."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src import (
    ModeloBoostChaveado,
    ModeloBoostLinear,
    ModeloBoostMedio,
    simular,
)
from src.benchmarks import PARAMETROS_NOMINAIS, TENSAO_SAIDA_ALVO
from utils.estilo_graficos import configurar_estilo_ieee


def main() -> None:
    configurar_estilo_ieee()
    parametros = PARAMETROS_NOMINAIS
    corrente_indutor_media = TENSAO_SAIDA_ALVO / (
        parametros.resistencia_carga * (1.0 - parametros.duty_cycle)
    )
    ponto_operacao = (corrente_indutor_media, TENSAO_SAIDA_ALVO)
    duracao = 30e-3
    frequencia_amostragem = 1e6
    quantidade_amostras = int(duracao * frequencia_amostragem) + 1
    tempos = np.linspace(0.0, duracao, quantidade_amostras)
    estado_inicial = np.zeros(2)

    resultado_medio = simular(
        ModeloBoostMedio(), tempos, parametros, estado_inicial=estado_inicial
    )
    resultado_chaveado = simular(
        ModeloBoostChaveado(), tempos, parametros, estado_inicial=estado_inicial
    )
    # O modelo linear usa estados de desvio em relação ao ponto de operação.
    perturbacao_duty_cycle = 0.0
    resultado_linear = simular(
        ModeloBoostLinear(
            corrente_indutor_media,
            TENSAO_SAIDA_ALVO,
            duty_perturbacao=perturbacao_duty_cycle,
        ),
        tempos,
        parametros,
        estado_inicial=estado_inicial - ponto_operacao,
    ) + np.asarray(ponto_operacao)

    fig, eixos = plt.subplots(2, 1, sharex=True, figsize=(7.16, 5.0))
    eixos[0].plot(tempos, resultado_medio[:, 0], label="Modelo médio", linewidth=2.4)
    eixos[0].plot(
        tempos,
        resultado_chaveado[:, 0],
        label="Modelo chaveado",
        linewidth=1.8,
        alpha=0.8,
    )
    eixos[0].plot(tempos, resultado_linear[:, 0], "--", linewidth=2.2, label="Modelo linear")
    eixos[0].set_ylabel("Corrente do indutor $i_L$ (A)")
    eixos[0].grid(True, alpha=0.3)
    eixos[0].legend()

    eixos[1].plot(tempos, resultado_medio[:, 1], label="Modelo médio", linewidth=2.4)
    eixos[1].plot(
        tempos,
        resultado_chaveado[:, 1],
        label="Modelo chaveado",
        linewidth=1.8,
        alpha=0.8,
    )
    eixos[1].plot(tempos, resultado_linear[:, 1], "--", linewidth=2.2, label="Modelo linear")
    eixos[1].set_xlabel("Tempo (s)")
    eixos[1].set_ylabel("Tensão de saída $v_o$ (V)")
    eixos[1].grid(True, alpha=0.3)
    eixos[1].legend()

    fig.suptitle("Conversor boost: modelos médio, chaveado e linear")
    fig.tight_layout()
    pasta_resultados = Path("resultados")
    pasta_resultados.mkdir(parents=True, exist_ok=True)
    caminho_figura = pasta_resultados / "comparacao_tres_modelos.png"
    fig.savefig(caminho_figura)
    print(f"Gráfico salvo em: {caminho_figura}")
    plt.show()


if __name__ == "__main__":
    main()
