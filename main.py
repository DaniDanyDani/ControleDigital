"""Executa e compara os modelos médio, chaveado e linear do boost."""

import json
from dataclasses import asdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src import (
    ModeloBoostChaveado,
    ModeloBoostLinear,
    ModeloBoostLinearMatrizes,
    ModeloBoostMedio,
    simular,
)
from src.benchmarks import PARAMETROS_NOMINAIS, TENSAO_SAIDA_ALVO
from utils.estilo_graficos import configurar_estilo_ieee
from utils.gerar_matrizes import gerar_matrizes


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
    resultado_matrizes = simular(
        ModeloBoostLinearMatrizes(corrente_indutor_media, TENSAO_SAIDA_ALVO),
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

    matrizes = gerar_matrizes()
    dados_matrizes = {
        "parametros_boost": {
            **asdict(parametros),
            "unidades": {
                "vin": "V",
                "indutancia": "H",
                "capacitancia": "F",
                "resistencia_carga": "ohm",
                "resistencia_indutor": "ohm",
                "corrente_carga_externa": "A",
                "duty_cycle": "adimensional",
                "frequencia_chaveamento": "Hz",
            },
        },
        "constantes_benchmark": {
            "tensao_saida_alvo_V": TENSAO_SAIDA_ALVO,
            "resistencia_carga_nominal_ohm": parametros.resistencia_carga,
            "resistencia_indutor_nominal_ohm": parametros.resistencia_indutor,
        },
        "ponto_operacao": {
            "duty_cycle": parametros.duty_cycle,
            "tensao_saida_V": TENSAO_SAIDA_ALVO,
            "corrente_indutor_A": corrente_indutor_media,
        },
        "simulacao": {
            "duracao_s": duracao,
            "frequencia_amostragem_Hz": frequencia_amostragem,
            "quantidade_amostras": int(quantidade_amostras),
            "estado_inicial_absoluto": estado_inicial.tolist(),
            "estado_inicial_linear_desvio": (estado_inicial - ponto_operacao).tolist(),
            "perturbacao_duty_cycle": 0.0,
            "perturbacao_vin_V": 0.0,
        },
        "entradas_separadas": {
            "equacao": "dx/dt = A*x + B_d*delta_D + B_v*delta_Vin",
            "A": matrizes["A"].tolist(),
            "B_d": matrizes["B_d"].tolist(),
            "B_v": matrizes["B_v"].tolist(),
        },
        "entradas_combinadas": {
            "equacao": "dx/dt = A*x + B*u; y = C*x + D*u",
            "ordem_entradas": ["delta_D", "delta_Vin"],
            "saida": "delta_v_o",
            "A": matrizes["A"].tolist(),
            "B": matrizes["B"].tolist(),
            "C": matrizes["C"].tolist(),
            "D": matrizes["D"].tolist(),
        },
    }
    caminho_json = pasta_resultados / "matrizes_modelo_linear.json"
    with caminho_json.open("w", encoding="utf-8") as arquivo:
        json.dump(dados_matrizes, arquivo, indent=2, ensure_ascii=False)
    print(f"Matrizes salvas em: {caminho_json}")

    caminho_figura = pasta_resultados / "comparacao_tres_modelos.png"
    fig.savefig(caminho_figura)
    print(f"Gráfico salvo em: {caminho_figura}")

    figura_matrizes, eixos_matrizes = plt.subplots(
        2, 1, sharex=True, figsize=(7.16, 5.0)
    )
    series = (
        (resultado_medio, "Modelo médio", "-", 2.4),
        (resultado_linear, "Modelo linear", "--", 2.0),
        (resultado_matrizes, "Modelo por matrizes", ":", 2.2),
    )
    for resultado, rotulo, estilo, largura in series:
        eixos_matrizes[0].plot(
            tempos, resultado[:, 0], linestyle=estilo, linewidth=largura, label=rotulo
        )
        eixos_matrizes[1].plot(
            tempos, resultado[:, 1], linestyle=estilo, linewidth=largura, label=rotulo
        )
    eixos_matrizes[0].set_ylabel("Corrente do indutor $i_L$ (A)")
    eixos_matrizes[1].set_ylabel("Tensão de saída $v_o$ (V)")
    eixos_matrizes[1].set_xlabel("Tempo (s)")
    for eixo in eixos_matrizes:
        eixo.grid(True, alpha=0.3)
        eixo.legend()
    figura_matrizes.suptitle("Comparação: modelo médio, linear e matricial")
    figura_matrizes.tight_layout()
    caminho_matrizes = pasta_resultados / "comparacao_modelo_matricial.png"
    figura_matrizes.savefig(caminho_matrizes)
    print(f"Comparação matricial salva em: {caminho_matrizes}")
    plt.show()


if __name__ == "__main__":
    main()
