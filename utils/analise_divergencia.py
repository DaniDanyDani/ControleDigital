"""Compara o modelo médio com sua linearização no benchmark nominal.

Execute com ``python -m utils.analise_divergencia``. O duty cycle varre uma
faixa de pontos de operação, mantendo fixo o ponto usado na linearização.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src import ModeloBoostLinear, ModeloBoostMedio, simular
from src.benchmarks import PARAMETROS_NOMINAIS, TENSAO_SAIDA_ALVO
from src.modelos import ParametrosBoost
from utils.estilo_graficos import configurar_estilo_ieee


def _tempo_acomodacao(
    tempos: NDArray[np.float64], resposta: NDArray[np.float64], tolerancia: float = 0.02
) -> tuple[float, NDArray[np.float64]]:
    """Estima acomodação a 2% e o valor final pela média dos últimos 5%."""
    inicio_janela = max(1, int(0.95 * resposta.shape[0]))
    valores_finais = np.mean(resposta[inicio_janela:], axis=0)
    escalas = np.maximum(np.abs(valores_finais), np.max(np.abs(resposta), axis=0) * 1e-6)
    dentro_faixa = np.all(
        np.abs(resposta - valores_finais) <= tolerancia * escalas,
        axis=1,
    )
    acomodado_desde = np.logical_and.accumulate(dentro_faixa[::-1])[::-1]
    indices = np.flatnonzero(acomodado_desde)
    tempo = float(tempos[indices[0]]) if indices.size else float("nan")
    return tempo, valores_finais


def _nrmse_percentual(
    referencia: NDArray[np.float64], estimativa: NDArray[np.float64]
) -> NDArray[np.float64]:
    erro = np.sqrt(np.mean((estimativa - referencia) ** 2, axis=0))
    escala = np.ptp(referencia, axis=0)
    escala = np.where(escala > 1e-12, escala, np.maximum(np.abs(referencia[-1]), 1e-12))
    return 100.0 * erro / escala


def analisar_faixa_duty(
    duties: NDArray[np.float64],
    duracao: float = 30e-3,
    frequencia_amostragem: float = 1e6,
    parametros_nominais: ParametrosBoost = PARAMETROS_NOMINAIS,
    tensao_saida_alvo: float = TENSAO_SAIDA_ALVO,
) -> list[dict[str, float]]:
    """Simula as duas respostas para cada duty e calcula as métricas.

    A planta não linear parte do equilíbrio nominal. O modelo linear recebe
    ``delta_D = D - D_nominal``, começa com desvios nulos e tem o ponto nominal
    somado às respostas para retornar às grandezas absolutas.
    """
    p0 = parametros_nominais
    if np.any((duties <= 0.0) | (duties >= 1.0)):
        raise ValueError("Todos os duty cycles do range precisam estar entre 0 e 1.")
    if duracao <= 0.0 or frequencia_amostragem <= 0.0:
        raise ValueError("A duração e a frequência de amostragem precisam ser positivas.")

    corrente_media = (
        tensao_saida_alvo / p0.resistencia_carga + p0.corrente_carga_externa
    ) / (1.0 - p0.duty_cycle)
    ponto_operacao = np.array([corrente_media, tensao_saida_alvo])
    tempos = np.linspace(
        0.0, duracao, int(duracao * frequencia_amostragem) + 1
    )
    resultados = []

    for duty in duties:
        parametros_ponto = replace(p0, duty_cycle=float(duty))
        resposta_media = simular(
            ModeloBoostMedio(), tempos, parametros_ponto, estado_inicial=ponto_operacao
        )
        resposta_linear = simular(
            ModeloBoostLinear(
                corrente_media,
                tensao_saida_alvo,
                duty_perturbacao=float(duty - p0.duty_cycle),
            ),
            tempos,
            p0,
            estado_inicial=(0.0, 0.0),
        ) + ponto_operacao

        t_acomodacao, final_medio = _tempo_acomodacao(tempos, resposta_media)
        _, final_linear = _tempo_acomodacao(tempos, resposta_linear)
        nrmse = _nrmse_percentual(resposta_media, resposta_linear)
        erro_final = 100.0 * np.abs(final_linear - final_medio) / np.maximum(
            np.abs(final_medio), 1e-12
        )
        resultados.append(
            {
                "duty": float(duty),
                "delta_duty_percentual": 100.0 * float(duty - p0.duty_cycle),
                "tempo_acomodacao_s": t_acomodacao,
                "corrente_final_media_A": float(final_medio[0]),
                "corrente_final_linear_A": float(final_linear[0]),
                "tensao_final_media_V": float(final_medio[1]),
                "tensao_final_linear_V": float(final_linear[1]),
                "nrmse_corrente_percentual": float(nrmse[0]),
                "nrmse_tensao_percentual": float(nrmse[1]),
                "erro_final_corrente_percentual": float(erro_final[0]),
                "erro_final_tensao_percentual": float(erro_final[1]),
            }
        )
    return resultados


def main() -> None:
    configurar_estilo_ieee()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duty-min", type=float, default=0.45, help="Limite inferior do sweep")
    parser.add_argument("--duty-max", type=float, default=0.56, help="Limite superior do sweep")
    parser.add_argument("--pontos", type=int, default=12, help="Quantidade de duties avaliados")
    parser.add_argument("--duracao", type=float, default=30e-3, help="Duração da simulação em segundos")
    parser.add_argument("--fs-amostragem", type=float, default=1e6, help="Amostragem em Hz")
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("resultados") / "divergencia_modelo_linear.png",
        help="Arquivo PNG para salvar o gráfico",
    )
    parser.add_argument(
        "--nao-exibir",
        action="store_true",
        help="Salva o gráfico sem abrir a janela interativa",
    )
    args = parser.parse_args()
    if args.pontos < 2 or args.duty_min >= args.duty_max:
        parser.error("Informe duty-min < duty-max e pelo menos dois pontos.")

    duties = np.linspace(args.duty_min, args.duty_max, args.pontos)
    dados = analisar_faixa_duty(duties, args.duracao, args.fs_amostragem)

    print(
        "D       deltaD(%)  ts(ms)  i_final_medio(A)  i_final_linear(A)  "
        "v_final_medio(V)  v_final_linear(V)  NRMSE_i(%)  NRMSE_v(%)  erro_v_final(%)"
    )
    for item in dados:
        print(
            f"{item['duty']:.5f}  {item['delta_duty_percentual']:+8.3f}  "
            f"{item['tempo_acomodacao_s'] * 1e3:7.3f}  "
            f"{item['corrente_final_media_A']:16.5f}  "
            f"{item['corrente_final_linear_A']:17.5f}  "
            f"{item['tensao_final_media_V']:15.5f}  "
            f"{item['tensao_final_linear_V']:16.5f}  "
            f"{item['nrmse_corrente_percentual']:10.3f}  "
            f"{item['nrmse_tensao_percentual']:10.3f}  "
            f"{item['erro_final_tensao_percentual']:15.3f}"
        )

    figura, eixos = plt.subplots(2, 1, sharex=True, figsize=(7.16, 5.0))
    eixos[0].plot(
        [d["duty"] for d in dados],
        [d["nrmse_corrente_percentual"] for d in dados],
        "o-",
        label="Corrente $i_L$",
    )
    eixos[0].plot(
        [d["duty"] for d in dados],
        [d["nrmse_tensao_percentual"] for d in dados],
        "s-",
        label="Tensão $v_o$",
    )
    eixos[0].set_ylabel("NRMSE da trajetória (%)")
    eixos[0].grid(True, alpha=0.3)
    eixos[0].legend()
    eixos[1].plot(
        [d["duty"] for d in dados],
        [d["erro_final_tensao_percentual"] for d in dados],
        "o-",
        color="tab:red",
        label="Erro final de $v_o$",
    )
    eixos[1].set_xlabel("Duty cycle do ponto avaliado")
    eixos[1].set_ylabel("Erro após acomodação (%)")
    eixos[1].grid(True, alpha=0.3)
    figura.suptitle("Divergência do modelo médio linearizado")
    figura.tight_layout()
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(args.saida, dpi=160)
    print(f"Gráfico salvo em: {args.saida}")
    if not args.nao_exibir:
        plt.show()


if __name__ == "__main__":
    main()
