"""Executa as simulações e análises ativadas em uma configuração de projeto."""

from dataclasses import asdict
import json

import matplotlib.pyplot as plt
import numpy as np

from .configuracao import ConfiguracaoProjeto
from .modelos import (
    ModeloBoostChaveado,
    ModeloBoostLinear,
    ModeloBoostLinearMatrizes,
    ModeloBoostMedio,
)
from .simulacao import simular
from utils.analise_estrutural import verificar_controlabilidade_observabilidade
from utils.analise_frequencial import (
    calcular_margens,
    calcular_polos_zeros,
    criar_sys,
    plotar_bode,
    plotar_lugar_raizes,
    plotar_polos_zeros,
)
from utils.analise_divergencia import analisar_faixa_duty
from utils.estilo_graficos import configurar_estilo_ieee
from utils.gerar_matrizes import gerar_matrizes


def executar(config: ConfiguracaoProjeto) -> None:
    """Executa somente os modelos e análises habilitados no INI."""
    configurar_estilo_ieee()
    config.saida.mkdir(parents=True, exist_ok=True)
    p = config.parametros
    il_media = (
        config.tensao_saida_alvo / p.resistencia_carga + p.corrente_carga_externa
    ) / (1.0 - p.duty_cycle)
    ponto = np.array([il_media, config.tensao_saida_alvo])
    n_amostras = int(config.duracao * config.frequencia_amostragem) + 1
    tempos = np.linspace(0.0, config.duracao, n_amostras)
    x0 = np.asarray(config.estado_inicial)
    classes = {
        "medio": ModeloBoostMedio,
        "chaveado": ModeloBoostChaveado,
        "linear": ModeloBoostLinear,
        "matricial": ModeloBoostLinearMatrizes,
    }
    respostas = {}

    if config.analises.get("simulacao", False):
        for nome, ativado in config.modelos.items():
            if not ativado:
                continue
            modelo = classes[nome]
            if nome in ("linear", "matricial"):
                modelo = modelo(il_media, config.tensao_saida_alvo)
                respostas[nome] = simular(modelo, tempos, p, x0 - ponto) + ponto
            else:
                respostas[nome] = simular(modelo(), tempos, p, x0)
        if respostas and config.plots.get("simulacao", False):
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.16, 5.0))
            for name, result in respostas.items():
                axes[0].plot(tempos, result[:, 0], label=name)
                axes[1].plot(tempos, result[:, 1], label=name)
            axes[0].set_ylabel("Corrente do indutor $i_L$ (A)")
            axes[1].set_ylabel("Tensão de saída $v_o$ (V)")
            axes[1].set_xlabel("Tempo (s)")
            for axis in axes:
                axis.grid(True, alpha=0.3)
                axis.legend()
            fig.tight_layout()
            fig.savefig(config.saida / "comparacao_modelos.png")

    needs_matrices = any(
        config.analises.get(key, False)
        for key in ("matrizes", "frequencial", "estrutural")
    )
    matrices = (
        gerar_matrizes(p, config.tensao_saida_alvo, il_media)
        if needs_matrices
        else None
    )
    if config.analises.get("matrizes", False):
        payload = {
            "parametros_boost": asdict(p),
            "ponto_operacao": {
                "tensao_saida_V": config.tensao_saida_alvo,
                "corrente_indutor_A": il_media,
                "duty_cycle": p.duty_cycle,
            },
            "entradas_separadas": {
                key: matrices[key].tolist() for key in ("A", "B_d", "B_v")
            },
            "entradas_combinadas": {
                key: matrices[key].tolist() for key in ("A", "B", "C", "D")
            },
        }
        (config.saida / "matrizes_modelo_linear.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    if config.analises.get("estrutural", False):
        result = verificar_controlabilidade_observabilidade(
            matrices["A"], matrices["B"], matrices["C"]
        )
        print(
            f"Controlável: {result['controlavel']} (posto {result['posto_Mc']}); "
            f"observável: {result['observavel']} (posto {result['posto_Mo']})."
        )

    if config.analises.get("frequencial", False):
        system = criar_sys(
            config.entrada_frequencial, p, config.tensao_saida_alvo, il_media
        )
        poles, zeros = calcular_polos_zeros(system)
        print(f"Polos: {poles}\nZeros: {zeros}")
        if config.plots.get("frequencial", False):
            plotar_polos_zeros(poles, zeros, config.saida / "polos_zeros.png")
            pm, gm, wgc, wpc = plotar_bode(system, config.saida / "bode.png")
            plotar_lugar_raizes(system, config.saida / "lugar_raizes.png")
        else:
            pm = gm = wgc = wpc = float("nan")
        print(f"Margem de fase: {pm:.3f} deg; margem de ganho: {gm:.3f} dB")

    if config.analises.get("divergencia", False):
        duties = np.linspace(config.duty_min, config.duty_max, config.quantidade_duties)
        dados = analisar_faixa_duty(
            duties,
            config.duracao,
            config.frequencia_amostragem,
            p,
            config.tensao_saida_alvo,
        )
        if config.plots.get("divergencia", False):
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.16, 5.0))
            axes[0].plot(
                duties,
                [item["nrmse_corrente_percentual"] for item in dados],
                "o-",
                label="NRMSE $i_L$",
            )
            axes[0].plot(
                duties,
                [item["nrmse_tensao_percentual"] for item in dados],
                "s-",
                label="NRMSE $v_o$",
            )
            axes[0].set_ylabel("NRMSE da trajetória (%)")
            axes[0].legend()
            axes[1].plot(
                duties,
                [item["erro_final_tensao_percentual"] for item in dados],
                "o-",
                color="tab:red",
            )
            axes[1].set_xlabel("Duty cycle")
            axes[1].set_ylabel("Erro final de $v_o$ (%)")
            for axis in axes:
                axis.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(config.saida / "divergencia_modelo_linear.png")
        for item in dados:
            print(
                f"D={item['duty']:.5f}: NRMSE iL={item['nrmse_corrente_percentual']:.3f}%, "
                f"NRMSE vo={item['nrmse_tensao_percentual']:.3f}%, "
                f"t_acomodacao={item['tempo_acomodacao_s']:.6g} s"
            )

    if config.mostrar_graficos:
        plt.show()
    else:
        plt.close("all")
