"""Análise de polos, zeros e resposta em frequência do modelo linear."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from src import ModeloBoostLinear
from src.benchmarks import PARAMETROS_NOMINAIS, TENSAO_SAIDA_ALVO
from src.modelos import ParametrosBoost
from utils.estilo_graficos import configurar_estilo_ieee


def criar_sys(
    entrada: str = "duty",
    parametros: ParametrosBoost = PARAMETROS_NOMINAIS,
    tensao_saida_media: float = TENSAO_SAIDA_ALVO,
    corrente_indutor_media: float | None = None,
) -> signal.StateSpace:
    """Cria sistema SISO para ``delta_D`` ou ``delta_Vin`` no ponto informado."""
    p = parametros
    if corrente_indutor_media is None:
        corrente_indutor_media = (
            tensao_saida_media / p.resistencia_carga + p.corrente_carga_externa
        ) / (1.0 - p.duty_cycle)
    modelo = ModeloBoostLinear(corrente_indutor_media, tensao_saida_media)
    a, b_d, b_v = modelo.matrizes(p)
    if entrada == "duty":
        b, d = b_d, np.zeros((1, 1))
    elif entrada == "vin":
        b, d = b_v, np.zeros((1, 1))
    else:
        raise ValueError("entrada deve ser 'duty' ou 'vin'.")
    c = np.array([[0.0, 1.0]])  # Saída: desvio de tensão delta_v_o.
    return signal.StateSpace(a, b, c, d)


def calcular_polos_zeros(sistema: signal.StateSpace) -> tuple[np.ndarray, np.ndarray]:
    """Retorna os polos e zeros do canal SISO em espaço de estados."""
    a = np.asarray(sistema.A)
    b = np.asarray(sistema.B)
    c = np.asarray(sistema.C)
    d = np.asarray(sistema.D)
    polos = np.linalg.eigvals(a)

    # Para o modelo de ordem 2, o numerador é C*adj(sI-A)*B + D*det(sI-A).
    tr_a = np.trace(a)
    denominador_adjunto = c @ b
    coeficiente_s = float(denominador_adjunto[0, 0])
    termo_constante = float((c @ (a - tr_a * np.eye(2)) @ b)[0, 0])
    if d[0, 0] != 0.0:
        raise ValueError("O cálculo dos zeros espera D=0 para este modelo.")
    zeros = (
        np.roots([coeficiente_s, termo_constante])
        if abs(coeficiente_s) > 1e-14
        else np.array([], dtype=complex)
    )
    return polos, np.asarray(zeros, dtype=complex)


def plotar_polos_zeros(
    polos: np.ndarray, zeros: np.ndarray, caminho: Path | None = None
) -> None:
    """Plota polos como X e zeros como O no plano complexo."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(polos.real, polos.imag, marker="x", s=100, linewidths=2.5, label="Polos")
    if zeros.size:
        ax.scatter(zeros.real, zeros.imag, marker="o", s=110, color="tab:red", linewidths=2, label="Zeros", zorder=5)
        for zero in zeros:
            ax.annotate(
                f"z = {zero.real:.3g}",
                (zero.real, zero.imag),
                xytext=(-8, 12),
                textcoords="offset points",
                ha="right",
                color="tab:red",
            )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Parte real (rad/s)")
    ax.set_ylabel("Parte imaginária (rad/s)")
    ax.set_title("Mapa de polos e zeros")
    if zeros.size:
        xmin = min(polos.real.min(), zeros.real.min())
        xmax = max(polos.real.max(), zeros.real.max())
        margem = 0.08 * (xmax - xmin)
        ax.set_xlim(xmin - margem, xmax + margem)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if caminho:
        fig.savefig(caminho)


def calcular_margens(
    frequencia: np.ndarray, magnitude_db: np.ndarray, fase_graus: np.ndarray
) -> tuple[float, float, float, float]:
    """Retorna margem de fase/ganho e frequências de cruzamento para unity feedback."""
    cruzamento_ganho = np.flatnonzero(
        (magnitude_db[:-1] >= 0.0) & (magnitude_db[1:] < 0.0)
    )
    if cruzamento_ganho.size:
        i = cruzamento_ganho[0]
        frac = magnitude_db[i] / (magnitude_db[i] - magnitude_db[i + 1])
        log_w = np.log10(frequencia[i]) + frac * (
            np.log10(frequencia[i + 1]) - np.log10(frequencia[i])
        )
        fase_cross = np.interp(log_w, np.log10(frequencia), fase_graus)
        w_gc = 10.0**log_w
        pm = 180.0 + fase_cross
    else:
        w_gc, pm = float("nan"), float("nan")

    diferenca_fase = fase_graus + 180.0
    cruzamento_fase = np.flatnonzero(diferenca_fase[:-1] * diferenca_fase[1:] <= 0.0)
    if cruzamento_fase.size:
        i = cruzamento_fase[0]
        frac = -diferenca_fase[i] / (diferenca_fase[i + 1] - diferenca_fase[i])
        log_w = np.log10(frequencia[i]) + frac * (
            np.log10(frequencia[i + 1]) - np.log10(frequencia[i])
        )
        gm_db = -np.interp(log_w, np.log10(frequencia), magnitude_db)
        w_pc = 10.0**log_w
    else:
        w_pc, gm_db = float("nan"), float("nan")
    return pm, gm_db, w_gc, w_pc


def _resposta_frequencia(
    sistema: signal.StateSpace, frequencia: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    identidade = np.eye(sistema.A.shape[0])
    resposta = np.array(
        [
            (sistema.C @ np.linalg.solve(1j * w * identidade - sistema.A, sistema.B)
             + sistema.D)[0, 0]
            for w in frequencia
        ]
    )
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(resposta), 1e-300))
    fase_graus = np.rad2deg(np.unwrap(np.angle(resposta)))
    return magnitude_db, fase_graus


def plotar_bode(
    sistema: signal.StateSpace,
    caminho: Path | None = None,
    caminho_margens: Path | None = None,
) -> tuple[float, float, float, float]:
    """Plota magnitude e fase da resposta em frequência do canal SISO."""
    polos = np.linalg.eigvals(sistema.A)
    frequencia = np.logspace(
        np.log10(max(np.min(np.abs(polos)) / 100.0, 1e-2)),
        np.log10(np.max(np.abs(polos)) * 100.0),
        600,
    )
    magnitude_db, fase_graus = _resposta_frequencia(sistema, frequencia)
    pm, gm_db, w_gc, w_pc = calcular_margens(frequencia, magnitude_db, fase_graus)
    fig, (ax_mag, ax_fase) = plt.subplots(2, 1, sharex=True, figsize=(7.16, 5.0))
    ax_mag.semilogx(frequencia, magnitude_db)
    ax_mag.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    if np.isfinite(w_gc):
        ax_mag.axvline(w_gc, color="tab:green", linestyle="--", linewidth=1.2)
        ax_mag.scatter([w_gc], [0], color="tab:green", zorder=5)
    if np.isfinite(w_pc):
        mag_pc = np.interp(np.log10(w_pc), np.log10(frequencia), magnitude_db)
        ax_mag.axvline(w_pc, color="tab:red", linestyle="--", linewidth=1.2)
        ax_mag.scatter([w_pc], [mag_pc], color="tab:red", zorder=5)
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_fase.semilogx(frequencia, fase_graus)
    ax_fase.axhline(-180.0, color="black", linewidth=0.8, linestyle=":")
    if np.isfinite(w_gc):
        phase_gc = np.interp(np.log10(w_gc), np.log10(frequencia), fase_graus)
        ax_fase.axvline(w_gc, color="tab:green", linestyle="--", linewidth=1.2)
        ax_fase.scatter([w_gc], [phase_gc], color="tab:green", zorder=5)
    if np.isfinite(w_pc):
        ax_fase.axvline(w_pc, color="tab:red", linestyle="--", linewidth=1.2)
    ax_fase.set_xlabel("Frequência (rad/s)")
    ax_fase.set_ylabel("Fase (graus)")
    ax_fase.grid(True, which="both", alpha=0.3)
    texto_pm = f"PM = {pm:.2f} deg @ {w_gc:.3g} rad/s" if np.isfinite(pm) else "PM = n/a"
    texto_gm = f"GM = {gm_db:.2f} dB @ {w_pc:.3g} rad/s" if np.isfinite(gm_db) else "GM = n/a"
    ax_mag.text(0.02, 0.03, texto_pm + "\n" + texto_gm, transform=ax_mag.transAxes, va="bottom")
    fig.tight_layout()
    if caminho:
        fig.savefig(caminho)
    if caminho_margens:
        fig.savefig(caminho_margens)
    return pm, gm_db, w_gc, w_pc


def plotar_lugar_raizes(
    sistema: signal.StateSpace, caminho: Path | None = None
) -> None:
    """Plota o lugar das raízes para realimentação unitária e ganho K >= 0."""
    a, b, c = sistema.A, sistema.B, sistema.C
    tr_a = np.trace(a)
    num = np.array(
        [float((c @ b)[0, 0]), float((c @ (a - tr_a * np.eye(2)) @ b)[0, 0])]
    )
    den = np.poly(a)
    ganhos = np.concatenate(([0.0], np.logspace(-4, 5, 700)))
    trajetorias = np.empty((ganhos.size, 2), dtype=complex)
    trajetorias[0] = np.roots(den)
    for idx, ganho in enumerate(ganhos[1:], start=1):
        raizes = np.roots(np.polyadd(den, ganho * num))
        if idx > 0:
            direto = np.sum(np.abs(raizes - trajetorias[idx - 1]))
            trocado = np.sum(np.abs(raizes[::-1] - trajetorias[idx - 1]))
            if trocado < direto:
                raizes = raizes[::-1]
        trajetorias[idx] = raizes

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for ramo in range(2):
        ax.plot(trajetorias[:, ramo].real, trajetorias[:, ramo].imag, linewidth=1.8)
    polos, zeros = calcular_polos_zeros(sistema)
    ax.scatter(polos.real, polos.imag, marker="x", s=90, linewidths=2, label="Polos")
    if zeros.size:
        ax.scatter(zeros.real, zeros.imag, marker="o", s=90, facecolors="none", linewidths=2, label="Zeros")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Parte real (rad/s)")
    ax.set_ylabel("Parte imaginária (rad/s)")
    ax.set_title("Lugar das raízes (realimentação unitária)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if caminho:
        fig.savefig(caminho)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", choices=("duty", "vin"), default="duty")
    args = parser.parse_args()
    configurar_estilo_ieee()

    sistema = criar_sys(args.entrada)
    polos, zeros = calcular_polos_zeros(sistema)
    print(f"Polos ({args.entrada}):\n{polos}")
    print(f"Zeros ({args.entrada}):\n{zeros}")

    pasta = Path("resultados")
    pasta.mkdir(exist_ok=True)
    plotar_polos_zeros(polos, zeros, pasta / f"polos_zeros_{args.entrada}.png")
    pm, gm, w_gc, w_pc = plotar_bode(sistema, pasta / f"bode_{args.entrada}.png")
    print(f"Margem de fase (unity feedback): {pm:.3f} deg; cruzamento: {w_gc:.3g} rad/s")
    print(f"Margem de ganho (unity feedback): {gm:.3f} dB; cruzamento: {w_pc:.3g} rad/s")
    plotar_lugar_raizes(sistema, pasta / f"lugar_raizes_{args.entrada}.png")
    plt.show()


if __name__ == "__main__":
    main()
