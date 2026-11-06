"""Gera e exibe a representação em espaço de estados do modelo linear."""

import numpy as np

from src import ModeloBoostLinear
from src.benchmarks import PARAMETROS_NOMINAIS, TENSAO_SAIDA_ALVO
from src.modelos import ParametrosBoost


def gerar_matrizes(
    parametros: ParametrosBoost = PARAMETROS_NOMINAIS,
    tensao_saida_media: float = TENSAO_SAIDA_ALVO,
    corrente_indutor_media: float | None = None,
) -> dict[str, np.ndarray]:
    """Monta as matrizes no ponto de operação informado ou no benchmark.

    O vetor combinado de entrada é ``u = [delta_D, delta_Vin]`` e a saída é
    ``delta_v_o``. Logo, a matriz de transmissão direta combinada tem duas
    colunas; separadamente, cada entrada tem ``D = [[0]]``.
    """
    p = parametros
    if corrente_indutor_media is None:
        corrente_indutor_media = (
            tensao_saida_media / p.resistencia_carga + p.corrente_carga_externa
        ) / (1.0 - p.duty_cycle)
    modelo = ModeloBoostLinear(corrente_indutor_media, tensao_saida_media)
    a, b_d, b_v = modelo.matrizes(p)
    c = np.array([[0.0, 1.0]])
    d_d = np.zeros((1, 1))
    d_v = np.zeros((1, 1))

    return {
        "A": a,
        "B_d": b_d,
        "B_v": b_v,
        "B": np.hstack((b_d, b_v)),
        "C": c,
        "D_d": d_d,
        "D_v": d_v,
        "D": np.hstack((d_d, d_v)),
    }


def main() -> None:
    matrizes = gerar_matrizes()
    print("Modelo com entradas separadas: dx/dt = A*x + B_d*delta_D + B_v*delta_Vin")
    for nome in ("A", "B_d", "B_v"):
        print(f"{nome} =\n{matrizes[nome]}\n")

    print("Modelo MIMO: dx/dt = A*x + B*u; y = C*x + D*u")
    print("u = [delta_D, delta_Vin]^T")
    for nome in ("A", "B", "C", "D"):
        print(f"{nome} =\n{matrizes[nome]}\n")
    print("Para saída de tensão absoluta, v_o = 24 V + y no ponto de operação.")


if __name__ == "__main__":
    main()
