"""Verifica controlabilidade e observabilidade por posto das matrizes."""

import numpy as np

from utils.gerar_matrizes import gerar_matrizes


def matriz_controlabilidade(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Constrói Mc = [B, AB, ..., A^(n-1)B]."""
    n = a.shape[0]
    return np.hstack([np.linalg.matrix_power(a, k) @ b for k in range(n)])


def matriz_observabilidade(a: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Constrói Mo = [C; CA; ..., CA^(n-1)]."""
    n = a.shape[0]
    return np.vstack([c @ np.linalg.matrix_power(a, k) for k in range(n)])


def verificar_controlabilidade_observabilidade(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> dict[str, object]:
    """Retorna matrizes, postos e indicadores de controlabilidade/observabilidade."""
    mc = matriz_controlabilidade(a, b)
    mo = matriz_observabilidade(a, c)
    n = a.shape[0]
    posto_mc = int(np.linalg.matrix_rank(mc))
    posto_mo = int(np.linalg.matrix_rank(mo))
    return {
        "Mc": mc,
        "posto_Mc": posto_mc,
        "controlavel": posto_mc == n,
        "Mo": mo,
        "posto_Mo": posto_mo,
        "observavel": posto_mo == n,
    }


def main() -> None:
    matrizes = gerar_matrizes()
    resultado = verificar_controlabilidade_observabilidade(
        matrizes["A"], matrizes["B"], matrizes["C"]
    )
    n = matrizes["A"].shape[0]
    print("--- Controlabilidade ---")
    print(f"Matriz Mc:\n{resultado['Mc']}")
    print(f"Posto de Mc: {resultado['posto_Mc']} (necessário: {n})")
    print("O sistema é controlável." if resultado["controlavel"] else "O sistema não é controlável.")
    print("\n--- Observabilidade ---")
    print(f"Matriz Mo:\n{resultado['Mo']}")
    print(f"Posto de Mo: {resultado['posto_Mo']} (necessário: {n})")
    print("O sistema é observável." if resultado["observavel"] else "O sistema não é observável.")


if __name__ == "__main__":
    main()
