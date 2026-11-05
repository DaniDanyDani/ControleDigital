"""Integração numérica dos modelos do conversor boost."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from .modelos import ModeloBoost, ParametrosBoost, VetorEstados


def simular(
    modelo: ModeloBoost,
    tempos: ArrayLike,
    parametros: ParametrosBoost,
    estado_inicial: ArrayLike = (0.0, 0.0),
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> NDArray[np.float64]:
    """Simula o modelo e retorna uma matriz ``(n_tempos, 2)``.

    As colunas do resultado são, nesta ordem, corrente do indutor ``i_L``
    (A) e tensão de saída ``v_o`` (V). Para o modelo chaveado, a integração
    é reiniciada em cada borda PWM.
    """
    vetor_tempo = np.asarray(tempos, dtype=float)
    x0 = np.asarray(estado_inicial, dtype=float)

    if vetor_tempo.ndim != 1 or vetor_tempo.size == 0:
        raise ValueError("tempos precisa ser um vetor não vazio.")
    if not np.all(np.isfinite(vetor_tempo)):
        raise ValueError("Todos os instantes de tempo precisam ser finitos.")
    if vetor_tempo.size > 1 and np.any(np.diff(vetor_tempo) <= 0.0):
        raise ValueError("Os instantes de tempo precisam ser estritamente crescentes.")
    if x0.shape != (2,) or not np.all(np.isfinite(x0)):
        raise ValueError("estado_inicial precisa conter [i_L, v_o] finitos.")
    if rtol <= 0.0 or atol <= 0.0:
        raise ValueError("rtol e atol precisam ser maiores que zero.")

    resultado = np.empty((vetor_tempo.size, 2), dtype=float)
    resultado[0] = x0
    if vetor_tempo.size == 1:
        return resultado

    inicio, fim = vetor_tempo[0], vetor_tempo[-1]
    bordas_modelo = modelo.tempos_descontinuidade(inicio, fim, parametros)
    bordas = np.unique(np.concatenate(([inicio, fim], bordas_modelo)))
    estados_atuais = x0.copy()

    for esquerda, direita in zip(bordas[:-1], bordas[1:]):
        indice_inicio = np.searchsorted(vetor_tempo, esquerda, side="right")
        indice_fim = np.searchsorted(vetor_tempo, direita, side="right")
        tempos_saida = vetor_tempo[indice_inicio:indice_fim]

        incluir_direita = tempos_saida.size == 0 or tempos_saida[-1] != direita
        tempos_integracao = (
            np.append(tempos_saida, direita) if incluir_direita else tempos_saida
        )
        instante_referencia = esquerda + (direita - esquerda) / 2.0

        solucao = solve_ivp(
            fun=lambda tempo, estados: modelo.derivadas_no_intervalo(
                tempo, estados, parametros, instante_referencia
            ),
            t_span=(esquerda, direita),
            y0=estados_atuais,
            t_eval=tempos_integracao,
            rtol=rtol,
            atol=atol,
        )
        if not solucao.success:
            raise RuntimeError(f"Falha na integração numérica: {solucao.message}")

        quantidade_saida = tempos_saida.size
        if quantidade_saida:
            resultado[indice_inicio:indice_fim] = solucao.y[:, :quantidade_saida].T
        estados_atuais = solucao.y[:, -1]

    return resultado
