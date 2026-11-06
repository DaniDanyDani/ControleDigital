import unittest

import numpy as np

from src import ModeloBoostChaveado, ModeloBoostLinear, ModeloBoostMedio, ParametrosBoost
from src.benchmarks import PARAMETROS_NOMINAIS, TENSAO_SAIDA_ALVO
from src.simulacao import simular


class TestModelosBoost(unittest.TestCase):
    def setUp(self):
        self.p = ParametrosBoost(12.0, 470e-6, 47e-6, 28.8, 0.125, duty_cycle=0.5, frequencia_chaveamento=50e3)

    def test_modelo_medio(self):
        actual = ModeloBoostMedio().derivadas(0.0, np.array([2.0, 18.0]), self.p)
        expected = [12/470e-6 - .5*18/470e-6 - .125*2/470e-6, .5*2/47e-6 - 18/(28.8*47e-6)]
        np.testing.assert_allclose(actual, expected)

    def test_modelo_chaveado(self):
        model = ModeloBoostChaveado()
        x = np.array([2.0, 18.0])
        np.testing.assert_allclose(model.derivadas(1e-6, x, self.p), [(12-.125*2)/470e-6, (-18/28.8)/47e-6])
        np.testing.assert_allclose(model.derivadas(15e-6, x, self.p), [(12-18-.125*2)/470e-6, (2-18/28.8)/47e-6])

    def test_parametros_invalidos(self):
        with self.assertRaises(ValueError):
            ParametrosBoost(12, 0, 47e-6, 28.8, .125)

    def test_modelo_linear_em_equilibrio(self):
        # Linear states are deviations, so zero deviation and zero input
        # perturbations must remain at the operating point.
        model = ModeloBoostLinear(1.704, 24.0)
        np.testing.assert_allclose(
            model.derivadas(0.0, np.zeros(2), self.p), [0.0, 0.0]
        )

    def test_linear_reproduz_medio_na_partida(self):
        p = PARAMETROS_NOMINAIS
        IL = TENSAO_SAIDA_ALVO / (p.resistencia_carga * (1 - p.duty_cycle))
        X = np.array([IL, TENSAO_SAIDA_ALVO])
        t = np.linspace(0, 10e-3, 2001)
        med = simular(ModeloBoostMedio(), t, p, (0.0, 0.0))
        lin = simular(ModeloBoostLinear(IL, TENSAO_SAIDA_ALVO), t, p, -X) + X
        np.testing.assert_allclose(lin, med, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
