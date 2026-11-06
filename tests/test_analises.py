import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.analise_frequencial import (
    calcular_polos_zeros,
    criar_sys,
    plotar_lugar_raizes,
)
from utils.gerar_matrizes import gerar_matrizes


class TestAnalises(unittest.TestCase):
    def test_matrizes_de_espaco_de_estados_tem_dimensoes_corretas(self):
        matrizes = gerar_matrizes()
        self.assertEqual(matrizes["A"].shape, (2, 2))
        self.assertEqual(matrizes["B"].shape, (2, 2))
        self.assertEqual(matrizes["C"].shape, (1, 2))
        self.assertEqual(matrizes["D"].shape, (1, 2))

    def test_polos_zeros_e_lugar_das_raizes(self):
        sistema = criar_sys("duty")
        polos, zeros = calcular_polos_zeros(sistema)
        self.assertEqual(polos.size, 2)
        self.assertTrue(np.all(polos.real < 0.0))
        self.assertEqual(zeros.size, 1)
        self.assertGreater(zeros[0].real, 0.0)
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "root_locus.png"
            plotar_lugar_raizes(sistema, caminho)
            self.assertTrue(caminho.is_file())


if __name__ == "__main__":
    unittest.main()
