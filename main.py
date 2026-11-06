"""Carrega um INI e executa as simulações e análises habilitadas nele."""

import argparse

from src.configuracao import carregar_configuracao
from src.pipeline import executar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configuracao",
        nargs="?",
        default="configuracoes/gerar_analisar_modelo_linear.ini",
        help="arquivo INI do projeto",
    )
    args = parser.parse_args()
    configuracao = carregar_configuracao(args.configuracao)
    print(f"Projeto: {configuracao.nome}")
    print(f"Resultados em: {configuracao.saida}")
    executar(configuracao)


if __name__ == "__main__":
    main()
