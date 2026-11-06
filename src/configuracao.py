"""Leitura e validação das configurações INI do conversor boost."""

from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path

from .modelos import ParametrosBoost


@dataclass(frozen=True)
class ConfiguracaoProjeto:
    nome: str
    parametros: ParametrosBoost
    tensao_saida_alvo: float
    duracao: float
    frequencia_amostragem: float
    estado_inicial: tuple[float, float]
    saida: Path
    modelos: dict[str, bool]
    analises: dict[str, bool]
    plots: dict[str, bool]
    entrada_frequencial: str
    duty_min: float
    duty_max: float
    quantidade_duties: int
    mostrar_graficos: bool


def carregar_configuracao(caminho: str | Path) -> ConfiguracaoProjeto:
    """Lê o INI, valida o tipo de projeto e constrói os parâmetros da planta."""
    arquivo = Path(caminho)
    parser = ConfigParser(interpolation=None)
    if not arquivo.is_file():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {arquivo}")
    parser.read(arquivo, encoding="utf-8")
    if parser.get("projeto", "tipo", fallback="").lower() != "conversorboost":
        raise ValueError("O campo [projeto] tipo deve ser ConversorBoost.")

    params = parser["parametros"]
    planta = ParametrosBoost(
        vin=params.getfloat("vin"),
        resistencia_carga=params.getfloat("resistencia_carga"),
        indutancia=params.getfloat("indutancia"),
        capacitancia=params.getfloat("capacitancia"),
        resistencia_indutor=params.getfloat("resistencia_indutor"),
        corrente_carga_externa=params.getfloat("corrente_carga_externa", fallback=0.0),
        duty_cycle=params.getfloat("duty_cycle"),
        frequencia_chaveamento=params.getfloat("frequencia_chaveamento"),
    )
    modelos_secao = parser["modelos"]
    analises_secao = parser["analises"]
    plots_secao = parser["plots"]
    simulacao = parser["simulacao"]
    analise_linear = parser["analise_linear"]
    estado = simulacao.get("estado_inicial", "0, 0").split(",")
    if len(estado) != 2:
        raise ValueError("[simulacao] estado_inicial deve conter i_L,v_o.")

    return ConfiguracaoProjeto(
        nome=parser.get("projeto", "nome", fallback=arquivo.stem),
        parametros=planta,
        tensao_saida_alvo=parser.getfloat("sinais", "tensao_saida_alvo"),
        duracao=simulacao.getfloat("duracao"),
        frequencia_amostragem=simulacao.getfloat("frequencia_amostragem"),
        estado_inicial=(float(estado[0]), float(estado[1])),
        saida=Path(parser.get("projeto", "pasta_resultados", fallback="resultados")),
        modelos={nome: modelos_secao.getboolean(nome, fallback=False) for nome in modelos_secao},
        analises={nome: analises_secao.getboolean(nome, fallback=False) for nome in analises_secao},
        plots={nome: plots_secao.getboolean(nome, fallback=False) for nome in plots_secao},
        entrada_frequencial=analise_linear.get("entrada_frequencial", fallback="duty"),
        duty_min=analise_linear.getfloat("duty_min", fallback=planta.duty_cycle - 0.05),
        duty_max=analise_linear.getfloat("duty_max", fallback=planta.duty_cycle + 0.05),
        quantidade_duties=analise_linear.getint("quantidade_duties", fallback=11),
        mostrar_graficos=simulacao.getboolean("mostrar_graficos", fallback=True),
    )
