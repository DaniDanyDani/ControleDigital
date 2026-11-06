# Simulação do conversor boost

Projeto em Python para comparar os modelos médio, chaveado e linear de um
conversor boost. Os estados absolutos são `x = [i_L, v_o]`; no modelo linear,
os estados representam desvios em relação ao ponto de operação.

## Estrutura

- `src/modelos.py`: parâmetros da planta e equações dos modelos médio e
  chaveado, além da linearização de pequenos sinais.
- `src/simulacao.py`: integração numérica e retorno dos estados para cada
  instante solicitado.
- `main.py`: configuração de um exemplo, execução dos dois modelos e gráfico
  comparativo.
- `tests/`: testes automatizados dos modelos e da simulação.

## Instalação e execução

```bash
python -m pip install -r requirements.txt
python main.py
```

Para executar todos os testes:

```bash
python -m unittest discover -s tests -v
```

## Divergência do modelo linearizado

O script abaixo varre duty cycles ao redor do ponto de linearização nominal,
compara as trajetórias do modelo médio e do modelo linear, informa os valores
finais e o tempo de acomodação, e salva um gráfico de NRMSE e erro final:

```bash
python -m utils.analise_divergencia --duty-min 0.45 --duty-max 0.56 --pontos 12
```

O ponto de linearização permanece fixo no benchmark. O gráfico é salvo por
padrão em `resultados/divergencia_modelo_linear.png`; essa pasta é ignorada
pelo Git. Use `--saida` para escolher outro caminho; `--duracao` e
`--fs-amostragem` ajustam a janela e a resolução temporal da simulação.

## Uso da API

```python
import numpy as np

from src import ModeloBoostChaveado, ParametrosBoost, simular

parametros = ParametrosBoost(
    vin=24.0,
    indutancia=1e-3,
    capacitancia=470e-6,
    resistencia_carga=20.0,
    resistencia_indutor=0.15,
    corrente_carga_externa=0.0,
    duty_cycle=0.5,
    frequencia_chaveamento=20e3,
)
tempos = np.linspace(0.0, 30e-3, 30_001)
estados = simular(ModeloBoostChaveado(), tempos, parametros)
```

`estados[:, 0]` contém `i_L` em ampères e `estados[:, 1]` contém `v_o` em
volts. O estado inicial padrão é `[0 A, 0 V]` e pode ser alterado pelo
argumento `estado_inicial`.

## Equações implementadas

### Benchmark nominal

Para `Vin = 12 V`, `R = 28.8 ohm`, `rL = 0.125 ohm` e `I_ext = 0`, o
equilíbrio do modelo médio com `v_o = 24 V` exige `D = 0.50883673`
(aproximadamente `50.8837%`). Esse valor considera a queda na resistência
série do indutor; o benchmark está centralizado em `src/benchmarks.py` e é
usado pelo `main.py`.

O modelo chaveado usa `I_ext` como uma corrente de carga adicional em paralelo
com o resistor `R`:

- Chave ligada: `di_L/dt = (Vin - r_L i_L)/L` e
  `dv_o/dt = (-v_o/R - I_ext)/C`.
- Chave desligada: `di_L/dt = (Vin - v_o - r_L i_L)/L` e
  `dv_o/dt = (i_L - v_o/R - I_ext)/C`.
- Modelo médio: `di_L/dt = Vin/L - (1-D)v_o/L - r_L i_L/L` e
  `dv_o/dt = (1-D)i_L/C - v_o/(RC) - I_ext/C`.

O PWM começa ligado em `t = 0`, permanece ligado durante `D / f_sw` e repete
com frequência `f_sw`. A integração do modelo chaveado é dividida nas bordas
de cada período e nas transições entre os estados ligado e desligado.
