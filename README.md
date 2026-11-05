# Simulação do conversor boost

Projeto em Python para comparar o modelo médio e o modelo chaveado de um
conversor boost. Os estados são `x = [i_L, v_o]`.

## Estrutura

- `src/modelos.py`: parâmetros da planta e equações dos modelos médio e
  chaveado.
- `src/simulacao.py`: integração numérica e retorno dos estados para cada
  instante solicitado.
- `main.py`: configuração de um exemplo, execução dos dois modelos e gráfico
  comparativo.

## Instalação e execução

```bash
python -m pip install -r requirements.txt
python main.py
```

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
