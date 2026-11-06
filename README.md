# Boost Converter Modeling and Analysis

Python project for simulating and analyzing averaged, switched, and small-signal linear models of a boost converter. The workflow is configured through INI files; `main.py` runs only the requested models and analyses.

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- Matplotlib

Create and activate a virtual environment, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run an example configuration from the repository root:

```powershell
python main.py configuracoes/gerar_analisar_modelo_linear.ini
```

To run the averaged, switched, linear, and matrix implementations together:

```powershell
python main.py configuracoes/gerar_analisar_modelos_atuais.ini
```

## Configuration

INI files in `configuracoes/` define plant parameters, output operating point, simulation time grid, enabled models, analyses, plot options, and output directory. Copy an example to create a new run configuration. Boolean options use `true` or `false`.

The supported model flags are `medio`, `chaveado`, `linear`, and `matricial`. Analysis flags include simulation, state-space matrix export, frequency analysis, structural analysis, and linearization divergence. Discrete-time models are not implemented.

Generated figures and JSON results go to the configured `pasta_resultados`; the default `resultados/` directory is ignored by Git.

## Available analyses

```powershell
# Sweep duty cycle and compare averaged and linearized trajectories
python -m utils.analise_divergencia --duty-min 0.45 --duty-max 0.56 --pontos 12

# Print state-space matrices
python -m utils.gerar_matrizes

# Pole-zero map, Bode response, margins, and root locus
python -m utils.analise_frequencial --entrada duty
python -m utils.analise_frequencial --entrada vin

# Controllability and observability
python -m utils.analise_estrutural
```

## Tests

Run the test suite from the repository root:

```powershell
python -m unittest discover -s tests -v
```

## Project layout

```text
configuracoes/  Example INI run configurations
src/            Converter models, numerical simulation, config loader, pipeline
utils/          Matrix generation and engineering analyses/plots
tests/          Automated unit and integration checks
main.py         INI-driven command-line entry point
```

## Model scope

The state vector is `[i_L, v_o]`. The switched and averaged models include inductor resistance `r_L`, load resistance `R`, and optional external output current `I_ext`. The linear model uses deviation states and input perturbations around its configured operating point.

The nominal example uses `Vin = 12 V`, `R = 28.8 ohm`, `L = 470 uH`, `C = 47 uF`, and `r_L = 0.125 ohm`. Its duty ratio is selected for a 24 V averaged-model operating point.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
