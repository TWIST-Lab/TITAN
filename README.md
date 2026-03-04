# Sagin_RT_Optimization

This repository contains tools for running UAV placement simulations.
All utilities require **Python 3.12**.

## Running a single simulation case

Use `run_case.py` with a TOML configuration file to evaluate one set of parameters. Results are appended to a CSV file.

```bash
python3 run_case.py example_case.toml --summary results/summary.csv
```

The `example_case.toml` file provides an example configuration. Edit the values to change the simulation environment and method. For bayesian methods (`bayesian`, `bayesian_stochastic`, `bayesian_aoi`), final UAV altitude is applied as `terrain_height(x,y) + z_min`, so every UAV remains at least `z_min` meters above local terrain. Setting `location_error` under `[environment]` controls the UE location error radius for digital twin accuracy evaluation. The environment section also accepts a `scenario` option for base station failure analysis. Valid values are `"Full Failure"`, `"1 BS Fail A"`, `"2 BS Fail A"` and `"No Failure"`. Set `uav_count = 0` in the `[case]` section to evaluate a scenario without UAVs. The `[case].method` field supports `"random"`, `"bayesian"`, `"bayesian_stochastic"`, `"bayesian_aoi"` and `"leo"`. `bayesian_aoi` uses AOI-aware loss with paper hyperparameters (`alpha=0.01`, `beta=1.0`, `gamma=0.8`, `d_min=400m`); in failure analysis, failed gNB coordinates are automatically used as AOI centers.

## Plotting results

After running several cases, generate summary graphs using `plot_results.py`:

```bash
python3 plot_results.py results/summary.csv
```

The script produces three separate PNG files for coverage ratio, mean SINR and
sum rate. They are saved in the same `results` directory as the summary CSV.
Pass `--location-tests` to compare different location errors with UAV count on
the x-axis.

## Running all cases from a folder

Generate multiple case files (see below) and execute them with `run_folder.py`.
Each case is run in a fresh Python process to avoid interference between
environments:

```bash
python3 run_folder.py configs --summary results/summary.csv
```

Each configuration result is appended to the chosen summary CSV.

## Generating configuration sweeps

Use `generate_cases.py` to create TOML configs varying the seed, UAV count,
method and optional UE location error radius:

```bash
python3 generate_cases.py configs --uav-count 2 --method random --seeds 1 2 3 --location-error 1.0
```

This writes `case_random_uav2_seed*_error_1.0.toml` files under `configs/`, ready for `run_folder.py` when a location error is specified.
