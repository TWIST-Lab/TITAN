#!/usr/bin/env python3
"""Run a single simulation case described by a TOML configuration."""

from __future__ import annotations

import argparse
import os
import tomllib
import pandas as pd

from sionna.rt import subcarrier_frequencies
from sionna.phy.constants import BOLTZMANN_CONSTANT

from SimulationEnvironment import SimulationEnvironment
from run_simulations import (
    evaluate_env,
    run_random_placement,
    run_bayesian_placement,
    run_bayesian_stochastic_placement,
    run_bayesian_aoi_placement,
    run_leo,
)

import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt

def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def build_environment(cfg: dict) -> SimulationEnvironment:
    env_cfg = cfg.get("environment", {})
    env = SimulationEnvironment(seed=env_cfg.get("seed", 44))

    fidelity_eval = env_cfg.get("fidelity_eval", False)
    if "samples_per_tx" in env_cfg:
        samples_per_tx = env_cfg["samples_per_tx"]
    else:
        samples_per_tx = 10**4 if fidelity_eval else 10**6
    if "eval_samples_per_tx" in env_cfg:
        eval_samples_per_tx = env_cfg["eval_samples_per_tx"]
    else:
        eval_samples_per_tx = 10**6 if fidelity_eval else samples_per_tx

    env.init_solvers(
        samples_per_tx=samples_per_tx,
        max_depth=env_cfg.get("max_depth", 8),
        refraction=env_cfg.get("refraction", False),
        diffuse=env_cfg.get("diffuse", True),
        eval_samples_per_tx=eval_samples_per_tx,
        fidelity_eval=fidelity_eval,
    )

    scenario = env_cfg.get("scenario", "Full Failure")

    if scenario != "Full Failure":
        env.add_tx("dummy", [0, 0, 1000], 23)
    else:
        env.add_base_station("tx1", [608, 73, 150], orientation=[0, 0, 0], color=(0, 0, 1))
        env.add_base_station("tx2", [-140, -300, 110], orientation=[0, 0, 0], color=(0.7, 0, 0.7))
        env.add_base_station("tx3", [-220, 220, 40], orientation=[0, 0, 0], color=(1, 0, 0))

    env.deploy_ues(
        env_cfg.get("ue_count", 50),
        location_error=env_cfg.get("location_error", 0.0),
    )


    if env_cfg.get("low_fidelity", False):
        env.rm_samples_per_tx = env_cfg.get("low_fidelity_samples_per_tx", 1000)
        env.rm_max_depth = env_cfg.get("low_fidelity_max_depth", 8)
    env.rm_refraction = True 

    if scenario != "Full Failure":
        env.scene.remove("dummy")
    else:
        env.scene.remove("tx1")
        env.scene.remove("tx2")
        env.scene.remove("tx3")

    env.txs.clear()
    env.uavs.clear()
    env.base_stations.clear()
    env.bs_positions.clear()
    env.bs_active.clear()

    if scenario == "Full Failure":
        pass  # Add no base stations
    elif scenario == "1 BS Fail":
        env.add_base_station("tx1", [608, 73, 150], orientation=[0, 0, 0], color=(0, 0, 1))
        env.add_base_station("tx2", [-140, -300, 110], orientation=[0, 0, 0], color=(0.7, 0, 0.7))
    elif scenario == "2 BS Fail":
        env.add_base_station("tx3", [-220, 220, 40], orientation=[0, 0, 0], color=(1, 0, 0))
    elif scenario == "No Failure":
        env.add_base_station("tx1", [608, 73, 150], orientation=[0, 0, 0], color=(0, 0, 1))
        env.add_base_station("tx2", [-140, -300, 110], orientation=[0, 0, 0], color=(0.7, 0, 0.7))
        env.add_base_station("tx3", [-220, 220, 40], orientation=[0, 0, 0], color=(1, 0, 0))
    else:
        raise ValueError(f"Unknown scenario '{scenario}'")

    #env.deploy_ues(
        #env_cfg.get("ue_count", 50),
        #location_error=env_cfg.get("location_error", 0.0),
    #)
    uav_count = cfg.get("case", {}).get("uav_count", 1)
    for i in range(uav_count):
        env.add_tx(f"uav_{i+1}", [0, 0, 500], 23)

    return env


def run_case(cfg: dict, debug: bool) -> dict:
    env = build_environment(cfg)

    case_cfg = cfg.get("case", {})
    uav_count = case_cfg.get("uav_count", 1)
    method = case_cfg.get("method", "random").lower()
    use_offset = case_cfg.get("use_ue_offset", False)
    location_error = cfg.get("environment", {}).get("location_error")

    opt_offset = use_offset
    eval_offset = use_offset
    if location_error is not None:
        opt_offset = True
        eval_offset = False

    if method == "random":
        result = run_random_placement(
            env,
            uav_count,
            xy_extent=case_cfg.get("xy_extent", 500),
            z_min=case_cfg.get("z_min", 50),
            z_max=case_cfg.get("z_max", 1000),
            seed=case_cfg.get("seed"),
            use_ue_offset=use_offset,
        )
    elif method == "bayesian_stochastic":
        result = run_bayesian_stochastic_placement(
            env,
            uav_count,
            xy_extent=case_cfg.get("xy_extent", 500),
            z_min=case_cfg.get("z_min", 50),
            z_max=case_cfg.get("z_max", 1000),
            n_trials=case_cfg.get("n_trials", 150),
            use_ue_offset=opt_offset,
            eval_offset=eval_offset,
            )
    elif method == "bayesian":
        result = run_bayesian_placement(
            env,
            uav_count,
            xy_extent=case_cfg.get("xy_extent", 500),
            z_min=case_cfg.get("z_min", 50),
            z_max=case_cfg.get("z_max", 1000),
            n_trials=case_cfg.get("n_trials", 150),
            use_ue_offset=opt_offset,
            eval_offset=eval_offset,
        )
    elif method == "bayesian_aoi":
        result = run_bayesian_aoi_placement(
            env,
            uav_count,
            xy_extent=case_cfg.get("xy_extent", 250),
            z_min=case_cfg.get("z_min", 50),
            z_max=case_cfg.get("z_max", 1000),
            n_trials=case_cfg.get("n_trials", 2500),
            use_ue_offset=opt_offset,
            eval_offset=eval_offset,
        )
    elif method == "leo":
        # LEO evaluation uses a smaller OFDM grid
        env.subcarrier_spacing = 15e3
        env.num_subcarriers = 12 * 25 # 300 * 15000 ~= 5MHz 
        env.no = BOLTZMANN_CONSTANT * env.noise_temp_k * env.subcarrier_spacing
        env.frequencies = subcarrier_frequencies(
            num_subcarriers=env.num_subcarriers,
            subcarrier_spacing=env.subcarrier_spacing,
        )
        env.scene.bandwidth = env.subcarrier_spacing * env.num_subcarriers
        result = run_leo(env, use_ue_offset=use_offset)
    else:
        raise ValueError(f"Unknown method '{method}'")


    if debug:
        env.call_rm_solver()
        env.render_scene()
        env.render_paths()
        plt.show()

    result["location_error"] = location_error
    result["gnb_count"] = len(env.base_stations)
    return result


def save_summary(result: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([
        {
            "placement": result["placement"],
            "uav_count": result["uav_count"],
            "gnb_count": result["gnb_count"],
            "coverage_ratio": result["coverage_ratio"],
            "mean_sinr_db": result["mean_sinr_db"],
            "sum_rate": result["sum_rate"],
            "fairness_index": result["fairness_index"],
            "spectral_efficiency": result["spectral_efficiency"],
            "location_error": result.get("location_error"),
        }
    ])
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single simulation case.")
    parser.add_argument("config", help="TOML configuration file")
    parser.add_argument(
        "--summary",
        default="results/summary.csv",
        help="CSV file to append summary results",
    )
    parser.add_argument("--debug", default=False, help="Show deployment")

    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_case(cfg, args.debug)
    save_summary(result, args.summary)
    


if __name__ == "__main__":
    main()
