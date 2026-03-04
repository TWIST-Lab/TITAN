#!/usr/bin/env python3
"""Generate multiple TOML configuration files for simulation cases."""

from __future__ import annotations

import argparse
import os
import tomllib


DEFAULT_BASE = "example_case.toml"


def load_base(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def dump_toml(data: dict) -> str:
    lines = []
    for section, values in data.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            if isinstance(value, bool):
                val = "true" if value else "false"
            elif isinstance(value, (int, float)):
                val = str(value)
            else:
                val = f'"{value}"'
            lines.append(f"{key} = {val}")
        lines.append("")
    return "\n".join(lines)


def write_config(cfg: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(dump_toml(cfg))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate case configuration files")
    parser.add_argument("output", help="Directory to place generated configs")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Base TOML file")
    parser.add_argument("--uav-count", type=int, required=True)
    parser.add_argument("--method", choices=["random", "bayesian", "bayesian_stochastic", "bayesian_aoi", "leo"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Seeds for environment and case")
    parser.add_argument("--n-trials", type=int, default=None, help="Number of trials for bayesian method")
    parser.add_argument(
        "--scenario",
        choices=["Full Failure", "1 BS Fail", "2 BS Fail", "No Failure"],
        default=None,
        help="Base station failure scenario",
    )
    parser.add_argument(
        "--location-error",
        type=float,
        default=None,
        help="UE location error radius [m]",
    )
    args = parser.parse_args()

    base_cfg = load_base(args.base)

    for seed in args.seeds:
        cfg = base_cfg.copy()
        env_cfg = cfg.setdefault("environment", {})
        env_cfg["seed"] = seed
        if args.location_error is not None:
            env_cfg["location_error"] = args.location_error
        if args.scenario is not None:
            env_cfg["scenario"] = args.scenario
        case = cfg.setdefault("case", {})
        case.update({
            "method": args.method,
            "uav_count": args.uav_count,
            "seed": seed,
        })
        if args.n_trials is not None and args.method in {"bayesian", "bayesian_stochastic", "bayesian_aoi"}:
            case["n_trials"] = args.n_trials
        filename = f"case_{args.method}_uav{args.uav_count}_seed{seed}"
        if args.scenario is not None:
            scenario_slug = args.scenario.lower().replace(" ", "_")
            filename += f"_scenario_{scenario_slug}"
        if args.location_error is not None:
            filename += f"_error_{args.location_error}"
        filename += ".toml"
        write_config(cfg, os.path.join(args.output, filename))
        print(f"Wrote {filename}")


if __name__ == "__main__":
    main()
