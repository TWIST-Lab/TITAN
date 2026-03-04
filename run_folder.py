#!/usr/bin/env python3
"""Run all simulation case configurations in a folder."""

from __future__ import annotations

import argparse
import glob
import os

import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all case configs in a directory")
    parser.add_argument("folder", help="Directory containing TOML configuration files")
    parser.add_argument(
        "--summary",
        default="results/summary.csv",
        help="CSV file where results will be appended",
    )
    args = parser.parse_args()

    toml_files = sorted(glob.glob(os.path.join(args.folder, "*.toml")))
    if not toml_files:
        print(f"No TOML files found in {args.folder}")
        return

    for path in toml_files:
        subprocess.run(
            [sys.executable, "run_case.py", path, "--summary", args.summary],
            check=True,
        )
        print(f"Finished {os.path.basename(path)}")


if __name__ == "__main__":
    main()
