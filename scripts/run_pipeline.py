#!/usr/bin/env python3
"""CLI to trigger pipeline runs."""

from __future__ import annotations

import argparse
import json
import sys

from src.config.pipeline_config import load_pipeline_config
from src.orchestration.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an AutoML pipeline")
    parser.add_argument("config", help="Path to the pipeline YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without running")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    print(f"Loaded pipeline config: {config.name}")

    if args.dry_run:
        print("Dry run — config is valid.")
        return

    print(f"Starting pipeline: {config.name}")
    result = run_pipeline(config)
    print("\n=== Pipeline Complete ===")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
