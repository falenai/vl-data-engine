#!/usr/bin/env python3
"""CLI: run the full VL data processing pipeline."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import DataPipeline, PipelineConfig


def parse_args():
    p = argparse.ArgumentParser(description="Run the VL data pipeline")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--input", help="Override input_path from config")
    p.add_argument("--output", help="Override output_path from config")
    return p.parse_args()


def main():
    args = parse_args()
    config = PipelineConfig.from_yaml(args.config)

    if args.input:
        config.input_path = args.input
    if args.output:
        config.output_path = args.output

    pipeline = DataPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
