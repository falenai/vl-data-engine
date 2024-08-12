#!/usr/bin/env python3
"""CLI: run quality filtering on a JSONL dataset."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quality_filter import FilterConfig, QualityFilterPipeline
from src.data_loader import JSONLDataset
from src.utils import setup_logging

import json
import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Filter image-caption pairs by quality")
    p.add_argument("--input", required=True, help="Input JSONL file")
    p.add_argument("--image-root", default="", help="Root dir for image paths")
    p.add_argument("--output", required=True, help="Output JSONL file")
    p.add_argument("--min-clip-score", type=float, default=0.22)
    p.add_argument("--min-text-len", type=int, default=10)
    p.add_argument("--max-text-len", type=int, default=512)
    p.add_argument("--min-word-count", type=int, default=3)
    p.add_argument("--no-clip", action="store_true", help="Disable CLIP filter")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device

    config = FilterConfig(
        min_text_len=args.min_text_len,
        max_text_len=args.max_text_len,
        min_word_count=args.min_word_count,
        min_clip_score=args.min_clip_score,
        use_clip_filter=not args.no_clip,
        clip_batch_size=args.batch_size,
    )
    filter_pipeline = QualityFilterPipeline(config, device=device)

    dataset = JSONLDataset(args.input, image_root=args.image_root)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    total = 0
    passed = 0
    with open(args.output, "w") as out_f:
        for batch in tqdm(loader, desc="Filtering"):
            images = batch["image"]
            captions = batch["caption"]
            _, kept_caps, kept_idx = filter_pipeline.filter_batch(images, captions)
            total += len(captions)
            passed += len(kept_caps)
            for cap in kept_caps:
                out_f.write(json.dumps({"caption": cap}, ensure_ascii=False) + "\n")

    stats = filter_pipeline.stats()
    logger.info(f"Filtering done: {passed}/{total} kept ({100*passed/(total+1e-6):.1f}%)")
    logger.info(f"Stats: {stats}")


if __name__ == "__main__":
    main()

# dry-run flag
