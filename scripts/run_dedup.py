#!/usr/bin/env python3
"""CLI: deduplicate a JSONL dataset using image pHash and text SimHash."""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deduplication import DedupPipeline
from src.utils import setup_logging

from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Deduplicate image-caption JSONL")
    p.add_argument("--input", required=True)
    p.add_argument("--image-root", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--image-method", choices=["phash", "dhash"], default="phash")
    p.add_argument("--image-threshold", type=int, default=5)
    p.add_argument("--text-threshold", type=int, default=3)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    dedup = DedupPipeline(
        image_method=args.image_method,
        image_threshold=args.image_threshold,
        text_threshold=args.text_threshold,
    )

    image_root = Path(args.image_root)
    kept = 0
    total = 0

    with open(args.input) as f_in, open(args.output, "w") as f_out:
        for line in tqdm(f_in, desc="Deduplicating"):
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)

            img_path = image_root / record.get("image", "")
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            caption = record.get("caption", "")
            key = record.get("key", caption[:40])

            if dedup.process(key=key, img=img, caption=caption):
                f_out.write(line + "\n")
                kept += 1

    stats = dedup.stats()
    logger.info(f"Dedup done: {kept}/{total} kept")
    logger.info(f"Stats: {stats}")


if __name__ == "__main__":
    main()
