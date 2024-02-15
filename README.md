# VL-Data-Engine

A scalable data processing pipeline for vision-language pretraining datasets.

## Overview

This toolkit provides a configurable pipeline for cleaning, filtering, deduplicating, and augmenting large-scale image-caption datasets (CC3M, LAION, WIT, etc.) before vision-language model pretraining.

Key features:
- Quality filtering with CLIP-based relevance scoring and aesthetic assessment
- Near-duplicate removal using perceptual hashing and SimHash
- Bilingual (zh/en) caption augmentation
- Scalable to billion-scale datasets via WebDataset format
- Fully configurable via YAML configs

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```bash
# Run the full pipeline on a shard of data
python scripts/run_pipeline.py \
    --config configs/default.yaml \
    --input /path/to/shards/{000000..000999}.tar \
    --output /path/to/output/

# Filter only (no augmentation)
python scripts/run_filter.py \
    --input data.jsonl \
    --min-clip-score 0.25 \
    --min-text-len 10
```

## License

MIT
