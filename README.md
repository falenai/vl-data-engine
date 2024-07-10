# VL-Data-Engine

> A scalable, configurable pipeline for cleaning and augmenting vision-language pretraining datasets.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)

## The Problem

Large-scale image-caption datasets scraped from the web are noisy. Text is often mismatched, duplicated, spammy, or machine-translated garbage. Training a vision-language model on this raw data wastes compute and degrades quality. This toolkit provides a practical pipeline to fix that.

## What It Does

```
Raw web data → [Quality Filter] → [Deduplication] → [Augmentation] → Clean dataset
```

Three main stages:

1. **Quality filtering** — CLIP-based image-text relevance scoring + text heuristics (spam patterns, length checks, repeated n-grams)
2. **Deduplication** — perceptual hashing (pHash/dHash) for near-duplicate images + SimHash for near-duplicate captions
3. **Caption augmentation** — template-based paraphrasing and optional bilingual (zh↔en) back-translation

## Installation

```bash
git clone https://github.com/falenai/vl-data-engine.git
cd vl-data-engine
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Full pipeline

```bash
python scripts/run_pipeline.py \
    --config configs/default.yaml \
    --input data/raw.jsonl \
    --output output/clean/
```

### Just filter

```bash
python scripts/run_filter.py \
    --input data/raw.jsonl \
    --image-root data/images/ \
    --output data/filtered.jsonl \
    --min-clip-score 0.25
```

### Just deduplicate

```bash
python scripts/run_dedup.py \
    --input data/filtered.jsonl \
    --image-root data/images/ \
    --output data/deduped.jsonl \
    --image-method phash \
    --image-threshold 5
```

## Configuration

Everything is controlled via YAML. Copy `configs/default.yaml` and adjust:

```yaml
filter:
  min_clip_score: 0.22       # CLIP cosine similarity threshold
  min_text_len: 10           # Minimum caption character length
  max_aspect_ratio: 3.0      # Max image aspect ratio
  use_aesthetic_filter: false # Aesthetic score (disabled by default)

run_dedup: true
dedup_image_method: "phash"  # or "dhash"
dedup_image_threshold: 5     # Hamming distance threshold

run_augmentation: false      # Enable caption augmentation
```

Dataset-specific configs are in `configs/`. Currently: `default.yaml`, `cc3m.yaml`.

## Benchmarks

On a sample of CC3M (1M pairs, single A100 GPU):

| Stage | Throughput | Retention |
|-------|-----------|-----------|
| Text filter | ~80k pairs/s | ~82% |
| CLIP filter (ViT-B/32) | ~4k pairs/s | ~68% |
| Deduplication | ~12k pairs/s | ~91% |

Full pipeline end-to-end: ~2.5k pairs/s on a single GPU.

## Dataset Input Format

The pipeline expects JSONL files, one record per line:

```json
{"image": "relative/path/to/image.jpg", "caption": "A dog playing fetch."}
{"image": "another/image.jpg", "caption": "Mountain landscape at sunset.", "score": 0.9}
```

## Roadmap

- [x] Text heuristic filters
- [x] CLIP-based relevance scoring
- [x] Perceptual hash deduplication
- [x] SimHash text deduplication
- [x] Template-based caption augmentation
- [x] Back-translation (zh↔en)
- [ ] WebDataset native output
- [ ] Distributed processing with Ray
- [ ] Aesthetic score predictor integration (LAION aesthetic model)
- [ ] GPU-accelerated pHash

## License

MIT
