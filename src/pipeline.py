"""Main pipeline orchestrator for VL data processing."""
import json
import logging
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import yaml
from tqdm import tqdm

from .augmentation import CaptionAugmentor
from .data_loader import JSONLDataset, jsonl_iter
from .deduplication import DedupPipeline
from .quality_filter import FilterConfig, QualityFilterPipeline
from .utils import ensure_dir, setup_logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    # I/O
    input_path: str = ""
    output_path: str = "output/"
    image_root: str = ""
    output_format: str = "jsonl"  # "jsonl" or "webdataset"

    # Filtering
    filter: FilterConfig = field(default_factory=FilterConfig)

    # Deduplication
    run_dedup: bool = True
    dedup_image_method: str = "phash"
    dedup_image_threshold: int = 5
    dedup_text_threshold: int = 3

    # Augmentation
    run_augmentation: bool = False
    augmentation_use_templates: bool = True
    augmentation_use_synonyms: bool = True
    augmentation_use_back_translation: bool = False

    # Compute
    device: str = "auto"
    num_workers: int = 4
    batch_size: int = 256

    # Logging
    log_level: str = "INFO"
    log_every: int = 1000

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        filter_cfg = FilterConfig(**raw.pop("filter", {}))
        return cls(filter=filter_cfg, **raw)

    def to_yaml(self, path: str):
        d = asdict(self)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class DataPipeline:
    """End-to-end data processing pipeline.

    Stages (in order):
    1. Load samples from JSONL or WebDataset
    2. Quality filter (text + image + CLIP)
    3. Deduplication
    4. Caption augmentation (optional)
    5. Write output
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = _resolve_device(config.device)
        setup_logging(config.log_level)

        self.quality_filter = QualityFilterPipeline(config.filter, device=self.device)
        self.dedup = DedupPipeline(
            image_method=config.dedup_image_method,
            image_threshold=config.dedup_image_threshold,
            text_threshold=config.dedup_text_threshold,
        ) if config.run_dedup else None
        self.augmentor = CaptionAugmentor(
            use_templates=config.augmentation_use_templates,
            use_synonyms=config.augmentation_use_synonyms,
            use_back_translation=config.augmentation_use_back_translation,
            device=self.device,
        ) if config.run_augmentation else None

        self._output_dir = ensure_dir(config.output_path)
        self._out_file = None
        self._stats = {
            "loaded": 0,
            "filter_passed": 0,
            "dedup_kept": 0,
            "written": 0,
            "augmented": 0,
        }

    def _open_output(self):
        out_path = self._output_dir / "output.jsonl"
        self._out_file = open(out_path, "w")
        logger.info(f"Writing to {out_path}")

    def _write_sample(self, record: dict):
        self._out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._stats["written"] += 1

    def run(self):
        """Execute the full pipeline."""
        self._open_output()
        t0 = time.time()

        dataset = JSONLDataset(
            self.config.input_path,
            image_root=self.config.image_root,
            image_size=224,
            min_score=0.0,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=(self.device == "cuda"),
        )

        for batch_idx, batch in enumerate(tqdm(loader, desc="Processing")):
            images = batch["image"]
            captions = batch["caption"]
            self._stats["loaded"] += len(captions)

            # Stage 1: Quality filter
            kept_imgs, kept_caps, kept_idx = self.quality_filter.filter_batch(images, captions)
            self._stats["filter_passed"] += len(kept_caps)

            # Stage 2: Deduplication (per-sample, in-memory)
            dedup_caps = []
            dedup_imgs = []
            if self.dedup and len(kept_caps) > 0:
                for img, cap in zip(kept_imgs, kept_caps):
                    # Convert tensor back to PIL for image hashing
                    from PIL import Image as PILImage
                    import torchvision.transforms.functional as TF
                    pil = TF.to_pil_image(img.clamp(0, 1))
                    if self.dedup.process(key=cap[:40], img=pil, caption=cap):
                        dedup_imgs.append(img)
                        dedup_caps.append(cap)
                self._stats["dedup_kept"] += len(dedup_caps)
            else:
                dedup_imgs = list(kept_imgs)
                dedup_caps = list(kept_caps)
                self._stats["dedup_kept"] += len(dedup_caps)

            # Stage 3: Augmentation + write
            for cap in dedup_caps:
                if self.augmentor:
                    variants = self.augmentor.augment(cap)
                    for v in variants:
                        self._write_sample({"caption": v})
                    self._stats["augmented"] += len(variants)
                else:
                    self._write_sample({"caption": cap})

            if batch_idx > 0 and batch_idx % (self.config.log_every // self.config.batch_size + 1) == 0:
                self._log_progress()

        self._out_file.close()
        elapsed = time.time() - t0
        self._log_final(elapsed)

        # Save filter stats
        stats_path = self._output_dir / "pipeline_stats.json"
        combined_stats = {
            "pipeline": self._stats,
            "filter": self.quality_filter.stats(),
            "dedup": self.dedup.stats() if self.dedup else {},
            "elapsed_seconds": elapsed,
        }
        with open(stats_path, "w") as f:
            json.dump(combined_stats, f, indent=2)
        logger.info(f"Stats written to {stats_path}")

    def _log_progress(self):
        s = self._stats
        logger.info(
            f"loaded={s['loaded']:,}  filter_passed={s['filter_passed']:,}  "
            f"dedup_kept={s['dedup_kept']:,}  written={s['written']:,}"
        )

    def _log_final(self, elapsed: float):
        s = self._stats
        logger.info("=" * 60)
        logger.info("Pipeline complete:")
        logger.info(f"  Loaded:         {s['loaded']:,}")
        logger.info(f"  Filter passed:  {s['filter_passed']:,}  ({100*s['filter_passed']/(s['loaded']+1e-6):.1f}%)")
        logger.info(f"  Dedup kept:     {s['dedup_kept']:,}")
        logger.info(f"  Written:        {s['written']:,}")
        logger.info(f"  Elapsed:        {elapsed:.1f}s")
