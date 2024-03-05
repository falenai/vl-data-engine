"""Data loading utilities for vision-language pretraining datasets."""
import io
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

_DEFAULT_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_STD = [0.229, 0.224, 0.225]


def build_image_transform(image_size: int = 224, augment: bool = False):
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(_DEFAULT_MEAN, _DEFAULT_STD),
        ])
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(_DEFAULT_MEAN, _DEFAULT_STD),
    ])


def _decode_image(data: bytes, transform) -> Optional[torch.Tensor]:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return transform(img)
    except Exception as e:
        logger.debug(f"Image decode failed: {e}")
        return None


def _decode_text(data: bytes) -> Optional[str]:
    try:
        text = data.decode("utf-8").strip()
        return text if len(text) >= 5 else None
    except Exception:
        return None


def create_wds_pipeline(
    shards: str,
    image_size: int = 224,
    batch_size: int = 256,
    num_workers: int = 4,
    shuffle_buffer: int = 5000,
    caption_key: str = "txt",
    image_key: str = "jpg",
    augment: bool = False,
) -> wds.WebLoader:
    """Build a WebDataset pipeline for image-caption pairs.

    Args:
        shards: Shard URL pattern (local glob or s3://)
        image_size: Target resolution for images
        batch_size: Mini-batch size
        num_workers: Parallel workers for loading
        shuffle_buffer: Shuffle buffer size; set 0 to disable
        caption_key: Key name for caption inside tar archives
        image_key: Key name for image inside tar archives
        augment: Enable random data augmentation

    Returns:
        WebLoader ready for iteration
    """
    transform = build_image_transform(image_size, augment=augment)

    def process_sample(sample: dict) -> Optional[dict]:
        img = _decode_image(sample.get(image_key, b""), transform)
        txt = _decode_text(sample.get(caption_key, b""))
        if img is None or txt is None:
            return None
        return {"image": img, "caption": txt, "key": sample.get("__key__", "")}

    def collate(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        return {
            "images": torch.stack([b["image"] for b in batch]),
            "captions": [b["caption"] for b in batch],
            "keys": [b["key"] for b in batch],
        }

    dset = wds.WebDataset(shards, handler=wds.warn_and_continue)
    if shuffle_buffer > 0:
        dset = dset.shuffle(shuffle_buffer)
    dset = dset.map(process_sample).select(lambda x: x is not None)

    return wds.WebLoader(
        dset.batched(batch_size, collation_fn=collate),
        num_workers=num_workers,
        batch_size=None,
        pin_memory=torch.cuda.is_available(),
    )


class JSONLDataset(Dataset):
    """Dataset backed by a JSONL file with image paths and captions.

    Expected format per line:
        {"image": "relative/path.jpg", "caption": "...", "score": 0.85}
    """

    def __init__(
        self,
        jsonl_path: str,
        image_root: str = "",
        image_size: int = 224,
        min_score: float = 0.0,
        augment: bool = False,
    ):
        self.image_root = Path(image_root)
        self.transform = build_image_transform(image_size, augment=augment)

        self.samples: List[dict] = []
        skipped = 0
        with open(jsonl_path) as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if float(item.get("score", 1.0)) >= min_score:
                        self.samples.append(item)
                    else:
                        skipped += 1
                except json.JSONDecodeError:
                    skipped += 1

        logger.info(
            f"Loaded {len(self.samples)} samples ({skipped} skipped) from {jsonl_path}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        img_path = self.image_root / item["image"]
        try:
            image = self.transform(Image.open(img_path).convert("RGB"))
        except Exception:
            image = torch.zeros(3, 224, 224)  # sentinel for corrupt images
        return {
            "image": image,
            "caption": item.get("caption", ""),
            "score": float(item.get("score", 1.0)),
        }


def jsonl_iter(path: str) -> Iterator[dict]:
    """Lazy iterator over a JSONL file."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass
# pass
# See also: webdataset docs
