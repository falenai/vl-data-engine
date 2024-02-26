"""Shared utility functions for the data pipeline."""
import hashlib
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

import ftfy
import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper()),
    )


def clean_text(text: str) -> str:
    """Fix encoding issues and normalize unicode."""
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def batch_iter(iterable, batch_size: int):
    """Yield successive batches from iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def count_words(text: str) -> int:
    """Word count, works for CJK and Latin scripts."""
    # CJK characters count as individual words
    cjk = re.findall(r"[一-鿿㐀-䶿]", text)
    latin = re.findall(r"\b[a-zA-Z]+\b", text)
    return len(cjk) + len(latin)


def is_mostly_ascii(text: str, threshold: float = 0.8) -> bool:
    if not text:
        return True
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / len(text) >= threshold


def has_repeated_ngrams(text: str, n: int = 4, max_repeat: int = 3) -> bool:
    """Check if text contains suspiciously repeated n-grams (spam indicator)."""
    words = text.lower().split()
    if len(words) < n * max_repeat:
        return False
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    from collections import Counter
    counts = Counter(ngrams)
    return any(v >= max_repeat for v in counts.values())


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_url_blacklist(path: Optional[str]) -> set:
    if path is None or not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def chunk_list(lst, chunk_size: int):
    """Split list into chunks of at most chunk_size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]

