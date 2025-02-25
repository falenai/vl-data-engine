"""Near-duplicate detection for image-caption datasets.

Supports:
- Exact URL/key deduplication
- Image perceptual hashing (pHash, dHash)
- Text SimHash for near-duplicate captions
"""
import hashlib
import logging
import struct
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ── Exact deduplication ──────────────────────────────────────────────────────


class ExactDeduplicator:
    """Tracks exact hashes to remove duplicates in a single pass."""

    def __init__(self):
        self._seen: Set[str] = set()

    def is_duplicate(self, key: str) -> bool:
        if key in self._seen:
            return True
        self._seen.add(key)
        return False

    def reset(self):
        self._seen.clear()

    def __len__(self):
        return len(self._seen)


# ── Perceptual hashing ───────────────────────────────────────────────────────


def phash(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Compute perceptual hash (pHash) of an image.

    Args:
        img: PIL Image
        hash_size: Size of the DCT matrix (default 8 → 64-bit hash)
        highfreq_factor: Upsampling factor for DCT input

    Returns:
        Integer representation of the hash
    """
    import scipy.fftpack as fft

    img_size = hash_size * highfreq_factor
    img = img.convert("L").resize((img_size, img_size), Image.LANCZOS)
    pixels = np.array(img, dtype=float)

    dct = fft.dct(fft.dct(pixels, axis=0), axis=1)
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low)
    bits = (dct_low > med).flatten()

    # Pack 64 bits into integer
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def dhash(img: Image.Image, hash_size: int = 8) -> int:
    """Compute difference hash (dHash) — faster than pHash."""
    img = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.array(img)
    diff = pixels[:, 1:] > pixels[:, :-1]
    h = 0
    for b in diff.flatten():
        h = (h << 1) | int(b)
    return h


def hamming_distance(h1: int, h2: int, bits: int = 64) -> int:
    xor = h1 ^ h2
    return bin(xor).count("1")


class ImageDeduplicator:
    """Near-duplicate image detection using perceptual hashing.

    Images with hamming distance <= threshold are considered duplicates.
    """

    def __init__(self, method: str = "phash", threshold: int = 5):
        assert method in ("phash", "dhash"), f"Unknown hash method: {method}"
        self.method = method
        self.threshold = threshold
        self._hashes: List[Tuple[int, str]] = []  # (hash, key) pairs

    def _compute_hash(self, img: Image.Image) -> int:
        if self.method == "phash":
            return phash(img)
        return dhash(img)

    def is_near_duplicate(self, img: Image.Image, key: str = "") -> bool:
        """Check if img is a near-duplicate of any stored image."""
        h = self._compute_hash(img)
        for stored_h, stored_key in self._hashes:
            if hamming_distance(h, stored_h) <= self.threshold:
                logger.debug(f"Near-dup: {key} ≈ {stored_key}")
                return True
        self._hashes.append((h, key))
        return False

    def add(self, img: Image.Image, key: str = ""):
        h = self._compute_hash(img)
        self._hashes.append((h, key))

    def reset(self):
        self._hashes.clear()


# ── SimHash for text ─────────────────────────────────────────────────────────


def _shingle(text: str, k: int = 3) -> List[str]:
    """Character k-shingles."""
    tokens = text.lower().split()
    if len(tokens) < k:
        return tokens
    return [" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)]


def simhash(text: str, bits: int = 64) -> int:
    """Compute SimHash fingerprint for text."""
    shingles = _shingle(text, k=3)
    if not shingles:
        return 0

    v = np.zeros(bits, dtype=int)
    for s in shingles:
        h = int(hashlib.md5(s.encode()).hexdigest(), 16) & ((1 << bits) - 1)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1

    fingerprint = 0
    for i in range(bits):
        if v[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def simhash_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


class TextDeduplicator:
    """Near-duplicate text detection using SimHash.

    Pairs with SimHash distance <= threshold are flagged as duplicates.
    """

    def __init__(self, threshold: int = 3, bits: int = 64):
        self.threshold = threshold
        self.bits = bits
        self._hashes: List[Tuple[int, str]] = []

    def is_near_duplicate(self, text: str, key: str = "") -> bool:
        h = simhash(text, self.bits)
        for stored_h, stored_key in self._hashes:
            if simhash_distance(h, stored_h) <= self.threshold:
                return True
        self._hashes.append((h, key))
        return False

    def reset(self):
        self._hashes.clear()


# ── Combined pipeline ────────────────────────────────────────────────────────


class DedupPipeline:
    """Two-stage deduplication: exact hash then near-dup via pHash + SimHash."""

    def __init__(
        self,
        image_method: str = "phash",
        image_threshold: int = 5,
        text_threshold: int = 3,
    ):
        self.exact = ExactDeduplicator()
        self.image_dedup = ImageDeduplicator(method=image_method, threshold=image_threshold)
        self.text_dedup = TextDeduplicator(threshold=text_threshold)
        self._stats = {"total": 0, "exact_dup": 0, "image_dup": 0, "text_dup": 0, "kept": 0}

    def process(self, key: str, img: Image.Image, caption: str) -> bool:
        """Return True if the sample should be kept (not a duplicate)."""
        self._stats["total"] += 1

        if self.exact.is_duplicate(key):
            self._stats["exact_dup"] += 1
            return False

        if self.image_dedup.is_near_duplicate(img, key):
            self._stats["image_dup"] += 1
            return False

        if self.text_dedup.is_near_duplicate(caption, key):
            self._stats["text_dup"] += 1
            return False

        self._stats["kept"] += 1
        return True

    def stats(self) -> dict:
        return dict(self._stats)

    def reset(self):
        self.exact.reset()
        self.image_dedup.reset()
        self.text_dedup.reset()
        for k in self._stats:
            self._stats[k] = 0


def batch_phash(images, hash_size: int = 8) -> list:
    """Compute pHash for a list of PIL images in batch."""
    return [phash(img, hash_size) for img in images]
# reviewed
# reviewed
# reviewed

# URL dedup from metadata field

# handle short text edge case

# tqdm progress bar

# fix reset bug

# parallel hashing

# numpy speedup
