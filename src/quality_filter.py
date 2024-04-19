"""Quality filtering for image-caption pairs.

Provides both text-based heuristic filters and model-based scoring
(CLIP similarity, aesthetic prediction).
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .utils import clean_text, count_words, has_repeated_ngrams

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Configuration for quality filters."""
    # Text filters
    min_text_len: int = 10
    max_text_len: int = 512
    min_word_count: int = 3
    max_word_count: int = 100
    allow_languages: List[str] = field(default_factory=lambda: ["en", "zh"])
    check_repeated_ngrams: bool = True

    # CLIP-based filter
    use_clip_filter: bool = True
    min_clip_score: float = 0.20
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    clip_batch_size: int = 256

    # Aesthetic filter
    use_aesthetic_filter: bool = False
    min_aesthetic_score: float = 4.5

    # Image filters
    min_image_size: int = 64
    max_aspect_ratio: float = 3.0


class TextQualityFilter:
    """Heuristic text quality checker."""

    # Common spam / boilerplate patterns
    _SPAM_PATTERNS = [
        re.compile(r"www\.\S+\.\S+"),           # URLs
        re.compile(r"\S+@\S+\.\S+"),             # emails
        re.compile(r"(?:buy|sell|click|free|win)\s+now", re.I),
        re.compile(r"[^\x00-\x7F]{20,}"),        # long non-ASCII runs (often OCR noise)
    ]

    def __init__(self, config: FilterConfig):
        self.config = config

    def check(self, text: str) -> Tuple[bool, str]:
        """Return (passes, reason). If passes is False, reason explains why."""
        text = clean_text(text)

        if len(text) < self.config.min_text_len:
            return False, "too_short"
        if len(text) > self.config.max_text_len:
            return False, "too_long"

        wc = count_words(text)
        if wc < self.config.min_word_count:
            return False, "too_few_words"
        if wc > self.config.max_word_count:
            return False, "too_many_words"

        for pat in self._SPAM_PATTERNS:
            if pat.search(text):
                return False, "spam_pattern"

        if self.config.check_repeated_ngrams and has_repeated_ngrams(text):
            return False, "repeated_ngrams"

        return True, "ok"

    def filter_batch(self, texts: List[str]) -> List[bool]:
        return [self.check(t)[0] for t in texts]


class CLIPScoreFilter:
    """Filter based on CLIP image-text cosine similarity."""

    def __init__(self, config: FilterConfig, device: Optional[str] = None):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None
        self._preprocess = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import open_clip
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.config.clip_model,
                pretrained=self.config.clip_pretrained,
                device=self.device,
            )
            self._tokenizer = open_clip.get_tokenizer(self.config.clip_model)
            self._model.eval()
            logger.info(f"Loaded CLIP {self.config.clip_model} on {self.device}")
        except ImportError:
            raise RuntimeError("open_clip_torch is required for CLIP scoring. pip install open-clip-torch")

    @torch.no_grad()
    def score_batch(
        self, images: torch.Tensor, captions: List[str]
    ) -> np.ndarray:
        """Compute CLIP cosine similarity for a batch of image-caption pairs."""
        self._load_model()
        images = images.to(self.device)
        tokens = self._tokenizer(captions).to(self.device)

        image_features = self._model.encode_image(images)
        text_features = self._model.encode_text(tokens)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        scores = (image_features * text_features).sum(dim=-1)
        return scores.cpu().numpy()

    def filter_batch(
        self, images: torch.Tensor, captions: List[str]
    ) -> List[bool]:
        scores = self.score_batch(images, captions)
        return [float(s) >= self.config.min_clip_score for s in scores]


class ImageQualityFilter:
    """Check basic image-level quality constraints."""

    def __init__(self, config: FilterConfig):
        self.config = config

    def check(self, image_tensor: torch.Tensor) -> Tuple[bool, str]:
        """image_tensor: CHW float tensor."""
        c, h, w = image_tensor.shape
        if h < self.config.min_image_size or w < self.config.min_image_size:
            return False, "image_too_small"
        aspect = max(h, w) / min(h, w)
        if aspect > self.config.max_aspect_ratio:
            return False, "bad_aspect_ratio"
        return True, "ok"


class QualityFilterPipeline:
    """Chains multiple quality filters together."""

    def __init__(self, config: FilterConfig, device: Optional[str] = None):
        self.config = config
        self.text_filter = TextQualityFilter(config)
        self.image_filter = ImageQualityFilter(config)
        self.clip_filter = CLIPScoreFilter(config, device) if config.use_clip_filter else None
        self._stats: Dict[str, int] = {
            "total": 0,
            "passed": 0,
            "failed_text": 0,
            "failed_image": 0,
            "failed_clip": 0,
        }

    def filter_batch(
        self,
        images: torch.Tensor,
        captions: List[str],
    ) -> Tuple[torch.Tensor, List[str], List[int]]:
        """Filter a batch, returning kept images, captions, and original indices."""
        bs = len(captions)
        self._stats["total"] += bs

        kept_indices = []
        text_mask = self.text_filter.filter_batch(captions)
        self._stats["failed_text"] += sum(1 for m in text_mask if not m)

        # Image checks
        img_mask = []
        for i, img in enumerate(images):
            ok, _ = self.image_filter.check(img)
            img_mask.append(ok and text_mask[i])
        self._stats["failed_image"] += sum(
            1 for i, m in enumerate(img_mask) if not m and text_mask[i]
        )

        candidate_idx = [i for i, m in enumerate(img_mask) if m]
        if not candidate_idx:
            self._stats["passed"] += 0
            return images[:0], [], []

        if self.clip_filter is not None:
            cand_imgs = images[candidate_idx]
            cand_caps = [captions[i] for i in candidate_idx]
            clip_mask = self.clip_filter.filter_batch(cand_imgs, cand_caps)
            self._stats["failed_clip"] += sum(1 for m in clip_mask if not m)
            kept_local = [j for j, m in enumerate(clip_mask) if m]
            kept_indices = [candidate_idx[j] for j in kept_local]
        else:
            kept_indices = candidate_idx

        self._stats["passed"] += len(kept_indices)

        if not kept_indices:
            return images[:0], [], []

        kept_imgs = images[kept_indices]
        kept_caps = [captions[i] for i in kept_indices]
        return kept_imgs, kept_caps, kept_indices

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def reset_stats(self):
        for k in self._stats:
            self._stats[k] = 0


# TODO: add aesthetic score predictor (LAION aesthetic model)


def filter_from_jsonl(jsonl_path: str, config: FilterConfig, device: str = "cpu") -> int:
    """Convenience function: count passing samples in a JSONL file (no images)."""
    f = TextQualityFilter(config)
    passed = 0
    with open(jsonl_path) as fp:
        for line in fp:
            import json
            try:
                item = json.loads(line)
                ok, _ = f.check(item.get("caption", ""))
                if ok:
                    passed += 1
            except Exception:
                pass
    return passed


class LanguageFilter:
    """Filter samples by detected language.
    
    Requires: langdetect (pip install langdetect)
    """
    
    def __init__(self, allowed: list = None):
        self.allowed = set(allowed or ["en", "zh-cn", "zh-tw"])
    
    def check(self, text: str) -> bool:
        try:
            from langdetect import detect
            lang = detect(text)
            return lang in self.allowed
        except Exception:
            return True  # fail open
# TODO: optimize this
# TODO: optimize this
# TODO: optimize this

# batch processing via multiprocessing.Pool

# vectorized implementation reduces overhead

# refactored
