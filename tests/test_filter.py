"""Tests for quality filter module."""
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quality_filter import FilterConfig, TextQualityFilter, ImageQualityFilter


@pytest.fixture
def config():
    return FilterConfig(
        min_text_len=10,
        max_text_len=200,
        min_word_count=3,
        max_word_count=50,
        use_clip_filter=False,
    )


class TestTextQualityFilter:
    def test_passes_valid_caption(self, config):
        f = TextQualityFilter(config)
        ok, reason = f.check("A dog playing in the park on a sunny afternoon.")
        assert ok

    def test_rejects_too_short(self, config):
        f = TextQualityFilter(config)
        ok, reason = f.check("Hi")
        assert not ok
        assert reason == "too_short"

    def test_rejects_url_spam(self, config):
        f = TextQualityFilter(config)
        ok, reason = f.check("Buy now at www.example.com great deals!")
        assert not ok

    def test_rejects_repeated_ngrams(self, config):
        f = TextQualityFilter(config)
        text = "the cat sat on the mat the cat sat on the mat the cat sat on the mat"
        ok, reason = f.check(text)
        assert not ok
        assert reason == "repeated_ngrams"

    def test_filter_batch(self, config):
        f = TextQualityFilter(config)
        texts = [
            "A beautiful sunset over the mountains.",
            "no",  # too short
            "A dog playing fetch in the park with children.",
        ]
        results = f.filter_batch(texts)
        assert results == [True, False, True]


class TestImageQualityFilter:
    def test_passes_valid_image(self, config):
        f = ImageQualityFilter(config)
        img = torch.rand(3, 224, 224)
        ok, reason = f.check(img)
        assert ok

    def test_rejects_too_small(self, config):
        f = ImageQualityFilter(config)
        img = torch.rand(3, 32, 32)
        ok, reason = f.check(img)
        assert not ok
        assert reason == "image_too_small"

    def test_rejects_bad_aspect_ratio(self, config):
        f = ImageQualityFilter(config)
        img = torch.rand(3, 224, 800)
        ok, reason = f.check(img)
        assert not ok
        assert reason == "bad_aspect_ratio"

# test empty string
