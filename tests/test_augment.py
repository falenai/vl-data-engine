# augmentation tests
import pytest

def test_template_augment_returns_list():
    from src.augmentation import template_augment
    result = template_augment('A dog playing fetch.', n=3)
    assert len(result) == 3

