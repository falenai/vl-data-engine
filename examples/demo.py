"""Quick demo: run text quality filter on a small dataset."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.quality_filter import FilterConfig, TextQualityFilter

sample_captions = [
    "A golden retriever playing fetch on a sunny beach.",
    "hi",  # too short
    "Buy now at www.example.com amazing deals on electronics!",  # spam
    "Two children building a sandcastle near the ocean waves.",
    "The quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog",
    "A mountainous landscape with snow-capped peaks reflecting in a crystal clear lake.",
]

config = FilterConfig(min_text_len=10, min_word_count=3, use_clip_filter=False)
f = TextQualityFilter(config)

print("Caption quality check results:")
for cap in sample_captions:
    ok, reason = f.check(cap)
    status = "✓" if ok else f"✗ ({reason})"
    print(f"  [{status}] {cap[:60]}")
