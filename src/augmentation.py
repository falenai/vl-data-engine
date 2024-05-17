"""Caption augmentation for bilingual (zh/en) vision-language datasets.

Provides template-based augmentation, synonym substitution, and
optional back-translation support when Helsinki-NLP models are available.
"""
import logging
import random
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Template-based augmentation ──────────────────────────────────────────────

_EN_TEMPLATES = [
    "{caption}",
    "A photo of {noun_phrase}.",
    "An image showing {caption_lower}.",
    "This picture depicts {noun_phrase}.",
    "{caption} — captured in a photograph.",
]

_FILLER_ADJ = [
    "bright", "colorful", "beautiful", "natural", "detailed",
    "clear", "vivid", "sharp", "striking", "interesting",
]

_POSITION_PHRASES = [
    "in the center", "on the left", "on the right",
    "in the background", "in the foreground", "at the top",
]


def _extract_noun_phrase(text: str) -> str:
    """Very rough noun phrase extraction — good enough for templates."""
    text = text.strip().rstrip(".")
    # Remove leading articles
    text = re.sub(r"^(A|An|The)\s+", "", text, flags=re.I)
    return text.lower() if text else text


def template_augment(caption: str, n: int = 1, seed: Optional[int] = None) -> List[str]:
    """Generate augmented variants of a caption using templates.

    Args:
        caption: Original English caption
        n: Number of variants to generate
        seed: Random seed for reproducibility

    Returns:
        List of augmented captions (may include original)
    """
    rng = random.Random(seed)
    noun_phrase = _extract_noun_phrase(caption)
    results = []

    for _ in range(n):
        tmpl = rng.choice(_EN_TEMPLATES)
        aug = tmpl.format(
            caption=caption,
            caption_lower=caption.lower(),
            noun_phrase=noun_phrase,
        )
        results.append(aug)

    return results


# ── Synonym substitution ─────────────────────────────────────────────────────

_SYNONYM_MAP: Dict[str, List[str]] = {
    "picture": ["photo", "image", "photograph", "shot"],
    "photo": ["picture", "image", "photograph", "snapshot"],
    "image": ["picture", "photo", "photograph"],
    "shows": ["depicts", "displays", "features", "illustrates"],
    "large": ["big", "huge", "sizable", "massive"],
    "small": ["tiny", "little", "miniature", "compact"],
    "beautiful": ["stunning", "gorgeous", "lovely", "attractive"],
    "sitting": ["resting", "perching", "positioned"],
    "standing": ["upright", "erect", "positioned upright"],
    "walking": ["strolling", "striding", "moving"],
    "holding": ["carrying", "grasping", "clutching"],
    "wearing": ["dressed in", "sporting", "donning"],
    "looking": ["gazing", "peering", "glancing"],
    "group": ["collection", "gathering", "cluster"],
    "outdoor": ["outside", "exterior", "open-air"],
    "indoor": ["inside", "interior", "enclosed"],
}


def synonym_substitute(caption: str, p: float = 0.3, seed: Optional[int] = None) -> str:
    """Replace words with synonyms with probability p."""
    rng = random.Random(seed)
    words = caption.split()
    result = []
    for w in words:
        w_lower = w.lower().rstrip(",.;:")
        punct = w[len(w_lower):]
        if w_lower in _SYNONYM_MAP and rng.random() < p:
            replacement = rng.choice(_SYNONYM_MAP[w_lower])
            # Preserve capitalization
            if w[0].isupper():
                replacement = replacement.capitalize()
            result.append(replacement + punct)
        else:
            result.append(w)
    return " ".join(result)


# ── Back-translation via Helsinki-NLP ────────────────────────────────────────


class BackTranslator:
    """Bilingual caption augmentation using en↔zh translation models.

    Requires: transformers (pip install transformers)
    Models: Helsinki-NLP/opus-mt-en-zh, Helsinki-NLP/opus-mt-zh-en
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._en2zh = None
        self._zh2en = None
        self._en2zh_tok = None
        self._zh2en_tok = None

    def _load(self):
        if self._en2zh is not None:
            return
        try:
            from transformers import MarianMTModel, MarianTokenizer
            logger.info("Loading Helsinki-NLP translation models...")
            self._en2zh_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
            self._en2zh = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh").to(self.device)
            self._zh2en_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
            self._zh2en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en").to(self.device)
        except ImportError:
            raise RuntimeError("transformers is required for back-translation")

    def _translate(self, texts: List[str], model, tokenizer) -> List[str]:
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=128, num_beams=4)
        return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    def en_to_zh(self, captions: List[str]) -> List[str]:
        """Translate English captions to Chinese."""
        self._load()
        return self._translate(captions, self._en2zh, self._en2zh_tok)

    def back_translate(self, captions: List[str]) -> List[str]:
        """Round-trip: en → zh → en for paraphrase augmentation."""
        self._load()
        zh = self._translate(captions, self._en2zh, self._en2zh_tok)
        return self._translate(zh, self._zh2en, self._zh2en_tok)


# ── Main augmentation interface ──────────────────────────────────────────────


class CaptionAugmentor:
    """Combined caption augmentation pipeline.

    Applies one or more augmentation strategies to caption batches.
    """

    def __init__(
        self,
        use_templates: bool = True,
        use_synonyms: bool = True,
        use_back_translation: bool = False,
        synonym_p: float = 0.3,
        templates_per_caption: int = 1,
        device: str = "cpu",
        seed: Optional[int] = 42,
    ):
        self.use_templates = use_templates
        self.use_synonyms = use_synonyms
        self.use_back_translation = use_back_translation
        self.synonym_p = synonym_p
        self.templates_per_caption = templates_per_caption
        self.seed = seed
        self._back_translator: Optional[BackTranslator] = None

        if use_back_translation:
            self._back_translator = BackTranslator(device=device)

    def augment(self, caption: str, caption_idx: int = 0) -> List[str]:
        """Return a list of augmented versions of the caption."""
        results = [caption]

        if self.use_templates:
            variants = template_augment(
                caption,
                n=self.templates_per_caption,
                seed=(self.seed or 0) + caption_idx,
            )
            results.extend(v for v in variants if v != caption)

        if self.use_synonyms:
            syn = synonym_substitute(
                caption,
                p=self.synonym_p,
                seed=(self.seed or 0) + caption_idx + 1000,
            )
            if syn != caption:
                results.append(syn)

        return results

    def augment_batch(self, captions: List[str]) -> List[List[str]]:
        augmented = [self.augment(c, i) for i, c in enumerate(captions)]

        if self.use_back_translation and self._back_translator:
            try:
                bt = self._back_translator.back_translate(captions)
                for i, (bt_cap, orig) in enumerate(zip(bt, captions)):
                    if bt_cap and bt_cap != orig:
                        augmented[i].append(bt_cap)
            except Exception as e:
                logger.warning(f"Back-translation failed: {e}")

        return augmented


def get_augmentor_from_config(cfg: dict) -> "CaptionAugmentor":
    """Build CaptionAugmentor from a flat config dict."""
    return CaptionAugmentor(
        use_templates=cfg.get("augmentation_use_templates", True),
        use_synonyms=cfg.get("augmentation_use_synonyms", True),
        use_back_translation=cfg.get("augmentation_use_back_translation", False),
        device=cfg.get("device", "cpu"),
    )

_SYNONYM_MAP['dark'] = ['dim', 'shadowy', 'gloomy']

# wip: back-translation

# batched back-translation fixed

# model caching for back-translation
