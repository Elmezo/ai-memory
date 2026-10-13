"""Shared text utilities: UTF-8 BOM handling, normalization, validation."""

from __future__ import annotations

import re
import unicodedata

BOM = "\ufeff"
_WS_RE = re.compile(r"\s+", re.UNICODE)

# Standalone messages that are never worth persisting as long-term memories.
TRIVIAL_PHRASES = frozenset(
    {
        "ok",
        "okay",
        "thanks",
        "thank you",
        "hi",
        "hello",
        "hey",
        "yes",
        "no",
        "تم",
        "تمام",
        "شكرا",
        "شكرًا",
        "مرحبا",
        "اهلا",
        "أهلا",
        "نعم",
        "لا",
    }
)


def strip_bom(text: str) -> str:
    """Remove leading/trailing Unicode BOM characters (safe for Arabic/UTF-8)."""
    if not isinstance(text, str):
        return text
    return text.replace(BOM, "").strip()


def normalize_text(text: str) -> str:
    """Normalize text for deduplication: NFKC, BOM strip, collapse whitespace,
    lowercase (Latin only effect; Arabic unaffected)."""
    text = strip_bom(text)
    text = unicodedata.normalize("NFKC", text)
    text = _WS_RE.sub(" ", text).strip()
    return text.casefold()


def tokenize(text: str) -> list[str]:
    return [t for t in normalize_text(text).split(" ") if t]


def jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity on normalized text."""
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_trivial(content: str, min_chars: int = 3) -> bool:
    """Reject empty, too-short, punctuation-only, or trivially small content."""
    normalized = normalize_text(content)
    if len(normalized) < min_chars:
        return True
    if not any(ch.isalnum() for ch in normalized):
        return True
    return normalized in TRIVIAL_PHRASES
