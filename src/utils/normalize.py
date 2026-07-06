"""Bangla text normalization.

Two levels:
- ``normalize_bangla``: light normalization used for transformer inputs.
  Keeps punctuation and emoji (informative for hate speech), fixes Unicode.
- ``clean_for_tfidf``: aggressive cleaning on top of ``normalize_bangla``
  for bag-of-words models (TF-IDF): strips punctuation/emoji/digits,
  collapses character elongation, lowercases Latin.
"""
import re
import unicodedata

# Zero-width/invisible characters and the U+FFFD replacement char that
# commonly pollute scraped Bangla text.
_INVISIBLE_RE = re.compile("[​‌‍⁠﻿­�]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_MENTION_RE = re.compile(r"@\w+")

_WS_RE = re.compile(r"\s+")

# Aggressive filter for TF-IDF: keep Bengali block, Latin letters, whitespace.
_TFIDF_KEEP_RE = re.compile(r"[^ঀ-৿a-zA-Z\s]")
_ELONGATION_RE = re.compile(r"(.)\1{2,}")


def normalize_bangla(text: str) -> str:
    """Light normalization for transformer models. Preserves punctuation."""
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE_RE.sub("", text)
    text = _CONTROL_RE.sub(" ", text)
    text = _URL_RE.sub(" URL ", text)
    text = _EMAIL_RE.sub(" EMAIL ", text)
    text = _MENTION_RE.sub(" USER ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def clean_for_tfidf(text: str) -> str:
    """Aggressive cleaning for bag-of-words models."""
    text = normalize_bangla(text)
    text = text.lower()
    # Collapse runs of 3+ identical chars (elongation like "নাাাা", "!!!!") to 2.
    text = _ELONGATION_RE.sub(r"\1\1", text)
    text = _TFIDF_KEEP_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()
