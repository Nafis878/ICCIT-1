"""Synthetic noisy/dialect-style augmentation for Bangla text.

IMPORTANT: this generates SYNTHETIC perturbations that imitate Bangla
social-media noise and informal spelling. It is rule-based and is NOT
manually verified dialect data — do not present it as a real dialect
corpus. The only real dialect data in this project is BIDWESH.

Operations (1–3 sampled per text, seeded):
- elongation:    repeat a word-final character 2–3x ("নাাা", "!!!")
- punct_noise:   add/duplicate/drop sentence punctuation (! ? । …)
- informal:      common informal/phonetic spelling substitutions
                 (word- and character-level, e.g. হয়েছে→হইছে, ী→ি)
- typo:          adjacent-character swap or single char drop in 1–2 words
- spacing:       drop/double a space
- romanize:      optional (--romanize), crude deterministic Bangla→Latin
                 transliteration of the whole text; experimental.

Outputs:
- data/processed/test_augmented.csv   (1:1 perturbed copy of test.csv)
- data/processed/train_augmented.csv  (train.csv + perturbed copies of a
                                       --frac sample, is_augmented flag)

Usage: python -m src.data.augment [--frac 0.5] [--seed 42] [--romanize]
"""
import argparse
import random
import re

import pandas as pd

from src.utils.common import DATA_PROCESSED, setup_utf8_stdout
from src.utils.normalize import clean_for_tfidf

# --- informal / phonetic variant tables ------------------------------------
# Word-level: standard -> common informal social-media spelling.
WORD_VARIANTS = {
    "কি": "কী", "কী": "কি",
    "না": "নাহ", "তো": "তোহ", "কেন": "ক্যান", "কেমন": "ক্যামন",
    "হয়েছে": "হইছে", "হয়েছি": "হইছি", "হবে": "হইবো", "হয়": "হইয়া",
    "গেছে": "গেসে", "আসছে": "আইতেছে", "করছে": "করতেছে",
    "করে": "কইরা", "বলে": "বইলা", "দিয়ে": "দিয়া", "নিয়ে": "নিয়া",
    "ভালো": "ভালা", "ভাল": "ভালা", "খারাপ": "খ্রাপ",
    "ছেলে": "পোলা", "মেয়ে": "মাইয়া", "মানুষ": "মানুস",
    "এখন": "এহন", "তখন": "তহন", "কথা": "কতা", "সাথে": "সাতে",
    "তোমার": "তুমার", "তোমরা": "তুমরা", "আমাদের": "আমগো", "তাদের": "তাগো",
    "খুব": "খুউব", "আচ্ছা": "আইচ্ছা", "একটা": "একখান", "কিছু": "কিসু",
    "পারে": "পারবো", "লাগে": "লাগতেসে", "বেশি": "বেশী", "বেশী": "বেশি",
}
# Character-level confusions common in careless typing (applied to at most
# a couple of positions so the text stays readable).
CHAR_VARIANTS = [
    ("ী", "ি"), ("ি", "ী"), ("ূ", "ু"), ("ণ", "ন"), ("ষ", "শ"),
    ("শ", "স"), ("ড়", "র"), ("ঢ়", "র"), ("চ্ছ", "চ্স"), ("ক্ষ", "খ"),
]

PUNCT_ADD = ["!", "!!", "?", "??", "...", "।", "!!!"]
_PUNCT_STRIP_RE = re.compile(r"[!?।…]+\s*$")

# Crude Bangla -> Latin ("Banglish") transliteration map; experimental.
ROMAN_MAP = {
    "অ": "o", "আ": "a", "ই": "i", "ঈ": "i", "উ": "u", "ঊ": "u", "ঋ": "ri",
    "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou",
    "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng",
    "চ": "ch", "ছ": "ch", "জ": "j", "ঝ": "jh", "ঞ": "n",
    "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh", "ণ": "n",
    "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n",
    "প": "p", "ফ": "f", "ব": "b", "ভ": "v", "ম": "m",
    "য": "j", "র": "r", "ল": "l", "শ": "sh", "ষ": "sh", "স": "s", "হ": "h",
    "ড়": "r", "ঢ়": "r", "য়": "y", "ৎ": "t", "ং": "ng", "ঃ": "h", "ঁ": "",
    "া": "a", "ি": "i", "ী": "i", "ু": "u", "ূ": "u", "ৃ": "ri",
    "ে": "e", "ৈ": "oi", "ো": "o", "ৌ": "ou", "্": "",
    "০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
    "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9", "।": ".",
}


def op_elongation(text: str, rng: random.Random) -> str:
    words = text.split()
    if not words:
        return text
    idx = rng.randrange(len(words))
    w = words[idx]
    if len(w) >= 2:
        words[idx] = w + w[-1] * rng.randint(1, 3)
    return " ".join(words)


def op_punct_noise(text: str, rng: random.Random) -> str:
    if rng.random() < 0.3 and _PUNCT_STRIP_RE.search(text):
        return _PUNCT_STRIP_RE.sub("", text).strip()
    return text + rng.choice(PUNCT_ADD)


def op_informal(text: str, rng: random.Random) -> str:
    words = text.split()
    replaced = False
    for i, w in enumerate(words):
        if w in WORD_VARIANTS and rng.random() < 0.8:
            words[i] = WORD_VARIANTS[w]
            replaced = True
    text = " ".join(words)
    # Fall back to (or occasionally add) a character-level confusion.
    if not replaced or rng.random() < 0.3:
        candidates = [(a, b) for a, b in CHAR_VARIANTS if a in text]
        if candidates:
            a, b = rng.choice(candidates)
            occurrences = [m.start() for m in re.finditer(re.escape(a), text)]
            pos = rng.choice(occurrences)
            text = text[:pos] + b + text[pos + len(a):]
    return text


def op_typo(text: str, rng: random.Random) -> str:
    words = text.split()
    eligible = [i for i, w in enumerate(words) if len(w) >= 4]
    if not eligible:
        return text
    for i in rng.sample(eligible, min(len(eligible), rng.randint(1, 2))):
        w = list(words[i])
        j = rng.randrange(len(w) - 1)
        if rng.random() < 0.5:
            w[j], w[j + 1] = w[j + 1], w[j]  # swap
        else:
            del w[j]  # drop
        words[i] = "".join(w)
    return " ".join(words)


def op_spacing(text: str, rng: random.Random) -> str:
    spaces = [m.start() for m in re.finditer(" ", text)]
    if not spaces:
        return text
    pos = rng.choice(spaces)
    if rng.random() < 0.5:
        return text[:pos] + text[pos + 1:]  # join two words
    return text[:pos] + "  " + text[pos + 1:]  # double space


def op_romanize(text: str, rng: random.Random) -> str:  # noqa: ARG001
    return "".join(ROMAN_MAP.get(ch, ch) for ch in text)


OPS = {
    "elongation": op_elongation,
    "punct_noise": op_punct_noise,
    "informal": op_informal,
    "typo": op_typo,
    "spacing": op_spacing,
}


def augment_text(text: str, rng: random.Random,
                 romanize: bool = False) -> tuple[str, str]:
    """Apply 1-3 random ops; returns (augmented_text, ops_used)."""
    if romanize and rng.random() < 0.5:
        return op_romanize(text, rng), "romanize"
    n_ops = rng.randint(1, 3)
    # 'informal' is the core dialect-style op — make it likely.
    names = (["informal"] if rng.random() < 0.7 else []) + rng.sample(
        sorted(OPS), k=n_ops)
    names = list(dict.fromkeys(names))[:3]
    for name in names:
        text = OPS[name](text, rng)
    return text, "+".join(names)


def augment_frame(df: pd.DataFrame, seed: int, romanize: bool) -> pd.DataFrame:
    rng = random.Random(seed)
    out = df.copy()
    aug = [augment_text(t, rng, romanize) for t in out["text"]]
    out["text"] = [a[0] for a in aug]
    out["aug_ops"] = [a[1] for a in aug]
    out["text_clean"] = out["text"].map(clean_for_tfidf)
    return out


def main() -> None:
    setup_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frac", type=float, default=0.5,
                        help="fraction of train rows to add as augmented copies")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--romanize", action="store_true",
                        help="also emit crude romanized variants (experimental)")
    args = parser.parse_args()

    test = pd.read_csv(DATA_PROCESSED / "test.csv")
    test_aug = augment_frame(test, args.seed, args.romanize)
    test_aug.to_csv(DATA_PROCESSED / "test_augmented.csv", index=False,
                    encoding="utf-8")
    changed = (test_aug["text"].values != test["text"].values).mean()
    print(f"test_augmented.csv: {len(test_aug)} rows "
          f"({changed:.1%} texts modified)")
    print("  op mix:", test_aug["aug_ops"].value_counts().head(8).to_dict())

    train = pd.read_csv(DATA_PROCESSED / "train.csv")
    sample = train.sample(frac=args.frac, random_state=args.seed)
    sample_aug = augment_frame(sample, args.seed + 1, args.romanize)
    sample_aug["is_augmented"] = True
    train_out = pd.concat([train.assign(is_augmented=False, aug_ops=""),
                           sample_aug], ignore_index=True)
    train_out.to_csv(DATA_PROCESSED / "train_augmented.csv", index=False,
                     encoding="utf-8")
    print(f"train_augmented.csv: {len(train_out)} rows "
          f"({len(sample_aug)} synthetic, frac={args.frac})")
    print("NOTE: synthetic rule-based noise — not verified dialect data.")


if __name__ == "__main__":
    main()
