"""Mine a Standard-Bangla -> regional-dialect word-substitution lexicon
from BIDWESH parallel sentences (DIA: Dialect-Informed Augmentation).

Uses ONLY sentence pairs whose source is in the BIDWESH *adapt* split, so
the held-out dialect benchmark never leaks into training. Alignment:
difflib opcodes over word sequences; within each 'replace' region, words
are paired by character-level similarity. Pairs must occur >= --min-freq
times and be sufficiently similar (>= --min-ratio) to filter noise.

Output: data/processed/dialect_lexicon.json
    {"per_dialect": {dialect: {std_word: [variants...]}},
     "merged": {std_word: [variants...]}, "stats": {...}}

Usage: python -m src.data.dialect_lexicon [--min-freq 2] [--min-ratio 0.5]
"""
import argparse
import difflib
import re
from collections import Counter, defaultdict

# NOTE: python re's \W matches Bengali combining vowel signs (category Mc),
# so strip an explicit punctuation set instead.
_PUNCT_CHARS = re.escape("।,.!?;:'\"()[]{}…~_+*/\\|-–—“”‘’#@&%^<>=")
_PUNCT_STRIP = re.compile(f"^[{_PUNCT_CHARS}\\s]+|[{_PUNCT_CHARS}\\s]+$")

import pandas as pd

from src.utils.common import (
    DATA_PROCESSED,
    DATA_RAW,
    read_csv_any,
    save_json,
    setup_utf8_stdout,
)
from src.utils.normalize import normalize_bangla

DIALECTS = ("Chittagong", "Noakhali", "Barishal")


def word_pairs(std_words: list[str], dia_words: list[str],
               min_ratio: float):
    """Yield (std_word, dialect_word) candidates from one sentence pair."""
    sm = difflib.SequenceMatcher(a=std_words, b=dia_words, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "replace":
            continue
        for k, sw_raw in enumerate(std_words[i1:i2]):
            sw = _PUNCT_STRIP.sub("", sw_raw)
            if len(sw) < 3:
                continue
            region = [_PUNCT_STRIP.sub("", w) for w in dia_words[j1:j2]]
            region = [w for w in region if len(w) >= 2]
            if not region:
                continue
            # positional hint + best character similarity in the region
            best, best_r = None, 0.0
            for m, dw in enumerate(region):
                r = difflib.SequenceMatcher(a=sw, b=dw).ratio()
                r -= 0.05 * abs(k - m)  # prefer nearby positions
                if r > best_r:
                    best, best_r = dw, r
            if best and best != sw and best_r >= min_ratio:
                yield sw, best


def main() -> None:
    setup_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--min-ratio", type=float, default=0.5)
    parser.add_argument("--max-variants", type=int, default=3,
                        help="variants kept per standard word (by freq)")
    args = parser.parse_args()

    rt = read_csv_any(DATA_RAW / "bidwesh" / "Regional Translated Texts.csv")
    rt["std_norm"] = rt["Standard Bangla"].map(normalize_bangla)

    bidwesh = pd.read_csv(DATA_PROCESSED / "bidwesh_test.csv")
    adapt_sources = set(
        bidwesh.loc[bidwesh["bidwesh_split"] == "adapt", "standard_bangla"])
    rt_adapt = rt[rt["std_norm"].isin(adapt_sources)]
    print(f"parallel sentences restricted to adapt split: "
          f"{len(rt_adapt)}/{len(rt)}")

    counts: dict[str, Counter] = {d.lower(): Counter() for d in DIALECTS}
    for _, row in rt_adapt.iterrows():
        std_words = row["std_norm"].split()
        for d in DIALECTS:
            dia = normalize_bangla(row[d]) if pd.notna(row[d]) else ""
            if not dia:
                continue
            for sw, dw in word_pairs(std_words, dia.split(), args.min_ratio):
                counts[d.lower()][(sw, dw)] += 1

    per_dialect: dict[str, dict] = {}
    merged_counter: Counter = Counter()
    for d, ctr in counts.items():
        lex = defaultdict(list)
        for (sw, dw), n in ctr.items():
            if n >= args.min_freq:
                lex[sw].append((dw, n))
                merged_counter[(sw, dw)] += n
        per_dialect[d] = {
            sw: [w for w, _ in sorted(vs, key=lambda x: -x[1])
                 ][:args.max_variants]
            for sw, vs in lex.items()}
        print(f"  {d}: {sum(ctr.values())} raw pairs -> "
              f"{len(per_dialect[d])} lexicon entries")

    merged = defaultdict(list)
    for (sw, dw), n in merged_counter.items():
        merged[sw].append((dw, n))
    merged_lex = {sw: [w for w, _ in sorted(vs, key=lambda x: -x[1])
                       ][:args.max_variants]
                  for sw, vs in merged.items()}
    print(f"merged lexicon: {len(merged_lex)} standard words")

    top = merged_counter.most_common(30)
    print("top pairs (std -> dialect, freq):")
    for (sw, dw), n in top:
        print(f"  {sw} -> {dw}  ({n})")

    save_json(
        {"per_dialect": per_dialect, "merged": merged_lex,
         "stats": {"adapt_sentences": int(len(rt_adapt)),
                   "min_freq": args.min_freq, "min_ratio": args.min_ratio,
                   "merged_entries": len(merged_lex)},
         "note": ("Mined only from BIDWESH adapt-split parallel sentences; "
                  "held-out test sources untouched.")},
        DATA_PROCESSED / "dialect_lexicon.json",
    )
    print(f"wrote {DATA_PROCESSED / 'dialect_lexicon.json'}")


if __name__ == "__main__":
    main()
