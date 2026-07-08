# Near-duplicate leakage audit

Char-3-gram TF-IDF cosine similarity, max over BD-SHS train+val.

## BIDWESH sources vs BD-SHS train+val
3060 unique sources; exact matches previously found: 28 rows.
- cosine >= 0.80: 193 sources / 518 rows (232 in the held-out test half)
- cosine >= 0.85: 136 sources / 360 rows (157 in the held-out test half)
- cosine >= 0.90: 71 sources / 188 rows (80 in the held-out test half)
- cosine >= 0.95: 27 sources / 62 rows (10 in the held-out test half)

Flag applied at 0.90; evaluation excludes 80 flagged rows from bidwesh_heldout.

## BD-SHS test vs train+val (official split kept)
- cosine >= 0.80: 319 test rows (6.3%)
- cosine >= 0.85: 214 test rows (4.3%)
- cosine >= 0.90: 106 test rows (2.1%)
- cosine >= 0.95: 42 test rows (0.8%)

Flag column `near_dup_train` written at 0.90; official test kept intact for comparability, slice analysis reports both subsets.
