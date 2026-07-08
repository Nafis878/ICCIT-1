# Datasets

All raw downloads live unchanged in `data/raw/<dataset>/`. Run
`python -m src.data.download` to (re)fetch everything that is publicly
accessible; per-file sizes and SHA-256 checksums are recorded in
`data/raw/download_status.json`.

Content warning: these datasets contain highly offensive, hateful and
obscene Bangla text. They are used here strictly for building and
evaluating hate speech *detection* systems.

---

## 1. BD-SHS (main dataset)

| | |
|---|---|
| Source | Kaggle: <https://www.kaggle.com/datasets/naurosromim/bdshs> |
| Code/paper repo | <https://github.com/naurosromim/hate-speech-dataset-for-Bengali-social-media> (notebooks only — **no data files**, verified) |
| Paper | Romim et al., *BD-SHS: A Benchmark Dataset for Learning to Detect Online Bangla Hate Speech in Different Social Contexts*, LREC 2022. <https://arxiv.org/abs/2206.00372> · <https://aclanthology.org/2022.lrec-1.552/> |
| License | MIT (per the GitHub repository license) |
| Size | 50,281 comments — official splits: train 40,224 / val 5,028 / test 5,029 |
| Files | `train.csv`, `val.csv`, `test.csv` |
| Columns | `sentence` (text) · `hate speech` (Task A: 1 = hate, 0 = not hate) · `target` (Task B, hate rows only: `ind`, `male`, `female`, `group`, `_`-joined combos, NaN for non-hate) · `type` (Task C, hate rows only: `slander`, `gender`, `religion`, `callToViolence`, `_`-joined combos, NaN for non-hate) |
| Download status | ✅ automatic (anonymous Kaggle endpoint `https://www.kaggle.com/api/v1/datasets/download/naurosromim/bdshs`) |

Manual fallback if the anonymous endpoint stops working: log in to Kaggle →
dataset page → *Download* → unzip into `data/raw/bdshs/` so that
`train.csv`, `val.csv`, `test.csv` sit in that folder. (Alternative:
configure `%USERPROFILE%\.kaggle\kaggle.json` and run
`kaggle datasets download -d naurosromim/bdshs -p data/raw/bdshs --unzip`.)

```bibtex
@inproceedings{romim2022bdshs,
  title     = {BD-SHS: A Benchmark Dataset for Learning to Detect Online Bangla Hate Speech in Different Social Contexts},
  author    = {Romim, Nauros and Ahmed, Mosahed and Islam, Md Saiful and Sen Sharma, Arnab and Talukder, Hriteshwar and Amin, Mohammad Ruhul},
  booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference (LREC)},
  pages     = {5153--5162},
  year      = {2022}
}
```

## 2. BIDWESH (real regional-dialect evaluation set)

| | |
|---|---|
| Source | Mendeley Data: <https://data.mendeley.com/datasets/bpkrvf882k/1> (DOI 10.17632/bpkrvf882k.1) |
| Paper | Fayaz et al., *BIDWESH: A Bangla Regional Based Hate Speech Detection Dataset*, 2025. <https://arxiv.org/abs/2507.16183> |
| License | CC BY 4.0 |
| Size | 3,061 source sentences × 3 dialects = 9,183 dialectal instances |
| Files | `BIDWESH Dataset.csv` (dialect texts + labels), `Regional Translated Texts.csv` (row-aligned Standard Bangla source + the 3 dialect translations; alignment verified 100%) |
| Columns | `Chittagong`, `Noakhali`, `Barishal` (dialect texts) · `target`, `type`, `hate speech` (same scheme as BD-SHS) · `Standard Bangla` (source sentence, second file) |
| Download status | ✅ automatic (Mendeley public API; the script falls back to system `curl` because the WAF rejects Python's TLS fingerprint) |
| Role in this project (protocol v2) | Grouped **40/10/50 adapt/dev/test split by source sentence** (`bidwesh_split` column, seed 42): the *adapt* half feeds the DIA lexicon mining and few-shot adaptation experiments; the *test* half (`bidwesh_heldout`, ~4.4K rows after excluding near-duplicates of BD-SHS train/val at char-3-gram cosine ≥ 0.9 — see `reports/leakage_audit.md`) is the held-out dialect benchmark. Sources overlapping BD-SHS train/val are forced into the adapt half. |

```bibtex
@misc{fayaz2025bidwesh,
  title  = {BIDWESH: A Bangla Regional Based Hate Speech Detection Dataset},
  author = {Fayaz, Azizul Hakim and Uddin, MD. Shorif and Bhuiyan, Rayhan Uddin and Sultana, Zakia and Islam, Md. Samiul and Paul, Bidyarthi and Muhammad, Tashreef and Manzoor, Shahriar},
  year   = {2025},
  eprint = {2507.16183},
  archivePrefix = {arXiv}
}
```

## 3. Bengali Hate Speech Dataset (~30K, Romim et al. 2021)

| | |
|---|---|
| Source | Kaggle: <https://www.kaggle.com/datasets/naurosromim/bengali-hate-speech-dataset> |
| Paper | Romim et al., *Hate Speech Detection in the Bengali Language: A Dataset and Its Baseline Evaluation*, IJCACI 2020 / Springer AISC. <https://arxiv.org/abs/2012.09686> |
| License | Not stated in the anonymous download metadata — check the Kaggle page before redistribution |
| Size | 30,000 comments (10,000 hate / 20,000 not hate) |
| Files | `Bengali hate speech .csv` (note the space before `.csv` — kept as-is) |
| Columns | `sentence` · `hate` (1/0) · `category` (comment *domain*: crime, entertainment, sports, religion, politics, celebrity, "Meme, TikTok and others" — **not** a hate-type label) |
| Download status | ✅ automatic (anonymous Kaggle endpoint) |
| Role in this project | Standardized to `data/processed/extra_bengali_hs_30k.csv`; **not merged** into the main BD-SHS experiments by default (kept for optional cross-dataset experiments) |

```bibtex
@inproceedings{romim2021hate,
  title     = {Hate Speech Detection in the Bengali Language: A Dataset and Its Baseline Evaluation},
  author    = {Romim, Nauros and Ahmed, Mosahed and Talukder, Hriteshwar and Islam, Md Saiful},
  booktitle = {Proceedings of International Joint Conference on Advances in Computational Intelligence (IJCACI)},
  year      = {2021},
  publisher = {Springer}
}
```

## 4. Bengali Hate Speech (Karim et al. 2020)

| | |
|---|---|
| Source | Hugging Face: <https://huggingface.co/datasets/rezacsedu/bn_hate_speech> |
| Paper | Karim et al., *Classification Benchmarks for Under-resourced Bengali Language based on Multichannel Convolutional-LSTM Network*, IEEE DSAA 2020. <https://arxiv.org/abs/2004.07807> |
| License | MIT |
| Size | 3,418 texts |
| Files | `bn_hate_speech_train.parquet` |
| Columns | `text` · `label` (5 hate *topics*: Personal, Political, Religious, Geopolitical, Gender abusive — **no non-hate class**, so it cannot be used for the binary task) |
| Download status | ✅ automatic (direct parquet URL) |
| Role in this project | Downloaded + documented for completeness; unused in the binary pipeline |

```bibtex
@inproceedings{karim2020classification,
  title     = {Classification Benchmarks for Under-resourced Bengali Language based on Multichannel Convolutional-LSTM Network},
  author    = {Karim, Md Rezaul and Chakravarthi, Bharathi Raja and McCrae, John P. and Cochez, Michael},
  booktitle = {2020 IEEE 7th International Conference on Data Science and Advanced Analytics (DSAA)},
  year      = {2020}
}
```

## 5. HS-BAN — ❌ no public download

| | |
|---|---|
| Paper | Romim et al., *HS-BAN: A Benchmark Dataset of Social Media Comments for Hate Speech Detection in Bangla*, 2021. <https://arxiv.org/abs/2112.01902> |
| Size (per paper) | ~50,000 labelled comments (40.17% hate) |
| Download status | ❌ **manual required** — the paper states "We will make the dataset available for public use", but no official link exists on arXiv, and the authors' BD-SHS GitHub repo contains no HS-BAN data (verified via the GitHub API file tree on 2026-07-05). |

Manual steps if you want HS-BAN: contact the authors (Shahjalal University
of Science and Technology / Fordham University groups behind BD-SHS), or
watch the authors' GitHub (<https://github.com/naurosromim>) for a release.
If you obtain it, drop the files into `data/raw/hs_ban/` and add an adapter
in `src/data/preprocess.py`.

---

## Synthetic augmentation disclaimer

`data/processed/test_augmented.csv` and `train_augmented.csv` are produced
by `src/data/augment.py` using **synthetic, rule-based noisy/dialect-style
perturbations** (spelling variants, character elongation, punctuation
noise). They are *not* manually verified dialect data and must not be
described as real dialect corpora. The only human-created dialect data in
this project is BIDWESH (dataset 2).
