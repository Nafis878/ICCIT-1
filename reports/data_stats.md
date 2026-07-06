# Processed data statistics

- `train.csv` (BD-SHS train): n=40181 | not_hate=20875 (52.0%), hate=19306 (48.0%)
- `val.csv` (BD-SHS val): n=5028 | not_hate=2612 (51.9%), hate=2416 (48.1%)
- `test.csv` (BD-SHS test): n=5028 | not_hate=2612 (51.9%), hate=2416 (48.1%)
- `bidwesh_test.csv`: n=8987 | not_hate=4551 (50.6%), hate=4436 (49.4%)
  - chittagong: n=3054 | not_hate=1545 (50.6%), hate=1509 (49.4%)
  - noakhali: n=3000 | not_hate=1512 (50.4%), hate=1488 (49.6%)
  - barishal: n=2933 | not_hate=1494 (50.9%), hate=1439 (49.1%)
  - rows whose Standard-Bangla source appears in BD-SHS train/val: 28 (0.3%) -> excluded in 'bidwesh-clean' eval
- `extra_bengali_hs_30k.csv`: n=29831 | not_hate=19874 (66.6%), hate=9957 (33.4%) (not used by default)

Label mapping: 0 = not_hate, 1 = hate (`label_mapping.json`).
