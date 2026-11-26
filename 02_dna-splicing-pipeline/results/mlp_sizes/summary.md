# Classifier-head size sweep

NT-v2-50M frozen embeddings, centre + previous token concatenated (`center_plus_prev_concat`, 1024 = 512 + 512). Test set: 2000 windows. Made by `scripts/15_mlp_sizes.py`, 2026-11-26 11:08 UTC.

Values are mean ± SD over seeds 0, 1, 2, with the range [min–max] in brackets.

| Head | Trainable params | Test macro-F1 | Test accuracy |
|---|---:|---|---|
| 1024 → 128 → 3 | 131,587 | 0.9398 ± 0.0010 [0.9388–0.9407] | 0.9400 ± 0.0010 [0.9390–0.9410] |
| 1024 → 256 → 3 (current) | 263,171 | 0.9381 ± 0.0033 [0.9343–0.9403] | 0.9383 ± 0.0033 [0.9345–0.9405] |
| 1024 → 512 → 3 | 526,339 | 0.9403 ± 0.0028 [0.9372–0.9428] | 0.9405 ± 0.0028 [0.9375–0.9430] |
| 1024 → 256 → 128 → 3 | 295,683 | 0.9318 ± 0.0030 [0.9300–0.9352] | 0.9322 ± 0.0029 [0.9305–0.9355] |

Epoch chosen on validation loss, per seed: 1024-128-3 2, 2, 2; 1024-256-3 2, 2, 2; 1024-512-3 2, 2, 2; 1024-256-128-3 1, 2, 1.

## Recipe

Same as `03_train_head.py` with its defaults: a stratified 10% validation split of the training set (random_state 0, the split 03's stored results used), StandardScaler fit on the other 90%, ReLU + Dropout(0.2) after each hidden layer, Adam (lr 0.001, weight decay 0.0), batch 128, up to 300 epochs, early stopping on validation loss with patience 20, and the weights of the best-validation-loss epoch restored. The seed changes only the initial weights and the batch order; the train / validation / test split is the same in every run.

## Checks and caveats

- Reproduction of 03: MATCH: 1024-256-3 seed 0 gives acc 0.9405, macro-F1 0.9403, epoch 2; phase1_results.json has acc 0.9405, macro-F1 0.9403, epoch 2.
- The SD covers training randomness only. The test set is fixed at 2000 windows; at accuracy 0.94 its binomial standard error is about 0.0053, so differences of that order could also change on a different test sample.

## Files

- `summary.csv`: the table above, machine-readable.
- `runs.csv`: one row per head and seed.
- `runs.json`: every run in full (per-class metrics, confusion matrix, epoch history).
- `confusion_1024-128-3.png`: test confusion matrix of 1024 → 128 → 3, mean ± SD over seeds.
- `confusion_1024-256-3.png`: test confusion matrix of 1024 → 256 → 3 (current), mean ± SD over seeds.
- `confusion_1024-512-3.png`: test confusion matrix of 1024 → 512 → 3, mean ± SD over seeds.
- `confusion_1024-256-128-3.png`: test confusion matrix of 1024 → 256 → 128 → 3, mean ± SD over seeds.
