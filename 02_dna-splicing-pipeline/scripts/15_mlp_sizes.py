#!/usr/bin/env python3
"""
Classifier-head size sweep (advisor request, 23 Nov 2026 meeting follow-up).

    python scripts/15_mlp_sizes.py

Trains four MLP heads on the frozen NT-v2-50M centre + previous-token
embeddings (variant center_plus_prev_concat, 1024 = 512 centre + 512 previous):

    1024 -> 128 -> 3
    1024 -> 256 -> 3          (the head of 03_train_head.py)
    1024 -> 512 -> 3
    1024 -> 256 -> 128 -> 3   (one extra hidden layer)

Recipe and data split are those of 03_train_head.py with its defaults: the same
stratified 10% validation split (random_state 0) carved out of train, the
StandardScaler fit on the remaining 90%, ReLU + Dropout(0.2) after every hidden
layer, Adam(lr 1e-3, no weight decay), batch 128, up to 300 epochs, and early
stopping on validation loss (patience 20) with the best-validation-loss weights
restored. Each head is trained with seeds 0, 1 and 2; the seed changes the weight
initialisation and the batch order, never the split. The test split is the
untouched 2000-row test set, scored with 03's own evaluate().

Writes only to results/mlp_sizes/:
    summary.md            the table (mean, SD and range over seeds) and notes
    summary.csv           the same numbers, machine-readable
    runs.csv              one row per head x seed
    runs.json             everything per run, including the epoch-by-epoch history
    confusion_<head>.png  test confusion matrix per head, mean over seeds
Nothing under data/ and no other results file is written.
"""

import copy
import csv
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np

from phase1_common import (EMB_DIR, RESULTS_DIR, RESULTS_JSON, SCRIPTS_DIR, label_names,
                           load_label_info, model_slug, read_json, write_json_atomic)

MODEL = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
VARIANT = "center_plus_prev_concat"
OUT = RESULTS_DIR / "mlp_sizes"
SEEDS = (0, 1, 2)
HEADS = [  # (key, hidden layer widths, display name)
    ("1024-128-3", [128], "1024 → 128 → 3"),
    ("1024-256-3", [256], "1024 → 256 → 3 (current)"),
    ("1024-512-3", [512], "1024 → 512 → 3"),
    ("1024-256-128-3", [256, 128], "1024 → 256 → 128 → 3"),
]
# 03_train_head.py defaults (the values its stored results were made with).
VAL_FRAC, SPLIT_SEED = 0.10, 0
MAX_EPOCHS, PATIENCE, LR, WEIGHT_DECAY, BATCH, DROPOUT = 300, 20, 1e-3, 0.0, 128, 0.2

# Chart tokens (dataviz reference palette): blue sequential ramp 100 -> 700.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"


def load_03():
    """03_train_head.py as a module (its name starts with a digit), for evaluate()."""
    spec = importlib.util.spec_from_file_location("train_head_03", SCRIPTS_DIR / "03_train_head.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_head(in_dim, hidden, n_classes):
    """Linear -> ReLU -> Dropout per hidden layer, then Linear to the classes.
    With hidden=[256] this is exactly 03's nn.Sequential, built in the same order,
    so the same seed gives the same initial weights."""
    from torch import nn

    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(DROPOUT)]
        d = h
    layers.append(nn.Linear(d, n_classes))
    return nn.Sequential(*layers)


def train_head(X_fit, y_fit, X_val, y_val, hidden, n_classes, seed):
    """The training loop of 03_train_head.train_mlp, with a configurable head."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    Xf = torch.tensor(X_fit, dtype=torch.float32)
    yf = torch.tensor(y_fit, dtype=torch.long)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.long)

    model = build_head(Xf.shape[1], hidden, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best = {"val_loss": float("inf"), "epoch": 0, "state": None}
    history = []
    since_best = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xf), generator=gen)
        total = 0.0
        for i in range(0, len(Xf), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xf[b]), yf[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            logits = model(Xv)
            val_loss = float(loss_fn(logits, yv))
            val_acc = float((logits.argmax(1) == yv).float().mean())
        history.append({"epoch": epoch, "train_loss": round(total / len(Xf), 5),
                        "val_loss": round(val_loss, 5), "val_acc": round(val_acc, 4)})
        if val_loss < best["val_loss"] - 1e-4:
            best = {"val_loss": val_loss, "val_acc": val_acc, "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
            since_best = 0
        else:
            since_best += 1
        if since_best >= PATIENCE:
            break

    model.load_state_dict(best["state"])
    model.eval()
    return model, best, history


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def confusion_figure(path, title, mats, names):
    """Mean confusion matrix over seeds. Colour = share of the true class (row);
    text = mean count ± SD over seeds, and that share as a percentage."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    mats = np.asarray(mats, dtype=float)              # (seeds, true, pred)
    mean, sd = mats.mean(0), mats.std(0, ddof=1)
    share = mean / mean.sum(1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2})
    fig, ax = plt.subplots(figsize=(5.6, 4.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    n = len(names)
    for i in range(n):
        for j in range(n):
            dark = share[i, j] > 0.5
            ax.text(j, i - 0.09, f"{mean[i, j]:.0f} ± {sd[i, j]:.0f}", ha="center", va="center",
                    fontsize=11.5, fontweight="bold", color="white" if dark else INK)
            ax.text(j, i + 0.17, f"{100 * share[i, j]:.1f}%", ha="center", va="center",
                    fontsize=9, color="white" if dark else INK_2)
    ax.set_xticks(range(n), names)
    ax.set_yticks(range(n), names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.tick_params(length=0)
    # 2px surface gap between cells.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Share of the true class", color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2, length=0)
    fig.text(0.04, 0.965, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.04, 0.905, f"NT-v2-50M centre + previous token · test set, n = {int(mean.sum())}"
             f" · mean ± SD over {len(mats)} seeds", ha="left", va="top", fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    m03 = load_03()
    emb = EMB_DIR / model_slug(MODEL) / VARIANT
    X_train, y_train, X_test, y_test = (np.load(emb / f"{n}.npy")
                                        for n in ("X_train", "y_train", "X_test", "y_test"))
    info = load_label_info()
    names = label_names(info)
    classes = sorted(names)
    to_idx = {c: i for i, c in enumerate(classes)}
    print(f"{MODEL} [{VARIANT}]: X_train {X_train.shape}, X_test {X_test.shape}")

    # 03's split: stratified 10% validation from train, random_state = 03's default seed.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=VAL_FRAC, stratify=y_train, random_state=SPLIT_SEED)
    scaler = StandardScaler().fit(X_fit)
    Xs_fit, Xs_val = scaler.transform(X_fit), scaler.transform(X_val)
    Xs_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    yi_fit = np.array([to_idx[v] for v in y_fit])
    yi_val = np.array([to_idx[v] for v in y_val])
    print(f"fit {len(X_fit)} / validation {len(X_val)} / test {len(X_test)}")

    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for key, hidden, label in HEADS:
        for seed in SEEDS:
            t0 = time.time()
            model, best, history = train_head(Xs_fit, yi_fit, Xs_val, yi_val, hidden,
                                              len(classes), seed)
            secs = time.time() - t0
            with torch.no_grad():
                y_pred = np.array(classes)[model(Xs_test).argmax(1).numpy()]
            met = m03.evaluate(y_test, y_pred, classes, names)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            runs.append({"head": key, "label": label, "hidden": hidden, "seed": seed,
                         "trainable_params": int(n_params),
                         "best_epoch": best["epoch"], "epochs_run": len(history),
                         "best_val_loss": round(best["val_loss"], 5),
                         "best_val_acc": round(best["val_acc"], 4),
                         "test_accuracy": met["accuracy"], "test_macro_f1": met["macro_f1"],
                         "per_class": met["per_class"],
                         "confusion_matrix": met["confusion_matrix"],
                         "train_seconds": round(secs, 1), "history": history})
            print(f"  {key:<15} seed {seed}  params {n_params:>7,}  epoch {best['epoch']:>3} "
                  f"(ran {len(history)})  val loss {best['val_loss']:.4f}  "
                  f"test acc {met['accuracy']:.4f}  macro-F1 {met['macro_f1']:.4f}  {secs:.1f}s")

    # Sanity check: the 256 head with seed 0 is exactly 03's run; compare with its stored numbers.
    stored = (read_json(RESULTS_JSON, default={}) or {}).get("pooling_variants", {}) \
        .get(MODEL, {}).get(VARIANT, {}).get("mlp")
    mine = next(r for r in runs if r["head"] == "1024-256-3" and r["seed"] == 0)
    if stored:
        same = (stored["accuracy"] == mine["test_accuracy"]
                and stored["macro_f1"] == mine["test_macro_f1"]
                and stored["confusion_matrix"]["matrix"] == mine["confusion_matrix"]["matrix"]
                and stored["training"]["best_epoch"] == mine["best_epoch"])
        check = (f"{'MATCH' if same else 'MISMATCH'}: 1024-256-3 seed 0 gives acc "
                 f"{mine['test_accuracy']}, macro-F1 {mine['test_macro_f1']}, epoch "
                 f"{mine['best_epoch']}; phase1_results.json has acc {stored['accuracy']}, "
                 f"macro-F1 {stored['macro_f1']}, epoch {stored['training']['best_epoch']}"
                 f"{'' if same else ' (or a different confusion matrix)'}.")
    else:
        same, check = None, "phase1_results.json has no stored 03 result for this variant."
    print("\n" + check)

    # ---- summary ----
    summary = []
    for key, hidden, label in HEADS:
        rs = [r for r in runs if r["head"] == key]
        f1, acc = [r["test_macro_f1"] for r in rs], [r["test_accuracy"] for r in rs]
        f1_m, f1_s = mean_sd(f1)
        acc_m, acc_s = mean_sd(acc)
        summary.append({"head": key, "label": label, "trainable_params": rs[0]["trainable_params"],
                        "n_seeds": len(rs),
                        "macro_f1_mean": round(f1_m, 4), "macro_f1_sd": round(f1_s, 4),
                        "macro_f1_min": min(f1), "macro_f1_max": max(f1),
                        "accuracy_mean": round(acc_m, 4), "accuracy_sd": round(acc_s, 4),
                        "accuracy_min": min(acc), "accuracy_max": max(acc),
                        "best_epochs": [r["best_epoch"] for r in rs]})
        confusion_figure(OUT / f"confusion_{key}.png", label,
                         [r["confusion_matrix"]["matrix"] for r in rs],
                         rs[0]["confusion_matrix"]["names"])

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        for s in summary:
            w.writerow({**s, "best_epochs": " ".join(map(str, s["best_epochs"]))})
    run_cols = ["head", "seed", "trainable_params", "best_epoch", "epochs_run", "best_val_loss",
                "best_val_acc", "test_accuracy", "test_macro_f1", "train_seconds"]
    with open(OUT / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=run_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(runs)

    n_test = len(y_test)
    best_acc = max(s["accuracy_mean"] for s in summary)
    se = (best_acc * (1 - best_acc) / n_test) ** 0.5
    lines = [
        "# Classifier-head size sweep",
        "",
        f"NT-v2-50M frozen embeddings, centre + previous token concatenated (`{VARIANT}`, "
        f"1024 = 512 + 512). Test set: {n_test} windows. Made by `scripts/15_mlp_sizes.py`, "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
        "",
        f"Values are mean ± SD over seeds {', '.join(map(str, SEEDS))}, with the range "
        "[min–max] in brackets.",
        "",
        "| Head | Trainable params | Test macro-F1 | Test accuracy |",
        "|---|---:|---|---|",
    ]
    for s in summary:
        lines.append(f"| {s['label']} | {s['trainable_params']:,} | "
                     f"{s['macro_f1_mean']:.4f} ± {s['macro_f1_sd']:.4f} "
                     f"[{s['macro_f1_min']:.4f}–{s['macro_f1_max']:.4f}] | "
                     f"{s['accuracy_mean']:.4f} ± {s['accuracy_sd']:.4f} "
                     f"[{s['accuracy_min']:.4f}–{s['accuracy_max']:.4f}] |")
    lines += [
        "",
        "Epoch chosen on validation loss, per seed: " + "; ".join(
            f"{s['head']} {', '.join(map(str, s['best_epochs']))}" for s in summary) + ".",
        "",
        "## Recipe",
        "",
        "Same as `03_train_head.py` with its defaults: a stratified 10% validation split of the "
        f"training set (random_state {SPLIT_SEED}, the split 03's stored results used), "
        "StandardScaler fit on the other 90%, ReLU + Dropout(0.2) after each hidden layer, "
        f"Adam (lr {LR}, weight decay {WEIGHT_DECAY}), batch {BATCH}, up to {MAX_EPOCHS} epochs, "
        f"early stopping on validation loss with patience {PATIENCE}, and the weights of the "
        "best-validation-loss epoch restored. The seed changes only the initial weights and "
        "the batch order; the train / validation / test split is the same in every run.",
        "",
        "## Checks and caveats",
        "",
        f"- Reproduction of 03: {check}",
        f"- The SD covers training randomness only. The test set is fixed at {n_test} windows; "
        f"at accuracy {best_acc:.2f} its binomial standard error is about {se:.4f}, so "
        "differences of that order could also change on a different test sample.",
        "",
        "## Files",
        "",
        "- `summary.csv`: the table above, machine-readable.",
        "- `runs.csv`: one row per head and seed.",
        "- `runs.json`: every run in full (per-class metrics, confusion matrix, epoch history).",
    ] + [f"- `confusion_{s['head']}.png`: test confusion matrix of {s['label']}, mean ± SD "
         "over seeds." for s in summary]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")

    write_json_atomic(OUT / "runs.json", {
        "model": MODEL, "variant": VARIANT, "seeds": list(SEEDS),
        "recipe": {"val_frac": VAL_FRAC, "split_random_state": SPLIT_SEED,
                   "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "lr": LR,
                   "weight_decay": WEIGHT_DECAY, "batch_size": BATCH, "dropout": DROPOUT,
                   "n_fit": int(len(X_fit)), "n_val": int(len(X_val)), "n_test": int(n_test)},
        "reproduces_03": same, "reproduction_check": check,
        "summary": summary, "runs": runs,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    print("\n" + "\n".join(lines[6:6 + 2 + len(summary)]))
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
