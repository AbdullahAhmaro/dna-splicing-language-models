#!/usr/bin/env python3
"""
Step 18: learning curve on the genome dataset. Does more training data help the head?

    python scripts/18_learning_curve.py               # embed (resumable), train, report
    python scripts/18_learning_curve.py --embed-only  # just the embedding

Embedding, once and resumable. NT-v2-50M on every window of the genome test file
(18,653 windows, chr20/21) and of the 120,000-window default train file (16_genome_dataset.py),
with 10_scan_gene.window_features: the last-layer vectors of the token covering base 300 and of
the token before it, float16, written in chunks of 2,048 windows with a progress file, so an
interrupted run carries on where it stopped. Stored in data/emb/<model>/genome/
(phase1_common.genome_emb_dir); splice_sites_all's embeddings are not touched. Head input:
pair_features(C, P, "center_plus_prev_concat"), as in every earlier run of this variant.

Training sets. Nested, class-balanced subsets of the train file: 8,000, 15,000, 30,000, 60,000
and 120,000 windows. Per seed, every class gets one seeded order (the none class interleaves its
four groups, decoy GT, decoy AG, exon interior, intron interior, in proportion), and size n takes
the first n/3 of each class. So each set lies inside the next, and every set keeps the class
balance and the none class's thirds. 120,000 is the whole file.

Head and validation: 03_train_head.py run with --seed s: a stratified 10% validation split
(random_state s), StandardScaler fit on the other 90%, 1024 -> 256 -> 3, Adam 1e-3, batch 128,
early stopping on validation loss with patience 20 (03's train_mlp, unchanged). Seeds 0, 1, 2:
the seed changes the subset, the validation split, the initial weights and the batch order.
Test: every head on the same 18,653 test windows, scored with 03's evaluate(). Decoy recall =
the share of the test set's GT (or AG) decoy windows that the head calls none.

Outputs:
  data/emb/<model>/genome/{train,test}.features.npy   (n, 2, 512) float16, rows as the parquet
  data/emb/<model>/genome/heads/head_n<size>_s<seed>.pt   every head, in 03's format
  results/learning_curve/summary.md, summary.csv, runs.csv, runs.json,
  results/learning_curve/learning_curve.png, confusion_8000.png, confusion_120000.png
"""

import argparse
import csv
import hashlib
import importlib
import sys
import time
from datetime import datetime, timezone

import numpy as np

from phase1_common import (GENOME_DEFAULT_DIR, GENOME_FULL_DIR, RESULTS_DIR, SEED, genome_emb_dir,
                           label_names, load_genome_split, load_label_info, pair_features,
                           read_json, write_json_atomic)

th = importlib.import_module("03_train_head")      # train_mlp, evaluate
sg = importlib.import_module("10_scan_gene")       # window_features (resumable embedding)
ms = importlib.import_module("15_mlp_sizes")       # confusion_figure, chart tokens

MODEL = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
VARIANT = "center_plus_prev_concat"
SIZES = (8_000, 15_000, 30_000, 60_000, 120_000)
SEEDS = (0, 1, 2)
CHUNK = 2048
EMB = genome_emb_dir(MODEL)
OUT = RESULTS_DIR / "learning_curve"
DECOYS = {"GT decoy": ("decoy", "donor-like (GT after)"),
          "AG decoy": ("decoy", "acceptor-like (AG before)")}
GROUPS = [("decoy", "donor-like (GT after)"), ("decoy", "acceptor-like (AG before)"),
          ("exon_interior", ""), ("intron_interior", "")]


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- embedding
def embed(split, loaded):
    """[C, P] float16 vectors of every window of one split, via sg.window_features. The windows
    are laid end to end, so the window centred on base 600*i + 300 is exactly row i."""
    path = GENOME_DEFAULT_DIR / f"{split}.parquet"
    seqs = load_genome_split(split, ["sequence"])["sequence"]
    stem = EMB / split
    fpath = stem.with_name(f"{split}.features.npy")
    ppath = stem.with_name(f"{split}.positions.npy")
    prog = stem.with_name(f"{split}.progress.json")
    mpath = stem.with_name(f"{split}.meta.json")
    sha = sha1(path)
    meta = read_json(mpath, default=None)
    if meta and meta["parquet_sha1"] != sha:            # the parquet was rebuilt: start over
        print(f"{split}: {path.name} changed since it was embedded; re-embedding")
        for p in (fpath, ppath, prog):
            p.unlink(missing_ok=True)
    positions = np.arange(len(seqs), dtype=np.int64) * sg.WINDOW + sg.CENTER
    if fpath.exists() and ppath.exists() and not prog.exists() \
            and np.array_equal(np.load(ppath), positions):
        print(f"{split}: {len(seqs):,} windows already embedded")
        return np.load(fpath, mmap_mode="r")
    EMB.mkdir(parents=True, exist_ok=True)
    write_json_atomic(mpath, {
        "parquet": str(path.relative_to(GENOME_DEFAULT_DIR.parent.parent.parent)),
        "parquet_sha1": sha, "n": len(seqs), "model": MODEL,
        "content": "features[i] = [C, P]: last-layer vector of the token covering base 300 "
                   "and of the token before it (10_scan_gene.window_features), float16, row i "
                   "= row i of the parquet",
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    if "model" not in loaded:
        import torch
        from transformers import AutoTokenizer

        from try_dna_models import load_model
        loaded["tok"] = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        loaded["model"], _ = load_model(MODEL)
        loaded["model"].eval()
        loaded["torch"] = torch
    print(f"{split}: embedding {len(seqs):,} windows (checkpoint every {CHUNK:,})", flush=True)
    t0 = time.time()
    feats, _ = sg.window_features("".join(seqs), positions, loaded["tok"], loaded["model"],
                                  loaded["torch"], stem, batch=32, chunk=CHUNK)
    meta = read_json(mpath)
    meta.update(finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                seconds_this_session=round(time.time() - t0, 1))
    write_json_atomic(mpath, meta)
    return feats


def features(F):
    F = np.asarray(F, dtype=np.float32)
    return pair_features(F[:, 0], F[:, 1], VARIANT)


# ----------------------------------------------------------------------------- subsets
def nested_order(labels, strata, seed):
    """Per class, a seeded order of its rows; within a class the strata (none: its four
    groups) are interleaved in proportion, so every prefix keeps their shares."""
    rng = np.random.default_rng([SEED, 18, seed])
    order = {}
    for c in sorted(set(labels.tolist())):
        rows = np.flatnonzero(labels == c)
        keys = np.empty(len(rows))
        for s in sorted(set(strata[rows].tolist())):
            m = np.flatnonzero(strata[rows] == s)
            rank = np.empty(len(m))
            rank[rng.permutation(len(m))] = np.arange(len(m))
            keys[m] = (rank + 0.5) / len(m)
        order[c] = rows[np.lexsort((rng.random(len(rows)), keys))]
    return order


def subset(order, n):
    classes = sorted(order)
    per = [n // len(classes) + (i < n % len(classes)) for i in range(len(classes))]
    return np.sort(np.concatenate([order[c][:k] for c, k in zip(classes, per)]))


# ----------------------------------------------------------------------------- report
def chart(path, summary, runs, n_test):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([s["n_train"] for s in summary], float)
    m = np.array([s["macro_f1_mean"] for s in summary])
    sd = np.array([s["macro_f1_sd"] for s in summary])
    line, light = ms.BLUE_RAMP[4], ms.BLUE_RAMP[1]
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": ms.INK,
                         "axes.labelcolor": ms.INK_2, "xtick.color": ms.INK_2,
                         "ytick.color": ms.INK_2})
    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=ms.SURFACE)
    ax.set_facecolor(ms.SURFACE)
    ax.fill_between(x, m - sd, m + sd, color=light, alpha=0.55, linewidth=0, zorder=1)
    for r in runs:
        ax.plot(r["n_train"], r["test_macro_f1"], "o", ms=4, color=ms.BLUE_RAMP[2],
                alpha=0.9, zorder=2, markeredgewidth=0)
    ax.plot(x, m, "-", color=line, lw=2, zorder=3)
    ax.plot(x, m, "o", ms=8, color=line, markeredgecolor=ms.SURFACE, markeredgewidth=2, zorder=4)
    for i in (0, len(x) - 1):
        ax.annotate(f"{m[i]:.3f}", (x[i], m[i]), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, fontweight="bold", color=ms.INK)
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{v / 1000:g}k" for v in x])
    ax.minorticks_off()
    lo, hi = min(r["test_macro_f1"] for r in runs), max(r["test_macro_f1"] for r in runs)
    pad = max(0.004, (hi - lo) * 0.25)
    ax.set_ylim(lo - pad, hi + 2 * pad)
    ax.set_xlabel("Training windows (class-balanced, nested; log scale)")
    ax.set_ylabel("Test macro-F1")
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.03, 0.965, "Test macro-F1 vs training-set size", ha="left", va="top",
             fontsize=13, fontweight="bold", color=ms.INK)
    fig.text(0.03, 0.905, f"NT-v2-50M centre + previous token · 03 head · genome test "
             f"(chr20/21), n = {n_test:,}\nLine = mean, band = ± SD, small dots = single seeds "
             f"({len(SEEDS)} per size)", ha="left", va="top", fontsize=8.5, color=ms.INK_2,
             linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, dpi=200, facecolor=ms.SURFACE)
    plt.close(fig)


def fmt(s, key, d=3):
    return (f"{s[key + '_mean']:.{d}f} ± {s[key + '_sd']:.{d}f} "
            f"[{s[key + '_min']:.{d}f}–{s[key + '_max']:.{d}f}]")


def main():
    ap = argparse.ArgumentParser(description="Learning curve on the genome dataset.")
    ap.add_argument("--embed-only", action="store_true")
    args = ap.parse_args()

    loaded = {}
    F_test = embed("test", loaded)
    F_train = embed("train", loaded)
    if args.embed_only:
        return 0
    loaded.clear()

    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    info = load_label_info()
    names = label_names(info)
    classes = sorted(names)
    to_idx = {c: i for i, c in enumerate(classes)}
    none = info["none_label"]

    tr = load_genome_split("train", ["name", "label"])
    te = load_genome_split("test", ["name", "label"])
    meta = {sp: load_genome_split(sp, ["name", "group", "subtype"], meta=True)
            for sp in ("train", "test")}
    stratum = {sp: dict(zip(meta[sp]["name"], zip(meta[sp]["group"], meta[sp]["subtype"])))
               for sp in meta}
    y_tr, y_te = np.array(tr["label"]), np.array(te["label"])
    s_tr = np.array([GROUPS.index(stratum["train"][n]) if y == none else -1
                     for n, y in zip(tr["name"], tr["label"])])
    g_te = [stratum["test"][n] for n in te["name"]]
    X_tr, X_te = features(F_train), features(F_test)
    print(f"train {X_tr.shape}, test {X_te.shape}")

    # consistency with 17, which embedded 8,000 + 2,000 of the same rows separately (float32)
    s17 = EMB.parent / "genome_sample"
    check17 = None
    if (s17 / "C_test.npy").exists():
        i17 = np.load(s17 / "idx_test.npy")
        a = features(np.stack([np.load(s17 / "C_test.npy"), np.load(s17 / "P_test.npy")], 1))
        d = np.abs(a - X_te[i17])
        check17 = {"rows": int(len(i17)), "max_abs_diff": float(d.max()),
                   "mean_abs_diff": float(d.mean()),
                   "mean_abs_value": float(np.abs(X_te[i17]).mean())}
        print(f"vs 17's separate embedding of {len(i17):,} test rows: max |diff| "
              f"{d.max():.4f}, mean |diff| {d.mean():.5f}")

    heads_dir = EMB / "heads"
    heads_dir.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in SEEDS:
        order = nested_order(y_tr, s_tr, seed)
        prev = None
        for n in SIZES:
            idx = subset(order, n)
            assert prev is None or np.isin(prev, idx).all(), "subsets are not nested"
            prev = idx
            X, y = X_tr[idx], y_tr[idx]
            a = argparse.Namespace(lr=1e-3, weight_decay=0.0, batch_size=128, max_epochs=300,
                                   patience=20, seed=seed, val_frac=0.10)
            X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=a.val_frac, stratify=y,
                                                          random_state=a.seed)
            scaler = StandardScaler().fit(X_fit)
            t0 = time.time()
            mlp, best, hist = th.train_mlp(
                scaler.transform(X_fit), np.array([to_idx[v] for v in y_fit]),
                scaler.transform(X_val), np.array([to_idx[v] for v in y_val]), len(classes), a)
            secs = time.time() - t0
            with torch.no_grad():
                lg = mlp(torch.tensor(scaler.transform(X_te), dtype=torch.float32))
            pred = np.array(classes)[lg.argmax(1).numpy()]
            met = th.evaluate(y_te, pred, classes, names)
            grp = {}
            for g in GROUPS:
                mk = np.array([x == g for x in g_te])
                grp[f"{g[0]} {g[1]}".strip()] = {"n": int(mk.sum()),
                                                 "called_none": round(float((pred[mk] == none).mean()), 4)}
            dec = {k: grp[f"{g[0]} {g[1]}"]["called_none"] for k, g in DECOYS.items()}
            torch.save({"state_dict": mlp.state_dict(), "in_dim": int(X.shape[1]), "hidden": 256,
                        "dropout": 0.2, "classes": [int(c) for c in classes],
                        "class_names": [names[c] for c in classes],
                        "scaler_mean": torch.tensor(scaler.mean_, dtype=torch.float32),
                        "scaler_scale": torch.tensor(scaler.scale_, dtype=torch.float32),
                        "model": MODEL, "variant": VARIANT, "train": "genome default train",
                        "n_train": n, "seed": seed}, heads_dir / f"head_n{n}_s{seed}.pt")
            runs.append({"n_train": n, "seed": seed, "n_fit": int(len(X_fit)),
                         "n_val": int(len(X_val)),
                         "train_class_counts": {names[c]: int((y == c).sum()) for c in classes},
                         "best_epoch": best["epoch"], "epochs_run": len(hist),
                         "best_val_loss": round(best["val_loss"], 5),
                         "best_val_acc": round(best["val_acc"], 4),
                         "test_accuracy": met["accuracy"], "test_macro_f1": met["macro_f1"],
                         "gt_decoy_recall": dec["GT decoy"], "ag_decoy_recall": dec["AG decoy"],
                         "negatives_called_none": grp, "per_class": met["per_class"],
                         "confusion_matrix": met["confusion_matrix"],
                         "train_seconds": round(secs, 1), "history": hist})
            print(f"  n {n:>7,} seed {seed}  epoch {best['epoch']:>3} (ran {len(hist):>3})  "
                  f"test acc {met['accuracy']:.4f}  macro-F1 {met['macro_f1']:.4f}  "
                  f"GT decoy {dec['GT decoy']:.3f}  AG decoy {dec['AG decoy']:.3f}  {secs:.0f}s",
                  flush=True)

    summary = []
    for n in SIZES:
        rs = [r for r in runs if r["n_train"] == n]
        s = {"n_train": n, "n_seeds": len(rs), "best_epochs": [r["best_epoch"] for r in rs]}
        for key, col in (("macro_f1", "test_macro_f1"), ("accuracy", "test_accuracy"),
                         ("gt_decoy_recall", "gt_decoy_recall"),
                         ("ag_decoy_recall", "ag_decoy_recall")):
            v = np.array([r[col] for r in rs], float)
            s.update({f"{key}_mean": round(float(v.mean()), 4),
                      f"{key}_sd": round(float(v.std(ddof=1)), 4),
                      f"{key}_min": float(v.min()), f"{key}_max": float(v.max())})
        summary.append(s)
    n_test = len(y_te)
    chart(OUT / "learning_curve.png", summary, runs, n_test)
    for n in (SIZES[0], SIZES[-1]):
        rs = [r for r in runs if r["n_train"] == n]
        ms.confusion_figure(OUT / f"confusion_{n}.png", f"Trained on {n:,} genome windows",
                            [r["confusion_matrix"]["matrix"] for r in rs],
                            rs[0]["confusion_matrix"]["names"])

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        for s in summary:
            w.writerow({**s, "best_epochs": " ".join(map(str, s["best_epochs"]))})
    cols = ["n_train", "seed", "n_fit", "n_val", "best_epoch", "epochs_run", "best_val_loss",
            "best_val_acc", "test_accuracy", "test_macro_f1", "gt_decoy_recall",
            "ag_decoy_recall", "train_seconds"]
    with open(OUT / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(runs)

    first, last = summary[0], summary[-1]
    gain = last["macro_f1_mean"] - first["macro_f1_mean"]
    p = last["macro_f1_mean"]
    se = (p * (1 - p) / n_test) ** 0.5
    n_dec = {k: sum(x == g for x in g_te) for k, g in DECOYS.items()}
    emb_meta = {sp: read_json(EMB / f"{sp}.meta.json") for sp in ("train", "test")}
    lines = [
        "# Learning curve: does more training data help?", "",
        f"_Made by `scripts/18_learning_curve.py` on "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}; every number below is in "
        "`runs.json`._", "",
        f"NT-v2-50M frozen, centre + previous token concatenated (`{VARIANT}`), the head and "
        "recipe of `03_train_head.py`. Training windows: nested, class-balanced subsets of the "
        "120,000-window default train file of the genome dataset (`16_genome_dataset.py`). "
        f"Test: the same {n_test:,} genome test windows (chr20/21) for every head. "
        f"Mean ± SD over seeds {', '.join(map(str, SEEDS))}, range [min–max] in brackets.", "",
        "![Test macro-F1 vs training-set size](learning_curve.png)", "",
        "| training windows | test macro-F1 | test accuracy | GT decoys called none | "
        "AG decoys called none | epoch chosen |",
        "|---:|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['n_train']:,} | {fmt(s, 'macro_f1')} | {fmt(s, 'accuracy')} | "
                     f"{fmt(s, 'gt_decoy_recall')} | {fmt(s, 'ag_decoy_recall')} | "
                     f"{', '.join(map(str, s['best_epochs']))} |")
    lines += [
        "",
        f"From {first['n_train']:,} to {last['n_train']:,} windows the mean test macro-F1 "
        f"changes by {gain:+.3f} ({first['macro_f1_mean']:.3f} → {last['macro_f1_mean']:.3f}). "
        f"Decoy recall = share of the test set's GT decoys (n = {n_dec['GT decoy']:,}) or AG "
        f"decoys (n = {n_dec['AG decoy']:,}) that the head labels none.", "",
        "## Confusion matrices", "",
        f"Mean ± SD over the {len(SEEDS)} seeds; colour and percentage = share of the true class.",
        "",
        f"![Confusion matrix, {SIZES[0]:,} training windows](confusion_{SIZES[0]}.png)",
        f"![Confusion matrix, {SIZES[-1]:,} training windows](confusion_{SIZES[-1]}.png)", "",
        "## Recipe", "",
        "- Embedding: once, for all 120,000 train and all test windows, with "
        "`10_scan_gene.window_features` (float16, checkpointed every "
        f"{CHUNK:,} windows); stored in `data/emb/<model>/genome/`, apart from splice_sites_all's "
        "embeddings. Head input = `pair_features(C, P, \"center_plus_prev_concat\")` (1024 values).",
        "- Subsets: per seed, a seeded order within each class (the none class interleaves decoy "
        "GT, decoy AG, exon interior and intron interior in proportion); size n takes the first "
        "n/3 of each class, so each set contains the smaller ones and keeps the class balance and "
        "the none class's thirds. 120,000 is the whole file.",
        "- Head: `03_train_head.py` with `--seed s`: stratified 10% validation split "
        "(random_state s), StandardScaler on the other 90%, 1024 → 256 → 3, Adam (lr 0.001), "
        "batch 128, early stopping on validation loss (patience 20), best epoch restored.",
        "- The seed changes the subset, the validation split, the initial weights and the batch "
        "order. At 120,000 the training windows are the same for every seed.", "",
        "## Checks and caveats", "",
        f"- Test-set noise: at macro-F1 {p:.3f} on {n_test:,} windows the binomial standard error "
        f"is about {se:.4f}; the seed SD does not include it, since the test set is fixed.",
        f"- Decoy recall rests on {n_dec['GT decoy']:,} GT and {n_dec['AG decoy']:,} AG decoy "
        "windows, so its own binomial standard error is about "
        f"{(last['gt_decoy_recall_mean'] * (1 - last['gt_decoy_recall_mean']) / n_dec['GT decoy']) ** 0.5:.3f}"
        f"–{(first['ag_decoy_recall_mean'] * (1 - first['ag_decoy_recall_mean']) / n_dec['AG decoy']) ** 0.5:.3f}; "
        "steps smaller than that between neighbouring sizes are noise.",
        "- The genome test set is not the splice_sites_all test set, so these numbers are not "
        "directly comparable with the 0.940 of `phase1_summary.md` (see `../genome_head_check.md`).",
    ]
    if check17:
        lines.append(f"- Embedding check: {check17['rows']:,} test windows were also embedded "
                     f"separately by `17_genome_head_check.py`; the head inputs agree to a mean "
                     f"absolute difference of {check17['mean_abs_diff']:.1e} (max "
                     f"{check17['max_abs_diff']:.4f}, float16 rounding; mean |value| "
                     f"{check17['mean_abs_value']:.3f}).")
    lines += ["", "## Files", "",
              "- `summary.csv`: the table, machine-readable. `runs.csv`: one row per size and seed.",
              "- `runs.json`: every run in full (per-class metrics, confusion matrix, every negative "
              "group, epoch history).",
              "- `learning_curve.png`, `confusion_8000.png`, `confusion_120000.png`.",
              "- Heads: `data/emb/<model>/genome/heads/head_n<size>_s<seed>.pt` (03's format; "
              "git-ignored)."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    write_json_atomic(OUT / "runs.json", {
        "model": MODEL, "variant": VARIANT, "sizes": list(SIZES), "seeds": list(SEEDS),
        "n_test": n_test, "embedding": emb_meta, "embedding_check_vs_17": check17,
        "summary": summary, "runs": runs,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    print("\n" + "\n".join(lines[8:10 + len(summary)]))
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
