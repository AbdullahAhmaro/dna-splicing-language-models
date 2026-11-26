#!/usr/bin/env python3
"""
Step 17: does the genome-wide dataset of 16 train our current head as well as splice_sites_all?

    python scripts/17_genome_head_check.py

The run that gave ~0.94 (results/phase1_results.json, NT-v2-50M, variant
center_plus_prev_concat, MLP head) used 8,000 train and 2,000 test windows. This step
repeats it on the genome dataset with the same sizes: a class-stratified 8,000 of the
default train file and 2,000 of the test file (chr20/21), seed 42, so 10,000 windows.

Embedding: the frozen NT-v2-50M, last-layer vector of the token covering base 300 (C) and of
the token before it (P), via 10_scan_gene.window_pairs (as 13 embeds its hard negatives);
features = phase1_common.pair_features(C, P, "center_plus_prev_concat").
Head: 03_train_head.train_mlp with 03's defaults (1024 -> 256 -> 3, Adam 1e-3, batch 128,
early stopping on a stratified 10% validation split, seed 0), scored with 03's evaluate().

Two cross-checks that need no new embedding:
  the new head on the old 2,000 splice_sites_all test windows (their saved vectors), and
  the old head (center_plus_prev_concat/head_mlp.pt) on the new 2,000 test windows.

Outputs: data/emb/<model>/genome_sample/ (vectors, row indices), results/genome_head_check.json
and results/genome_head_check.md.
"""

import argparse
import hashlib
import importlib
import sys
import time
from datetime import datetime, timezone

import numpy as np

from phase1_common import (DATA_DIR, EMB_DIR, GENOME_DEFAULT_DIR, GENOME_FULL_DIR, RESULTS_DIR,
                           RESULTS_JSON, SEED, label_names, load_label_info, model_slug,
                           pair_features, read_json, stratified_subsample, write_json_atomic)

th = importlib.import_module("03_train_head")
sg = importlib.import_module("10_scan_gene")

MODEL = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
VARIANT = "center_plus_prev_concat"
N_TRAIN, N_TEST = 8000, 2000
DEF_DIR = GENOME_DEFAULT_DIR
META_TEST = GENOME_FULL_DIR / "test_meta.parquet"
OUT = EMB_DIR / model_slug(MODEL) / "genome_sample"
OUT_JSON = RESULTS_DIR / "genome_head_check.json"
OUT_MD = RESULTS_DIR / "genome_head_check.md"


def sha1(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


def embed(split, seqs, idx, sha):
    """C and P vectors of the sampled windows (cached; redone if the parquet changed)."""
    meta = read_json(OUT / f"{split}_meta.json", default=None)
    if (meta and meta["parquet_sha1"] == sha and (OUT / f"C_{split}.npy").exists()
            and np.array_equal(np.load(OUT / f"idx_{split}.npy"), idx)):
        print(f"{split}: embeddings up to date ({len(idx):,} windows)")
        return np.load(OUT / f"C_{split}.npy"), np.load(OUT / f"P_{split}.npy"), meta["seconds"]
    import torch
    from transformers import AutoTokenizer

    from try_dna_models import load_model

    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model, _ = load_model(MODEL)
    model.eval()
    texts = [seqs[i] for i in idx]
    t0 = time.time()
    Cs, Ps = [], []
    for i in range(0, len(texts), 512):
        C, P = sg.window_pairs(texts[i:i + 512], tok, model, torch, batch=32)
        Cs.append(C)
        Ps.append(P)
        done = min(i + 512, len(texts))
        print(f"  {split}: {done:,}/{len(texts):,} windows ({done / (time.time() - t0):.1f}/s)",
              flush=True)
    C, P = np.concatenate(Cs).astype(np.float32), np.concatenate(Ps).astype(np.float32)
    secs = round(time.time() - t0, 1)
    np.save(OUT / f"C_{split}.npy", C)
    np.save(OUT / f"P_{split}.npy", P)
    np.save(OUT / f"idx_{split}.npy", idx)
    write_json_atomic(OUT / f"{split}_meta.json", {
        "parquet": str((DEF_DIR / f"{split}.parquet").relative_to(DATA_DIR.parent)),
        "parquet_sha1": sha, "n": int(len(idx)), "seconds": secs,
        "content": "C = last-layer vector of the token covering base 300, P = the token before "
                   "it (10_scan_gene.window_pairs); rows = idx_<split>.npy into the parquet"})
    return C, P, secs


def main():
    argparse.ArgumentParser(description="Head check on the genome dataset.").parse_args()
    import pyarrow.parquet as pq
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    info = load_label_info()
    names = label_names(info)
    classes = sorted(names)
    data, X, y, secs = {}, {}, {}, {}
    for split, n in (("train", N_TRAIN), ("test", N_TEST)):
        path = DEF_DIR / f"{split}.parquet"
        data[split] = pq.read_table(path, columns=["sequence", "name", "label"]).to_pydict()
        labels = np.array(data[split]["label"])
        idx = stratified_subsample(labels, n, SEED)
        C, P, secs[split] = embed(split, data[split]["sequence"], idx, sha1(path))
        X[split] = pair_features(C, P, VARIANT)
        y[split] = labels[idx]
        data[split]["idx"] = idx

    # 03's recipe, unchanged
    args = argparse.Namespace(lr=1e-3, weight_decay=0.0, batch_size=128, max_epochs=300,
                              patience=20, seed=0, val_frac=0.10)
    X_fit, X_val, y_fit, y_val = train_test_split(X["train"], y["train"], test_size=args.val_frac,
                                                  stratify=y["train"], random_state=args.seed)
    scaler = StandardScaler().fit(X_fit)
    to_idx = {c: i for i, c in enumerate(classes)}
    mlp, best, hist = th.train_mlp(scaler.transform(X_fit), np.array([to_idx[v] for v in y_fit]),
                                   scaler.transform(X_val), np.array([to_idx[v] for v in y_val]),
                                   len(classes), args)

    def predict(Xs):
        with torch.no_grad():
            lg = mlp(torch.tensor(scaler.transform(Xs), dtype=torch.float32))
        return np.array(classes)[lg.argmax(1).numpy()]
    pred = predict(X["test"])
    new_on_new = th.evaluate(y["test"], pred, classes, names)
    th.print_report(f"NT-v2-50M [{VARIANT}] head trained on genome train, genome test "
                    f"(n={len(pred)})", new_on_new)

    # cross-checks
    old_dir = EMB_DIR / model_slug(MODEL) / VARIANT
    Xo, yo = np.load(old_dir / "X_test.npy"), np.load(old_dir / "y_test.npy")
    new_on_old = th.evaluate(yo, predict(Xo), classes, names)
    pr, cls = th.head_probs(old_dir / "head_mlp.pt", X["test"])
    old_on_new = th.evaluate(y["test"], np.array(cls)[pr.argmax(1)], classes, names)
    ref = read_json(RESULTS_JSON)["pooling_variants"][MODEL][VARIANT]["mlp"]
    print(f"\nnew head on old test: acc {new_on_old['accuracy']:.4f} F1 {new_on_old['macro_f1']:.4f}")
    print(f"old head on new test: acc {old_on_new['accuracy']:.4f} F1 {old_on_new['macro_f1']:.4f}")
    print(f"earlier (old train -> old test): acc {ref['accuracy']:.4f} F1 {ref['macro_f1']:.4f}")

    # how the new head does on each kind of negative in the new test sample
    meta = pq.read_table(META_TEST, columns=["name", "group", "subtype"]).to_pydict()
    grp = {n: f"{g} {s}".strip() for n, g, s in zip(meta["name"], meta["group"], meta["subtype"])}
    tn = [data["test"]["name"][i] for i in data["test"]["idx"]]
    by_group = {}
    for g in sorted({grp[n] for n, t in zip(tn, y["test"]) if t == 0}):
        m = np.array([grp[n] == g for n in tn])
        by_group[g] = {"n": int(m.sum()), "called_none": round(float((pred[m] == 0).mean()), 4)}

    res = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "model": MODEL, "variant": VARIANT, "n_train": int(len(y["train"])),
           "n_test": int(len(y["test"])), "sample_seed": SEED,
           "embedding_seconds": secs,
           "training": {"n_fit": int(len(X_fit)), "n_val": int(len(X_val)),
                        "best_epoch": best["epoch"], "epochs_run": len(hist),
                        "best_val_acc": round(best["val_acc"], 4)},
           "genome_train_on_genome_test": new_on_new,
           "genome_head_on_old_test": new_on_old,
           "old_head_on_genome_test": old_on_new,
           "earlier_old_train_on_old_test": {"accuracy": ref["accuracy"],
                                             "macro_f1": ref["macro_f1"]},
           "genome_test_negatives_called_none": by_group}
    write_json_atomic(OUT_JSON, res)
    pc = new_on_new["per_class"]
    md = ["# Head check on the genome-wide dataset", "",
          f"_Generated by `scripts/17_genome_head_check.py` on {res['generated_utc'][:16]} UTC; "
          "numbers in `results/genome_head_check.json`._", "",
          f"NT-v2-50M, `{VARIANT}`, 03's MLP head and recipe. {N_TRAIN:,} train windows "
          f"(class-stratified from the default genome train file) and {N_TEST:,} test windows "
          "(from chr20/21), seed 42, the sizes of the earlier run.", "",
          "| trained on | tested on | accuracy | macro-F1 |", "|---|---|---:|---:|",
          f"| splice_sites_all (earlier, `phase1_results.json`) | splice_sites_all test | "
          f"{ref['accuracy']:.3f} | {ref['macro_f1']:.3f} |",
          f"| **genome** | **genome test (chr20/21)** | **{new_on_new['accuracy']:.3f}** | "
          f"**{new_on_new['macro_f1']:.3f}** |",
          f"| genome | splice_sites_all test | {new_on_old['accuracy']:.3f} | "
          f"{new_on_old['macro_f1']:.3f} |",
          f"| splice_sites_all (saved head) | genome test | {old_on_new['accuracy']:.3f} | "
          f"{old_on_new['macro_f1']:.3f} |", "",
          "Genome head on genome test, per class F1: " + ", ".join(
              f"{k} {v['f1']:.3f}" for k, v in pc.items()) + ". Negatives called none: " +
          ", ".join(f"{g} {v['called_none']:.3f} (n={v['n']})" for g, v in by_group.items()) + ".",
          ""]
    OUT_MD.write_text("\n".join(md))
    print(f"wrote {OUT_JSON} and {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
