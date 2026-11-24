#!/usr/bin/env python3
"""
Phase 1, step 3: train a small classifier on top of the frozen embeddings.

    python scripts/03_train_head.py --model InstaDeepAI/nucleotide-transformer-v2-50m-multi-species
    python scripts/03_train_head.py --model zhihan1996/DNABERT-2-117M --variant center_w25

Reads data/emb/<model-slug>/{X,y}_{train,test}.npy written by 02_embed.py (or,
with --variant, data/emb/<model-slug>/<variant>/ written by 02b_embed_center.py)
and fits:

  * MLP head (PyTorch): input -> 256 -> ReLU -> Dropout(0.2) -> num_classes,
    Adam, early stopping on a stratified 10% validation split carved out of
    train (best-validation-loss weights are restored).
  * Plain sklearn LogisticRegression on the same vectors, fit on the full train
    set, as a second reference point.

Both get standardised inputs (StandardScaler fit on their training rows only).
For each: accuracy, macro-F1, per-class precision/recall/F1 and the confusion
matrix on the test split. Per-row test predictions are saved next to the
embeddings as pred_test_mlp.npy / pred_test_logreg.npy, and the trained MLP
head (weights + input scaler) as head_mlp.pt. Metrics are merged into
results/phase1_results.json
under models[<model name>] (full-window mean) or
pooling_variants[<model name>][<variant>]; other entries are left alone.

--hardneg (centre/previous-token variants only): the same MLP recipe, trained on
the training windows PLUS the "train"-split hard negatives of
13_hard_negatives.py (all labelled none). Saved as head_mlp_hardneg.pt and
pred_test_mlp_hardneg.npy; metrics under hardneg_heads[<model>][<variant>],
together with how this head and the original head_mlp.pt label the HELD-OUT
hard-negative genes (never used for training or selection). No logistic
regression in this mode.

Any head scoring above 0.97 accuracy is flagged loudly: the strand-aware GT/AG
rule already gets ~0.89 from the motif alone, so a near-perfect score would
suggest the head reads the centre motif literally rather than using context.
"""

import argparse
import copy
import sys
import time
from datetime import datetime, timezone

import numpy as np

from phase1_common import (EMB_DIR, RESULTS_JSON, SCAN_VARIANTS, label_names,
                           load_hard_negatives, load_label_info, model_slug, read_json,
                           short_name, update_json)

FLAG_ACCURACY = 0.97


def evaluate(y_true, y_pred, classes, names):
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes,
                                                 zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=classes,
                                         average="macro", zero_division=0)), 4),
        "per_class": {
            names[c]: {"label": int(c), "precision": round(float(p[i]), 4),
                       "recall": round(float(r[i]), 4), "f1": round(float(f[i]), 4),
                       "support": int(s[i])}
            for i, c in enumerate(classes)
        },
        "confusion_matrix": {
            "rows": "true label", "cols": "predicted label",
            "labels": [int(c) for c in classes],
            "names": [names[c] for c in classes],
            "matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        },
    }


def print_report(title, m):
    print(f"\n--- {title} ---")
    print(f"accuracy {m['accuracy']:.4f}   macro-F1 {m['macro_f1']:.4f}")
    print(f"{'class':>10} {'prec':>7} {'recall':>7} {'f1':>7} {'n':>6}")
    for name, v in m["per_class"].items():
        print(f"{name:>10} {v['precision']:7.3f} {v['recall']:7.3f} {v['f1']:7.3f} {v['support']:6d}")
    cm = m["confusion_matrix"]
    print("confusion (rows = true, cols = predicted): " + " ".join(cm["names"]))
    for name, row in zip(cm["names"], cm["matrix"]):
        print(f"{name:>10} " + " ".join(f"{v:6d}" for v in row))


def train_mlp(X_fit, y_fit, X_val, y_val, n_classes, args):
    import torch
    from torch import nn

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    Xf = torch.tensor(X_fit, dtype=torch.float32)
    yf = torch.tensor(y_fit, dtype=torch.long)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.long)

    model = nn.Sequential(
        nn.Linear(Xf.shape[1], 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, n_classes))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best = {"val_loss": float("inf"), "epoch": 0, "state": None}
    history = []
    since_best = 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        perm = torch.randperm(len(Xf), generator=gen)
        total = 0.0
        for i in range(0, len(Xf), args.batch_size):
            b = perm[i:i + args.batch_size]
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
        if epoch == 1 or epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  train loss {total / len(Xf):.4f}  "
                  f"val loss {val_loss:.4f}  val acc {val_acc:.4f}")
        if since_best >= args.patience:
            print(f"  early stop at epoch {epoch} (no val-loss improvement for "
                  f"{args.patience} epochs); restoring epoch {best['epoch']}")
            break

    model.load_state_dict(best["state"])
    model.eval()
    return model, best, history


def head_probs(path, X):
    """Class probabilities of a saved MLP head (head_mlp*.pt) on raw inputs X."""
    import torch
    from torch import nn

    h = torch.load(path)
    mlp = nn.Sequential(nn.Linear(h["in_dim"], h["hidden"]), nn.ReLU(), nn.Dropout(h["dropout"]),
                        nn.Linear(h["hidden"], len(h["classes"])))
    mlp.load_state_dict(h["state_dict"])
    mlp.eval()
    x = (torch.as_tensor(X, dtype=torch.float32) - h["scaler_mean"]) / h["scaler_scale"]
    with torch.no_grad():
        return torch.softmax(mlp(x), dim=-1).numpy(), h["classes"]


def hardneg_eval(hn, rows, heads, names):
    """How each saved head labels the given hard-negative rows, per group."""
    out = {}
    for kind, path in heads.items():
        if not path.exists():
            continue
        pr, cls = head_probs(path, hn["X"][rows])
        pred = np.array(cls)[pr.argmax(1)]
        none = [c for c in cls if names[c] == "none"][0]
        res = {}
        for g in sorted(set(hn["group"][rows])):
            m = hn["group"][rows] == g
            res[g] = {"n": int(m.sum()),
                      **{f"called_{names[c]}": round(float((pred[m] == c).mean()), 4) for c in cls},
                      "mean_p_site": round(float(1 - pr[m, cls.index(none)].mean()), 4)}
        res["all"] = {"n": int(len(rows)),
                      "called_none": round(float((pred == none).mean()), 4)}
        out[kind] = res
    return out


def main():
    ap = argparse.ArgumentParser(description="Train MLP + logistic-regression heads.")
    ap.add_argument("--model", required=True, help="Hugging Face repo id used in 02_embed.py")
    ap.add_argument("--variant", default=None,
                    help="pooling variant folder from 02b_embed_center.py (default: full-window mean)")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hardneg", action="store_true",
                    help="also train on the hard negatives of 13_hard_negatives.py")
    args = ap.parse_args()

    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    emb = EMB_DIR / model_slug(args.model)
    if args.variant:
        emb = emb / args.variant
    need = ["X_train", "y_train", "X_test", "y_test"]
    missing = [n for n in need if not (emb / f"{n}.npy").exists()]
    if missing:
        maker = "02b_embed_center.py" if args.variant else "02_embed.py"
        print(f"Missing {missing} in {emb}. Run {maker} --model {args.model} first.",
              file=sys.stderr)
        return 1
    X_train, y_train, X_test, y_test = (np.load(emb / f"{n}.npy") for n in need)
    meta = read_json(emb / "meta.json", default={}) or {}

    info = load_label_info()
    names = label_names(info)
    classes = sorted(names)
    if info["status"] != "inferred":
        print("WARNING: label semantics are unclear (see results/dataset_format.md); "
              "class names below are placeholders.")
    variant = args.variant or "full_mean"
    print(f"Model {args.model} [{variant}]: X_train {X_train.shape}, X_test {X_test.shape}, "
          f"classes {[f'{c}={names[c]}' for c in classes]}")
    hn, n_orig = None, len(y_train)
    if args.hardneg:
        if args.variant not in SCAN_VARIANTS:
            print(f"--hardneg needs a centre/previous-token variant {SCAN_VARIANTS}",
                  file=sys.stderr)
            return 1
        hn = load_hard_negatives(args.model, args.variant)
        tr = hn["split"] == "train"
        X_train = np.concatenate([X_train, hn["X"][tr]]).astype(np.float32)
        y_train = np.concatenate([y_train, np.full(int(tr.sum()), info["none_label"],
                                                   dtype=y_train.dtype)])
        print(f"+ {int(tr.sum())} hard negatives (label none) from the training genes of "
              f"13_hard_negatives.py: " + ", ".join(
                  f"{g} {int((hn['group'][tr] == g).sum())}" for g in sorted(set(hn["group"]))))

    # ----- MLP head -----
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=args.val_frac, stratify=y_train, random_state=args.seed)
    scaler = StandardScaler().fit(X_fit)
    to_idx = {c: i for i, c in enumerate(classes)}
    yi_fit = np.array([to_idx[v] for v in y_fit])
    yi_val = np.array([to_idx[v] for v in y_val])

    print(f"\nMLP head: fit on {len(X_fit)}, early-stop on {len(X_val)} validation rows")
    t0 = time.time()
    mlp, best, history = train_mlp(scaler.transform(X_fit), yi_fit,
                                   scaler.transform(X_val), yi_val, len(classes), args)
    mlp_secs = time.time() - t0
    with torch.no_grad():
        logits = mlp(torch.tensor(scaler.transform(X_test), dtype=torch.float32))
    y_pred_mlp = np.array(classes)[logits.argmax(1).numpy()]
    # The trained head itself (weights + input scaler), so later steps such as
    # 10_scan_gene.py apply exactly this classifier to new sequence.
    torch.save({"state_dict": mlp.state_dict(), "in_dim": int(X_train.shape[1]),
                "hidden": 256, "dropout": 0.2, "classes": [int(c) for c in classes],
                "class_names": [names[c] for c in classes],
                "scaler_mean": torch.tensor(scaler.mean_, dtype=torch.float32),
                "scaler_scale": torch.tensor(scaler.scale_, dtype=torch.float32),
                "model": args.model, "variant": args.variant or "full_mean",
                "hard_negatives": int(len(y_train) - n_orig)},
               emb / ("head_mlp_hardneg.pt" if hn is not None else "head_mlp.pt"))
    mlp_metrics = evaluate(y_test, y_pred_mlp, classes, names)
    mlp_metrics["training"] = {
        "architecture": f"Linear({X_train.shape[1]},256) -> ReLU -> Dropout(0.2) -> "
                        f"Linear(256,{len(classes)})",
        "optimizer": f"Adam(lr={args.lr}, weight_decay={args.weight_decay})",
        "batch_size": args.batch_size, "max_epochs": args.max_epochs,
        "patience": args.patience, "val_frac": args.val_frac, "seed": args.seed,
        "n_fit": int(len(X_fit)), "n_val": int(len(X_val)),
        "best_epoch": best["epoch"], "epochs_run": len(history),
        "best_val_loss": round(best["val_loss"], 5), "best_val_acc": round(best["val_acc"], 4),
        "preprocessing": "StandardScaler fit on the n_fit rows",
        "train_seconds": round(mlp_secs, 1),
        "history": history,
    }
    print_report(f"{short_name(args.model)} [{variant}]{' + hard negatives' if hn else ''} "
                 f"+ MLP head (test, n={len(y_test)})", mlp_metrics)
    if hn is not None:
        np.save(emb / "pred_test_mlp_hardneg.npy", y_pred_mlp.astype(np.int64))
        hold = np.flatnonzero(hn["split"] == "holdout")
        ev = hardneg_eval(hn, hold, {"original": emb / "head_mlp.pt",
                                     "hardneg": emb / "head_mlp_hardneg.pt"}, names)
        print(f"\nHeld-out hard-negative genes ({len(hold)} windows), share called 'none':")
        for g in ev["hardneg"]:
            if g == "all":
                continue
            print(f"  {g:<16} original head {ev.get('original', {}).get(g, {}).get('called_none', float('nan')):.3f}"
                  f"   hard-negative head {ev['hardneg'][g]['called_none']:.3f}")
        entry = {"short_name": short_name(args.model), "variant": variant,
                 "n_train_original": n_orig, "n_hard_negatives": int(len(y_train) - n_orig),
                 "n_test": int(len(y_test)), "mlp": mlp_metrics, "holdout_hard_negatives": ev,
                 "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        update_json(RESULTS_JSON, ["hardneg_heads", args.model, args.variant], entry)
        print(f"\nSaved under ['hardneg_heads', {args.model!r}, {args.variant!r}] in {RESULTS_JSON}")
        return 0

    # ----- logistic regression -----
    print("\nLogisticRegression: fit on the full training set")
    t0 = time.time()
    logreg = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=5000, random_state=args.seed))
    logreg.fit(X_train, y_train)
    lr_secs = time.time() - t0
    y_pred_lr = logreg.predict(X_test)
    lr_metrics = evaluate(y_test, y_pred_lr, classes, names)
    # Per-row test predictions, for analyses such as 09_motif_analysis.py.
    np.save(emb / "pred_test_mlp.npy", y_pred_mlp.astype(np.int64))
    np.save(emb / "pred_test_logreg.npy", np.asarray(y_pred_lr, dtype=np.int64))
    lr_metrics["training"] = {
        "estimator": "StandardScaler -> LogisticRegression(lbfgs, C=1.0, max_iter=5000)",
        "n_fit": int(len(X_train)),
        "n_iter": int(np.max(logreg[-1].n_iter_)),
        "train_seconds": round(lr_secs, 1),
    }
    print_report(f"{short_name(args.model)} [{variant}] + logistic regression "
                 f"(test, n={len(y_test)})", lr_metrics)

    flags = [h for h, m in (("mlp", mlp_metrics), ("logreg", lr_metrics))
             if m["accuracy"] > FLAG_ACCURACY]
    for h in flags:
        acc = mlp_metrics["accuracy"] if h == "mlp" else lr_metrics["accuracy"]
        print("\n" + "!" * 78)
        print(f"!!! FLAG: {short_name(args.model)} [{variant}] {h} accuracy {acc:.4f} > "
              f"{FLAG_ACCURACY}.")
        print("!!! Suspiciously high: the head may be reading the centre motif literally")
        print("!!! rather than using context. Inspect before trusting this number.")
        print("!" * 78)

    entry = {
        "short_name": short_name(args.model),
        "variant": variant,
        "flag_above_0_97": flags,
        "embedding_dir": str(emb.relative_to(EMB_DIR.parent.parent)),
        "embedding_dim": int(X_train.shape[1]),
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "embedding_run": meta.get("run_key"),
        "label_semantics": info["status"],
        "label_names": {str(c): names[c] for c in classes},
        "mlp": mlp_metrics,
        "logreg": lr_metrics,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    keys = (["pooling_variants", args.model, args.variant] if args.variant
            else ["models", args.model])
    update_json(RESULTS_JSON, keys, entry)
    print(f"\nSaved under {keys} in {RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
