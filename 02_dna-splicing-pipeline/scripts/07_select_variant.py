#!/usr/bin/env python3
"""
Phase 1, step 7: choose each model's pooling variant WITHOUT looking at the test set.

    python scripts/07_select_variant.py

Before this step, the "best" pooling variant was read off the test-set table,
which flatters the winner. Protocol here:

  1. Carve a SELECTION split (10%, class-stratified, seed 42) out of the 8,000
     training rows. It is a different split from the MLP head's own
     early-stopping split (03 uses seed 0 on the full train set); the overlap
     that occurs by chance is reported.
  2. For every pooling variant, train the exact 03 MLP recipe on the remaining
     90% (which carves its own early-stopping split from that 90%) and score it
     on the selection split.
  3. Pick the variant with the best selection-split accuracy (macro-F1 breaks
     ties). The test set is not read for this decision.
  4. The final number is that ONE variant's test score from 03, where the head
     is refit on all 8,000 training rows (standard "select on validation, refit,
     test once").

Writes variant_selection into results/phase1_results.json and prints how far
the reported number moved from the test-selected (leaky) best.

--hardneg: the same protocol for the head trained with the hard negatives of
13_hard_negatives.py, over the variants a per-position scan can use (centre /
previous token). The selection split is the SAME original rows as above plus
whole hard-negative genes (seeded, ~10% of the training-split hard negatives;
the held-out genes of 13 are never touched). For each variant two selection
heads are trained on the remaining rows: one without hard negatives (= the
current recipe) and one with them, and both are scored on the same selection
split. The variant is chosen by the hard-negative head's accuracy on the whole
selection split; the final number is that variant's 03 --hardneg test score.
Writes variant_selection_hardneg.
"""

import argparse
import contextlib
import importlib
import io
import sys
import types

import numpy as np

from phase1_common import (EMB_DIR, MODELS, RESULTS_JSON, SCAN_VARIANTS, SEED, label_names,
                           load_hard_negatives, load_label_info, model_slug, read_json,
                           update_json)

VARIANTS = ["full_mean", "center_tok", "center_w10", "center_w25", "center_w50",
            "concat_c25_full", "prev_tok", "center_plus_prev", "center_plus_prev_concat"]
th = importlib.import_module("03_train_head")      # reuse the exact head recipe


def variant_dir(repo, v):
    base = EMB_DIR / model_slug(repo)
    return base if v == "full_mean" else base / v


def refit_entry(res, repo, v):
    """The 03 result for this variant (head trained on all train rows)."""
    if v == "full_mean":
        return res.get("models", {}).get(repo)
    return res.get("pooling_variants", {}).get(repo, {}).get(v)


def main():
    ap = argparse.ArgumentParser(description="Validation-only pooling-variant selection.")
    ap.add_argument("--sel-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=SEED, help="seed for the selection split")
    ap.add_argument("--head-seed", type=int, default=0, help="seed of the 03 head recipe")
    ap.add_argument("--hardneg", action="store_true",
                    help="select the hard-negative head's variant (see docstring)")
    args = ap.parse_args()
    if args.hardneg:
        return main_hardneg(args)

    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    res = read_json(RESULTS_JSON) or {}
    info = load_label_info()
    classes = sorted(label_names(info))
    head_args = types.SimpleNamespace(seed=args.head_seed, lr=1e-3, weight_decay=0.0,
                                      batch_size=128, max_epochs=300, patience=20)
    out = {
        "protocol": (f"selection split = {args.sel_frac:.0%} of the training rows, "
                     f"class-stratified, seed {args.seed}; selection heads trained on the "
                     "other rows with the 03 MLP recipe (own early-stopping split inside); "
                     "variant chosen by selection-split accuracy; final number = that "
                     "variant's 03 test score (head refit on all training rows)"),
        "sel_frac": args.sel_frac, "seed": args.seed, "models": {},
    }
    failed = False

    for repo, short in MODELS.items():
        avail = [v for v in VARIANTS if (variant_dir(repo, v) / "X_train.npy").exists()]
        if not avail:
            print(f"{short}: no embeddings found, skipping")
            failed = True
            continue
        ys = {v: np.load(variant_dir(repo, v) / "y_train.npy") for v in avail}
        y = ys[avail[0]]
        if not all(np.array_equal(ys[v], y) for v in avail):
            print(f"{short}: training labels differ between variants; refusing to compare.")
            failed = True
            continue
        n = len(y)
        rows = np.arange(n)
        fit_rows, sel_rows = train_test_split(rows, test_size=args.sel_frac, stratify=y,
                                              random_state=args.seed)
        # The 03 head's early-stopping rows on the full train set, for the overlap report.
        _, es03 = train_test_split(rows, test_size=0.10, stratify=y, random_state=args.head_seed)
        overlap = int(np.intersect1d(sel_rows, es03).size)

        print(f"\n=== {short}: selection split {len(sel_rows)} rows, selection heads fit on "
              f"{len(fit_rows)} (overlap with 03's early-stopping rows: {overlap} rows, "
              f"chance level ~{len(sel_rows) * 0.10:.0f})")
        val, sel_test = {}, {}
        for v in avail:
            d = variant_dir(repo, v)
            X = np.load(d / "X_train.npy")
            X_test, y_test = np.load(d / "X_test.npy"), np.load(d / "y_test.npy")
            Xf, Xes, yf, yes = train_test_split(X[fit_rows], y[fit_rows], test_size=0.10,
                                                stratify=y[fit_rows], random_state=args.head_seed)
            sc = StandardScaler().fit(Xf)
            to_idx = {c: i for i, c in enumerate(classes)}
            with contextlib.redirect_stdout(io.StringIO()):
                mlp, best, _ = th.train_mlp(sc.transform(Xf), np.array([to_idx[t] for t in yf]),
                                            sc.transform(Xes), np.array([to_idx[t] for t in yes]),
                                            len(classes), head_args)

            def predict(Xa):
                with torch.no_grad():
                    logits = mlp(torch.tensor(sc.transform(Xa), dtype=torch.float32))
                return np.array(classes)[logits.argmax(1).numpy()]

            pv = predict(X[sel_rows])
            val[v] = {"accuracy": round(float(accuracy_score(y[sel_rows], pv)), 4),
                      "macro_f1": round(float(f1_score(y[sel_rows], pv, labels=classes,
                                                       average="macro")), 4),
                      "best_epoch": best["epoch"]}
            pt = predict(X_test)    # reported only; never used for the decision
            sel_test[v] = {"accuracy": round(float(accuracy_score(y_test, pt)), 4),
                           "macro_f1": round(float(f1_score(y_test, pt, labels=classes,
                                                            average="macro")), 4)}
            print(f"  {v:<16} selection-split acc {val[v]['accuracy']:.4f}  "
                  f"macro-F1 {val[v]['macro_f1']:.4f}")

        chosen = max(avail, key=lambda v: (val[v]["accuracy"], val[v]["macro_f1"],
                                           -VARIANTS.index(v)))
        refit = {v: refit_entry(res, repo, v) for v in avail}
        missing = [v for v in avail if not refit[v]]
        if missing:
            print(f"  WARNING: no 03 test result for {missing}; run 03 first.")
        final = refit[chosen]["mlp"] if refit[chosen] else None
        leaky_v = max((v for v in avail if refit[v]),
                      key=lambda v: refit[v]["mlp"]["accuracy"], default=None)
        leaky = refit[leaky_v]["mlp"] if leaky_v else None
        moved = (final["accuracy"] - leaky["accuracy"]) if final and leaky else None
        out["models"][repo] = {
            "short_name": short,
            "n_selection": int(len(sel_rows)), "n_selection_fit": int(len(fit_rows)),
            "overlap_with_03_early_stopping_rows": overlap,
            "selection_split": val,
            "selection_head_test_for_reference_only": sel_test,
            "selected_variant": chosen,
            "selected_val_accuracy": val[chosen]["accuracy"],
            "selected_val_macro_f1": val[chosen]["macro_f1"],
            "final_test": ({"accuracy": final["accuracy"], "macro_f1": final["macro_f1"],
                            "n": int(sum(c["support"] for c in final["per_class"].values()))}
                           if final else None),
            "leaky_test_selected": ({"variant": leaky_v, "accuracy": leaky["accuracy"],
                                     "macro_f1": leaky["macro_f1"]} if leaky else None),
            "moved_vs_leaky_points": round(moved * 100, 2) if moved is not None else None,
        }
        print(f"  -> selected on validation: {chosen} (val acc {val[chosen]['accuracy']:.4f})")
        if final:
            print(f"  -> final test score (03 head refit on all train rows): accuracy "
                  f"{final['accuracy']:.4f}, macro-F1 {final['macro_f1']:.4f}")
        if leaky:
            same = "same variant" if leaky_v == chosen else f"leaky choice was {leaky_v}"
            print(f"  -> test-selected (leaky) best was {leaky['accuracy']:.4f}; the number "
                  f"moved {moved * 100:+.1f} points ({same})")

    update_json(RESULTS_JSON, ["variant_selection"], out)
    print(f"\nSaved under 'variant_selection' in {RESULTS_JSON}")
    return 1 if failed else 0


def main_hardneg(args):
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    res = read_json(RESULTS_JSON) or {}
    info = load_label_info()
    names = label_names(info)
    classes = sorted(names)
    to_idx = {c: i for i, c in enumerate(classes)}
    none = info["none_label"]
    head_args = types.SimpleNamespace(seed=args.head_seed, lr=1e-3, weight_decay=0.0,
                                      batch_size=128, max_epochs=300, patience=20)
    out = {"protocol": (f"selection split = the {args.sel_frac:.0%} original training rows of "
                        "variant_selection (class-stratified, seed "
                        f"{args.seed}) + whole hard-negative genes (seeded, ~{args.sel_frac:.0%} of "
                        "the training-split hard negatives); two selection heads per variant "
                        "(without / with hard negatives) on the remaining rows, 03 MLP recipe; "
                        "variant chosen by the hard-negative head's accuracy on the whole "
                        "selection split; final number = that variant's 03 --hardneg test score"),
           "sel_frac": args.sel_frac, "seed": args.seed, "variants": list(SCAN_VARIANTS),
           "models": {}}
    failed = False
    for repo, short in MODELS.items():
        avail = [v for v in SCAN_VARIANTS if (variant_dir(repo, v) / "X_train.npy").exists()]
        try:
            hns = {v: load_hard_negatives(repo, v) for v in avail}
        except (FileNotFoundError, ValueError) as exc:
            print(f"{short}: {exc}")
            failed = True
            continue
        y = np.load(variant_dir(repo, avail[0]) / "y_train.npy")
        rows = np.arange(len(y))
        fit_rows, sel_rows = train_test_split(rows, test_size=args.sel_frac, stratify=y,
                                              random_state=args.seed)
        hn0 = hns[avail[0]]
        tr = np.flatnonzero(hn0["split"] == "train")
        genes = np.array(sorted(set(hn0["gene"][tr])))
        sel_genes, n_sel = [], 0
        for g in genes[np.random.default_rng(args.seed).permutation(len(genes))]:
            if n_sel >= args.sel_frac * len(tr):
                break
            sel_genes.append(g)
            n_sel += int((hn0["gene"][tr] == g).sum())
        in_sel = np.isin(hn0["gene"][tr], sel_genes)
        hn_sel, hn_fit = tr[in_sel], tr[~in_sel]
        print(f"\n=== {short}: selection split = {len(sel_rows)} original rows + {len(hn_sel)} "
              f"hard negatives from {len(sel_genes)} genes; selection heads fit on "
              f"{len(fit_rows)} original rows (+ {len(hn_fit)} hard negatives)")
        val = {}
        for v in avail:
            X = np.load(variant_dir(repo, v) / "X_train.npy")
            Xh = hns[v]["X"]
            fits = {"original": (X[fit_rows], y[fit_rows]),
                    "hardneg": (np.concatenate([X[fit_rows], Xh[hn_fit]]),
                                np.concatenate([y[fit_rows], np.full(len(hn_fit), none, y.dtype)]))}
            val[v] = {}
            for kind, (Xa, ya) in fits.items():
                Xf, Xes, yf, yes = train_test_split(Xa, ya, test_size=0.10, stratify=ya,
                                                    random_state=args.head_seed)
                sc = StandardScaler().fit(Xf)
                with contextlib.redirect_stdout(io.StringIO()):
                    mlp, best, _ = th.train_mlp(sc.transform(Xf), np.array([to_idx[t] for t in yf]),
                                                sc.transform(Xes), np.array([to_idx[t] for t in yes]),
                                                len(classes), head_args)

                def predict(Xp):
                    with torch.no_grad():
                        lg = mlp(torch.tensor(sc.transform(Xp), dtype=torch.float32))
                    return np.array(classes)[lg.argmax(1).numpy()]
                po, ph = predict(X[sel_rows]), predict(Xh[hn_sel])
                yall = np.concatenate([y[sel_rows], np.full(len(hn_sel), none, y.dtype)])
                pall = np.concatenate([po, ph])
                grp = hns[v]["group"][hn_sel]
                val[v][kind] = {
                    "accuracy_all": round(float(accuracy_score(yall, pall)), 4),
                    "macro_f1_all": round(float(f1_score(yall, pall, labels=classes,
                                                         average="macro")), 4),
                    "accuracy_original_rows": round(float(accuracy_score(y[sel_rows], po)), 4),
                    "hard_negatives_called_none": round(float((ph == none).mean()), 4),
                    "hard_negatives_called_none_by_group": {
                        g: round(float((ph[grp == g] == none).mean()), 4) for g in sorted(set(grp))},
                    "best_epoch": best["epoch"]}
                r = val[v][kind]
                print(f"  {v:<24} {kind:<8} all {r['accuracy_all']:.4f}  original rows "
                      f"{r['accuracy_original_rows']:.4f}  hard negatives called none "
                      f"{r['hard_negatives_called_none']:.3f}")
        chosen = max(avail, key=lambda v: (val[v]["hardneg"]["accuracy_all"],
                                           val[v]["hardneg"]["macro_f1_all"],
                                           -SCAN_VARIANTS.index(v)))
        final = res.get("hardneg_heads", {}).get(repo, {}).get(chosen)
        cur = res.get("variant_selection", {}).get("models", {}).get(repo, {})
        out["models"][repo] = {
            "short_name": short, "n_selection_original": int(len(sel_rows)),
            "n_selection_hard_negatives": int(len(hn_sel)), "selection_genes": sel_genes,
            "n_fit_original": int(len(fit_rows)), "n_fit_hard_negatives": int(len(hn_fit)),
            "selection_split": val, "selected_variant": chosen,
            "selected_val_accuracy": val[chosen]["hardneg"]["accuracy_all"],
            "current_head_selected_variant": cur.get("selected_variant"),
            "final_test": ({"accuracy": final["mlp"]["accuracy"],
                            "macro_f1": final["mlp"]["macro_f1"],
                            "n": final["n_test"]} if final else None),
        }
        print(f"  -> selected on validation: {chosen} (all-rows val acc "
              f"{val[chosen]['hardneg']['accuracy_all']:.4f})")
        if final:
            print(f"  -> final test score (03 --hardneg, head refit on all training rows + "
                  f"hard negatives): accuracy {final['mlp']['accuracy']:.4f}")
        else:
            print(f"  WARNING: no 03 --hardneg result for {chosen}; run 03 first.")
            failed = True
    update_json(RESULTS_JSON, ["variant_selection_hardneg"], out)
    print(f"\nSaved under 'variant_selection_hardneg' in {RESULTS_JSON}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
