#!/usr/bin/env python3
"""
Phase 1, step 4: a baseline with no machine learning, only the canonical splice motifs.

    python scripts/04_baseline.py                 # same 2,000-row test subset as 02_embed.py
    python scripts/04_baseline.py --max-test -1   # if the models were embedded on the full test set

Biology: an intron almost always starts with GT (the donor site) and ends
with AG (the acceptor site).

Rule A, exactly as specified:
    look at the two letters at offset k:
        "GT" -> donor,  "AG" -> acceptor,  anything else -> neither.
    Every offset k is scanned. The best one is reported with its test accuracy
    and macro-F1, along with the full offset-vs-accuracy curve. Choosing k on
    the test set flatters the rule, so the offset chosen on the TRAIN split is
    reported too.

Rule B, strand-aware (added because of what 01_get_data.py found):
    the windows are written on the genome's + strand, so about half of all
    sites appear reverse-complemented (GT becomes AC, AG becomes CT). Donor
    and acceptor motifs also sit at different offsets. Rule B checks
        donor    if GT at k1 or AC at k2
        acceptor if AG at k3 or CT at k4      (donor wins if both match)
    Each k is picked on the TRAIN split only, then scored on test.

The mapping from motif to label value comes from data/label_info.json
(written by 01). If 01 could not settle the mapping, this script says so
loudly and reports every possible assignment instead of silently picking one.
"""

import argparse
import itertools
import sys

import numpy as np

from phase1_common import (EMB_DIR, RESULTS_JSON, SEED, dinuc_hits, label_names,
                           load_label_info, load_splits, seq_matrix,
                           stratified_subsample, update_json)


def scores(y, pred, classes):
    from sklearn.metrics import accuracy_score, f1_score
    return (float(accuracy_score(y, pred)),
            float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)))


def full_metrics(y, pred, classes, names):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    acc, f1 = scores(y, pred, classes)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": round(acc, 4), "macro_f1": round(f1, 4),
        "per_class": {names[c]: {"label": int(c), "precision": round(float(p[i]), 4),
                                 "recall": round(float(r[i]), 4), "f1": round(float(f[i]), 4),
                                 "support": int(s[i])} for i, c in enumerate(classes)},
        "confusion_matrix": {"rows": "true label", "cols": "predicted label",
                             "labels": [int(c) for c in classes],
                             "names": [names[c] for c in classes],
                             "matrix": confusion_matrix(y, pred, labels=classes).tolist()},
    }


def rule_a(hits_gt, hits_ag, k, donor, acceptor, none):
    pred = np.full(hits_gt.shape[0], none)
    pred[hits_gt[:, k]] = donor
    pred[hits_ag[:, k]] = acceptor
    return pred


def scan_rule_a(mat, y, donor, acceptor, none, classes):
    gt, ag = dinuc_hits(mat, "GT"), dinuc_hits(mat, "AG")
    acc, f1 = np.zeros(gt.shape[1]), np.zeros(gt.shape[1])
    for k in range(gt.shape[1]):
        acc[k], f1[k] = scores(y, rule_a(gt, ag, k, donor, acceptor, none), classes)
    return acc, f1


def best_offset_one_vs_rest(hits, is_target):
    """Offset where 'motif present' best predicts 'is this class' (F1, one-vs-rest)."""
    tp = (hits & is_target[:, None]).sum(0)
    fp = (hits & ~is_target[:, None]).sum(0)
    fn = (~hits & is_target[:, None]).sum(0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    k = int(np.argmax(f1))
    return k, float(f1[k])


def rule_b(mat, k, donor, acceptor, none):
    d = dinuc_hits(mat, "GT")[:, k["GT"]] | dinuc_hits(mat, "AC")[:, k["AC"]]
    a = dinuc_hits(mat, "AG")[:, k["AG"]] | dinuc_hits(mat, "CT")[:, k["CT"]]
    pred = np.full(mat.shape[0], none)
    pred[a] = acceptor
    pred[d] = donor
    return pred, int((d & a).sum())


def main():
    ap = argparse.ArgumentParser(description="GT/AG motif baseline (no ML).")
    ap.add_argument("--max-test", type=int, default=2000,
                    help="must match 02_embed.py so all methods see the same test rows (-1 = all)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    ds = load_splits()
    info = load_label_info()
    names = label_names(info)
    classes = sorted(names)

    tr_mat, tr_y = seq_matrix(ds["train"]["sequence"]), np.array(ds["train"]["label"])
    te_mat, te_y = seq_matrix(ds["test"]["sequence"]), np.array(ds["test"]["label"])
    sub = stratified_subsample(te_y, args.max_test, args.seed)

    # Same test rows as the embedding runs?
    matches = []
    for p in sorted(EMB_DIR.glob("*/idx_test.npy")):
        matches.append(bool(np.array_equal(np.load(p), sub)))
    same_rows = all(matches) if matches else None
    if matches and not same_rows:
        print("WARNING: some embedding runs used a different test subset than "
              f"--max-test {args.max_test}; baseline and model numbers are not on the same rows.")

    print(f"Test split: {len(te_y)} rows; model-matched subset: {len(sub)} rows "
          f"(seed {args.seed}, max_test {args.max_test})")

    # ----- label mapping -----
    if info["status"] == "inferred":
        mappings = [(info["donor_label"], info["acceptor_label"], info["none_label"])]
        print(f"Label mapping (inferred by 01 from motif enrichment, not documented by the "
              f"dataset): donor={mappings[0][0]}, acceptor={mappings[0][1]}, none={mappings[0][2]}")
    else:
        mappings = list(itertools.permutations(classes, 3))
        print("WARNING: label semantics are UNCLEAR (01 could not map labels to "
              "donor/acceptor/none). Reporting rule A under every possible assignment; "
              "the best one was picked by looking at test labels, so treat it as an "
              "optimistic upper bound, not a real baseline.")

    out = {"label_semantics": info["status"], "label_note": info["note"],
           "label_names": {str(c): names[c] for c in classes},
           "test_rows": {"full_test_n": int(len(te_y)), "model_subset_n": int(len(sub)),
                         "model_subset_max_test": args.max_test, "seed": args.seed,
                         "subset_matches_embedding_runs": same_rows}}

    # ----- rule A: single offset, GT->donor, AG->acceptor -----
    best = None
    for donor, acceptor, none in mappings:
        acc_te, f1_te = scan_rule_a(te_mat, te_y, donor, acceptor, none, classes)
        k = int(np.argmax(acc_te))
        if best is None or acc_te[k] > best["acc"][best["k"]]:
            best = {"map": (donor, acceptor, none), "acc": acc_te, "f1": f1_te, "k": k}
    donor, acceptor, none = best["map"]
    k_test = best["k"]
    acc_tr, _ = scan_rule_a(tr_mat, tr_y, donor, acceptor, none, classes)
    k_train = int(np.argmax(acc_tr))

    gt_te, ag_te = dinuc_hits(te_mat, "GT"), dinuc_hits(te_mat, "AG")
    pred_a = rule_a(gt_te, ag_te, k_test, donor, acceptor, none)
    pred_a_tr = rule_a(gt_te, ag_te, k_train, donor, acceptor, none)
    rule_a_out = {
        "description": "dinucleotide at offset k: GT -> donor, AG -> acceptor, else none; "
                       "every offset scanned",
        "mapping": {"donor": int(donor), "acceptor": int(acceptor), "none": int(none)},
        "best_offset": k_test,
        "best_offset_selected_on": "test accuracy (as requested; optimistic)",
        "full_test": full_metrics(te_y, pred_a, classes, names),
        "model_subset": full_metrics(te_y[sub], pred_a[sub], classes, names),
        "offset_selected_on_train": k_train,
        "full_test_at_train_offset": full_metrics(te_y, pred_a_tr, classes, names),
        "curve_full_test": {"offset": list(range(len(best["acc"]))),
                            "accuracy": [round(float(v), 4) for v in best["acc"]],
                            "macro_f1": [round(float(v), 4) for v in best["f1"]]},
    }
    if info["status"] != "inferred":
        rule_a_out["WARNING"] = ("label mapping unclear; mapping chosen by best test accuracy "
                                 "over all assignments")
    out["single_offset_rule"] = rule_a_out
    top = np.argsort(best["acc"])[::-1][:5]
    print("\nRule A (as specified: one offset, GT->donor, AG->acceptor)")
    print("  top offsets by test accuracy: " +
          ", ".join(f"{int(t)}: {best['acc'][t]:.4f}" for t in top))
    print(f"  best offset {k_test}: accuracy {rule_a_out['full_test']['accuracy']:.4f}, "
          f"macro-F1 {rule_a_out['full_test']['macro_f1']:.4f} (full test, n={len(te_y)})")
    print(f"  same offset on the model subset (n={len(sub)}): accuracy "
          f"{rule_a_out['model_subset']['accuracy']:.4f}, macro-F1 "
          f"{rule_a_out['model_subset']['macro_f1']:.4f}")
    print(f"  offset chosen on TRAIN instead: {k_train} -> test accuracy "
          f"{rule_a_out['full_test_at_train_offset']['accuracy']:.4f}"
          + ("  (same offset)" if k_train == k_test else ""))
    background = float(np.median(best["acc"]))
    print(f"  median accuracy over all offsets: {background:.4f} (= mostly 'predict none')")

    # ----- rule B: strand-aware, separate offsets chosen on train -----
    if info["status"] == "inferred":
        is_d, is_a = tr_y == donor, tr_y == acceptor
        ks, f1s = {}, {}
        for motif, target in (("GT", is_d), ("AC", is_d), ("AG", is_a), ("CT", is_a)):
            ks[motif], f1s[motif] = best_offset_one_vs_rest(dinuc_hits(tr_mat, motif), target)
        pred_b, conflicts = rule_b(te_mat, ks, donor, acceptor, none)
        out["strand_aware_rule"] = {
            "description": "donor if GT@k_GT or AC@k_AC (reverse complement); acceptor if "
                           "AG@k_AG or CT@k_CT; donor wins if both; else none",
            "offsets_selected_on_train": ks,
            "train_one_vs_rest_f1": {m: round(v, 4) for m, v in f1s.items()},
            "test_rows_matching_both": conflicts,
            "full_test": full_metrics(te_y, pred_b, classes, names),
            "model_subset": full_metrics(te_y[sub], pred_b[sub], classes, names),
        }
        rb = out["strand_aware_rule"]
        print("\nRule B (strand-aware, offsets picked on train)")
        print(f"  offsets: GT@{ks['GT']}, AC@{ks['AC']} -> donor; "
              f"AG@{ks['AG']}, CT@{ks['CT']} -> acceptor")
        print(f"  full test (n={len(te_y)}): accuracy {rb['full_test']['accuracy']:.4f}, "
              f"macro-F1 {rb['full_test']['macro_f1']:.4f}   ({conflicts} rows matched both)")
        print(f"  model subset (n={len(sub)}): accuracy {rb['model_subset']['accuracy']:.4f}, "
              f"macro-F1 {rb['model_subset']['macro_f1']:.4f}")
        for nm, v in rb["full_test"]["per_class"].items():
            print(f"    {nm:>9}: precision {v['precision']:.3f} recall {v['recall']:.3f} "
                  f"f1 {v['f1']:.3f}")
    else:
        out["strand_aware_rule"] = {"skipped": "label mapping unclear"}

    update_json(RESULTS_JSON, ["baseline"], out)
    print(f"\nSaved under 'baseline' in {RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
