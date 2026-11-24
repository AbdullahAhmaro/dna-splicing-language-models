#!/usr/bin/env python3
"""
Phase 1, step 9: what can the language model do that the GT/AG motif rule cannot?

    python scripts/09_motif_analysis.py

On the same 2,000 test windows the models were scored on, compare the
both-strands motif rule (04) with each model's validation-selected
pooling variant (07, predictions saved by 03):

  1. True splice sites, split into CANONICAL (the expected motif is at the
     expected offset on either strand: donor GT@301 or AC@298, acceptor AG@298
     or CT@301) and NON-CANONICAL (neither). The rule can only fire on the
     motif, so its recall on non-canonical sites is zero by construction. Any
     recall there has to come from something other than the motif.
  2. Non-sites (label none), split by whether a "decoy" motif sits at those
     same offsets. The rule calls every decoy a splice site, so it is wrong on
     all of them by construction.
  3. For canonical sites with one motif orientation: is the motif inside the
     token that covers base 300 (the centre token), split across its boundary,
     or entirely in a neighbouring token?
  4. Control: could whole-window base composition (a known dataset artefact:
     negative windows are more AT-rich) explain the decoy rejection? A
     logistic regression on mono- and dinucleotide frequencies, trained on
     motif-bearing training windows, gates the rule. Its threshold is set to
     keep the same share of canonical sites as each model. The threshold is
     set on the test sites, which favours the control.

Writes results/motif_analysis.md and results/motif_analysis.json. 06 copies
the tables into phase1_summary.md.
"""

import argparse
import math
import sys
from datetime import datetime, timezone

import numpy as np

from phase1_common import (EMB_DIR, MODELS, RESULTS_DIR, RESULTS_JSON, SEED,
                           load_label_info, load_splits, model_slug, read_json,
                           seq_matrix, stratified_subsample, write_json_atomic)

OUT_MD = RESULTS_DIR / "motif_analysis.md"
OUT_JSON = RESULTS_DIR / "motif_analysis.json"
CENTER_BASE = 300


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def cell(mask, correct):
    """Rate with k/n; small groups also get a 95% Wilson interval."""
    n = int(mask.sum())
    if n == 0:
        return "n/a", None
    k = int((correct & mask).sum())
    lo, hi = wilson(k, n)
    txt = f"{k / n:.3f} ({k}/{n})"
    if n < 100:
        txt += f" [{lo:.2f}–{hi:.2f}]"
    return txt, {"k": k, "n": n, "rate": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def composition(seqs):
    """Mono- (4) and dinucleotide (16) frequencies of each whole window."""
    m = seq_matrix(seqs)
    code = np.full(256, -1, dtype=np.int64)
    for i, c in enumerate("ACGT"):
        code[ord(c)] = i
    x = code[m]
    mono = np.stack([(x == i).mean(1) for i in range(4)], axis=1)
    pair = 4 * x[:, :-1] + x[:, 1:]
    di = np.stack([(pair == j).mean(1) for j in range(16)], axis=1)
    return np.hstack([mono, di])


def main():
    ap = argparse.ArgumentParser(description="Canonical vs non-canonical motif analysis.")
    ap.add_argument("--max-train", type=int, default=8000,
                    help="training rows for the composition control (same subsample as 02)")
    ap.add_argument("--max-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    res = read_json(RESULTS_JSON) or {}
    info = load_label_info()
    names = {int(k): v for k, v in info["names"].items()}
    D, A, N = info["donor_label"], info["acceptor_label"], info["none_label"]
    rb = res.get("baseline", {}).get("strand_aware_rule", {})
    sel = res.get("variant_selection", {}).get("models", {})
    if not rb or "offsets_selected_on_train" not in rb:
        print("No strand-aware baseline in phase1_results.json; run 04 first.", file=sys.stderr)
        return 1
    k = rb["offsets_selected_on_train"]       # same offsets the reported rule used

    ds = load_splits()
    y_all = np.array(ds["test"]["label"])
    idx = stratified_subsample(y_all, args.max_test, args.seed)
    rows = ds["test"].select(idx.tolist())
    S, y = rows["sequence"], np.array(rows["label"])

    def at(motif, off):
        return np.array([s[off:off + 2] == motif for s in S])

    d_plus, d_minus = at("GT", k["GT"]), at("AC", k["AC"])
    a_plus, a_minus = at("AG", k["AG"]), at("CT", k["CT"])
    d_motif, a_motif = d_plus | d_minus, a_plus | a_minus
    rule = np.full(len(y), N)
    rule[a_motif] = A
    rule[d_motif] = D                     # donor wins ties, exactly as in 04

    # ----- each model's validation-selected variant and its saved predictions -----
    preds, chosen, spans = {}, {}, {}
    for repo, short in MODELS.items():
        s = sel.get(repo)
        if not s:
            print(f"{short}: no validation-selected variant (run 07); skipping")
            continue
        v = s["selected_variant"]
        d = EMB_DIR / model_slug(repo)
        vd = d if v == "full_mean" else d / v
        pf = vd / "pred_test_mlp.npy"
        if not pf.exists():
            print(f"{short}: {pf} missing (rerun 03 for {v}); skipping")
            continue
        if not np.array_equal(np.load(vd / "idx_test.npy"), idx):
            print(f"{short}: test rows differ from --max-test {args.max_test}; skipping")
            continue
        p = np.load(pf)
        acc = float((p == y).mean())
        if s.get("final_test") and abs(acc - s["final_test"]["accuracy"]) > 1e-4:
            print(f"WARNING {short}: saved predictions give accuracy {acc:.4f}, but 07 reports "
                  f"{s['final_test']['accuracy']:.4f}")
        preds[short], chosen[short] = p, v
        ix = np.load(d / "center_tokens_index.npz")
        test_rows = np.flatnonzero(ix["split"] == "test")
        assert np.array_equal(ix["dataset_row"][test_rows], idx), "centre-token index misaligned"
        ptr, ts, te = ix["ptr"], ix["tok_start"], ix["tok_end"]
        sp = []
        for i in test_rows:
            a, b = ptr[i], ptr[i + 1]
            c = np.flatnonzero((ts[a:b] <= CENTER_BASE) & (te[a:b] > CENTER_BASE))[0]
            sp.append((int(ts[a + c]), int(te[a + c])))
        spans[short] = np.array(sp)
        print(f"{short}: using `{v}` (selected on validation), test accuracy {acc:.4f}")

    methods = {"motif rule": rule, **{f"{m}": p for m, p in preds.items()}}
    correct = {m: p == y for m, p in methods.items()}

    # ----- table 1: canonical vs non-canonical, plus decoys -----
    groups = [
        ("**True splice sites**", None),
        (f"Donor sites, canonical (GT@{k['GT']} or AC@{k['AC']})", (y == D) & d_motif),
        ("Donor sites, non-canonical (neither motif)", (y == D) & ~d_motif),
        (f"Acceptor sites, canonical (AG@{k['AG']} or CT@{k['CT']})", (y == A) & a_motif),
        ("Acceptor sites, non-canonical (neither motif)", (y == A) & ~a_motif),
        ("All splice sites, canonical", ((y == D) & d_motif) | ((y == A) & a_motif)),
        ("**All splice sites, non-canonical**", ((y == D) & ~d_motif) | ((y == A) & ~a_motif)),
        ("**Not splice sites** (correct = called *none*)", None),
        ("Non-sites with a decoy motif at those offsets", (y == N) & (d_motif | a_motif)),
        ("Non-sites with no motif at those offsets", (y == N) & ~(d_motif | a_motif)),
    ]
    head = ["group", "n", "motif rule"] + [f"{m} (`{chosen[m]}`)" for m in preds]
    t1 = ["| " + " | ".join(head) + " |", "|---|---:|" + "---:|" * (len(head) - 2)]
    t1_json = []
    for label, mask in groups:
        if mask is None:
            t1.append(f"| {label} | | " + " | " * (len(head) - 3) + " |")
            continue
        cells, js = [], {"group": label.strip("*"), "n": int(mask.sum())}
        for m in methods:
            txt, j = cell(mask, correct[m])
            cells.append(txt)
            js[m] = j
        t1.append(f"| {label} | {int(mask.sum())} | " + " | ".join(cells) + " |")
        t1_json.append(js)

    # ----- table 2: what the non-canonical sites are -----
    nc = np.flatnonzero(((y == D) & ~d_motif) | ((y == A) & ~a_motif))
    t2 = ["| row in test split | true label | bases 296–305 (centre = 300) | bases 298–299 | "
          "bases 301–302 | motif rule | " + " | ".join(preds) + " |",
          "|---:|---|---|---|---|---|" + "---|" * len(preds)]
    kinds = []
    for i in nc[:40]:
        s = S[i]
        kind = ("GC donor (+ strand, GC@301)" if s[301:303] == "GC" and y[i] == D else
                "GC donor (− strand, GC@298)" if s[298:300] == "GC" and y[i] == D else "other")
        kinds.append(kind)
        ctx = f"`{s[296:300]}·{s[300:306]}`"
        pr = " | ".join(("✓ " if preds[m][i] == y[i] else "✗ ") + names[int(preds[m][i])]
                        for m in preds)
        t2.append(f"| {int(idx[i])} | {names[int(y[i])]} | {ctx} | {s[298:300]} | {s[301:303]} | "
                  f"{'✓ ' if rule[i] == y[i] else '✗ '}{names[int(rule[i])]} | {pr} |")
    kinds_all = [("GC donor (+ strand, GC@301)" if S[i][301:303] == "GC" and y[i] == D else
                  "GC donor (− strand, GC@298)" if S[i][298:300] == "GC" and y[i] == D else
                  "other") for i in nc]
    kind_counts = {kd: kinds_all.count(kd) for kd in sorted(set(kinds_all))}

    # ----- table 3: motif inside the centre token or in a neighbour -----
    def own_motif(i):
        """(offset, orientation) of the site's own motif if exactly one orientation is present."""
        if y[i] == D:
            hits = [(k["GT"], "+ GT")] * bool(d_plus[i]) + [(k["AC"], "− AC")] * bool(d_minus[i])
        elif y[i] == A:
            hits = [(k["AG"], "+ AG")] * bool(a_plus[i]) + [(k["CT"], "− CT")] * bool(a_minus[i])
        else:
            return None
        return hits[0] if len(hits) == 1 else ("both" if hits else None)

    loc_order = ["inside the centre token", "split across the centre token's edge",
                 "entirely in a neighbouring token", "both orientations present"]
    t3 = ["| model (pooling) | where the site's motif sits | n | donor / acceptor | motif rule | "
          "model |", "|---|---|---:|---|---:|---:|"]
    t3_json = {}
    for m in preds:
        loc = np.array([""] * len(y), dtype=object)
        for i in np.flatnonzero((y == D) | (y == A)):
            om = own_motif(i)
            if om is None:
                continue
            if om == "both":
                loc[i] = loc_order[3]
                continue
            off = om[0]
            s0, e0 = spans[m][i]
            if s0 <= off and off + 2 <= e0:
                loc[i] = loc_order[0]
            elif off < e0 and off + 2 > s0:
                loc[i] = loc_order[1]
            else:
                loc[i] = loc_order[2]
        t3_json[m] = {}
        for L in loc_order:
            mask = loc == L
            if not mask.any():
                continue
            r_txt, r_j = cell(mask, correct["motif rule"])
            m_txt, m_j = cell(mask, correct[m])
            nd, na = int((mask & (y == D)).sum()), int((mask & (y == A)).sum())
            t3.append(f"| {m} (`{chosen[m]}`) | {L} | {int(mask.sum())} | {nd} / {na} | "
                      f"{r_txt} | {m_txt} |")
            t3_json[m][L] = {"n": int(mask.sum()), "donors": nd, "acceptors": na,
                             "motif_rule": r_j, "model": m_j}

    # ----- table 4: base-composition control for the decoy rows -----
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    tr_idx = stratified_subsample(np.array(ds["train"]["label"]), args.max_train, args.seed)
    tr = ds["train"].select(tr_idx.tolist())
    Str, ytr = tr["sequence"], np.array(tr["label"])

    def motif_mask(seqs):
        def has(mo, off):
            return np.array([q[off:off + 2] == mo for q in seqs])
        return has("GT", k["GT"]) | has("AC", k["AC"]) | has("AG", k["AG"]) | has("CT", k["CT"])

    m_tr, m_te = motif_mask(Str), d_motif | a_motif
    comp = LogisticRegression(max_iter=5000, class_weight="balanced").fit(
        composition(Str)[m_tr], (ytr[m_tr] != N).astype(int))
    p_site = comp.predict_proba(composition(S))[:, 1]
    can = ((y == D) & d_motif) | ((y == A) & a_motif)
    auc = float(roc_auc_score((y[m_te] != N).astype(int), p_site[m_te]))
    t4 = ["| model | model keeps (canonical sites right) | model rejects decoys | "
          "composition gate rejects decoys, at the same share of sites kept |",
          "|---|---:|---:|---:|"]
    t4_json = {"composition_auc_decoys_vs_canonical": round(auc, 4), "models": {}}
    decoy_mask = (y == N) & m_te
    for m in preds:
        keep = float((correct[m] & can).sum() / can.sum())
        thr = float(np.quantile(p_site[can], 1 - keep))
        comp_rej = float((p_site[decoy_mask] < thr).mean())
        mod_rej = float(correct[m][decoy_mask].mean())
        t4.append(f"| {m} (`{chosen[m]}`) | {keep:.3f} | **{mod_rej:.3f}** | {comp_rej:.3f} |")
        t4_json["models"][m] = {"canonical_kept": round(keep, 4), "model_rejects": round(mod_rej, 4),
                                "composition_rejects_at_same_keep": round(comp_rej, 4)}

    # ----- table 5: acceptor-token hypothesis (centre vs previous-token heads) -----
    hyp_variants = ["center_tok", "prev_tok", "center_plus_prev", "center_plus_prev_concat"]
    groups5 = [("donor, + strand: GT@301", (y == D) & d_plus & ~d_minus),
               ("donor, − strand: AC@298", (y == D) & d_minus & ~d_plus),
               ("**acceptor, + strand: AG@298**", (y == A) & a_plus & ~a_minus),
               ("acceptor, − strand: CT@301", (y == A) & a_minus & ~a_plus),
               ("all acceptors", y == A), ("all donors", y == D), ("non-sites (none)", y == N)]
    t5, t5_json = [], {}
    for repo, short in MODELS.items():
        d = EMB_DIR / model_slug(repo)
        avail = [v for v in hyp_variants if (d / v / "pred_test_mlp.npy").exists()
                 and np.array_equal(np.load(d / v / "idx_test.npy"), idx)]
        if not avail:
            continue
        pv = {v: np.load(d / v / "pred_test_mlp.npy") for v in avail}
        t5.append(f"\n**{short}**\n")
        t5.append("| group | n | " + " | ".join(f"`{v}`" for v in avail) + " |")
        t5.append("|---|---:|" + "---:|" * len(avail))
        t5_json[short] = {}
        for label, mask in groups5:
            nm = int(mask.sum())
            rec = {v: float((pv[v][mask] == y[mask]).mean()) if nm else float("nan") for v in avail}
            t5_json[short][label.strip("*")] = {"n": nm, **{v: round(r, 4) for v, r in rec.items()}}
            cells = []
            for v in avail:
                delta = "" if v == "center_tok" or "center_tok" not in rec else \
                    f" ({(rec[v] - rec['center_tok']) * 100:+.1f})"
                cells.append(f"{rec[v]:.3f}{delta}")
            t5.append(f"| {label} | {nm} | " + " | ".join(cells) + " |")
    sel_v = {MODELS[r]: v["selected_variant"] for r, v in sel.items() if r in MODELS}

    # ----- prose computed from the numbers -----
    nc_mask = ((y == D) & ~d_motif) | ((y == A) & ~a_motif)
    decoy = (y == N) & (d_motif | a_motif)
    lines = []
    lines.append(f"- **Non-canonical sites are rare here:** {int(nc_mask.sum())} of "
                 f"{int(((y == D) | (y == A)).sum())} true splice sites in these "
                 f"{len(y):,} test windows ({', '.join(f'{v} {kd}' for kd, v in kind_counts.items())}). "
                 "The rule gets every one of them wrong by construction.")
    for m in preds:
        kk = int((correct[m] & nc_mask).sum())
        lo, hi = wilson(kk, int(nc_mask.sum()))
        kd = int((correct[m] & decoy).sum())
        lines.append(f"- **{m}** (`{chosen[m]}`) recovered **{kk}/{int(nc_mask.sum())}** "
                     f"non-canonical sites (95% CI {lo:.2f}–{hi:.2f}) and correctly rejected "
                     f"**{kd}/{int(decoy.sum())}** decoy windows, where the rule is right on "
                     "none of either group.")
    lines.append("- The non-canonical group is too small to carry a strong claim on its own "
                 "(see the interval). The decoy row is the larger piece of evidence: those "
                 "windows contain exactly the motif the rule looks for, at exactly the right "
                 "offset, and only context can tell that they are not splice sites.")

    md = ["# Motif analysis: what the language model does that the GT/AG rule cannot\n",
          f"_Generated by `scripts/09_motif_analysis.py` on "
          f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Same {len(y):,} test "
          "windows as every other number in phase 1._\n",
          "**How to read this.** The motif rule (04) calls a window a donor if it has GT at "
          f"offset {k['GT']} (or AC at {k['AC']}, the same motif on the other strand), and an "
          f"acceptor if it has AG at {k['AG']} (or CT at {k['CT']}). Otherwise it says *none*. "
          "Each model column is the frozen language model with the MLP head and the pooling "
          "variant chosen on a validation split (07), never on test. Cells give the fraction "
          "correct, (correct / total), and a 95% interval for groups under 100. For splice "
          "sites, correct means the right site type (recall). For non-sites, it means calling "
          "them *none*.\n",
          "## 1. Canonical vs non-canonical sites, and decoys\n",
          *t1, "",
          *lines, "",
          "## 2. The non-canonical sites, one by one\n",
          "Context shows bases 296–299 · 300–305. ✓ = right site type, ✗ = wrong.\n",
          *t2, "",
          "GC-AG introns (a GC donor instead of GT) are the known minor class of human introns, "
          "about 1% of them. GC reads as GC on both strands (it is its own reverse complement), "
          "so a − strand GC donor shows GC at offset 298.\n",
          "## 3. Is the motif inside the centre token?\n",
          "The centre token is the token covering base 300. NT-v2 always cuts the window into "
          "6-mers from base 0, so its centre token is always bases 300–305. It contains "
          "GT@301 (+ strand donors) and CT@301 (− strand acceptors), but AG@298 and AC@298 "
          "sit in the previous token. DNABERT-2's tokens vary in length, so its centre token "
          "varies per window. Only canonical sites with exactly one motif orientation can be "
          "placed. Sites showing both orientations are listed separately.\n",
          *t3, "",
          "If a head only read the letters inside the centre token, its recall would collapse "
          "in the *neighbouring token* rows. Similar recall in both rows means the centre "
          "token's vector also encodes its neighbours (attention mixes context into every "
          "token).\n",
          "## 4. Control: is it just base composition?\n",
          "Negative windows in this dataset are more AT-rich than splice-site windows "
          "(see `dataset_format.md`), so a model could reject decoys from overall composition "
          "rather than splice signals. This control gives the motif rule a gate that only sees "
          "composition: mono- and dinucleotide frequencies of the whole window, logistic "
          "regression trained on the motif-bearing training windows. Its threshold is set to "
          "keep the same share of canonical sites as each model. That threshold is set on the "
          "test sites themselves, which favours the control. Composition alone separates "
          f"decoys from real sites with AUC {auc:.2f}.\n",
          *t4, "",
          "If composition explained the models' advantage, the last two columns would be "
          "close. A large gap means the models use sequence information beyond overall base "
          "composition.\n",
          "## 5. Acceptor-token hypothesis: does the previous token help acceptors?\n",
          "NT-v2's centre token covers bases 300–305, so a + strand donor's GT (301) is "
          "inside it, but a + strand acceptor's AG (298) is in the previous token. A gene "
          "scan in transcript orientation only ever sees these + strand geometries. So the "
          "row that tests the hypothesis is **+ strand acceptors (AG@298)**. Recall per "
          "variant, with the change vs `center_tok` in points in brackets. Same 2,000 test "
          "windows; heads trained by 03, previous-token variants built by "
          "`12_repool_center.py`. Chosen on validation (07): "
          + ", ".join(f"{k} `{v}`" for k, v in sel_v.items()) + ".\n",
          *t5, ""]
    OUT_MD.write_text("\n".join(md) + "\n")
    write_json_atomic(OUT_JSON, {
        "n_test_windows": int(len(y)), "rule_offsets": k,
        "models": {m: {"variant": chosen[m]} for m in preds},
        "groups": t1_json,
        "non_canonical_kinds": kind_counts,
        "non_canonical_rows": [int(idx[i]) for i in nc],
        "motif_location_vs_centre_token": t3_json,
        "composition_control": t4_json,
        "acceptor_token_hypothesis": t5_json,
    })
    print("\n".join(t1))
    print("\n".join(lines))
    print("\n".join(t3))
    print("\n".join(t4))
    print(f"\nWrote {OUT_MD} and {OUT_JSON}")
    return 0 if preds else 1


if __name__ == "__main__":
    sys.exit(main())
