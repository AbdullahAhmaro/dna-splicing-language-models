#!/usr/bin/env python3
"""
Phase 1, step 11: report on the whole-gene scans from 10_scan_gene.py.

    python scripts/11_scan_report.py

Metrics (each gene's canonical transcript span only; flanks and other
transcripts' exons outside it are excluded):
  * Top-k accuracy (the SpliceAI metric): a gene with k true donors gives the k
    highest-scoring donor positions; report the fraction within 2 bases of a
    true donor. Same for acceptors. Picks are de-duplicated (no two picks
    within 2 bases of each other) and each true site can be matched only once.
    DNABERT-2 scores whole tokens, so a DNABERT-2 pick is a token span and
    counts if a true site lies within 2 bases of that span.
  * AUROC over every scored position: true sites vs all other positions more
    than 2 bases from a true site of that type.
  * 95% bootstrap intervals: GENES are resampled with replacement (2,000
    resamples, seed 42), because sites within one gene share a sequence
    context and are not independent. The pooled top-k and AUROC are recomputed
    on each resample (AUROC exactly, from a gene-by-gene matrix of
    site-vs-position comparisons). Differences between two tracks use the same
    resamples (paired); a difference is called real when its 95% interval
    excludes 0.
  * False positives at thresholds 0.5 / 0.9 / 0.99, counted per scored
    position (per base for NT-v2, per token for DNABERT-2): at or above the
    threshold and more than 2 bases from every true site of that type, with
    where they are (canonical exon / intron, motif-bearing, on another
    transcript's site).
  * Matched recall (hard-negative head vs current head): the current head's
    threshold fixes how many true sites are found; the other head's threshold
    is lowered or raised until it finds as many, and their false positives are
    compared at that equal recall.
  * Composition control (as in 09): the GT/AG rule gated by a base-composition
    model, at the same number of true sites found.
  * Context check: true sites and 500 random positions per gene also scored in
    600-nt windows, for the long-context tracks.

Writes results/position_scan.json, results/position_scan_summary.md,
results/scan_<GENE>.png (one per gene), results/scan_tracks_ci.png and
results/scan_hardneg_fp.png.
"""

import importlib
import math
import sys
from datetime import datetime, timezone

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

from phase1_common import (DATA_DIR, MODELS, RESULTS_DIR, RESULTS_JSON, SEED,  # noqa: E402
                           read_json, stratified_subsample, write_json_atomic)

SCAN_DIR = DATA_DIR / "scan"
GENE_LIST = DATA_DIR / "genes" / "scan_gene_list.json"
OUT_JSON = RESULTS_DIR / "position_scan.json"
OUT_MD = RESULTS_DIR / "position_scan_summary.md"
TOL = 2
THRESHOLDS = (0.5, 0.9, 0.99)
CLASSES = {"donor": 2, "acceptor": 1}          # columns of the score track
COMP = str.maketrans("ACGT", "TGCA")
N_BOOT = 2000

SURFACE, TEXT, TEXT_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
COLOR = {"donor": "#2a78d6", "acceptor": "#eb6834"}
EXON, EXON_BAND = "#52514e", "#ecebe6"
HEAD_COLOR = {"original": "#8a8984", "hardneg": "#6250d6"}     # neutral = current, violet = new
HEAD_TINT = {"original": "#d6d5d0", "hardneg": "#c4bdf1"}      # lighter step = intronic share


# ---------------------------------------------------------------- metrics
def unit_table(scores, units, b0, b1):
    """One row per scoring unit in the gene body (a base for NT-v2, a token for
    DNABERT-2): its score and the first/last base it covers."""
    idx = np.arange(b0, b1)
    ok = np.isfinite(scores[b0:b1]) & (units[b0:b1] >= 0)
    idx, u, sc = idx[ok], units[b0:b1][ok], scores[b0:b1][ok]
    uniq, first, inv = np.unique(u, return_index=True, return_inverse=True)
    lo = np.full(len(uniq), np.iinfo(np.int64).max)
    hi = np.full(len(uniq), -1)
    np.minimum.at(lo, inv, idx)
    np.maximum.at(hi, inv, idx)
    return sc[first], lo, hi


def near_site(lo, hi, sites):
    """True for units within TOL bases of any of the given sites."""
    s = np.asarray(sites)
    if s.size == 0:
        return np.zeros(len(lo), bool)
    return ((s[None, :] >= lo[:, None] - TOL) & (s[None, :] <= hi[:, None] + TOL)).any(axis=1)


def top_k(score, lo, hi, sites, k):
    """Top-k accuracy with de-duplicated picks and one-to-one matching (tolerance TOL)."""
    order = np.argsort(-score, kind="stable")
    picks = []
    for j in order:
        if any(lo[j] - TOL <= ph and pl - TOL <= hi[j] for pl, ph in picks):
            continue
        picks.append((int(lo[j]), int(hi[j])))
        if len(picks) == k:
            break
    free = set(sites)
    hits = []
    for pl, ph in picks:
        near = [s for s in free if pl - TOL <= s <= ph + TOL]
        if near:
            free.discard(min(near, key=lambda s: min(abs(s - pl), abs(s - ph))))
            hits.append(True)
        else:
            hits.append(False)
    return sum(hits), picks, hits


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]                     # average ranks for ties
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ------------------------------------------------------- composition control
def composition_model():
    """The 09 gate: site-vs-none logistic regression on window composition,
    trained on motif-bearing training windows of the same 8,000-row subsample."""
    from sklearn.linear_model import LogisticRegression

    from phase1_common import load_label_info, load_splits

    m09 = importlib.import_module("09_motif_analysis")
    info = load_label_info()
    k = (read_json(RESULTS_JSON) or {})["baseline"]["strand_aware_rule"]["offsets_selected_on_train"]
    ds = load_splits()
    tr = ds["train"].select(stratified_subsample(np.array(ds["train"]["label"]), 8000,
                                                 SEED).tolist())
    S, y = tr["sequence"], np.array(tr["label"])

    def has(mo, off):
        return np.array([q[off:off + 2] == mo for q in S])
    motif = has("GT", k["GT"]) | has("AC", k["AC"]) | has("AG", k["AG"]) | has("CT", k["CT"])
    clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(
        m09.composition(S)[motif], (y[motif] != info["none_label"]).astype(int))
    return clf, m09.composition


def control_candidates(gene, seq, cls, comp_clf, comp_fn):
    """GT (donor) / AG (acceptor) positions in transcript orientation, each scored by the
    composition gate on its 600-nt window (read on the genome + strand, as in training)."""
    b0, b1 = gene["eval_region_idx"]
    if cls == "donor":
        cand = np.array([i for i in range(b0, b1) if seq[i + 1:i + 3] == "GT"])
        sites = gene["donors_idx"]
    else:
        cand = np.array([i for i in range(b0, b1) if seq[i - 2:i] == "AG"])
        sites = gene["acceptors_idx"]
    wins = [seq[i - 300:i + 300] for i in cand]
    if gene["strand"] == -1:
        wins = [w.translate(COMP)[::-1] for w in wins]
    p = comp_clf.predict_proba(comp_fn(wins))[:, 1]
    is_site = np.isin(cand, np.asarray(sites))      # a canonical motif sits exactly at its site
    return p, is_site


def control_fp_at(p, is_site, n_found):
    """False positives left by the gate when it keeps exactly n_found true sites
    (None if the rule cannot find that many: non-canonical sites)."""
    if n_found == 0:
        return 0
    sp = np.sort(p[is_site])[::-1]
    if n_found > len(sp):
        return None
    return int((p[~is_site] >= sp[n_found - 1]).sum())


# ---------------------------------------------------------------- figure
def binned_max(v, width):
    n = math.ceil(len(v) / width)
    pad = np.full(n * width, 0.0)
    pad[:len(v)] = np.nan_to_num(v, nan=0.0)
    return pad.reshape(n, width).max(axis=1)


def figure(sym, gene, tracks, topk_res, out_png):
    b0, b1 = gene["gene_body_idx"]                  # draw the whole gene body
    e0, e1 = gene["eval_region_idx"]                # canonical transcript span = scored
    length_kb = (b1 - b0) / 1000
    width = max(1, math.ceil((b1 - b0) / 1500))          # bases per drawn bin
    xb = (np.arange(math.ceil((b1 - b0) / width)) * width + width / 2) / 1000
    n = len(tracks)
    H = 1.75 + 2.45 * n
    fig = plt.figure(figsize=(12.5, H), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(1 + n, 1, height_ratios=[0.34] + [1] * n, hspace=0.36,
                          left=0.1, right=0.985, top=1 - 1.12 / H, bottom=0.6 / H)

    ax0 = fig.add_subplot(gs[0])
    ax0.set_facecolor(SURFACE)
    for yy, spans, col, lab in ((0.45, gene["exon_spans_idx"], EXON, "canonical"),
                                (-0.45, gene["alt_exon_spans_idx"], "#bdbbb4",
                                 "other transcripts")):
        if not spans:
            continue
        lo_ = min(a for a, _ in spans)
        hi_ = max(b for _, b in spans)
        ax0.plot([(lo_ - b0) / 1000, (hi_ - b0) / 1000], [yy, yy], color=col, linewidth=0.9,
                 zorder=1)
        for s_, e_ in spans:
            ax0.add_patch(Rectangle(((s_ - b0) / 1000, yy - 0.28), (e_ - s_ + 1) / 1000, 0.56,
                                    facecolor=col, edgecolor="none", zorder=2))
        ax0.text(-0.008, yy, lab, transform=ax0.get_yaxis_transform(), ha="right",
                 va="center", fontsize=8, color=TEXT_2)
    ax0.set_ylim(-0.85, 0.85)
    ax0.set_yticks([])
    ax0.set_xlim(0, length_kb)
    ax0.tick_params(axis="x", labelbottom=False, length=0)
    for side in ("top", "right", "left", "bottom"):
        ax0.spines[side].set_visible(False)

    for r, (short, sc) in enumerate(tracks.items(), start=1):
        ax = fig.add_subplot(gs[r], sharex=ax0)
        ax.set_facecolor(SURFACE)
        for s, e in gene["exon_spans_idx"]:
            ax.axvspan((s - b0) / 1000, (e + 1 - b0) / 1000, color=EXON_BAND, zorder=0,
                       linewidth=0)
        for cls, sign in (("donor", 1), ("acceptor", -1)):
            col = CLASSES[cls]
            y = binned_max(sc[b0:b1, col], width)
            ax.fill_between(xb, 0, sign * y, step="mid", color=COLOR[cls], alpha=0.85,
                            linewidth=0, zorder=2)
            sites = [(s - b0) / 1000 for s in gene[f"{cls}s_idx"]]
            ax.vlines(sites, 0, sign * 1.0, color=TEXT, linewidth=0.9, alpha=0.75, zorder=4)
            ax.scatter(sites, [sign * 1.12] * len(sites), marker="v" if sign > 0 else "^",
                       s=26, color=COLOR[cls], edgecolors=TEXT, linewidths=0.5, zorder=5,
                       clip_on=False)
        ax.axhline(0, color=MUTED, linewidth=0.6, zorder=3)
        for a_, b_ in ((b0, e0), (e1, b1)):
            if b_ - a_ > 50:
                ax.axvspan((a_ - b0) / 1000, (b_ - b0) / 1000, color=SURFACE, alpha=0.72,
                           zorder=4.5, linewidth=0)
                if r == 1 and (b_ - a_) > 0.25 * (b1 - b0):     # label only wide regions
                    ax.text((a_ + b_) / 2000 - b0 / 1000, 0.62,
                            "outside the canonical transcript:\nnot used in the metrics",
                            ha="center", va="center", fontsize=8, color=TEXT_2, zorder=6)
        ax.set_ylim(-1.2, 1.2)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticklabels(["1", "0.5", "0", "0.5", "1"], color=TEXT_2, fontsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=1)
        ax.tick_params(axis="both", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.text(-0.062, 0.76, "donor\nscore", transform=ax.transAxes, ha="center",
                va="center", fontsize=8.5, color=COLOR["donor"])
        ax.text(-0.062, 0.24, "acceptor\nscore", transform=ax.transAxes, ha="center",
                va="center", fontsize=8.5, color=COLOR["acceptor"])
        tk = topk_res[short]
        ax.set_title(f"{short}   top-k: donors {tk['donor']['hits']}/{tk['donor']['k']}, "
                     f"acceptors {tk['acceptor']['hits']}/{tk['acceptor']['k']}",
                     loc="left", fontsize=10, color=TEXT, pad=11)
        if r < n:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("position along the gene body, 5′ → 3′ (kb)", color=TEXT_2,
                          fontsize=9)
            ax.tick_params(axis="x", colors=TEXT_2, labelsize=8)

    fig.text(0.1, 1 - 0.14 / H, f"{sym}: splice-site scores along the whole gene",
             fontsize=13.5, color=TEXT, ha="left", va="top")
    fig.text(0.1, 1 - 0.44 / H,
             f"chr{gene['chrom']}:{gene['start']:,}–{gene['end']:,}, {gene['strand_symbol']} strand "
             f"({gene['assembly']}), canonical transcript {gene['transcript_id']}, on a held-out "
             "test chromosome.\nFrozen language model + per-position MLP head at every "
             f"position, drawn as the maximum per {width}-base bin. Grey bands: canonical exons.",
             fontsize=8.3, color=TEXT_2, ha="left", va="top")
    handles = [Patch(color=COLOR["donor"], label="donor score"),
               Patch(color=COLOR["acceptor"], label="acceptor score"),
               Line2D([], [], color=TEXT, alpha=0.75, linewidth=0.9, marker="v",
                      markerfacecolor=COLOR["donor"], markeredgecolor=TEXT, markersize=6,
                      label="true donor (last exon base)"),
               Line2D([], [], color=TEXT, alpha=0.75, linewidth=0.9, marker="^",
                      markerfacecolor=COLOR["acceptor"], markeredgecolor=TEXT, markersize=6,
                      label="true acceptor (first exon base)"),
               Patch(color=EXON_BAND, label="exon")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.095, 1 - 0.8 / H),
               ncols=5, frameon=False, fontsize=8.5, labelcolor=TEXT, handlelength=1.6)
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- per-gene records
def motif_flags(seq, cls):
    """Per base: does the canonical motif sit where a real site of this type has it
    (donor: GT right after the base; acceptor: AG right before it)?"""
    a = np.frombuffer(seq.encode(), np.uint8)
    f = np.zeros(len(a), bool)
    if cls == "donor":
        f[:-2] = (a[1:-1] == ord("G")) & (a[2:] == ord("T"))
    else:
        f[2:] = (a[:-2] == ord("A")) & (a[1:-1] == ord("G"))
    return f


def unit_any(flag, lo, hi):
    cs = np.concatenate([[0], np.cumsum(flag)])
    return (cs[hi + 1] - cs[lo]) > 0


def class_record(sc, un, gene, cls, exonic, motif):
    """Everything the pooled metrics and the bootstrap need, for one gene, track, class."""
    b0, b1 = gene["eval_region_idx"]
    col = CLASSES[cls]
    sites = list(gene[f"{cls}s_idx"])
    s_u, lo, hi = unit_table(sc[:, col], un, b0, b1)
    hits, picks, hl = top_k(s_u, lo, hi, sites, len(sites))
    near = near_site(lo, hi, sites)
    det = np.array([s_u[(lo <= s + TOL) & (hi >= s - TOL)].max(initial=-np.inf) for s in sites])
    span = np.arange(b0, b1)
    far = np.min(np.abs(span[:, None] - np.asarray(sites)[None, :]), axis=1) > TOL
    neg = sc[span[far], col]
    fp = ~near
    return {"hits": hits, "k": len(sites), "picks": picks, "hit_flags": hl,
            "pos": sc[sites, col].astype(float), "neg_sorted": np.sort(neg[np.isfinite(neg)]),
            "det": det, "fp_score": s_u[fp], "fp_exon": exonic[lo[fp]],
            "fp_alt": near_site(lo[fp], hi[fp], gene[f"alt_{cls}s_idx"]),
            "fp_motif": unit_any(motif, lo[fp], hi[fp]), "units": int(len(s_u))}


def fp_counts(recs, t):
    """Pooled false positives at threshold t over a list of per-gene class records."""
    sc = np.concatenate([r["fp_score"] for r in recs])
    m = sc >= t
    ex = np.concatenate([r["fp_exon"] for r in recs])[m]
    return {"threshold": float(t), "sites_found": int(sum((r["det"] >= t).sum() for r in recs)),
            "sites": int(sum(r["k"] for r in recs)), "false_positives": int(m.sum()),
            "in_exons": int(ex.sum()), "in_introns": int((~ex).sum()),
            "motif_bearing": int(np.concatenate([r["fp_motif"] for r in recs])[m].sum()),
            "on_other_transcript_sites": int(np.concatenate([r["fp_alt"] for r in recs])[m].sum())}


def threshold_for(recs, n_found):
    """Highest threshold at which the pooled records find at least n_found true sites."""
    if n_found <= 0:
        return math.inf
    det = np.sort(np.concatenate([r["det"] for r in recs]))[::-1]
    return float(det[min(n_found, len(det)) - 1])


# ---------------------------------------------------------------- bootstrap
def u_matrix(recs):
    """U[g, h] = sum over true sites of gene g and negatives of gene h of
    1[site > negative] + 0.5 * 1[tie]: the pooled AUROC of any multiset of genes
    follows from it exactly."""
    G = len(recs)
    U = np.zeros((G, G))
    npos = np.array([np.isfinite(r["pos"]).sum() for r in recs], float)
    nneg = np.array([len(r["neg_sorted"]) for r in recs], float)
    for g, rg in enumerate(recs):
        p = rg["pos"][np.isfinite(rg["pos"])]
        for h, rh in enumerate(recs):
            lt = np.searchsorted(rh["neg_sorted"], p, "left")
            le = np.searchsorted(rh["neg_sorted"], p, "right")
            U[g, h] = lt.sum() + 0.5 * (le - lt).sum()
    return U, npos, nneg


def boot_values(recs, W):
    """Point estimate and bootstrap replicates of pooled top-k and AUROC."""
    hits = np.array([r["hits"] for r in recs], float)
    k = np.array([r["k"] for r in recs], float)
    U, npos, nneg = u_matrix(recs)
    return {"top_k": (hits.sum() / k.sum(), (W @ hits) / (W @ k)),
            "auroc": (U.sum() / (npos.sum() * nneg.sum()),
                      np.einsum("bg,gh,bh->b", W, U, W) / ((W @ npos) * (W @ nneg)))}


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


# ---------------------------------------------------------------- tracks
VSHORT = {"center_tok": "centre token", "prev_tok": "previous token",
          "center_plus_prev": "centre+previous (mean)",
          "center_plus_prev_concat": "centre ++ previous token"}
CTX = {"long": "long-context scan", "window": "600-nt windows"}
HEAD_NAME = {"original": "current head", "hardneg": "hard-negative head",
             "finetuned": "fine-tuned backbone + head"}      # also the display order of the heads
# "finetuned": the track of kaggle/finetune_scan_genes.ipynb, brought in by 14_import_finetuned_tracks.py
HEAD_SUFFIX = {"hardneg": " · hard-negative head", "finetuned": " · fine-tuned backbone"}


def track_label(repo, variant, context, head):
    return (f"{MODELS.get(repo, repo)} · {VSHORT.get(variant, variant)} · {CTX[context]}"
            + HEAD_SUFFIX.get(head, ""))


def short_tag(t):
    return (f"{MODELS[t['repo']]}, {'600-nt windows' if t['context'] == 'window' else 'long context'}"
            + ("" if t["variant"] in (t["selected"],) else f", `{t['variant']}`")
            + f", {HEAD_NAME[t['head']]}")


# ---------------------------------------------------------------- figures
def forest_figure(T, P, order, out_png):
    """Pooled top-k and AUROC with 95% gene-bootstrap intervals, one row per track.
    AUROC is drawn as 1 - AUROC on an inverted log axis (labelled in AUROC), so the
    differences near 0.99 stay visible next to values near 0.8."""
    n = len(order)
    H = 1.5 + 0.55 * n
    fig, axes = plt.subplots(1, 2, figsize=(12.5, H), dpi=200, sharey=True,
                             gridspec_kw={"wspace": 0.08, "left": 0.30, "right": 0.975,
                                          "top": 1 - 0.92 / H, "bottom": 0.62 / H})
    fig.patch.set_facecolor(SURFACE)
    ys = np.arange(n)[::-1]

    def tr(metric, v):
        return np.asarray(v, float) if metric == "top_k" else 1 - np.asarray(v, float)
    for ax, metric, xlab in ((axes[0], "top_k", "top-k accuracy (higher is better)"),
                             (axes[1], "auroc", "AUROC, log scale of 1 − AUROC "
                                                "(right is better)")):
        ax.set_facecolor(SURFACE)
        for y, lb in zip(ys, order):
            for cls, dy in (("donor", 0.14), ("acceptor", -0.14)):
                v = P[lb][cls][metric]
                ax.plot(tr(metric, v["ci95"]), [y + dy] * 2, color=COLOR[cls], linewidth=2,
                        solid_capstyle="round", zorder=2)
                ax.scatter([tr(metric, v["value"])], [y + dy], s=34, color=COLOR[cls],
                           edgecolors=SURFACE, linewidths=1.6, zorder=3)
        ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
        ax.tick_params(axis="both", length=0, colors=TEXT_2, labelsize=8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.set_xlabel(xlab, color=TEXT_2, fontsize=8.8)
    axes[0].set_xlim(0, max(0.05, max(P[lb][c]["top_k"]["ci95"][1] for lb in order
                                      for c in CLASSES) * 1.08))
    lo = min(1 - P[lb][c]["auroc"]["ci95"][1] for lb in order for c in CLASSES)
    hi = max(1 - P[lb][c]["auroc"]["ci95"][0] for lb in order for c in CLASSES)
    ax = axes[1]
    ax.set_xscale("log")
    ax.set_xlim(hi * 1.4, max(lo / 1.4, 1e-4))          # inverted: better to the right
    ticks = [t for t in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
             if lo / 1.4 <= t <= hi * 1.4]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{1 - t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([lb.replace(" · hard-negative head", "\n· hard-negative head")
                             for lb in order], fontsize=8.3, color=TEXT)
    axes[0].set_ylim(-0.6, n - 0.4)
    G = len(next(iter(P.values()))["donor"]["per_gene_hits"])
    fig.text(0.012, 1 - 0.14 / H, "Whole-gene scans: pooled top-k and AUROC with 95% intervals",
             fontsize=12.5, color=TEXT, ha="left", va="top")
    fig.text(0.012, 1 - 0.47 / H,
             f"{G} genes on the held-out test chromosomes. Intervals come from resampling genes "
             f"({N_BOOT:,} bootstrap resamples); overlapping intervals are not shown to differ.",
             fontsize=8.3, color=TEXT_2, ha="left", va="top")
    handles = [Line2D([], [], color=COLOR[c], marker="o", linewidth=2, markersize=5.5,
                      markeredgecolor=SURFACE, label=f"{c}s") for c in CLASSES]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.975, 1 - 0.1 / H),
               ncols=2, frameon=False, fontsize=8.5, labelcolor=TEXT)
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def hardneg_fp_figure(HN, out_png):
    """False positives at matched recall, current vs hard-negative head, split exon / intron."""
    models = list(HN)
    fig, axes = plt.subplots(2, len(models), figsize=(6.4 * len(models), 7.4), dpi=200,
                             squeeze=False, gridspec_kw={"hspace": 0.55, "wspace": 0.22,
                                                         "left": 0.075, "right": 0.985,
                                                         "top": 0.79, "bottom": 0.09})
    fig.patch.set_facecolor(SURFACE)
    for j, m in enumerate(models):
        for i, cls in enumerate(CLASSES):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            rows = HN[m]["matched"][cls]
            x = np.arange(len(rows))
            for off, head in ((-0.2, "original"), (0.2, "hardneg")):
                ex = np.array([r[head]["in_exons"] for r in rows])
                intr = np.array([r[head]["in_introns"] for r in rows])
                ax.bar(x + off, ex, width=0.36, color=HEAD_COLOR[head], zorder=2,
                       edgecolor=SURFACE, linewidth=1)
                ax.bar(x + off, intr, bottom=ex, width=0.36, color=HEAD_TINT[head], zorder=2,
                       edgecolor=SURFACE, linewidth=1)
                for xi, tot in zip(x + off, ex + intr):
                    ax.text(xi, tot, f"{tot:,}", ha="center", va="bottom", fontsize=7.3,
                            color=TEXT_2)
            ax.set_xticks(x)
            ax.set_xticklabels([f"current head ≥ {r['threshold']}\n{r['sites_found']}/"
                                f"{r['sites']} sites found"
                                + (" †" if r["sites_found"] == r["sites"] else "") for r in rows],
                               fontsize=7.8, color=TEXT_2)
            ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
            ax.tick_params(axis="both", length=0, labelsize=7.8, colors=TEXT_2)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(MUTED)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
            ax.set_title(f"{m}: {cls} false positives", loc="left", fontsize=9.8, color=TEXT,
                         pad=8)
    fig.text(0.075, 0.975, "Hard negatives: false positives at matched recall",
             fontsize=13, color=TEXT, ha="left", va="top")
    fig.text(0.075, 0.935,
             "Each pair finds the same number of true sites: the current head at its threshold, "
             "the hard-negative head at the threshold that finds as many.\nDark segment = false "
             "positives inside canonical exons; light segment = inside introns. Pooled over all "
             "scanned genes. † = every site found: the match is set by the single hardest site.",
             fontsize=8.3, color=TEXT_2, ha="left", va="top")
    handles = [Patch(color=HEAD_COLOR["original"], label="current head, in exons"),
               Patch(color=HEAD_TINT["original"], label="current head, in introns"),
               Patch(color=HEAD_COLOR["hardneg"], label="hard-negative head, in exons"),
               Patch(color=HEAD_TINT["hardneg"], label="hard-negative head, in introns")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.07, 0.885), ncols=4,
               frameon=False, fontsize=8.3, labelcolor=TEXT)
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Report on the whole-gene scans.")
    ap.add_argument("--genes", nargs="*", default=None,
                    help="restrict to these genes (default: every gene in the scan list)")
    args = ap.parse_args()
    glist = read_json(GENE_LIST) or {"genes": []}
    order = [g["symbol"] for g in glist["genes"]]
    genes = [g for g in order if (SCAN_DIR / g / "gene.json").exists()
             and (args.genes is None or g in args.genes)]
    if not genes:
        print("No scans found; run 10_scan_gene.py first.", file=sys.stderr)
        return 1
    res = read_json(RESULTS_JSON) or {}
    sel = {"original": {r: v["selected_variant"] for r, v in
                        res.get("variant_selection", {}).get("models", {}).items()},
           "hardneg": {r: v["selected_variant"] for r, v in
                       res.get("variant_selection_hardneg", {}).get("models", {}).items()}}
    sel["finetuned"] = sel["original"]      # the fine-tune uses the variant 07 chose for the current head

    # tracks present for every gene
    metas = {s: read_json(SCAN_DIR / s / "scan_meta.json", default={}) or {} for s in genes}
    names = set.intersection(*[{n for n, m in metas[s].items()
                                if (SCAN_DIR / s / f"{n}.scores.npy").exists()
                                and m.get("model") in MODELS} for s in genes])
    skipped = sorted(set.union(*[set(metas[s]) for s in genes]) - names)
    T = {}
    for n in sorted(names):
        m = metas[genes[0]][n]
        head = m.get("head", "original")
        T[track_label(m["model"], m["variant"], m["context"], head)] = {
            "name": n, "repo": m["model"], "variant": m["variant"], "context": m["context"],
            "head": head, "selected": sel[head].get(m["model"])}
    for repo in MODELS:
        ctxs = {t["context"] for t in T.values() if t["repo"] == repo
                and t["head"] == "original" and t["variant"] == t["selected"]}
        best = "window" if "window" in ctxs else "long"
        for t in T.values():
            if t["repo"] == repo:
                t["role"] = ("headline" if t["variant"] == t["selected"] and t["context"] == best
                             else "reference")
    order_t = sorted(T, key=lambda lb: (list(MODELS).index(T[lb]["repo"]), T[lb]["role"] != "headline",
                                        list(HEAD_NAME).index(T[lb]["head"]),
                                        T[lb]["context"] != "window",
                                        T[lb]["variant"] != T[lb]["selected"]))
    headline = [lb for lb in order_t if T[lb]["role"] == "headline"]

    comp_clf, comp_fn = composition_model()
    recs = {lb: {c: [] for c in CLASSES} for lb in T}
    ctrl = {}
    out = {"genes": {}, "tracks": {lb: {k: v for k, v in t.items() if k != "name"}
                                   for lb, t in T.items()},
           "tracks_skipped_incomplete": skipped, "tolerance_bases": TOL,
           "thresholds": list(THRESHOLDS), "bootstrap": {"unit": "gene", "resamples": N_BOOT,
                                                         "seed": SEED, "interval": "95% percentile"}}
    bases = {"all": 0, "exon": 0}
    for sym in genes:
        gd = SCAN_DIR / sym
        gene = read_json(gd / "gene.json")
        seq = gene["seq_scanned"]
        b0, b1 = gene["eval_region_idx"]
        exonic = np.zeros(gene["length_scanned"], bool)
        for s, e in gene["exon_spans_idx"]:
            exonic[s:e + 1] = True
        bases["all"] += b1 - b0
        bases["exon"] += int(exonic[b0:b1].sum())
        motif = {c: motif_flags(seq, c) for c in CLASSES}
        ctrl[sym] = {c: control_candidates(gene, seq, c, comp_clf, comp_fn) for c in CLASSES}
        g0, g1 = gene["gene_body_idx"]
        gres = {"set": gene.get("gene_set", "original"), "strand": gene["strand_symbol"],
                "chrom": gene["chrom"], "start": gene["start"], "end": gene["end"],
                "length": b1 - b0, "gene_length": g1 - g0, "n_transcripts": gene["n_transcripts"],
                "exonic_fraction": float(exonic[b0:b1].mean()), "exons": len(gene["exons"]),
                "donors": len(gene["donors_idx"]), "acceptors": len(gene["acceptors_idx"]),
                "donor_motifs": gene["donor_motifs"], "acceptor_motifs": gene["acceptor_motifs"],
                "transcript_id": gene["transcript_id"], "gene_id": gene["gene_id"],
                "why_chosen": gene["why_chosen"], "tracks": {}}
        panels, topk_res = {}, {}
        for lb in order_t:
            t = T[lb]
            sc = np.load(gd / f"{t['name']}.scores.npy")
            un = np.load(gd / f"{t['name']}.units.npy")
            if t["role"] == "headline":
                panels[lb] = sc
            topk_res[lb] = {}
            gt = {"scan": metas[sym][t["name"]], "classes": {}}
            for cls in CLASSES:
                r = class_record(sc, un, gene, cls, exonic, motif[cls])
                recs[lb][cls].append(r)
                topk_res[lb][cls] = {"hits": r["hits"], "k": r["k"]}
                gt["classes"][cls] = {
                    "top_k": {"k": r["k"], "hits": r["hits"],
                              "picks": [{"span_idx": list(p), "hit": h}
                                        for p, h in zip(r["picks"], r["hit_flags"])]},
                    "score_at_true_sites": [round(float(v), 4) for v in r["pos"]],
                    "false_positives": [fp_counts([r], th) for th in THRESHOLDS]}
            gres["tracks"][lb] = gt
        png = RESULTS_DIR / f"scan_{sym}.png"
        figure(sym, gene, panels, topk_res, png)
        gres["figure"] = str(png.relative_to(RESULTS_DIR.parent))
        out["genes"][sym] = gres
        print(f"{sym}: " + "; ".join(f"{T[lb]['context'][:4]}/{T[lb]['head'][:4]} "
                                     f"{MODELS[T[lb]['repo']][:5]} d {v['donor']['hits']}/{v['donor']['k']} "
                                     f"a {v['acceptor']['hits']}/{v['acceptor']['k']}"
                                     for lb, v in topk_res.items() if T[lb]["role"] == "headline"))

    # ---- pooled metrics with gene-bootstrap intervals
    G = len(genes)
    W = np.random.default_rng(SEED).multinomial(G, [1 / G] * G, size=N_BOOT).astype(float)
    BV, P = {}, {}
    for lb in order_t:
        P[lb], BV[lb] = {}, {}
        for cls in CLASSES:
            rl = recs[lb][cls]
            bv = boot_values(rl, W)
            BV[lb][cls] = bv
            hits, k = sum(r["hits"] for r in rl), sum(r["k"] for r in rl)
            P[lb][cls] = {
                "top_k": {"hits": hits, "k": k, "value": bv["top_k"][0], "ci95": ci(bv["top_k"][1])},
                "auroc": {"value": bv["auroc"][0], "ci95": ci(bv["auroc"][1])},
                "per_gene_hits": {s: [r["hits"], r["k"]] for s, r in zip(genes, rl)},
                "false_positives": [fp_counts(rl, th) for th in THRESHOLDS],
            }
            sets = {}
            for s, r in zip(genes, rl):
                sets.setdefault(out["genes"][s]["set"], [0, 0])
                sets[out["genes"][s]["set"]][0] += r["hits"]
                sets[out["genes"][s]["set"]][1] += r["k"]
            P[lb][cls]["top_k_by_gene_set"] = sets
            if T[lb]["role"] == "headline":
                for f in P[lb][cls]["false_positives"]:
                    tot, un = 0, []
                    for s, r in zip(genes, rl):
                        p, is_site = ctrl[s][cls]
                        c = control_fp_at(p, is_site, int((r["det"] >= f["threshold"]).sum()))
                        if c is None:
                            un.append(s)
                        else:
                            tot += c
                    f["control_false_positives"] = tot
                    f["control_unreachable_genes"] = un
                P[lb][cls]["rule_alone"] = {
                    "false_positives": int(sum((~ctrl[s][cls][1]).sum() for s in genes)),
                    "true_sites_found": int(sum(ctrl[s][cls][1].sum() for s in genes))}
    out["pooled"] = P

    # ---- paired comparisons
    def find(repo, variant, context, head):
        return next((lb for lb, t in T.items() if t["repo"] == repo and t["variant"] == variant
                     and t["context"] == context and t["head"] == head), None)
    NT, DB = list(MODELS)
    so, sh = sel["original"], sel["hardneg"]
    specs = [
        ("context", "NT-v2: 600-nt windows vs long context (same head)",
         find(NT, so.get(NT), "long", "original"), find(NT, so.get(NT), "window", "original")),
        ("head", "NT-v2: centre ++ previous vs centre token (long context)",
         find(NT, "center_tok", "long", "original"), find(NT, so.get(NT), "long", "original")),
        ("head", "DNABERT-2: centre ++ previous vs centre token (long context)",
         find(DB, "center_tok", "long", "original"), find(DB, so.get(DB), "long", "original")),
        ("model", "NT-v2 (600-nt windows) vs DNABERT-2 (long context), current heads",
         find(DB, so.get(DB), "long", "original"), find(NT, so.get(NT), "window", "original")),
        ("hardneg", "NT-v2 (600-nt windows): hard-negative head vs current head",
         find(NT, so.get(NT), "window", "original"), find(NT, sh.get(NT), "window", "hardneg")),
        ("hardneg", "DNABERT-2 (long context): hard-negative head vs current head",
         find(DB, so.get(DB), "long", "original"), find(DB, sh.get(DB), "long", "hardneg")),
        ("finetune", "NT-v2 (600-nt windows): fine-tuned backbone vs frozen backbone, current head",
         find(NT, so.get(NT), "window", "original"), find(NT, so.get(NT), "window", "finetuned")),
        ("finetune", "NT-v2 (600-nt windows): fine-tuned backbone vs frozen backbone with the "
                     "hard-negative head (reference)",
         find(NT, sh.get(NT), "window", "hardneg"), find(NT, so.get(NT), "window", "finetuned")),
    ]
    comps = []
    for kind, title, a, b in specs:
        if not a or not b:
            continue
        c = {"kind": kind, "title": title, "baseline": a, "new": b, "delta": {}}
        for cls in CLASSES:
            for metric in ("top_k", "auroc"):
                d0 = BV[b][cls][metric][0] - BV[a][cls][metric][0]
                db = BV[b][cls][metric][1] - BV[a][cls][metric][1]
                lo, hi = ci(db)
                c["delta"][f"{metric}_{cls}"] = {
                    "value": float(d0), "ci95": [lo, hi],
                    "verdict": "higher" if lo > 0 else "lower" if hi < 0 else "not distinguishable",
                    "share_of_resamples_above_0": float((db > 0).mean())}
        comps.append(c)
    out["comparisons"] = comps

    # ---- hard negatives: matched recall, fixed thresholds, where the false positives are
    HN = {}
    for repo in MODELS:
        ctxh = "window" if find(repo, so.get(repo), "window", "original") and \
            find(repo, sh.get(repo), "window", "hardneg") else "long"
        a, b = find(repo, so.get(repo), ctxh, "original"), find(repo, sh.get(repo), ctxh, "hardneg")
        if not a or not b:
            continue
        h = {"baseline": a, "new": b, "context": ctxh, "matched": {}, "fixed": {}}
        for cls in CLASSES:
            ra, rb = recs[a][cls], recs[b][cls]
            h["fixed"][cls] = [{"threshold": th, "original": fp_counts(ra, th),
                                "hardneg": fp_counts(rb, th)} for th in THRESHOLDS]
            rows = []
            for th in THRESHOLDS:
                fa = fp_counts(ra, th)
                tb = threshold_for(rb, fa["sites_found"])
                fb = fp_counts(rb, tb) if math.isfinite(tb) else \
                    {k: 0 for k in fa} | {"threshold": tb, "sites": fa["sites"]}
                rows.append({"threshold": th, "sites_found": fa["sites_found"], "sites": fa["sites"],
                             "hardneg_threshold": tb, "original": fa, "hardneg": fb})
            h["matched"][cls] = rows
        HN[MODELS[repo]] = h
    out["hardneg"] = HN
    out["positives"] = {
        "genes": G, "sites": sum(P[order_t[0]][c]["top_k"]["k"] for c in CLASSES),
        "donors": P[order_t[0]]["donor"]["top_k"]["k"],
        "acceptors": P[order_t[0]]["acceptor"]["top_k"]["k"],
        "bases": bases["all"], "exon_bases": bases["exon"],
        "exonic_fraction": bases["exon"] / bases["all"],
        "one_site_per_bases": round(bases["all"] / sum(P[order_t[0]][c]["top_k"]["k"] for c in CLASSES)),
        "one_donor_per_bases": round(bases["all"] / P[order_t[0]]["donor"]["top_k"]["k"])}

    # ---- context check (tracks that have it for every gene)
    out["context_check"] = {}
    for lb in order_t:
        files = [SCAN_DIR / s / f"{T[lb]['name']}.context_check.npz" for s in genes]
        if not all(f.exists() for f in files):
            continue
        acc = {c: {k: [] for k in ("ls", "lr", "ws", "wr", "la", "wa")} for c in CLASSES}
        for f in files:
            z = np.load(f)
            nd, na = int(z["n_donors"]), int(z["n_acceptors"])
            for cls, col in CLASSES.items():
                sl = slice(0, nd) if cls == "donor" else slice(nd, nd + na)
                rnd = slice(nd + na, None)
                a_ = acc[cls]
                a_["ls"] += z["long"][sl, col].tolist()
                a_["ws"] += z["window"][sl, col].tolist()
                a_["lr"] += z["long"][rnd, col].tolist()
                a_["wr"] += z["window"][rnd, col].tolist()
                a_["la"] += z["long"][:, col].tolist()
                a_["wa"] += z["window"][:, col].tolist()
        out["context_check"][lb] = {cls: {
            "auroc_long": auroc(a_["ls"], a_["lr"]), "auroc_window": auroc(a_["ws"], a_["wr"]),
            "mean_site_long": float(np.nanmean(a_["ls"])), "mean_site_window": float(np.mean(a_["ws"])),
            "corr_long_window": float(np.corrcoef(a_["la"], a_["wa"])[0, 1]),
            "n_sites": len(a_["ls"]), "n_random": len(a_["lr"])} for cls, a_ in acc.items()}

    forest_figure(T, P, order_t, RESULTS_DIR / "scan_tracks_ci.png")
    if HN:
        hardneg_fp_figure(HN, RESULTS_DIR / "scan_hardneg_fp.png")
    out["order"] = order_t
    out["headline"] = headline
    out["gene_list"] = {k: v for k, v in glist.items() if k != "genes"}
    write_json_atomic(OUT_JSON, out)
    write_summary(out, res)
    print(f"Wrote {OUT_JSON} and {OUT_MD}")
    return 0


# ---------------------------------------------------------------- summary
def fmt_ci(v, c, d=2):
    return f"{v:.{d}f} [{c[0]:.{d}f}, {c[1]:.{d}f}]"


def fmt_delta(x, d=2):
    s = f"{x['value']:+.{d}f} [{x['ci95'][0]:+.{d}f}, {x['ci95'][1]:+.{d}f}]"
    return f"**{s}**" if x["verdict"] != "not distinguishable" else s


def ratio(a, b):
    return b / a if a else (math.inf if b else 1.0)


def fp_verdict(row):
    """Did the exonic false positives at matched recall go down, stay, or move?
    'down' = at most 0.75x, 'up' = at least 1.33x the current head's count."""
    a, b = row["original"], row["hardneg"]
    if a["false_positives"] < 30:
        return "too few to judge"
    r_ex, r_in = ratio(a["in_exons"], b["in_exons"]), ratio(a["in_introns"], b["in_introns"])
    down, up = 0.75, 1 / 0.75
    if r_ex <= down:
        return "moved into introns" if r_in >= up else "improved"
    if r_ex >= up:
        return "worse"
    if r_in >= up:
        return "exons same, introns worse"
    if r_in <= down:
        return "exons same, introns better"
    return "stayed the same"


def fmt_ratio(r):
    return "∞" if r == math.inf else f"×{r:.2f}"


def write_summary(out, res):
    G, P, pos, T, C = out["genes"], out["pooled"], out["positives"], out["tracks"], out["comparisons"]
    order, headline, HN = out["order"], out["headline"], out["hardneg"]
    hnj = read_json(RESULTS_DIR / "hard_negatives.json", default={}) or {}
    vs_o = res.get("variant_selection", {}).get("models", {})
    vs_h = res.get("variant_selection_hardneg", {}).get("models", {})
    pv, hh = res.get("pooling_variants", {}), res.get("hardneg_heads", {})
    n_add = sum(g["set"] == "added" for g in G.values())
    n_orig = len(G) - n_add

    def tag(lb):
        t = T[lb]
        return (f"{MODELS[t['repo']]}, {'600-nt windows' if t['context'] == 'window' else 'long context'}"
                + ("" if t["variant"] == t["selected"] else f", `{t['variant']}`")
                + f", {HEAD_NAME[t['head']]}")

    def describe(c):
        real = [(k, v) for k, v in c["delta"].items() if v["verdict"] != "not distinguishable"]
        same = [k for k, v in c["delta"].items() if v["verdict"] == "not distinguishable"]
        nice = {"top_k_donor": "top-k donors", "top_k_acceptor": "top-k acceptors",
                "auroc_donor": "AUROC donors", "auroc_acceptor": "AUROC acceptors"}
        parts = [f"{nice[k]} {v['verdict']} ({v['value']:+.{3 if 'auroc' in k else 2}f})"
                 for k, v in real]
        txt = ("; ".join(parts) + ". " if parts else "no difference is distinguishable from "
               "gene-to-gene noise. ")
        if parts and same:
            txt += "Not distinguishable: " + ", ".join(nice[k] for k in same) + "."
        return txt

    md = ["# Position scan: the per-position splice classifier applied along whole genes\n",
          f"_Generated by `scripts/11_scan_report.py` on "
          f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from the scans written "
          "by `scripts/10_scan_gene.py`._\n",
          "**What was done.** A training window labelled \"centre is a donor\" is a label for "
          "one position (base 300), and the pooling chosen on validation reads only the token "
          "covering that base and the token just before it. So the trained head is already a "
          f"per-position classifier. Here the frozen DNA model is run along {len(G)} whole genes "
          "on the held-out test chromosomes, and that head is applied at every position to give "
          "a donor score and an acceptor score per base. This round adds (1) "
          f"{n_add} more genes, for statistical power, with gene-bootstrap confidence intervals "
          "on every pooled number, and (2) a head trained with hard negatives (exon interiors, "
          "intron interiors and GT/AG decoys from training-chromosome genes).\n"]

    def bottom_lines():
        out_ = {}
        for repo, short in MODELS.items():
            h = HN.get(short)
            vh = vs_h.get(repo)
            if not h or not vh:
                continue
            v = vh["selected_variant"]
            tc = pv.get(repo, {}).get(vs_o.get(repo, {}).get("selected_variant"), {}).get("mlp")
            th = hh.get(repo, {}).get(v, {}).get("mlp")
            c = next((c for c in C if c["baseline"] == h["baseline"] and c["new"] == h["new"]), None)
            if not (tc and th and c):
                continue
            verd = [x["verdict"] for x in c["delta"].values()]
            scan = ("better" if "higher" in verd and "lower" not in verd else
                    "worse" if "lower" in verd and "higher" not in verd else
                    "mixed" if "higher" in verd else "not distinguishable")
            fpv = [fp_verdict(r) for cls in CLASSES for r in h["matched"][cls]]
            n_imp = sum(x == "improved" for x in fpv)
            n_bad = sum(x in ("worse", "moved into introns", "exons same, introns worse")
                        for x in fpv)
            tots = [ratio(r["original"]["false_positives"], r["hardneg"]["false_positives"])
                    for cls in CLASSES for r in h["matched"][cls]
                    if r["sites_found"] < r["sites"] and r["original"]["false_positives"] >= 30]
            med = float(np.median(tots)) if tots else 1.0
            better = "Modestly better" if med > 2 / 3 else "Better"
            rc = {k: (tc["per_class"][k]["recall"], th["per_class"][k]["recall"])
                  for k in ("donor", "acceptor", "none")}
            worse_bench = th["accuracy"] < tc["accuracy"]
            bench = (f"On the dataset's own test windows it is slightly worse: accuracy "
                     f"{tc['accuracy']:.4f} → {th['accuracy']:.4f} "
                     f"({(th['accuracy'] - tc['accuracy']) * 100:+.1f} points); it trades site "
                     f"recall (donors {rc['donor'][0]:.3f} → {rc['donor'][1]:.3f}, acceptors "
                     f"{rc['acceptor'][0]:.3f} → {rc['acceptor'][1]:.3f}) for none recall "
                     f"({rc['none'][0]:.3f} → {rc['none'][1]:.3f}), as expected when about half "
                     "the training rows are negatives."
                     if worse_bench else
                     f"On the dataset's own test windows it is not worse: accuracy "
                     f"{tc['accuracy']:.4f} → {th['accuracy']:.4f}.")
            genes_txt = {"better": "better", "worse": "worse", "mixed": "mixed",
                         "not distinguishable": "not distinguishable from the current head"}[scan]
            tail_ = ", slightly worse on the benchmark windows" if worse_bench else ""
            if scan == "worse" or n_bad > n_imp:
                overall = "**Worse overall: a negative result.**"
            elif scan == "better" and n_bad == 0 and n_imp > 0:
                overall = f"**{better} at the task it targets{tail_}.**"
            elif scan in ("better", "not distinguishable") and n_imp > n_bad:
                overall = (f"**{'Mostly, modestly' if med > 2 / 3 else 'Mostly'} better at the "
                           f"task it targets{tail_}, with exceptions (below).**"
                           if scan == "better" else
                           "**Fewer exonic false positives, but no threshold-free gain along "
                           f"the genes{tail_}.**")
            else:
                overall = "**No clear gain.**"
            out_[short] = (overall, f"{bench} Threshold-free, along the genes: {genes_txt}. "
                                    f"{describe(c)} At matched recall the exonic false positives "
                                    f"are *improved* at {n_imp} of {len(fpv)} operating points and "
                                    f"worse or displaced at {n_bad}; total false positives at "
                                    f"equal recall change by a median ×{med:.2f} (operating points "
                                    "below full recall with ≥ 30 false positives).")
        return out_
    bottom = bottom_lines()
    ex_kb = pos["exon_bases"] / 1000
    in_kb = (pos["bases"] - pos["exon_bases"]) / 1000

    def density(f):
        di = f["in_introns"] / in_kb
        return (f["in_exons"] / ex_kb) / di if di else math.inf

    def plain(m):
        """One plain sentence per model: did the exonic false-positive problem improve?"""
        h = HN[m]
        parts, vs = [], []
        for cls in CLASSES:
            r = next(x for x in h["matched"][cls] if x["threshold"] == 0.9)
            a, b = r["original"], r["hardneg"]
            v = fp_verdict(r)
            vs.append(v)
            parts.append(f"{cls}s **{v}** (exonic ×{ratio(a['in_exons'], b['in_exons']):.2f}, "
                         f"intronic ×{ratio(a['in_introns'], b['in_introns']):.2f}, total "
                         f"×{ratio(a['false_positives'], b['false_positives']):.2f}; exon:intron "
                         f"density {density(a):.1f} → {density(b):.1f})")
        if all(v == "improved" for v in vs):
            head_ = "Improved"
        elif any(v == "moved into introns" for v in vs):
            head_ = "Moved elsewhere (into introns)"
        elif any(v == "improved" for v in vs):
            head_ = "Partly improved"
        elif any(v in ("worse", "exons same, introns worse") for v in vs):
            head_ = "Not improved"
        else:
            head_ = "Stayed the same"
        moved = any(v in ("moved into introns", "exons same, introns worse") for v in vs)
        return (f"**{head_}**" + ("" if moved else ", and the false positives did not move into "
                                  "introns") + ": at equal recall (the current head's 0.9 "
                "operating point), " + "; ".join(parts) + ".")

    def premise(m):
        h = HN[m]
        out_ = []
        for cls in CLASSES:
            a = next(x for x in h["matched"][cls] if x["threshold"] == 0.9)["original"]
            if a["false_positives"]:
                out_.append(f"{cls}s {a['in_exons'] / a['false_positives']:.0%} exonic, "
                            f"{density(a):.1f}× the intronic density")
        return "; ".join(out_)

    # ---- short answers
    md.append("## Short answers\n")
    md.append(f"- **Sites now:** {pos['donors']} donors and {pos['acceptors']} acceptors in "
              f"{len(G)} genes (the first scans had 21 + 21 in {n_orig}). Every pooled top-k and "
              "AUROC below carries a 95% interval from resampling genes.")
    for c in C:
        md.append(f"- **{c['title']}:** {describe(c)}")
    for m, h in HN.items():
        r9 = {cls: next(r for r in h["matched"][cls] if r["threshold"] == 0.9) for cls in CLASSES}
        bits = []
        for cls, r in r9.items():
            a, b = r["original"], r["hardneg"]
            bits.append(f"{cls}s: exonic false positives {a['in_exons']:,} → {b['in_exons']:,}, "
                        f"intronic {a['in_introns']:,} → {b['in_introns']:,} "
                        f"(**{fp_verdict(r)}**)")
        md.append(f"- **Hard negatives, {m} ({'600-nt windows' if h['context'] == 'window' else 'long context'}). "
                  "Exonic false-positive problem:** " + plain(m) + " "
                  + ("Overall: " + bottom[m][0] + " " if m in bottom else "")
                  + "Details in section 2.")
    md.append("")

    md.append("> **Read this first: class imbalance.** Both heads were trained on windows: the "
              "current head on class-balanced ones (about one third each), the hard-negative head "
              "on the same windows plus hard negatives, so about half \"none\". Along these genes, "
              f"true sites are about **1 in {pos['one_site_per_bases']:,} positions** "
              f"({pos['sites']} sites in {pos['bases']:,} scored bases), about 1 in "
              f"{pos['one_donor_per_bases']:,} for each type. Nothing was re-calibrated for that. "
              "A per-window accuracy of 0.94 does not carry over: a small per-position "
              "false-positive rate times thousands of negative positions outnumbers the true "
              "sites. Top-k and AUROC need no threshold, so they are the fairest comparisons.\n")

    # ---- genes
    md.append("## Genes\n")
    gl = out.get("gene_list", {})
    md.append(f"All {len(G)} genes are on chr20 and chr21, the dataset's held-out test chromosomes "
              "(checked against the dataset's own `name` column: train = chr1–19, X, Y; test = "
              "chr20, chr21). The first three were picked by hand in round 4. The other "
              f"{n_add} were picked by a fixed, seeded rule from an Ensembl listing "
              f"(release {', '.join(map(str, gl.get('ensembl_release') or ['?']))}): "
              f"{gl.get('rule_text', '').replace('>=', '≥').replace(' x ', ' × ')}. "
              f"{gl.get('n_candidates', '?')} genes qualified. "
              "Sequences and transcripts are cached in `data/genes/`; the list and the rule in "
              "`data/genes/scan_gene_list.json`. **Metrics use only the span of each gene's "
              "canonical transcript**; real sites of other transcripts inside it are counted "
              "separately in the false-positive tables.\n")
    md.append("| gene | set | location (GRCh38) | strand | scored span | exons (share of span) | "
              "donors | acceptors | non-canonical sites |")
    md.append("|---|---|---|:---:|---:|---:|---:|---:|---|")
    for sym, g in G.items():
        nc = [f"{cls} {mo}" for cls in ("donor", "acceptor") for mo in g[f"{cls}_motifs"]
              if mo != {"donor": "GT", "acceptor": "AG"}[cls]]
        md.append(f"| **{sym}** | {g['set']} | chr{g['chrom']}:{g['start']:,}–{g['end']:,} | "
                  f"**{g['strand']}** | {g['length'] / 1000:.1f} kb | {g['exons']} "
                  f"({g['exonic_fraction']:.0%}) | {g['donors']} | {g['acceptors']} | "
                  f"{', '.join(nc) or '–'} |")
    n_minus = sum(g["strand"] == "-" for g in G.values())
    md.append(f"\n{n_minus} of the {len(G)} genes are on the − strand and are scanned "
              "reverse-complemented (5′→3′ of the transcript). Exons cover "
              f"{pos['exonic_fraction']:.0%} of the {pos['bases']:,} scored bases.\n")

    # ---- tracks
    md.append("## Tracks\n")
    md.append("| track | head | how the gene is scanned |")
    md.append("|---|---|---|")
    g0 = next(iter(G.values()))
    for lb in order:
        sc = g0["tracks"][lb]["scan"]
        if sc.get("context") == "window":
            how = ("a 600-nt window centred on every base of the canonical span, one forward pass "
                   "per base: the training geometry exactly"
                   + ("; the hard-negative head is applied to the same cached token vectors"
                      if T[lb]["head"] == "hardneg" else
                      "; run on a GPU by `kaggle/finetune_scan_genes.ipynb` with the fine-tuned "
                      "backbone" if T[lb]["head"] == "finetuned" else ""))
        else:
            how = (f"{sc.get('phases')} passes shifted by 0–{sc.get('phases', 1) - 1} bases, so "
                   "every base starts a 6-mer token once; score = the token starting at that base"
                   if sc.get("assign") == "start" else
                   "1 pass; score = the BPE token covering that base (tokens average ~5 bases)")
            how = (f"tiles of {sc.get('tile_bases'):,} bases, {sc.get('overlap_bases'):,} overlap, "
                   f"{sc.get('edge_discard_bases')} bases discarded at tile edges; {how}")
        head = (f"`{T[lb]['variant']}` MLP, {'03 + hard negatives (13)' if T[lb]['head'] == 'hardneg' else '03'}"
                + (", chosen on validation (07)" if T[lb]["variant"] == T[lb]["selected"] else
                   ", previous round (reference)"))
        if T[lb]["head"] == "finetuned":
            head = (f"`{T[lb]['variant']}` MLP trained together with the backbone (full fine-tune "
                    "on the same 8,000 training windows as 03)")
        md.append(f"| {lb}{' **(headline)**' if lb in headline else ''} | {head} | {how} |")
    md.append("")

    # ---- section 1
    md.append("## 1. Top-k and AUROC across all genes, with 95% intervals\n")
    md.append("Top-k: for a gene with k true donors, take its k highest-scoring donor positions "
              f"(de-duplicated, none within {TOL} bases of another) and count those within {TOL} "
              "bases of a true donor, each site matched once; pooled = hits ÷ sites over all "
              "genes. AUROC: every true site against every other scored position. **Intervals "
              f"resample genes** ({N_BOOT:,} bootstrap resamples, seed {SEED}), because sites in one "
              "gene share their context; this is wider and more honest than resampling sites.\n")
    md.append("| track | top-k donors | top-k acceptors | AUROC donors | AUROC acceptors |")
    md.append("|---|---:|---:|---:|---:|")
    for lb in order:
        d, a = P[lb]["donor"], P[lb]["acceptor"]
        md.append(f"| {lb} | {d['top_k']['hits']}/{d['top_k']['k']} = "
                  f"{fmt_ci(d['top_k']['value'], d['top_k']['ci95'])} | "
                  f"{a['top_k']['hits']}/{a['top_k']['k']} = "
                  f"{fmt_ci(a['top_k']['value'], a['top_k']['ci95'])} | "
                  f"{fmt_ci(d['auroc']['value'], d['auroc']['ci95'], 3)} | "
                  f"{fmt_ci(a['auroc']['value'], a['auroc']['ci95'], 3)} |")
    rand = (2 * TOL + 1) / pos["one_donor_per_bases"]
    md.append(f"\nRandom picks would give a top-k of about {rand:.3f}. Figure: "
              "`results/scan_tracks_ci.png`.\n")

    md.append("### Which differences between tracks are real?\n")
    md.append("Each row compares two tracks on the same bootstrap resamples (new minus "
              "baseline). **Bold** = the 95% interval excludes 0, so the difference is real at "
              "this sample size; plain = not distinguishable from gene-to-gene noise.\n")
    md.append("| comparison (new vs baseline) | Δ top-k donors | Δ top-k acceptors | "
              "Δ AUROC donors | Δ AUROC acceptors |")
    md.append("|---|---:|---:|---:|---:|")
    for c in C:
        dd = c["delta"]
        md.append(f"| {c['title']} | {fmt_delta(dd['top_k_donor'])} | "
                  f"{fmt_delta(dd['top_k_acceptor'])} | {fmt_delta(dd['auroc_donor'], 3)} | "
                  f"{fmt_delta(dd['auroc_acceptor'], 3)} |")
    md.append("")
    for c in C:
        md.append(f"- **{c['title']}.** {describe(c)}")
    md.append("")

    md.append(f"**The first {n_orig} genes vs the {n_add} new ones.** The round-4 and round-5 "
              f"top-k numbers came from {n_orig} genes. Same tracks, split by gene set:\n")
    md.append(f"| track | site type | first {n_orig} genes | {n_add} added genes | all {len(G)} |")
    md.append("|---|---|---:|---:|---:|")
    for lb in order:
        for cls in CLASSES:
            s = P[lb][cls]["top_k_by_gene_set"]
            o, a = s.get("original", [0, 0]), s.get("added", [0, 0])
            t = P[lb][cls]["top_k"]
            md.append(f"| {tag(lb)} | {cls}s | {o[0]}/{o[1]} = {o[0] / max(o[1], 1):.2f} | "
                      f"{a[0]}/{a[1]} = {a[0] / max(a[1], 1):.2f} | "
                      f"{t['hits']}/{t['k']} = {t['value']:.2f} |")
    md.append("")
    hyp = (read_json(RESULTS_DIR / "motif_analysis.json") or {}).get("acceptor_token_hypothesis", {})
    for short, rows in hyp.items():
        r = rows.get("acceptor, + strand: AG@298", {})
        sv = next((t["selected"] for t in T.values() if MODELS[t["repo"]] == short
                   and t["head"] == "original"), None)
        if "center_tok" in r and sv in r:
            dlt = (r[sv] - r["center_tok"]) * 100
            md.append(f"- *Test windows (round 5, unchanged):* {short} recall on + strand "
                      f"acceptors is {r['center_tok']:.3f} with `center_tok` and {r[sv]:.3f} with "
                      f"`{sv}` ({dlt:+.1f} points; n = {r['n']}). Source: section 5 of "
                      "`motif_analysis.md`.")
    md.append("")

    # ---- section 2: hard negatives
    md.append("## 2. Hard negatives\n")
    md.append("The current head's false positives sit mostly inside exons, because the "
              "training negatives were mostly intronic and intergenic. `13_hard_negatives.py` "
              "mines negatives that look like what the scan sees, and the same MLP recipe is "
              "retrained with them.\n")
    if hnj:
        ch, rule, cnt = hnj["chromosomes"], hnj["rule"], hnj["counts"]

        def nat(cs):
            return sorted(cs, key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else 0, c))
        ch = {**ch, "dataset_train": nat(ch["dataset_train"]), "dataset_test": nat(ch["dataset_test"])}
        md.append("### What was mined\n")
        md.append(f"Genes come **only from training chromosomes** "
                  f"(chr{', chr'.join(ch['used_for_mining'])}; read from the dataset: train = "
                  f"chr{', chr'.join(ch['dataset_train'])}, test = chr{', chr'.join(ch['dataset_test'])}; "
                  f"{ch['note']}). {rule['genes']} protein-coding genes were drawn from "
                  f"{rule['regions']} seeded random 5-Mb regions (canonical transcript "
                  "protein-coding, 5–60 kb, ≥ 4 exons, no overlapping protein-coding gene), up to "
                  f"{rule['per_gene_per_group']} windows per group per gene. "
                  f"**{rule['holdout_genes']} genes are held out entirely** for evaluation. "
                  "\"Annotated splice site\" means an internal exon boundary of any transcript of "
                  "any gene in the region, on either strand, so no alternative site is used as a "
                  "negative. Windows are 600 nt of the genome + strand with the position at "
                  "offset 300, the dataset's own convention.\n")
        md.append("| group | definition | training genes | held-out genes |")
        md.append("|---|---|---:|---:|")
        defs = {"exon_interior": f"inside a canonical exon, ≥ {rule['min_distance_to_any_annotated_site']['exon_interior']} "
                                 "bases from every annotated splice site",
                "intron_interior": f"inside a canonical intron, ≥ {rule['min_distance_to_any_annotated_site']['intron_interior']} "
                                   "bases from every annotated splice site, outside every annotated exon",
                "decoy": "GT right after the base or AG right before it in the gene's orientation "
                         "(where a real donor / acceptor has it), ≥ "
                         f"{rule['min_distance_to_any_annotated_site']['decoy']} bases from every "
                         "annotated splice site"}
        for g in ("exon_interior", "intron_interior", "decoy"):
            md.append(f"| **{g.replace('_', ' ')}** | {defs[g]} | {cnt['train'][g]:,} | "
                      f"{cnt['holdout'][g]:,} |")
        md.append(f"| **total** | | **{cnt['train']['total']:,}** | {cnt['holdout']['total']:,} |")
        sub = hnj.get("by_subtype", {})
        dec = {k.split(" / ")[1]: 0 for k in sub if k.startswith("decoy")}
        for k, v in sub.items():
            if k.startswith("decoy"):
                dec[k.split(" / ")[1]] += v
        dloc = {loc: sum(v for k, v in sub.items() if k.startswith("decoy") and k.endswith(loc))
                for loc in ("exon", "intron")}
        cc = hnj.get("convention_check", {})
        mi = hnj.get("interior_windows_that_also_carry_a_motif", {})
        md.append("\nDecoys: " + ", ".join(f"{v:,} {k}" for k, v in dec.items())
                  + f"; {dloc['exon']:,} in exons, {dloc['intron']:,} in introns. "
                  "Interior windows also carry the motif by chance: "
                  + ", ".join(f"{g.replace('_', ' ')} {sum(v.values()):,}" for g, v in mi.items())
                  + ". Convention check on these genes' own canonical sites: "
                  + ", ".join(f"{k}s {v['with_canonical_motif']}/{v['sites']} show the motif at the "
                              "training offset" for k, v in cc.items())
                  + ". Counts per gene: `results/hard_negatives.json`.\n")

    md.append("### Training and validation (same recipe, same 07 protocol)\n")
    md.append("The MLP architecture and training recipe are unchanged (03). Selection follows "
              "07: the same 10% of original training rows, plus ~10% of the hard negatives taken "
              "as whole genes, are held out. Two selection heads per variant, without and with "
              "hard negatives, are scored on that split. The variant is chosen by the "
              "hard-negative head's accuracy over the whole split, then refit on everything "
              "(03 `--hardneg`) and tested once on the original test windows (chr20/chr21).\n")
    md.append("| model | variant (current → hard-neg) | validation: original rows | validation: "
              "hard negatives called none | test accuracy | test recall donor / acceptor / none |")
    md.append("|---|---|---:|---:|---:|---:|")
    for repo, short in MODELS.items():
        vh = vs_h.get(repo)
        if not vh:
            continue
        v = vh["selected_variant"]
        s = vh["selection_split"][v]
        cur_v = vs_o.get(repo, {}).get("selected_variant")
        tc = pv.get(repo, {}).get(cur_v, {}).get("mlp")
        th = hh.get(repo, {}).get(v, {}).get("mlp")

        def rec3(m):
            return " / ".join(f"{m['per_class'][c]['recall']:.3f}" for c in ("donor", "acceptor", "none"))
        md.append(f"| {short} | `{cur_v}` → `{v}` | {s['original']['accuracy_original_rows']:.4f} → "
                  f"{s['hardneg']['accuracy_original_rows']:.4f} | "
                  f"{s['original']['hard_negatives_called_none']:.3f} → "
                  f"{s['hardneg']['hard_negatives_called_none']:.3f} | "
                  + (f"{tc['accuracy']:.4f} → {th['accuracy']:.4f} | {rec3(tc)} → {rec3(th)} |"
                     if tc and th else "n/a | n/a |"))
    md.append("\nValidation columns: the two selection heads of 07 `--hardneg` on the same "
              "held-out rows. Test columns: the final heads (current = `03`, hard-neg = "
              "`03 --hardneg`), on the dataset's 2,000 test windows.\n")

    md.append("**Held-out hard-negative genes** (never used to train or select either head): "
              "share of windows each head calls \"none\".\n")
    md.append("| model | group | windows | current head | hard-negative head |")
    md.append("|---|---|---:|---:|---:|")
    for repo, short in MODELS.items():
        v = vs_h.get(repo, {}).get("selected_variant")
        ev = hh.get(repo, {}).get(v, {}).get("holdout_hard_negatives", {})
        for g in ("exon_interior", "intron_interior", "decoy"):
            if g in ev.get("hardneg", {}):
                md.append(f"| {short} | {g.replace('_', ' ')} | {ev['hardneg'][g]['n']:,} | "
                          f"{ev.get('original', {}).get(g, {}).get('called_none', float('nan')):.3f} | "
                          f"{ev['hardneg'][g]['called_none']:.3f} |")
    md.append("")

    md.append("### Gene scan: top-k and AUROC, current vs hard-negative head\n")
    md.append("From section 1 (same genes, same intervals):\n")
    md.append("| track | top-k donors | top-k acceptors | AUROC donors | AUROC acceptors |")
    md.append("|---|---:|---:|---:|---:|")
    for m, h in HN.items():
        for lb in (h["baseline"], h["new"]):
            d, a = P[lb]["donor"], P[lb]["acceptor"]
            md.append(f"| {tag(lb)} | {fmt_ci(d['top_k']['value'], d['top_k']['ci95'])} | "
                      f"{fmt_ci(a['top_k']['value'], a['top_k']['ci95'])} | "
                      f"{fmt_ci(d['auroc']['value'], d['auroc']['ci95'], 3)} | "
                      f"{fmt_ci(a['auroc']['value'], a['auroc']['ci95'], 3)} |")
        c = next((c for c in C if c["baseline"] == h["baseline"] and c["new"] == h["new"]), None)
        if c:
            dd = c["delta"]
            md.append(f"| *Δ (hard-neg − current)* | {fmt_delta(dd['top_k_donor'])} | "
                      f"{fmt_delta(dd['top_k_acceptor'])} | {fmt_delta(dd['auroc_donor'], 3)} | "
                      f"{fmt_delta(dd['auroc_acceptor'], 3)} |")
    md.append("")

    md.append("### False positives at fixed thresholds\n")
    md.append("Per scored position (per base for NT-v2, per token for DNABERT-2), at or above the "
              f"threshold and more than {TOL} bases from every true site of that type, pooled over "
              f"all {len(G)} scored spans ({pos['bases']:,} bases, {pos['exonic_fraction']:.0%} "
              "exonic). Raw thresholds are not at equal recall: the hard-negative head's "
              "probabilities are lower overall, so compare with the matched-recall table below.\n")
    md.append("| model | site type | threshold | head | true sites found | **false positives** | "
              "in exons | in introns | exonic share |")
    md.append("|---|---|---:|---|---:|---:|---:|---:|---:|")
    for m, h in HN.items():
        for cls in CLASSES:
            for r in h["fixed"][cls]:
                for head in ("original", "hardneg"):
                    f = r[head]
                    sh_ = f"{f['in_exons'] / f['false_positives']:.0%}" if f["false_positives"] else "n/a"
                    md.append(f"| {m} | {cls} | {r['threshold']} | {HEAD_NAME[head]} | "
                              f"{f['sites_found']}/{f['sites']} | **{f['false_positives']:,}** | "
                              f"{f['in_exons']:,} | {f['in_introns']:,} | {sh_} |")
    md.append("")

    md.append("### False positives at matched recall\n")
    md.append("The current head at 0.5, 0.9 and 0.99 finds a certain number of true sites. The "
              "hard-negative head's threshold is set (on these genes) to the highest value that "
              "finds at least as many. Same recall, so the false-positive counts are directly "
              "comparable. Verdict, from the exonic and intronic counts (down = at most ×0.75, "
              "up = at least ×1.33): exonic down and intronic not up = *improved*; exonic down "
              "but intronic up = *moved into introns*; exonic up = *worse*; exonic unchanged = "
              "*stayed the same* (or says which way the intronic ones moved); fewer than 30 false "
              "positives for the current head = *too few to judge*. Where the current "
              "head already finds **every** site (marked †), the matched threshold is set by the "
              "single hardest site, which makes that row much noisier than the others. Figure: "
              "`results/scan_hardneg_fp.png`.\n")
    md.append("| model | site type | operating point | sites found | current head: FPs (exon / "
              "intron) | hard-neg threshold | hard-neg head: FPs (exon / intron) | total | exonic | "
              "intronic | verdict |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for m, h in HN.items():
        for cls in CLASSES:
            for r in h["matched"][cls]:
                a, b = r["original"], r["hardneg"]
                full = " †" if r["sites_found"] == r["sites"] else ""
                md.append(f"| {m} | {cls} | current ≥ {r['threshold']}{full} | {r['sites_found']}/"
                          f"{r['sites']} | {a['false_positives']:,} ({a['in_exons']:,} / "
                          f"{a['in_introns']:,}) | {r['hardneg_threshold']:.4g} | "
                          f"{b['false_positives']:,} ({b['in_exons']:,} / {b['in_introns']:,}) | "
                          f"{fmt_ratio(ratio(a['false_positives'], b['false_positives']))} | "
                          f"{fmt_ratio(ratio(a['in_exons'], b['in_exons']))} | "
                          f"{fmt_ratio(ratio(a['in_introns'], b['in_introns']))} | "
                          f"**{fp_verdict(r)}** |")
    md.append("")

    md.append("### Where the false positives are (matched recall, the 0.9 operating point)\n")
    ex_kb = pos["exon_bases"] / 1000
    in_kb = (pos["bases"] - pos["exon_bases"]) / 1000
    md.append(f"Density = false positives per kb of exon ({ex_kb:,.1f} kb scored) or of intron "
              f"({in_kb:,.1f} kb). An exon:intron density ratio of 1 means no preference for "
              "exons. *Motif-bearing* = GT right after (donors) or AG right before (acceptors) "
              "the position, exactly where a real site has it.\n")
    md.append("| model | site type | head | false positives | per kb exon | per kb intron | "
              "exon : intron density | motif-bearing | on another transcript's site |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for m, h in HN.items():
        for cls in CLASSES:
            r = next(x for x in h["matched"][cls] if x["threshold"] == 0.9)
            for head in ("original", "hardneg"):
                f = r[head]
                de, di = f["in_exons"] / ex_kb, f["in_introns"] / in_kb
                fp = max(f["false_positives"], 1)
                md.append(f"| {m} | {cls} | {HEAD_NAME[head]} | {f['false_positives']:,} | "
                          f"{de:.1f} | {di:.2f} | {de / di if di else math.inf:.1f} | "
                          f"{f['motif_bearing'] / fp:.0%} | {f['on_other_transcript_sites']} |")
    md.append("")

    md.append("### Verdict\n")
    md.append(f"**First, the premise.** Exons are {pos['exonic_fraction']:.0%} of the scored "
              "sequence. With the current head at the 0.9 operating point, false positives are: "
              + "; ".join(f"{m} {premise(m)}" for m in HN)
              + ". So exons do carry more false positives per kb than introns, as suspected, "
              "but most false positives were intronic all along, simply because introns are most "
              "of a gene. Removing every exonic false positive could at best cut the total by "
              "that exonic share.\n")
    md.append("**Did the exonic false-positive problem improve, stay the same, or move "
              "elsewhere?**\n")
    for m in HN:
        md.append(f"- **{m}:** {plain(m)}")
    md.append("")
    for short, (overall, txt) in bottom.items():
        md.append(f"- **{short}, bottom line.** {overall} {txt}")
    md.append("")
    md.append("Per site type and operating point:\n")
    for m, h in HN.items():
        c = next((c for c in C if c["baseline"] == h["baseline"] and c["new"] == h["new"]), None)
        lines = []
        for cls in CLASSES:
            vs = [fp_verdict(r) for r in h["matched"][cls]]
            r9 = next(r for r in h["matched"][cls] if r["threshold"] == 0.9)
            a, b = r9["original"], r9["hardneg"]
            lines.append(f"{cls}s: at the 0.9 operating point ({r9['sites_found']}/{r9['sites']} "
                         f"sites found by both) exonic false positives {a['in_exons']:,} → "
                         f"{b['in_exons']:,} ({fmt_ratio(ratio(a['in_exons'], b['in_exons']))}), "
                         f"intronic {a['in_introns']:,} → {b['in_introns']:,} "
                         f"({fmt_ratio(ratio(a['in_introns'], b['in_introns']))}), total "
                         f"{a['false_positives']:,} → {b['false_positives']:,}: **{vs[1]}** "
                         f"(at 0.5 / 0.9 / 0.99: {' / '.join(vs)})")
        md.append(f"- **{m}.** " + "; ".join(lines) + ".")
        if c:
            md.append(f"  Threshold-free: {describe(c)}")
    md.append("")

    # ---- per-gene top-k
    md.append("## Top-k per gene\n")
    md.append("donor hits / acceptor hits, each out of the gene's k sites.\n")
    md.append("| gene | k | " + " | ".join(tag(lb) for lb in headline) + " |")
    md.append("|---|---:|" + "---:|" * len(headline))
    for sym, g in G.items():
        cells = []
        for lb in headline:
            tc = g["tracks"][lb]["classes"]
            cells.append(f"{tc['donor']['top_k']['hits']} / {tc['acceptor']['top_k']['hits']}")
        md.append(f"| {sym} ({g['strand']}) | {g['donors']} | " + " | ".join(cells) + " |")
    md.append("")

    # ---- composition control
    md.append("## Composition control\n")
    lb0 = headline[0]
    md.append("Same idea as section 4 of `motif_analysis.md`, at every position of every gene. The "
              "GT/AG rule in transcript orientation proposes every GT as a donor and every AG as an "
              f"acceptor ({P[lb0]['donor']['rule_alone']['false_positives']:,} false donor and "
              f"{P[lb0]['acceptor']['rule_alone']['false_positives']:,} false acceptor candidates). "
              "A gate that only sees the base composition of the surrounding 600 nt (the 09 "
              "logistic regression) keeps the highest-scoring candidates, per gene, until it has "
              "found as many true sites as the model did. That cut-off is set on the genes' own "
              "sites, which favours the control. Genes where the model found a site the rule "
              "cannot (non-canonical) are *unreachable* and add nothing to the control, which "
              "understates it.\n")
    md.append("| track | site type | threshold | sites found | model FPs | composition-gated rule FPs "
              "| model ÷ control | verdict |")
    md.append("|---|---|---:|---:|---:|---:|---:|---|")
    for lb in headline:
        for cls in CLASSES:
            for f in P[lb][cls]["false_positives"]:
                ctrl_, mf = f["control_false_positives"], f["false_positives"]
                un = f["control_unreachable_genes"]
                if f["sites_found"] == 0:
                    who, rt = "n/a (no sites found)", "n/a"
                else:
                    r = mf / ctrl_ if ctrl_ else math.inf
                    rt = f"{r:.2f}" if ctrl_ else "∞"
                    who = (f"**model, {1 / r:.1f}× fewer**" if r < 0.75 else
                           "similar (within 25%)" if r <= 1 / 0.75 else f"control, {r:.1f}× fewer")
                    if un:
                        who += f" (control unreachable in {len(un)} gene{'s' if len(un) > 1 else ''})"
                md.append(f"| {tag(lb)} | {cls} | {f['threshold']} | {f['sites_found']}/{f['sites']} "
                          f"| {mf:,} | {ctrl_:,} | {rt} | {who} |")
    md.append("")

    md.append("## Why a high AUROC still gives a low top-k\n")
    md.append("An AUROC of 0.99 means about 1% of positions outscore a typical true site. A gene "
              f"has ~{pos['one_donor_per_bases']:,} positions per donor, so roughly a dozen "
              "positions per gene score above a typical true donor. That is the class-imbalance "
              "effect in one sentence: a good per-window classifier is not yet a good "
              "per-position detector.\n")

    if out["context_check"]:
        md.append("## Context check: long-context tracks vs training-style windows\n")
        md.append("For the long-context tracks, 10 also scored every true site and 500 random "
                  "positions per gene in 600-nt windows exactly like training.\n")
        md.append("| track | site type | AUROC, long-context scan | AUROC, 600-nt windows | mean "
                  "score at true sites (long / window) | correlation long vs window |")
        md.append("|---|---|---:|---:|---:|---:|")
        for lb, cc in out["context_check"].items():
            for cls, c in cc.items():
                md.append(f"| {lb} | {cls} | {c['auroc_long']:.3f} | {c['auroc_window']:.3f} | "
                          f"{c['mean_site_long']:.2f} / {c['mean_site_window']:.2f} | "
                          f"{c['corr_long_window']:.2f} |")
        md.append("\nThese AUROCs use random positions, not every position, so they are higher "
                  "than the section 1 values.\n")

    md.append("## Figures\n")
    md.append("- `results/scan_tracks_ci.png`: pooled top-k and AUROC with 95% intervals, every "
              "track.")
    if HN:
        md.append("- `results/scan_hardneg_fp.png`: false positives at matched recall, current vs "
                  "hard-negative head, split into exons and introns.")
    md.append("- `results/scan_<GENE>.png`, one per gene: exon structure and the donor (up) / "
              "acceptor (down) scores of the headline tracks along the gene, true sites marked. "
              + ", ".join(f"`{sym}`" for sym in G) + ".\n")

    md.append("## Honest limitations\n")
    md.append(f"- **Balanced training, imbalanced reality.** True sites are about 1 in "
              f"{pos['one_site_per_bases']:,} positions; neither head was calibrated for that, so "
              "thresholded calls still come with hundreds to thousands of false positives.")
    md.append(f"- **{len(G)} genes.** The intervals are from resampling these genes. They cover "
              "gene-to-gene variation among genes like these (protein-coding, 4–25 kb, on two "
              "chromosomes), not all human genes.")
    md.append("- **Matched-recall thresholds are set on the test genes.** That is fine for "
              "comparing the two heads with each other (both get the same advantage), but the "
              "thresholds themselves are not a deployable operating point.")
    md.append("- **Hard negatives are labelled from annotation.** An unannotated real site would "
              "be a wrong negative; excluding every annotated boundary of every transcript in the "
              "region on both strands keeps this rare, not impossible.")
    md.append("- **Resolution.** NT-v2 gets a score per base. DNABERT-2 points to a whole token "
              "(~5 bases), so its ±2-base tolerance is effectively wider and its false positives "
              "are counted per token.")
    md.append("- **Canonical transcript only.** Real sites of other transcripts inside the span "
              "count as false positives; the tables show how many.\n")
    OUT_MD.write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    sys.exit(main())
