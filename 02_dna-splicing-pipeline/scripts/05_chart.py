#!/usr/bin/env python3
"""
Phase 1, step 5: grouped bar charts of accuracy and macro-F1.

    python scripts/05_chart.py

Reads results/phase1_results.json and writes (200 dpi):
  results/phase1_comparison.png       methods side by side (full-window mean pooling)
  results/phase1_pooling_variants.png every pooling variant x model, with the
                                      strand-aware GT/AG rule as a reference line
                                      (only when 02b/03 variant results exist)

Groups: the GT/AG rule exactly as specified, the strand-aware GT/AG rule,
NT-v2-50M + MLP head, and DNABERT-2 + MLP head. Every number comes from the
same test rows the models were scored on. A method that failed or was not run
is shown as "not run" instead of being silently dropped.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

from phase1_common import MODELS, RESULTS_DIR, RESULTS_JSON, read_json  # noqa: E402

OUT_PNG = RESULTS_DIR / "phase1_comparison.png"
OUT_VARIANTS_PNG = RESULTS_DIR / "phase1_pooling_variants.png"
VARIANT_ORDER = ["full_mean", "center_tok", "center_w10", "center_w25", "center_w50",
                 "concat_c25_full", "prev_tok", "center_plus_prev", "center_plus_prev_concat"]
VARIANT_LABELS = {
    "full_mean": "full-window\nmean",
    "center_tok": "centre\ntoken",
    "center_w10": "bases\n290-310",
    "center_w25": "bases\n275-325",
    "center_w50": "bases\n250-350",
    "concat_c25_full": "275-325\n+ full mean",
    "prev_tok": "previous\ntoken",
    "center_plus_prev": "centre +\nprevious\n(mean)",
    "center_plus_prev_concat": "centre ++\nprevious\n(concat)",
}

# Palette: categorical slots 1-2 of the validated default palette (blue, orange).
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
SERIES = {"accuracy": ("Accuracy", "#2a78d6"), "macro_f1": ("Macro-F1", "#eb6834")}


def baseline_scores(rule):
    """Prefer the model-matched test subset; fall back to the full test split."""
    if not rule or "skipped" in rule:
        return None, None
    for key in ("model_subset", "full_test"):
        if key in rule:
            return rule[key], key
    return None, None


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="y", colors=TEXT_2, length=0)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


def variant_metrics(res, repo, variant):
    if variant == "full_mean":
        return res.get("models", {}).get(repo, {}).get("mlp")
    return res.get("pooling_variants", {}).get(repo, {}).get(variant, {}).get("mlp")


def pooling_chart(res):
    """Small multiples: one panel per model, one group per pooling variant."""
    if not res.get("pooling_variants"):
        print("No pooling-variant results yet; skipping", OUT_VARIANTS_PNG.name)
        return
    rb = res.get("baseline", {}).get("strand_aware_rule", {})
    ref, _ = baseline_scores(rb)

    fig, axes = plt.subplots(len(MODELS), 1, figsize=(12.5, 8.6), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(VARIANT_ORDER))
    width = 0.38
    n_test = set()
    for ax, (repo, short) in zip(np.atleast_1d(axes), MODELS.items()):
        style_axis(ax)
        if ref:
            ax.axhline(ref["accuracy"], color=TEXT_2, linewidth=1.1,
                       linestyle=(0, (5, 3)), zorder=4)
        for i, v in enumerate(VARIANT_ORDER):
            m = variant_metrics(res, repo, v)
            if m is None:
                ax.text(x[i], 0.05, "not run", ha="center", va="bottom", fontsize=8.5,
                        color=MUTED, style="italic")
                continue
            n_test.add(sum(c["support"] for c in m["per_class"].values()))
            for j, (key, (_, color)) in enumerate(SERIES.items()):
                val = m[key]
                xi = x[i] + (j - 0.5) * width
                ax.bar(xi, val, width, color=color, edgecolor=SURFACE, linewidth=1.0, zorder=3)
                ax.text(xi, val + 0.012, f"{val:.3f}", ha="center", va="bottom",
                        fontsize=7, color=TEXT, zorder=5,
                        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.4))
        chosen = res.get("variant_selection", {}).get("models", {}).get(repo, {}).get(
            "selected_variant")
        ax.set_xticks(x, [VARIANT_LABELS[v] + ("\n▲ chosen on\nvalidation" if v == chosen else "")
                          for v in VARIANT_ORDER], fontsize=8.5, color=TEXT)
        ax.set_xlim(-0.6, len(VARIANT_ORDER) - 0.4)
        ax.set_title(f"{short} + MLP head (frozen backbone)", loc="left", fontsize=10.5,
                     color=TEXT, pad=8)
    np.atleast_1d(axes)[0].set_ylabel("Score on held-out test windows", color=TEXT_2)

    handles = [Patch(color=c, label=lbl) for lbl, c in SERIES.values()]
    if ref:
        handles.append(Line2D([], [], color=TEXT_2, linewidth=1.1, linestyle=(0, (5, 3)),
                              label=f"Strand-aware GT/AG rule, no learning "
                                    f"(acc {ref['accuracy']:.3f}, F1 {ref['macro_f1']:.3f})"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, 0.915),
               frameon=False, ncols=3, fontsize=9, labelcolor=TEXT, handlelength=1.6)
    fig.suptitle("Splice-site classification by pooling variant", x=0.055, y=0.99,
                 ha="left", fontsize=12.5, color=TEXT)
    n_txt = " / ".join(f"{n:,}" for n in sorted(n_test)) or "?"
    fig.text(0.055, 0.925, "Mean of last-layer token vectors over different base ranges of the "
             f"600-nt window (splice junction at ~300). Test n = {n_txt}.",
             fontsize=8.5, color=TEXT_2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2.0)
    fig.savefig(OUT_VARIANTS_PNG, dpi=200, facecolor=SURFACE)
    print(f"Saved {OUT_VARIANTS_PNG}")


def main():
    res = read_json(RESULTS_JSON)
    if not res:
        print(f"{RESULTS_JSON} not found or empty; run steps 03/04 first.", file=sys.stderr)
        return 1

    base = res.get("baseline", {})
    models = res.get("models", {})
    groups = []          # (label, metrics-dict-or-None, n)

    a, _ = baseline_scores(base.get("single_offset_rule"))
    k = base.get("single_offset_rule", {}).get("best_offset")
    groups.append((f"GT/AG rule\none offset ({k}),\nas specified", a))
    b, _ = baseline_scores(base.get("strand_aware_rule"))
    groups.append(("GT/AG rule\nstrand-aware\n(no learning)", b))
    for repo, short in MODELS.items():
        m = models.get(repo, {}).get("mlp")
        groups.append((f"{short}\n+ MLP head\n(frozen)", m))

    ns = {m["n"] for _, m in groups[:2] if m} | {models[r]["n_test"] for r in MODELS if r in models}
    n_label = " / ".join(f"{n:,}" for n in sorted(ns)) or "?"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(8.6, 4.9), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = np.arange(len(groups))
    width = 0.34
    for j, (key, (label, color)) in enumerate(SERIES.items()):
        offs = (j - 0.5) * width
        for i, (_, m) in enumerate(groups):
            if m is None:
                continue
            v = m[key]
            ax.bar(x[i] + offs, v, width, color=color, edgecolor=SURFACE, linewidth=1.2,
                   zorder=3)
            ax.text(x[i] + offs, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=TEXT, zorder=4)
    for i, (_, m) in enumerate(groups):
        if m is None:
            ax.text(x[i], 0.05, "not run /\nfailed", ha="center", va="bottom",
                    fontsize=9, color=MUTED, style="italic")

    chance = 1 / 3
    ax.axhline(chance, color=MUTED, linewidth=0.9, linestyle=(0, (3, 3)), zorder=2)

    ax.set_xticks(x, [g[0] for g in groups], fontsize=9, color=TEXT)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Score on held-out test windows", color=TEXT_2)
    ax.tick_params(axis="y", colors=TEXT_2, length=0)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)

    ax.set_title("Splice-site classification: donor / acceptor / neither",
                 loc="left", fontsize=12, color=TEXT, pad=26)
    ax.text(0, 1.035, f"600-nt windows, GENCODE human; test n = {n_label}. "
            "Language-model backbones frozen, embeddings mean-pooled.",
            transform=ax.transAxes, fontsize=8.5, color=TEXT_2, va="bottom")
    handles = [Patch(color=c, label=lbl) for lbl, c in SERIES.values()]
    handles.append(Line2D([], [], color=MUTED, linewidth=0.9, linestyle=(0, (3, 3)),
                          label="Chance (1/3)"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False,
              ncols=3, fontsize=9, labelcolor=TEXT, handlelength=1.0, handleheight=1.0)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, facecolor=SURFACE)
    print(f"Saved {OUT_PNG}")
    for label, m in groups:
        flat = label.replace("\n", " ")
        print(f"  {flat:<45} " + (f"acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}"
                                  if m else "not run / failed"))
    pooling_chart(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
