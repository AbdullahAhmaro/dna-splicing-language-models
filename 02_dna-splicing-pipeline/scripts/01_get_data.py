#!/usr/bin/env python3
"""
Phase 1, step 1: download the splice-site dataset, cache it, document its format.

    python scripts/01_get_data.py            # download once, then reuse the disk cache
    python scripts/01_get_data.py --refresh  # force a fresh download

Outputs
    data/splice_sites_all/            dataset cached with save_to_disk (later steps read this)
    data/label_info.json              what each label value means + where the motifs sit
    results/dataset_format.md         human-readable description (for the advisor)
    results/splice_sites_all_sample200.csv   200 rows, stratified by label

The dataset card does not say which label value is donor / acceptor / none, so
this script works it out from the sequences themselves: a donor site starts
the intron with GT, an acceptor ends it with AG. Whichever class has those
letters at one fixed position in (nearly) every row is that kind of site.
"""

import argparse
import collections
import shutil
import sys
from datetime import datetime, timezone

import numpy as np

from phase1_common import (DATA_DIR, DATASET_CONFIG, DATASET_DIR, DATASET_REPO,
                           LABEL_INFO_PATH, RESULTS_DIR, SEED, dinuc_hits,
                           read_json, seq_matrix, stratified_subsample,
                           write_json_atomic)

FORMAT_MD = RESULTS_DIR / "dataset_format.md"
SAMPLE_CSV = RESULTS_DIR / "splice_sites_all_sample200.csv"

# GT starts an intron (donor), AG ends it (acceptor). AC and CT are the same two
# motifs read off the opposite DNA strand (reverse complements).
MOTIFS = {"GT": "donor, + strand", "AC": "donor, - strand (rev. comp. of GT)",
          "AG": "acceptor, + strand", "CT": "acceptor, - strand (rev. comp. of AG)"}


def download():
    """The revised repo no longer exposes named configs (only 'default'), so the
    splice_sites_all task is loaded straight from its parquet files."""
    from datasets import load_dataset

    files = {s: f"{DATASET_CONFIG}/{s}.parquet" for s in ("train", "test")}
    print(f"Downloading {DATASET_REPO} :: {files}")
    ds = load_dataset(DATASET_REPO, data_files=files)
    sha = None
    try:
        from huggingface_hub import HfApi
        sha = HfApi().dataset_info(DATASET_REPO).sha
    except Exception as exc:                    # noqa: BLE001 - informational only
        print(f"  (could not read dataset commit id: {exc})")
    return ds, sha


def peak(profile):
    k = int(np.argmax(profile))
    return k, float(profile[k])


def infer_label_semantics(mat, y):
    """
    Decide which label value is donor / acceptor / neither from motif enrichment
    on the TRAIN split. Returns a dict that is written to data/label_info.json.
    """
    classes = sorted(int(c) for c in set(y.tolist()))
    hits = {d: dinuc_hits(mat, d) for d in MOTIFS}
    prof = {c: {d: hits[d][y == c].mean(axis=0) for d in MOTIFS} for c in classes}

    # Strand-combined enrichment: peak frequency on + strand plus on - strand.
    donor_score = {c: peak(prof[c]["GT"])[1] + peak(prof[c]["AC"])[1] for c in classes}
    acc_score = {c: peak(prof[c]["AG"])[1] + peak(prof[c]["CT"])[1] for c in classes}
    donor = max(classes, key=donor_score.get)
    acceptor = max(classes, key=acc_score.get)
    rest = [c for c in classes if c not in (donor, acceptor)]

    offsets = {
        "donor_GT_plus": peak(prof[donor]["GT"])[0],
        "donor_AC_minus": peak(prof[donor]["AC"])[0],
        "acceptor_AG_plus": peak(prof[acceptor]["AG"])[0],
        "acceptor_CT_minus": peak(prof[acceptor]["CT"])[0],
    }

    def union(c, a, ka, b, kb):
        m = y == c
        return float((hits[a][m, ka] | hits[b][m, kb]).mean())

    per_class = {}
    for c in classes:
        per_class[c] = {
            **{f"{d}_peak": {"offset": peak(prof[c][d])[0],
                              "fraction": round(peak(prof[c][d])[1], 4)} for d in MOTIFS},
            "GT@donor_off_or_AC@donor_off": round(union(
                c, "GT", offsets["donor_GT_plus"], "AC", offsets["donor_AC_minus"]), 4),
            "AG@acc_off_or_CT@acc_off": round(union(
                c, "AG", offsets["acceptor_AG_plus"], "CT", offsets["acceptor_CT_minus"]), 4),
        }

    cov_donor = per_class[donor]["GT@donor_off_or_AC@donor_off"]
    cov_acc = per_class[acceptor]["AG@acc_off_or_CT@acc_off"]
    none_rate = None
    clear = donor != acceptor and len(rest) == 1
    if clear:
        none = rest[0]
        none_rate = max(per_class[none]["GT@donor_off_or_AC@donor_off"],
                        per_class[none]["AG@acc_off_or_CT@acc_off"])
        clear = cov_donor >= 0.9 and cov_acc >= 0.9 and none_rate <= 0.3

    if clear:
        names = {donor: "donor", acceptor: "acceptor", rest[0]: "none"}
        status = "inferred"
        note = ("Label names are NOT documented in the dataset card; they were inferred "
                "from canonical-motif enrichment on the training split. The evidence is "
                f"strong: {cov_donor:.1%} of label-{donor} rows carry GT (or its reverse "
                f"complement AC) at one fixed offset, {cov_acc:.1%} of label-{acceptor} rows "
                f"carry AG (or CT), versus at most {none_rate:.1%} of label-{rest[0]} rows.")
    else:
        names = {c: f"class_{c}" for c in classes}
        status = "unclear"
        note = ("Could NOT confidently map label values to donor/acceptor/none from motif "
                "enrichment. Treat any donor/acceptor naming downstream as a guess.")

    return {
        "status": status,
        "note": note,
        "label_values": classes,
        "names": {str(k): v for k, v in sorted(names.items())},
        "donor_label": donor if clear else None,
        "acceptor_label": acceptor if clear else None,
        "none_label": rest[0] if clear else None,
        "motif_offsets_0based": offsets,
        "per_class_evidence": {str(k): v for k, v in per_class.items()},
        "scores": {"donor_score": {str(k): round(v, 4) for k, v in donor_score.items()},
                   "acceptor_score": {str(k): round(v, 4) for k, v in acc_score.items()}},
    }


def wrap_seq(seq, width=100):
    return "\n".join(f"{i:>4}  {seq[i:i + width]}" for i in range(0, len(seq), width))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--refresh", action="store_true",
                    help="re-download even if the disk cache exists")
    args = ap.parse_args()

    from datasets import load_from_disk

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sha = None
    if DATASET_DIR.exists() and not args.refresh:
        print(f"Using cached dataset at {DATASET_DIR}")
        ds = load_from_disk(str(DATASET_DIR))
        old = read_json(LABEL_INFO_PATH, default={}) or {}
        sha = old.get("dataset_commit")
    else:
        ds, sha = download()
        if DATASET_DIR.exists():
            shutil.rmtree(DATASET_DIR)
        ds.save_to_disk(str(DATASET_DIR))
        print(f"Cached to {DATASET_DIR}")

    splits = list(ds.keys())
    columns = ds[splits[0]].column_names
    print(ds)

    # ---------- per-split facts ----------
    facts = {}
    for s in splits:
        seqs = ds[s]["sequence"]
        lengths = np.array([len(x) for x in seqs])
        labels = np.array(ds[s]["label"])
        chars = collections.Counter("".join(seqs))
        name_suffix = np.array([int(n.rsplit("|", 1)[1]) for n in ds[s]["name"]])
        facts[s] = {
            "n": len(seqs),
            "len_min": int(lengths.min()), "len_median": float(np.median(lengths)),
            "len_max": int(lengths.max()),
            "labels": dict(sorted(collections.Counter(labels.tolist()).items())),
            "chars": dict(sorted(chars.items())),
            "tasks": sorted(set(ds[s]["task"])),
            "name_suffix_equals_label": bool((name_suffix == labels).all()),
        }
        print(f"\n[{s}] {facts[s]['n']} rows  length min/median/max = "
              f"{facts[s]['len_min']}/{facts[s]['len_median']:.0f}/{facts[s]['len_max']}")
        print(f"  label counts: {facts[s]['labels']}")
        print(f"  characters:   {facts[s]['chars']}")

    # ---------- what the labels mean ----------
    train_mat = seq_matrix(ds["train"]["sequence"])
    train_y = np.array(ds["train"]["label"])
    info = infer_label_semantics(train_mat, train_y)
    info["dataset_repo"] = DATASET_REPO
    info["dataset_config"] = DATASET_CONFIG
    info["dataset_commit"] = sha
    info["sequence_length"] = int(train_mat.shape[1])
    info["label_granularity"] = "window"

    test_mat = seq_matrix(ds["test"]["sequence"])
    test_y = np.array(ds["test"]["label"])
    gc = {}
    for split_name, mat, y in (("train", train_mat, train_y), ("test", test_mat, test_y)):
        gc[split_name] = {str(c): round(float(np.isin(mat[y == c], [ord("G"), ord("C")]).mean()), 4)
                          for c in info["label_values"]}
    info["gc_fraction_by_class"] = gc

    # How often the strand-specific motif shows up: tells us the windows are
    # written on the genome's + strand, not on the gene's strand.
    if info["status"] == "inferred":
        d, a = info["donor_label"], info["acceptor_label"]
        o = info["motif_offsets_0based"]
        plus_d = float(dinuc_hits(train_mat, "GT")[train_y == d, o["donor_GT_plus"]].mean())
        minus_d = float(dinuc_hits(train_mat, "AC")[train_y == d, o["donor_AC_minus"]].mean())
        plus_a = float(dinuc_hits(train_mat, "AG")[train_y == a, o["acceptor_AG_plus"]].mean())
        minus_a = float(dinuc_hits(train_mat, "CT")[train_y == a, o["acceptor_CT_minus"]].mean())
        info["strand_mix"] = {"donor_plus": round(plus_d, 4), "donor_minus": round(minus_d, 4),
                              "acceptor_plus": round(plus_a, 4), "acceptor_minus": round(minus_a, 4)}

    write_json_atomic(LABEL_INFO_PATH, info)
    print(f"\nLabel semantics: {info['status'].upper()}  names = {info['names']}")
    print(f"  {info['note']}")
    print(f"  motif offsets (0-based): {info['motif_offsets_0based']}")
    if "strand_mix" in info:
        print(f"  strand mix: {info['strand_mix']}")
    print(f"  wrote {LABEL_INFO_PATH}")

    # ---------- 200-row sample ----------
    idx = stratified_subsample(train_y, 200, SEED)
    ds["train"].select(idx.tolist()).to_csv(str(SAMPLE_CSV), index=False)
    print(f"  wrote {SAMPLE_CSV} (200 train rows, stratified by label, seed {SEED})")

    # ---------- dataset_format.md ----------
    names = {int(k): v for k, v in info["names"].items()}
    L = info["sequence_length"]
    o = info["motif_offsets_0based"]
    md = []
    md.append("# Dataset format: splice_sites_all\n")
    md.append(f"_Generated by `scripts/01_get_data.py` on "
              f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}._\n")
    md.append("## Source\n")
    md.append(f"- Hugging Face dataset: `{DATASET_REPO}`, task `{DATASET_CONFIG}`")
    md.append(f"- Files: `{DATASET_CONFIG}/train.parquet`, `{DATASET_CONFIG}/test.parquet`"
              " (the repo no longer exposes a named `splice_sites_all` config, only `default`,"
              " so the task's parquet files are loaded directly)")
    if sha:
        md.append(f"- Dataset commit: `{sha}`")
    md.append("- Origin (dataset card): human splice sites from GENCODE V44, level-3 (automated) "
              "transcripts excluded. Test set is held-out chromosomes.")
    md.append(f"- Cached locally at `data/splice_sites_all/` (later steps never re-download).\n")

    md.append("## Splits and sizes\n")
    md.append("| split | rows | length min / median / max | label counts |")
    md.append("|---|---:|---|---|")
    for s in splits:
        f = facts[s]
        lc = ", ".join(f"{k} ({names.get(k, '?')}): {v}" for k, v in f["labels"].items())
        md.append(f"| {s} | {f['n']:,} | {f['len_min']} / {f['len_median']:.0f} / "
                  f"{f['len_max']} | {lc} |")
    md.append("")
    md.append(f"Distinct label values: `{info['label_values']}`. "
              f"Characters in sequences: {', '.join(sorted(facts['train']['chars']))} "
              "(no N or lowercase).\n")

    md.append("## Columns\n")
    md.append("| column | type | content |")
    md.append("|---|---|---|")
    desc = {
        "sequence": f"DNA string, always {L} nt, upper-case A/C/G/T",
        "name": "`chrom:start-end\\|label`: genomic coordinates of the window; the suffix after `\\|` "
                + ("always equals `label`" if all(f["name_suffix_equals_label"] for f in facts.values())
                   else "does NOT always equal `label`"),
        "label": "integer class for the whole window (see below)",
        "task": f"constant: {facts['train']['tasks']}",
    }
    feats = ds["train"].features
    for c in columns:
        md.append(f"| `{c}` | {feats[c]} | {desc.get(c, '')} |")
    md.append("")

    md.append("## Does a row label a WINDOW or a POSITION?\n")
    md.append(f"**Each row labels an entire WINDOW.** Every row is one {L}-nt sequence with exactly "
              "one integer label; there is no per-nucleotide label and no column giving a "
              "position inside the sequence.\n")
    md.append(f"The window is positioned so that the candidate splice junction sits at its "
              f"centre (around offset {L // 2}). The label answers *\"is the centre of this window "
              "a donor site, an acceptor site, or neither?\"*. So the task is **3-way "
              "classification of a fixed window**, not **locating** the cut point in a long "
              "sequence. The advisor's framing from meeting 3 (\"output a number 0 to 8000 "
              "for the cut point\") is a different, harder task that this dataset does not "
              "directly provide.\n")

    md.append("## What each label value means\n")
    md.append(f"**Status: {info['status'].upper()}.** {info['note']}\n")
    md.append("| label | meaning | rows with GT@{gt} or AC@{ac} | rows with AG@{ag} or CT@{ct} |"
              .format(gt=o["donor_GT_plus"], ac=o["donor_AC_minus"],
                      ag=o["acceptor_AG_plus"], ct=o["acceptor_CT_minus"]))
    md.append("|---:|---|---:|---:|")
    for c in info["label_values"]:
        ev = info["per_class_evidence"][str(c)]
        md.append(f"| {c} | {names[c]} | {ev['GT@donor_off_or_AC@donor_off']:.1%} | "
                  f"{ev['AG@acc_off_or_CT@acc_off']:.1%} |")
    md.append("")
    md.append("Offsets are 0-based positions in the sequence string; `GT@301` means "
              "`sequence[301:303] == 'GT'`.\n")

    if "strand_mix" in info:
        sm = info["strand_mix"]
        md.append("### Important: sequences are NOT strand-normalised\n")
        md.append(f"Only about half of the splice-site windows show the textbook motif. Donors: "
                  f"{sm['donor_plus']:.1%} have `GT` at offset {o['donor_GT_plus']}, and "
                  f"{sm['donor_minus']:.1%} instead have `AC` at offset {o['donor_AC_minus']}, "
                  "which is `GT` read on the opposite strand. Acceptors: "
                  f"{sm['acceptor_plus']:.1%} have `AG` at offset {o['acceptor_AG_plus']}, and "
                  f"{sm['acceptor_minus']:.1%} have `CT` at offset {o['acceptor_CT_minus']} "
                  "(reverse complement of `AG`).\n")
        md.append("So the windows are written on the genome's + strand, whatever strand the "
                  "gene is on. Genes on the − strand appear reverse-complemented. Two "
                  "consequences:\n")
        md.append("1. A model has to recognise a splice site in both orientations.")
        md.append(f"2. A rule that only looks for `GT`/`AG` at one offset can catch at most "
                  "about half of one class. Donor `GT` "
                  f"(offset {o['donor_GT_plus']}) and acceptor `AG` (offset "
                  f"{o['acceptor_AG_plus']}) are also at different offsets. "
                  "`04_baseline.py` therefore reports the rule exactly as specified *and* a "
                  "strand-aware variant.\n")

    md.append("### Other differences between classes\n")
    md.append("GC fraction by class (train): " + ", ".join(
        f"label {c} ({names[c]}): {gc['train'][str(c)]:.1%}" for c in info["label_values"])
        + ". Negative windows are noticeably more AT-rich than splice-site windows, so "
        "overall base composition alone carries some signal. That is worth knowing when "
        "reading the results of a mean-pooled embedding.\n")

    md.append("## Three example rows (train split, one per label, full content)\n")
    for c in info["label_values"]:
        i = int(np.flatnonzero(train_y == c)[0])
        row = ds["train"][i]
        md.append(f"### Row {i}: label {row['label']} ({names[c]})\n")
        md.append(f"- `name`: `{row['name']}`")
        md.append(f"- `task`: `{row['task']}`")
        md.append(f"- `label`: `{row['label']}`")
        md.append(f"- `sequence` ({len(row['sequence'])} nt, left column = 0-based offset):\n")
        md.append("```\n" + wrap_seq(row["sequence"]) + "\n```\n")
        print(f"\nExample row {i} (label {row['label']} = {names[c]}): name={row['name']}")
        print(wrap_seq(row["sequence"]))

    md.append("## Files\n")
    md.append("- `results/splice_sites_all_sample200.csv`: 200 training rows, stratified by "
              f"label (seed {SEED})")
    md.append("- `data/label_info.json`: the label mapping and motif evidence used by later steps")
    FORMAT_MD.write_text("\n".join(md) + "\n")
    print(f"\nWrote {FORMAT_MD}")

    if info["status"] != "inferred":
        print("WARNING: label semantics unclear, see dataset_format.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
