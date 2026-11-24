#!/usr/bin/env python3
"""
Phase 1, step 14: bring the fine-tuned model's gene scan back from Kaggle.

    python scripts/14_import_finetuned_tracks.py ~/Downloads/finetuned_scan_tracks.zip
    python scripts/14_import_finetuned_tracks.py <zip> --no-report     # import only, do not run 11

kaggle/finetune_scan_genes.ipynb fine-tunes NT-v2-50M (Run B of finetune_ntv2.ipynb)
and window-scans the 15 genes with it, exactly as `10_scan_gene.py --context window`
does with the frozen model. It writes finetuned_scan_tracks.zip:

    manifest.json                                   Run B test accuracy, versions, per-gene facts
    <GENE>/<track>.scores.npy, <track>.units.npy    same format as 10 writes
    <GENE>/scan_meta.finetuned.json                 the scan_meta.json entry for the track
    <track> = <model>.center_plus_prev_concat.finetuned.window

This script checks the zip against the local scans BEFORE writing anything:
  * not a SMOKE run, every gene of data/genes/scan_gene_list.json present;
  * each gene was scanned on the same sequence (sha256 of the scanned sequence
    equals data/scan/<GENE>/gene.json) and at the same positions as the frozen
    "NT-v2, 600-nt windows" track (identical units array), so the two tracks
    score the very same windows;
  * scores are (L, 3) float32, finite and summing to 1 where scored, NaN elsewhere.
Then it copies the two track files into data/scan/<GENE>/, adds the track to each
scan_meta.json (which is how 11_scan_report.py finds tracks), keeps the manifest
as results/finetuned_scan_manifest.json, and runs 11_scan_report.py.

11 then reports the new track like any other, and adds two paired comparisons on
the same gene-bootstrap resamples: fine-tuned vs the frozen current head (same
training windows, same scan windows: the comparison that matters) and fine-tuned
vs the frozen hard-negative head (reference). This script prints those rows.

Nothing is deleted. If a fine-tuned track is already there and differs, the old
files are moved to superseded/finetuned_tracks_<time>/ first. The report files 11
is about to rewrite (position_scan.json / _summary.md) are copied to
superseded/results_before_finetuned_import_<time>/.
"""

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

from phase1_common import (DATA_DIR, MODELS, RESULTS_DIR, ROOT, SCRIPTS_DIR, model_slug,
                           read_json, write_json_atomic)

SCAN_DIR = DATA_DIR / "scan"
GENE_LIST = DATA_DIR / "genes" / "scan_gene_list.json"
SUPERSEDED = ROOT / "superseded"
MANIFEST_OUT = RESULTS_DIR / "finetuned_scan_manifest.json"
NT = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
VARIANT = "center_plus_prev_concat"
HEAD = "finetuned"
TRACK = f"{model_slug(NT)}.{VARIANT}.{HEAD}.window"
FROZEN = f"{model_slug(NT)}.{VARIANT}.window"            # the frozen current-head window track


def npy(z, name):
    return np.load(io.BytesIO(z.read(name)), allow_pickle=False)


def validate(z):
    """Everything that must hold before a single file is written. Returns (manifest, per-gene data)."""
    names = set(z.namelist())
    if "manifest.json" not in names:
        raise SystemExit("no manifest.json in the zip: not a finetuned_scan_tracks.zip")
    man = json.loads(z.read("manifest.json"))
    if man.get("smoke"):
        raise SystemExit("this zip is from a SMOKE run (one gene, partial scan, tiny training "
                         "sample). Set SMOKE = False in the notebook and rerun; nothing imported.")
    if man.get("track") != TRACK or man.get("head") != HEAD or man.get("model") != NT:
        raise SystemExit(f"unexpected track in the zip: {man.get('track')!r} (expected {TRACK!r})")
    genes = [g["symbol"] for g in read_json(GENE_LIST)["genes"]]
    missing = [g for g in genes if g not in man.get("genes", {})]
    if missing:
        raise SystemExit(f"genes missing from the zip: {missing}; nothing imported")
    data = {}
    for sym in genes:
        gd = SCAN_DIR / sym
        gene = read_json(gd / "gene.json")
        if gene is None or not (gd / f"{FROZEN}.units.npy").exists():
            raise SystemExit(f"{sym}: local scan missing ({gd}); run 10_scan_gene.py --context window first")
        for suffix in ("scores.npy", "units.npy"):
            if f"{sym}/{TRACK}.{suffix}" not in names:
                raise SystemExit(f"{sym}/{TRACK}.{suffix} missing from the zip")
        sc, un = npy(z, f"{sym}/{TRACK}.scores.npy"), npy(z, f"{sym}/{TRACK}.units.npy")
        L = gene["length_scanned"]
        m = man["genes"][sym]
        sha = hashlib.sha256(gene["seq_scanned"].encode()).hexdigest()
        if m["seq_scanned_sha256"] != sha or m["length_scanned"] != L:
            raise SystemExit(f"{sym}: Kaggle scanned a different sequence than data/scan/{sym}/gene.json")
        if sc.shape != (L, 3) or sc.dtype != np.float32 or un.shape != (L,):
            raise SystemExit(f"{sym}: wrong array shape/dtype: scores {sc.shape} {sc.dtype}, units {un.shape}")
        frozen_un = np.load(gd / f"{FROZEN}.units.npy")
        if not np.array_equal(un, frozen_un):
            raise SystemExit(f"{sym}: scored positions differ from the frozen window track "
                             f"({int((un >= 0).sum()):,} vs {int((frozen_un >= 0).sum()):,} positions)")
        scored = un >= 0
        if not np.isfinite(sc[scored]).all() or not np.isnan(sc[~scored]).all():
            raise SystemExit(f"{sym}: scores are not finite exactly at the scored positions")
        if not np.allclose(sc[scored].sum(1), 1.0, atol=1e-3):
            raise SystemExit(f"{sym}: class scores do not sum to 1")
        meta = json.loads(z.read(f"{sym}/scan_meta.finetuned.json"))
        if list(meta) != [TRACK] or meta[TRACK].get("head") != HEAD or meta[TRACK].get("model") != NT:
            raise SystemExit(f"{sym}: unexpected scan_meta entry {list(meta)}")
        data[sym] = (sc, un, meta[TRACK])
    return man, data


def fmt(x, d):
    return f"{x['value']:+.{d}f} [{x['ci95'][0]:+.{d}f}, {x['ci95'][1]:+.{d}f}] {x['verdict']}"


def print_numbers():
    """The rows that matter, read back from results/position_scan.json (written by 11)."""
    ps = read_json(RESULTS_DIR / "position_scan.json")
    T, P = ps["tracks"], ps["pooled"]

    def find(context, head):
        return next((lb for lb, t in T.items() if t["repo"] == NT and t["variant"] == VARIANT
                     and t["context"] == context and t["head"] == head), None)
    rows = [("frozen, current head", find("window", "original")),
            ("frozen, hard-negative head", find("window", "hardneg")),
            ("fine-tuned", find("window", HEAD))]
    if rows[-1][1] is None:
        raise SystemExit("11_scan_report.py ran but the fine-tuned track is not in position_scan.json; "
                         f"see 'tracks_skipped_incomplete': {ps.get('tracks_skipped_incomplete')}")
    print(f"\nNT-v2-50M, 600-nt windows, {len(ps['genes'])} genes (pooled, 95% gene-bootstrap intervals)")
    print(f"{'':<28}{'top-k donors':>24}{'top-k acceptors':>24}{'AUROC donors':>26}{'AUROC acceptors':>26}")
    for title, lb in rows:
        if lb is None:
            continue
        d, a = P[lb]["donor"], P[lb]["acceptor"]

        def cell(v, dd):
            return f"{v['value']:.{dd}f} [{v['ci95'][0]:.{dd}f}, {v['ci95'][1]:.{dd}f}]"
        print(f"{title:<28}{d['top_k']['hits']:>4}/{d['top_k']['k']:<4}{cell(d['top_k'], 2):>15}"
              f"{a['top_k']['hits']:>5}/{a['top_k']['k']:<4}{cell(a['top_k'], 2):>15}"
              f"{cell(d['auroc'], 3):>26}{cell(a['auroc'], 3):>26}")
    print("\nPaired differences on the same gene resamples (new minus baseline; 'higher'/'lower' = "
          "the 95% interval excludes 0):")
    for c in ps["comparisons"]:
        if c["kind"] != "finetune":
            continue
        print(f"\n{c['title']}")
        for key, label, d in (("top_k_donor", "top-k donors", 2), ("top_k_acceptor", "top-k acceptors", 2),
                              ("auroc_donor", "AUROC donors", 3), ("auroc_acceptor", "AUROC acceptors", 3)):
            print(f"    {label:<16} {fmt(c['delta'][key], d)}")
    man = read_json(MANIFEST_OUT) or {}
    rb = man.get("run_b", {})
    print(f"\nFine-tuned model behind this track: Run B test accuracy {rb.get('test_accuracy')} on "
          f"{rb.get('n_test')} test windows (best epoch {rb.get('best_epoch')} of {rb.get('epochs')}); "
          f"source: {MANIFEST_OUT.relative_to(ROOT)}")
    print(f"Full report: {(RESULTS_DIR / 'position_scan_summary.md').relative_to(ROOT)}, "
          f"figure results/scan_tracks_ci.png")


def main():
    ap = argparse.ArgumentParser(description="Import the fine-tuned scan tracks from Kaggle and rerun 11.")
    ap.add_argument("zip", type=Path, help="finetuned_scan_tracks.zip downloaded from Kaggle")
    ap.add_argument("--no-report", action="store_true", help="do not run 11_scan_report.py afterwards")
    args = ap.parse_args()
    if NT not in MODELS:
        raise SystemExit(f"{NT} is not in phase1_common.MODELS")
    if not args.zip.exists():
        raise SystemExit(f"{args.zip} not found")

    with zipfile.ZipFile(args.zip) as z:
        bad = z.testzip()
        if bad:
            raise SystemExit(f"zip is corrupt at {bad}; download it again")
        man, data = validate(z)
    n_pos = sum(int((un >= 0).sum()) for _, un, _ in data.values())
    print(f"{args.zip.name}: {len(data)} genes, {n_pos:,} scored positions, all checks passed "
          f"(same sequences and same positions as the frozen window track)")
    print(f"  Run B on Kaggle: test accuracy {man['run_b']['test_accuracy']}, device {man.get('gpu') or man.get('device')}, "
          f"fp16 {man.get('fp16_autocast')}")

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    moved = 0
    for sym, (sc, un, meta) in data.items():
        gd = SCAN_DIR / sym
        for suffix, arr in (("scores.npy", sc), ("units.npy", un)):
            dst = gd / f"{TRACK}.{suffix}"
            if dst.exists():
                if np.array_equal(np.load(dst), arr, equal_nan=True):
                    continue
                old = SUPERSEDED / f"finetuned_tracks_{stamp}" / sym
                old.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dst), str(old / dst.name))       # never delete: archive the old track
                moved += 1
            np.save(dst, arr)
        scan_meta = read_json(gd / "scan_meta.json", default={}) or {}
        scan_meta[TRACK] = meta
        write_json_atomic(gd / "scan_meta.json", scan_meta)
    write_json_atomic(MANIFEST_OUT, {**man, "imported_from": args.zip.name,
                                     "imported_at": datetime.now().isoformat(timespec="seconds")})
    print(f"  wrote {TRACK}.scores.npy / .units.npy into data/scan/<GENE>/ for {len(data)} genes, "
          f"updated each scan_meta.json, saved {MANIFEST_OUT.relative_to(ROOT)}"
          + (f"; {moved} older fine-tuned track files moved to superseded/finetuned_tracks_{stamp}/" if moved else ""))
    if args.no_report:
        print("Skipped 11_scan_report.py (--no-report).")
        return 0

    keep = SUPERSEDED / f"results_before_finetuned_import_{stamp}"
    for name in ("position_scan.json", "position_scan_summary.md"):
        if (RESULTS_DIR / name).exists():
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copy2(RESULTS_DIR / name, keep / name)
    print(f"  previous position_scan.json / _summary.md copied to {keep.relative_to(ROOT)}/")
    print("\nRunning 11_scan_report.py ...", flush=True)
    r = subprocess.run([sys.executable, str(SCRIPTS_DIR / "11_scan_report.py")], cwd=str(SCRIPTS_DIR))
    if r.returncode != 0:
        raise SystemExit(f"11_scan_report.py failed (exit {r.returncode}); the tracks are imported, "
                         "rerun it after fixing: python scripts/11_scan_report.py")
    print_numbers()
    return 0


if __name__ == "__main__":
    sys.exit(main())
