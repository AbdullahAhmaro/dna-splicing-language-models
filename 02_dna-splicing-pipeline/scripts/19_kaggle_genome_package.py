#!/usr/bin/env python3
"""
Step 19: a small upload package of the genome dataset for kaggle/genome_nt2500m.ipynb.

    python scripts/19_kaggle_genome_package.py

Reads the default files of 16_genome_dataset.py (the 120,000-window class-balanced train file and
the full chr20/21 test file, 18,653 windows) and keeps only what the notebook needs:

  genome_train.parquet  sequence, label, chrom                               (row order = default train)
  genome_test.parquet   sequence, label, chrom, gene_id, gene, group         (row order = default test)
  manifest.json         row and class counts, column meanings, sha256 of every package file
                        and of the source parquets

gene_id is the cluster the notebook's bootstrap resamples (genes, not windows); group says which
kind of window each row is (donor, acceptor, decoy_GT, decoy_AG, exon_interior, intron_interior),
from the full split's test_meta.parquet.

Output: data/genome/kaggle_upload/genome-splice-sites/ (git-ignored, like all of data/).
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from phase1_common import DATA_DIR, GENOME_DEFAULT_DIR, GENOME_DIR, load_label_info, write_json_atomic

OUT = GENOME_DIR / "kaggle_upload" / "genome-splice-sites"
TEST_CHROMS = {"chr20", "chr21"}
GROUP = {("decoy", "donor-like (GT after)"): "decoy_GT",
         ("decoy", "acceptor-like (AG before)"): "decoy_AG"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(table, path):
    pq.write_table(table, path, compression="zstd", compression_level=19)


def main():
    argparse.ArgumentParser(description="Kaggle upload package of the genome dataset.").parse_args()
    names = {int(k): v for k, v in load_label_info()["names"].items()}
    src = {s: GENOME_DEFAULT_DIR / f"{s}.parquet" for s in ("train", "test")}
    meta_path = GENOME_DIR / "splice_sites_genome_full" / "test_meta.parquet"
    OUT.mkdir(parents=True, exist_ok=True)

    tr = pq.read_table(src["train"], columns=["sequence", "label", "chrom", "name"])
    te = pq.read_table(src["test"], columns=["sequence", "label", "chrom", "name"])
    meta = pq.read_table(meta_path, columns=["name", "label", "group", "subtype", "gene", "gene_id"])

    # checks before anything is written
    assert tr.num_rows == 120_000, tr.num_rows
    assert set(tr["chrom"].to_pylist()).isdisjoint(TEST_CHROMS), "a train window comes from chr20/21"
    assert set(te["chrom"].to_pylist()) <= TEST_CHROMS
    assert meta["name"].to_pylist() == te["name"].to_pylist(), "test_meta rows do not line up with test"
    assert meta["label"].to_pylist() == te["label"].to_pylist()
    for t in (tr, te):
        seqs = t["sequence"].to_pylist()
        assert all(len(s) == 600 for s in seqs)
    assert not set(te["sequence"].to_pylist()) & set(tr["sequence"].to_pylist()), "test sequence in train"
    group = [GROUP.get((g, s), g) for g, s in zip(meta["group"].to_pylist(), meta["subtype"].to_pylist())]
    assert set(group) == {"donor", "acceptor", "decoy_GT", "decoy_AG", "exon_interior", "intron_interior"}
    assert all(g for g in meta["gene_id"].to_pylist()), "a test window has no gene"

    train = pa.table({"sequence": tr["sequence"],
                      "label": pa.array(tr["label"].to_numpy().astype(np.int8)),
                      "chrom": tr["chrom"].dictionary_encode()})
    test = pa.table({"sequence": te["sequence"],
                     "label": pa.array(te["label"].to_numpy().astype(np.int8)),
                     "chrom": te["chrom"].dictionary_encode(),
                     "gene_id": meta["gene_id"].dictionary_encode(),
                     "gene": meta["gene"].dictionary_encode(),
                     "group": pa.array(group).dictionary_encode()})
    files = {"genome_train.parquet": train, "genome_test.parquet": test}
    for f, t in files.items():
        write(t, OUT / f)

    def counts(t, col):
        v, n = np.unique(np.asarray(t[col].to_pylist()), return_counts=True)
        return {str(a): int(b) for a, b in zip(v, n)}

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "built_by": "02_dna-splicing-pipeline/scripts/19_kaggle_genome_package.py",
        "dataset": "genome-wide splice-site windows (GRCh38, MANE Select v1.3; negatives checked "
                   "against GENCODE v50), built by scripts/16_genome_dataset.py",
        "labels": {str(k): v for k, v in names.items()},
        "window": "600 nt, genome + strand; the label is for base 300 (0-based)",
        "split": "train = every chromosome except chr20 and chr21 (chr1-19, chr22, X, Y; the 120,000-window "
                 "class-balanced default file); "
                 "test = every window on chr20 and chr21",
        "columns": {
            "sequence": "the 600-nt window",
            "label": "0 none, 1 acceptor, 2 donor",
            "chrom": "chromosome of the window",
            "gene_id": "(test) NCBI GeneID of the MANE gene the window was taken from; "
                       "the notebook's bootstrap resamples these",
            "gene": "(test) gene symbol",
            "group": "(test) donor, acceptor, or the kind of negative: decoy_GT (GT at the donor "
                     "offset, not a site in any GENCODE transcript), decoy_AG (AG at the acceptor "
                     "offset, same), exon_interior, intron_interior",
        },
        "files": {f: {"rows": t.num_rows, "columns": t.column_names,
                      "bytes": (OUT / f).stat().st_size, "sha256": sha256(OUT / f),
                      "labels": counts(t, "label")} for f, t in files.items()},
        "test_groups": counts(test, "group"),
        "test_genes": len(set(meta["gene_id"].to_pylist())),
        "sources": {str(p.relative_to(DATA_DIR.parent)): sha256(p)
                    for p in (src["train"], src["test"], meta_path)},
    }
    write_json_atomic(OUT / "manifest.json", manifest)
    total = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
    for f, v in manifest["files"].items():
        print(f"{f}: {v['rows']:,} rows, {v['bytes'] / 1e6:.1f} MB, labels {v['labels']}")
    print(f"test groups {manifest['test_groups']}, {manifest['test_genes']} genes")
    print(f"package: {OUT} ({total / 1e6:.1f} MB in {len(list(OUT.iterdir()))} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
