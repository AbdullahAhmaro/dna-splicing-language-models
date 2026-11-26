"""
Shared paths and helpers for the phase-1 splice-site pipeline (scripts 01-06).

Kept in one place so every step agrees on where files live, which test
sequences were used, and what each label value means.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent                      # dna-phase/
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
EMB_DIR = DATA_DIR / "emb"
DATASET_DIR = DATA_DIR / "splice_sites_all"    # datasets.save_to_disk() cache
LABEL_INFO_PATH = DATA_DIR / "label_info.json"
RESULTS_JSON = RESULTS_DIR / "phase1_results.json"
EMBED_STATS_JSON = RESULTS_DIR / "embed_stats.json"
RUN_STEPS_TSV = RESULTS_DIR / "run_steps.tsv"

# The genome-wide dataset of 16_genome_dataset.py (GRCh38 + MANE Select, same format).
GENOME_DIR = DATA_DIR / "genome"
GENOME_DEFAULT_DIR = GENOME_DIR / "splice_sites_genome"      # 120,000 balanced train + full test
GENOME_FULL_DIR = GENOME_DIR / "splice_sites_genome_full"    # every window, + *_meta.parquet

DATASET_REPO = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
DATASET_CONFIG = "splice_sites_all"
SEED = 42

# Display names used in tables and the chart.
MODELS = {
    "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species": "NT-v2-50M",
    "zhihan1996/DNABERT-2-117M": "DNABERT-2",
}


def short_name(repo: str) -> str:
    return MODELS.get(repo, repo.split("/")[-1])


def model_slug(repo: str) -> str:
    """'zhihan1996/DNABERT-2-117M' -> 'zhihan1996__DNABERT-2-117M' (safe as a folder name)."""
    return repo.replace("/", "__")


def load_splits():
    """The dataset as cached on disk by 01_get_data.py (no network needed)."""
    from datasets import load_from_disk

    if not DATASET_DIR.exists():
        raise SystemExit(f"{DATASET_DIR} not found. Run 01_get_data.py first.")
    return load_from_disk(str(DATASET_DIR))


def load_genome_split(split: str, columns=None, full: bool = False, meta: bool = False) -> dict:
    """One split of the genome dataset (16_genome_dataset.py) as a dict of columns:
    the default files (120,000-window balanced train, full chr20/21 test) unless full=True;
    meta=True reads the full split's per-row annotations (kind, group, gene, ...) instead."""
    import pyarrow.parquet as pq

    if meta:
        path = GENOME_FULL_DIR / f"{split}_meta.parquet"
    else:
        path = (GENOME_FULL_DIR if full else GENOME_DEFAULT_DIR) / f"{split}.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run 16_genome_dataset.py first.")
    return pq.read_table(path, columns=columns).to_pydict()


def genome_emb_dir(repo: str):
    """Where embeddings of the genome dataset go; kept apart from splice_sites_all's."""
    return EMB_DIR / model_slug(repo) / "genome"


def dataset_chromosomes() -> dict:
    """Chromosomes present in each dataset split, read from the `name` column
    (`chrom:start-end|label`), without the 'chr' prefix."""
    ds = load_splits()
    return {s: sorted({n.split(":")[0].removeprefix("chr") for n in ds[s]["name"]})
            for s in ds}


def stratified_subsample(labels, n: int, seed: int = SEED) -> np.ndarray:
    """
    Sorted row indices of a class-stratified random subsample of size n.

    n < 0 (or n >= len) means "use every row". Same labels + n + seed always
    give the same indices, so 02 (embedding) and 04 (baseline) score exactly
    the same test sequences.
    """
    labels = np.asarray(labels)
    total = len(labels)
    if n is None or n < 0 or n >= total:
        return np.arange(total)
    from sklearn.model_selection import train_test_split

    idx, _ = train_test_split(np.arange(total), train_size=n,
                              stratify=labels, random_state=seed)
    return np.sort(idx)


def read_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json_atomic(path: Path, obj) -> None:
    """Write via a temp file + rename so a crash never leaves half a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def update_json(path: Path, keys: list[str], value) -> None:
    """Set obj[keys[0]][keys[1]]... = value in a JSON file, keeping everything else."""
    obj = read_json(path, default={}) or {}
    node = obj
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
    write_json_atomic(path, obj)


def load_label_info() -> dict:
    info = read_json(LABEL_INFO_PATH)
    if info is None:
        raise SystemExit(f"{LABEL_INFO_PATH} not found. Run 01_get_data.py first.")
    return info


def label_names(info: dict) -> dict[int, str]:
    return {int(k): v for k, v in info["names"].items()}


def seq_matrix(seqs) -> np.ndarray:
    """Equal-length DNA strings -> (N, L) uint8 array of ASCII codes."""
    seqs = list(seqs)
    length = len(seqs[0])
    if any(len(s) != length for s in seqs):
        raise ValueError("sequences have different lengths")
    buf = "".join(seqs).upper().encode("ascii")
    return np.frombuffer(buf, dtype=np.uint8).reshape(len(seqs), length)


def dinuc_hits(mat: np.ndarray, dinuc: str) -> np.ndarray:
    """(N, L-1) bool: True where positions k, k+1 spell `dinuc`."""
    a, b = (ord(c) for c in dinuc.upper())
    return (mat[:, :-1] == a) & (mat[:, 1:] == b)


# Heads the per-position scan can apply: built from the centre token (C) and the
# token just before it (P). 12_repool_center.py built the P-based training
# vectors from the float16 copy of the tokens, so the scan rounds them the same way.
SCAN_VARIANTS = ("center_tok", "prev_tok", "center_plus_prev", "center_plus_prev_concat")
FROM_F16 = {"prev_tok", "center_plus_prev", "center_plus_prev_concat"}


def pair_features(C, P, variant):
    """Head input for a centre/previous-token variant from the two token vectors."""
    if variant in FROM_F16:
        C = C.astype(np.float16).astype(np.float32)
        P = P.astype(np.float16).astype(np.float32)
    if variant == "center_tok":
        return C
    if variant == "prev_tok":
        return P
    if variant == "center_plus_prev":
        return (C + P) / 2
    return np.concatenate([C, P], axis=-1)


HARDNEG_TSV = DATA_DIR / "hard_negatives" / "hard_negatives.tsv"


def load_hard_negatives(repo: str, variant: str) -> dict:
    """Hard-negative windows of 13_hard_negatives.py as head inputs for one variant:
    X (n, d), and per row its split (train / holdout), group and gene."""
    import csv

    d = EMB_DIR / model_slug(repo) / "hardneg"
    if not HARDNEG_TSV.exists() or not (d / "C.npy").exists():
        raise FileNotFoundError(f"hard negatives missing ({HARDNEG_TSV}, {d}); "
                                "run 13_hard_negatives.py first")
    with open(HARDNEG_TSV) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    C, P = np.load(d / "C.npy"), np.load(d / "P.npy")
    if len(rows) != len(C):
        raise ValueError(f"{d}: {len(C)} embedded windows but {len(rows)} TSV rows; rerun 13")
    return {"X": pair_features(C, P, variant),
            "split": np.array([r["split"] for r in rows]),
            "group": np.array([r["group"] for r in rows]),
            "gene": np.array([r["gene_id"] for r in rows])}
