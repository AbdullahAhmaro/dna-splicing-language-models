#!/usr/bin/env python3
"""
Download the splice-site dataset and show what is inside it.

This is the data for the task the advisor set: given a DNA sequence, say
where it should be cut. Each example is a short DNA sequence with a label:
donor site, acceptor site, or not a splice site.

Usage:
    python scripts/get_splice_data.py
"""

from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DATASET = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
CONFIG = "splice_sites_all"


def main():
    from datasets import load_dataset

    print(f"Downloading {DATASET} / {CONFIG} ...")
    ds = load_dataset(DATASET, CONFIG)
    print(ds)

    for split in ds:
        rows = ds[split]
        print(f"\n--- {split}: {len(rows)} examples ---")
        print("columns:", rows.column_names)

        first = rows[0]
        for key, value in first.items():
            shown = value if not isinstance(value, str) or len(value) <= 80 \
                else value[:80] + f"... ({len(value)} letters)"
            print(f"  {key}: {shown}")

        label_col = next(
            (c for c in rows.column_names if c.lower() in ("label", "labels")), None
        )
        if label_col:
            counts = {}
            for v in rows[label_col]:
                counts[v] = counts.get(v, 0) + 1
            print(f"  label counts: {dict(sorted(counts.items()))}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "splice_sites_sample.csv"
    ds["train"].select(range(min(200, len(ds["train"])))).to_csv(out, index=False)
    print(f"\nFirst 200 training rows saved to {out}")
    print("Open it in a spreadsheet to see the sequences and their labels.")


if __name__ == "__main__":
    main()
