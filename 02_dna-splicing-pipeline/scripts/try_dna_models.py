#!/usr/bin/env python3
"""
Try several small DNA language models and report which ones actually run
on this machine (CPU only). Saves the embeddings of the first one that works.

Usage:
    python scripts/try_dna_models.py                 # try all candidates
    python scripts/try_dna_models.py --only nt50     # try one by short name
    python scripts/try_dna_models.py --list          # just show the candidates

Why a list and not one model: several DNA models ship their own Python code
on the Hub ("trust_remote_code"), written against older versions of
transformers. Some of them break on new versions. Rather than guess, we try
them in order of how likely they are to just work and report what happened.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# A 600-letter human DNA fragment (random-looking but fixed, so every run is
# comparable). Replace with a real sequence any time; nothing depends on it.
TEST_SEQ = (
    "ATGGCCTTTGAAGTCCTGACCAGGTAAGTCTGCAGACCTTGGCTAACTTGCAGGTCAGTTCCTGGAAGCT"
    "GACCCAGGTGAGTACCTGGCCTTAGCAGTCCTGAGGTCAGCTTGGACCTTGCAGGTAACCTGAGTCCTGA"
    "GCTTAGGCTGACCTTGAGGTCAGCCTGGAAGTCCTGACCAGGTAAGTCTGCAGACCTTGGCTAACTTGCA"
    "GGTCAGTTCCTGGAAGCTGACCCAGGTGAGTACCTGGCCTTAGCAGTCCTGAGGTCAGCTTGGACCTTGC"
    "AGGTAACCTGAGTCCTGAGCTTAGGCTGACCTTGAGGTCAGCCTGGAAGTCCTGACCAGGTAAGTCTGCA"
    "GACCTTGGCTAACTTGCAGGTCAGTTCCTGGAAGCTGACCCAGGTGAGTACCTGGCCTTAGCAGTCCTGA"
    "GGTCAGCTTGGACCTTGCAGGTAACCTGAGTCCTGAGCTTAGGCTGACCTTGAGGTCAGCCTGGAAGTCC"
    "TGACCAGGTAAGTCTGCAGACCTTGGCTAACTTGCAGGTCAGTTCCTGGAAGCTGACCCAGGTGAGTACC"
    "TGGCCTTAGCAGTCCTGAGGTCAGCTTGGACCTTGCAGGTAACCTGAGTCCTGAGCTTAGGCTGACC"
)

# Ordered most-likely-to-work first.
CANDIDATES = [
    {
        "key": "nt50",
        "repo": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        "params": "50M",
        "tokens": "6-mers",
        "note": "Smallest Nucleotide Transformer. Plain PyTorch, no triton.",
    },
    {
        "key": "nt100",
        "repo": "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        "params": "100M",
        "tokens": "6-mers",
        "note": "Same family, one size up (~400 MB download).",
    },
    {
        "key": "dnabert1",
        "repo": "zhihan1996/DNA_bert_6",
        "params": "~89M",
        "tokens": "6-mers (space separated)",
        "note": "Original DNABERT. Input must be pre-split into 6-mers.",
    },
    {
        "key": "dnabert2",
        "repo": "zhihan1996/DNABERT-2-117M",
        "params": "117M",
        "tokens": "BPE",
        "note": "The one the advisor named. Needs: pip install einops",
    },
]


def to_kmers(seq, k=6):
    """DNABERT-1 wants 'ATGGCC TGGCCT GGCCTT ...' instead of a raw string."""
    return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))


def load_model(repo):
    """
    Load a model, trying each of the Auto classes in turn.

    Why: a model on the Hub only registers the Auto classes its authors chose.
    The Nucleotide Transformer registers AutoModelForMaskedLM but not AutoModel,
    so asking for AutoModel gives "Unrecognized configuration class". Trying
    both covers either case. Returns (model, which_class_worked).
    """
    from transformers import AutoModel, AutoModelForMaskedLM

    errors = []
    for cls in (AutoModel, AutoModelForMaskedLM):
        try:
            model = cls.from_pretrained(repo, trust_remote_code=True)
            return model, cls.__name__
        except Exception as exc:                # noqa: BLE001
            errors.append(f"{cls.__name__}: {type(exc).__name__}: {exc}")
    raise RuntimeError(" | ".join(errors))


def get_hidden(model, inputs, torch):
    """
    Pull the per-token feature vectors out, whatever shape the model returns.

    A plain AutoModel hands back last_hidden_state directly. A masked-language
    model hands back word-score predictions instead, so we ask it for its
    hidden states and take the last layer.
    """
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state
    if getattr(out, "hidden_states", None):
        return out.hidden_states[-1]
    if isinstance(out, (tuple, list)):
        return out[0]
    raise RuntimeError("could not find per-token vectors in the model output")


def try_one(cand, seq, save_embeddings):
    import torch
    from transformers import AutoTokenizer

    repo = cand["repo"]
    print(f"\n=== {cand['key']}: {repo} ({cand['params']}) ===")
    print(f"    {cand['note']}")

    record = dict(cand)
    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        model, via = load_model(repo)
        model.eval()
        load_s = time.time() - t0

        text = to_kmers(seq) if "DNA_bert" in repo else seq
        inputs = tok(text, return_tensors="pt", truncation=True)
        n_tokens = inputs["input_ids"].shape[1]
        max_len = getattr(tok, "model_max_length", None)
        truncated = bool(max_len and max_len < 1e6 and n_tokens >= max_len)

        t1 = time.time()
        hidden = get_hidden(model, inputs, torch)
        infer_s = time.time() - t1

        per_token = hidden[0]                 # (tokens, dim)
        whole_seq = per_token.mean(dim=0)     # (dim,)

        print(f"    OK  tokens={n_tokens}  embedding shape={tuple(per_token.shape)}"
              f"  (loaded via {via})")
        if truncated:
            print(f"        NOTE: input was cut at the model's limit of {max_len} "
                  f"tokens. Longer sequences will be clipped.")
        print(f"        load {load_s:.1f}s   forward pass {infer_s:.2f}s")
        print(f"        first 5 numbers of token 1: "
              f"{[round(float(x), 4) for x in per_token[0, :5]]}")

        record.update(
            status="ok",
            loaded_via=via,
            n_tokens=int(n_tokens),
            truncated=truncated,
            max_tokens=int(max_len) if truncated else None,
            embedding_dim=int(per_token.shape[-1]),
            load_seconds=round(load_s, 1),
            forward_seconds=round(infer_s, 2),
        )

        if save_embeddings:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = RESULTS_DIR / f"{cand['key']}_embeddings.pt"
            torch.save(
                {"repo": repo, "sequence": seq,
                 "per_token": per_token, "whole_sequence": whole_seq},
                out_path,
            )
            print(f"        saved -> {out_path}")
            record["saved_to"] = str(out_path)

    except Exception as exc:                    # noqa: BLE001 - we want every failure
        reason = f"{type(exc).__name__}: {exc}"
        print(f"    FAILED  {reason.splitlines()[0][:300]}")
        record.update(status="failed", error=reason[:2000])

    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run a single candidate by key")
    ap.add_argument("--list", action="store_true", help="list candidates and exit")
    ap.add_argument("--seq", help="use your own DNA sequence instead of the test one")
    args = ap.parse_args()

    if args.list:
        for c in CANDIDATES:
            print(f"{c['key']:10s} {c['params']:6s} {c['repo']}")
        return 0

    seq = (args.seq or TEST_SEQ).strip().upper()
    todo = [c for c in CANDIDATES if not args.only or c["key"] == args.only]
    if not todo:
        print(f"No candidate named {args.only!r}. Use --list to see the keys.")
        return 2

    import torch
    import transformers
    print(f"python {platform.python_version()}  torch {torch.__version__}  "
          f"transformers {transformers.__version__}")
    print(f"sequence length: {len(seq)} letters")

    records, saved_one = [], False
    for cand in todo:
        rec = try_one(cand, seq, save_embeddings=not saved_one)
        if rec["status"] == "ok" and not saved_one:
            saved_one = True
        records.append(rec)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = RESULTS_DIR / "model_report.json"
    report.write_text(json.dumps(
        {"python": platform.python_version(),
         "torch": torch.__version__,
         "transformers": transformers.__version__,
         "sequence_length": len(seq),
         "results": records}, indent=2))

    ok = [r for r in records if r["status"] == "ok"]
    print("\n" + "=" * 60)
    print(f"{len(ok)} of {len(records)} models ran. Report: {report}")
    for r in records:
        mark = "ok    " if r["status"] == "ok" else "FAILED"
        print(f"  {mark}  {r['key']:10s} {r['repo']}")
    if not ok:
        print("\nNothing ran. Send the report file to Claude and it will sort it out.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
