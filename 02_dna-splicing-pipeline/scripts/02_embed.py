#!/usr/bin/env python3
"""
Phase 1, step 2: turn every DNA window into one vector using a frozen DNA language model.

    python scripts/02_embed.py --model InstaDeepAI/nucleotide-transformer-v2-50m-multi-species
    python scripts/02_embed.py --model zhihan1996/DNABERT-2-117M
    python scripts/02_embed.py --model zhihan1996/DNABERT-2-117M --max-train -1 --max-test -1   # full set

How it works
    1. Take a class-balanced random subset (fixed seed) of train and test.
       Pass -1 to use every row.
    2. Tokenise in batches with padding and truncation, then run the model
       under torch.no_grad(). The model is never updated (frozen backbone).
    3. Average the last hidden layer over real tokens only. The attention mask
       decides which tokens count, so padding never enters the average.
    4. Save float32 arrays to data/emb/<model-slug>/{X_train,y_train,X_test,y_test}.npy

Interrupt-safe: every 500 sequences a checkpoint is written to
data/emb/<model-slug>/ckpt/. Rerun the same command and it carries on from
the last checkpoint. If the finished arrays already exist for the same
settings the script exits straight away (use --force to redo them).

Tokens per sequence and seconds per sequence go to results/embed_stats.json.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np

from phase1_common import (EMB_DIR, EMBED_STATS_JSON, SEED, load_splits, model_slug,
                           read_json, short_name, stratified_subsample, update_json,
                           write_json_atomic)
# Reuse the loaders that already work for these models (AutoModel vs
# AutoModelForMaskedLM, and models that return plain tuples).
from try_dna_models import get_hidden, load_model, to_kmers


def fmt_time(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class Progress:
    """tqdm-style progress line. Redraws in place on a terminal; when output
    goes to a log file it prints a fresh line every `log_every` seconds."""

    def __init__(self, label, total, done=0, log_every=20.0):
        self.label, self.total, self.done = label, total, done
        self.start_done = done
        self.t0 = time.time()
        self.last_print = 0.0
        self.log_every = log_every
        self.tty = sys.stdout.isatty()

    def update(self, n):
        self.done += n
        now = time.time()
        if self.tty or now - self.last_print >= self.log_every or self.done >= self.total:
            self.last_print = now
            self._print()

    def _print(self):
        elapsed = time.time() - self.t0
        new = self.done - self.start_done
        rate = new / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else float("nan")
        frac = self.done / self.total if self.total else 1.0
        bar = "#" * int(frac * 25)
        line = (f"[{self.label}] {self.done:>6}/{self.total} {frac:4.0%} |{bar:<25}| "
                f"{rate:5.1f} seq/s  elapsed {fmt_time(elapsed)}  "
                f"ETA {fmt_time(eta) if eta == eta else '?'}")
        if self.tty:
            print("\r" + line, end="", flush=True)
        else:
            print(line, flush=True)

    def close(self):
        if self.tty:
            print(flush=True)


def indices_digest(idx):
    return hashlib.sha1(np.ascontiguousarray(idx, dtype=np.int64).tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description="Embed splice windows with a frozen DNA model.")
    ap.add_argument("--model", required=True, help="Hugging Face repo id")
    ap.add_argument("--max-train", type=int, default=8000,
                    help="train rows to embed, class-stratified (-1 = all 30,000)")
    ap.add_argument("--max-test", type=int, default=2000,
                    help="test rows to embed, class-stratified (-1 = all 3,000)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--checkpoint-every", type=int, default=500,
                    help="write a resumable checkpoint every N sequences")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    ap.add_argument("--force", action="store_true", help="ignore existing outputs/checkpoints")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    if args.threads:
        torch.set_num_threads(args.threads)

    ds = load_splits()
    idx = {"train": stratified_subsample(ds["train"]["label"], args.max_train, args.seed),
           "test": stratified_subsample(ds["test"]["label"], args.max_test, args.seed)}

    out_dir = EMB_DIR / model_slug(args.model)
    ckpt_dir = out_dir / "ckpt"
    run_key = {
        "model": args.model, "seed": args.seed,
        "max_train": args.max_train, "max_test": args.max_test,
        "n_train": int(len(idx["train"])), "n_test": int(len(idx["test"])),
        "idx_train_sha1": indices_digest(idx["train"]),
        "idx_test_sha1": indices_digest(idx["test"]),
        "pooling": "mean over attention_mask==1 tokens of the last hidden layer",
    }

    print(f"Model: {args.model}")
    print(f"Rows : train {run_key['n_train']} / {len(ds['train'])}, "
          f"test {run_key['n_test']} / {len(ds['test'])}  (seed {args.seed})")
    print(f"Out  : {out_dir}")

    final = {n: out_dir / f"{n}.npy" for n in ("X_train", "y_train", "X_test", "y_test")}
    meta_path = out_dir / "meta.json"
    old_meta = read_json(meta_path)
    if (not args.force and all(p.exists() for p in final.values())
            and old_meta and old_meta.get("run_key") == run_key):
        print("Embeddings for these exact settings already exist; nothing to do "
              "(pass --force to recompute).")
        return 0

    ckpt_meta_path = ckpt_dir / "run_key.json"
    if ckpt_dir.exists():
        if args.force or read_json(ckpt_meta_path) != run_key:
            print("Existing checkpoint is for different settings (or --force): discarding it.")
            shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(ckpt_meta_path, run_key)

    # ----- model -----
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model, via = load_model(args.model)
    model.eval()
    load_seconds = time.time() - t_load
    limits = [getattr(tok, "model_max_length", None),
              getattr(model.config, "max_position_embeddings", None), 2048]
    max_len = int(min(x for x in limits if x and x < 1_000_000))
    kmer_input = "DNA_bert" in args.model         # DNABERT-1 needs space-separated 6-mers
    print(f"Loaded via {via} in {load_seconds:.1f}s; max_length={max_len}; "
          f"torch threads={torch.get_num_threads()}")

    def embed(seqs):
        texts = [to_kmers(s) if kmer_input else s for s in seqs]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len)
        hidden = get_hidden(model, enc, torch)              # (B, T, D), no_grad inside
        mask = enc["attention_mask"]
        if hidden.shape[:2] != mask.shape:
            raise RuntimeError(f"hidden {tuple(hidden.shape)} vs mask {tuple(mask.shape)}")
        m = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return pooled.float().cpu().numpy(), mask.sum(dim=1).cpu().numpy().astype(np.int32)

    # ----- embed both splits, chunk by chunk -----
    t_run = time.time()
    n_new = n_resumed = 0
    arrays = {}
    try:
        with torch.no_grad():
            for split in ("train", "test"):
                rows = ds[split].select(idx[split].tolist())
                seqs, labels = rows["sequence"], np.asarray(rows["label"], dtype=np.int64)
                n = len(seqs)
                starts = list(range(0, n, args.checkpoint_every))
                done_before = sum(min(args.checkpoint_every, n - s) for s in starts
                                  if (ckpt_dir / f"{split}_{s:07d}.npz").exists())
                if done_before:
                    print(f"[{split}] resuming: {done_before}/{n} already in checkpoints")
                n_resumed += done_before
                prog = Progress(split, n, done=done_before)
                for s in starts:
                    path = ckpt_dir / f"{split}_{s:07d}.npz"
                    if path.exists():
                        continue
                    chunk = seqs[s:s + args.checkpoint_every]
                    xs, ts = [], []
                    t_chunk = time.time()
                    for b in range(0, len(chunk), args.batch_size):
                        x, t = embed(chunk[b:b + args.batch_size])
                        xs.append(x)
                        ts.append(t)
                        prog.update(len(x))
                    secs = time.time() - t_chunk
                    tmp = ckpt_dir / f"{split}_{s:07d}.tmp.npz"
                    np.savez(tmp, X=np.concatenate(xs), ntok=np.concatenate(ts),
                             secs=np.float64(secs))
                    os.replace(tmp, path)
                    n_new += len(chunk)
                prog.close()

                parts = [np.load(ckpt_dir / f"{split}_{s:07d}.npz") for s in starts]
                arrays[split] = {
                    "X": np.concatenate([p["X"] for p in parts]).astype(np.float32),
                    "y": labels,
                    "ntok": np.concatenate([p["ntok"] for p in parts]),
                    "secs": float(sum(float(p["secs"]) for p in parts)),
                }
                if len(arrays[split]["X"]) != n:
                    raise RuntimeError(f"{split}: expected {n} rows, got {len(arrays[split]['X'])}")
    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoints are kept; rerun the same command to resume.")
        return 130

    # ----- save -----
    for split in ("train", "test"):
        np.save(final[f"X_{split}"], arrays[split]["X"])
        np.save(final[f"y_{split}"], arrays[split]["y"])
        np.save(out_dir / f"idx_{split}.npy", idx[split])

    ntok = np.concatenate([arrays[s]["ntok"] for s in ("train", "test")])
    total_secs = arrays["train"]["secs"] + arrays["test"]["secs"]
    n_total = len(ntok)
    stats = {
        "model": args.model,
        "short_name": short_name(args.model),
        "loaded_via": via,
        "embedding_dim": int(arrays["train"]["X"].shape[1]),
        "n_train": run_key["n_train"],
        "n_test": run_key["n_test"],
        "tokens_per_sequence": {
            "mean": round(float(ntok.mean()), 2), "median": float(np.median(ntok)),
            "min": int(ntok.min()), "max": int(ntok.max()),
            "note": "non-padding tokens incl. special tokens (CLS/SEP), 600-nt input",
        },
        "max_length": max_len,
        "n_truncated": int((ntok >= max_len).sum()),
        "seconds_per_sequence": round(total_secs / n_total, 5),
        "sequences_per_second": round(n_total / total_secs, 2),
        "embed_compute_seconds": round(total_secs, 1),
        "model_load_seconds": round(load_seconds, 1),
        "wall_seconds_this_run": round(time.time() - t_run, 1),
        "n_embedded_this_run": n_new,
        "n_resumed_from_checkpoint": n_resumed,
        "batch_size": args.batch_size,
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json_atomic(meta_path, {"run_key": run_key, "stats": stats})
    update_json(EMBED_STATS_JSON, [args.model], stats)
    shutil.rmtree(ckpt_dir)

    print(f"\nSaved {out_dir}/X_train.npy {arrays['train']['X'].shape}, "
          f"X_test.npy {arrays['test']['X'].shape}")
    print(f"tokens/sequence mean {stats['tokens_per_sequence']['mean']} "
          f"(min {stats['tokens_per_sequence']['min']}, max {stats['tokens_per_sequence']['max']}), "
          f"truncated {stats['n_truncated']}")
    print(f"{stats['seconds_per_sequence'] * 1000:.1f} ms/sequence "
          f"({stats['sequences_per_second']} seq/s), compute {fmt_time(total_secs)}")
    print(f"Stats -> {EMBED_STATS_JSON}")
    print(json.dumps({"tokens_per_sequence": stats["tokens_per_sequence"]["mean"],
                      "seconds_per_sequence": stats["seconds_per_sequence"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
