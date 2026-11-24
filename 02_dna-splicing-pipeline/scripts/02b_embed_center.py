#!/usr/bin/env python3
"""
Phase 1, step 2b: position-aware pooling around the centre of each window.

    python scripts/02b_embed_center.py --model InstaDeepAI/nucleotide-transformer-v2-50m-multi-species
    python scripts/02b_embed_center.py --model zhihan1996/DNABERT-2-117M

Why: the label depends on the ~2 bases at the window centre (offsets 298-302),
and averaging all ~100 tokens (02_embed.py) dilutes that signal. This script
maps every token back to the bases it covers and pools only near the centre.

Token -> base mapping
    fast tokenizer (DNABERT-2): return_offsets_mapping
    slow tokenizer (NT-v2):     cumulative lengths of convert_ids_to_tokens(),
                                asserting they add up to the sequence length
                                and spell the sequence exactly.

Variants (base ranges are 0-based and inclusive; a token counts if it
overlaps the range by at least one base). All come from ONE forward pass:
    center_tok       the single token covering base 300
    center_w10       mean over tokens overlapping bases 290-310
    center_w25       mean over tokens overlapping bases 275-325
    center_w50       mean over tokens overlapping bases 250-350
    concat_c25_full  center_w25 ++ the full-window mean (same definition as 02)
Each is saved to data/emb/<model-slug>/<variant>/{X,y,idx}_{train,test}.npy

Raw per-token hidden states for tokens overlapping bases 240-360 are also
saved, once, as float16:
    data/emb/<model-slug>/center_tokens.npy        (total_tokens, hidden_dim)
    data/emb/<model-slug>/center_tokens_index.npz  ptr / n_tokens per sequence,
        split, dataset row, label, and each token's base span (tok_start, tok_end)
With these, any other pooling width inside 240-360 needs no new forward pass.

Same seeded 8,000 / 2,000 class-stratified subsample as 02 (phase1_common).
Checkpoints every 500 sequences and resumes after an interruption, like 02.
"""

import argparse
import importlib
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
from try_dna_models import get_hidden, load_model

_emb02 = importlib.import_module("02_embed")       # reuse Progress / helpers, unmodified
Progress, fmt_time, indices_digest = _emb02.Progress, _emb02.fmt_time, _emb02.indices_digest

CENTER_BASE = 300
WINDOWS = {"center_w10": (290, 310), "center_w25": (275, 325), "center_w50": (250, 350)}
RAW_WINDOW = (240, 360)
POOLED = ["full_mean", "center_tok", *WINDOWS]            # computed per sequence
VARIANTS = ["center_tok", *WINDOWS, "concat_c25_full"]    # saved as variant folders


def main():
    ap = argparse.ArgumentParser(description="Centre-pooled embeddings (one forward pass).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-train", type=int, default=8000)
    ap.add_argument("--max-test", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    if args.threads:
        torch.set_num_threads(args.threads)

    ds = load_splits()
    idx = {"train": stratified_subsample(ds["train"]["label"], args.max_train, args.seed),
           "test": stratified_subsample(ds["test"]["label"], args.max_test, args.seed)}
    out_dir = EMB_DIR / model_slug(args.model)
    ckpt_dir = out_dir / "ckpt_center"
    run_key = {
        "model": args.model, "seed": args.seed,
        "max_train": args.max_train, "max_test": args.max_test,
        "n_train": int(len(idx["train"])), "n_test": int(len(idx["test"])),
        "idx_train_sha1": indices_digest(idx["train"]),
        "idx_test_sha1": indices_digest(idx["test"]),
        "center_base": CENTER_BASE, "windows_inclusive": WINDOWS, "raw_window": RAW_WINDOW,
    }
    # JSON turns tuples into lists; normalise now so the saved key compares equal.
    run_key = json.loads(json.dumps(run_key))
    print(f"Model: {args.model}")
    print(f"Rows : train {run_key['n_train']}, test {run_key['n_test']} (seed {args.seed})")

    meta_path = out_dir / "center_meta.json"
    expected = [out_dir / v / f"{n}.npy" for v in VARIANTS
                for n in ("X_train", "y_train", "X_test", "y_test")]
    expected += [out_dir / "center_tokens.npy", out_dir / "center_tokens_index.npz"]
    old = read_json(meta_path)
    if not args.force and old and old.get("run_key") == run_key and all(p.exists() for p in expected):
        print("Centre-pooled embeddings for these settings already exist; nothing to do "
              "(--force to recompute).")
        return 0

    if ckpt_dir.exists() and (args.force or read_json(ckpt_dir / "run_key.json") != run_key):
        print("Existing centre checkpoint is for different settings (or --force): discarding.")
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(ckpt_dir / "run_key.json", run_key)

    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model, via = load_model(args.model)
    model.eval()
    limits = [getattr(tok, "model_max_length", None),
              getattr(model.config, "max_position_embeddings", None), 2048]
    max_len = int(min(x for x in limits if x and x < 1_000_000))
    special_ids = np.array(sorted(set(tok.all_special_ids)))
    mapping = "offset_mapping (fast tokenizer)" if tok.is_fast else \
        "reconstructed from convert_ids_to_tokens lengths (slow tokenizer)"
    print(f"Loaded via {via} in {time.time() - t_load:.1f}s; token->base mapping: {mapping}")

    def spans(enc, i, seq):
        """Per-token [start, end) base offsets for row i; -1 for special/pad tokens."""
        am = enc["attention_mask"][i].numpy().astype(bool)
        # Special tokens are found by id: NT's slow tokenizer returns a
        # special_tokens_mask that assumes an EOS token it never adds (off by one).
        body = am & ~np.isin(enc["input_ids"][i].numpy(), special_ids)
        start = np.full(len(am), -1, dtype=np.int64)
        end = np.full(len(am), -1, dtype=np.int64)
        if tok.is_fast:
            off = enc["offset_mapping"][i].numpy()
            start[body], end[body] = off[body, 0], off[body, 1]
        else:
            pieces = tok.convert_ids_to_tokens(enc["input_ids"][i][body].tolist())
            lens = np.array([len(p) for p in pieces])
            assert int(lens.sum()) == len(seq), \
                f"reconstructed length {int(lens.sum())} != sequence length {len(seq)}"
            assert "".join(pieces) == seq, "reconstructed tokens do not spell the sequence"
            end[body] = np.cumsum(lens)
            start[body] = end[body] - lens
        assert start[body].min() == 0 and end[body].max() == len(seq), "tokens do not cover the sequence"
        return am, body, start, end

    def embed(seqs):
        enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_len,
                  return_offsets_mapping=tok.is_fast)
        model_inputs = {k: v for k, v in enc.items() if k != "offset_mapping"}
        hidden = get_hidden(model, model_inputs, torch).float().cpu().numpy()   # (B, T, D)
        out = {k: [] for k in POOLED}
        raw, raw_n, raw_s, raw_e, n_pooled = [], [], [], [], {k: [] for k in POOLED}
        for i, seq in enumerate(seqs):
            am, body, start, end = spans(enc, i, seq)
            h = hidden[i]
            sel = {"full_mean": am}                       # identical definition to 02_embed.py
            sel["center_tok"] = body & (start <= CENTER_BASE) & (end > CENTER_BASE)
            assert sel["center_tok"].sum() == 1, "expected exactly one token covering base 300"
            for name, (lo, hi) in WINDOWS.items():
                sel[name] = body & (start <= hi) & (end > lo)
            for k in POOLED:
                out[k].append(h[sel[k]].mean(axis=0))
                n_pooled[k].append(int(sel[k].sum()))
            r = body & (start <= RAW_WINDOW[1]) & (end > RAW_WINDOW[0])
            raw.append(h[r].astype(np.float16))
            raw_n.append(int(r.sum()))
            raw_s.append(start[r])
            raw_e.append(end[r])
        return ({k: np.stack(v).astype(np.float32) for k, v in out.items()},
                np.concatenate(raw), np.array(raw_n), np.concatenate(raw_s),
                np.concatenate(raw_e), {k: np.array(v) for k, v in n_pooled.items()})

    t_run = time.time()
    n_new = 0
    data = {}
    try:
        with torch.no_grad():
            for split in ("train", "test"):
                rows = ds[split].select(idx[split].tolist())
                seqs, labels = rows["sequence"], np.asarray(rows["label"], dtype=np.int64)
                n = len(seqs)
                starts = list(range(0, n, args.checkpoint_every))
                done = sum(min(args.checkpoint_every, n - s) for s in starts
                           if (ckpt_dir / f"{split}_{s:07d}.npz").exists())
                if done:
                    print(f"[{split}] resuming: {done}/{n} already in checkpoints")
                prog = Progress(split, n, done=done)
                for s in starts:
                    path = ckpt_dir / f"{split}_{s:07d}.npz"
                    if path.exists():
                        continue
                    chunk = seqs[s:s + args.checkpoint_every]
                    parts = []
                    t0 = time.time()
                    for b in range(0, len(chunk), args.batch_size):
                        parts.append(embed(chunk[b:b + args.batch_size]))
                        prog.update(len(parts[-1][2]))
                    pooled = {k: np.concatenate([p[0][k] for p in parts]) for k in POOLED}
                    npool = {k: np.concatenate([p[5][k] for p in parts]) for k in POOLED}
                    tmp = ckpt_dir / f"{split}_{s:07d}.tmp.npz"
                    np.savez(tmp, **{f"X_{k}": v for k, v in pooled.items()},
                             **{f"n_{k}": v for k, v in npool.items()},
                             raw=np.concatenate([p[1] for p in parts]),
                             raw_n=np.concatenate([p[2] for p in parts]),
                             raw_start=np.concatenate([p[3] for p in parts]).astype(np.int16),
                             raw_end=np.concatenate([p[4] for p in parts]).astype(np.int16),
                             secs=np.float64(time.time() - t0))
                    os.replace(tmp, path)
                    n_new += len(chunk)
                prog.close()
                data[split] = {"files": [ckpt_dir / f"{split}_{s:07d}.npz" for s in starts],
                               "y": labels, "n": n}
    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoints are kept; rerun the same command to resume.")
        return 130

    # ----- assemble pooled variants -----
    pooled, npool, secs = {}, {}, 0.0
    for split in ("train", "test"):
        loaded = [np.load(f) for f in data[split]["files"]]
        pooled[split] = {k: np.concatenate([z[f"X_{k}"] for z in loaded]) for k in POOLED}
        npool[split] = {k: np.concatenate([z[f"n_{k}"] for z in loaded]) for k in POOLED}
        secs += sum(float(z["secs"]) for z in loaded)
        assert len(pooled[split]["full_mean"]) == data[split]["n"]
        pooled[split]["concat_c25_full"] = np.concatenate(
            [pooled[split]["center_w25"], pooled[split]["full_mean"]], axis=1)

    # Cross-check: the full-window mean must match what 02_embed.py saved.
    check = {}
    for split in ("train", "test"):
        prev_x, prev_y = out_dir / f"X_{split}.npy", out_dir / f"y_{split}.npy"
        if prev_x.exists():
            px = np.load(prev_x)
            same_shape = px.shape == pooled[split]["full_mean"].shape
            diff = float(np.abs(px - pooled[split]["full_mean"]).max()) if same_shape else None
            y_ok = bool(np.array_equal(np.load(prev_y), data[split]["y"]))
            check[split] = {"max_abs_diff_vs_02": diff, "labels_equal": y_ok}
            if diff is None or diff > 1e-3 or not y_ok:
                print(f"WARNING: {split} full-window mean does not match 02_embed.py output "
                      f"(max diff {diff}, labels equal {y_ok}). Rows may be misaligned.")
        else:
            check[split] = "02_embed.py output not found; not compared"
    print(f"Full-window mean vs 02_embed.py: {check}")

    for v in VARIANTS:
        vdir = out_dir / v
        vdir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "test"):
            np.save(vdir / f"X_{split}.npy", pooled[split][v].astype(np.float32))
            np.save(vdir / f"y_{split}.npy", data[split]["y"])
            np.save(vdir / f"idx_{split}.npy", idx[split])
        write_json_atomic(vdir / "meta.json", {"run_key": run_key, "variant": v,
                                               "dim": int(pooled["train"][v].shape[1])})

    # ----- raw centre tokens, written straight into one memory-mapped file -----
    counts = {s: np.concatenate([np.load(f)["raw_n"] for f in data[s]["files"]])
              for s in ("train", "test")}
    n_tok = np.concatenate([counts["train"], counts["test"]])
    dim = pooled["train"]["full_mean"].shape[1]
    raw_path = out_dir / "center_tokens.npy"
    tmp_raw = out_dir / "center_tokens.tmp.npy"
    mm = np.lib.format.open_memmap(tmp_raw, mode="w+", dtype=np.float16,
                                   shape=(int(n_tok.sum()), dim))
    pos, tok_start, tok_end = 0, [], []
    for split in ("train", "test"):
        for f in data[split]["files"]:
            z = np.load(f)
            mm[pos:pos + len(z["raw"])] = z["raw"]
            pos += len(z["raw"])
            tok_start.append(z["raw_start"])
            tok_end.append(z["raw_end"])
    mm.flush()
    del mm
    os.replace(tmp_raw, raw_path)
    np.savez(out_dir / "center_tokens_index.npz",
             ptr=np.concatenate([[0], np.cumsum(n_tok)]).astype(np.int64),
             n_tokens=n_tok.astype(np.int32),
             split=np.array(["train"] * len(counts["train"]) + ["test"] * len(counts["test"])),
             dataset_row=np.concatenate([idx["train"], idx["test"]]).astype(np.int64),
             label=np.concatenate([data["train"]["y"], data["test"]["y"]]),
             tok_start=np.concatenate(tok_start), tok_end=np.concatenate(tok_end),
             raw_window_inclusive=np.array(RAW_WINDOW))

    n_total = run_key["n_train"] + run_key["n_test"]
    stats = {
        "model": args.model, "short_name": short_name(args.model),
        "token_to_base": mapping,
        "tokens_pooled_per_sequence": {
            k: {"mean": round(float(np.concatenate([npool[s][k] for s in npool]).mean()), 2),
                "min": int(min(npool[s][k].min() for s in npool)),
                "max": int(max(npool[s][k].max() for s in npool))} for k in POOLED},
        "raw_center_tokens": {"file": str(raw_path.relative_to(EMB_DIR.parent.parent)),
                              "shape": [int(n_tok.sum()), int(dim)], "dtype": "float16",
                              "tokens_per_sequence_mean": round(float(n_tok.mean()), 2),
                              "bytes": int(raw_path.stat().st_size)},
        "full_mean_check_vs_02": check,
        "seconds_per_sequence": round(secs / n_total, 5),
        "embed_compute_seconds": round(secs, 1),
        "wall_seconds_this_run": round(time.time() - t_run, 1),
        "n_embedded": n_total,
        "n_embedded_this_run": n_new,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json_atomic(meta_path, {"run_key": run_key, "stats": stats})
    update_json(EMBED_STATS_JSON, ["center_pooling", args.model], stats)
    shutil.rmtree(ckpt_dir)

    print("\nTokens pooled per sequence (mean, min-max):")
    for k, v in stats["tokens_pooled_per_sequence"].items():
        print(f"  {k:<11} {v['mean']:7.2f}  ({v['min']}-{v['max']})")
    print(f"Raw centre tokens: {raw_path} {tuple(stats['raw_center_tokens']['shape'])} float16, "
          f"{stats['raw_center_tokens']['bytes'] / 1e6:.0f} MB")
    print(f"Variants saved under {out_dir}/<variant>/ : {', '.join(VARIANTS)}")
    print(f"{stats['seconds_per_sequence'] * 1000:.1f} ms/sequence, compute {fmt_time(secs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
