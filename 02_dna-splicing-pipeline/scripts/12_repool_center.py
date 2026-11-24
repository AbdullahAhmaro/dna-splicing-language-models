#!/usr/bin/env python3
"""
Phase 1, step 12: extra pooling variants built from the saved centre tokens (no forward pass).

    python scripts/12_repool_center.py

Hypothesis being tested: NT-v2's centre token covers bases 300-305, so a donor's
GT (offset 301) falls inside it, while an acceptor's AG (offset 298) falls in
the previous token. If the head mostly reads its own token, adding the
previous token should help acceptors.

02b_embed_center.py saved the last-layer vector of every token overlapping
bases 240-360 (float16, data/emb/<model>/center_tokens.npy, with each token's
base span). From those, this script builds:

    prev_tok                 the token immediately before the centre token
    center_plus_prev         mean of the centre token and the previous token
    center_plus_prev_concat  the two vectors concatenated (2 x hidden size)

"Centre token" = the token covering base 300, exactly as in center_tok. The new
variants use the float16 copy; re-pooling center_tok from it reproduces the
float32 vectors to within float16 rounding, which is checked and reported.

Saved like every other variant: data/emb/<model>/<variant>/{X,y,idx}_{train,test}.npy,
so 03_train_head.py --variant <name> and 07_select_variant.py work unchanged.
"""

import sys

import numpy as np

from phase1_common import EMB_DIR, MODELS, model_slug, write_json_atomic

CENTER_BASE = 300
NEW_VARIANTS = ["prev_tok", "center_plus_prev", "center_plus_prev_concat"]


def main():
    failed = False
    for repo, short in MODELS.items():
        d = EMB_DIR / model_slug(repo)
        raw_path, ix_path = d / "center_tokens.npy", d / "center_tokens_index.npz"
        if not raw_path.exists() or not ix_path.exists():
            print(f"{short}: {raw_path.name} missing; run 02b_embed_center.py first")
            failed = True
            continue
        raw = np.load(raw_path, mmap_mode="r")
        ix = np.load(ix_path)
        ptr, ts, te, split = ix["ptr"], ix["tok_start"], ix["tok_end"], ix["split"]
        n = len(ptr) - 1
        cen = np.empty((n, raw.shape[1]), np.float32)
        prev = np.empty((n, raw.shape[1]), np.float32)
        prev_span = np.empty((n, 2), np.int64)
        for i in range(n):
            a, b = ptr[i], ptr[i + 1]
            c = np.flatnonzero((ts[a:b] <= CENTER_BASE) & (te[a:b] > CENTER_BASE))
            assert len(c) == 1, f"row {i}: {len(c)} tokens cover base {CENTER_BASE}"
            c = int(c[0])
            assert c >= 1, f"row {i}: no previous token saved"
            assert te[a + c - 1] == ts[a + c], f"row {i}: previous token is not adjacent"
            cen[i] = raw[a + c]
            prev[i] = raw[a + c - 1]
            prev_span[i] = (ts[a + c - 1], te[a + c - 1])
        built = {"prev_tok": prev,
                 "center_plus_prev": (cen + prev) / 2,
                 "center_plus_prev_concat": np.concatenate([cen, prev], axis=1)}

        # Same rows, same order as the existing variants?
        ref = d / "center_tok"
        rows = {s: np.flatnonzero(split == s) for s in ("train", "test")}
        for s in ("train", "test"):
            assert np.array_equal(ix["dataset_row"][rows[s]], np.load(ref / f"idx_{s}.npy")), \
                f"{short}: {s} rows do not match center_tok"
        diff = max(float(np.abs(cen[rows[s]] - np.load(ref / f"X_{s}.npy")).max())
                   for s in ("train", "test"))

        for v, X in built.items():
            vd = d / v
            vd.mkdir(parents=True, exist_ok=True)
            for s in ("train", "test"):
                np.save(vd / f"X_{s}.npy", X[rows[s]].astype(np.float32))
                np.save(vd / f"y_{s}.npy", np.load(ref / f"y_{s}.npy"))
                np.save(vd / f"idx_{s}.npy", np.load(ref / f"idx_{s}.npy"))
            write_json_atomic(vd / "meta.json", {
                "variant": v, "dim": int(X.shape[1]),
                "source": "re-pooled from center_tokens.npy (float16) by 12_repool_center.py",
                "center_token": f"token covering base {CENTER_BASE}",
                "previous_token": "the token immediately before it"})
        spans = prev_span[:, 1] - prev_span[:, 0]
        print(f"{short}: built {', '.join(NEW_VARIANTS)} for {n} windows; previous token "
              f"covers bases {int(np.median(prev_span[:, 0]))}-{int(np.median(prev_span[:, 1])) - 1} "
              f"(median), length {spans.min()}-{spans.max()}; centre token re-pooled from "
              f"float16 differs from center_tok by at most {diff:.4f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
