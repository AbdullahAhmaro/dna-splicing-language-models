#!/usr/bin/env python3
"""
Phase 1, step 10: scan whole genes with the per-position splice classifier.

    python scripts/10_scan_gene.py                     # every gene in the list, both models
    python scripts/10_scan_gene.py --genes SOD1 --models zhihan1996/DNABERT-2-117M
    python scripts/10_scan_gene.py --context window --models <NT-v2>   # training geometry
    python scripts/10_scan_gene.py --head hardneg      # the hard-negative head (13 + 03 + 07)

Idea: a training window labelled "centre is a donor" is a label for one
position (base 300), and the validation-chosen pooling (07) reads only the
token covering that base and the token before it. So the trained head is
already a per-position classifier. This script runs the frozen DNA model along
a whole gene and applies that head at every position.

Genes: all on chr20/chr21, the dataset's held-out TEST chromosomes (checked
against the dataset's own `name` column: no training window comes from them),
so the head never trained on any of their splice sites. The list lives in
data/genes/scan_gene_list.json:
  * original (round 4): MMP9 (+, 13 exons), SOD1 (+, 5 exons, a GC donor),
    PCNA (-, 6 exons).
  * added (for statistical power): built once by select_added_genes() from an
    Ensembl listing of chr20/chr21 with a fixed rule and seed (see SELECT):
    protein-coding genes whose canonical transcript is protein-coding, spans
    4-25 kb, has >= 4 exons and overlaps no other protein-coding gene on either
    strand; stratified into 3 span bins x 3 exon-count bins, one seeded pick
    per stratum (alternating strands), then more from the exon-rich strata
    until there are >= 12 added genes and >= 85 added introns.
Each gene is fetched from the Ensembl REST API (GRCh38) and cached in
data/genes/<GENE>.json with 1,000 bases of flank on each side.

Coordinates: every gene is scanned in its own transcript orientation. A
- strand gene is reverse-complemented before scanning, and every position is
mapped back to the genome in the outputs. Site positions follow the training
geometry: a donor is the LAST exon base (GT follows it), an acceptor is the
FIRST exon base (AG precedes it).

--context long (default):
  * One pass if the sequence fits the model's context, otherwise overlapping
    tiles with >= 1,000 bases of overlap. Predictions within ~500 bases of
    any tile edge are discarded.
  * NT-v2 cuts DNA into fixed 6-mers from the start of its input, and in
    training the centre token always STARTED at the scored base. So the gene
    is scanned 6 times, shifted by 0-5 bases. Every base is then the start of
    a token in exactly one pass and gets its own score.
  * DNABERT-2 uses variable-length BPE tokens. As in training, a base is
    scored by the token that covers it, so neighbouring bases inside one token
    share a score.
--context window: score every base of the canonical transcript span from its
  own 600-nt window centred on that base, exactly the geometry the head was
  trained on. One forward pass per base (NT-v2 here: ~23 windows/s). The two
  token vectors each head needs (the token covering the base and the one
  before it) are cached per gene, float16, in <model>.window.features.npy, so
  any other centre/previous-token head is applied without a new forward pass.
  Checkpointed and resumable.

Heads (--head): `original` = the 03 head on the 8,000 training windows;
`hardneg` = the same recipe trained on those windows plus the hard negatives
of 13_hard_negatives.py. Default variant = the one 07 chose on validation for
that head; --variant overrides. Supported: center_tok, prev_tok,
center_plus_prev, center_plus_prev_concat.

Outputs per gene, in data/scan/<GENE>/ (<track> = <model>.<variant>[.hardneg].<long|window>):
  gene.json                       coordinates, strand, sequence, true sites
  <track>.scores.npy              (L, 3) float32 per-base [none, acceptor, donor]; NaN = not scored
  <track>.units.npy               (L,) scoring unit per base (-1 = not scored), for de-duplication
  <track>.hidden.npy / .tokens.npz  (long) centre-token hidden states, float16, and their spans
  <track>.context_check.npz       (long, original head, chosen variant) scores at true sites +
                                  500 random positions, also computed in 600-nt windows
  <model>.window.features.npy     (window) (n, 2, hidden) float16 [centre, previous] token
                                  vectors at <model>.window.positions.npy
  scan_meta.json                  how each track was produced
"""

import argparse
import json
import sys
import time
import urllib.request

import numpy as np

from phase1_common import (DATA_DIR, EMB_DIR, MODELS, RESULTS_JSON,
                           SCAN_VARIANTS, dataset_chromosomes, model_slug, pair_features,
                           read_json, write_json_atomic)
from try_dna_models import get_hidden, load_model

GENES_DIR = DATA_DIR / "genes"
SCAN_DIR = DATA_DIR / "scan"
GENE_LIST = GENES_DIR / "scan_gene_list.json"
ENSEMBL = "https://rest.ensembl.org"
FLANK = 1000
ORIGINAL_GENES = {
    "MMP9": ("ENSG00000100985",
             "chr20 (held-out test chromosome), + strand, 7.7 kb, 13 exons: many splice sites in a compact gene"),
    "SOD1": ("ENSG00000142168",
             "chr21 (held-out test chromosome), + strand, 9.3 kb, 5 exons; intron 1 has a non-canonical GC donor"),
    "PCNA": ("ENSG00000132646",
             "chr20 (held-out test chromosome), - strand, 12.7 kb, 6 exons: exercises the reverse-complement path"),
}
# Rule for the added genes (select_added_genes); fixed so the list is reproducible.
SELECT = {
    "seed": 42,
    "span_bins": [[4000, 10000], [10000, 16000], [16000, 25001]],     # canonical span, bases
    "exon_bins": [[4, 7], [7, 13], [13, 1000]],                        # canonical exon count
    "min_added_genes": 12,
    "min_added_introns": 85,
    # second round: exon-rich strata first, small before large (cheapest introns first)
    "top_up_order": [[0, 2], [1, 2], [0, 1], [1, 1], [2, 2], [2, 1]],
}


def test_chromosomes():
    """(test, train) chromosome sets of the dataset, from its own `name` column."""
    ch = dataset_chromosomes()
    return set(ch["test"]), set(ch["train"])


def ensembl_release():
    try:
        return json.loads(fetch("/info/data"))["releases"]
    except Exception:                          # noqa: BLE001 - informational only
        return None


def list_region_genes(chrom, length, step=5_000_000):
    """All Ensembl genes on one chromosome (the overlap endpoint allows 5 Mb per request)."""
    out = {}
    for s in range(1, length + 1, step):
        e = min(s + step - 1, length)
        for g in json.loads(fetch(f"/overlap/region/human/{chrom}:{s}-{e}?feature=gene")):
            out[g["id"]] = g
        time.sleep(0.2)
    return list(out.values())


def lookup_many(ids, batch=50):
    """POST /lookup/id with transcripts and exons, in batches."""
    out = {}
    for i in range(0, len(ids), batch):
        out.update(json.loads(fetch("/lookup/id", data={"ids": ids[i:i + batch], "expand": 1})))
        time.sleep(0.3)
    return out


def canonical_summary(g):
    """Span / exon count of a looked-up gene's canonical transcript (None if unusable)."""
    if not g:
        return None
    can = [t for t in g.get("Transcript", []) if t.get("is_canonical")]
    if len(can) != 1:
        return None
    t = can[0]
    return {"symbol": g.get("display_name") or g["id"], "gene_id": g["id"],
            "chrom": g["seq_region_name"], "strand": g["strand"], "start": g["start"],
            "end": g["end"], "tx_biotype": t["biotype"], "span": t["end"] - t["start"] + 1,
            "exons": len(t["Exon"]), "n_transcripts": len(g["Transcript"])}


def select_added_genes():
    """Build data/genes/scan_gene_list.json once (rule and seed in SELECT)."""
    test, train = test_chromosomes()
    if test & train:
        raise SystemExit(f"dataset train and test share chromosomes {test & train}")
    GENES_DIR.mkdir(parents=True, exist_ok=True)
    raw = GENES_DIR / "_ensembl_test_chrom_genes.json"
    if raw.exists():
        listing = json.loads(raw.read_text())
    else:
        print(f"Listing Ensembl genes on chr{', chr'.join(sorted(test))} ...")
        listing = {"release": ensembl_release(), "genes": []}
        for c in sorted(test):
            length = json.loads(fetch(f"/info/assembly/homo_sapiens/{c}"))["length"]
            listing["genes"] += list_region_genes(c, length)
        raw.write_text(json.dumps(listing))
    pc = [g for g in listing["genes"] if g.get("biotype") == "protein_coding"]
    pre = [g["id"] for g in pc if 3000 <= g["end"] - g["start"] + 1 <= 40000]
    lk = GENES_DIR / "_ensembl_test_chrom_lookup.json"
    looked = json.loads(lk.read_text()) if lk.exists() else {}
    missing = [i for i in pre if i not in looked]
    if missing:
        print(f"Looking up {len(missing)} candidate genes ...")
        looked.update(lookup_many(missing))
        lk.write_text(json.dumps(looked))
    orig_ids = {gid for gid, _ in ORIGINAL_GENES.values()}
    span_lo, span_hi = SELECT["span_bins"][0][0], SELECT["span_bins"][-1][1]
    cands = []
    for gid in sorted(pre):
        c = canonical_summary(looked.get(gid))
        if (c is None or gid in orig_ids or c["tx_biotype"] != "protein_coding"
                or not span_lo <= c["span"] < span_hi or c["exons"] < SELECT["exon_bins"][0][0]
                or c["chrom"] not in test):
            continue
        overl = [p for p in pc if p["id"] != gid and p["seq_region_name"] == c["chrom"]
                 and p["start"] <= c["end"] and p["end"] >= c["start"]]
        if overl:
            continue
        cands.append(c)

    def stratum(c):
        si = next(i for i, (a, b) in enumerate(SELECT["span_bins"]) if a <= c["span"] < b)
        ei = next(i for i, (a, b) in enumerate(SELECT["exon_bins"]) if a <= c["exons"] < b)
        return si, ei
    rng = np.random.default_rng(SELECT["seed"])
    pools = {}
    for si in range(len(SELECT["span_bins"])):
        for ei in range(len(SELECT["exon_bins"])):
            p = [c for c in cands if stratum(c) == (si, ei)]
            pools[(si, ei)] = [p[i] for i in rng.permutation(len(p))]
    chosen = []

    def take(key, why):
        minus = sum(c["strand"] == -1 for c in chosen)
        want = -1 if minus <= len(chosen) - minus else 1      # keep the strands balanced
        p = pools.get(key, [])
        if not p:
            return
        j = next((i for i, c in enumerate(p) if c["strand"] == want), 0)
        c = p.pop(j)
        c["stratum"], c["round"] = key, why
        chosen.append(c)
    for key in pools:
        take(key, "one per stratum")
    for key in map(tuple, SELECT["top_up_order"]):
        if (len(chosen) >= SELECT["min_added_genes"]
                and sum(c["exons"] - 1 for c in chosen) >= SELECT["min_added_introns"]):
            break
        take(key, "top-up (exon-rich strata)")

    def label(lo, hi, unit):
        return f"{lo // 1000}-{round(hi / 1000)} kb" if unit == "kb" else \
            (f"{lo}-{hi - 1} exons" if hi < 1000 else f"{lo}+ exons")
    genes = [{"symbol": s, "gene_id": gid, "set": "original", "why": why}
             for s, (gid, why) in ORIGINAL_GENES.items()]
    for c in chosen:
        sb, eb = SELECT["span_bins"][c["stratum"][0]], SELECT["exon_bins"][c["stratum"][1]]
        genes.append({"symbol": c["symbol"], "gene_id": c["gene_id"], "set": "added",
                      "why": (f"chr{c['chrom']} (held-out test chromosome), "
                              f"{'+' if c['strand'] == 1 else '-'} strand, {c['span'] / 1000:.1f} kb, "
                              f"{c['exons']} exons; stratum {label(*sb, 'kb')} / "
                              f"{label(*eb, 'exons')}, seeded pick ({c['round']})")})
    rec = {"ensembl_release": listing.get("release"), "rule": SELECT,
           "rule_text": "protein-coding genes on the test chromosomes whose canonical transcript "
                        "is protein-coding, spans 4-25 kb, has >= 4 exons and overlaps no other "
                        "protein-coding gene; one seeded pick per span x exon-count stratum "
                        "(strands alternating), then exon-rich strata until >= 12 genes and "
                        ">= 85 introns",
           "n_candidates": len(cands), "genes": genes}
    write_json_atomic(GENE_LIST, rec)
    print(f"Selected {len(chosen)} added genes from {len(cands)} candidates "
          f"({sum(c['exons'] - 1 for c in chosen)} introns); saved {GENE_LIST}")
    return rec


def load_gene_list():
    rec = read_json(GENE_LIST) if GENE_LIST.exists() else select_added_genes()
    return {g["symbol"]: g for g in rec["genes"]}
# Per-model scanning geometry (see docstring).
SCAN = {
    "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species":
        {"phases": 6, "assign": "start", "tile": 12000, "overlap": 1002},
    "zhihan1996/DNABERT-2-117M":
        {"phases": 1, "assign": "cover", "tile": 2000, "overlap": 1000},
}
WINDOW, CENTER = 600, 300
COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(s):
    return s.translate(COMP)[::-1]


def fetch(path, ctype="application/json", tries=5, data=None):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(
                ENSEMBL + path, data=None if data is None else json.dumps(data).encode(),
                headers={"Content-Type": ctype, "Accept": ctype,
                         "User-Agent": "dna-phase-scan/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read().decode()
        except Exception as exc:                # noqa: BLE001 - retry any network error
            last = exc
            print(f"  Ensembl request failed ({exc}); retry {i + 1}/{tries}")
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"Ensembl request failed: {path}: {last}")


def get_gene(symbol, gene_id, chroms=None):
    """Ensembl lookup + + strand genomic sequence with flanks; cached in data/genes/.
    `chroms`: the chromosomes the gene must be on (the dataset's test chromosomes)."""
    GENES_DIR.mkdir(parents=True, exist_ok=True)
    cache = GENES_DIR / f"{symbol}.json"
    old = json.loads(cache.read_text()) if cache.exists() else None
    if old and "transcripts" in old:
        if chroms is not None and old["chrom"] not in chroms:
            raise SystemExit(f"{symbol} is on chr{old['chrom']}, not a test chromosome "
                             f"({sorted(chroms)}); refusing to scan it")
        return old
    print(f"Fetching {symbol} ({gene_id}) from Ensembl ...")
    g = json.loads(fetch(f"/lookup/id/{gene_id}?expand=1"))
    if chroms is not None and g["seq_region_name"] not in chroms:
        raise SystemExit(f"{symbol} is on chr{g['seq_region_name']}, not a test chromosome "
                         f"({sorted(chroms)}); refusing to scan it")
    canon = [t for t in g["Transcript"] if t.get("is_canonical")]
    if len(canon) != 1:
        raise RuntimeError(f"{symbol}: expected one canonical transcript, got {len(canon)}")
    t = canon[0]
    lo, hi = g["start"] - FLANK, g["end"] + FLANK
    if old and old["region_start"] == lo and old["region_end"] == hi:
        seq = old["seq_plus"]                       # sequence already cached
    else:
        seq = fetch(f"/sequence/region/human/{g['seq_region_name']}:{lo}..{hi}:1"
                    "?content-type=text/plain", "text/plain").strip().upper()
    if len(seq) != hi - lo + 1:
        raise RuntimeError(f"{symbol}: got {len(seq)} bases, expected {hi - lo + 1}")
    rec = {
        "symbol": symbol, "gene_id": g["id"], "chrom": g["seq_region_name"],
        "start": g["start"], "end": g["end"], "strand": g["strand"],
        "assembly": g.get("assembly_name"), "biotype": g["biotype"],
        "transcript_id": t["id"], "transcript_biotype": t["biotype"],
        "exons": sorted([[e["start"], e["end"]] for e in t["Exon"]]),
        "transcript_start": t["start"], "transcript_end": t["end"],
        "transcripts": [{"id": x["id"], "biotype": x["biotype"],
                         "is_canonical": bool(x.get("is_canonical")),
                         "exons": sorted([[e["start"], e["end"]] for e in x["Exon"]])}
                        for x in g["Transcript"]],
        "flank": FLANK, "region_start": lo, "region_end": hi, "seq_plus": seq,
        "fetched_from": ENSEMBL,
    }
    cache.write_text(json.dumps(rec))
    return rec


def gene_frame(rec):
    """Sequence in transcript orientation plus true sites and coordinate maps."""
    lo, hi, strand = rec["region_start"], rec["region_end"], rec["strand"]
    seq = rec["seq_plus"] if strand == 1 else revcomp(rec["seq_plus"])

    def to_idx(genomic):
        return genomic - lo if strand == 1 else hi - genomic

    ex = sorted(rec["exons"], key=lambda e: e[0], reverse=(strand == -1))   # transcript order
    if strand == 1:
        spans = [(to_idx(s), to_idx(e)) for s, e in ex]
    else:
        spans = [(to_idx(e), to_idx(s)) for s, e in ex]
    donors = [end for _, end in spans[:-1]]          # last exon base before each intron
    acceptors = [start for start, _ in spans[1:]]    # first exon base after each intron
    body = (to_idx(rec["start"] if strand == 1 else rec["end"]),
            to_idx(rec["end"] if strand == 1 else rec["start"]) + 1)
    tx = (to_idx(rec["transcript_start"] if strand == 1 else rec["transcript_end"]),
          to_idx(rec["transcript_end"] if strand == 1 else rec["transcript_start"]) + 1)

    def sites_of(exons):
        e = sorted(exons, key=lambda x: x[0], reverse=(strand == -1))
        sp = [(to_idx(a), to_idx(b)) if strand == 1 else (to_idx(b), to_idx(a)) for a, b in e]
        return [x[1] for x in sp[:-1]], [x[0] for x in sp[1:]], sp

    alt_d, alt_a, alt_spans = set(), set(), []
    for t in rec["transcripts"]:
        if t["is_canonical"]:
            continue
        d_, a_, sp_ = sites_of(t["exons"])
        alt_d.update(d_)
        alt_a.update(a_)
        alt_spans += sp_
    return {
        "seq": seq, "donors": donors, "acceptors": acceptors, "exon_spans": spans,
        "body": body, "transcript_span": tx,
        "alt_donors": sorted(alt_d), "alt_acceptors": sorted(alt_a),
        "alt_exon_spans": sorted(set(map(tuple, alt_spans))),
        "donor_motifs": [seq[d + 1:d + 3] for d in donors],
        "acceptor_motifs": [seq[a - 2:a] for a in acceptors],
        "donor_genomic": [lo + d if strand == 1 else hi - d for d in donors],
        "acceptor_genomic": [lo + a if strand == 1 else hi - a for a in acceptors],
    }


HEADS = {"original": "head_mlp.pt", "hardneg": "head_mlp_hardneg.pt"}
SELECTION_KEY = {"original": "variant_selection", "hardneg": "variant_selection_hardneg"}


def selected_variant(repo, head_kind="original"):
    sel = (read_json(RESULTS_JSON) or {}).get(
        SELECTION_KEY[head_kind], {}).get("models", {}).get(repo, {})
    return sel.get("selected_variant", "center_tok")


def load_head(repo, variant, torch, head_kind="original"):
    from torch import nn

    if variant not in SCAN_VARIANTS:
        raise SystemExit(f"{repo}: variant {variant!r} is not a per-position head; the scan "
                         f"supports {SCAN_VARIANTS}")
    path = EMB_DIR / model_slug(repo) / variant / HEADS[head_kind]
    if not path.exists():
        raise SystemExit(f"{path} missing; rerun 03_train_head.py --variant {variant}"
                         + (" --hardneg" if head_kind == "hardneg" else ""))
    h = torch.load(path)
    mlp = nn.Sequential(nn.Linear(h["in_dim"], h["hidden"]), nn.ReLU(), nn.Dropout(h["dropout"]),
                        nn.Linear(h["hidden"], len(h["classes"])))
    mlp.load_state_dict(h["state_dict"])
    mlp.eval()

    def head(C, P):
        x = pair_features(np.asarray(C, np.float32), np.asarray(P, np.float32), variant)
        x = (torch.as_tensor(x, dtype=torch.float32) - h["scaler_mean"]) / h["scaler_scale"]
        with torch.no_grad():
            return torch.softmax(mlp(x), dim=-1).numpy()
    return head, h


def track_name(slug, variant, context, head_kind="original"):
    return f"{slug}.{variant}{'.hardneg' if head_kind == 'hardneg' else ''}.{context}"


def token_spans(tok, enc, text):
    """[start, end) base offsets of the non-special tokens of one encoded string."""
    ids = enc["input_ids"][0].numpy()
    body = ~np.isin(ids, np.array(sorted(set(tok.all_special_ids))))
    if tok.is_fast:
        off = enc["offset_mapping"][0].numpy()
        st, en = off[body, 0], off[body, 1]
    else:
        pieces = tok.convert_ids_to_tokens(ids[body].tolist())
        lens = np.array([len(p) for p in pieces])
        assert "".join(pieces) == text, "tokens do not spell the input"
        en = np.cumsum(lens)
        st = en - lens
    return body, st.astype(np.int64), en.astype(np.int64)


def scan(seq, tok, model, head, cfg, torch):
    """Long-context scan. Returns per-base scores, unit ids, and per-token records."""
    L = len(seq)
    scores = np.full((L, 3), np.nan, dtype=np.float32)
    units = np.full(L, -1, dtype=np.int64)
    rec = {"hidden": [], "start": [], "end": [], "phase": [], "tile": []}
    T, O = cfg["tile"], cfg["overlap"]
    step, cut = T - O, O // 2
    uid = 0
    n_tiles = 0
    for ph in range(cfg["phases"]):
        sub = seq[ph:]
        starts = list(range(0, max(len(sub) - O, 1), step))
        for ti, s0 in enumerate(starts):
            text = sub[s0:s0 + T]
            enc = tok(text, return_tensors="pt", return_offsets_mapping=tok.is_fast)
            inputs = {k: v for k, v in enc.items() if k != "offset_mapping"}
            if inputs["input_ids"].shape[1] > getattr(tok, "model_max_length", 10 ** 9):
                raise RuntimeError(f"tile of {len(text)} bases has {inputs['input_ids'].shape[1]} "
                                   "tokens, over the model limit; lower the tile size")
            with torch.no_grad():
                hid = get_hidden(model, inputs, torch)[0].float().numpy()
            body, st, en = token_spans(tok, enc, text)
            hid = hid[body]
            g_st, g_en = st + s0 + ph, en + s0 + ph               # global base offsets
            keep_lo, keep_hi = s0 + ph + cut, s0 + ph + len(text) - cut
            if cfg["assign"] == "start":
                keep = (g_st >= keep_lo) & (g_st < keep_hi) & (en - st == 6)
            else:
                keep = (g_en > keep_lo) & (g_st < keep_hi)
            j = np.flatnonzero(keep)
            assert (j >= 1).all(), "a kept token has no previous token"
            p = head(hid[j], hid[j - 1])
            for (a, b), sc in zip(zip(g_st[keep], g_en[keep]), p):
                if cfg["assign"] == "start":
                    scores[a] = sc
                    units[a] = uid
                else:
                    lo_, hi_ = max(a, keep_lo), min(b, keep_hi)
                    scores[lo_:hi_] = sc
                    units[lo_:hi_] = uid
                uid += 1
            rec["hidden"].append(hid[keep].astype(np.float16))
            rec["start"].append(g_st[keep])
            rec["end"].append(g_en[keep])
            rec["phase"].append(np.full(int(keep.sum()), ph))
            rec["tile"].append(np.full(int(keep.sum()), ti))
            n_tiles += 1
    return scores, units, {k: np.concatenate(v) for k, v in rec.items()}, n_tiles


def window_pairs(texts, tok, model, torch, batch=32):
    """For each 600-nt text: the last-layer vector of the token covering base 300 (C)
    and of the token just before it (P), exactly as the training windows were pooled."""
    special = np.array(sorted(set(tok.all_special_ids)))
    out, out_prev = [], []
    for b in range(0, len(texts), batch):
        tx = texts[b:b + batch]
        enc = tok(tx, return_tensors="pt", padding=True, return_offsets_mapping=tok.is_fast)
        inputs = {k: v for k, v in enc.items() if k != "offset_mapping"}
        with torch.no_grad():
            hid = get_hidden(model, inputs, torch).float().numpy()
        for i in range(len(tx)):
            ids = enc["input_ids"][i].numpy()
            body = (enc["attention_mask"][i].numpy() == 1) & ~np.isin(ids, special)
            if tok.is_fast:
                off = enc["offset_mapping"][i].numpy()
                st, en = off[:, 0], off[:, 1]
            else:
                pieces = tok.convert_ids_to_tokens(ids[body].tolist())
                lens = np.array([len(x) for x in pieces])
                st = np.full(len(ids), -1)
                en = np.full(len(ids), -1)
                en[body] = np.cumsum(lens)
                st[body] = en[body] - lens
            c = np.flatnonzero(body & (st <= CENTER) & (en > CENTER))
            assert len(c) == 1
            bi = np.flatnonzero(body)
            k = int(np.searchsorted(bi, c[0]))
            assert k >= 1, "centre token has no previous token"
            out.append(hid[i, c[0]])
            out_prev.append(hid[i, bi[k - 1]])
    return np.stack(out), np.stack(out_prev)


def window_scores(seq, positions, tok, model, head, torch, batch=16):
    """Head scores at given positions, each from its own 600-nt window (training geometry)."""
    C, P = window_pairs([seq[p - CENTER:p - CENTER + WINDOW] for p in positions],
                        tok, model, torch, batch=batch)
    return head(C, P)


def window_features(seq, positions, tok, model, torch, stem, batch=32, chunk=2048):
    """[centre, previous] token vectors (float16) of a 600-nt window centred on every
    given position, cached in <stem>.features.npy (n, 2, hidden) with the positions in
    <stem>.positions.npy. Resumable: <stem>.progress.json exists only while incomplete."""
    fpath = stem.with_name(stem.name + ".features.npy")
    ppath = stem.with_name(stem.name + ".positions.npy")
    prog = stem.with_name(stem.name + ".progress.json")
    same = fpath.exists() and ppath.exists() and np.array_equal(np.load(ppath), positions)
    if same and not prog.exists():
        return np.load(fpath, mmap_mode="r"), False
    if same:
        done = int(read_json(prog)["done"])
        feats = np.lib.format.open_memmap(fpath, mode="r+")
        print(f"    resuming: {done:,}/{len(positions):,} windows already embedded")
    else:
        done = 0
        write_json_atomic(prog, {"done": 0})            # marks the cache incomplete first
        np.save(ppath, positions)
        feats = np.lib.format.open_memmap(fpath, mode="w+", dtype=np.float16,
                                          shape=(len(positions), 2, model.config.hidden_size))
    t0, start_done, last = time.time(), done, 0.0
    while done < len(positions):
        ps = positions[done:done + chunk]
        C, P = window_pairs([seq[p - CENTER:p - CENTER + WINDOW] for p in ps],
                            tok, model, torch, batch=batch)
        feats[done:done + len(ps)] = np.stack([C, P], axis=1).astype(np.float16)
        feats.flush()
        done += len(ps)
        write_json_atomic(prog, {"done": done})
        el = time.time() - t0
        rate = (done - start_done) / el if el > 0 else 0
        if time.time() - last > 20 or done == len(positions):
            last = time.time()
            print(f"    {done:>6,}/{len(positions):,} windows  {rate:5.1f} windows/s  "
                  f"ETA {(len(positions) - done) / rate / 60 if rate else float('nan'):.1f} min",
                  flush=True)
    prog.unlink()
    del feats
    return np.load(fpath, mmap_mode="r"), True


def score_features(feats, head, chunk=8192):
    out = np.empty((len(feats), 3), np.float32)
    for i in range(0, len(feats), chunk):
        f = np.asarray(feats[i:i + chunk], np.float32)
        out[i:i + chunk] = head(f[:, 0], f[:, 1])
    return out


HEAD_TRAINED_ON = {
    "original": "class-balanced 600-nt windows (~1/3 per class)",
    "hardneg": "the same windows plus the hard negatives of 13_hard_negatives.py (exon "
               "interiors, intron interiors, GT/AG decoys from training-chromosome genes)",
}


def main():
    ap = argparse.ArgumentParser(description="Per-position splice-site scan of whole genes.")
    ap.add_argument("--genes", nargs="*", default=["all"],
                    help="gene symbols, or 'all' / 'original' / 'added' (default: all)")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--n-random", type=int, default=500,
                    help="random gene positions for the context check")
    ap.add_argument("--head", choices=list(HEADS), default="original",
                    help="original: 03 head on the training windows; hardneg: 03 head trained "
                         "with the hard negatives of 13 as well")
    ap.add_argument("--variant", default=None,
                    help="head to apply (default: the variant 07 chose on validation)")
    ap.add_argument("--context", choices=["long", "window"], default="long",
                    help="long: one pass / long tiles (default); window: a 600-nt window "
                         "centred on every scored base, exactly as in training")
    ap.add_argument("--context-check", choices=["auto", "yes", "no"], default="auto",
                    help="long context only; auto = only for the original head's chosen variant")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    test, train = test_chromosomes()
    if test & train:
        raise SystemExit(f"dataset train and test share chromosomes {sorted(test & train)}")
    glist = load_gene_list()
    wanted = []
    for g in args.genes:
        if g in ("all", "original", "added"):
            wanted += [s for s, r in glist.items() if g == "all" or r["set"] == g]
        elif g in glist:
            wanted.append(g)
        else:
            raise SystemExit(f"unknown gene {g!r}; known: {list(glist)}")
    wanted = list(dict.fromkeys(wanted))

    frames = {}
    for sym in wanted:
        rec = get_gene(sym, glist[sym]["gene_id"], chroms=test)
        fr = gene_frame(rec)
        frames[sym] = (rec, fr)
        bad = set(fr["seq"]) - set("ACGT")
        L = len(fr["seq"])
        print(f"\n{sym} {rec['gene_id']} chr{rec['chrom']}:{rec['start']}-{rec['end']} "
              f"strand {'+' if rec['strand'] == 1 else '-'} ({rec['assembly']}), canonical "
              f"{rec['transcript_id']}, {len(rec['exons'])} exons; scanned {L:,} bases "
              f"(gene + {FLANK} flank each side, transcript orientation)"
              + (f"; non-ACGT characters: {sorted(bad)}" if bad else ""))
        print(f"  why: {glist[sym]['why']}")
        print(f"  donors    {len(fr['donors'])}: motifs {fr['donor_motifs']}")
        print(f"  acceptors {len(fr['acceptors'])}: motifs {fr['acceptor_motifs']}")
        out = SCAN_DIR / sym
        out.mkdir(parents=True, exist_ok=True)
        write_json_atomic(out / "gene.json", {
            **{k: v for k, v in rec.items() if k != "seq_plus"},
            "strand_symbol": "+" if rec["strand"] == 1 else "-",
            "why_chosen": glist[sym]["why"], "gene_set": glist[sym]["set"],
            "chromosome_check": f"chr{rec['chrom']} is a dataset TEST chromosome "
                                f"(test: {sorted(test)}; never in train)",
            "scan_orientation": "transcript (reverse-complemented)" if rec["strand"] == -1
                                else "transcript (= genome + strand)",
            "length_scanned": L, "gene_body_idx": list(fr["body"]),
            "eval_region_idx": list(fr["transcript_span"]),
            "eval_region_note": "canonical transcript span (its pre-mRNA); metrics use only this",
            "n_transcripts": len(rec["transcripts"]),
            "alt_donors_idx": fr["alt_donors"], "alt_acceptors_idx": fr["alt_acceptors"],
            "alt_exon_spans_idx": [list(x) for x in fr["alt_exon_spans"]],
            "donors_idx": fr["donors"], "acceptors_idx": fr["acceptors"],
            "donors_genomic": fr["donor_genomic"], "acceptors_genomic": fr["acceptor_genomic"],
            "donor_motifs": fr["donor_motifs"], "acceptor_motifs": fr["acceptor_motifs"],
            "exon_spans_idx": fr["exon_spans"],
            "site_convention": "donor = last exon base (GT follows); acceptor = first exon "
                               "base (AG precedes); indices are 0-based in the scanned sequence",
            "seq_scanned": fr["seq"],
        })

    for repo in args.models:
        cfg = SCAN[repo]
        slug = model_slug(repo)
        chosen = selected_variant(repo, args.head)
        variant = args.variant or chosen
        name = track_name(slug, variant, args.context, args.head)
        todo = [s for s in wanted
                if args.force or not (SCAN_DIR / s / f"{name}.scores.npy").exists()]
        if not todo:
            print(f"\n{MODELS.get(repo, repo)} `{variant}` ({args.head} head) {args.context}: "
                  "all genes already scanned (--force to redo)")
            continue
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        model, _ = load_model(repo)
        model.eval()
        head, h = load_head(repo, variant, torch, args.head)
        do_cc = args.context == "long" and (
            args.context_check == "yes" or
            (args.context_check == "auto" and args.head == "original" and variant == chosen))
        how = (f"{cfg['phases']} phase(s), tiles of {cfg['tile']} bases, overlap {cfg['overlap']}, "
               f"score assigned to the token {'start' if cfg['assign'] == 'start' else 'span'}"
               if args.context == "long" else
               f"a {WINDOW}-nt window centred on every scored base (training geometry)")
        print(f"\n=== {MODELS.get(repo, repo)}: {args.head} head `{variant}` "
              f"({'chosen on validation' if variant == chosen else 'explicit'}); {how}")
        for sym in todo:
            rec, fr = frames[sym]
            seq, out = fr["seq"], SCAN_DIR / sym
            b0, b1 = fr["body"]
            e0, e1 = fr["transcript_span"]
            meta = read_json(out / "scan_meta.json", default={}) or {}
            t0 = time.time()
            if args.context == "long":
                scores, units, toks, n_tiles = scan(seq, tok, model, head, cfg, torch)
                secs = time.time() - t0
                cov = float(np.isfinite(scores[e0:e1, 0]).mean())
                np.save(out / f"{name}.scores.npy", scores)
                np.save(out / f"{name}.units.npy", units)
                np.save(out / f"{name}.hidden.npy", toks.pop("hidden"))
                np.savez(out / f"{name}.tokens.npz", **toks)
                meta[name] = {"model": repo, "variant": variant, "context": "long",
                              "head": args.head,
                              "phases": cfg["phases"], "assign": cfg["assign"],
                              "tile_bases": cfg["tile"], "overlap_bases": cfg["overlap"],
                              "edge_discard_bases": cfg["overlap"] // 2, "forward_passes": n_tiles,
                              "tokens_kept": int(len(toks["start"])), "scored_span_coverage": cov,
                              "scan_seconds": round(secs, 1)}
                if do_cc:
                    # context check: same head, 600-nt windows exactly as in training
                    rng = np.random.default_rng(0)
                    sites = np.array(fr["donors"] + fr["acceptors"])
                    pool = np.arange(max(b0, CENTER), min(b1, len(seq) - CENTER))
                    pool = pool[np.min(np.abs(pool[:, None] - sites[None, :]), axis=1) > 2]
                    rand = np.sort(rng.choice(pool, min(args.n_random, len(pool)), replace=False))
                    pos = np.concatenate([fr["donors"], fr["acceptors"], rand]).astype(np.int64)
                    t1 = time.time()
                    win = window_scores(seq, pos, tok, model, head, torch)
                    np.savez(out / f"{name}.context_check.npz", positions=pos, window=win,
                             long=scores[pos], n_donors=len(fr["donors"]),
                             n_acceptors=len(fr["acceptors"]))
                    meta[name]["context_check_seconds"] = round(time.time() - t1, 1)
                print(f"  {sym}: {n_tiles} forward passes in {secs:.1f}s")
            else:
                positions = np.arange(e0, e1)          # every base of the canonical span
                feats, computed = window_features(seq, positions, tok, model, torch,
                                                  out / f"{slug}.window")
                t_feat = time.time() - t0
                scores = np.full((len(seq), 3), np.nan, dtype=np.float32)
                scores[positions] = score_features(feats, head)
                secs = time.time() - t0
                units = np.full(len(seq), -1, dtype=np.int64)
                units[positions] = positions
                np.save(out / f"{name}.scores.npy", scores)
                np.save(out / f"{name}.units.npy", units)
                prev = meta.get(name, {})
                meta[name] = {"model": repo, "variant": variant, "context": "window",
                              "head": args.head,
                              "window_bases": WINDOW, "positions_scored": int(len(positions)),
                              "forward_passes": int(len(positions)),
                              "scored_span_coverage": float(np.isfinite(scores[e0:e1, 0]).mean()),
                              "features_cache": f"{slug}.window.features.npy",
                              "scan_seconds": round(t_feat, 1) if computed
                              else prev.get("scan_seconds", round(secs, 1)),
                              "scored_from_cached_features": not computed}
                print(f"  {sym}: {len(positions):,} windows "
                      + (f"embedded in {t_feat / 60:.1f} min" if computed else
                         "scored from cached features") + f"; head applied in {secs - t_feat:.1f}s")
            meta[name]["head_trained_on"] = HEAD_TRAINED_ON[args.head]
            write_json_atomic(out / "scan_meta.json", meta)
            d, a = fr["donors"], fr["acceptors"]
            print(f"    p(donor) at true donors {[round(float(v), 2) for v in scores[d, 2]]}; "
                  f"p(acceptor) at true acceptors {[round(float(v), 2) for v in scores[a, 1]]}")
    print(f"\nOutputs in {SCAN_DIR}/<GENE>/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
