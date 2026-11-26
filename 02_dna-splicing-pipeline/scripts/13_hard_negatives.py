#!/usr/bin/env python3
"""
Phase 1, step 13: hard negatives mined from genes on the TRAINING chromosomes.

    python scripts/13_hard_negatives.py             # mine (Ensembl, cached) + embed, both models
    python scripts/13_hard_negatives.py --mine-only

Why. Along whole genes the head's false positives sit mostly inside exons. The
dataset's "none" windows are mostly intronic or intergenic, so "inside an exon,
not at its edge" was hardly ever shown as a negative. This step mines windows
the head should call "none" from real genes, in three groups:

  exon_interior    inside an exon of the gene's canonical transcript, >= 20 bases
                   from every annotated splice site
  intron_interior  inside a canonical intron, >= 20 bases from every annotated
                   splice site and outside every annotated exon
  decoy            GT right after the position (donor-like) or AG right before
                   it (acceptor-like) in the gene's own orientation, i.e. the
                   motif exactly where a real site has it, but >= 10 bases from
                   every annotated splice site

"Annotated splice site" = an internal exon boundary of ANY transcript of ANY
gene in the region, on either strand (Ensembl exon overlap), not only the
canonical transcript, so alternative sites are never used as negatives.

Chromosome split. Read from the dataset's own `name` column: training windows
come from chr1-19, X and Y; test windows from chr20 and chr21; chr22 is in
neither split and is not used. Genes come only from the training chromosomes
(chrY is skipped: few genes, and its pseudo-autosomal regions are shared with
chrX). The script refuses to run if a gene lands on a test chromosome.

Genes. Seeded random 5-Mb regions of the training chromosomes (weighted by
length) are listed in Ensembl. Candidates: protein-coding genes whose canonical
transcript is protein-coding, spans 5-60 kb, has >= 4 exons, and that overlap
no other protein-coding gene; at most 2 per region. HOLDOUT_GENES of them (seeded)
are held out entirely: they are used only to evaluate heads, never to train or
select them.

Windows. 600 nt of the genome + strand with the position at offset 300, the
dataset's own convention, so a - strand gene's decoy shows AC@298 / CT@301
exactly like a real - strand site in the training data. The script checks the
convention on the canonical sites of these genes first.

Embedding. For both models, the last-layer vector of the token covering base
300 (C) and of the token before it (P), computed exactly like the scan's window
features (10_scan_gene.window_pairs) and the training variants (02b, 12).

Outputs:
  data/hard_negatives/genes/<gene_id>.json   cached Ensembl records (+ sequence)
  data/hard_negatives/hard_negatives.tsv     one row per window (group, gene, split, ...)
  data/emb/<model>/hardneg/{C,P}.npy         float32 token vectors, rows as in the TSV
  results/hard_negatives.json                counts per group / split / subtype
"""

import argparse
import csv
import hashlib
import importlib
import json
import sys
import time

import numpy as np

from phase1_common import (DATA_DIR, EMB_DIR, MODELS, RESULTS_DIR, SEED, dataset_chromosomes,
                           model_slug, read_json, write_json_atomic)

sg = importlib.import_module("10_scan_gene")       # Ensembl helpers + window_pairs

HN_DIR = DATA_DIR / "hard_negatives"
GENE_CACHE = HN_DIR / "genes"
TSV = HN_DIR / "hard_negatives.tsv"
OUT_JSON = RESULTS_DIR / "hard_negatives.json"
WINDOW, CENTER = 600, 300
FLANK = 400                       # sequence fetched around the gene (>= half a window)
N_REGIONS = 60
REGION = 5_000_000
N_GENES = 75
HOLDOUT_GENES = 15
PER_REGION = 2
PER_GENE = 20                     # windows per group per gene (decoys: half donor-like)
MIN_DIST = {"exon_interior": 20, "intron_interior": 20, "decoy": 10}
GROUPS = list(MIN_DIST)
FIELDS = ["row", "split", "group", "subtype", "gene", "gene_id", "chrom", "strand",
          "pos", "in_canonical_exon", "motif_at_canonical_offset", "sequence"]


def train_chromosomes():
    ch = dataset_chromosomes()
    test, train = set(ch["test"]), set(ch["train"])
    if test & train:
        raise SystemExit(f"dataset train and test share chromosomes {sorted(test & train)}")
    use = sorted(train - {"Y"}, key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else 0, c))
    return use, test, ch


def candidate_genes(chroms, rng):
    """Seeded regions -> filtered protein-coding genes (cached listings)."""
    GENE_CACHE.mkdir(parents=True, exist_ok=True)
    asm_path = HN_DIR / "_assembly.json"
    if asm_path.exists():
        asm = json.loads(asm_path.read_text())
    else:
        info = json.loads(sg.fetch("/info/assembly/homo_sapiens"))
        asm = {r["name"]: r["length"] for r in info["top_level_region"]
               if r.get("coord_system") == "chromosome"}
        asm_path.write_text(json.dumps(asm))
    lengths = np.array([asm[c] for c in chroms], float)
    regions = []
    for _ in range(N_REGIONS):
        c = chroms[rng.choice(len(chroms), p=lengths / lengths.sum())]
        s = int(rng.integers(1, asm[c] - REGION))
        regions.append((c, s, s + REGION - 1))
    listing_path = HN_DIR / "_region_genes.json"
    listing = json.loads(listing_path.read_text()) if listing_path.exists() else {}
    for c, s, e in regions:
        key = f"{c}:{s}-{e}"
        if key not in listing:
            listing[key] = [g for g in json.loads(sg.fetch(f"/overlap/region/human/{key}?feature=gene"))
                            if g.get("biotype") == "protein_coding"]
            listing_path.write_text(json.dumps(listing))
            time.sleep(0.2)
    pre = sorted({g["id"] for c, s, e in regions for g in listing[f"{c}:{s}-{e}"]
                  if 5000 <= g["end"] - g["start"] + 1 <= 80000})
    lk_path = HN_DIR / "_lookup.json"
    looked = json.loads(lk_path.read_text()) if lk_path.exists() else {}
    missing = [i for i in pre if i not in looked]
    if missing:
        print(f"Looking up {len(missing)} genes in Ensembl ...")
        looked.update(sg.lookup_many(missing))
        lk_path.write_text(json.dumps(looked))
    picked, seen = [], set()
    for c, s, e in regions:
        pcs = listing[f"{c}:{s}-{e}"]
        ok = []
        for g in sorted(pcs, key=lambda g: g["id"]):
            cs = sg.canonical_summary(looked.get(g["id"]))
            if (cs is None or g["id"] in seen or cs["tx_biotype"] != "protein_coding"
                    or not 5000 <= cs["span"] <= 60000 or cs["exons"] < 4):
                continue
            if any(p["id"] != g["id"] and p["start"] <= cs["end"] and p["end"] >= cs["start"]
                   for p in pcs):
                continue
            ok.append(g["id"])
        for i in rng.permutation(len(ok))[:PER_REGION]:
            picked.append(ok[i])
            seen.add(ok[i])
        if len(picked) >= N_GENES:
            break
    return picked[:N_GENES], looked, len(regions)


def gene_record(gid, looked):
    """Canonical exons, sequence (+ strand, FLANK on each side) and every annotated
    splice site / exon in the region on both strands; cached per gene."""
    path = GENE_CACHE / f"{gid}.json"
    if path.exists():
        return json.loads(path.read_text())
    g = looked[gid]
    t = next(x for x in g["Transcript"] if x.get("is_canonical"))
    c, lo, hi = g["seq_region_name"], g["start"] - FLANK, g["end"] + FLANK
    seq = sg.fetch(f"/sequence/region/human/{c}:{lo}..{hi}:1?content-type=text/plain",
                   "text/plain").strip().upper()
    if len(seq) != hi - lo + 1:
        raise RuntimeError(f"{gid}: got {len(seq)} bases, expected {hi - lo + 1}")
    exons = json.loads(sg.fetch(f"/overlap/region/human/{c}:{lo}-{hi}?feature=exon"))
    by_tx = {}
    for e in exons:
        by_tx.setdefault(e["Parent"], []).append((e["start"], e["end"]))
    sites = set()
    for ex in by_tx.values():
        ex = sorted(set(ex))
        if len(ex) < 2:
            continue
        sites.update(s for s, _ in ex[1:])          # every exon start but the first
        sites.update(e for _, e in ex[:-1])          # every exon end but the last
    rec = {"gene_id": gid, "symbol": g.get("display_name") or gid, "chrom": c,
           "strand": g["strand"], "start": g["start"], "end": g["end"],
           "transcript_id": t["id"], "tx_start": t["start"], "tx_end": t["end"],
           "exons": sorted([[e["start"], e["end"]] for e in t["Exon"]]),
           "region_lo": lo, "region_hi": hi, "seq_plus": seq,
           "annotated_sites": sorted(sites),
           "annotated_exons": sorted({(e["start"], e["end"]) for e in exons}),
           "n_region_transcripts": len(by_tx)}
    path.write_text(json.dumps(rec))
    time.sleep(0.2)
    return rec


def window(rec, p):
    i = p - rec["region_lo"]
    return rec["seq_plus"][i - CENTER:i + CENTER]


def motif_of(w, strand):
    """Which canonical motif the window carries at the site offsets, in the gene's orientation."""
    if strand == 1:
        d, a = w[301:303] == "GT", w[298:300] == "AG"
    else:
        d, a = w[298:300] == "AC", w[301:303] == "CT"
    return "donor-like" if d and not a else "acceptor-like" if a and not d else \
        "both" if d and a else "none"


def canonical_sites(rec):
    ex = sorted(rec["exons"])
    ends, starts = [e for _, e in ex[:-1]], [s for s, _ in ex[1:]]
    # + strand: exon ends are donors; - strand: exon starts (genome) are donors
    return (ends, starts) if rec["strand"] == 1 else (starts, ends)


def site_distance(pos, sites):
    """Distance from each position to the nearest annotated site (10**9 if there is none)."""
    sites = np.unique(np.asarray(sites, dtype=np.int64))
    if len(sites) == 0:
        return np.full(len(pos), 10 ** 9)
    k = np.searchsorted(sites, pos)
    left = sites[np.clip(k - 1, 0, len(sites) - 1)]
    right = sites[np.clip(k, 0, len(sites) - 1)]
    return np.minimum(np.abs(pos - left), np.abs(right - pos))


def window_flags(rec, pos):
    """For every position p: does window(rec, p) hold 600 bases, all A/C/G/T (ok), and
    what motif_of(window(rec, p), strand) says ('none' where not ok). The same answers
    as building each window string, without building them (genes can span Mb)."""
    s = np.frombuffer(rec["seq_plus"].encode("ascii"), dtype=np.uint8)
    i = np.asarray(pos, dtype=np.int64) - rec["region_lo"]
    inside = (i - CENTER >= 0) & (i + CENTER <= len(s))
    ii = np.clip(i, CENTER, len(s) - CENTER)
    bad = np.concatenate([[0], np.cumsum(~np.isin(s, np.frombuffer(b"ACGT", np.uint8)))])
    ok = inside & (bad[ii + CENTER] - bad[ii - CENTER] == 0)

    def at(off, pair):                       # window[300+off : 302+off] == pair
        j = np.clip(ii + off, 0, len(s) - 2)
        return (s[j] == ord(pair[0])) & (s[j + 1] == ord(pair[1]))
    if rec["strand"] == 1:
        d, a = at(1, "GT"), at(-2, "AG")
    else:
        d, a = at(-2, "AC"), at(1, "CT")
    motif = np.where(d & ~a, "donor-like", np.where(a & ~d, "acceptor-like",
                                                    np.where(d & a, "both", "none")))
    motif[~ok] = "none"
    return ok, motif


def mine_gene(rec, rng, per_gene=PER_GENE):
    """Draw per_gene windows per group (decoys: half donor-like) from one gene."""
    lo, hi = rec["tx_start"], rec["tx_end"]
    pos = np.arange(lo, hi + 1)
    dist = site_distance(pos, rec["annotated_sites"])
    in_exon = np.zeros(len(pos), bool)
    for s, e in rec["exons"]:
        in_exon[max(s, lo) - lo:min(e, hi) - lo + 1] = True
    in_any_exon = np.zeros(len(pos), bool)
    for s, e in rec["annotated_exons"]:
        if e >= lo and s <= hi:
            in_any_exon[max(s, lo) - lo:min(e, hi) - lo + 1] = True
    ok, motif = window_flags(rec, pos)
    pools = {
        "exon_interior": ok & in_exon & (dist >= MIN_DIST["exon_interior"]),
        "intron_interior": ok & ~in_exon & ~in_any_exon & (dist >= MIN_DIST["intron_interior"]),
    }
    dec = ok & (dist >= MIN_DIST["decoy"])
    taken = np.zeros(len(pos), bool)
    rows = []

    def take(mask, n, group, subtype):
        idx = np.flatnonzero(mask & ~taken)
        for j in idx[rng.permutation(len(idx))[:n]]:
            taken[j] = True
            rows.append({"group": group, "subtype": subtype, "pos": int(pos[j]),
                         "in_canonical_exon": bool(in_exon[j]),
                         "motif_at_canonical_offset": motif[j],
                         "sequence": window(rec, int(pos[j]))})
    # decoys first, so the interior groups are drawn from what is left
    take(dec & (motif == "donor-like"), per_gene // 2, "decoy", "donor-like (GT after)")
    take(dec & (motif == "acceptor-like"), per_gene - per_gene // 2, "decoy",
         "acceptor-like (AG before)")
    for g in ("exon_interior", "intron_interior"):
        take(pools[g], per_gene, g, "")
    return rows, {"canonical_span": int(hi - lo + 1), "exon_pool": int(pools["exon_interior"].sum()),
                  "intron_pool": int(pools["intron_interior"].sum())}


def mine(args):
    chroms, test, ch = train_chromosomes()
    rng = np.random.default_rng(SEED)
    gids, looked, n_regions = candidate_genes(chroms, rng)
    if len(gids) < N_GENES:
        raise SystemExit(f"only {len(gids)} candidate genes; raise N_REGIONS")
    hold = set(np.array(gids)[np.random.default_rng(SEED + 1).permutation(len(gids))[:HOLDOUT_GENES]])
    rows, per_gene, conv = [], {}, {"donor": [0, 0], "acceptor": [0, 0]}
    for gi, gid in enumerate(gids):
        rec = gene_record(gid, looked)
        if rec["chrom"] in test or rec["chrom"] not in chroms:
            raise SystemExit(f"{gid} is on chr{rec['chrom']}, not a training chromosome")
        # convention check: real canonical sites must show the dataset's motif geometry
        dons, accs = canonical_sites(rec)
        for kind, ss in (("donor", dons), ("acceptor", accs)):
            for s in ss:
                m = motif_of(window(rec, s), rec["strand"])
                conv[kind][0] += m in (f"{kind}-like", "both")
                conv[kind][1] += 1
        r, info = mine_gene(rec, np.random.default_rng([SEED, gi]))
        split = "holdout" if gid in hold else "train"
        for x in r:
            x.update(split=split, gene=rec["symbol"], gene_id=gid, chrom=rec["chrom"],
                     strand="+" if rec["strand"] == 1 else "-")
        rows += r
        per_gene[gid] = {"symbol": rec["symbol"], "chrom": rec["chrom"], "split": split,
                         "strand": rec["strand"], "exons": len(rec["exons"]), **info,
                         "windows": {g: sum(x["group"] == g for x in r) for g in GROUPS}}
        print(f"  {gi + 1:>2}/{len(gids)} {rec['symbol']:<12} chr{rec['chrom']:<2} "
              f"{'+' if rec['strand'] == 1 else '-'} {split:<7} span {info['canonical_span']:>6,} "
              f"-> {per_gene[gid]['windows']}")
    for i, x in enumerate(rows):
        x["row"] = i
    HN_DIR.mkdir(parents=True, exist_ok=True)
    with open(TSV, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    counts = {}
    for sp in ("train", "holdout"):
        counts[sp] = {g: sum(x["split"] == sp and x["group"] == g for x in rows) for g in GROUPS}
        counts[sp]["total"] = sum(counts[sp][g] for g in GROUPS)
    sub = {}
    for x in rows:
        k = (x["group"] + (f" / {x['subtype']}" if x["subtype"] else "")
             + (" / exon" if x["in_canonical_exon"] else " / intron"))
        sub[k] = sub.get(k, 0) + 1
    motif_in_interior = {g: {m: sum(x["group"] == g and x["motif_at_canonical_offset"] == m
                                    for x in rows) for m in ("donor-like", "acceptor-like", "both")}
                         for g in ("exon_interior", "intron_interior")}
    res = {
        "chromosomes": {"dataset_train": ch["train"], "dataset_test": ch["test"],
                        "used_for_mining": chroms,
                        "note": "chr22 is in neither dataset split; chrY skipped"},
        "rule": {"regions": n_regions, "region_bases": REGION, "genes": len(gids),
                 "holdout_genes": HOLDOUT_GENES, "per_gene_per_group": PER_GENE,
                 "min_distance_to_any_annotated_site": MIN_DIST, "seed": SEED,
                 "window": "genome + strand, position at offset 300 (dataset convention)"},
        "convention_check": {k: {"with_canonical_motif": v[0], "sites": v[1]}
                             for k, v in conv.items()},
        "counts": counts, "by_subtype": dict(sorted(sub.items())),
        "interior_windows_that_also_carry_a_motif": motif_in_interior,
        "genes": per_gene, "tsv": str(TSV.relative_to(DATA_DIR.parent)),
        "tsv_sha1": hashlib.sha1(TSV.read_bytes()).hexdigest(),
    }
    write_json_atomic(OUT_JSON, res)
    print(f"\nConvention check (canonical sites of these genes with the dataset's motif "
          f"geometry): donors {conv['donor'][0]}/{conv['donor'][1]}, acceptors "
          f"{conv['acceptor'][0]}/{conv['acceptor'][1]}")
    for sp, c in counts.items():
        print(f"{sp:<8} " + ", ".join(f"{g} {c[g]}" for g in GROUPS) + f"  (total {c['total']})")
    return res


def load_rows():
    with open(TSV) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def embed(repo, rows, sha):
    import torch
    from transformers import AutoTokenizer

    from try_dna_models import load_model

    out = EMB_DIR / model_slug(repo) / "hardneg"
    meta = read_json(out / "meta.json", default=None)
    if meta and meta.get("tsv_sha1") == sha and (out / "C.npy").exists():
        print(f"{MODELS[repo]}: hard-negative embeddings up to date ({meta['n']} windows)")
        return
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model, _ = load_model(repo)
    model.eval()
    texts = [r["sequence"] for r in rows]
    t0 = time.time()
    Cs, Ps = [], []
    for i in range(0, len(texts), 512):
        C, P = sg.window_pairs(texts[i:i + 512], tok, model, torch, batch=32)
        Cs.append(C)
        Ps.append(P)
        print(f"  {MODELS[repo]}: {min(i + 512, len(texts)):,}/{len(texts):,} windows "
              f"({(min(i + 512, len(texts))) / (time.time() - t0):.1f}/s)", flush=True)
    np.save(out / "C.npy", np.concatenate(Cs).astype(np.float32))
    np.save(out / "P.npy", np.concatenate(Ps).astype(np.float32))
    write_json_atomic(out / "meta.json", {
        "n": len(texts), "tsv_sha1": sha, "hidden": int(Cs[0].shape[1]),
        "rows": "same order as data/hard_negatives/hard_negatives.tsv",
        "content": "C = last-layer vector of the token covering base 300, P = the token "
                   "before it (as 10_scan_gene.window_pairs); float32",
        "seconds": round(time.time() - t0, 1)})
    print(f"{MODELS[repo]}: embedded {len(texts):,} windows in {(time.time() - t0) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser(description="Mine and embed hard negatives.")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--mine-only", action="store_true")
    ap.add_argument("--remine", action="store_true", help="rebuild the TSV from the cached genes")
    args = ap.parse_args()
    if args.remine or not TSV.exists() or not OUT_JSON.exists():
        mine(args)
    else:
        c = read_json(OUT_JSON)["counts"]
        print(f"Hard negatives already mined ({TSV}): " + "; ".join(
            f"{sp} " + ", ".join(f"{g} {c[sp][g]}" for g in GROUPS) for sp in c))
    if args.mine_only:
        return 0
    rows = load_rows()
    sha = hashlib.sha1(TSV.read_bytes()).hexdigest()
    for repo in args.models:
        embed(repo, rows, sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
