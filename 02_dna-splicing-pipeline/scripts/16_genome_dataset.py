#!/usr/bin/env python3
"""
Step 16: a genome-wide splice-site dataset (GRCh38 + MANE Select), in the format of
splice_sites_all, built with the window code of 13_hard_negatives.py.

    python scripts/16_genome_dataset.py             # download (once), build, check
    python scripts/16_genome_dataset.py --rebuild   # rebuild the parquet files from the sources

Why. splice_sites_all has 33,000 windows; the advisor asked for 70,000-100,000+.

Sources, downloaded once into data/genome/source/ (git-ignored; the exact files, sizes and
checksums are written to data/genome/README.md):
  GRCh38.p14 genome FASTA, NCBI RefSeq GCF_000001405.40      } the pair OpenSpliceAI's MANE
  MANE v1.3 RefSeq GFF, MANE Select transcripts only         } training guide links
  GENCODE v50 comprehensive GTF (reference chromosomes), used only to keep negatives
  away from every annotated splice site of every transcript.

Windows. 13_hard_negatives.window(): 600 nt of the genome + strand with the position at
offset 300; a donor is the last exon base, an acceptor the first exon base (13's
canonical_sites), so + strand donors show GT@301 and - strand donors AC@298, exactly as in
splice_sites_all. Labels as data/label_info.json: 0 none, 1 acceptor, 2 donor.

  positives  every donor and acceptor of every MANE Select transcript on chr1-22, X, Y
  negatives  13_hard_negatives.mine_gene() on every MANE gene, i.e. 13's three groups with
             13's distances: GT/AG decoys at the site offset >= 10 nt from any annotated site,
             exon interiors >= 20 nt from any site, intron interiors >= 20 nt from any site
             and outside every annotated exon. "Annotated" = any GENCODE v50 transcript
             (all biotypes and levels, both strands) plus MANE. Each gene gets negatives in
             proportion to its intron count, so genes weigh the same in every class; per split
             the three groups are then cut to equal thirds (decoys half GT, half AG) and the
             none class to the size of the donor and acceptor classes.

Split by chromosome as splice_sites_all: test = chr20 + chr21, train = every other
chromosome. The script stops if a train window comes from chr20/21.

Outputs (data/ is git-ignored):
  data/genome/grch38/chr*.seq                         primary chromosomes, upper case, one line
  data/genome/splice_sites_genome_full/{train,test}.parquet        every window
  data/genome/splice_sites_genome_full/{train,test}_meta.parquet   row by row: kind, group,
                                                       gene, transcript, strand, motif
  data/genome/splice_sites_genome/{train,test}.parquet   default: a seeded, class-balanced
                                                       120,000-window train + the full test
  data/genome/README.md                               files, versions, layout
  results/genome_dataset.json, results/genome_dataset.md   counts and checks
Parquet columns: those of splice_sites_all (sequence, name, label, task) plus chrom and
position (1-based GRCh38 coordinate of the base at offset 300).
"""

import argparse
import gzip
import hashlib
import importlib
import json
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

from phase1_common import (DATA_DIR, GENOME_DEFAULT_DIR, GENOME_DIR, GENOME_FULL_DIR,
                           RESULTS_DIR, SEED, dinuc_hits, load_label_info, load_splits,
                           read_json, seq_matrix, write_json_atomic)

hn = importlib.import_module("13_hard_negatives")   # window, canonical_sites, mine_gene
gd = importlib.import_module("01_get_data")         # infer_label_semantics (motif check)

SRC = GENOME_DIR / "source"
CHROM_DIR = GENOME_DIR / "grch38"
FULL_DIR = GENOME_FULL_DIR
DEF_DIR = GENOME_DEFAULT_DIR
README = GENOME_DIR / "README.md"
OUT_JSON = RESULTS_DIR / "genome_dataset.json"
OUT_MD = RESULTS_DIR / "genome_dataset.md"

NCBI = "https://ftp.ncbi.nlm.nih.gov"
ASM = f"{NCBI}/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14"
MANE = f"{NCBI}/refseq/MANE/MANE_human/release_1.3"
GENCODE = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50"
SOURCES = {
    "genome": ("GCF_000001405.40_GRCh38.p14_genomic.fna.gz",
               f"{ASM}/GCF_000001405.40_GRCh38.p14_genomic.fna.gz",
               "GRCh38.p14 genome, NCBI RefSeq assembly GCF_000001405.40 (OpenSpliceAI's link)"),
    "assembly_report": ("GCF_000001405.40_GRCh38.p14_assembly_report.txt",
                        f"{ASM}/GCF_000001405.40_GRCh38.p14_assembly_report.txt",
                        "sequence names and lengths of that assembly"),
    "genome_md5": ("GCF_000001405.40_GRCh38.p14_md5checksums.txt", f"{ASM}/md5checksums.txt",
                   "NCBI's published MD5 checksums of that assembly folder"),
    "mane": ("MANE.GRCh38.v1.3.refseq_genomic.gff.gz",
             f"{MANE}/MANE.GRCh38.v1.3.refseq_genomic.gff.gz",
             "MANE v1.3 RefSeq GFF (OpenSpliceAI's link); MANE Select transcripts are used"),
    "mane_versions": ("MANE.GRCh38.v1.3.README_versions.txt", f"{MANE}/README_versions.txt",
                      "MANE v1.3 release versions (RefSeq annotation release, Ensembl release)"),
    "gencode": ("gencode.v50.annotation.gtf.gz", f"{GENCODE}/gencode.v50.annotation.gtf.gz",
                "GENCODE v50 comprehensive annotation, reference chromosomes"),
    "gencode_md5": ("gencode.v50.MD5SUMS", f"{GENCODE}/MD5SUMS",
                    "GENCODE's published MD5 checksums of release 50"),
}
CHROMS = [f"chr{c}" for c in [*range(1, 23), "X", "Y"]]
TEST_CHROMS = {"chr20", "chr21"}
TASK = "splice_sites_genome"
DEFAULT_PER_CLASS = 40_000
OVERSAMPLE = 1.10            # negatives drawn per gene before the per-split cut to thirds
SITE_MARGIN = 1000           # annotated sites fetched this far beyond a gene's region
COLUMNS = ["sequence", "name", "label", "task", "chrom", "position"]
META_COLUMNS = ["name", "chrom", "position", "label", "kind", "group", "subtype", "strand",
                "gene", "gene_id", "transcript_id", "ensembl_transcript", "in_canonical_exon",
                "motif_at_canonical_offset"]
NEG_KINDS = [("decoy", "donor-like (GT after)"), ("decoy", "acceptor-like (AG before)"),
             ("exon_interior", ""), ("intron_interior", "")]


def src(key):
    return SRC / SOURCES[key][0]


# ----------------------------------------------------------------------------- sources
def download():
    SRC.mkdir(parents=True, exist_ok=True)
    for key, (name, url, _) in SOURCES.items():
        path = SRC / name
        if path.exists():
            continue
        print(f"Downloading {url}")
        tmp = path.with_name(name + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
        tmp.rename(path)


def checksums():
    """Size, MD5 and SHA-256 of every source file (cached by size + mtime), and whether the
    MD5 matches the one NCBI / GENCODE publish."""
    cache_path = SRC / "_checksums.json"
    cache = read_json(cache_path, default={}) or {}
    published = {}
    for line in open(src("genome_md5")):
        md5, f = line.split()
        published[f.removeprefix("./")] = md5
    for line in open(src("gencode_md5")):
        md5, f = line.split()
        published[f] = md5
    out = {}
    for key, (name, url, what) in SOURCES.items():
        p = SRC / name
        st = p.stat()
        c = cache.get(name)
        if not c or c["bytes"] != st.st_size or c["mtime"] != st.st_mtime:
            h5, h256 = hashlib.md5(), hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(1 << 24):
                    h5.update(chunk)
                    h256.update(chunk)
            c = {"bytes": st.st_size, "mtime": st.st_mtime, "md5": h5.hexdigest(),
                 "sha256": h256.hexdigest()}
            cache[name] = c
        out[key] = {"file": name, "url": url, "what": what, "bytes": c["bytes"],
                    "md5": c["md5"], "sha256": c["sha256"],
                    "published_md5": published.get(name),
                    "md5_matches_published": (published.get(name) == c["md5"])
                    if name in published else None,
                    "downloaded_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc)
                    .strftime("%Y-%m-%d")}
    write_json_atomic(cache_path, cache)
    bad = [k for k, v in out.items() if v["md5_matches_published"] is False]
    if bad:
        raise SystemExit(f"MD5 mismatch against the published checksum: {bad}")
    return out


def extract_chromosomes():
    """Primary chromosomes of the RefSeq FASTA -> data/genome/grch38/chrN.seq (upper case,
    no line breaks), checked against the lengths in the assembly report."""
    man_path = CHROM_DIR / "manifest.json"
    if man_path.exists():
        return read_json(man_path)
    names = {}
    for line in open(src("assembly_report")):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if f[1] == "assembled-molecule" and f[9] in CHROMS:
            names[f[6]] = (f[9], int(f[8]))          # RefSeq accession -> (chrN, length)
    CHROM_DIR.mkdir(parents=True, exist_ok=True)
    man, out, cur, n = {}, None, None, 0

    def close():
        if out is not None:
            out.close()
            c, length = names[cur]
            if n != length:
                raise SystemExit(f"{c}: {n} bases written, assembly report says {length}")
            (CHROM_DIR / f"{c}.seq.part").rename(CHROM_DIR / f"{c}.seq")
            man[c] = {"accession": cur, "length": n, "file": f"{c}.seq"}
            print(f"  {c:<6} {cur:<13} {n:>12,} bases")
    print(f"Extracting the primary chromosomes from {src('genome').name} ...")
    with gzip.open(src("genome"), "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                close()
                acc = line[1:].split()[0]
                cur, out, n = (acc, open(CHROM_DIR / f"{names[acc][0]}.seq.part", "w"), 0) \
                    if acc in names else (None, None, 0)
            elif out is not None:
                s = line.rstrip("\n").upper()
                out.write(s)
                n += len(s)
    close()
    if sorted(man) != sorted(CHROMS):
        raise SystemExit(f"missing chromosomes: {sorted(set(CHROMS) - set(man))}")
    write_json_atomic(man_path, man)
    return man


def load_chrom(c):
    return (CHROM_DIR / f"{c}.seq").read_text()


def gff_attrs(col):
    return dict(kv.split("=", 1) for kv in col.split(";") if "=" in kv)


def parse_mane():
    """MANE Select transcripts (not MANE Plus Clinical) with their exons."""
    tx, exons = {}, defaultdict(list)
    with gzip.open(src("mane"), "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[2] not in ("mRNA", "exon"):
                continue
            a = gff_attrs(p[8])
            if a.get("tag") != "MANE Select":
                continue
            if p[2] == "mRNA":
                dbx = dict(x.split(":", 1) for x in a.get("Dbxref", "").split(",") if ":" in x)
                tx[a["ID"]] = {"chrom": p[0], "strand": 1 if p[6] == "+" else -1,
                               "start": int(p[3]), "end": int(p[4]), "symbol": a.get("gene"),
                               "gene_id": f"GeneID:{dbx.get('GeneID')}",
                               "transcript_id": a.get("transcript_id"),
                               "ensembl_transcript": dbx.get("Ensembl", "")}
            else:
                exons[a["Parent"]].append((int(p[3]), int(p[4])))
    out = []
    for tid, t in tx.items():
        ex = sorted(set(exons[tid]))
        if not ex or ex[0][0] != t["start"] or ex[-1][1] != t["end"]:
            raise SystemExit(f"{tid}: exons do not span the transcript")
        out.append({**t, "exons": [list(e) for e in ex]})
    return out


def parse_gencode():
    """Every exon of every GENCODE transcript, and every internal exon boundary (every exon
    start but the first, every exon end but the last, as 13's gene_record), per chromosome."""
    rx = re.compile(r'transcript_id "([^"]+)"')
    per_tx, chrom_of, header = defaultdict(list), {}, []
    with gzip.open(src("gencode"), "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                header.append(line.strip())
                continue
            p = line.split("\t", 8)
            if p[2] != "exon":
                continue
            t = rx.search(p[8]).group(1)
            per_tx[t].append((int(p[3]), int(p[4])))
            chrom_of[t] = p[0]
    sites, exons = defaultdict(set), defaultdict(set)
    for t, ex in per_tx.items():
        c = chrom_of[t]
        ex = sorted(set(ex))
        exons[c].update(ex)
        if len(ex) >= 2:
            sites[c].update(s for s, _ in ex[1:])
            sites[c].update(e for _, e in ex[:-1])
    return sites, exons, {"transcripts": len(per_tx), "header": header}


# ----------------------------------------------------------------------------- build
def gene_record(t, seq, sites_c, exons_c):
    """A MANE transcript as 13_hard_negatives' gene record (same fields), from the genome."""
    lo, hi = t["start"] - hn.FLANK, t["end"] + hn.FLANK
    if lo < 1 or hi > len(seq):
        raise SystemExit(f"{t['transcript_id']}: region runs off {t['chrom']}")
    a, b = np.searchsorted(sites_c, [lo - SITE_MARGIN, hi + SITE_MARGIN + 1])
    ov = exons_c[(exons_c[:, 0] <= hi) & (exons_c[:, 1] >= lo)]
    return {"gene_id": t["gene_id"], "symbol": t["symbol"], "chrom": t["chrom"],
            "strand": t["strand"], "start": t["start"], "end": t["end"],
            "transcript_id": t["transcript_id"], "tx_start": t["start"], "tx_end": t["end"],
            "exons": t["exons"], "region_lo": lo, "region_hi": hi, "seq_plus": seq[lo - 1:hi],
            "annotated_sites": sites_c[a:b], "annotated_exons": [tuple(e) for e in ov.tolist()]}


def quotas(genes, per_group):
    """Even per-gene quotas, proportional to intron counts, summing to OVERSAMPLE x per_group
    (seeded order for the rounding); even so decoys split exactly half GT / half AG."""
    k = np.array([len(g["exons"]) - 1 for g in genes], float)
    order = np.random.default_rng(SEED).permutation(len(genes))
    r = np.round(np.cumsum(k[order]) * (OVERSAMPLE * per_group / 2) / k.sum()).astype(int)
    q = np.zeros(len(genes), int)
    q[order] = 2 * np.diff(np.concatenate([[0], r]))
    return q


def build():
    info = load_label_info()
    lab = {"none": info["none_label"], "acceptor": info["acceptor_label"],
           "donor": info["donor_label"]}
    if lab != {"none": 0, "acceptor": 1, "donor": 2}:
        raise SystemExit(f"unexpected label mapping {lab}")
    print("Parsing MANE ...")
    mane = parse_mane()
    n_mane = len(mane)
    on_primary = [t for t in mane if t["chrom"] in CHROMS]
    genes = sorted([t for t in on_primary if len(t["exons"]) >= 2],
                   key=lambda t: (CHROMS.index(t["chrom"]), t["start"], t["transcript_id"]))
    print(f"  {n_mane:,} MANE Select transcripts, {len(on_primary):,} on chr1-22/X/Y, "
          f"{len(genes):,} with at least one intron")
    print("Parsing GENCODE ...")
    sites, exons, gc_info = parse_gencode()
    for t in on_primary:                         # MANE itself counts as annotated too
        ex = [tuple(e) for e in t["exons"]]
        exons[t["chrom"]].update(ex)
        sites[t["chrom"]].update(s for s, _ in ex[1:])
        sites[t["chrom"]].update(e for _, e in ex[:-1])
    sites = {c: np.array(sorted(v), dtype=np.int64) for c, v in sites.items()}
    exons = {c: np.array(sorted(v), dtype=np.int64).reshape(-1, 2) for c, v in exons.items()}
    print(f"  {gc_info['transcripts']:,} GENCODE transcripts; "
          f"{sum(len(v) for v in sites.values()):,} distinct annotated splice sites")

    split_of = {t["transcript_id"]: "test" if t["chrom"] in TEST_CHROMS else "train"
                for t in genes}
    q = np.zeros(len(genes), int)
    for sp in ("train", "test"):
        idx = [i for i, t in enumerate(genes) if split_of[t["transcript_id"]] == sp]
        introns = sum(len(genes[i]["exons"]) - 1 for i in idx)
        q[idx] = quotas([genes[i] for i in idx], introns / 3)

    pos_rows, neg_rows = [], []
    conv = {"donor": [0, 0], "acceptor": [0, 0]}
    skipped = Counter()
    t0 = time.time()
    for c in CHROMS:
        gi_c = [i for i, t in enumerate(genes) if t["chrom"] == c]
        if not gi_c:
            continue
        seq = load_chrom(c)
        for gi in gi_c:
            t = genes[gi]
            rec = gene_record(t, seq, sites[c], exons[c])
            base = {"chrom": c, "strand": "+" if t["strand"] == 1 else "-", "gene": t["symbol"],
                    "gene_id": t["gene_id"], "transcript_id": t["transcript_id"],
                    "ensembl_transcript": t["ensembl_transcript"],
                    "split": split_of[t["transcript_id"]]}
            for kind, ss in zip(("donor", "acceptor"), hn.canonical_sites(rec)):
                for p in ss:
                    w = hn.window(rec, p)
                    if len(w) != hn.WINDOW or not set(w) <= set("ACGT"):
                        skipped[kind] += 1
                        continue
                    m = hn.motif_of(w, rec["strand"])
                    conv[kind][0] += m in (f"{kind}-like", "both")
                    conv[kind][1] += 1
                    pos_rows.append({**base, "position": int(p), "label": lab[kind],
                                     "kind": kind, "group": kind, "subtype": "",
                                     "in_canonical_exon": True,
                                     "motif_at_canonical_offset": m, "sequence": w})
            if q[gi]:
                rows, _ = hn.mine_gene(rec, np.random.default_rng([SEED, gi]), per_gene=int(q[gi]))
                for r in rows:
                    neg_rows.append({**base, "position": r["pos"], "label": lab["none"],
                                     "kind": "none", "group": r["group"], "subtype": r["subtype"],
                                     "in_canonical_exon": r["in_canonical_exon"],
                                     "motif_at_canonical_offset": str(r["motif_at_canonical_offset"]),
                                     "sequence": r["sequence"]})
        del seq
        print(f"  {c:<6} {len(gi_c):>5} genes  -> {len(pos_rows):>8,} sites, "
              f"{len(neg_rows):>8,} negatives so far  ({time.time() - t0:.0f}s)", flush=True)

    # positives: one row per (chrom, position); a position that is a site of two kinds is dropped
    by_pos = defaultdict(list)
    for r in pos_rows:
        by_pos[(r["chrom"], r["position"])].append(r)
    conflicts = sum(len({r["label"] for r in v}) > 1 for v in by_pos.values())
    dup_same = sum(len(v) > 1 and len({r["label"] for r in v}) == 1 for v in by_pos.values())
    pos_rows = [v[0] for v in by_pos.values() if len({r["label"] for r in v}) == 1]
    # negatives: one row per (chrom, position) (overlapping genes can draw the same base)
    seen, uniq = set(), []
    for r in neg_rows:
        k = (r["chrom"], r["position"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    neg_dups = len(neg_rows) - len(uniq)
    neg_rows = uniq

    # per split: none = mean of donors and acceptors, in equal thirds (decoys half GT, half AG)
    splits, cut = {}, {}
    for sp in ("train", "test"):
        P = [r for r in pos_rows if r["split"] == sp]
        nd = sum(r["kind"] == "donor" for r in P)
        na = sum(r["kind"] == "acceptor" for r in P)
        g = (round((nd + na) / 2) // 3) // 2 * 2
        pools = {k: [r for r in neg_rows if r["split"] == sp and (r["group"], r["subtype"]) == k]
                 for k in NEG_KINDS}
        need = {k: g // 2 if k[0] == "decoy" else g for k in NEG_KINDS}
        short = {f"{k[0]} {k[1]}".strip(): (len(pools[k]), need[k])
                 for k in NEG_KINDS if len(pools[k]) < need[k]}
        if short:
            raise SystemExit(f"{sp}: not enough negatives {short}; raise OVERSAMPLE")
        rng = np.random.default_rng([SEED, 7, int(sp == "test")])
        N = []
        for k in NEG_KINDS:
            pick = np.sort(rng.choice(len(pools[k]), need[k], replace=False))
            N += [pools[k][i] for i in pick]
        cut[sp] = {"drawn": {f"{k[0]} {k[1]}".strip(): len(pools[k]) for k in NEG_KINDS},
                   "kept": {f"{k[0]} {k[1]}".strip(): need[k] for k in NEG_KINDS}}
        rows = P + N
        perm = np.random.default_rng([SEED, int(sp == "test")]).permutation(len(rows))
        splits[sp] = [rows[i] for i in perm]
    bad = [r for r in splits["train"] if r["chrom"] in TEST_CHROMS]
    if bad or any(r["chrom"] not in TEST_CHROMS for r in splits["test"]):
        raise SystemExit("chromosome split violated")
    return splits, {
        "mane_select_transcripts": n_mane, "mane_on_primary_chromosomes": len(on_primary),
        "mane_with_introns": len(genes),
        "mane_off_primary": sorted({t["chrom"] for t in mane if t["chrom"] not in CHROMS}),
        "gencode": {"transcripts": gc_info["transcripts"], "header": gc_info["header"]},
        "annotated_sites_total": int(sum(len(v) for v in sites.values())),
        "convention_check": {k: {"with_canonical_motif": v[0], "sites": v[1]}
                             for k, v in conv.items()},
        "positives_skipped_non_ACGT": dict(skipped), "positions_with_conflicting_labels": conflicts,
        "positions_shared_by_two_transcripts_same_label": dup_same,
        "negative_duplicates_removed": neg_dups, "negatives_per_group": cut,
    }


def name_of(r, offset):
    lo = r["position"] - 301 + offset
    return f"{r['chrom']}:{lo}-{lo + hn.WINDOW}|{r['label']}"


def name_convention(old):
    """Which `name` coordinates splice_sites_all uses: for every old row, compare the
    sequence with the genome at start..end read as 0-based half-open (offset 0) or shifted
    by one. Returns the offset to add to pos - 301 and the match rates."""
    by_c = defaultdict(list)
    for sp in old:
        for n, s in zip(old[sp]["name"], old[sp]["sequence"]):
            c, rng_ = n.split("|")[0].split(":")
            a, b = map(int, rng_.split("-"))
            by_c[c].append((a, b, s))
    hits, total = Counter(), 0
    for c, rows in by_c.items():
        seq = load_chrom(c)
        for a, b, s in rows:
            total += 1
            hits[0] += seq[a:b] == s
            hits[1] += seq[a - 1:b - 1] == s
    rates = {k: hits[k] / total for k in (0, 1)}
    best = max(rates, key=rates.get)
    if rates[best] < 0.99:
        raise SystemExit(f"old windows do not match the genome under either convention: {rates}")
    return best, {"rows": total, "match_0based_halfopen": round(rates[0], 5),
                  "match_shifted_by_1": round(rates[1], 5)}


def write_parquet(rows, path, columns, offset):
    import pyarrow as pa
    import pyarrow.parquet as pq

    types = {"sequence": pa.string(), "name": pa.string(), "label": pa.int32(),
             "task": pa.string(), "chrom": pa.string(), "position": pa.int64(),
             "kind": pa.string(), "group": pa.string(), "subtype": pa.string(),
             "strand": pa.string(), "gene": pa.string(), "gene_id": pa.string(),
             "transcript_id": pa.string(), "ensembl_transcript": pa.string(),
             "in_canonical_exon": pa.bool_(), "motif_at_canonical_offset": pa.string()}
    data = {c: [] for c in columns}
    for r in rows:
        for c in columns:
            data[c].append(name_of(r, offset) if c == "name" else TASK if c == "task" else r[c])
    tbl = pa.table({c: pa.array(data[c], type=types[c]) for c in columns})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, path, compression="zstd")


def default_train(train):
    """Seeded, class-balanced 120,000: 40,000 donors, 40,000 acceptors, 40,000 none in
    thirds (decoys 6,667 GT + 6,667 AG, exon 13,333, intron 13,333)."""
    third = DEFAULT_PER_CLASS // 3                                   # 13,333
    decoy = (DEFAULT_PER_CLASS - 2 * third) // 2                     # 6,667 GT + 6,667 AG
    need = {("donor", ""): DEFAULT_PER_CLASS, ("acceptor", ""): DEFAULT_PER_CLASS,
            NEG_KINDS[0]: decoy, NEG_KINDS[1]: decoy, NEG_KINDS[2]: third, NEG_KINDS[3]: third}
    assert sum(need[k] for k in NEG_KINDS) == DEFAULT_PER_CLASS
    rng = np.random.default_rng([SEED, 120])
    keep = []
    for k, n in need.items():
        idx = [i for i, r in enumerate(train) if (r["group"], r["subtype"]) == k]
        keep += list(np.array(idx)[rng.choice(len(idx), n, replace=False)])
    return [train[i] for i in sorted(keep)]


# ----------------------------------------------------------------------------- checks
def read_split(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pydict()


def motif_check(seqs, y):
    """01_get_data's label-semantics check, unchanged, plus its strand mix."""
    mat = seq_matrix(seqs)
    y = np.asarray(y)
    res = gd.infer_label_semantics(mat, y)
    o = res["motif_offsets_0based"]
    if res["status"] == "inferred":
        d, a = res["donor_label"], res["acceptor_label"]
        res["strand_mix"] = {
            "donor_plus": round(float(dinuc_hits(mat, "GT")[y == d, o["donor_GT_plus"]].mean()), 4),
            "donor_minus": round(float(dinuc_hits(mat, "AC")[y == d, o["donor_AC_minus"]].mean()), 4),
            "acceptor_plus": round(float(dinuc_hits(mat, "AG")[y == a, o["acceptor_AG_plus"]].mean()), 4),
            "acceptor_minus": round(float(dinuc_hits(mat, "CT")[y == a, o["acceptor_CT_minus"]].mean()), 4)}
    res["gc_by_class"] = {str(c): round(float(np.isin(mat[y == c], [ord("G"), ord("C")]).mean()), 4)
                          for c in sorted(set(y.tolist()))}
    return res


def checks(build_info, sources, man):
    import pyarrow.parquet as pq
    from datasets import Features

    old = load_splits()
    offset, conv_old = name_convention(old)
    full = {sp: read_split(FULL_DIR / f"{sp}.parquet") for sp in ("train", "test")}
    meta = {sp: read_split(FULL_DIR / f"{sp}_meta.parquet") for sp in ("train", "test")}
    dflt = {sp: read_split(DEF_DIR / f"{sp}.parquet") for sp in ("train", "test")}
    names = {0: "none", 1: "acceptor", 2: "donor"}

    def counts(d):
        return {sp: {**{names[k]: v for k, v in sorted(Counter(d[sp]["label"]).items())},
                     "total": len(d[sp]["label"])} for sp in d}
    groups = {sp: dict(sorted(Counter(f"{g} {s}".strip() for g, s, k in zip(
        meta[sp]["group"], meta[sp]["subtype"], meta[sp]["kind"]) if k == "none").items()))
        for sp in meta}
    dgroups = {}
    mk = {n: (g, s) for sp in meta for n, g, s in zip(meta[sp]["name"], meta[sp]["group"],
                                                       meta[sp]["subtype"])}
    for sp in dflt:
        dgroups[sp] = dict(sorted(Counter(f"{mk[n][0]} {mk[n][1]}".strip() for n, y in zip(
            dflt[sp]["name"], dflt[sp]["label"]) if y == 0).items()))
    genes = {sp: len(set(meta[sp]["gene_id"])) for sp in meta}
    pos_genes = {sp: len({g for g, k in zip(meta[sp]["gene_id"], meta[sp]["kind"]) if k != "none"})
                 for sp in meta}
    dnames = set(dflt["train"]["name"])
    dgenes = {"train": len({g for n, g in zip(meta["train"]["name"], meta["train"]["gene_id"])
                            if n in dnames})}

    # chromosome split
    split_chroms = {sp: sorted(set(full[sp]["chrom"]), key=CHROMS.index) for sp in full}
    assert not set(split_chroms["train"]) & TEST_CHROMS
    assert set(split_chroms["test"]) <= TEST_CHROMS
    # every window really is the genome at its name, and position is base 300
    spot = {}
    for sp in full:
        rng = np.random.default_rng([SEED, 3])
        idx = rng.choice(len(full[sp]["name"]), min(2000, len(full[sp]["name"])), replace=False)
        by_c = defaultdict(list)
        for i in idx:
            by_c[full[sp]["chrom"][i]].append(i)
        ok = 0
        for c, ii in by_c.items():
            seq = load_chrom(c)
            for i in ii:
                n = full[sp]["name"][i]
                a, b = map(int, n.split("|")[0].split(":")[1].split("-"))
                ok += (seq[a - offset:b - offset] == full[sp]["sequence"][i]
                       and full[sp]["sequence"][i][300] == seq[full[sp]["position"][i] - 1]
                       and int(n.split("|")[1]) == full[sp]["label"][i])
        spot[sp] = {"checked": int(len(idx)), "match": int(ok)}

    # schema: same features as splice_sites_all for the shared columns
    old_feat = old["train"].features
    new_feat = Features.from_arrow_schema(pq.read_schema(DEF_DIR / "train.parquet"))
    schema_same = all(str(old_feat[c]) == str(new_feat[c]) for c in old_feat)

    # motif check of 01 on the default train split and on the test split
    motif = {"default_train": motif_check(dflt["train"]["sequence"], dflt["train"]["label"]),
             "test": motif_check(dflt["test"]["sequence"], dflt["test"]["label"])}

    # old test windows in the new chr20/21 test split
    new_test_seq = defaultdict(set)
    for s, y in zip(full["test"]["sequence"], full["test"]["label"]):
        new_test_seq[s].add(y)
    new_test_names = set(full["test"]["name"])
    ot = old["test"]
    reap = {"old_test_rows": len(ot["label"]), "by_class": {}}
    same_seq = same_lab = same_name = 0
    for n, s, y in zip(ot["name"], ot["sequence"], ot["label"]):
        cls = reap["by_class"].setdefault(names[y], {"old": 0, "same_sequence_and_label": 0})
        cls["old"] += 1
        if s in new_test_seq:
            same_seq += 1
            if y in new_test_seq[s]:
                same_lab += 1
                cls["same_sequence_and_label"] += 1
        same_name += n in new_test_names
    reap.update(same_sequence=same_seq, same_sequence_and_label=same_lab, same_name=same_name,
                old_test_chroms=sorted({n.split(":")[0] for n in ot["name"]}))
    old_train_seq = set(old["train"]["sequence"])
    leak = {"new_test_sequences_also_in_new_train":
            len(set(full["test"]["sequence"]) & set(full["train"]["sequence"])),
            "new_test_sequences_in_old_train": len(set(full["test"]["sequence"]) & old_train_seq),
            "duplicate_sequences_within_train":
            len(full["train"]["sequence"]) - len(set(full["train"]["sequence"]))}

    res = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": sources, "chromosomes": man,
        "build": build_info,
        "name_convention": {"rule": "chrom:start-end|label with start = position - 301 + offset, "
                                    "end = start + 600 (0-based half-open window)",
                            "offset": offset, "old_dataset_rows_vs_genome": conv_old},
        "counts_full": counts(full), "counts_default": counts(dflt),
        "none_groups_full": groups, "none_groups_default": dgroups,
        "genes_full": genes, "genes_with_sites_full": pos_genes, "genes_default_train": dgenes,
        "split_chromosomes": split_chroms, "genome_spot_check": spot,
        "schema_same_as_splice_sites_all": schema_same,
        "motif_check": motif, "old_test_in_new_test": reap, "sequence_overlap": leak,
        "files": {p.relative_to(DATA_DIR.parent).as_posix(): p.stat().st_size
                  for d in (FULL_DIR, DEF_DIR) for p in sorted(d.glob("*.parquet"))},
    }
    write_json_atomic(OUT_JSON, res)
    return res


# ----------------------------------------------------------------------------- reports
def pct(x):
    return f"{100 * x:.1f}%"


def motif_lines(m):
    e = m["per_class_evidence"]
    return (f"donors with GT@301 or AC@298: {pct(e['2']['GT@donor_off_or_AC@donor_off'])}; "
            f"acceptors with AG@298 or CT@301: {pct(e['1']['AG@acc_off_or_CT@acc_off'])}; "
            f"none rows with either: {pct(e['0']['GT@donor_off_or_AC@donor_off'])} / "
            f"{pct(e['0']['AG@acc_off_or_CT@acc_off'])}")


def write_reports(res):
    s, b = res["sources"], res["build"]
    cf, cd = res["counts_full"], res["counts_default"]
    mo = res["motif_check"]
    gtf_hdr = [h.removeprefix("##") for h in b["gencode"]["header"]]
    lines = [
        "# data/genome: genome-wide splice-site windows (GRCh38 + MANE Select)", "",
        f"_Written by `scripts/16_genome_dataset.py` on {res['generated_utc'][:10]}. "
        "Everything in this folder is git-ignored; the script rebuilds it. Counts and checks: "
        "`results/genome_dataset.md`._", "",
        "## Source files (`source/`)", "",
        "| file | what | bytes | MD5 | published MD5 matches |", "|---|---|---:|---|---|"]
    for k, v in s.items():
        m = {True: "yes", False: "NO", None: "n/a (none published)"}[v["md5_matches_published"]]
        lines.append(f"| [`{v['file']}`]({v['url']}) | {v['what']} | {v['bytes']:,} | "
                     f"`{v['md5']}` | {m} |")
    mane_v = (SRC / SOURCES["mane_versions"][0]).read_text().strip().replace("\t", ": ")
    lines += ["", "Versions:", "",
              "- Genome: GRCh38.p14, NCBI RefSeq assembly `GCF_000001405.40`. Only the 24 "
              "assembled chromosomes (chr1-22, X, Y) are used, extracted to `grch38/` "
              "(upper case; lengths checked against the assembly report).",
              "- MANE: release 1.3 (" + "; ".join(mane_v.splitlines()) + "). "
              f"{b['mane_select_transcripts']:,} MANE Select transcripts, "
              f"{b['mane_on_primary_chromosomes']:,} on chr1-22/X/Y (the rest sit on fix/alt "
              f"patches and are not used), {b['mane_with_introns']:,} with at least one intron. "
              "MANE Plus Clinical transcripts are ignored.",
              "- GENCODE: " + "; ".join(h for h in gtf_hdr if h.startswith(("description", "date")))
              + f". {b['gencode']['transcripts']:,} transcripts, all biotypes and levels.",
              "- SHA-256 of every file: `results/genome_dataset.json` (`sources`).", "",
              "## Built files", "",
              "| path | rows | content |", "|---|---:|---|",
              f"| `splice_sites_genome_full/train.parquet` | {cf['train']['total']:,} | every "
              "window, train chromosomes (all but chr20, chr21) |",
              f"| `splice_sites_genome_full/test.parquet` | {cf['test']['total']:,} | every "
              "window, chr20 and chr21 |",
              "| `splice_sites_genome_full/{train,test}_meta.parquet` | same | row by row: kind, "
              "negative group, gene, MANE transcript, strand, motif at the site offsets |",
              f"| `splice_sites_genome/train.parquet` | {cd['train']['total']:,} | **default**: "
              "seeded, class-balanced sample of the full train split |",
              f"| `splice_sites_genome/test.parquet` | {cd['test']['total']:,} | **default**: "
              "the full test split |", "",
              "Columns: those of `splice_sites_all` (`sequence`, `name`, `label`, `task`) plus "
              "`chrom` and `position`.", "",
              "- `sequence`: 600 nt of the genome + strand, upper case A/C/G/T only. As in "
              "`splice_sites_all`, - strand sites appear reverse-complemented.",
              "- `name`: `chrom:start-end|label`, 0-based half-open window coordinates, the "
              "convention of `splice_sites_all` (checked: "
              f"{pct(res['name_convention']['old_dataset_rows_vs_genome']['match_0based_halfopen'])}"
              " of its 33,000 windows equal the genome at their name).",
              "- `label`: 0 none, 1 acceptor, 2 donor (`data/label_info.json`).",
              f"- `task`: `{TASK}`.",
              "- `chrom`: `chr1` ... `chrY`. `position`: 1-based GRCh38 coordinate of the base at "
              "offset 300; for a site that is the exon-side base (donor: last exon base, "
              "acceptor: first exon base).", "",
              "Read with `datasets.load_dataset(\"parquet\", data_files={\"train\": ..., "
              "\"test\": ...})` or `pandas.read_parquet`.", ""]
    README.write_text("\n".join(lines))

    ng, nd = res["none_groups_full"], res["none_groups_default"]
    ro = res["old_test_in_new_test"]
    md = ["# Genome-wide splice-site dataset: counts and checks", "",
          f"_Generated by `scripts/16_genome_dataset.py` on {res['generated_utc'][:16]} UTC. "
          "Files and versions: `data/genome/README.md` (git-ignored), numbers: "
          "`results/genome_dataset.json`._", "",
          "Sources: GRCh38.p14 (NCBI `GCF_000001405.40`) + MANE v1.3 Select, the pair "
          "OpenSpliceAI trains on; GENCODE v50 comprehensive only to keep negatives off real "
          "sites. Windows, labels, centring and strand convention as `splice_sites_all`, built "
          "with `13_hard_negatives.window` / `canonical_sites` / `mine_gene`. Test = chr20 + "
          "chr21, train = every other chromosome.", "",
          "## Counts", "",
          "| file | split | none | acceptor | donor | total | genes |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for lab, c, gn in (("full", cf, res["genes_full"]),
                       ("default", cd, {"train": res["genes_default_train"]["train"],
                                        "test": res["genes_full"]["test"]})):
        for sp in ("train", "test"):
            x = c[sp]
            md.append(f"| {lab} | {sp} | {x['none']:,} | {x['acceptor']:,} | {x['donor']:,} | "
                      f"{x['total']:,} | {gn[sp]:,} |")
    md += ["", "None class by group (full train / full test / default train): " + "; ".join(
        f"{g} {ng['train'][g]:,} / {ng['test'][g]:,} / {nd['train'][g]:,}" for g in ng["train"]) + ".",
        "", f"MANE: {b['mane_select_transcripts']:,} Select transcripts, "
        f"{b['mane_on_primary_chromosomes']:,} on chr1-22/X/Y, {b['mane_with_introns']:,} with "
        f"an intron. Site windows dropped for non-ACGT bases: {b['positives_skipped_non_ACGT'] or 0}; "
        f"positions that are a donor of one transcript and an acceptor of another (dropped): "
        f"{b['positions_with_conflicting_labels']}.", "",
        "## Checks", "",
        f"- Chromosome split: train chromosomes {', '.join(res['split_chromosomes']['train'])}; "
        f"test {', '.join(res['split_chromosomes']['test'])}. No train window from chr20/21.",
        f"- Every window equals the genome at its `name`, and `position` is base 300: "
        + ", ".join(f"{sp} {v['match']:,}/{v['checked']:,}" for sp, v in res["genome_spot_check"].items())
        + " sampled rows.",
        f"- Parquet schema of the shared columns identical to `splice_sites_all`: "
        f"{'yes' if res['schema_same_as_splice_sites_all'] else 'NO'}.",
        f"- 13's convention check (MANE sites whose window shows the canonical motif at the "
        f"site offset): donors {b['convention_check']['donor']['with_canonical_motif']:,}/"
        f"{b['convention_check']['donor']['sites']:,}, acceptors "
        f"{b['convention_check']['acceptor']['with_canonical_motif']:,}/"
        f"{b['convention_check']['acceptor']['sites']:,}.", "",
        "### Motif check of `01_get_data.py`", "",
        "| | splice_sites_all train | genome default train | genome test |", "|---|---:|---:|---:|"]
    old_li = load_label_info()["per_class_evidence"]
    rows_m = [("donors: GT@301 or AC@298", "2", "GT@donor_off_or_AC@donor_off"),
              ("acceptors: AG@298 or CT@301", "1", "AG@acc_off_or_CT@acc_off"),
              ("none: GT@301 or AC@298", "0", "GT@donor_off_or_AC@donor_off"),
              ("none: AG@298 or CT@301", "0", "AG@acc_off_or_CT@acc_off"),
              ("acceptors: GT@301 or AC@298", "1", "GT@donor_off_or_AC@donor_off"),
              ("donors: AG@298 or CT@301", "2", "AG@acc_off_or_CT@acc_off")]
    for t, c, k in rows_m:
        md.append(f"| {t} | {pct(old_li[c][k])} | {pct(mo['default_train']['per_class_evidence'][c][k])} | "
                  f"{pct(mo['test']['per_class_evidence'][c][k])} |")
    sm = mo["default_train"].get("strand_mix", {})
    md += ["", f"Label mapping re-inferred from the motifs alone: "
           f"{mo['default_train']['status']} {mo['default_train']['names']} (default train), "
           f"{mo['test']['status']} {mo['test']['names']} (test); motif offsets "
           f"{mo['default_train']['motif_offsets_0based']}. Strand mix (default train): " +
           ", ".join(f"{k} {pct(v)}" for k, v in sm.items()) + ". GC by class (default train): " +
           ", ".join(f"{k} {pct(v)}" for k, v in mo["default_train"]["gc_by_class"].items()) +
           ". The none rows carry motifs by design: a third of them are GT/AG decoys.", "",
           "### Old test windows in the new chr20/21 split", "",
           f"Of the {ro['old_test_rows']:,} `splice_sites_all` test windows "
           f"({', '.join(ro['old_test_chroms'])}), **{ro['same_sequence_and_label']:,}** reappear "
           f"in the new full test split with the same sequence and label "
           f"({ro['same_name']:,} with the identical `name`): " + ", ".join(
               f"{k} {v['same_sequence_and_label']:,}/{v['old']:,}" for k, v in ro["by_class"].items())
           + ".", "",
           f"Sequence overlap: {res['sequence_overlap']['new_test_sequences_also_in_new_train']:,} "
           "test sequences also occur in the train split (identical 600 nt on another "
           f"chromosome); {res['sequence_overlap']['duplicate_sequences_within_train']:,} duplicate "
           "sequences inside train.", ""]
    OUT_MD.write_text("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description="Build the genome-wide splice-site dataset.")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the parquet files")
    args = ap.parse_args()
    download()
    sources = checksums()
    man = extract_chromosomes()
    info_path = FULL_DIR / "_build_info.json"
    if args.rebuild or not (DEF_DIR / "train.parquet").exists() or not info_path.exists():
        old = load_splits()
        offset, _ = name_convention(old)
        splits, build_info = build()
        for sp, rows in splits.items():
            write_parquet(rows, FULL_DIR / f"{sp}.parquet", COLUMNS, offset)
            write_parquet(rows, FULL_DIR / f"{sp}_meta.parquet", META_COLUMNS, offset)
        write_parquet(default_train(splits["train"]), DEF_DIR / "train.parquet", COLUMNS, offset)
        write_parquet(splits["test"], DEF_DIR / "test.parquet", COLUMNS, offset)
        write_json_atomic(info_path, build_info)
        del splits
    build_info = read_json(info_path)
    res = checks(build_info, sources, man)
    write_reports(res)
    cf, cd = res["counts_full"], res["counts_default"]
    for lab, c in (("full", cf), ("default", cd)):
        for sp in ("train", "test"):
            print(f"{lab:<8}{sp:<6} " + ", ".join(f"{k} {v:,}" for k, v in c[sp].items()))
    print(f"genes: {res['genes_full']}")
    print("motif check (default train): " + motif_lines(res["motif_check"]["default_train"]))
    print("motif check (test):          " + motif_lines(res["motif_check"]["test"]))
    ro = res["old_test_in_new_test"]
    print(f"old test windows in new test: {ro['same_sequence_and_label']}/{ro['old_test_rows']}")
    print(f"wrote {README}, {OUT_MD}, {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
