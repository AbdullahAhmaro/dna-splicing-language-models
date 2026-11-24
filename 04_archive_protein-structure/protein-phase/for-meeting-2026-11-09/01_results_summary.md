# Ubiquitin — first runs

**Sequence (76 amino acids):**
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG

**Real shape:** PDB ID 1UBQ (lab-measured, X-ray, 1.8 Å). File: reference/1UBQ_real_structure.pdb

## AlphaFold 2 (ColabFold v1.6.3, T4 GPU)

| Rank | Model | Avg. pLDDT | pTM |
|---|---|---|---|
| 1 | model 4 | 95.9 | 0.841 |
| 2 | model 3 | 95.5 | 0.835 |
| 3 | model 2 | 95.3 | 0.832 |
| 4 | model 1 | 95.2 | 0.822 |
| 5 | model 5 | 94.0 | 0.806 |

- Run time: ~2 min for all 5 models (first model 42 s incl. one-time setup, then ~18 s each).
- Found ~20,000 related sequences (see alphafold/alphafold_similar_proteins_coverage.png).
- Low confidence only at the tail (blocks ~72–76, down to ~53) — the tail is floppy in real life
  (see alphafold/alphafold_confidence_plddt.png).
- Best file: alphafold/alphafold_best_model.pdb (rank 1 = model 4)
- Ranks 2–5: alphafold/other_models/alphafold_model_rank2.pdb … rank5.pdb
- Raw ColabFold output (scores, MSA, log): alphafold/raw/

## ESMFold (Hugging Face facebook/esmfold_v1, T4 GPU)

- Avg. pLDDT: 90.5 (file stores 0.905 — same scale x100)
- Low confidence: blocks 74–76 (60, 51, 45) — same tail.
- Run time: 2.6 s (prediction only, from the "Took ... s" line; model loading not included)
- File: esmfold/esmfold_model.pdb

## Comparison (RMSD on CA atoms, after overlapping)

| Compared | RMSD |
|---|---|
| AlphaFold best model vs ESMFold, all 76 blocks | 0.58 Å |
| AlphaFold best model vs ESMFold, blocks 1–70 (no tail) | 0.17 Å |
| AlphaFold vs real (1UBQ), all 76 blocks | 0.80 Å |
| AlphaFold vs real (1UBQ), blocks 1–70 (no tail) | 0.37 Å |
| ESMFold vs real (1UBQ), all 76 blocks | 0.84 Å |
| ESMFold vs real (1UBQ), blocks 1–70 (no tail) | 0.40 Å |

Computed with notebooks/compare_rmsd.py (chain A, ATOM records only; each row is
overlapped on exactly the blocks it measures).

Takeaway: two very different methods reach practically the same shape; they only
differ on the floppy tail. ESMFold is a bit less confident (no relatives search).
Both are also very close to the real shape: under 0.4 Å without the tail, ~0.8 Å with it.

## ESM-2 (facebook/esm2_t33_650M_UR50D, CPU, under 1 min)

- Output: one feature vector per amino acid, shape 76 x 1280. File: esm2/ubiquitin_esm2_650M_embeddings.pt
- Sanity check: the vector for Ile44 is most similar to its neighbours (43, 45) and to other
  hydrophobic residues (3, 23, 70) — the vectors capture residue type and local environment.
- Note: ESMFold already runs ESM-2 internally; this was run standalone to inspect the vectors.
- Script: notebooks/esm2_embeddings_cpu.py

## First alanine scan (I44A, ESMFold, CPU)

| Measure | Wild type | I44A |
|---|---|---|
| Mean pLDDT | 90.5 | 89.9 |
| pLDDT at residue 44 | 94.0 | 93.0 |
| RMSD vs wild type, all 76 | — | 0.23 Å |
| RMSD vs wild type, 1–70 | — | 0.12 Å |

- Largest movement is at the flexible C-terminal tail (residue 76: 1.3 Å), not at residue 44 (0.32 Å)
  (CA atoms, after overlapping on residues 1–70).
- Interpretation: the fold is essentially unchanged, which matches experiment — I44A ubiquitin still
  folds. But I44 is part of the hydrophobic patch that binding partners grip, and I44A is known to
  abolish that binding. So backbone RMSD alone does not detect a functionally critical mutation.
  Open question for the project: what signal should we use instead (side-chain atoms, ESM-2
  embedding shift, folding with a binding partner, NMR restraints)?
- Files: esmfold/alanine_scan/esmfold_wildtype.pdb and esmfold/alanine_scan/esmfold_I44A.pdb.
  esmfold_wildtype.pdb is byte-identical to esmfold/esmfold_model.pdb (checked with SHA-256).
- Script: notebooks/alanine_swap_cpu.py

## Still to do
- [x] Compare both against 1UBQ
- [x] Run ESM-2 on its own (feature vectors, 76 x 1280)
- [x] Optional: first alanine swap (I44A) with ESMFold
- [ ] Run the full alanine scan over all 76 positions (28 and 46 are already alanine)
- [ ] Decide what signal detects a functional change (side-chain atoms, ESM-2 embedding shift,
      folding with a binding partner, NMR restraints)
