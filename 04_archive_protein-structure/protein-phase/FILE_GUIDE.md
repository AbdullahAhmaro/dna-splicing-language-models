# File guide

What each important file is, and whether you need it. Old file names: see RENAMES.md.

**Need it?** — **Yes** = you use it now · **Next** = for an upcoming step · **Keep** = don't open it day to day, but don't delete it

## Project root

| File | What it is | Need it? |
|---|---|---|
| README.md | Project goal, meeting time, folder list | Yes |
| FILE_GUIDE.md | This file | Yes |
| RENAMES.md | Old file name → new file name, for everything moved or renamed on 2026-11-08 | Keep (when you meet an old name) |
| for-meeting-2026-11-09/ | Copies of what you present on Monday 9 Nov, numbered in presenting order (01–09) | Yes, for the meeting |
| study-notes/Study_Notes_2_Proteins_AlphaFold_ESMFold.pdf | 8-page study notes: proteins, AlphaFold, ESMFold, how to read the results (incl. the 1UBQ RMSDs) | Yes, for learning |
| study-notes/old_versions/Study_Notes_2_Proteins_AlphaFold_ESMFold_v1.pdf | Earlier 7-page version of the study notes, before the 1UBQ comparison | Keep |

## meetings/

| File | What it is | Need it? |
|---|---|---|
| meeting-01_intro-dna-language-models.md | Meeting 1 transcript (translated): language models, DNA, schedule | Keep |
| meeting-02_alphafold-esm-peptides.md | Meeting 2 transcript: the AlphaFold / ESM tasks and the alanine-scanning goal | Yes, it defines the current tasks |

## proposal/

| File | What it is | Need it? |
|---|---|---|
| README.md | Placeholder; proposal due the 18th (date to confirm) | Next |

## notebooks/

| File | What it is | Need it? |
|---|---|---|
| notebooks.md | Links to the Colab notebooks and how to run each script | Yes |
| esmfold_hf.py | ESMFold in Colab on GPU (Hugging Face); made esmfold/esmfold_model.pdb | Yes, to re-run or fold variants |
| compare_rmsd.py | RMSD after overlapping (CA atoms) between AlphaFold, ESMFold and 1UBQ; runs locally, numpy only | Yes |
| esm2_embeddings_cpu.py | ESM-2 feature vectors (76 × 1280) on CPU; made esm2/ubiquitin_esm2_650M_embeddings.pt | Yes |
| alanine_swap_cpu.py | ESMFold on CPU: folds normal + I44A, prints pLDDT and RMSD; made esmfold/alanine_scan/ | Yes, the base for the full alanine scan |
| esm2_embeddings.py | GPU version of esm2_embeddings_cpu.py | Keep (when a GPU is free) |
| alanine_swap.py | GPU version of the I44A swap, run after esmfold_hf.py | Keep (when a GPU is free) |

## proteins/ubiquitin/

| File | What it is | Need it? |
|---|---|---|
| results.md | All ubiquitin numbers (pLDDT, pTM, run times, RMSD, ESM-2, I44A) and what they mean | Yes, the main summary |
| alphafold/alphafold_best_model.pdb | AlphaFold 2's best 3D model (rank 1 = model 4, pLDDT 95.9) | Yes |
| alphafold/alphafold_confidence_plddt.png | AlphaFold confidence (pLDDT) per amino acid, all 5 models | Yes |
| alphafold/alphafold_similar_proteins_coverage.png | How many related sequences AlphaFold found, per position (~20,000) | Yes |
| alphafold/alphafold_pae_advanced.png | Predicted aligned error between every pair of positions (advanced) | Keep (only if asked) |
| alphafold/other_models/alphafold_model_rank2.pdb … rank5.pdb | AlphaFold's 4 other models (rank 2 = model 3, 3 = model 2, 4 = model 1, 5 = model 5) | Keep |
| alphafold/raw/log.txt | ColabFold run log: time and pLDDT/pTM per model | Keep |
| alphafold/raw/cite.bibtex | Citations for ColabFold / AlphaFold and the databases it used | Next (for the report) |
| alphafold/raw/*_scores_rank_00N_*.json | Per-amino-acid pLDDT and PAE numbers for each model | Keep |
| alphafold/raw/ (everything else) | Untouched ColabFold output: alignment (.a3m), MSA search files (_env/), settings (config.json), input (.csv), done marker | Keep |
| esmfold/esmfold_model.pdb | ESMFold's 3D model (pLDDT 90.5, stored as 0.905) | Yes |
| esmfold/alanine_scan/esmfold_I44A.pdb | ESMFold model of the I44A mutant (Ile44 → Ala; pLDDT 89.9) | Yes |
| esmfold/alanine_scan/esmfold_wildtype.pdb | ESMFold model of normal ubiquitin from the alanine-scan run; byte-identical to esmfold/esmfold_model.pdb | Yes, the baseline for every alanine swap |
| reference/1UBQ_real_structure.pdb | The real, lab-measured ubiquitin structure (PDB 1UBQ, X-ray, 1.8 Å) | Yes |
| reference/README.md | Where the 1UBQ file came from | Keep |
| esm2/ubiquitin_esm2_650M_embeddings.pt | ESM-2 feature vectors, shape 76 × 1280 (PyTorch tensor; open with torch.load) | Yes |
| esm2/README.md | What the .pt file is and how it was made | Keep |
