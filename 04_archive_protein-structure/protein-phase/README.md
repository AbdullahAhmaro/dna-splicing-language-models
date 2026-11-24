# Final Project — Josh & Tala (advisor: Yehuda Hassin)

Goal: do "alanine scanning" on a computer instead of in a lab — swap one amino acid
to alanine, let an AI (AlphaFold 2 / ESMFold) predict the new shape, and compare it
with the original. Later this may also use NMR data.

Weekly meeting: Mondays 11:30.

## Folders
- for-meeting-2026-11-09/ — copies of what we present on Mon 9 Nov, numbered in presenting order
- meetings/   — transcripts of advisor meetings
- proteins/   — one folder per protein (and later, per alanine swap)
    - ubiquitin/
        - alphafold/  — AlphaFold 2 (ColabFold): alphafold_best_model.pdb + 3 plots
            - other_models/ — the 4 other AlphaFold models (rank 2–5)
            - raw/          — untouched ColabFold output (scores, alignment, log, citations)
        - esmfold/    — ESMFold model (esmfold_model.pdb)
            - alanine_scan/ — ESMFold models for alanine swaps (wild type + I44A so far)
        - esm2/       — ESM-2 feature vectors (ubiquitin_esm2_650M_embeddings.pt, 76 x 1280)
        - reference/  — the real lab-measured shape, 1UBQ (1UBQ_real_structure.pdb)
        - results.md  — all numbers and what they mean
- notebooks/  — links + the code we ran in Colab (GPU and *_cpu.py versions), and compare_rmsd.py (runs locally)
- proposal/   — the project proposal (due the 18th — confirm date)
- study-notes/ — study notes (PDF); old_versions/ keeps earlier versions

FILE_GUIDE.md — what each file is and whether you need it.
RENAMES.md — old file names → new names.

Study guide (cheat sheet): https://claude.ai/artifact/Fq4wkb8faGUBb4ZLzjxs2g
