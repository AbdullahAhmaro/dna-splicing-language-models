# Renames (2026-11-08)

Every file that was moved or renamed, old name → new name. Nothing was deleted.
Paths are from the project folder. `U/` = `proteins/ubiquitin/`, `AF/` = `proteins/ubiquitin/alphafold/`.

## Renamed

| Old name | New name |
|---|---|
| AF/ubiquitin_7686d_unrelaxed_rank_001_alphafold2_ptm_model_4_seed_000.pdb | AF/alphafold_best_model.pdb |
| AF/ubiquitin_7686d_unrelaxed_rank_002_alphafold2_ptm_model_3_seed_000.pdb | AF/other_models/alphafold_model_rank2.pdb |
| AF/ubiquitin_7686d_unrelaxed_rank_003_alphafold2_ptm_model_2_seed_000.pdb | AF/other_models/alphafold_model_rank3.pdb |
| AF/ubiquitin_7686d_unrelaxed_rank_004_alphafold2_ptm_model_1_seed_000.pdb | AF/other_models/alphafold_model_rank4.pdb |
| AF/ubiquitin_7686d_unrelaxed_rank_005_alphafold2_ptm_model_5_seed_000.pdb | AF/other_models/alphafold_model_rank5.pdb |
| AF/ubiquitin_7686d_plddt.png | AF/alphafold_confidence_plddt.png |
| AF/ubiquitin_7686d_coverage.png | AF/alphafold_similar_proteins_coverage.png |
| AF/ubiquitin_7686d_pae.png | AF/alphafold_pae_advanced.png |
| U/esmfold/ubiquitin_esmfold.pdb | U/esmfold/esmfold_model.pdb |

## Moved, name unchanged

| Old name | New name |
|---|---|
| AF/ubiquitin_7686d_predicted_aligned_error_v1.json | AF/raw/ubiquitin_7686d_predicted_aligned_error_v1.json |
| AF/ubiquitin_7686d_scores_rank_001_alphafold2_ptm_model_4_seed_000.json | AF/raw/ubiquitin_7686d_scores_rank_001_alphafold2_ptm_model_4_seed_000.json |
| AF/ubiquitin_7686d_scores_rank_002_alphafold2_ptm_model_3_seed_000.json | AF/raw/ubiquitin_7686d_scores_rank_002_alphafold2_ptm_model_3_seed_000.json |
| AF/ubiquitin_7686d_scores_rank_003_alphafold2_ptm_model_2_seed_000.json | AF/raw/ubiquitin_7686d_scores_rank_003_alphafold2_ptm_model_2_seed_000.json |
| AF/ubiquitin_7686d_scores_rank_004_alphafold2_ptm_model_1_seed_000.json | AF/raw/ubiquitin_7686d_scores_rank_004_alphafold2_ptm_model_1_seed_000.json |
| AF/ubiquitin_7686d_scores_rank_005_alphafold2_ptm_model_5_seed_000.json | AF/raw/ubiquitin_7686d_scores_rank_005_alphafold2_ptm_model_5_seed_000.json |
| AF/ubiquitin_7686d.a3m | AF/raw/ubiquitin_7686d.a3m |
| AF/ubiquitin_7686d_env/ (folder, 5 files) | AF/raw/ubiquitin_7686d_env/ |
| AF/log.txt | AF/raw/log.txt |
| AF/config.json | AF/raw/config.json |
| AF/ubiquitin_7686d.csv | AF/raw/ubiquitin_7686d.csv |
| AF/ubiquitin_7686d.done.txt | AF/raw/ubiquitin_7686d.done.txt |
| AF/cite.bibtex | AF/raw/cite.bibtex |
| Study_Notes_2_Proteins_AlphaFold_ESMFold.pdf | study-notes/Study_Notes_2_Proteins_AlphaFold_ESMFold.pdf |

## Copied for the meeting (originals stay where they are)

| Original | Copy |
|---|---|
| U/results.md | for-meeting-2026-11-09/01_results_summary.md |
| AF/alphafold_confidence_plddt.png | for-meeting-2026-11-09/02_alphafold_confidence_plddt.png |
| AF/alphafold_similar_proteins_coverage.png | for-meeting-2026-11-09/03_alphafold_similar_proteins_coverage.png |
| AF/alphafold_best_model.pdb | for-meeting-2026-11-09/04_alphafold_best_model.pdb |
| U/esmfold/esmfold_model.pdb | for-meeting-2026-11-09/05_esmfold_model.pdb |
| U/reference/1UBQ_real_structure.pdb | for-meeting-2026-11-09/06_real_structure_1UBQ.pdb |
| study-notes/Study_Notes_2_Proteins_AlphaFold_ESMFold.pdf | for-meeting-2026-11-09/07_study_notes.pdf |
| U/esmfold/alanine_scan/esmfold_I44A.pdb | for-meeting-2026-11-09/08_esmfold_I44A.pdb |
| U/esmfold/alanine_scan/esmfold_wildtype.pdb | for-meeting-2026-11-09/09_esmfold_wildtype.pdb |

01_results_summary.md was refreshed from U/results.md on 2026-11-09 (adds the ESM-2 and I44A sections).

## Study notes replaced with a new version (2026-11-09)

| Old name | New name |
|---|---|
| study-notes/Study_Notes_2_Proteins_AlphaFold_ESMFold.pdf (7 pages, v1) | study-notes/old_versions/Study_Notes_2_Proteins_AlphaFold_ESMFold_v1.pdf |
| Study_Notes_2_Proteins_AlphaFold_ESMFold.pdf (new 8-page version, saved in the project root) | study-notes/Study_Notes_2_Proteins_AlphaFold_ESMFold.pdf |

for-meeting-2026-11-09/07_study_notes.pdf was overwritten with the new version; the old
content is the v1 file above.

## ESM-2 and I44A runs, moved in from the project root (2026-11-09)

| Old name | New name |
|---|---|
| ubiquitin_esm2_650M.pt | U/esm2/ubiquitin_esm2_650M_embeddings.pt |
| ubiquitin_normal_esmfold.pdb | U/esmfold/alanine_scan/esmfold_wildtype.pdb |
| ubiquitin_I44A_esmfold.pdb | U/esmfold/alanine_scan/esmfold_I44A.pdb |

U/esmfold/alanine_scan/esmfold_wildtype.pdb is byte-identical to U/esmfold/esmfold_model.pdb; both are kept.

## New files

| File | Where it came from |
|---|---|
| U/reference/1UBQ_real_structure.pdb | https://files.rcsb.org/download/1UBQ.pdb (saved unchanged; RCSB calls it 1UBQ.pdb) |
| notebooks/compare_rmsd.py | New RMSD script |
| notebooks/esm2_embeddings_cpu.py | CPU version of esm2_embeddings.py (2026-11-09) |
| notebooks/alanine_swap_cpu.py | CPU version of esmfold_hf.py + alanine_swap.py in one script (2026-11-09) |
| FILE_GUIDE.md, RENAMES.md | New guides |

Note: Colab still saves new runs under the old-style names (ColabFold: `<jobname>_unrelaxed_rank_001_…pdb`,
esmfold_hf.py: `ubiquitin_esmfold.pdb`, alanine_swap_cpu.py: `ubiquitin_normal_esmfold.pdb` and
`ubiquitin_I44A_esmfold.pdb`, esm2_embeddings_cpu.py: `ubiquitin_esm2_650M.pt`). Rename them the same way
when you add them here.
