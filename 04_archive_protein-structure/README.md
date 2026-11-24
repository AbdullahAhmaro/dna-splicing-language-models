# Archive: protein-structure phase (not live work)

The project started on proteins: predict the 3D shape of ubiquitin with AlphaFold 2
and ESMFold, compare both with the real structure (PDB 1UBQ), and try a first
computer "alanine scan" (the I44A mutant). The advisor then reassigned us from the
peptide/protein project to the DNA project (splice-site prediction), while other
students continue on peptides. **None of this is live work.** It is kept as evidence
of what we did, so nothing here should be deleted.

## When the reassignment happened

No file states the date outright. What the files do show:

- The last protein-phase work is dated **2026-11-09**: `protein-phase/for-meeting-2026-11-09/`
  was prepared for the meeting on Monday 9 Nov, and `protein-phase/RENAMES.md` records
  changes on 2026-11-08 and 2026-11-09.
- Meeting 3 is "the meeting where the project was confirmed as DNA (not peptides)"
  (`../02_dna-splicing-pipeline/meetings/meeting-03_dna-splicing-task.md`). In it the
  students report the AlphaFold/ESMFold Colab runs, which is the 9 Nov material.
- The DNA-phase start-here notes, written by **2026-11-15**, already say "the advisor
  confirmed Josh and Tala are on the DNA project while other students do peptides".

So the reassignment happened between 9 and 15 Nov 2026. Most likely it was at the
Monday 9 Nov meeting (meeting 3), but no file says so in words.

## What is in here

`protein-phase/` is the old project folder (it was `FinalProject/FinalProject/`),
moved here on 2026-11-22 with its inner structure unchanged, so its own guides still
work. Start with:

- `protein-phase/README.md`: the original project description (protein phase)
- `protein-phase/FILE_GUIDE.md`: what every file is
- `protein-phase/RENAMES.md`: the renames done on 2026-11-08/09
- `protein-phase/proteins/ubiquitin/results.md`: all the numbers (pLDDT, RMSD, I44A)

| Folder | Contents |
|---|---|
| `proteins/ubiquitin/alphafold/` | AlphaFold 2 (ColabFold) outputs: best model, ranks 2–5, plots, raw ColabFold output |
| `proteins/ubiquitin/esmfold/` | ESMFold model; `alanine_scan/` holds the wild-type and I44A PDBs |
| `proteins/ubiquitin/esm2/` | ESM-2 embedding file `ubiquitin_esm2_650M_embeddings.pt` (76 × 1280) |
| `proteins/ubiquitin/reference/` | the real structure, 1UBQ |
| `notebooks/` | the Colab/CPU scripts that produced the above |
| `study-notes/` | the protein study-notes PDFs (current and `old_versions/`) |
| `for-meeting-2026-11-09/` | the old meeting folder, unchanged |
| `meetings/meeting-02_alphafold-esm-peptides.md` | the meeting that set the protein tasks |

## Moved out of this folder on 2026-11-22 (still exist, elsewhere)

The old `FILE_GUIDE.md` and `README.md` still list these at their old paths:

- `proposal/README.md` is now `../01_proposal/README.md`
- `meetings/meeting-01_intro-dna-language-models.md` is now
  `../02_dna-splicing-pipeline/meetings/`. Meeting 1 introduced DNA language models and
  the frozen-model-plus-small-head design the DNA pipeline uses, so it belongs with the
  live work.

Note: `for-meeting-2026-11-09/DNA Language Models — Cheat Sheet.pdf` (dated 29 Oct) is
DNA study material, not protein material. It was left in place because the whole
2026-11-09 meeting folder was archived as one unit.
