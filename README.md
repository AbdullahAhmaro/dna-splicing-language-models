# Final project: predicting splice sites with a DNA language model

JCE Software Engineering final project (תשפ"ו). We take a pre-trained DNA language model,
**freeze it**, and train only a small MLP head on its feature vectors to say whether a
position in human DNA is a splice **donor** (exon→intron), an **acceptor** (intron→exon),
or neither.

- **Students:** Abdullah Ahmaro and Tala Harbawi. Older notes call Abdullah "Josh".
- **Advisor:** Dr. Yehuda Hassin. Weekly meeting: Mondays 11:30.
  GPU-cluster access goes through Effi (the request email is drafted on the laptop, not in
  this repo).
- **Logbook:** [LOGBOOK.md](LOGBOOK.md) has one entry per advisor meeting, newest first:
  what we presented, what the advisor said, and the tasks for the next week.
- **History:** the project began on protein structure (AlphaFold/ESMFold on ubiquitin,
  alanine scanning). The advisor then moved us to DNA, between 9 and 15 Nov 2026
  (see `04_archive_protein-structure/README.md`).

## Current state (phase 1 done, last full run 2026-11-22)

Numbers are from `02_dna-splicing-pipeline/results/`.

- **Window benchmark** (InstaDeepAI `splice_sites_all`, 8,000 train / 2,000 test windows,
  test windows only from chr20/21): NT-v2-50M with a centre + previous-token head reaches
  **0.941** accuracy, against **0.888** for a strand-aware GT/AG motif rule. DNABERT-2-117M
  reaches 0.827. Averaging the whole window instead gives only 0.535 / 0.549.
- **Whole genes** (15 genes on chr20/21, 127 donors + 127 acceptors): NT-v2 *ranks*
  sites well (donor AUROC **0.994**) but *picks* them badly (donor top-k **0.39**,
  acceptor 0.19).
- **Hard negatives** (exon/intron interiors and GT/AG decoys from training chromosomes):
  acceptor top-k rose by +0.08 [+0.03, +0.13], and exonic false positives fell by a
  quarter to a third.
- NT 2.5B on the genome dataset (120k train): macro-F1 0.979 vs 0.956 for NT-v2 50M,
  +0.022 [+0.019, +0.026] (genes resampled). Beats NT-v2 50M trained on all 529k windows (0.966).
- **Open:** which phase-2 direction to take (fine-tuning, longer context, per-position
  labels; see `03_for-meeting-2026-11-23/00_results_summary.pdf`). Also open: GPU-cluster
  access, and the proposal due date (the 8 Nov placeholder says "the 18th, confirm").

## Folders

| Folder | What is in it |
|---|---|
| `01_proposal/` | The current Hebrew proposal `.docx` (saved 17 Nov) and the department's blank template |
| `02_dna-splicing-pipeline/` | The live work: `scripts/` (numbered steps 01–14 plus `run_all.sh`), `kaggle/` (notebooks run on a Kaggle GPU: the two fine-tuning ones and `compare_models.ipynb`, which compares frozen models of different sizes), `results/` (reports, figures, JSON, logs), and `requirements-frozen.txt` (exact package versions). On the laptop it also holds `data/` (dataset and embedding caches, about 2 GB), `dna-venv/`, and the transcripts of meetings 1 and 3; these are not in the repo. It was named `dna-phase/` until 2026-11-22, and older logs still use that name. |
| `03_for-meeting-2026-11-23/` | Renamed copies of what to show on Mon 23 Nov, in showing order. Start with `INDEX.md`. |
| `04_archive_protein-structure/` | The finished protein phase (AlphaFold, ESMFold, alanine scan, ESM-2, study notes, the 9 Nov meeting folder). Kept as a record, not live work. |
| `05_meeting-2026-11-23_recording-transcript/` | In the repo: only `scripts/`, the two scripts that transcribed the 23 Nov meeting. The recording and the transcripts stay on the laptop. |

## How to reproduce

Build one virtual environment, `dna-venv`, then run one command:

```bash
cd 02_dna-splicing-pipeline
bash scripts/setup_local_env.sh   # builds dna-venv/ (needed once after cloning): CPU torch + transformers 4.49.0
bash scripts/run_all.sh           # the one command: reruns every step; log in results/run.log
```

- **Environment:** `02_dna-splicing-pipeline/dna-venv/` (Python 3.14.4, torch 2.15.0+cpu,
  transformers 4.49.0). It runs on a CPU and needs no GPU. `run_all.sh` activates the venv
  itself, so you don't have to source anything first. The venv is not in the repo;
  `setup_local_env.sh` builds it, and `requirements-frozen.txt` lists the exact versions
  of the working one.
- **After cloning:** `data/` is not in the repo, so the first run starts from an empty
  cache: it downloads the dataset, the two models and the Ensembl gene records, and
  recomputes every embedding (see **Time**). A run from a fresh clone has not been tried
  yet; every run so far was on the laptop that already had the caches.
- **Fine-tuned track:** `run_all.sh` covers the frozen-model pipeline only. The fine-tuned
  NT-v2 scan is made on a Kaggle GPU by `kaggle/finetune_scan_genes.ipynb` and brought in
  with `python scripts/14_import_finetuned_tracks.py <finetuned_scan_tracks.zip>` after
  `run_all.sh` has finished. Its files live under `data/scan/`, so they are not in the
  repo either; `results/finetuned_scan_manifest.json` records the run.
- **Time:** about 10 minutes on the laptop CPU (9m 35s on 2026-11-22, 8m 39s on 2026-11-20),
  with the cached embeddings in `data/emb/` and `data/scan/`.
  From an empty cache, the embedding passes alone took about an hour (8m 49s + 25m 35s
  + 7m 19s + 19m 45s, per `results/phase1_summary.md`). The NT-v2 window scan adds about
  2 h the first time (comment in `run_all.sh`). The dataset and the Ensembl gene records
  are downloaded once, the first time, and read from `data/` after that.
- **Defaults:** 8,000 train / 2,000 test windows. Use
  `MAX_TRAIN=-1 MAX_TEST=-1 bash scripts/run_all.sh` for the full dataset.
- **Outputs:** the reports (`phase1_summary.md`, `position_scan_summary.md`,
  `motif_analysis.md`, `dataset_format.md`) and figures are regenerated in place.
  `results/run_steps.tsv` has the time of each step.

## What is not in this repo

`.gitignore` keeps these on the laptop only:

- `02_dna-splicing-pipeline/dna-venv/` (1.8 GB) and `02_dna-splicing-pipeline/data/`
  (2.3 GB: dataset, embedding caches, gene scans, trained heads). Both are rebuilt by the
  two commands above.
- Model weights and saved tensors (`.pt`, `.bin`, `.safetensors`, `.ckpt`), including the
  small embedding files `results/dnabert1_embeddings.pt`, `results/nt50_embeddings.pt`
  and the archived `ubiquitin_esm2_650M_embeddings.pt`.
- The meeting recording and all meeting transcripts (`meetings/` folders and everything in
  `05_meeting-2026-11-23_recording-transcript/` except its scripts). `LOGBOOK.md` is the
  record of the meetings. Some older notes in this repo still point at those transcript
  files.
- The draft email to Effi (`02_dna-splicing-pipeline/email_to_effi.md`).
- Python caches (`__pycache__/`), and any single file over 5 MB.
