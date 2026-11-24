# Logbook: advisor meetings

Final project of Abdullah Ahmaro and Tala Harbawi, advisor Dr. Yehuda Hassin (JCE).
One entry per meeting, newest first, at most three lines: what we presented, what the
advisor said, the tasks for next week. Add each new meeting at the top.

---

**23 Nov 2026** — Presented a frozen NT-v2-50M with a small MLP classifier for 3-class splice-site prediction (donor / acceptor / none): 94% accuracy reading the centre tokens vs 54% averaging the whole window; DNABERT-2 reached 83%; a first fine-tuning test reached 96.7%.
Advisor: report F1 and a confusion matrix for every result, keep the stronger model, and understand the code ourselves; ~8k examples is far too few, especially for fine-tuning.
Tasks: find or build a splicing dataset of 70k–100k+ examples; try other MLP sizes and depths; run on the college GPU server once access is granted; open the repo and logbook.

**16 Nov 2026** — Showed four DNA language models (NT-v2 50M and 100M, DNABERT-1, DNABERT-2) running locally on CPU, with their embedding outputs and how each tokenises the same sequence.
Advisor defined the task: splice-site prediction — given a DNA sequence, find where the cuts are — as a classifier on top of a frozen DNA model.
Tasks: find a splicing dataset, build and train the classifier on top of the model, and write the project proposal.

**9 Nov 2026 (date approximate)** — Presented our AlphaFold 2 and ESMFold runs on ubiquitin, done on Google Colab, where the GPU credits ran out within about 15 minutes.
Advisor: the project is confirmed as DNA, not peptides. Take a small DNA language model that runs on our own PC (about 1B parameters or less, e.g. DNABERT-2) and train a classifier on top of it for splicing: given a DNA sequence, say where it is cut.
Tasks: find a small DNA language model and run it locally, without Colab or a GPU; email Effi from our college addresses for GPU-cluster access; then add the splicing task on top of the chosen model.

**2 Nov 2026 (date approximate)** — Presented what we had read in the first week about amino acids, peptide bonds and how a protein chain folds.
Advisor: introduced AlphaFold 2, ESM-2 and ESMFold (a language model gives feature vectors, a small network on top predicts the fold) and NMR data; the goal at this stage was alanine scanning on a computer instead of in the lab.
Tasks: pick a real protein and run AlphaFold 2 on it; run ESM-2 and ESMFold and compare with AlphaFold; read about NMR; email our names and addresses for GPU-cluster access.

**Late Oct 2026 (date approximate)** — First meeting, nothing to present yet.
Advisor: explained Transformers and language models, DNA as a four-letter language, and the design we still use: take a pre-trained model, freeze it, and train only a small network on its feature vectors. Weekly meeting set for Mondays at 11:30.
Tasks: spend the first two weeks learning the basics of DNA, amino acids, proteins and DNA language models, with ChatGPT or Claude as a tutor and self-tests; the proposal is due on the 18th.

---

## Notes on the dates

There have been five meetings. Only meeting 5 (23 Nov 2026) has a dated recording. The
entry for meeting 4 (16 Nov 2026) was written by us from memory; the folders hold no notes
or recording of that meeting. The three earlier entries were reconstructed from the meeting transcripts (kept on the laptop, not in this
repo), and no file states their dates, so they are marked approximate. What the files do
show:

- **Meeting 3 (about 9 Nov):** `04_archive_protein-structure/protein-phase/for-meeting-2026-11-09/`
  was prepared for Monday 9 Nov, the transcript reports that material, and it was saved on 15 Nov.
- **Meeting 2 (about 2 Nov):** it set the AlphaFold task, which was run on 6 Nov (ColabFold
  log), and its transcript was saved on 8 Nov. It ended with "next week, Monday at 11:30".
- **Meeting 1 (late Oct):** the DNA cheat sheet written for its study task is dated 29 Oct,
  so the meeting was on or before that day.
