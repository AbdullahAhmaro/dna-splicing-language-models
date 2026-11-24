> **Note added 2026-11-22:** these are the week-1 setup notes (written 2026-11-15). This folder
> was called `dna-phase/` until 2026-11-22. The current overview, status and the one-command
> rerun are in `../README.md`; the results are in `results/` (start with `results/phase1_summary.md`).

# DNA phase — start here

The project moved from peptides to **DNA**. The advisor confirmed Josh and Tala are
on the DNA project while other students do peptides.

**The task:** given a DNA sequence, predict **where it should be cut** — the
boundaries between introns (cut out) and exons (kept). This is called *splicing*.
Many diseases happen because a mutation makes the body cut in the wrong place.

**The architecture** is the same one from meeting 1: take a pre-trained DNA
language model, **freeze it**, and train a small classifier on top of its feature
vectors.

## What the advisor asked for this week

1. Pick a small DNA language model on Hugging Face (he said ~1B parameters or
   less, and named DNABERT-2 as an example).
2. **Get it running locally first** — on the laptop, no GPU, no Colab credits.
3. Email Effi from both college email addresses for GPU cluster access.

The splicing task itself comes after, once a model runs.

## Do this, in order

### 1. Build the environment (once, ~10 minutes)

    bash scripts/setup_local_env.sh

This creates `dna-venv/` in this folder and installs CPU-only PyTorch plus a
pinned `transformers`. It does not touch the system Python.

### 2. Try the models

    source dna-venv/bin/activate
    python scripts/try_dna_models.py

It tries four small DNA models in order, prints which ones loaded and how long
each took, saves the embeddings of the first one that worked into `results/`, and
writes `results/model_report.json`.

Expect some of them to fail — that is the point of trying several. If all four
fail, send `results/model_report.json` to Claude.

To try just one:

    python scripts/try_dna_models.py --only nt50

### 3. Email Effi

The text is in `email_to_effi.md`. Fill in both college addresses and send.

### 4. Optional, for next week: look at the splice data

    python scripts/get_splice_data.py

Downloads the splice-site dataset and prints what is inside: ~30,000 sequences,
each labelled donor site / acceptor site / not a splice site.

## The candidate models

| Key | Model | Size | Why it is on the list |
|---|---|---|---|
| `nt50` | InstaDeepAI/nucleotide-transformer-v2-50m-multi-species | 50M | Smallest. Plain PyTorch, most likely to just run. |
| `nt100` | InstaDeepAI/nucleotide-transformer-v2-100m-multi-species | 100M | Same family, one size up (~400 MB). |
| `dnabert1` | zhihan1996/DNA_bert_6 | ~89M | The original DNABERT. Input must be split into 6-mers. |
| `dnabert2` | zhihan1996/DNABERT-2-117M | 117M | The one the advisor named. Needs `triton` on some setups, which is why it is last. |

A note on sizes: the advisor suggested ~1B parameters. These are all 50–120M,
which is 10–20× smaller. That is good news, not a problem — DNA models are small
compared to text models, and these are the standard ones in the splicing
literature. A 1B DNA model (Evo) exists but would be slow on a CPU. Worth
mentioning at the next meeting.

## What is still open

- [ ] Which model to settle on (decide after the report)
- [ ] Which exact framing of the task: classify a short window as donor /
      acceptor / neither, or predict a cut position inside a long sequence.
      The advisor described the second ("output a number 0 to 8000"); the ready
      -made datasets use the first. Worth asking him.
- [ ] Proposal, due the 18th (confirm the date in the JCE system)
