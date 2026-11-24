# Meeting Mon 2026-11-23: what to show, in order

Every file here is a **copy**. The originals stay in `../02_dna-splicing-pipeline/results/`
(or `../01_proposal/`), under their original names, which are given in brackets. Every number
below was read from those result files after the 2026-11-22 rerun. "NT-v2" is
nucleotide-transformer-v2-50M and "DNABERT-2" is DNABERT-2-117M. Both are frozen, and only
a small MLP head is trained.

| # | File | What it shows | The one point to make |
|---|---|---|---|
| 00 | `00_results_summary.pdf` (already here, made 2026-11-22) | 5-page written summary of phase 1, ending with the phase-2 plan | Handout. It ends with the question for the advisor: the datasets label a 600-nt window, but the task he described predicts a position. Should phase 2 train on per-position labels? |
| 01 | `01_dataset-format_what-the-labels-mean.md` [`dataset_format.md`] | The dataset: 30,000 train / 3,000 test windows of 600 nt, one label per window, test windows only from chr20/21 | The labels are **inferred, not documented**: 0 = none, 1 = acceptor, 2 = donor. Evidence: 98.5% of label-2 rows carry GT (or AC) at one fixed offset, and 99.8% of label-1 rows carry AG (or CT). |
| 02 | `02_report_phase1-summary_window-benchmark.md` [`phase1_summary.md`] | All window-classification results, the no-ML baseline, and how the pooling variant was chosen on validation | NT-v2 with the centre + previous-token head scores **0.941** accuracy on 2,000 test windows, against **0.888** for the strand-aware GT/AG rule. That is +5.2 points, outside the ±2.0-point noise. |
| 03 | `03_chart_pooling-variants_accuracy-vs-GT-AG-rule.png` [`phase1_pooling_variants.png`] | MLP accuracy for each way of pooling the token vectors, per model; the dashed line is the 0.888 rule | Pooling matters more than the model. Averaging the whole window gives **0.535** (NT-v2) and 0.549 (DNABERT-2), below the rule. Reading only the centre tokens gives 0.941 and 0.827. |
| 04 | `04_report_motif-analysis_beyond-the-GT-AG-rule.md` [`motif_analysis.md`] | What the model gets right that the motif rule cannot: decoys, non-canonical sites, a composition-only control | 184 test windows carry GT/AG at exactly the right offset but are not splice sites. The rule gets **0/184**, NT-v2 rejects **131/184** and DNABERT-2 rejects 150/184. A base-composition-only gate rejects just 4.3% at NT-v2's recall. |
| 05 | `05_report_position-scan_whole-gene-results.md` [`position_scan_summary.md`] | The window head applied at every position of 15 genes on chr20/21 (127 donors, 127 acceptors), with gene-bootstrap 95% intervals, hard negatives, and false-positive tables | True sites are **1 in 701** positions. NT-v2 donor AUROC is **0.994**, but top-k is only **0.39** (50/127). It ranks sites well and fails to pick them out. |
| 06 | `06_chart_whole-gene-tracks_topk-and-AUROC-95CI.png` [`scan_tracks_ci.png`] | Pooled top-k and AUROC with 95% intervals for every scan track | NT-v2 beats DNABERT-2 on all four metrics, with intervals that exclude zero. Donor top-k: +0.21 [+0.10, +0.33]. |
| 07 | `07_gene-scan_MMP9_scores-along-gene.png` [`scan_MMP9.png`] | MMP9 (chr20, + strand, 13 exons, 12 donors / 12 acceptors): donor and acceptor scores at every base, for the four headline tracks | Every one of the 12 true donors scores ≥ 0.86 on the NT-v2 window head, yet top-k is only **2/12**, because the true sites are buried in false positives. With the hard-negative head this rises to **4/12** donors and **7/12** acceptors (from 4/12), the largest per-gene acceptor gain on that track. ⚠ See note 1. |
| 08 | `08_gene-scan_PCNA_scores-along-gene.png` [`scan_PCNA.png`] | PCNA (chr20, − strand, 6 exons, 5 donors / 5 acceptors), same four tracks; the top strip also draws the exons of PCNA's other transcripts | Donor top-k is **3/5** on all four tracks. The other transcripts add an exon at 0–1.1 kb, upstream of the canonical transcript and outside the scored span. The DNABERT-2 long-context track scores that region high. ⚠ See note 2. |
| 09 | `09_chart_hard-negatives_false-positives-exon-vs-intron.png` [`scan_hardneg_fp.png`] | False positives at matched recall, current vs hard-negative head, split into exons and introns | NT-v2 donors at the 0.9 operating point: exonic false positives **1,005 → 655**, but total only **3,773 → 3,153**. 2,768 of the 3,773 were intronic from the start, so retraining on exon negatives cannot fix most of them. |
| 10 | `10_proposal_hebrew_splice-sites_2026-11-17.docx` [`01_proposal/הצעת_פרויקט_שחבור_DNA.docx`] | The current Hebrew proposal (saved 2026-11-17) | The proposal set its own test: if the model does not clearly beat a GT/AG-only baseline, the language model adds nothing. Phase 1 passes it, 0.941 vs 0.888 (row 02). |

## Notes: check these before you say them

1. **MMP9 is not the best donor result in the current files.** On the headline track
   in this figure (NT-v2, 600-nt windows) it gets 2/12 donors, joint last of the 15 genes
   (with MTG2). It *is* the best donor gene on the older reference track, NT-v2
   centre-token head with a long-context scan: **9/12**, the most donors found in any gene,
   and tied best by fraction with YTHDF1 (3/4). That track is in
   `05_report_position-scan_whole-gene-results.md` but is **not drawn in this figure**.
   On screen, the panel title will say "donors 2/12".
2. **The PCNA "extra exon" point is visual only.** No result file measures it. The
   upstream exon lies outside the canonical transcript, so the figure greys that region out
   and no metric counts it ("outside the canonical transcript: not used in the metrics").
   The NT-v2 window tracks do not score it at all, and the DNABERT-2 scores there are high
   across a broad stretch, not one sharp peak at an exon edge. The files say PCNA has 16
   annotated transcripts. Treat this as an observation to raise, not a result.
