# Superseded pipeline outputs (kept, not used)

`results_before_rerun_2026-11-22/` holds the versions of seven `results/` files as they
were before `scripts/run_all.sh` was rerun on 2026-11-22 18:34. That rerun checked that
the pipeline still worked after the folder was renamed from `dna-phase/`. It rewrote these
files in place, and `run_all.sh` always overwrites `phase1_results.prev.json`.

Checked on 2026-11-22: **no result changed.** Every accuracy, F1, top-k, AUROC and
false-positive count is identical, and all figures came out byte-identical. The only
differences are the "Generated on" timestamps, the step timings, and `train_seconds`/
`finished_utc` fields in the JSON. The old `phase1_results.prev.json` (from the
2026-11-20 02:46 run) also differed from its successor only in those timing fields.

They are kept only because the project rule is never to delete anything. `results/run.log`
is append-only, so its old content is still in the live file.
