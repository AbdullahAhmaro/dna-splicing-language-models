# Notebooks

- AlphaFold 2 (ColabFold): https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb
  Paste the sequence, set jobname, Runtime -> Run all. Runtime type: T4 GPU.
- ESMFold: the ColabFold ESMFold notebook currently fails on Colab (Python 3.13, "No module named openfold").
  Use esmfold_hf.py instead, in a new Colab notebook with T4 GPU (+ High-RAM if available).
  Our ESMFold notebook is saved in Josh's Google Drive (Untitled0.ipynb) — add its link here.
- ESM-2 feature vectors: esm2_embeddings.py
- First alanine swap (optional): alanine_swap.py — run after esmfold_hf.py in the same notebook.
- No GPU? Colab often has no free GPU. Use the CPU versions instead: esm2_embeddings_cpu.py and
  alanine_swap_cpu.py (no .cuda(), no device_map, no .half()). Both run fine on CPU for a 76-residue
  protein. alanine_swap_cpu.py stands alone: it folds the normal and the I44A version itself.
- RMSD comparison: compare_rmsd.py — runs on your own computer, needs only numpy
  (`pip install numpy`, then `python notebooks/compare_rmsd.py`). Prints the RMSD table in
  proteins/ubiquitin/results.md (AlphaFold, ESMFold and 1UBQ, all 76 blocks and 1–70).
