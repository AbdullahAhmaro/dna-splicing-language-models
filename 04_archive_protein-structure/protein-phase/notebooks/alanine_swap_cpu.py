# CPU version of esmfold_hf.py + alanine_swap.py in one script (no device_map, no .half()) —
# for when Colab has no free GPU. Fine on CPU for a 76-residue protein. In Colab, first run:
# !pip install -q transformers accelerate
import numpy as np, torch
from transformers import EsmForProteinFolding

model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", use_safetensors=True, low_cpu_mem_usage=True).eval()
model.trunk.set_chunk_size(64)

seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
mut = seq[:43] + "A" + seq[44:]          # position 44: I -> A  (I44A)

with torch.no_grad():
    pdb = model.infer_pdb(seq)
    pdb_mut = model.infer_pdb(mut)
open("ubiquitin_normal_esmfold.pdb", "w").write(pdb)
open("ubiquitin_I44A_esmfold.pdb", "w").write(pdb_mut)

def ca(p):
    return np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                     for l in p.splitlines() if l.startswith("ATOM") and l[12:16].strip() == "CA"])

def plddt(p):   # B-factor column of the CA atoms (0-1 scale here; x100 = pLDDT)
    return np.array([float(l[60:66]) for l in p.splitlines() if l.startswith("ATOM") and l[12:16].strip() == "CA"])

def rmsd(P, Q):
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(P.T @ Q)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return np.sqrt((((P @ U @ D @ Vt) - Q) ** 2).sum(1).mean())

for name, p in (("normal", pdb), ("I44A", pdb_mut)):
    print(f"{name}: mean pLDDT {100 * plddt(p).mean():.1f}, at residue 44 {100 * plddt(p)[43]:.1f}")
print(f"RMSD normal vs I44A, all 76: {rmsd(ca(pdb), ca(pdb_mut)):.2f} Å")
print(f"RMSD normal vs I44A, 1-70:   {rmsd(ca(pdb)[:70], ca(pdb_mut)[:70]):.2f} Å")

from google.colab import files
files.download("ubiquitin_normal_esmfold.pdb")
files.download("ubiquitin_I44A_esmfold.pdb")
