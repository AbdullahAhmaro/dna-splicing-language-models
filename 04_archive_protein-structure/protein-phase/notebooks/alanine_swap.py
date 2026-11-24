# Run AFTER esmfold_hf.py, in the same Colab notebook (uses model, seq, pdb from it).
import numpy as np, torch

mut = seq[:43] + "A" + seq[44:]          # position 44: I -> A  (I44A)
with torch.no_grad():
    pdb_mut = model.infer_pdb(mut)
open("ubiquitin_I44A_esmfold.pdb", "w").write(pdb_mut)

def ca(p):
    return np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                     for l in p.splitlines() if l.startswith("ATOM") and l[12:16].strip() == "CA"])

def rmsd(P, Q):
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(P.T @ Q)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return np.sqrt((((P @ U @ D @ Vt) - Q) ** 2).sum(1).mean())

print(f"RMSD normal vs I44A: {rmsd(ca(pdb), ca(pdb_mut)):.2f} Å")
