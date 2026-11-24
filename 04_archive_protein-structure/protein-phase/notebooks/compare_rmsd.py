# Run locally (needs only numpy):  python notebooks/compare_rmsd.py
# RMSD on CA atoms after overlapping (Kabsch superposition) for AlphaFold, ESMFold and
# the real structure 1UBQ — once for all 76 residues, once for 1-70 (without the floppy tail).
from pathlib import Path
import numpy as np

UBQ = Path(__file__).resolve().parent.parent / "proteins" / "ubiquitin"
FILES = {
    "AlphaFold": UBQ / "alphafold" / "alphafold_best_model.pdb",
    "ESMFold":   UBQ / "esmfold" / "esmfold_model.pdb",
    "1UBQ":      UBQ / "reference" / "1UBQ_real_structure.pdb",
}
PAIRS = [("AlphaFold", "1UBQ"), ("ESMFold", "1UBQ"), ("AlphaFold", "ESMFold")]

def ca(path, chain="A", first=1, last=76):
    # {residue number: (residue name, xyz)}. ATOM lines only, so waters (HETATM HOH) are skipped;
    # if an atom has alternate positions, keep the first one.
    out = {}
    for l in open(path):
        if l.startswith("ATOM") and l[12:16].strip() == "CA" and l[21] == chain and l[16] in " A":
            r = int(l[22:26])
            if first <= r <= last and r not in out:
                out[r] = (l[17:20], np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]))
    return out

def rmsd(P, Q):
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(P.T @ Q)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return np.sqrt((((P @ U @ D @ Vt) - Q) ** 2).sum(1).mean())

atoms = {name: ca(path) for name, path in FILES.items()}
for name, a in atoms.items():
    assert sorted(a) == list(range(1, 77)), f"{name}: expected CA atoms for residues 1-76, got {len(a)}"

print("| Compared | All 76 residues | Residues 1-70 (no tail) |")
print("|---|---|---|")
for a, b in PAIRS:
    A, B = atoms[a], atoms[b]
    assert all(A[r][0] == B[r][0] for r in A), f"{a} and {b} have different amino acids"
    cells = []
    for last in (76, 70):
        res = range(1, last + 1)   # superimpose on exactly the residues being measured
        cells.append(rmsd(np.array([A[r][1] for r in res]), np.array([B[r][1] for r in res])))
    print(f"| {a} vs {b} | {cells[0]:.2f} Å | {cells[1]:.2f} Å |")
