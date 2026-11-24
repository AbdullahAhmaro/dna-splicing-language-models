# Run in Google Colab (Runtime type: T4 GPU). In Colab, first run:
# !pip install -q transformers accelerate

import time, torch
from transformers import AutoTokenizer, EsmForProteinFolding

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", use_safetensors=True, low_cpu_mem_usage=True, device_map="cuda")
model.esm = model.esm.half()      # saves GPU memory on the T4
model.trunk.set_chunk_size(64)

seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

start = time.time()
with torch.no_grad():
    pdb = model.infer_pdb(seq)
print(f"Took {time.time() - start:.1f} s")

open("ubiquitin_esmfold.pdb", "w").write(pdb)

# average confidence = B-factor column of the CA atoms (0-1 scale here; x100 = pLDDT)
ca = [float(l[60:66]) for l in pdb.splitlines() if l.startswith("ATOM") and l[12:16].strip() == "CA"]
print(f"Mean pLDDT: {sum(ca)/len(ca):.3f}")

from google.colab import files
files.download("ubiquitin_esmfold.pdb")
