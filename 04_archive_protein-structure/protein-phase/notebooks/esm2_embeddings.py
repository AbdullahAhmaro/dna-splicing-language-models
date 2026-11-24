# Run in Google Colab (Runtime type: T4 GPU). In Colab, first run:
# !pip install -q transformers

import torch
from transformers import AutoTokenizer, EsmModel

seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm2 = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").cuda().eval()

inputs = tok(seq, return_tensors="pt").to("cuda")
with torch.no_grad():
    emb = esm2(**inputs).last_hidden_state[0, 1:-1]   # drop start/end tokens

print(emb.shape)            # (76, 1280): one vector per amino acid
protein_vec = emb.mean(0)   # (1280,): one vector for the whole protein

torch.save(emb.cpu(), "ubiquitin_esm2_650M.pt")
from google.colab import files
files.download("ubiquitin_esm2_650M.pt")
