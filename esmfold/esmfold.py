import json
import os

import numpy as np
import torch
from transformers import EsmForProteinFolding

seqs = {
    "Ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "Calmodulin-1": "MADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTIDFPEFLTMMARKMKDTDSEEEIREAFRVFDKDGNGYISAAELRHVMTNLGEKLTDEEVDEMIREADIDGDGQVNYEEFVQMMTAK",
    "Alpha-synuclein": "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA",
}

os.makedirs("out", exist_ok=True)

model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval()
model.esm = model.esm.half()
model = model.cuda()
model.trunk.set_chunk_size(32)

for name, seq in seqs.items():
    with torch.no_grad():
        out = model.infer(seq)
    out["plddt"] = out["plddt"] * 100

    with open(f"out/{name}.pdb", "w") as f:
        f.write(model.output_to_pdb(out)[0])

    scores = {
        "plddt": out["plddt"][0, :, 1].tolist(),
        "pae": out["predicted_aligned_error"][0].tolist(),
        "ptm": out["ptm"].item(),
    }
    with open(f"out/{name}.json", "w") as f:
        json.dump(scores, f)

    print(
        name,
        "pLDDT:",
        round(sum(scores["plddt"]) / len(seq), 1),
        "pTM:",
        round(scores["ptm"], 2),
    )
