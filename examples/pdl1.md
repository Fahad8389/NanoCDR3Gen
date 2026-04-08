# Example: PD-L1 (human)

A worked example of setting up NanoCDR3Gen against human PD-L1 IgV domain, the same target used by the mBER benchmark.

## 1. Get the PDB

Download a PD-L1 structure from RCSB PDB. mBER uses the structure from their own repository; a common public alternative is `4ZQK` (PD-L1 in complex with an antibody). Any PD-L1 structure with the IgV domain resolved will work.

```bash
curl -O https://files.rcsb.org/download/4ZQK.pdb
```

## 2. Hotspots (from mBER)

The mBER example notebook uses these hotspot residues on PD-L1 chain A:

| Residue | Role |
|---|---|
| A54 | IgV cleft |
| A56 | IgV cleft |
| A66 | IgV cleft |
| A115 | C-terminal loop |

## 3. Truncate to a 25 Å sphere

```python
from Bio.PDB import PDBParser, PDBIO, Select
import numpy as np

HOTSPOTS = [54, 56, 66, 115]
RADIUS = 25.0

p = PDBParser(QUIET=True).get_structure("pdl1", "4ZQK.pdb")
chain_A = p[0]["A"]

hot_coords = [r["CA"].coord for r in chain_A if r.id[1] in HOTSPOTS and "CA" in r]
keep = set()
for r in chain_A:
    if r.id[0] != " " or "CA" not in r:
        continue
    for h in hot_coords:
        if np.linalg.norm(r["CA"].coord - h) <= RADIUS:
            keep.add(r.id[1])
            break

class Sel(Select):
    def accept_chain(self, c): return c.id == "A"
    def accept_residue(self, r): return r.id[0] == " " and r.id[1] in keep

io = PDBIO()
io.set_structure(p)
io.save("pdbs/pdl1_trunc.pdb", Sel())
```

## 4. Extract the truncated sequence

```python
from Bio.PDB import PDBParser

three_to_one = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

p = PDBParser(QUIET=True).get_structure("pdl1", "pdbs/pdl1_trunc.pdb")
residues = [r for r in p[0]["A"] if r.id[0] == " "]
sequence = "".join(three_to_one.get(r.get_resname(), "X") for r in residues)
print(sequence)
```

Paste the printed sequence into `TARGET_SEQUENCE` in `nanocdr3gen.py`.

## 5. Fill in the TARGET BLOCK

```python
TARGET_NAME     = "pdl1"
TARGET_LABEL    = "Human PD-L1 IgV domain (25A truncation around A54/A56/A66/A115)"
TARGET_SEQUENCE = "<paste from step 4>"
TARGET_PDB      = "pdl1_trunc.pdb"
```

## 6. Set the jobs

Good starting point for PD-L1:

```python
JOBS = [
    (14, 15, 42, 0.3),
    (16, 15, 42, 0.3),
]
```

30 designs, about 3 hours on an A100 running the two jobs in parallel, roughly $26.

## 7. Launch

```bash
modal run --detach nanocdr3gen.py
```

## Expected result

On a concave target like PD-L1, you should see AF3 iPTM ≥ 0.80 on 25-35% of designs after validating the final sequences on an independent fold model (AlphaFold3 server is the cleanest option). The Protenix iPTM printed during the run is a design-time score and should not be used for ranking.
