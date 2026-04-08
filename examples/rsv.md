# Example: RSV F apex antigenic site

A worked example of setting up NanoCDR3Gen against the apex antigenic site on pre-fusion RSV F, the same epitope targeted by nirsevimab (a licensed antibody drug).

## 1. Get the PDB

```bash
curl -O https://files.rcsb.org/download/5UDC.pdb
```

`5UDC` is pre-fusion RSV F A2 in complex with the Fab fragment of nirsevimab (MEDI8897). This gives you ground-truth contact residues to use as hotspots.

## 2. Hotspots (nirsevimab contacts)

Heavy chain contacts from the Fab, picked as the core of the concave apex cleft:

| Residue (chain A) | Region |
|---|---|
| A63, A65, A68 | F2 region |
| A201, A209 | F1 HRA loop |
| A295 | C-terminal contact |

This is a **discontinuous epitope** (the F2 and F1 regions come together in the folded protein), so the truncated sphere will pull in residues from both regions.

## 3. Truncate to a 25 Å sphere around all hotspots

```python
from Bio.PDB import PDBParser, PDBIO, Select
import numpy as np

HOTSPOTS = [63, 65, 68, 201, 209, 295]
RADIUS = 25.0

p = PDBParser(QUIET=True).get_structure("rsvf", "5UDC.pdb")
chain_A = p[0]["A"]

hot_coords = []
for rnum in HOTSPOTS:
    for r in chain_A:
        if r.id[0] == " " and r.id[1] == rnum and "CA" in r:
            hot_coords.append(r["CA"].coord)

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
io.save("pdbs/rsv_apex_trunc.pdb", Sel())
```

Expected output: ~160-170 residues spanning both F2 (early residues) and F1 (HRA loop region).

## 4. Extract the truncated sequence

```python
from Bio.PDB import PDBParser

three_to_one = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

p = PDBParser(QUIET=True).get_structure("rsvf", "pdbs/rsv_apex_trunc.pdb")
residues = sorted([r for r in p[0]["A"] if r.id[0] == " "], key=lambda r: r.id[1])
sequence = "".join(three_to_one.get(r.get_resname(), "X") for r in residues)
print(sequence)
```

## 5. Fill in the TARGET BLOCK

```python
TARGET_NAME     = "rsv_apex"
TARGET_LABEL    = "RSV F pre-fusion, apex antigenic site (25A around nirsevimab contacts)"
TARGET_SEQUENCE = "<paste from step 4>"
TARGET_PDB      = "rsv_apex_trunc.pdb"
```

## 6. Set the jobs

```python
JOBS = [
    (14, 7, 42, 0.3),
    (16, 8, 42, 0.3),
]
```

15 designs, about 1.5 hours on an A100, roughly $13.

## 7. Launch

```bash
modal run --detach nanocdr3gen.py
```

## Notes on epitope shape

The apex of pre-fusion RSV F is a **concave cleft** formed by HRA folding over the globular head. This is the shape class where NanoCDR3Gen performs best.

By contrast, the **site II palivizumab epitope** on the same protein is a flat α-helix around residues 254-277. If you point NanoCDR3Gen at that epitope instead, expect a 0% hit rate. The pipeline (and most current de novo nanobody tools) does not handle flat helical surfaces well. When possible, choose concave epitopes over flat ones.
