# NanoCDR3Gen

A minimal, single-file pipeline for de novo VHH (nanobody) design against a user-specified target, built on [escalante-bio/mosaic](https://github.com/escalante-bio/mosaic) and designed to run on [Modal](https://modal.com) A100 GPUs.

NanoCDR3Gen optimizes CDR1, CDR2, and CDR3 positions on a humanized VH3 hybrid framework via backpropagation through [Protenix](https://github.com/bytedance/Protenix), with AbLang and ESM-C pseudolikelihoods biasing sequences toward natural antibody patterns.

## Why this exists

Most published de novo nanobody design pipelines (mBER, RFantibody) are optimized for **binders**. They generate thousands of designs per target and filter aggressively on predicted interface quality. That works as a proof of concept for AI-driven design, but a binder is not the same as a therapeutic candidate — a drug-grade nanobody also needs to look like a natural antibody (low immunogenicity risk) and behave well in solution (manufacturability, formulation).

NanoCDR3Gen is built around the opposite question: **given a small compute budget, can I reach published state-of-the-art hit rates while keeping nativeness and solubility in a therapeutic-viable range?** The goal is good candidates, not just binders. The pipeline pushes AbLang (antibody language model) pseudolikelihood inside the loss and post-scores every design with AbNatiV2 nativeness and CamSol solubility.

On PD-L1, NanoCDR3Gen reaches a 33% AF3 iPTM ≥ 0.80 hit rate from 30 designs, against mBER's reported ~40% on best hotspot from libraries of ~2,600-7,900 designs per target — roughly the same per-design hit rate with ~100× fewer designs, and with every design carrying nativeness and solubility scores alongside the binding prediction.

See [results and methodology notes](#results-and-methodology-notes) below.

## Quick start

### 1. Install Modal and authenticate

```bash
pip install modal
modal setup
```

### 2. Prepare your target PDB

Truncate your target around the binding hotspots and place the result at `./pdbs/<TARGET>_trunc.pdb`. The mBER paper recommends a **25 Å sphere** around hotspot residues.

Example with BioPython:

```python
from Bio.PDB import PDBParser, PDBIO, Select
import numpy as np

HOTSPOTS = [54, 56, 66, 115]  # residue numbers on chain A
RADIUS = 25.0

p = PDBParser(QUIET=True).get_structure("target", "my_target.pdb")
chain_A = p[0]["A"]

hot_coords = [r["CA"].coord for r in chain_A if r.id[1] in HOTSPOTS]
keep = set()
for r in chain_A:
    if r.id[0] != " " or "CA" not in r:
        continue
    for h in hot_coords:
        if np.linalg.norm(r["CA"].coord - h) <= RADIUS:
            keep.add(r.id[1])
            break

class Sel(Select):
    def accept_chain(self, c):
        return c.id == "A"
    def accept_residue(self, r):
        return r.id[0] == " " and r.id[1] in keep

io = PDBIO()
io.set_structure(p)
io.save("./pdbs/my_target_trunc.pdb", Sel())
```

### 3. Fill in the TARGET BLOCK in `nanocdr3gen.py`

```python
TARGET_NAME     = "pdl1"
TARGET_LABEL    = "Human PD-L1 IgV domain, 25A trunc"
TARGET_SEQUENCE = "AFTVTVPKDLYVVSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKV"
TARGET_PDB      = "pdl1_trunc.pdb"
```

### 4. Edit the JOBS list

```python
JOBS = [
    (14, 15, 42, 0.3),   # CDR3 length 14, 15 designs, seed 42, AbLang weight 0.3
    (16, 15, 42, 0.3),   # CDR3 length 16, 15 designs, seed 42, AbLang weight 0.3
]
```

### 5. Launch

```bash
modal run --detach nanocdr3gen.py
```

`--detach` lets the run survive terminal disconnects. A single job (15 designs) takes about 90 minutes on an A100 and costs roughly $13.

### 6. Results

- `results_<TARGET>_LIVE.json` is written incrementally and supports resume on re-launch.
- `results_<TARGET>_<TIMESTAMP>.json` is the final timestamped output.

Each design is named `{target}_cdr{N}_abl{NN}_d{NN}` (e.g. `pdl1_cdr14_abl03_d00`).

## Ranking designs

**Do not rank final designs by the Protenix iPTM printed during the run.** Protenix is the design-time scoring model and is biased toward sequences the optimizer produced. Always validate with an independent fold model:

- [AlphaFold3 Server](https://alphafoldserver.com/) (recommended)
- [Boltz](https://github.com/jwohlwend/boltz)
- [Chai-1](https://github.com/chaidiscovery/chai-lab)

A final AF3 iPTM ≥ 0.80 is the field-standard cutoff for a confident predicted interaction.

## Recommended parameters

| Parameter | Recommended | Notes |
|---|---|---|
| AbLang weight | **0.3** | See the weight study in the methodology notes. 0.1 gives higher variance with a lower tail. |
| CDR3 length | 14 or 16 | Target-dependent. Sweep if unsure. |
| Truncation radius | 25 Å | Matches mBER. |
| Designs per job | 15-20 | Smaller batches finish within Modal's 4 h default timeout. |

## Results and methodology notes

**Binding.** On concave druggable targets (PD-L1, RSV F apex), NanoCDR3Gen reached AF3 iPTM ≥ 0.80 hit rates of 25-35% from batches of 15-30 designs. On flat helical epitopes, the hit rate was 0%. **Epitope shape dominates every tunable in this pipeline** — it is the strongest single finding from all the experiments run during development.

**Nativeness.** Every design is post-scored with AbNatiV2 VHH2. Under the default AbLang weight (0.3), designs reach ABN2 means of 0.66-0.70 across targets, with tail populations above 0.70 (a common "native-looking" threshold). This is noticeably better than running the same pipeline at AbLang weight 0.1, which is what many open-source defaults use.

**Solubility.** Every design is also post-scored with CamSol. Under the default settings, designs reach CamSol means of 0.73-0.83, with most designs above the +0.5 "highly soluble" threshold. The solubility lift was an unintended side effect of increasing the AbLang weight, not a dedicated optimization — natural antibody sequences tend to avoid hydrophobic patches that adversarial loss-driven designs sometimes stumble into.

**Methodology validation.** The AbLang weight was studied in a paired 103-design A/B (0.1 vs 0.3). Means barely moved, but the hit rate at the usable threshold lifted 3.5× for 0.3. Fixing CDR1 to a cropped camelid germline was tested and rejected (binding collapsed). These decisions are baked into the defaults in this script.

**All numbers reported here are in-silico AF3 / AbNatiV2 / CamSol predictions.** No wet-lab validation has been performed on designs from this pipeline at the time of release. Wet-lab characterization (expression and SPR) is the required next step before any therapeutic claim can be made.

## Dependencies

Installed automatically inside the Modal container:

- [mosaic](https://github.com/escalante-bio/mosaic) (escalante-bio) — composable backprop design framework
- [Protenix](https://github.com/bytedance/Protenix) (ByteDance) — AlphaFold3-class structure prediction
- [AbLang](https://github.com/oxpig/AbLang) — antibody language model (Olsen et al., Bioinformatics 2022)
- [ESM-C](https://github.com/evolutionaryscale/esm) (EvolutionaryScale) — general protein language model
- JAX (with CUDA), equinox, gemmi

## Framework

The scaffold is a humanized VH3 hybrid derived from **h-NbBCII10** (Vincke et al., *J. Biol. Chem.* 2009, PDB 3EAK) with CDR1 = 7, CDR2 = 7, and variable CDR3. The framework is hardcoded in `build_framework()` and can be replaced with any other VHH scaffold by editing that function.

## Citation

If you use this code, please cite the underlying libraries and models:

- **mosaic**: escalante-bio, <https://github.com/escalante-bio/mosaic>
- **Protenix**: Chen et al., 2025, <https://github.com/bytedance/Protenix>
- **AbLang**: Olsen et al., *Bioinformatics* 38(7), 2022, <https://doi.org/10.1093/bioinformatics/btac051>
- **ESM-C**: EvolutionaryScale, 2024, <https://github.com/evolutionaryscale/esm>
- **mBER** (framework choice and truncation method): Stenger-Smith et al., *bioRxiv* 10.1101/2025.09.26.678877

## License

MIT (see `LICENSE`).

## Disclaimer

This software is research code. Designs produced by this pipeline are computational predictions and have not been experimentally validated by the author at the time of release. Do not use for clinical applications without independent wet-lab characterization.
