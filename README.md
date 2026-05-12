# NanoCDR3Gen

A minimal, single-file pipeline for de novo VHH (nanobody) design against a user-specified target, built on [escalante-bio/mosaic](https://github.com/escalante-bio/mosaic) and designed to run on [Modal](https://modal.com) A100 / H200 GPUs.

NanoCDR3Gen optimizes CDR1, CDR2, and CDR3 positions on a real **h-NbBCII10** humanized VH3 scaffold via backpropagation through [Protenix v2](https://github.com/bytedance/Protenix), with [AbLang2](https://github.com/oxpig/AbLang2), [ESM-C](https://github.com/evolutionaryscale/esm), [AbNatiV2](https://gitlab.developers.cam.ac.uk/ch/sormanni/abnativ) per-region nativeness, and [OmniLib fold-stability CNN](https://github.com/antoinekoehl/omnilib-ml) signals biasing sequences toward natural, foldable antibody patterns.

---

## Latest update

The pipeline has been substantially upgraded since the initial release. The current entry point is `current_version/nanocdr3gen_v2_run007_omnifold.py`. The original single-file release (`nanocdr3gen.py`) still works and is preserved unchanged.

### What's new vs the initial release

| Component | Before update | Now |
|---|---|---|
| Framework | humanized VH3 hybrid (CDR1=7, CDR2=7) | **real h-NbBCII10** (Vincke 2009, PDB 3EAK; full IMGT lengths CDR1=11, CDR2=8) |
| Structure scorer | Protenix v1, global paratope | Protenix v2, CDR3-only paratope |
| Antibody LM | AbLang v1 (heavy-only) | AbLang2 paired |
| Nativeness signal | post-score only (AbNatiV2) | **AbNatiV2 per-region in-loop loss** |
| Fold stability | not measured | **OmniLib fold-stability CNN in-loop loss** (Wan *et al.*, *Nat. Struct. Mol. Biol.* 2026) |
| Architecture | single-stage soft + MCMC | adds hot-spot redesign with accept/revert |

### Latest results — PD-L1 (n=10 designs, exploratory batch, May 2026)

A small batch of 10 designs at three CDR3 lengths (12, 14, 16) was scored on four independent axes. This is a development-time snapshot, not a benchmark — CDR3 lengths and n were chosen for development cost, not to compete with library-scale pipelines.

| Metric (scorer) | mean | best | hit-rate |
|---|---|---|---|
| Protenix v2 iPTM (design-time scorer) | 0.862 | 0.943 | 7/10 ≥ 0.85 |
| **AF3 iPTM** (field-standard binding cross-validator, best-of-5-seeds) | 0.606 | 0.860 | 2/10 ≥ 0.80 |
| AbNatiV2 overall (nativeness) | +0.589 | +0.634 | — |
| OmniLib P(high stability) (fold) | 0.749 | 0.961 | 7/10 ≥ 0.75 |

Per-CDR3-length breakdown:

| CDR3 | n | Protenix v2 iPTM mean | AF3 iPTM mean | AbNatiV2 mean | OmniLib mean |
|---|---|---|---|---|---|
| 12 | 2 | 0.802 | 0.470 | +0.608 | 0.835 |
| 14 | 6 | 0.868 | 0.587 | +0.588 | 0.808 |
| 16 | 2 | 0.906 | 0.800 | +0.575 | 0.487 |

**Effect of the new in-loop losses, holding everything else fixed.** Same real h-NbBCII10 framework, same Protenix v2 design-time scorer, same AbLang2, same ESM-C, same seed; the only thing that changes between the two columns is whether AbNatiV2 and OmniLib are in the loss tree.

| Metric | Without in-loop nativeness + fold (n=6) | With in-loop nativeness + fold (n=6, CDR3=14 subset) |
|---|---|---|
| Protenix v2 iPTM mean | 0.729 | 0.868 |
| Protenix v2 iPTM ≥ 0.85 | 2/6 (33%) | 5/6 (83%) |
| AbNatiV2 overall mean | +0.562 | +0.588 |
| OmniLib P(high stability) mean | 0.504 | 0.808 |
| OmniLib P(high stability) ≥ 0.75 | 1/6 (17%) | 5/6 (83%) |

*This isolates the effect of adding the two new in-loop loss terms. The Protenix v2 iPTM column lifts on the design-time scorer; whether that lift survives AF3 cross-validation is a separate question, addressed in the table above. The AbNatiV2 and OmniLib improvements are on the same models in both columns and are independent of any structure scorer.*

**Honest takeaway.** The two new in-loop signals work on their own metrics. AF3 cross-validation on the full n=10 batch shows the design-time Protenix v2 iPTM does not fully translate to AF3 iPTM — a known disagreement on antibody-antigen interfaces. The two AF3 binders (AF3 iPTM ≥ 0.80) in the n=10 batch have low fold scores; the two best fold-scoring designs failed AF3 binding. No design is strong on all four axes simultaneously. Larger batches and per-axis weight calibration are the natural next steps.

Run with:

```bash
modal run --detach current_version/nanocdr3gen_v2_run007_omnifold.py
modal run --detach current_version/nanocdr3gen_v2_run008_omnifold_cdr12_16.py
```

---

## Initial release — `nanocdr3gen.py` (before update)

The sections below describe the original single-file release. Everything from "Why this exists" to "Disclaimer" reflects the pipeline as published in v0.1 (April 2026); numbers and methodology in those sections are from that snapshot and have not been edited. For the current pipeline see "Latest update" above.

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

**Nativeness.** Every design is post-scored with AbNatiV2 VHH2. Under the default AbLang weight (0.3), designs reach AbNatiV2 means of 0.66-0.70 across targets, with tail populations above 0.70 (a common "native-looking" threshold). This is noticeably better than running the same pipeline at AbLang weight 0.1, which is what many open-source defaults use.

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

If you use this code, please cite the underlying libraries and models.

**Core (both releases):**

- **mosaic**: escalante-bio, <https://github.com/escalante-bio/mosaic>
- **ESM-C**: EvolutionaryScale, 2024, <https://github.com/evolutionaryscale/esm>
- **mBER** (framework choice and truncation method): Stenger-Smith et al., *bioRxiv* 10.1101/2025.09.26.678877
- **h-NbBCII10 framework**: Vincke et al., *J. Biol. Chem.* 284(5):3273-3284, 2009, <https://doi.org/10.1074/jbc.M806889200>

**Used in the original release (`nanocdr3gen.py`):**

- **Protenix** (v1): Chen et al., 2025, <https://github.com/bytedance/Protenix>
- **AbLang** (v1): Olsen et al., *Bioinformatics* 38(7), 2022, <https://doi.org/10.1093/bioinformatics/btac051>

**Added in the updated pipeline (`current_version/nanocdr3gen_v2_run007_omnifold.py`):**

- **Protenix v2**: ByteDance, 2026, <https://github.com/bytedance/Protenix>
- **AbLang2** (paired): Olsen et al., *bioRxiv* 2024.02.26.582143, 2024, <https://github.com/oxpig/AbLang2>
- **AbNatiV2** (per-region nativeness, in-loop loss): Ramon et al., *Nat. Mach. Intell.* 6:74-91, 2024 (V2 update bioRxiv 2025), <https://gitlab.developers.cam.ac.uk/ch/sormanni/abnativ>
- **OmniLib fold-stability CNN** (in-loop loss): Wan et al., *Nat. Struct. Mol. Biol.* 2026, <https://doi.org/10.1038/s41594-026-01804-9>, <https://github.com/antoinekoehl/omnilib-ml>

## License

MIT (see `LICENSE`).

## Disclaimer

This software is research code. Designs produced by this pipeline are computational predictions and have not been experimentally validated by the author at the time of release. Do not use for clinical applications without independent wet-lab characterization.
