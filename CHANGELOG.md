# Changelog

All notable changes to NanoCDR3Gen.

## Update — May 2026

New entry point: `current_version/nanocdr3gen_v2_run007_omnifold.py`. The original `nanocdr3gen.py` is preserved unchanged.

### Added

- **Real h-NbBCII10 framework.** Replaces the v0.1 humanized VH3 hybrid scaffold (which mixed FR2/CDR1 across sources). Every framework position is now the actual h-NbBCII10 sequence (Vincke et al., *J. Biol. Chem.* 2009, PDB 3EAK), with full IMGT-length CDR masks (CDR1=11, CDR2=8 designable positions; v0.1 used 7/7).
- **AbNatiV2 per-region in-loop loss.** Per-region differentiable nativeness term (CDR1, CDR2, CDR3 weighted independently). Replaces previous post-score-only nativeness use. JAX port of AbNatiV2 verified numerically against the official PyTorch checkpoint (max |diff| ~1e-7).
- **OmniLib fold-stability CNN in-loop loss.** Wraps the pretrained CNN from Wan et al., *Nat. Struct. Mol. Biol.* 2026 (`github.com/antoinekoehl/omnilib-ml`) as a differentiable Mosaic LossTerm. Pushes designs toward `P(high stability) ≥ 0.5` during optimization. JAX port verified against PyTorch (max |diff| ~1e-7).
- **AbLang v2 paired.** Replaces AbLang v1; single unified focal-loss model that reduces germline bias.
- **Protenix v2 with per-position paratope.** `BinderTargetContact.paratope_idx` restricted to CDR3 positions only, freeing CDR1+CDR2 from explicit binding pressure so nativeness/fold gradients can drive them.
- **Hot-spot redesign architecture (optional).** Three-stage pipeline: baseline design → identify low-nativeness positions → redesign only hot positions with accept/revert (rejects redesigns that crash iPTM or fail to improve AbNatiV2).
- **AA-composition caps.** Soft caps on cysteine (CDR1+CDR2), glutamate (CDR1), and lysine (CDR2) based on a 906K natural-VHH composition analysis.

### Changed

- Switched to Protenix v2 (April 2026 ByteDance release). pLDDT now returned in [0,1]; multiplied by 100 in result JSON for cross-version comparability.
- Default Modal GPU bumped from A100 to A100/H200 80 GB for memory headroom (combined AbNatiV2 + OmniLib fold CNN + Protenix v2 in-loop adds ~30 GB peak).

### Validated

- **Natural VHH ceiling on AbNatiV2 measured.** 64 camelid VHHs from SAbDab scored: AbNatiV2 overall median +0.80, OmniLib P(high stability) median 0.68. Used as anchor for "natural-looking" targets.
- **AbNatiV2 and OmniLib fold stability are independent signals.** Measured on 64 naturals + 18 designs, Pearson(AbNatiV2 overall, OmniLib P(high stability)) ≈ −0.07. Optimizing one does not automatically fix the other — both terms are now in the loss.

### Measured results on PD-L1 (n=10 exploratory batch, CDR3=12/14/16)

| Metric (scorer) | mean | best | hit-rate |
|---|---|---|---|
| Protenix v2 iPTM (design-time scorer) | 0.862 | 0.943 | 7/10 ≥ 0.85 |
| AF3 iPTM (field-standard cross-validator, best-of-5-seeds) | 0.606 | 0.860 | 2/10 ≥ 0.80 |
| AbNatiV2 overall (nativeness) | +0.589 | +0.634 | — |
| OmniLib P(high stability) (fold) | 0.749 | 0.961 | 7/10 ≥ 0.75 |

Notes on these numbers:

- This is a small exploratory batch, not a benchmark.
- Protenix v2 iPTM is the design-time scorer and is biased toward sequences the optimizer produced. AF3 iPTM is the independent cross-validator. The two disagree substantially on individual designs; on this batch AF3 returns 0.20-0.50 iPTM units below Protenix v2 for several designs, consistent with published reports on antibody-antigen interfaces.
- The two AF3 binders in this batch (AF3 iPTM ≥ 0.80) have low OmniLib fold scores; the two highest-fold designs failed AF3 binding. No design in the batch is strong on all four axes simultaneously. Per-axis weight calibration and larger batches are the natural next step.
- The v0.1 README reported a 33% AF3 iPTM ≥ 0.80 hit rate on PD-L1 from n=30; the current AF3 hit rate is 20% on n=10 (not directly comparable due to small n and different CDR3-length distribution).

---

## v0.1 — April 2026

Initial public release. Single-file pipeline (`nanocdr3gen.py`): real h-NbBCII10 framework, Protenix v1 backprop, AbLang v1 + ESM-C pseudolikelihoods, AbNatiV2 + CamSol post-scoring. PD-L1 result reported: 33% AF3 iPTM ≥ 0.80 hit rate from 30 designs.
