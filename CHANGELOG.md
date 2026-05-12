# Changelog

All notable changes to NanoCDR3Gen.

## Update — May 2026

New entry point: `current_version/nanocdr3gen_v2_run007_omnifold.py`. The original `nanocdr3gen.py` is preserved unchanged.

### Added

- **AbNatiV2 per-region in-loop loss.** Per-region differentiable nativeness term (CDR1, CDR2, CDR3 weighted independently). Replaces previous post-score-only nativeness use. JAX port of AbNatiV2 verified numerically against the official PyTorch checkpoint (max |diff| ~1e-7).
- **OmniLib fold-stability CNN in-loop loss.** Wraps the pretrained CNN from Wan et al., *Nat. Struct. Mol. Biol.* 2026 (`github.com/antoinekoehl/omnilib-ml`) as a differentiable Mosaic LossTerm. Pushes designs toward `P(high stability) ≥ 0.5` during optimization. JAX port verified against PyTorch (max |diff| ~1e-7).
- **AbLang v2 paired.** Replaces AbLang v1; single unified focal-loss model that reduces germline bias.
- **Protenix v2 with per-position paratope.** `BinderTargetContact.paratope_idx` restricted to CDR3 positions only, freeing CDR1+CDR2 from explicit binding pressure so nativeness/fold gradients can drive them.
- **Hot-spot redesign architecture (optional).** Three-stage pipeline: baseline design → identify low-nativeness positions → redesign only hot positions with accept/revert (rejects redesigns that crash iPTM or fail to improve V2).
- **AA-composition caps.** Soft caps on cysteine (CDR1+CDR2), glutamate (CDR1), and lysine (CDR2) based on a 906K natural-VHH composition analysis.

### Changed

- Switched to Protenix v2 (April 2026 ByteDance release). pLDDT now returned in [0,1]; multiplied by 100 in result JSON for cross-version comparability.
- Default Modal GPU bumped from A100 to A100/H200 80 GB for memory headroom (combined ABN2 + fold CNN + Protenix v2 in-loop adds ~30 GB peak).

### Validated

- **Natural VHH ceiling on V2 measured.** 64 camelid VHHs from SAbDab scored: V2 overall median +0.80, fold P(high) median 0.68. Used as anchor for "natural-looking" targets.
- **V2 and fold are independent signals.** Measured on 64 naturals + 18 designs, Pearson(V2-overall, fold) ≈ −0.07. Optimizing one does not automatically fix the other — both terms are now in the loss.

### Measured improvement on PD-L1 (n=6, single batch)

| Metric | Before update | After update | Δ |
|---|---|---|---|
| iPTM mean | 0.729 | 0.868 | +0.14 |
| iPTM ≥ 0.85 | 33% | 83% | +50pp |
| V2 overall mean | +0.562 | +0.588 | +0.03 |
| Fold P(high) mean | 0.504 | 0.808 | +0.30 |
| Fold ≥ 0.75 | 17% | 83% | +66pp |

---

## v0.1 — April 2026

Initial public release. Single-file pipeline (`nanocdr3gen.py`): real h-NbBCII10 framework, Protenix v1 backprop, AbLang v1 + ESM-C pseudolikelihoods, AbNatiV2 + CamSol post-scoring. PD-L1 result reported: 33% AF3 iPTM ≥ 0.80 hit rate from 30 designs.
