"""
NanoCDR3Gen — single-target VHH design template

A minimal Modal script for designing VHH (single-domain antibody) binders to a
target protein, built on escalante-bio/mosaic. Optimizes CDR1, CDR2, and CDR3
positions on the h-NbBCII10 humanized framework via backpropagation through
Protenix (an open-source AlphaFold3-class structure prediction model).

USAGE
-----
1. Install Modal and authenticate
       pip install modal
       modal setup

2. Truncate your target PDB around the binding hotspots and place the
   truncated file at:
       ./pdbs/<TARGET>_trunc.pdb
   The mBER paper recommends a 25 A sphere around hotspot residues
   (Stenger-Smith et al. bioRxiv 2025.09.26.678877).

3. Fill in the TARGET BLOCK below (4 lines).

4. Edit the JOBS list (CDR3 length, designs per condition, AbLang weight).

5. Launch (use --detach so the run survives terminal disconnects):
       modal run --detach nanocdr3gen.py

OUTPUT
------
- results_<TARGET>_LIVE.json          incremental, resume-safe
- results_<TARGET>_<TIMESTAMP>.json   final timestamped output

DESIGN ID NAMING
----------------
{target}_cdr{N}_abl{NN}_d{NN}
  Example: pdl1_cdr14_abl03_d00
  abl01 = AbLang weight 0.1
  abl03 = AbLang weight 0.3

RANKING DESIGNS
---------------
The Protenix iPTM printed during the run is a design-time score and should NOT
be used to rank final designs. Always validate with an independent fold model
(AlphaFold3 / AF3 server, Boltz, or Chai) before claiming a "hit". An AF3 iPTM
threshold of >= 0.80 is the field-standard cutoff for a confident interaction.

DEPENDENCIES (installed inside the Modal container)
----------------------------------------------------
- mosaic (escalante-bio):  https://github.com/escalante-bio/mosaic
- Protenix (ByteDance):    https://github.com/bytedance/Protenix
- AbLang:                  Olsen et al., Bioinformatics 2022
- ESM-C (EvolutionaryScale): https://github.com/evolutionaryscale/esm
- JAX, equinox, gemmi

CITATION
--------
If you use this code, please cite mosaic and the underlying models above, plus
the mBER paper which provides the framework choice and truncation method:

  Stenger-Smith et al., "mBER: Controllable de novo antibody design with
  million-scale experimental screening", bioRxiv 10.1101/2025.09.26.678877

LICENSE
-------
MIT License (see LICENSE file)
"""

import modal
from pathlib import Path

# ============================================================================
# TARGET BLOCK — fill in for your target
# ============================================================================
TARGET_NAME     = ""   # Short identifier used in design_id, e.g. "pdl1", "her2"
TARGET_LABEL    = ""   # Human-readable name for printouts and JSON metadata
TARGET_SEQUENCE = ""   # Truncated amino acid sequence (the part used for design)
TARGET_PDB      = ""   # PDB filename in ./pdbs/, e.g. "PDL1_trunc.pdb"

# Notes (optional but recommended for reproducibility):
# Source PDB         :
# Chain              :
# Hotspots           :
# Truncation radius  :
# Mutations          :
# Other              :
# ============================================================================


# ============================================================================
# JOBS — design conditions to run
# Each tuple: (cdr3_length, n_designs, seed, ablang_weight)
# Recommended starting point: AbLang weight 0.3, CDR3 lengths 12 to 18
# ============================================================================
JOBS = [
    # (14, 5, 42, 0.3),
    # (16, 5, 42, 0.3),
]
# ============================================================================


# ============================================================================
# FRAMEWORK — h-NbBCII10 humanized scaffold (Vincke et al. JBC 2009, PDB 3EAK)
# CDR positions are masked with "X" and filled in by the design loop.
# Modify only if you want a different VHH framework.
# ============================================================================
def build_framework(cdr3_len: int) -> str:
    return (
        "EVQLVESGGGLVQPGGSLRLSCAASG"                     # FR1 (26 aa)
        + "X" * 7                                         # CDR1 (7 aa, designed)
        + "AGWFRQAPGKEREFVAA"                             # FR2 (17 aa)
        + "X" * 7                                         # CDR2 (7 aa, designed)
        + "N" + "NADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYC"   # FR3 (38 aa)
        + "X" * cdr3_len                                  # CDR3 (variable, designed)
        + "WGQGTLVTVSS"                                   # FR4 (11 aa)
    )
# ============================================================================


# ----------------------------------------------------------------------------
#  Below this line: design loop, Modal image, and save logic.
#  No edits needed unless you are changing the underlying mosaic pipeline.
# ----------------------------------------------------------------------------

assert TARGET_NAME, "TARGET_NAME is empty — fill in the TARGET BLOCK before running"
assert TARGET_SEQUENCE, "TARGET_SEQUENCE is empty"
assert TARGET_PDB, "TARGET_PDB is empty"
assert JOBS, "JOBS list is empty"

PDB_PATH = f"/root/{TARGET_PDB}"

def download_protenix():
    from mosaic.models.protenix import Protenix2025
    Protenix2025()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands("git clone https://github.com/escalante-bio/mosaic.git")
    .workdir("mosaic")
    .run_commands("sed -i '/jopenfold3/d' pyproject.toml")
    .run_commands("uv pip install --system -r pyproject.toml")
    .run_commands("uv pip install --system jax[cuda]")
    .run_commands("uv pip install --system .")
    .run_commands("uv pip install --system esm")
    .run_function(download_protenix)
    .run_commands("uv pip install --system equinox")
    .env({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"})
    .add_local_file(f"./pdbs/{TARGET_PDB}", PDB_PATH)
)

app = modal.App(f"nanocdr3gen-{TARGET_NAME}")


@app.function(gpu="A100", image=image, timeout=14400)
def design(cdr3_len: int, n_designs: int = 5, seed: int = 42,
           ablang_weight: float = 0.3):
    """One Modal job: design `n_designs` VHHs at the given CDR3 length."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    import gemmi

    from mosaic.optimizers import simplex_APGM, gradient_MCMC
    import mosaic.losses.structure_prediction as sp
    from mosaic.losses.ablang import AbLangPseudoLikelihood, load_ablang
    from mosaic.losses.esmc import ESMCPseudoLikelihood, load_esmc
    from mosaic.losses.transformations import SetPositions
    from mosaic.structure_prediction import TargetChain
    from mosaic.models.protenix import Protenix2025
    from mosaic.common import TOKENS

    print(f"\n{'='*60}")
    print(f"Target  : {TARGET_LABEL or TARGET_NAME}  ({len(TARGET_SEQUENCE)} aa)")
    print(f"CDR3    : {cdr3_len}")
    print(f"AbLang  : {ablang_weight}")
    print(f"Designs : {n_designs}")
    print(f"PDB     : {PDB_PATH}")
    print(f"{'='*60}")

    masked_framework_sequence = build_framework(cdr3_len)
    target_structure = gemmi.read_structure(PDB_PATH)
    template_chain = target_structure[0][0]
    print(f"Template: {template_chain.name} "
          f"({len(list(template_chain.get_polymer()))} residues)")

    ablang, ablang_tokenizer = load_ablang("heavy")
    ablang_pll = AbLangPseudoLikelihood(
        model=ablang, tokenizer=ablang_tokenizer, stop_grad=True
    )
    ESMCPLL = ESMCPseudoLikelihood(load_esmc("esmc_300m"), stop_grad=True)
    protenix = Protenix2025()

    design_features, design_structure = protenix.target_only_features(
        chains=[
            TargetChain(masked_framework_sequence, use_msa=True),
            TargetChain(TARGET_SEQUENCE, use_msa=True, template_chain=template_chain),
        ],
    )

    structure_loss = (
        sp.BinderTargetContact(
            paratope_idx=np.array(
                [i for (i, c) in enumerate(masked_framework_sequence) if c == "X"]
            )
        )
        + 0.05 * sp.TargetBinderPAE()
        + 0.05 * sp.BinderTargetPAE()
        + 0.025 * sp.IPTMLoss()
        + 0.4 * sp.WithinBinderPAE()
        + 0.025 * sp.pTMEnergy()
        + 0.1 * sp.PLDDTLoss()
    )

    model_loss = protenix.build_multisample_loss(
        loss=structure_loss, features=design_features,
        recycling_steps=2, sampling_steps=20,
    )

    loss = SetPositions.from_sequence(
        wildtype=masked_framework_sequence,
        loss=0.1 * ESMCPLL + 2 * model_loss + ablang_weight * ablang_pll,
    )

    results = []
    num_designed_residues = len(
        [c for c in masked_framework_sequence if c == "X"]
    )

    for i in range(n_designs):
        print(f"\n--- Design {i+1}/{n_designs} ---")
        rng = np.random.randint(1000000) + seed + i

        _pssm = 0.5 * jax.random.gumbel(
            key=jax.random.key(rng),
            shape=(num_designed_residues, 20),
        )

        _, partial_pssm = simplex_APGM(
            loss_function=loss, x=_pssm,
            n_steps=50, stepsize=1.5 * np.sqrt(_pssm.shape[0]),
            momentum=0.2, scale=1.00, serial_evaluation=False,
            logspace=True, max_gradient_norm=1.0,
        )
        _, partial_pssm = simplex_APGM(
            loss_function=loss, x=partial_pssm,
            n_steps=30, stepsize=0.5 * np.sqrt(_pssm.shape[0]),
            momentum=0.0, scale=1.1, serial_evaluation=False,
            logspace=False, max_gradient_norm=1.0,
        )
        _, partial_pssm = simplex_APGM(
            loss_function=loss, x=jnp.log(partial_pssm + 1e-5),
            n_steps=30, stepsize=0.25 * np.sqrt(_pssm.shape[0]),
            momentum=0.0, scale=1.1, serial_evaluation=False,
            logspace=True, max_gradient_norm=1.0,
        )

        s_mcmc = gradient_MCMC(
            loss=loss, sequence=jax.device_put(partial_pssm.argmax(-1)),
            steps=30, fix_loss_key=False, proposal_temp=1e-5,
            max_path_length=1,
        )

        final_pssm = loss.sequence(jax.nn.one_hot(s_mcmc, 20))
        full_sequence = "".join(TOKENS[k] for k in final_pssm.argmax(-1))

        prediction = protenix.predict(
            PSSM=final_pssm, writer=design_structure,
            features=design_features, recycling_steps=10,
            key=jax.random.key(rng),
        )

        # Standard design ID: {target}_cdr{N}_abl{NN}_d{NN}
        abl_str = f"{int(round(ablang_weight*10)):02d}"
        design_id = f"{TARGET_NAME}_cdr{cdr3_len}_abl{abl_str}_d{i:02d}"

        print(f"Design   : {design_id}")
        print(f"Sequence : {full_sequence}")
        print(f"iPTM     : {prediction.iptm:.3f}    (design-time only — validate externally)")
        print(f"pLDDT    : {prediction.plddt.mean():.1f}")

        results.append({
            "design_id": design_id,
            "target": TARGET_NAME,
            "target_label": TARGET_LABEL,
            "target_seq_len": len(TARGET_SEQUENCE),
            "cdr3_len": cdr3_len,
            "ablang_weight": ablang_weight,
            "sequence": full_sequence,
            "iptm": float(prediction.iptm),
            "plddt": float(prediction.plddt.mean()),
        })

    return results


@app.local_entrypoint()
def main():
    """Run all jobs in JOBS, save incrementally, support resume on re-launch."""
    import json
    from datetime import datetime

    SAVE_FILE = Path(f"results_{TARGET_NAME}_LIVE.json")
    all_results = []

    if SAVE_FILE.exists():
        with open(SAVE_FILE) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} previous results from {SAVE_FILE}")
        done_ids = {r["design_id"] for r in all_results}
    else:
        done_ids = set()

    def save():
        with open(SAVE_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  [saved {len(all_results)} designs to {SAVE_FILE}]")

    remaining = []
    for cdr3, n, seed, abl in JOBS:
        abl_str = f"{int(round(abl*10)):02d}"
        first_id = f"{TARGET_NAME}_cdr{cdr3}_abl{abl_str}_d00"
        if first_id in done_ids:
            print(f"Skipping cdr{cdr3} abl{abl_str} (already done)")
            continue
        remaining.append((cdr3, n, seed, abl))

    if not remaining:
        print("All jobs already completed.")
    else:
        total = sum(j[1] for j in remaining)
        print(f"\nLaunching {len(remaining)} jobs ({total} designs) on Modal A100")
        print()
        for result_list in design.starmap(remaining):
            all_results.extend(result_list)
            save()
            r = result_list[0]
            best = max(result_list, key=lambda x: x["iptm"])
            print(f"  cdr{r['cdr3_len']} abl{r['ablang_weight']}: "
                  f"best iPTM={best['iptm']:.3f}")

    final_path = Path(
        f"results_{TARGET_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2)

    all_results.sort(key=lambda x: x["iptm"], reverse=True)
    print(f"\n{'='*60}")
    print(f"ALL DONE — {len(all_results)} designs")
    print("="*60)
    print("\nTop 10 by Protenix iPTM (design-time score, validate with AF3):")
    for r in all_results[:10]:
        print(f"  {r['design_id']}: iPTM={r['iptm']:.3f} pLDDT={r['plddt']:.1f}")
    print(f"\nSaved to {final_path}")
    print(f"Live file:  {SAVE_FILE}")
