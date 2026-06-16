#!/usr/bin/env python3
"""
scripts/setup_l1_gwh050_mwp120.py

One-time setup for the L1 closure test at gWH=0.50, mWp=12 TeV.
Run this ONCE locally before condor_submit. Takes ~1 min.

What it does:
  1. Creates the output directory structure
  2. Runs smefit PROJ --noise L0 to generate noiseless BSM pseudo-data
  3. Builds the experimental covariance and saves the Cholesky factor
     (each condor job uses this to draw its own L1 noise realisation)
  4. Writes the base NS runcard template (condor jobs fill in data_path + result_ID)
  5. Creates logs/ directory ready for condor output

Usage:
    cd fullpipeline
    python scripts/setup_l1_gwh050_mwp120.py
"""
import os, sys, yaml, json, subprocess
import numpy as np
from pathlib import Path
from scipy.linalg import block_diag

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))

from scripts.run_pipeline import (
    DATASETS, DS_NAMES, SM_DATA, THEORY, SMEFIT, ENV, make_rge
)
from models.wprime_constrained import WPrimeConstrainedModel

# ── Model ────────────────────────────────────────────────────────────────────
GWH  = 0.50
MWP  = 12.0
GWLF = GWH / 3.0
MODEL = WPrimeConstrainedModel(
    gWH=GWH, gWLf11=GWLF, gWLf22=GWLF, gWLf33=GWLF, gWqf33=GWLF, mWp=MWP
)
TAG     = "wprime_l1_gwh050_mwp120"
OUT_DIR = PIPELINE / "results" / TAG


def main():
    print(f"Model  : {MODEL}")
    print(f"Tag    : {TAG}")
    print(f"OutDir : {OUT_DIR}")

    # ── 1. Create directory structure ────────────────────────────────────────
    for d in ["projections/l0", "runcards", "fits", "logs"]:
        (OUT_DIR / d).mkdir(parents=True, exist_ok=True)
    print("\n[1] Directories created.")

    # ── 2. smefit PROJ --noise L0 ────────────────────────────────────────────
    l0_dir  = OUT_DIR / "projections" / "l0"
    rc_dir  = OUT_DIR / "runcards"
    c_eft   = MODEL.eft_coefficients()

    proj_rc = {
        "projections_path": str(l0_dir),
        "commondata_path":  SM_DATA,
        "theory_path":      THEORY,
        "datasets":         DATASETS,
        "coefficients":     {op: {"constrain": True, "value": float(c)}
                             for op, c in c_eft.items() if abs(c) > 1e-15},
        "use_quad":         False,
        "use_t0":           False,
        "use_theory_covmat": False,
        "rge": make_rge(MWP),
    }
    proj_rc_path = rc_dir / "proj_l0.yaml"
    with open(proj_rc_path, "w") as f:
        yaml.dump(proj_rc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\n[2] Running smefit PROJ --noise L0 ...")
    r = subprocess.run([SMEFIT, "PROJ", str(proj_rc_path), "--noise", "L0"], env=ENV)
    if r.returncode != 0:
        raise RuntimeError(f"smefit PROJ failed (exit {r.returncode})")

    n_written = sum(1 for ds in DS_NAMES if (l0_dir / f"{ds}.yaml").exists())
    print(f"    {n_written}/{len(DS_NAMES)} dataset yamls written to {l0_dir}")

    # ── 3. Build covariance Cholesky (stat + sys, no theory covmat) ──────────
    print("\n[3] Building experimental covariance ...")
    C_blocks, slices, idx = [], {}, 0

    for ds in DATASETS:
        sm_path = f"{SM_DATA}/{ds['name']}.yaml"
        sm      = yaml.safe_load(open(sm_path))
        dc      = sm["data_central"]
        dc      = [dc] if not isinstance(dc, list) else list(dc)
        n       = len(dc)

        stat = sm.get("statistical_error", [0.0] * n)
        stat = list(stat) if isinstance(stat, list) else [stat] * n
        C_ds = np.diag([float(e) ** 2 for e in stat])

        sys_mat = sm.get("systematics", None)
        if sys_mat is not None:
            S = np.array(sys_mat, dtype=float)
            if S.ndim == 1:
                S = S.reshape(1, n)
            C_ds += S.T @ S

        # regularise tiny/zero diagonal entries so Cholesky doesn't fail
        min_var = 1e-30
        for i in range(n):
            C_ds[i, i] = max(C_ds[i, i], min_var)

        C_blocks.append(C_ds)
        slices[ds["name"]] = (idx, idx + n)
        idx += n

    C_full = block_diag(*C_blocks)
    L_chol = np.linalg.cholesky(C_full)
    np.save(OUT_DIR / "projections" / "cov_chol.npy", L_chol)
    np.save(OUT_DIR / "projections" / "cov_slices.npy",
            np.array([(name, s, e) for name, (s, e) in slices.items()], dtype=object),
            allow_pickle=True)
    print(f"    Cholesky saved ({C_full.shape[0]} observables)")

    # ── 4. Base NS runcard template ───────────────────────────────────────────
    print("\n[4] Writing base NS runcard template ...")
    coeff_block = MODEL.uv_coeff_block()

    base_rc = {
        "result_ID":          "__RESULT_ID__",
        "result_path":        str(OUT_DIR / "fits"),
        "data_path":          "__DATA_PATH__",
        "theory_path":        THEORY,
        "use_quad":           False,
        "use_t0":             False,
        "use_theory_covmat":  False,
        "uv_couplings":       True,
        "n_samples":          10000,
        "nlive":              300,
        "lepsilon":           0.01,
        "target_evidence_unc": 0.1,
        "target_post_unc":    0.1,
        "frac_remain":        0.01,
        "datasets":           DATASETS,
        "rge":                make_rge(MWP),
        "coefficients":       coeff_block,
    }
    template_path = rc_dir / "base_ns_template.yaml"
    with open(template_path, "w") as f:
        yaml.dump(base_rc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"    Template: {template_path}")

    # ── 5. Summary ───────────────────────────────────────────────────────────
    print(f"""
Setup complete.

  L0 projection : {l0_dir}
  NS template   : {template_path}
  Cholesky      : {OUT_DIR}/projections/cov_chol.npy
  Logs dir      : {OUT_DIR}/logs/

Now submit 50 condor jobs:

    condor_submit /data/theorie/wtelman/submit_l1_gwh050_mwp120.sub
""")


if __name__ == "__main__":
    main()
