#!/usr/bin/env python3
"""
fullpipeline/scripts/generate_projections.py

Generate BSM pseudo-data projections for any UV model.

Uses the linearized theory matrix from the smefit database:
    data_BSM_i = data_SM_i + Σ_j  K_j[i] * c_j

where K_j[i] = LO[op_j][i]  (theory file, linear coefficient for observable i)
and   c_j = model.eft_coefficients()[op_j]  (Wilson coefficient at matching scale)

This is exact in the linear EFT approximation (leading order in c/Λ²).
Operators not present in a dataset's theory file contribute 0 (by convention).

Usage:
    python generate_projections.py --model zprime --gZH 0.12 --gZl 0.04 --mZp 1.0
    python generate_projections.py --model wprime --gWH 0.12 --mWp 1.0

Outputs:
    fullpipeline/projections/<tag>/FCCee_*.yaml    (BSM pseudo-data files)
    fullpipeline/projections/<tag>/coverage.txt    (operator coverage report)
"""

import os, sys, json, yaml, argparse
import numpy as np
from pathlib import Path

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))

DB      = "/data/theorie/wtelman/smefit_database"
SM_DATA = f"{DB}/commondata_projections_L0"
THEORY  = f"{DB}/theory"

# FCC-ee datasets to use in the pipeline
DATASETS = [
    "FCCee_ww_161GeV", "FCCee_ww_240GeV", "FCCee_ww_365GeV",
    "FCCee_Rb_240GeV", "FCCee_Rc_240GeV",
    "FCCee_Rmu_240GeV", "FCCee_Rtau_240GeV",
    "FCCee_bb_Afb_240GeV", "FCCee_cc_Afb_240GeV",
    "FCCee_mumu_Afb_240GeV", "FCCee_sigmaHad_240GeV",
    "FCCee_tautau_Afb_240GeV",
    "FCCee_Rb_365GeV", "FCCee_Rc_365GeV",
    "FCCee_Rmu_365GeV", "FCCee_Rtau_365GeV",
    "FCCee_bb_Afb_365GeV", "FCCee_cc_Afb_365GeV",
    "FCCee_mumu_Afb_365GeV", "FCCee_sigmaHad_365GeV",
    "FCCee_tautau_Afb_365GeV",
    "FCCee_Wwidth", "FCCee_Zdata",
    "FCCee_zh_240GeV", "FCCee_zh_WW_240GeV", "FCCee_zh_ZZ_240GeV",
    "FCCee_zh_aZ_240GeV", "FCCee_zh_aa_240GeV", "FCCee_zh_tautau_240GeV",
    "FCCee_zh_365GeV", "FCCee_zh_WW_365GeV", "FCCee_zh_ZZ_365GeV",
    "FCCee_zh_aa_365GeV", "FCCee_zh_tautau_365GeV",
]


def generate_projections(model, out_dir: str, verbose: bool = True):
    """
    Generate BSM pseudo-data projections for the given model.

    Parameters
    ----------
    model   : instance with .eft_coefficients() -> Dict[str, float]
    out_dir : output directory for BSM yaml files
    verbose : print per-dataset coverage summary

    Returns
    -------
    coverage : dict  {dataset: {op: K_dot_c contribution}}
    """
    os.makedirs(out_dir, exist_ok=True)

    # Get Wilson coefficients at matching scale
    c_eft = model.eft_coefficients()
    ops   = list(c_eft.keys())

    if verbose:
        print(f"\nGenerating projections for {model}")
        print(f"  EFT coefficients at matching scale:")
        for op, c in c_eft.items():
            print(f"    {op:12s}: {c:+.6f} TeV^-2")
        print(f"  Output: {out_dir}\n")

    total_signal = {}    # dataset -> total |Σ K_j * c_j|
    op_coverage  = {}    # dataset -> list of ops with theory predictions
    n_covered    = {op: 0 for op in ops}

    for ds in DATASETS:
        # ── Load SM central values ──────────────────────────────────────────
        sm_path = f"{SM_DATA}/{ds}.yaml"
        if not os.path.exists(sm_path):
            if verbose:
                print(f"  WARNING: SM data not found: {ds}")
            continue

        sm_yaml = yaml.safe_load(open(sm_path))
        dc_sm   = sm_yaml["data_central"]
        scalar  = not isinstance(dc_sm, list)
        dc_sm   = [dc_sm] if scalar else list(dc_sm)
        n_obs   = len(dc_sm)

        # ── Load theory predictions ─────────────────────────────────────────
        th_path = f"{THEORY}/{ds}.json"
        if not os.path.exists(th_path):
            if verbose:
                print(f"  WARNING: Theory file not found: {ds}")
            # Copy SM data as-is (no signal)
            sm_out = dict(sm_yaml)
            with open(f"{out_dir}/{ds}.yaml", "w") as f:
                yaml.dump(sm_out, f, default_flow_style=False, allow_unicode=True)
            continue

        th = json.load(open(th_path))
        lo = th.get("LO", {})

        # ── Apply EFT shifts ────────────────────────────────────────────────
        delta     = np.zeros(n_obs)
        ops_used  = []

        for op, c in c_eft.items():
            if abs(c) < 1e-15:
                continue
            if op not in lo:
                continue

            k_op = lo[op]
            if isinstance(k_op, (int, float)):
                k_op = [k_op] * n_obs
            k_op = np.array(k_op)

            if len(k_op) != n_obs:
                if verbose:
                    print(f"  WARNING: {ds} {op}: K length {len(k_op)} != n_obs {n_obs}, skipping")
                continue

            contribution = c * k_op
            delta += contribution
            ops_used.append(op)
            n_covered[op] += 1

        # ── Write BSM yaml ──────────────────────────────────────────────────
        dc_bsm = (np.array(dc_sm) + delta).tolist()
        bsm_yaml = dict(sm_yaml)
        bsm_yaml["data_central"] = dc_bsm[0] if scalar else dc_bsm

        with open(f"{out_dir}/{ds}.yaml", "w") as f:
            yaml.dump(bsm_yaml, f, default_flow_style=False, allow_unicode=True)

        total_signal[ds] = float(np.sqrt(np.mean(delta**2)))  # RMS signal
        op_coverage[ds]  = ops_used

        if verbose and ops_used:
            rms = total_signal[ds]
            print(f"  {ds:<40s}  RMS_delta={rms:.4e}  ops={ops_used}")

    # ── Coverage report ────────────────────────────────────────────────────
    report_lines = [
        f"# Z' projection coverage report",
        f"# Model: {model}",
        f"# EFT coefficients: {c_eft}",
        f"#",
        f"# Operator coverage (how many datasets have theory predictions):",
    ]
    for op in ops:
        report_lines.append(f"#   {op:12s}: {n_covered[op]}/{len(DATASETS)} datasets")
    report_lines += [
        f"#",
        f"# Per-dataset signal size (RMS of delta vector):",
    ]
    for ds, rms in sorted(total_signal.items(), key=lambda x: -x[1]):
        report_lines.append(f"#   {ds:<42s}: RMS={rms:.4e}  ops={op_coverage.get(ds,[])}")

    with open(f"{out_dir}/coverage.txt", "w") as f:
        f.write("\n".join(report_lines) + "\n")

    if verbose:
        print(f"\n  Coverage report: {out_dir}/coverage.txt")
        print(f"  Operator dataset coverage:")
        for op in ops:
            print(f"    {op:12s}: {n_covered[op]} datasets")

    return op_coverage


def parse_args():
    p = argparse.ArgumentParser(description="Generate BSM projections for any UV model")
    p.add_argument("--model", choices=["wprime", "zprime"], required=True)
    p.add_argument("--tag",   default=None)
    # W' params
    p.add_argument("--gWH",  type=float, default=0.12)
    p.add_argument("--mWp",  type=float, default=1.0)
    p.add_argument("--gWLf", type=float, default=None)
    p.add_argument("--gWqf", type=float, default=None)
    # Z' params
    p.add_argument("--gZH",  type=float, default=0.12)
    p.add_argument("--gZl",  type=float, default=0.04)
    p.add_argument("--mZp",  type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    from models.wprime import WPrimeModel
    from models.zprime import ZPrimeModel

    args = parse_args()

    if args.model == "wprime":
        gWLf = args.gWLf if args.gWLf is not None else args.gWH / 3
        gWqf = args.gWqf if args.gWqf is not None else args.gWH / 3
        model = WPrimeModel(gWH=args.gWH, gWLf11=gWLf, gWLf22=gWLf,
                            gWLf33=gWLf, gWqf33=gWqf, mWp=args.mWp)
        default_tag = f"wprime_gwh{int(args.gWH*100):03d}_mwp{int(args.mWp*10):03d}"

    elif args.model == "zprime":
        model = ZPrimeModel(gZH=args.gZH, gZl=args.gZl, mZp=args.mZp)
        default_tag = f"zprime_gzh{int(args.gZH*100):03d}_mzp{int(args.mZp*10):03d}"

    tag     = args.tag or default_tag
    out_dir = str(PIPELINE / "projections" / tag)

    generate_projections(model, out_dir, verbose=True)
    print(f"\nProjections ready: {out_dir}")
    print(f"Next: python run_pipeline.py --model {args.model} --tag {tag} --proj {out_dir}")
