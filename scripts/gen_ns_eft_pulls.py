"""
scripts/gen_ns_eft_pulls.py

Generate Metric 2c (NS EFT coefficient pull) plots for any existing L1 fit directory.
Reads fit_results.json files, computes per-operator NS posterior pulls, and saves:
  metric2c_ns_eft_pull_distribution.png/pdf
  metric2c_ns_eft_pull_bars.png/pdf

Usage:
    python scripts/gen_ns_eft_pulls.py --gWH 0.12 --mWp 3.0 --model wprime
    python scripts/gen_ns_eft_pulls.py --gWH 0.12 --mWp 3.0 --model wprime_universal
"""

import sys, argparse
from pathlib import Path

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "scripts"))

from wprime_l1_analysis import collect_ns_eft_pulls, plot_metric2_ns_eft_pulls


def get_model(args):
    gf = args.gWH / 3.0
    if args.model == "wprime_universal":
        from models.wprime_universal import WPrimeUniversalModel
        return WPrimeUniversalModel(gWH=args.gWH, mWp=args.mWp)
    elif args.model == "wprime":
        from models.wprime import WPrimeModel
        return WPrimeModel(gWH=args.gWH, gWLf11=gf, gWLf22=gf,
                           gWLf33=gf, gWqf33=gf, mWp=args.mWp)
    elif args.model == "wprime_constrained":
        from models.wprime_constrained import WPrimeConstrainedModel
        return WPrimeConstrainedModel(gWH=args.gWH, gWLf11=gf, gWLf22=gf,
                                      gWLf33=gf, gWqf33=gf, mWp=args.mWp)
    else:
        raise ValueError(f"Unknown model: {args.model}")


def get_out_dir(args):
    g  = int(round(args.gWH * 100))
    m  = int(round(args.mWp * 10))
    if args.model == "wprime_universal":
        return PIPELINE / "results" / f"wprime_l1u_gwh{g:03d}_mwp{m:03d}"
    else:
        return PIPELINE / "results" / f"wprime_l1_gwh{g:03d}_mwp{m:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gWH",   type=float, required=True)
    ap.add_argument("--mWp",   type=float, required=True)
    ap.add_argument("--model", default="wprime",
                    choices=["wprime", "wprime_constrained", "wprime_universal"])
    args = ap.parse_args()

    model   = get_model(args)
    out_dir = get_out_dir(args)

    if not out_dir.exists():
        print(f"ERROR: output directory not found: {out_dir}")
        sys.exit(1)

    print(f"Model  : {model!r}")
    print(f"Out dir: {out_dir}")
    print(f"Computing NS EFT pulls...")

    suffix = "" if args.model == "wprime" else f"_{args.model.replace('wprime_', '')}"
    ns_eft_pulls = collect_ns_eft_pulls(None, out_dir, model)
    plot_metric2_ns_eft_pulls(ns_eft_pulls, model, out_dir, suffix=suffix)

    print(f"Saved: {out_dir}/metric2c_ns_eft_pull_distribution{suffix}.png")
    print(f"Saved: {out_dir}/metric2c_ns_eft_pull_bars{suffix}.png")


if __name__ == "__main__":
    main()
