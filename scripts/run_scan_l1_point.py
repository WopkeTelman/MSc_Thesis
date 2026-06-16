#!/usr/bin/env python3
"""
scripts/run_scan_l1_point.py

Condor worker: generates all n_reps L1 projections for ONE grid point.
Called by run_scan_l1_condor.sh with --job $(Process).

After all 99 condor jobs finish, run:
    python scripts/run_pipeline.py --model wprime_constrained \
        --gWH 0.5 --mWp 6.0 --scan-l1 --n-l1-reps 50
That step is fast (projections already exist) and produces the plots.
"""
import sys, os, argparse
import numpy as np
from pathlib import Path

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))

from scripts.run_pipeline import generate_projections, DATASETS, DS_NAMES
from models.wprime_constrained import WPrimeConstrainedModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job",    type=int, required=True, help="Grid point index (0-98)")
    p.add_argument("--gWH",   type=float, default=0.5)
    p.add_argument("--mWp",   type=float, default=6.0)
    p.add_argument("--n-reps", type=int,  default=50)
    args = p.parse_args()

    g0, m0 = args.gWH, args.mWp
    coupling_grid = sorted(set(round(v, 4)
        for v in np.linspace(g0 * 0.25, g0 * 3.0, 9).tolist()))
    mass_grid     = sorted(set(round(v, 2)
        for v in np.linspace(m0 * 0.1,  m0 * 3.0, 11).tolist()))

    nm     = len(mass_grid)
    ic     = args.job // nm
    im     = args.job %  nm
    g_val  = coupling_grid[ic]
    m_val  = mass_grid[im]

    print(f"Job {args.job:3d}/{len(coupling_grid)*nm - 1}  "
          f"gWH={g_val:.4f}  mWp={m_val:.2f} TeV  n_reps={args.n_reps}")

    # build model at this grid point (scale fermion couplings with gWH)
    gWLf  = (g0 / 3) * (g_val / g0)
    model = WPrimeConstrainedModel(
        gWH=g_val, gWLf11=gWLf, gWLf22=gWLf, gWLf33=gWLf, gWqf33=gWLf, mWp=m_val
    )

    # same scan_tag as run_scan_l1 in run_pipeline.py
    scan_tag  = (f"wprime_constrained_l1scan_gWH{int(g_val*100):03d}"
                 f"_mWp{int(m_val*10):03d}")
    point_dir = PIPELINE / "projections" / scan_tag

    def _has_datasets(d):
        return (os.path.isdir(d) and
                any(f.endswith(".yaml") and not f.startswith("proj_")
                    for f in os.listdir(d)))

    n_done = 0
    for r in range(args.n_reps):
        rep_dir = str(point_dir / f"rep{r:03d}")
        if _has_datasets(rep_dir):
            print(f"  rep{r:03d}: already exists — skipping")
            n_done += 1
            continue
        print(f"  rep{r:03d}: generating L1 projection ...", flush=True)
        generate_projections(model, rep_dir, noise_level="L1")
        n_done += 1

    print(f"\nDone: {n_done}/{args.n_reps} replicas for "
          f"gWH={g_val:.4f}, mWp={m_val:.2f} TeV")


if __name__ == "__main__":
    main()
