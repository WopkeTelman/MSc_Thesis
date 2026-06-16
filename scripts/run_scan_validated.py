#!/usr/bin/env python3
"""
run_scan_validated.py

Validates the analytic discovery region by running actual smefit A fits at
each scan grid point. Compares fit-based q vs analytic q.

For each (gWH, mWp) point in the existing scan:
  1. Reads existing BSM projection (fullpipeline/projections/wprime_scan_*)
  2. Runs smefit A (full SMEFT fit on BSM data)
  3. Computes q_fit = chi2_SM - chi2_best = chi2_SM + 2*max_loglikelihood
  4. Converts to sigma (ndof = Fisher rank)
  5. Plots analytic vs fit-based discovery region side by side

Output: fullpipeline/results/wprime_gwh012_mwp010_scan_validated/
Does NOT touch any existing scan files.

Usage:
    python run_scan_validated.py [--skip-existing]
"""

import os, sys, yaml, json, subprocess, argparse
import numpy as np
from pathlib import Path
from scipy.stats import chi2 as chi2_dist, norm as norm_dist
from scipy.linalg import block_diag

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))

DB      = "/data/theorie/wtelman/smefit_database"
SMEFIT  = "/data/theorie/wtelman/miniconda3/envs/smefit-dev/bin/smefit"
SM_DATA = f"{DB}/commondata_projections_L0"
THEORY  = f"{DB}/theory"

ENV = {**os.environ,
       "MPLCONFIGDIR": "/data/theorie/wtelman/.mplconfig",
       "XDG_CACHE_HOME": "/data/theorie/wtelman/.cache"}

RGE = {"init_scale": 1000.0, "obs_scale": "dynamic",
       "smeft_accuracy": "integrate", "yukawa": "top", "adm_QCD": False}

DS_NAMES = [
    "FCCee_ww_161GeV", "FCCee_ww_240GeV", "FCCee_ww_365GeV",
    "FCCee_Rb_240GeV", "FCCee_Rc_240GeV", "FCCee_Rmu_240GeV", "FCCee_Rtau_240GeV",
    "FCCee_bb_Afb_240GeV", "FCCee_cc_Afb_240GeV", "FCCee_mumu_Afb_240GeV",
    "FCCee_sigmaHad_240GeV", "FCCee_tautau_Afb_240GeV",
    "FCCee_Rb_365GeV", "FCCee_Rc_365GeV", "FCCee_Rmu_365GeV", "FCCee_Rtau_365GeV",
    "FCCee_bb_Afb_365GeV", "FCCee_cc_Afb_365GeV", "FCCee_mumu_Afb_365GeV",
    "FCCee_sigmaHad_365GeV", "FCCee_tautau_Afb_365GeV",
    "FCCee_Wwidth", "FCCee_Zdata",
    "FCCee_zh_240GeV", "FCCee_zh_WW_240GeV", "FCCee_zh_ZZ_240GeV",
    "FCCee_zh_aZ_240GeV", "FCCee_zh_aa_240GeV", "FCCee_zh_tautau_240GeV",
    "FCCee_zh_365GeV", "FCCee_zh_WW_365GeV", "FCCee_zh_ZZ_365GeV",
    "FCCee_zh_aa_365GeV", "FCCee_zh_tautau_365GeV",
]
DATASETS = [{"name": ds, "order": "NLO_EW" if "zh" in ds else "LO"} for ds in DS_NAMES]


def q_to_sigma(q, k):
    if q <= 0:
        return 0.0
    p = chi2_dist.sf(q, df=k)
    return float(norm_dist.isf(p / 2)) if p > 0 else 99.0


def compute_fisher_rank(ops):
    """Rank of Fisher matrix F = K^T C^{-1} K (number of constrained directions)."""
    K_blocks, Ci_blocks = [], []
    for ds in DS_NAMES:
        sm_path = f"{SM_DATA}/{ds}.yaml"
        th_path = f"{THEORY}/{ds}.json"
        if not os.path.exists(sm_path) or not os.path.exists(th_path):
            continue
        sm = yaml.safe_load(open(sm_path))
        th = json.load(open(th_path))
        dc = sm["data_central"]
        dc = [dc] if not isinstance(dc, list) else list(dc)
        n  = len(dc)
        stat = sm.get("statistical_error", [0.0] * n)
        stat = list(stat) if isinstance(stat, list) else [stat] * n
        C = np.diag([float(e)**2 if float(e) > 0 else (0.01 * abs(float(d)))**2
                     for e, d in zip(stat, dc)])
        sys_mat = sm.get("systematics", None)
        if sys_mat is not None:
            S = np.array(sys_mat, dtype=float)
            if S.ndim == 2 and S.shape[0] == n:
                C += S @ S.T
        th_cov = th.get("theory_cov", None)
        if th_cov is not None:
            T = np.array(th_cov, dtype=float)
            if T.shape == (n, n):
                C += T
        lo   = th.get("LO", {})
        K_ds = np.zeros((n, len(ops)))
        for j, op in enumerate(ops):
            k = lo.get(op, 0.0)
            K_ds[:, j] = [float(k[i]) if isinstance(k, list) and i < len(k)
                          else float(k) for i in range(n)]
        K_blocks.append(K_ds)
        Ci_blocks.append(np.linalg.inv(C))
    K  = np.vstack(K_blocks)
    Ci = block_diag(*Ci_blocks)
    return int(np.linalg.matrix_rank(K.T @ Ci @ K))


def parse_discovery_table(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            rows.append({
                "gWH":                  float(vals[0]),
                "mWp":                  float(vals[1]),
                "chi2_SM":              float(vals[6]),
                "sigma_best1_analytic": float(vals[2]),
                "sigma_full_analytic":  float(vals[4]),
                "sigma_uv_analytic":    float(vals[5]),
            })
    return rows


def make_smeft_runcard(result_id, proj_dir, ops, truth):
    return {
        "result_ID":         result_id,
        "result_path":       None,
        "data_path":         proj_dir,
        "theory_path":       THEORY,
        "use_quad":          False,
        "use_t0":            True,
        "use_theory_covmat": True,
        "uv_couplings":      False,
        "n_samples":         10000,
        "datasets":          DATASETS,
        "coefficients": {
            op: {"min": -max(0.01, min(1.0, 20 * abs(truth.get(op, 0.01)))),
                 "max":  max(0.01, min(1.0, 20 * abs(truth.get(op, 0.01))))}
            for op in ops
        },
        "rge": RGE,
    }


def plot_results(results, plt_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    gWH_vals = sorted(set(r["gWH"] for r in results))
    mWp_vals = sorted(set(r["mWp"] for r in results))
    nc, nm   = len(gWH_vals), len(mWp_vals)

    sigma_fit_grid      = np.full((nc, nm), np.nan)
    sigma_analytic_grid = np.full((nc, nm), np.nan)

    for r in results:
        ic = gWH_vals.index(r["gWH"])
        im = mWp_vals.index(r["mWp"])
        sigma_fit_grid[ic, im]      = r["sigma_fit"]
        sigma_analytic_grid[ic, im] = r["sigma_full_analytic"]

    mass_arr     = np.array(mWp_vals)
    coupling_arr = np.array(gWH_vals)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, sigma_grid, title in [
        (axes[0], sigma_analytic_grid, "Analytic (profile likelihood)"),
        (axes[1], sigma_fit_grid,      "Fit-based (smefit A)"),
    ]:
        grid = np.nan_to_num(sigma_grid, nan=0.0)
        ax.contourf(mass_arr, coupling_arr, grid,
                    levels=[5.0, 1e4], colors=["#ADD8E6"], alpha=0.5)
        ax.contourf(mass_arr, coupling_arr, grid,
                    levels=[0, 5.0], colors=["#DCDCDC"], alpha=0.5)
        try:
            cs5 = ax.contour(mass_arr, coupling_arr, grid,
                             levels=[5.0], colors=["#1f77b4"], linewidths=[2.5])
            ax.clabel(cs5, fmt=r"$5\sigma$", fontsize=10, inline=True)
        except Exception:
            pass
        try:
            cs3 = ax.contour(mass_arr, coupling_arr, grid,
                             levels=[3.0], colors=["#1f77b4"], linewidths=[1.5],
                             linestyles=["--"])
            ax.clabel(cs3, fmt=r"$3\sigma$", fontsize=10, inline=True)
        except Exception:
            pass
        ax.set_xlabel(r"$m_{W'}$ [TeV]", fontsize=12)
        ax.set_ylabel(r"$|g_{WH}|$", fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.grid(True, which="major", alpha=0.2, lw=0.5)

    fig.suptitle(r"FCC-ee $W'$ discovery reach: analytic vs fit-validated (full SMEFT)",
                 fontsize=12)
    plt.tight_layout()
    out_png = f"{plt_dir}/discovery_region_validated.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out_png}")


def main():
    p = argparse.ArgumentParser(description="Validate discovery region with smefit A fits")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip scan points where fit_results.json already exists")
    args = p.parse_args()

    from models.wprime import WPrimeModel

    # Output — never touches existing scan
    out_dir  = str(PIPELINE / "results" / "wprime_gwh012_mwp010_scan_validated")
    fits_dir = f"{out_dir}/fits"
    rc_dir   = f"{out_dir}/runcards"
    plt_dir  = f"{out_dir}/plots"
    for d in [fits_dir, rc_dir, plt_dir]:
        os.makedirs(d, exist_ok=True)

    # Read scan grid from existing discovery table
    table_path = str(PIPELINE / "results" / "wprime_gwh012_mwp010_scan" / "discovery_table.txt")
    rows = parse_discovery_table(table_path)
    print(f"\n  Scan grid: {len(rows)} points")

    # Fisher rank (same for all points — operator set is fixed)
    dummy = WPrimeModel(gWH=0.12, gWLf11=0.04, gWLf22=0.04, gWLf33=0.04, gWqf33=0.04, mWp=1.0)
    ops   = dummy.OPERATORS
    rank  = compute_fisher_rank(ops)
    print(f"  Fisher rank = {rank}  (ndof for full SMEFT sigma)")
    print(f"  Output: {out_dir}\n")

    results = []

    for i, row in enumerate(rows):
        gWH = row["gWH"]
        mWp = row["mWp"]
        chi2_sm = row["chi2_SM"]

        scan_tag = f"wprime_scan_gWH{int(gWH * 100):03d}_mWp{int(mWp * 10):03d}"
        proj_dir = str(PIPELINE / "projections" / scan_tag)

        if not os.path.exists(proj_dir) or not os.listdir(proj_dir):
            print(f"  [{i+1:3d}/{len(rows)}] {scan_tag}: no projection — skipping")
            continue

        model  = WPrimeModel(gWH=gWH, gWLf11=gWH/3, gWLf22=gWH/3,
                             gWLf33=gWH/3, gWqf33=gWH/3, mWp=mWp)
        truth  = model.eft_coefficients()

        result_id = f"validated_{scan_tag}"
        rc_path   = f"{rc_dir}/{result_id}.yaml"
        fit_path  = f"{fits_dir}/{result_id}/fit_results.json"

        # Write runcard
        rc = make_smeft_runcard(result_id, proj_dir, ops, truth)
        rc["result_path"] = fits_dir
        with open(rc_path, "w") as f:
            yaml.dump(rc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        # Run smefit A
        if os.path.exists(fit_path) and args.skip_existing:
            print(f"  [{i+1:3d}/{len(rows)}] {scan_tag}: fit exists — skipping")
        else:
            r = subprocess.run([SMEFIT, "A", rc_path], env=ENV,
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  [{i+1:3d}/{len(rows)}] {scan_tag}: FAILED")
                continue

        if not os.path.exists(fit_path):
            print(f"  [{i+1:3d}/{len(rows)}] {scan_tag}: no fit_results.json — skipping")
            continue

        res          = json.load(open(fit_path))
        max_loglike  = res["max_loglikelihood"]
        chi2_best    = -2 * max_loglike
        q_fit        = chi2_sm - chi2_best
        sigma_fit    = q_to_sigma(q_fit, rank)

        print(f"  [{i+1:3d}/{len(rows)}] gWH={gWH:.3f} mWp={mWp:.2f}  "
              f"chi2_SM={chi2_sm:6.2f}  chi2_best={chi2_best:6.3f}  "
              f"q={q_fit:6.2f}  sigma_fit={sigma_fit:.2f}σ  "
              f"(analytic={row['sigma_full_analytic']:.2f}σ)")

        results.append({**row, "chi2_best": chi2_best,
                        "q_fit": q_fit, "sigma_fit": sigma_fit})

    if not results:
        print("No results to plot.")
        return

    # Save table
    tbl_path = f"{out_dir}/validated_table.txt"
    with open(tbl_path, "w") as f:
        f.write("# Validated discovery scan — smefit A at each grid point\n")
        f.write(f"# {'gWH':>8}  {'mWp':>6}  {'chi2_SM':>9}  {'chi2_best':>9}  "
                f"{'q_fit':>7}  {'sigma_fit':>9}  {'sigma_analytic':>14}\n")
        for r in results:
            f.write(f"  {r['gWH']:>8.4f}  {r['mWp']:>6.2f}  {r['chi2_SM']:>9.2f}  "
                    f"{r['chi2_best']:>9.4f}  {r['q_fit']:>7.2f}  "
                    f"{r['sigma_fit']:>9.3f}  {r['sigma_full_analytic']:>14.3f}\n")
    print(f"\n  Saved: {tbl_path}")

    plot_results(results, plt_dir)

    print(f"\n  Done. {len(results)}/{len(rows)} points completed.")


if __name__ == "__main__":
    main()
