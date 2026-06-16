#!/usr/bin/env python3
"""
run_uv_validated.py

Validates the analytic UV discovery reach by running a single smefit NS fit
at each (gWH, mWp) grid point on the Asimov (noiseless) BSM dataset.

For each grid point:
  1. Uses existing Asimov projections from the discovery scan
     (fullpipeline/projections/wprime_scan_*)
  2. Runs smefit NS with UV couplings free (tight priors ±PRIOR_SCALE × |truth|)
  3. Computes q_NS = chi2_SM + 2 * max_loglikelihood
  4. Compares to analytic q_analytic = c^T F c

This directly answers: does NS recover the analytic significance on noiseless data?
If yes: the analytic result is validated. The gap seen with noisy toys was purely
statistical scatter, not a systematic bias.

Output: results/uv_validated/
    fits/    -- NS fit results
    runcards/ -- NS runcards
    plots/   -- NS vs analytic comparison
    tables/  -- uv_validated_table.txt

Usage:
    python run_uv_validated.py [--prior-scale 3.0] [--skip-existing]
"""

import os, sys, yaml, json, subprocess, argparse
import numpy as np
from pathlib import Path
from scipy.stats import chi2 as chi2_dist, norm as norm_dist
from multiprocessing import Pool
import time

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))

DB      = "/data/theorie/wtelman/smefit_database"
SMEFIT  = "/data/theorie/wtelman/miniconda3/envs/smefit-dev/bin/smefit"
THEORY  = f"{DB}/theory"

ENV = {**os.environ,
       "MPLCONFIGDIR": "/data/theorie/wtelman/.mplconfig",
       "XDG_CACHE_HOME": "/data/theorie/wtelman/.cache"}

DATASETS = [
    {"name": "FCCee_ww_161GeV",         "order": "LO"},
    {"name": "FCCee_ww_240GeV",         "order": "LO"},
    {"name": "FCCee_ww_365GeV",         "order": "LO"},
    {"name": "FCCee_Rb_240GeV",         "order": "LO"},
    {"name": "FCCee_Rc_240GeV",         "order": "LO"},
    {"name": "FCCee_Rmu_240GeV",        "order": "LO"},
    {"name": "FCCee_Rtau_240GeV",       "order": "LO"},
    {"name": "FCCee_bb_Afb_240GeV",     "order": "LO"},
    {"name": "FCCee_cc_Afb_240GeV",     "order": "LO"},
    {"name": "FCCee_mumu_Afb_240GeV",   "order": "LO"},
    {"name": "FCCee_sigmaHad_240GeV",   "order": "LO"},
    {"name": "FCCee_tautau_Afb_240GeV", "order": "LO"},
    {"name": "FCCee_Rb_365GeV",         "order": "LO"},
    {"name": "FCCee_Rc_365GeV",         "order": "LO"},
    {"name": "FCCee_Rmu_365GeV",        "order": "LO"},
    {"name": "FCCee_Rtau_365GeV",       "order": "LO"},
    {"name": "FCCee_bb_Afb_365GeV",     "order": "LO"},
    {"name": "FCCee_cc_Afb_365GeV",     "order": "LO"},
    {"name": "FCCee_mumu_Afb_365GeV",   "order": "LO"},
    {"name": "FCCee_sigmaHad_365GeV",   "order": "LO"},
    {"name": "FCCee_tautau_Afb_365GeV", "order": "LO"},
    {"name": "FCCee_Wwidth",            "order": "LO"},
    {"name": "FCCee_Zdata",             "order": "LO"},
    {"name": "FCCee_zh_240GeV",         "order": "NLO_EW"},
    {"name": "FCCee_zh_WW_240GeV",      "order": "NLO_EW"},
    {"name": "FCCee_zh_ZZ_240GeV",      "order": "NLO_EW"},
    {"name": "FCCee_zh_aZ_240GeV",      "order": "NLO_EW"},
    {"name": "FCCee_zh_aa_240GeV",      "order": "NLO_EW"},
    {"name": "FCCee_zh_tautau_240GeV",  "order": "NLO_EW"},
    {"name": "FCCee_zh_365GeV",         "order": "NLO_EW"},
    {"name": "FCCee_zh_WW_365GeV",      "order": "NLO_EW"},
    {"name": "FCCee_zh_ZZ_365GeV",      "order": "NLO_EW"},
    {"name": "FCCee_zh_aa_365GeV",      "order": "NLO_EW"},
    {"name": "FCCee_zh_tautau_365GeV",  "order": "NLO_EW"},
]

RGE = {"init_scale": 1000.0, "obs_scale": "dynamic",
       "smeft_accuracy": "integrate", "yukawa": "top", "adm_QCD": False}

FERMION_RATIO = 1.0 / 3.0

SMEFIT_SRC = "/data/theorie/wtelman/smefit_release/src"

# 17 SMEFT operators present in smefit theory files for these datasets
# (OQQ1, OQQ8, OQl33, OQl3M, Oll2222, Oll2233, Oll2332, Oll3333 are absent)
SMEFT_OPS_17 = [
    "O3pQ3", "O3pl1", "O3pl2", "O3pl3",
    "OQl13", "OQl1M",
    "Obp",
    "Oll1111", "Oll1122", "Oll1133", "Oll1221", "Oll1331",
    "Op", "OpBox", "OpQM", "Otap", "Otp",
]

# Constrain structure for the 17 operators: copied from example runcard.
# Format: each term is {coupling_name: [coeff, power]}.
# Total: c_op = product_i (coeff_i * coupling_i ^ power_i)
# NOTE: UV couplings here are "effective" = physical / mWp, so that e.g.
#   OpBox = -0.375 * gWH_eff^2 = -0.375 * (gWH/mWp)^2  (no explicit mWp^2)
## Constrain format: c_op = sum_i prod_j (a_j * coupling_j^n_j)
## Each LIST ITEM is one PRODUCT TERM (a dict with multiple keys → multiply all factors).
## Multiple list items → sum of products.
## For a single product of N couplings: put all N factors in ONE dict.
SMEFT_CONSTRAIN_17 = {
    # single-coupling products (one-key dicts are fine as-is)
    "OpBox":   [{"gWH": [-0.375,                  2]}],
    "Obp":     [{"gWH": [-0.006002165432052166,    2]}],
    "Otp":     [{"gWH": [-0.2480703588615627,      2]}],
    "Otap":    [{"gWH": [-0.0025559460452279554,   2]}],
    "Op":      [{"gWH": [-0.12938347743146458,     2]}],
    "Oll1111": [{"gWLf11": [-0.125, 2]}],
    # two-coupling products: ALL factors in a SINGLE dict
    "OpQM":    [{"gWH": [0.25,  1], "gWqf33": [1.0, 1]}],
    "O3pQ3":   [{"gWH": [-0.25, 1], "gWqf33": [1.0, 1]}],
    "O3pl1":   [{"gWH": [-0.25, 1], "gWLf11": [1.0, 1]}],
    "O3pl2":   [{"gWH": [-0.25, 1], "gWLf22": [1.0, 1]}],
    "O3pl3":   [{"gWH": [-0.25, 1], "gWLf33": [1.0, 1]}],
    "Oll1221": [{"gWLf11": [-0.25, 1], "gWLf22": [1.0, 1]}],
    "OQl13":   [{"gWLf11": [-1.0,  1], "gWqf33": [1.0, 1]}],
    "OQl1M":   [{"gWLf11": [ 1.0,  1], "gWqf33": [1.0, 1]}],
    "Oll1122": [{"gWLf11": [0.125, 1], "gWLf22": [1.0, 1]}],
    "Oll1133": [{"gWLf11": [0.125, 1], "gWLf33": [1.0, 1]}],
    "Oll1331": [{"gWLf11": [0.25,  1], "gWLf33": [1.0, 1]}],
}


def q_to_sigma(q, ndof=5):
    if q <= 0 or ndof <= 0:
        return 0.0
    p = chi2_dist.sf(q, df=ndof)
    if p <= 0:
        return 99.0
    return max(0.0, float(norm_dist.isf(p)))


def make_uv_runcard(result_id, proj_dir, fits_dir, gWH, mWp, prior_scale):
    """Build runcard with proper UV->SMEFT constrain mappings.

    The UV couplings in the runcard are 'effective' (= physical / mWp), so
    that e.g. OpBox = -0.375 * gWH_eff^2 with no explicit mWp^2 factor.
    SMEFT operators are constrained (derived) quantities; UV couplings are free.
    """
    f = gWH * FERMION_RATIO
    # Effective UV couplings at truth point
    truth_eff = {
        "gWH":    gWH / mWp,
        "gWLf11": f   / mWp,
        "gWLf22": f   / mWp,
        "gWLf33": f   / mWp,
        "gWqf33": f   / mWp,
    }

    def prior(val):
        lo = max(1e-6, val / prior_scale)
        hi = val * prior_scale
        return {"min": lo, "max": hi}

    # SMEFT operators: constrained from UV couplings
    coefficients = {}
    for op, terms in SMEFT_CONSTRAIN_17.items():
        coefficients[op] = {"constrain": terms}
    # UV couplings: free parameters with tight priors around truth
    for uv_name, val in truth_eff.items():
        coefficients[uv_name] = prior(val)

    return {
        "result_ID":           result_id,
        "result_path":         fits_dir,
        "data_path":           proj_dir,
        "theory_path":         THEORY,
        "use_quad":            False,
        "use_t0":              False,
        "use_theory_covmat":   True,
        "uv_couplings":        True,
        "n_samples":           5000,
        "nlive":               300,
        "lepsilon":            0.01,
        "target_evidence_unc": 0.1,
        "target_post_unc":     0.1,
        "frac_remain":         0.01,
        "datasets":            DATASETS,
        "coefficients":        coefficients,
        "rge":                 RGE,
    }


def compute_chi2_sm_smefit(point_proj):
    """Compute chi2_SM = (Asimov data - SM theory)^T C^{-1} (same) using smefit.

    This uses the 17-operator smefit basis.  At c=0 (SM hypothesis), chi2 is
    purely a function of the stored Asimov data and the SM theory predictions —
    no EFT K matrix is needed, so the operator list doesn't affect the result.
    """
    if SMEFIT_SRC not in sys.path:
        sys.path.insert(0, SMEFIT_SRC)
    from smefit.loader import load_datasets
    ops = {op: {} for op in SMEFT_OPS_17}
    loaded = load_datasets(
        point_proj, DATASETS, ops,
        False,   # use_quad
        True,    # use_theory_covmat
        False,   # use_t0
        False,   # use_multiplicative_prescription
        "LO",    # default_order
        THEORY,  # theory_path
        None,    # rot_to_fit_basis
        False,   # has_uv_couplings
        False,   # has_external_chi2
    )
    diff = loaded.Commondata - loaded.SMTheory
    return float(diff @ loaded.InvCovMat @ diff)


def parse_discovery_table(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            rows.append({
                "gWH":              float(vals[0]),
                "mWp":              float(vals[1]),
                "chi2_SM":          float(vals[6]),
                "sigma_uv_analytic": float(vals[5]),
            })
    return rows


def run_one(args):
    (row, proj_dir, fits_dir, rc_dir, prior_scale, skip_existing) = args

    gWH     = row["gWH"]
    mWp     = row["mWp"]
    chi2_sm = row["chi2_SM_smefit"]   # from smefit 17-op basis, same as NS uses

    scan_tag  = f"wprime_scan_gWH{int(gWH * 100):03d}_mWp{int(mWp * 10):03d}"
    result_id = f"uv_validated_v2_{scan_tag}"
    rc_path   = f"{rc_dir}/{result_id}.yaml"
    fit_path  = f"{fits_dir}/{result_id}/fit_results.json"
    point_proj = f"{proj_dir}/{scan_tag}"

    if not os.path.exists(point_proj) or not os.listdir(point_proj):
        print(f"  SKIP {scan_tag}: no projection")
        return None

    # Write runcard
    rc = make_uv_runcard(result_id, point_proj, fits_dir, gWH, mWp, prior_scale)
    with open(rc_path, "w") as f:
        yaml.dump(rc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    if os.path.exists(fit_path) and skip_existing:
        print(f"  CACHED {scan_tag}")
    else:
        t0 = time.time()
        r  = subprocess.run([SMEFIT, "NS", rc_path], env=ENV,
                            capture_output=True, text=True, timeout=1800)
        dt = time.time() - t0
        if r.returncode != 0 or not os.path.exists(fit_path):
            print(f"  FAILED {scan_tag}  ({dt:.0f}s)")
            return None

    res    = json.load(open(fit_path))
    max_ll = res["max_loglikelihood"]
    q_ns   = chi2_sm + 2 * max_ll
    sigma_ns = q_to_sigma(q_ns, ndof=5)

    print(f"  {scan_tag}  chi2_SM={chi2_sm:7.3f}  "
          f"max_ll={max_ll:8.3f}  q_NS={q_ns:7.3f}  "
          f"sigma_NS={sigma_ns:.2f}  "
          f"(analytic_table={row['sigma_uv_analytic']:.2f})")

    return dict(gWH=gWH, mWp=mWp,
                chi2_sm=chi2_sm,
                chi2_sm_table=row["chi2_SM"],
                max_ll=max_ll, q_ns=q_ns, sigma_ns=sigma_ns,
                sigma_analytic=row["sigma_uv_analytic"])


def plot_results(results, plt_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D

    gWH_vals = sorted(set(r["gWH"] for r in results))
    mWp_vals = sorted(set(r["mWp"] for r in results))
    nc, nm   = len(gWH_vals), len(mWp_vals)
    mass_arr     = np.array(mWp_vals)
    coupling_arr = np.array(gWH_vals)

    def make_grid(key):
        g = np.full((nc, nm), np.nan)
        for r in results:
            if r["gWH"] in gWH_vals and r["mWp"] in mWp_vals:
                g[gWH_vals.index(r["gWH"]), mWp_vals.index(r["mWp"])] = r[key]
        return np.clip(np.nan_to_num(g, nan=0.0), 0, 50)

    grid_ns       = make_grid("sigma_ns")
    grid_analytic = make_grid("sigma_analytic")

    # ── Figure 1: side-by-side contour maps ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, grid, title in [
        (axes[0], grid_analytic, "Analytic UV (profile likelihood)"),
        (axes[1], grid_ns,       "NS Asimov UV (smefit NS)"),
    ]:
        ax.contourf(mass_arr, coupling_arr, grid,
                    levels=[5.0, 51], colors=["#ADD8E6"], alpha=0.5)
        ax.contourf(mass_arr, coupling_arr, grid,
                    levels=[0, 5.0], colors=["#DCDCDC"], alpha=0.5)
        for lev, ls, lw, lbl in [(5.0, "-", 2.5, r"$5\sigma$"),
                                  (3.0, "--", 1.5, r"$3\sigma$")]:
            try:
                cs = ax.contour(mass_arr, coupling_arr, grid,
                                levels=[lev], colors=["#1f77b4"],
                                linewidths=[lw], linestyles=[ls])
                ax.clabel(cs, fmt=lbl, fontsize=9, inline=True)
            except Exception:
                pass
        ax.set_xlabel(r"$m_{W'}$ [TeV]", fontsize=12)
        ax.set_ylabel(r"$g_{WH}$", fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.grid(True, which="major", alpha=0.2, lw=0.5)

    fig.suptitle(r"UV discovery reach validation: analytic vs NS (Asimov data)",
                 fontsize=12)
    plt.tight_layout()
    out1 = f"{plt_dir}/uv_validated_sidebyside.png"
    plt.savefig(out1, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out1}")

    # ── Figure 2: overlay — NS vs analytic 5sigma contour ────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(mass_arr[0], mass_arr[-1])
    ax.set_ylim(coupling_arr[0], coupling_arr[-1])

    ax.contourf(mass_arr, coupling_arr, grid_ns,
                levels=[5.0, 51], colors=["#ADD8E6"], alpha=0.35)
    ax.contourf(mass_arr, coupling_arr, grid_ns,
                levels=[0, 5.0], colors=["#DCDCDC"], alpha=0.35)

    legend_els = []
    for grid, col, ls, lw, lbl in [
        (grid_ns,       "#1f77b4", "-",  2.5, r"NS Asimov ($5\sigma$)"),
        (grid_analytic, "#d62728", "--", 2.0, r"Analytic ($5\sigma$)"),
    ]:
        try:
            cs = ax.contour(mass_arr, coupling_arr, grid,
                            levels=[5.0], colors=[col],
                            linewidths=[lw], linestyles=[ls])
            ax.clabel(cs, fmt=r"$5\sigma$", fontsize=9, inline=True)
        except Exception:
            pass
        legend_els.append(Line2D([0],[0], color=col, lw=lw, ls=ls, label=lbl))

    ax.legend(handles=legend_els, fontsize=11, loc="upper right", framealpha=0.9)
    ax.set_xlabel(r"$m_{W'}$ [TeV]", fontsize=13)
    ax.set_ylabel(r"$g_{WH}$", fontsize=13)
    ax.set_title(r"UV discovery validation: NS vs analytic (Asimov)", fontsize=12)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(True, which="major", alpha=0.2, lw=0.5)
    plt.tight_layout()
    out2 = f"{plt_dir}/uv_validated_overlay.png"
    plt.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out2}")

    # ── Figure 3: scatter plot — sigma_NS vs sigma_analytic ──────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    xs = [r["sigma_analytic"] for r in results]
    ys = [r["sigma_ns"]       for r in results]
    ax.scatter(xs, ys, s=20, alpha=0.6, color="#1f77b4")
    lim = max(max(xs), max(ys)) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x (perfect agreement)")
    ax.set_xlabel(r"$\sigma$ analytic (UV profile)", fontsize=12)
    ax.set_ylabel(r"$\sigma$ NS (Asimov)", fontsize=12)
    ax.set_title("NS vs analytic: point-by-point comparison", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out3 = f"{plt_dir}/uv_validated_scatter.png"
    plt.savefig(out3, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out3}")


def main():
    p = argparse.ArgumentParser(
        description="Validate UV discovery reach with Asimov NS fits")
    p.add_argument("--prior-scale",   type=float, default=3.0)
    p.add_argument("--n-cores",       type=int,   default=60)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    out_dir  = str(PIPELINE / "results" / "uv_validated")
    fits_dir = f"{out_dir}/fits"
    rc_dir   = f"{out_dir}/runcards"
    plt_dir  = f"{out_dir}/plots"
    tbl_dir  = f"{out_dir}/tables"
    proj_dir = str(PIPELINE / "projections")
    for d in [fits_dir, rc_dir, plt_dir, tbl_dir]:
        os.makedirs(d, exist_ok=True)

    # Read scan grid from existing discovery table
    table_path = str(PIPELINE / "results" / "wprime_gwh012_mwp010_scan" /
                     "discovery_table.txt")
    rows = parse_discovery_table(table_path)
    print(f"\n  Scan grid: {len(rows)} points")
    print(f"  Prior scale: ±{args.prior_scale} × |truth|")
    print(f"  N cores: {args.n_cores}")
    print(f"  Output: {out_dir}\n")

    # Precompute chi2_SM_smefit for each point (sequential, fast).
    # This uses smefit's 17-op basis on the Asimov data — the same basis
    # the NS fit uses, so chi2_SM is consistent with chi2_UV_best.
    print("  Precomputing chi2_SM_smefit for all grid points...")
    for row in rows:
        scan_tag   = (f"wprime_scan_gWH{int(row['gWH']*100):03d}"
                      f"_mWp{int(row['mWp']*10):03d}")
        point_proj = f"{proj_dir}/{scan_tag}"
        if os.path.exists(point_proj) and os.listdir(point_proj):
            chi2_sm_s = compute_chi2_sm_smefit(point_proj)
        else:
            chi2_sm_s = float("nan")
        row["chi2_SM_smefit"] = chi2_sm_s
        print(f"    {scan_tag}  chi2_SM_table={row['chi2_SM']:.3f}"
              f"  chi2_SM_smefit={chi2_sm_s:.3f}")
    print()

    job_args = [(row, proj_dir, fits_dir, rc_dir,
                 args.prior_scale, args.skip_existing)
                for row in rows]

    print(f"  Launching {len(job_args)} NS runs in parallel...\n")
    t0 = time.time()
    with Pool(processes=args.n_cores) as pool:
        raw = pool.map(run_one, job_args)
    dt = time.time() - t0

    results = [r for r in raw if r is not None]
    n_ok = len(results)
    print(f"\n  {n_ok}/{len(rows)} completed  (wall time: {dt/60:.1f} min)")

    if not results:
        print("No results.")
        return

    # Save table
    tbl_path = f"{tbl_dir}/uv_validated_table.txt"
    with open(tbl_path, "w") as f:
        f.write("# UV discovery validation: NS on Asimov data vs analytic\n")
        f.write(f"# prior_scale={args.prior_scale}, ndof_uv=5\n")
        f.write(f"# {'gWH':>8}  {'mWp':>6}  {'chi2_SM_sf':>10}  {'chi2_SM_tbl':>11}  "
                f"{'max_ll':>9}  {'q_NS':>8}  {'sigma_NS':>9}  {'sigma_analytic':>14}\n")
        for r in results:
            f.write(f"  {r['gWH']:>8.4f}  {r['mWp']:>6.2f}  "
                    f"{r['chi2_sm']:>10.4f}  {r['chi2_sm_table']:>11.4f}  "
                    f"{r['max_ll']:>9.4f}  {r['q_ns']:>8.4f}  "
                    f"{r['sigma_ns']:>9.3f}  {r['sigma_analytic']:>14.3f}\n")
    print(f"  Saved: {tbl_path}")

    plot_results(results, plt_dir)
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
