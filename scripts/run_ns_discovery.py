#!/usr/bin/env python3
"""
run_ns_discovery.py

NS-based UV discovery reach for W' at FCC-ee.

For each (gWH, mWp) grid point and each of N_TOYS signal-injected pseudo-datasets:
  1. Generate noisy BSM pseudo-data:  d_i = d_SM + K c_truth + eps_i
  2. Write as projection yaml files (one per dataset)
  3. Write UV coupling runcard (tight priors: ±PRIOR_SCALE × truth)
  4. Run smefit NS  → max_loglikelihood
  5. q_i = chi2_SM_i + 2 * max_loglikelihood
           where chi2_SM_i = (d_i - d_SM)^T C^{-1} (d_i - d_SM)

All NS runs launched in parallel (multiprocessing, n_cores workers).
After all runs: median and 68% band of q over toys → sigma_UV_NS(gWH, mWp).

Output: results/ns_discovery/
    fits/       -- smefit NS output per toy
    projections/ -- pseudo-data yaml files
    runcards/   -- NS runcards
    tables/     -- discovery_ns_table.txt
    plots/      -- NS-based contour overlaid on analytic

Usage:
    python run_ns_discovery.py [--n-toys 15] [--n-cores 60] [--seed 42]
                               [--prior-scale 3.0] [--no-theory-covmat]
"""

import os, sys, argparse, json, yaml, subprocess, time
import numpy as np
from pathlib import Path
from scipy.stats import chi2 as chi2_dist, norm as norm_dist
from scipy.linalg import block_diag
from multiprocessing import Pool

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))

DB      = "/data/theorie/wtelman/smefit_database"
SMEFIT  = "/data/theorie/wtelman/miniconda3/envs/smefit-dev/bin/smefit"
SM_DATA = f"{DB}/commondata_projections_L0"
THEORY  = f"{DB}/theory"

ENV = {**os.environ,
       "MPLCONFIGDIR": "/data/theorie/wtelman/.mplconfig",
       "XDG_CACHE_HOME": "/data/theorie/wtelman/.cache"}

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
DATASETS = [{"name": ds, "order": "NLO_EW" if "zh" in ds else "LO"}
            for ds in DS_NAMES]

OPERATORS = [
    "O3pQ3", "O3pl1", "O3pl2", "O3pl3", "OQQ1", "OQQ8", "OQl13", "OQl1M",
    "OQl33", "OQl3M", "Obp", "Oll1111", "Oll1122", "Oll1133", "Oll1221",
    "Oll1331", "Oll2222", "Oll2233", "Oll2332", "Oll3333", "Op", "OpBox",
    "OpQM", "Otap", "Otp"
]

RGE = {"init_scale": 1000.0, "obs_scale": "dynamic",
       "smeft_accuracy": "integrate", "yukawa": "top", "adm_QCD": False}

GWH_GRID = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.33, 0.36]
MWP_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
FERMION_RATIO = 1.0 / 3.0


# ── EFT matching ───────────────────────────────────────────────────────────────

def eft_coefficients(gWH, gWLf11, gWLf22, gWLf33, gWqf33, mWp):
    g, lf11, lf22, lf33, qf33, m2 = gWH, gWLf11, gWLf22, gWLf33, gWqf33, mWp**2
    return {
        "O3pQ3":   (-0.25 * g * qf33) / m2,
        "O3pl1":   (-0.25 * g * lf11) / m2,
        "O3pl2":   (-0.25 * g * lf22) / m2,
        "O3pl3":   (-0.25 * g * lf33) / m2,
        "OQQ1":    (0.08333333333333333 * qf33**2) / m2,
        "OQQ8":    (-1.0 * qf33**2) / m2,
        "OQl13":   (-1.0 * lf11 * qf33) / m2,
        "OQl1M":   (1.0  * lf11 * qf33) / m2,
        "OQl33":   (-1.0 * lf33 * qf33) / m2,
        "OQl3M":   (1.0  * lf33 * qf33) / m2,
        "Obp":     (-0.006002165432052166 * g**2) / m2,
        "Oll1111": (-0.125 * lf11**2) / m2,
        "Oll1122": (0.125  * lf11 * lf22) / m2,
        "Oll1133": (0.125  * lf11 * lf33) / m2,
        "Oll1221": (-0.25  * lf11 * lf22) / m2,
        "Oll1331": (0.25   * lf11 * lf33) / m2,
        "Oll2222": (0.125  * lf22**2) / m2,
        "Oll2233": (0.125  * lf22 * lf33) / m2,
        "Oll2332": (0.25   * lf22 * lf33) / m2,
        "Oll3333": (-0.125 * lf33**2) / m2,
        "Op":      (-0.12938347743146458 * g**2) / m2,
        "OpBox":   (-0.375 * g**2) / m2,
        "OpQM":    (0.25   * g * qf33) / m2,
        "Otap":    (-0.0025559460452279554 * g**2) / m2,
        "Otp":     (-0.2480703588615627   * g**2) / m2,
    }


def eft_vec(gWH, mWp):
    f = gWH * FERMION_RATIO
    c = eft_coefficients(gWH, f, f, f, f, mWp)
    return np.array([c.get(op, 0.0) for op in OPERATORS])


# ── Build K and C ──────────────────────────────────────────────────────────────

def build_KC(use_theory_covmat=True):
    K_blocks, C_blocks = [], []
    for ds in DS_NAMES:
        sm_path = f"{SM_DATA}/{ds}.yaml"
        th_path = f"{THEORY}/{ds}.json"
        if not os.path.exists(sm_path) or not os.path.exists(th_path):
            continue
        sm = yaml.safe_load(open(sm_path))
        th = json.load(open(th_path))
        lo = th.get("LO", {})
        dc = sm["data_central"]
        dc = [dc] if not isinstance(dc, list) else list(dc)
        n  = len(dc)

        stat = sm.get("statistical_error", [0.0] * n)
        stat = list(stat) if isinstance(stat, list) else [stat] * n
        C_ds = np.diag([float(e)**2 if float(e) > 0 else (0.01 * abs(float(d)))**2
                        for e, d in zip(stat, dc)])
        sys_mat = sm.get("systematics", None)
        if sys_mat is not None:
            S = np.array(sys_mat, dtype=float)
            if S.ndim == 2 and S.shape[0] == n:
                C_ds += S @ S.T
        if use_theory_covmat:
            th_cov = th.get("theory_cov", None)
            if th_cov is not None:
                T = np.array(th_cov, dtype=float)
                if T.shape == (n, n):
                    C_ds += T

        K_ds = np.zeros((n, len(OPERATORS)))
        for j, op in enumerate(OPERATORS):
            k = lo.get(op, 0.0)
            if isinstance(k, list):
                K_ds[:, j] = [float(k[i]) if i < len(k) else 0.0 for i in range(n)]
            else:
                K_ds[:, j] = float(k)

        K_blocks.append(K_ds)
        C_blocks.append(C_ds)

    K = np.vstack(K_blocks)
    C = block_diag(*C_blocks)
    return K, C


# ── Pseudo-data generation ─────────────────────────────────────────────────────

def generate_toy_projections(gWH, mWp, toy_idx, proj_base, seed, K, C):
    """
    Generate one noisy signal pseudo-dataset and write projection yaml files.
    d_toy = d_SM + K c_truth + eps,   eps ~ N(0, C)

    Returns (proj_dir, chi2_SM_toy) where chi2_SM_toy = eps_total^T C^{-1} eps_total
    and eps_total = K c_truth + eps (the full deviation from SM).
    """
    tag      = f"gWH{int(gWH*1000):04d}_mWp{int(mWp*100):04d}_toy{toy_idx:03d}"
    proj_dir = f"{proj_base}/{tag}"
    os.makedirs(proj_dir, exist_ok=True)

    # Check if already generated
    already = all(os.path.exists(f"{proj_dir}/{ds}.yaml") for ds in DS_NAMES)

    rng = np.random.default_rng(seed + toy_idx + int(gWH * 1e6) + int(mWp * 1e4))

    # Signal vector (noiseless)
    c = eft_vec(gWH, mWp)
    signal = K @ c   # (n_obs,)

    # Noise
    L = np.linalg.cholesky(C)
    eps = L @ rng.standard_normal(L.shape[0])

    delta = signal + eps   # full deviation from SM

    # chi2_SM for this toy
    Ci = np.linalg.inv(C)
    chi2_sm = float(delta @ Ci @ delta)

    if not already:
        # Write projection files — one per dataset
        offset = 0
        for ds in DS_NAMES:
            sm_path = f"{SM_DATA}/{ds}.yaml"
            if not os.path.exists(sm_path):
                continue
            sm = yaml.safe_load(open(sm_path))
            dc = sm["data_central"]
            dc = [dc] if not isinstance(dc, list) else list(dc)
            n  = len(dc)

            d_toy = [float(dc[i]) + float(delta[offset + i]) for i in range(n)]
            offset += n

            # Write modified yaml
            out = dict(sm)
            out["data_central"] = d_toy if n > 1 else d_toy[0]
            with open(f"{proj_dir}/{ds}.yaml", "w") as f:
                yaml.dump(out, f, default_flow_style=False, allow_unicode=True,
                          sort_keys=False)

    return proj_dir, chi2_sm


def make_uv_runcard(result_id, data_path, gWH, mWp, fits_dir, prior_scale):
    """UV coupling runcard with tight priors centered on truth."""
    f = gWH * FERMION_RATIO

    # UV parameter truth values
    truth_uv = {
        "gWH":    gWH,
        "gWLf11": f,
        "gWLf22": f,
        "gWLf33": f,
        "gWqf33": f,
    }
    # Tight priors: ±prior_scale * |truth| (min 0.01)
    def prior(val):
        half = max(0.01, prior_scale * abs(val))
        return {"min": -half, "max": half}

    return {
        "result_ID":         result_id,
        "result_path":       fits_dir,
        "data_path":         data_path,
        "theory_path":       THEORY,
        "use_quad":          True,
        "use_t0":            True,
        "use_theory_covmat": True,
        "uv_couplings":      True,
        "n_samples":         5000,
        "nlive":             200,
        "lepsilon":          0.01,
        "target_evidence_unc": 0.1,
        "target_post_unc":   0.1,
        "frac_remain":       0.01,
        "datasets":          DATASETS,
        "coefficients":      {op: prior(truth_uv[op]) for op in truth_uv},
        "rge":               RGE,
    }


# ── Single NS run (called in parallel) ────────────────────────────────────────

def run_one(args):
    """
    Run one NS fit for one (gWH, mWp, toy_idx) combination.
    Returns dict with results or None on failure.
    """
    (gWH, mWp, toy_idx, proj_base, fits_dir, rc_dir,
     seed, prior_scale, K, C, skip_existing) = args

    tag       = f"gWH{int(gWH*1000):04d}_mWp{int(mWp*100):04d}_toy{toy_idx:03d}"
    result_id = f"ns_disc_{tag}"
    fit_path  = f"{fits_dir}/{result_id}/fit_results.json"

    # Generate pseudo-data
    proj_dir, chi2_sm = generate_toy_projections(
        gWH, mWp, toy_idx, proj_base, seed, K, C)

    # Check if fit already done
    if os.path.exists(fit_path) and skip_existing:
        res = json.load(open(fit_path))
        max_ll = res["max_loglikelihood"]
        q = chi2_sm + 2 * max_ll
        return dict(gWH=gWH, mWp=mWp, toy=toy_idx,
                    chi2_sm=chi2_sm, max_ll=max_ll, q=q, status="cached")

    # Write runcard
    rc = make_uv_runcard(result_id, proj_dir, gWH, mWp, fits_dir, prior_scale)
    rc_path = f"{rc_dir}/{result_id}.yaml"
    with open(rc_path, "w") as f:
        yaml.dump(rc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Run smefit NS
    t0 = time.time()
    r  = subprocess.run([SMEFIT, "NS", rc_path], env=ENV,
                        capture_output=True, text=True, timeout=1800)
    dt = time.time() - t0

    if r.returncode != 0 or not os.path.exists(fit_path):
        print(f"  FAILED: {tag}  ({dt:.0f}s)\n{r.stderr[-500:]}")
        return dict(gWH=gWH, mWp=mWp, toy=toy_idx,
                    chi2_sm=chi2_sm, max_ll=None, q=None, status="failed")

    res    = json.load(open(fit_path))
    max_ll = res["max_loglikelihood"]
    q      = chi2_sm + 2 * max_ll

    print(f"  OK: {tag}  chi2_SM={chi2_sm:.2f}  max_ll={max_ll:.3f}  "
          f"q={q:.2f}  ({dt:.0f}s)")

    return dict(gWH=gWH, mWp=mWp, toy=toy_idx,
                chi2_sm=chi2_sm, max_ll=max_ll, q=q, status="ok")


# ── Significance conversion ────────────────────────────────────────────────────

def q_to_sigma(q, ndof):
    if q <= 0 or ndof <= 0:
        return 0.0
    p = chi2_dist.sf(q, df=ndof)
    if p <= 0:
        return 99.0
    return max(0.0, float(norm_dist.isf(p)))


# ── Aggregate results per grid point ──────────────────────────────────────────

def aggregate(toy_results, ndof):
    """
    For one (gWH, mWp) point: collect q values over toys, compute
    median sigma and 16th/84th percentile band.
    """
    qs = [r["q"] for r in toy_results if r["q"] is not None]
    if not qs:
        return dict(n_ok=0, q_median=np.nan,
                    sigma_median=np.nan, sigma_lo=np.nan, sigma_hi=np.nan)
    qs = np.array(qs)
    q_med  = float(np.median(qs))
    q_lo   = float(np.percentile(qs, 16))
    q_hi   = float(np.percentile(qs, 84))
    return dict(
        n_ok      = len(qs),
        q_median  = q_med,
        sigma_median = q_to_sigma(q_med, ndof),
        sigma_lo     = q_to_sigma(q_lo,  ndof),
        sigma_hi     = q_to_sigma(q_hi,  ndof),
    )


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_results(grid_results, plt_dir, analytic_table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D

    gWH_vals = sorted(set(r["gWH"] for r in grid_results))
    mWp_vals = sorted(set(r["mWp"] for r in grid_results))
    mass_arr     = np.array(mWp_vals)
    coupling_arr = np.array(gWH_vals)
    nc, nm = len(gWH_vals), len(mWp_vals)

    def make_ns_grid(key):
        g = np.full((nc, nm), np.nan)
        for r in grid_results:
            if r["gWH"] in gWH_vals and r["mWp"] in mWp_vals:
                g[gWH_vals.index(r["gWH"]), mWp_vals.index(r["mWp"])] = r[key]
        return np.clip(np.nan_to_num(g, nan=0.0), 0, 50)

    # Load analytic UV sigma for comparison
    analytic = {}
    if analytic_table and os.path.exists(analytic_table):
        with open(analytic_table) as f:
            for line in f:
                if line.startswith("#") or not line.strip(): continue
                v = line.split()
                analytic[(float(v[0]), float(v[1]))] = float(v[11])  # H1_uv

    def make_analytic_grid():
        g = np.full((nc, nm), np.nan)
        for (gw, mw), s in analytic.items():
            if gw in gWH_vals and mw in mWp_vals:
                g[gWH_vals.index(gw), mWp_vals.index(mw)] = s
        return np.clip(np.nan_to_num(g, nan=0.0), 0, 50)

    grid_ns_med  = make_ns_grid("sigma_median")
    grid_ns_lo   = make_ns_grid("sigma_lo")
    grid_ns_hi   = make_ns_grid("sigma_hi")
    grid_analytic = make_analytic_grid()

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(mass_arr[0], mass_arr[-1])
    ax.set_ylim(coupling_arr[0], coupling_arr[-1])

    # Shade NS median discovery region
    ax.contourf(mass_arr, coupling_arr, grid_ns_med,
                levels=[5.0, 51], colors=["#ADD8E6"], alpha=0.35)
    ax.contourf(mass_arr, coupling_arr, grid_ns_med,
                levels=[0, 5.0], colors=["#DCDCDC"], alpha=0.35)

    legend_els = []

    # 68% band: fill between sigma_lo and sigma_hi contours at 5sigma
    try:
        ax.contourf(mass_arr, coupling_arr, grid_ns_hi,
                    levels=[5.0, 51], colors=["#ADD8E6"], alpha=0.3)
        ax.contourf(mass_arr, coupling_arr, grid_ns_lo,
                    levels=[5.0, 51], colors=["white"], alpha=0.4)
        legend_els.append(
            plt.matplotlib.patches.Patch(color="#ADD8E6", alpha=0.6,
                                         label="NS 68% band"))
    except Exception:
        pass

    # NS median 5sigma contour
    try:
        cs = ax.contour(mass_arr, coupling_arr, grid_ns_med,
                        levels=[5.0], colors=["#1f77b4"], linewidths=[2.5])
        ax.clabel(cs, fmt=r"$5\sigma$ (NS)", fontsize=9, inline=True)
        legend_els.append(Line2D([0],[0], color="#1f77b4", lw=2.5,
                                  label=r"UV-NS median ($5\sigma$)"))
    except Exception:
        pass

    # Analytic H1 pseudo-data contour for comparison
    if grid_analytic.max() > 0:
        try:
            cs2 = ax.contour(mass_arr, coupling_arr, grid_analytic,
                             levels=[5.0], colors=["#d62728"],
                             linewidths=[2.0], linestyles=["--"])
            ax.clabel(cs2, fmt=r"$5\sigma$ (analytic)", fontsize=9, inline=True)
            legend_els.append(Line2D([0],[0], color="#d62728", lw=2.0, ls="--",
                                      label=r"UV-analytic ($5\sigma$, H1 toys)"))
        except Exception:
            pass

    ax.legend(handles=legend_els, fontsize=10, loc="upper right", framealpha=0.9)
    ax.set_xlabel(r"$m_{W'}$ [TeV]", fontsize=13)
    ax.set_ylabel(r"$g_{WH}$", fontsize=13)
    ax.set_title(r"FCC-ee $W'$ UV discovery reach: NS fits vs analytic",
                 fontsize=12)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(True, which="major", alpha=0.2, lw=0.5)
    plt.tight_layout()
    out = f"{plt_dir}/ns_discovery_uv.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="NS-based UV discovery reach for W' at FCC-ee")
    p.add_argument("--n-toys",           type=int,   default=15)
    p.add_argument("--n-cores",          type=int,   default=60)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--prior-scale",      type=float, default=3.0,
                   help="Prior half-width = prior_scale * |truth| for UV params")
    p.add_argument("--no-theory-covmat", action="store_true")
    p.add_argument("--skip-existing",    action="store_true",
                   help="Skip NS runs where fit_results.json already exists")
    args = p.parse_args()

    use_theory_covmat = not args.no_theory_covmat

    out_base  = str(PIPELINE / "results" / "ns_discovery")
    proj_base = f"{out_base}/projections"
    fits_dir  = f"{out_base}/fits"
    rc_dir    = f"{out_base}/runcards"
    plt_dir   = f"{out_base}/plots"
    tbl_dir   = f"{out_base}/tables"
    for d in [proj_base, fits_dir, rc_dir, plt_dir, tbl_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 64)
    print("  NS-BASED UV DISCOVERY REACH  —  W' at FCC-ee")
    print(f"  N_toys      = {args.n_toys}")
    print(f"  N_cores     = {args.n_cores}")
    print(f"  Prior scale = ±{args.prior_scale} × |truth|")
    print(f"  Seed        = {args.seed}")
    print(f"  Theory covmat: {'on' if use_theory_covmat else 'off'}")
    print(f"  Grid        : {len(GWH_GRID)} × {len(MWP_GRID)} = "
          f"{len(GWH_GRID)*len(MWP_GRID)} points  ×  {args.n_toys} toys = "
          f"{len(GWH_GRID)*len(MWP_GRID)*args.n_toys} NS runs")
    print(f"  Output      : {out_base}")
    print("=" * 64)

    print("\n[1] Building K, C from theory DB...")
    K, C = build_KC(use_theory_covmat)
    print(f"    K: {K.shape}, C: {C.shape}")

    # Fisher rank for UV (approximate as rank of K @ J subspace)
    # Use rank=5 (UV has 5 free parameters) as ndof
    ndof_uv = 5

    # Build all job arguments
    jobs = []
    for gWH in GWH_GRID:
        for mWp in MWP_GRID:
            for toy_idx in range(args.n_toys):
                jobs.append((
                    gWH, mWp, toy_idx,
                    proj_base, fits_dir, rc_dir,
                    args.seed, args.prior_scale,
                    K, C, args.skip_existing
                ))

    n_jobs = len(jobs)
    print(f"\n[2] Launching {n_jobs} NS runs on {args.n_cores} cores...")
    t_start = time.time()

    with Pool(processes=args.n_cores) as pool:
        all_results = pool.map(run_one, jobs)

    t_total = time.time() - t_start
    n_ok     = sum(1 for r in all_results if r and r["status"] in ("ok", "cached"))
    n_failed = sum(1 for r in all_results if r and r["status"] == "failed")
    print(f"\n  Completed: {n_ok}/{n_jobs} OK,  {n_failed} failed  "
          f"(wall time: {t_total/60:.1f} min)")

    # Aggregate per grid point
    print("\n[3] Aggregating results...")
    grid_results = []
    for gWH in GWH_GRID:
        for mWp in MWP_GRID:
            toys = [r for r in all_results
                    if r and r["gWH"] == gWH and r["mWp"] == mWp]
            agg  = aggregate(toys, ndof_uv)
            grid_results.append(dict(gWH=gWH, mWp=mWp, **agg))
            print(f"  gWH={gWH:.3f}  mWp={mWp:.2f}  n_ok={agg['n_ok']}  "
                  f"q_med={agg['q_median']:.2f}  "
                  f"sigma={agg['sigma_median']:.2f}  "
                  f"[{agg['sigma_lo']:.2f}, {agg['sigma_hi']:.2f}]")

    # Save table
    tbl_path = f"{tbl_dir}/ns_discovery_table.txt"
    with open(tbl_path, "w") as f:
        f.write(f"# NS-based UV discovery reach: W' at FCC-ee\n")
        f.write(f"# N_toys={args.n_toys}, prior_scale={args.prior_scale}, "
                f"seed={args.seed}\n")
        f.write(f"# ndof_uv={ndof_uv}\n")
        f.write(f"# {'gWH':>8}  {'mWp':>6}  {'n_ok':>5}  {'q_med':>8}  "
                f"{'sig_med':>8}  {'sig_lo':>8}  {'sig_hi':>8}\n")
        for r in grid_results:
            f.write(f"  {r['gWH']:>8.4f}  {r['mWp']:>6.2f}  {r['n_ok']:>5}  "
                    f"{r['q_median']:>8.3f}  {r['sigma_median']:>8.3f}  "
                    f"{r['sigma_lo']:>8.3f}  {r['sigma_hi']:>8.3f}\n")
    print(f"\n  Saved: {tbl_path}")

    # Plot
    analytic_table = str(PIPELINE / "results" / "discovery_reach" /
                         "tables" / "discovery_reach_table.txt")
    print("\n[4] Plotting...")
    plot_results(grid_results, plt_dir, analytic_table)

    print(f"\n{'='*64}")
    print(f"  DONE  —  results in {out_base}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
