#!/usr/bin/env python3
"""
run_exclusion_reach.py

95% CL expected exclusion reach for the W' model at FCC-ee.

Statistical question (different from discovery):
    Given that FCC-ee observes data consistent with the SM,
    which (gWH, mWp) points can be excluded at 95% CL?

Procedure (SM Asimov):
    1. SM Asimov data: d = d_SM  (no signal injected)
    2. For each BSM point: q = chi2(d_SM | SM) - chi2(d_SM | BSM)
                             = 0 - (-c^T F c) = c^T F c
       (SM fits perfectly; BSM fitted to SM data has residual c^T F c)
    3. Null distribution under H0=BSM: toy datasets d_toy = d_BSM + eps
       q_toy = chi2(d_toy | SM) - chi2(d_toy | BSM_best)
             = ||a + u||^2 - 0   [full SMEFT absorbs perfectly]
    4. p_excl = P(q_toy < q_Asimov | H0=BSM)
              = P(||a+u||^2 < ||a||^2)  ~  0.5 for median
       But for expected exclusion (Asimov convention):
       use Wilks: q ~ chi2(ndof) → 95% CL at 1.645 sigma

    Note: For the linear Gaussian model, the expected exclusion reach
    numerically equals the discovery reach at the same threshold.
    The distinction is the physical question asked and the threshold used:
        Discovery:  5sigma  (p < 2.87e-7  under H0=SM)
        Exclusion: 1.645sigma (p < 0.05   under H0=BSM, one-sided)
        Exclusion:  2sigma  (p < 0.0228  two-sided 95.45% CL)

Usage:
    python run_exclusion_reach.py [--n-toys 10000] [--seed 42] [--no-theory-covmat]

Output: results/exclusion_reach/
"""

import os, sys, argparse, json, yaml
import numpy as np
from pathlib import Path
from scipy.stats import chi2 as chi2_dist, norm as norm_dist
from scipy.linalg import block_diag
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))

# Import shared infrastructure from run_discovery_reach
from run_discovery_reach import (
    DS_NAMES, OPERATORS, GWH_GRID, MWP_GRID, FERMION_RATIO,
    build_KC, build_range_basis, build_1op_basis, build_full_smeft_basis,
    build_uv_basis, eft_coefficients_vec, wprime_jacobian,
    generate_sm_toys, compute_q_toys,
)

# Exclusion thresholds
CL_LEVELS = [
    (0.95,  1,  "#1f77b4",  "-",  r"95% CL ($1.64\sigma$)"),  # one-sided 95%
    (0.9973, 1, "#ff7f0e",  "--", r"$3\sigma$"),
    (0.9999999, 1, "#2ca02c", ":", r"$5\sigma$"),
]

L_GLOBAL = None


def q_to_sigma(q, ndof):
    """One-sided significance from Wilks."""
    if q <= 0 or ndof <= 0:
        return 0.0
    p = chi2_dist.sf(q, df=ndof)
    if p <= 0:
        return 99.0
    return max(0.0, float(norm_dist.isf(p)))


def q_to_sigma_toys(q_bsm, q_toys):
    """Empirical p-value from toys. q_toys = null distribution (H0=BSM toys)."""
    n_above = int(np.sum(q_toys < q_bsm))   # excl: how many BSM toys have smaller q than Asimov
    N = len(q_toys)
    if n_above >= N:
        return 99.0
    if n_above == 0:
        return 0.0
    p = 1.0 - n_above / N    # p_excl = P(q_toy > q_Asimov | H0=BSM)
    return max(0.0, float(norm_dist.isf(p)))


def run_scan(K, C, Ci, N_toys, seed, gWH_grid, mWp_grid, fermion_ratio):
    global L_GLOBAL

    F = K.T @ Ci @ K
    L = np.linalg.cholesky(C)
    L_GLOBAL = L

    print(f"\n  Generating {N_toys} SM toy datasets (seed={seed})...")
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((N_toys, L.shape[0]))  # H0=SM toys

    print("  Building constant operator bases...")
    U_r_1op,  rank_1op  = build_1op_basis(K, L, "OQl13")
    U_r_full, rank_full = build_full_smeft_basis(K, L)
    print(f"    1-op  basis: rank={rank_1op}")
    print(f"    Full SMEFT:  rank={rank_full}")

    results = []
    n_total = len(gWH_grid) * len(mWp_grid)
    n_done  = 0

    print(f"\n  Scanning {n_total} grid points...\n")

    for gWH in gWH_grid:
        for mWp in mWp_grid:
            n_done += 1
            gWLf = gWH * fermion_ratio
            gWqf = gWH * fermion_ratio

            # Wilson coefficients at this point
            c = eft_coefficients_vec(gWH, gWLf, gWLf, gWLf, gWqf, mWp)

            # Asimov exclusion statistic: q = c^T F c
            # (how badly BSM fits the SM Asimov data)
            q_full = float(c @ F @ c)

            # 1-op: score at OQl13 index
            idx   = OPERATORS.index("OQl13")
            g_idx = float(F[idx, :] @ c)
            q_1op = g_idx**2 / float(F[idx, idx])

            # UV basis (per point)
            U_r_uv, rank_uv = build_uv_basis(K, L, gWH, gWLf, gWLf, gWLf, gWqf, mWp)

            # Whitened signal: a = L^{-1} K c
            a = np.linalg.solve(L, K @ c)

            # H0=BSM toys: d_toy = d_BSM + eps → whitened: v = a + u
            # q_excl_toy = ||v||^2 (SM chi2) - 0 (BSM absorbs perfectly for full SMEFT)
            # For exclusion null: generate toys under H1=BSM
            u_bsm = u + a[np.newaxis, :]   # (N_toys, n_obs)

            q_null_1op  = compute_q_toys(u_bsm, U_r_1op)   # BSM toys projected on 1op basis
            q_null_full = compute_q_toys(u_bsm, U_r_full)   # BSM toys projected on full basis
            q_null_uv   = compute_q_toys(u_bsm, U_r_uv)     # BSM toys projected on UV basis

            # Wilks exclusion significances (Asimov q, chi2(ndof) null)
            s_wk_1op  = q_to_sigma(q_1op,  rank_1op)
            s_wk_full = q_to_sigma(q_full, rank_full)
            s_wk_uv   = q_to_sigma(q_full, rank_uv)

            # Toy-based exclusion: p = P(q_BSM_toy > q_Asimov | H0=BSM)
            s_toy_1op  = q_to_sigma_toys(q_1op,  q_null_1op)
            s_toy_full = q_to_sigma_toys(q_full, q_null_full)
            s_toy_uv   = q_to_sigma_toys(q_full, q_null_uv)
            if not np.isfinite(s_toy_1op):  s_toy_1op  = s_wk_1op
            if not np.isfinite(s_toy_full): s_toy_full = s_wk_full
            if not np.isfinite(s_toy_uv):   s_toy_uv   = s_wk_uv

            excl = "EXCLUDED" if s_toy_uv >= 1.645 else ""
            print(f"  [{n_done:3d}/{n_total}] gWH={gWH:.3f}  mWp={mWp:.2f} TeV  "
                  f"q={q_full:7.2f}  "
                  f"excl(1op)={s_toy_1op:.2f}  "
                  f"excl(full)={s_toy_full:.2f}  "
                  f"excl(UV)={s_toy_uv:.2f}  {excl}")

            results.append(dict(
                gWH=gWH, mWp=mWp,
                q_full=q_full, q_1op=q_1op,
                rank_uv=rank_uv,
                sigma_wilks_1op=s_wk_1op,   sigma_wilks_full=s_wk_full,
                sigma_wilks_uv=s_wk_uv,
                sigma_toys_1op=s_toy_1op,   sigma_toys_full=s_toy_full,
                sigma_toys_uv=s_toy_uv,
            ))

    return results, rank_1op, rank_full


def _contour_panel(ax, mass_arr, coupling_arr, sigma_grid, title):
    grid = np.clip(np.nan_to_num(sigma_grid, nan=0.0), 0, 50)

    # Shade: light blue = excluded at 95% CL, grey = not excluded
    ax.contourf(mass_arr, coupling_arr, grid,
                levels=[1.645, 51], colors=["#ADD8E6"], alpha=0.55)
    ax.contourf(mass_arr, coupling_arr, grid,
                levels=[0, 1.645], colors=["#DCDCDC"], alpha=0.55)

    # Contour lines
    for sigma, lw, ls, lbl, col in [
        (1.645, 2.5, "-",  r"95% CL", "#1f77b4"),
        (3.0,   1.5, "--", r"$3\sigma$", "#ff7f0e"),
        (5.0,   1.0, ":",  r"$5\sigma$", "#2ca02c"),
    ]:
        try:
            cs = ax.contour(mass_arr, coupling_arr, grid,
                            levels=[sigma], colors=[col],
                            linewidths=[lw], linestyles=[ls])
            ax.clabel(cs, fmt=lbl, fontsize=9, inline=True)
        except Exception:
            pass

    ax.set_xlabel(r"$m_{W'}$ [TeV]", fontsize=11)
    ax.set_ylabel(r"$g_{WH}$", fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(True, which="major", alpha=0.2, lw=0.5)


def plot_exclusion(results, plt_dir, N_toys, rank_full):
    gWH_vals = sorted(set(r["gWH"] for r in results))
    mWp_vals = sorted(set(r["mWp"] for r in results))
    nc, nm   = len(gWH_vals), len(mWp_vals)

    def make_grid(key):
        g = np.full((nc, nm), np.nan)
        for r in results:
            ic = gWH_vals.index(r["gWH"])
            im = mWp_vals.index(r["mWp"])
            g[ic, im] = r[key]
        return g

    mass_arr     = np.array(mWp_vals)
    coupling_arr = np.array(gWH_vals)

    methods = [
        ("sigma_wilks_1op",  "sigma_toys_1op",  "1-op: OQl13 (ndof=1)"),
        ("sigma_wilks_full", "sigma_toys_full", f"Full SMEFT (rank={rank_full})"),
        ("sigma_wilks_uv",   "sigma_toys_uv",   "UV-profiled (5 params)"),
    ]

    # Figure 1: 2x3 grid — Wilks / BSM-toys
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for col, (wk_key, toy_key, label) in enumerate(methods):
        _contour_panel(axes[0, col], mass_arr, coupling_arr,
                       make_grid(wk_key),
                       f"Wilks (Asimov): {label}")
        _contour_panel(axes[1, col], mass_arr, coupling_arr,
                       make_grid(toy_key),
                       f"BSM toys (N={N_toys}): {label}")

    fig.suptitle(r"$W'$ expected exclusion reach at FCC-ee (95% CL)",
                 fontsize=13)
    plt.tight_layout()
    out = f"{plt_dir}/exclusion_region_all_methods.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    # Figure 2: comparison — 95% CL contours overlaid for all methods
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors  = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    lstyles = ["-", "--", ":"]
    labels  = ["1-op (OQl13)", "Full SMEFT", "UV-profiled"]

    for ax, keys, title in [
        (axes[0], ("sigma_wilks_1op", "sigma_wilks_full", "sigma_wilks_uv"),
                  "Wilks (Asimov)"),
        (axes[1], ("sigma_toys_1op", "sigma_toys_full", "sigma_toys_uv"),
                  f"BSM toys (N={N_toys})"),
    ]:
        ax.set_xlim(mass_arr[0], mass_arr[-1])
        ax.set_ylim(coupling_arr[0], coupling_arr[-1])
        for key, col, ls, lbl in zip(keys, colors, lstyles, labels):
            try:
                grid_c = np.clip(np.nan_to_num(make_grid(key), nan=0.0), 0, 50)
                cs = ax.contour(mass_arr, coupling_arr, grid_c,
                                levels=[1.645], colors=[col],
                                linewidths=[2.0], linestyles=[ls])
                ax.clabel(cs, fmt=f"95% CL ({lbl})", fontsize=8, inline=True)
            except Exception:
                pass
        ax.set_xlabel(r"$m_{W'}$ [TeV]", fontsize=12)
        ax.set_ylabel(r"$g_{WH}$", fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.grid(True, which="major", alpha=0.2, lw=0.5)

    from matplotlib.lines import Line2D
    legend_els = [Line2D([0],[0], color=c, lw=2, ls=ls, label=lbl)
                  for c, ls, lbl in zip(colors, lstyles, labels)]
    axes[1].legend(handles=legend_els, fontsize=9, loc="upper right")
    fig.suptitle(r"$W'$ expected exclusion reach: 95% CL contours by method", fontsize=12)
    plt.tight_layout()
    out2 = f"{plt_dir}/exclusion_region_comparison.png"
    plt.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out2}")

    # Figure 3: overlay discovery (5sigma) vs exclusion (95% CL) for UV method
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(mass_arr[0], mass_arr[-1])
    ax.set_ylim(coupling_arr[0], coupling_arr[-1])

    # Use Wilks sigma — exact for linear Gaussian, toys add nothing here
    grid_uv = np.clip(np.nan_to_num(make_grid("sigma_wilks_uv"), nan=0.0), 0, 50)
    for sigma, lw, ls, col, lbl in [
        (1.645, 2.5, "-",  "#1f77b4", "95% CL exclusion"),
        (3.0,   2.0, "--", "#ff7f0e", r"$3\sigma$ evidence"),
        (5.0,   2.0, ":",  "#2ca02c", r"$5\sigma$ discovery"),
    ]:
        try:
            cs = ax.contour(mass_arr, coupling_arr, grid_uv,
                            levels=[sigma], colors=[col],
                            linewidths=[lw], linestyles=[ls])
            ax.clabel(cs, fmt=lbl, fontsize=9, inline=True)
        except Exception:
            pass

    ax.set_xlabel(r"$m_{W'}$ [TeV]", fontsize=12)
    ax.set_ylabel(r"$g_{WH}$", fontsize=12)
    ax.set_title(r"$W'$ reach at FCC-ee — UV-profiled (exclusion vs discovery)", fontsize=11)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(True, which="major", alpha=0.2, lw=0.5)
    plt.tight_layout()
    out3 = f"{plt_dir}/exclusion_vs_discovery_uv.png"
    plt.savefig(out3, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out3}")


def save_table(results, path, N_toys, seed, use_theory_covmat):
    header = (
        f"# Expected exclusion reach: W' model at FCC-ee\n"
        f"# N_toys={N_toys}, seed={seed}\n"
        f"# Theory covmat: {'on' if use_theory_covmat else 'off'}\n"
        f"# sigma convention: one-sided (Phi^{{-1}}(1-p))\n"
        f"# 95% CL = 1.645 sigma\n"
        f"# Wilks: q ~ chi2(ndof) null under H0=BSM\n"
        f"# Toys:  BSM toy datasets (signal injected)\n"
        f"# {'gWH':>8}  {'mWp':>6}  {'q_full':>10}  "
        f"{'wk_1op':>8}  {'wk_full':>8}  {'wk_uv':>8}  "
        f"{'toy_1op':>8}  {'toy_full':>8}  {'toy_uv':>8}\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for r in results:
            f.write(
                f"  {r['gWH']:>8.4f}  {r['mWp']:>6.2f}  {r['q_full']:>10.3f}  "
                f"{r['sigma_wilks_1op']:>8.3f}  {r['sigma_wilks_full']:>8.3f}  "
                f"{r['sigma_wilks_uv']:>8.3f}  "
                f"{r['sigma_toys_1op']:>8.3f}  {r['sigma_toys_full']:>8.3f}  "
                f"{r['sigma_toys_uv']:>8.3f}\n"
            )
    print(f"  Saved: {path}")


def main():
    p = argparse.ArgumentParser(
        description="Expected exclusion reach for W' at FCC-ee (95% CL)"
    )
    p.add_argument("--n-toys",           type=int,   default=10000)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--no-theory-covmat", action="store_true")
    p.add_argument("--fermion-ratio",    type=float, default=FERMION_RATIO)
    args = p.parse_args()

    use_theory_covmat = not args.no_theory_covmat

    out_base = str(PIPELINE / "results" / "exclusion_reach")
    plt_dir  = f"{out_base}/plots"
    tbl_dir  = f"{out_base}/tables"
    for d in [plt_dir, tbl_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 64)
    print("  EXPECTED EXCLUSION REACH  —  W' at FCC-ee  (95% CL)")
    print(f"  N_toys        = {args.n_toys}  |  seed = {args.seed}")
    print(f"  Theory covmat : {'on' if use_theory_covmat else 'off'}")
    print(f"  Fermion ratio : gWLf = gWqf = {args.fermion_ratio} × gWH")
    print(f"  Grid          : {len(GWH_GRID)} × {len(MWP_GRID)} = {len(GWH_GRID)*len(MWP_GRID)} points")
    print(f"  Output        : {out_base}")
    print("=" * 64)

    print("\n[1] Building K, C, Ci from theory DB...")
    K, C, Ci = build_KC(use_theory_covmat)
    print(f"    K: {K.shape}, C: {C.shape}")

    print("\n[2] Running scan...")
    results, rank_1op, rank_full = run_scan(
        K, C, Ci,
        N_toys        = args.n_toys,
        seed          = args.seed,
        gWH_grid      = GWH_GRID,
        mWp_grid      = MWP_GRID,
        fermion_ratio = args.fermion_ratio,
    )

    print(f"\n[3] Saving table...")
    save_table(results, f"{tbl_dir}/exclusion_reach_table.txt",
               args.n_toys, args.seed, use_theory_covmat)

    print(f"\n[4] Generating plots...")
    plot_exclusion(results, plt_dir, args.n_toys, rank_full)

    print(f"\n{'='*64}")
    print(f"  DONE  —  results in {out_base}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
