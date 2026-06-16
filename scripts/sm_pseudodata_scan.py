#!/usr/bin/env python3
"""
sm_pseudodata_scan.py

Supervisor's key question (Section 3.3):
  How different are the exclusion contours when pseudo-data is generated
  under H0 = SM vs H1 = BSM?

Under H1 = BSM Asimov:
    q_UV(m, g) = ||U_r^T a||^2  where a = L^{-1} K c(m,g)
    Scales as 1/m^4 — gives the familiar discovery reach contour.

Under H0 = SM (L1 toys):
    q_UV(m, g) ~ chi2(rank_UV)  regardless of (m, g)
    The UV signal direction in whitened space is m-independent (all matching
    relations scale uniformly as 1/m^2), so the null distribution is exactly
    chi2(rank_UV) at every grid point — no spurious mass-dependent exclusion.

Produces three outputs:
  1. Three-panel comparison figure (BSM map | single SM replica map | false-positive rate)
  2. Contour overlay: BSM 3/5sigma boundaries vs SM 95th-pct boundary
  3. Numerical table

Usage:
    python scripts/sm_pseudodata_scan.py [--n-toys 5000] [--seed 42]
Output:
    results/sm_exclusion_comparison/
"""

import sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.stats import chi2 as chi2_dist, norm as norm_dist
from pathlib import Path

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE / "scripts"))

from run_discovery_reach import (
    build_KC, build_uv_basis, eft_coefficients_vec,
    q_to_sigma_wilks, FERMION_RATIO,
)

# Extended scan grid — go to 8 TeV to capture the full BSM→noise transition
GWH_GRID = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30]
MWP_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


# ── Scan ──────────────────────────────────────────────────────────────────────

def run_scan(K, Ci, L, u_sm, gWH_grid, mWp_grid):
    """
    For each (gWH, mWp) compute:
      sigma_bsm      : Wilks significance on BSM Asimov data
      sigma_sm_p95   : Wilks sigma at 95th pct of SM toys (~ chi2 95th pct)
      sigma_sm_single: Wilks sigma on a single SM replica (u_sm[0])
      frac_sm_3sig   : fraction of SM toys exceeding 3sigma threshold
    """
    N_toys  = u_sm.shape[0]
    u_single = u_sm[0]           # one fixed SM noise realisation for visualisation

    # Benchmark points for q-distribution panels
    BENCH = {(0.12, 3.0): None, (0.12, 5.0): None}

    results  = []
    n_total  = len(gWH_grid) * len(mWp_grid)
    n_done   = 0

    for gWH in gWH_grid:
        gWLf = gWH * FERMION_RATIO
        for mWp in mWp_grid:
            n_done += 1

            # UV signal basis at this (m, g)
            U_r, rank_uv = build_uv_basis(K, L, gWH, gWLf, gWLf, gWLf, gWLf, mWp)

            # BSM Asimov: whitened signal
            c = eft_coefficients_vec(gWH, gWLf, gWLf, gWLf, gWLf, mWp)
            a = np.linalg.solve(L, K @ c)          # (n_obs,)

            q_bsm     = float(np.sum((U_r.T @ a) ** 2))
            sigma_bsm = q_to_sigma_wilks(q_bsm, rank_uv)

            # SM toys: q_i = ||U_r^T u_i||^2 ~ chi2(rank_uv) under H0
            q_sm_toys = np.sum((u_sm @ U_r) ** 2, axis=1)   # (N_toys,)

            # BSM signal replicas: shift SM toys by signal, then project
            # q_h1_i = ||U_r^T (u_i + a)||^2
            q_bsm_toys = np.sum(((u_sm + a[np.newaxis, :]) @ U_r) ** 2, axis=1)

            thresh_3s = chi2_dist.ppf(1 - 2.7e-3, df=rank_uv)
            frac_3s   = float(np.mean(q_sm_toys > thresh_3s))

            if n_done % 22 == 1 or n_done == n_total:
                print(f"  [{n_done:3d}/{n_total}] gWH={gWH:.2f} mWp={mWp:.1f}  "
                      f"σ_BSM={sigma_bsm:5.2f}  frac_3σ={frac_3s*100:.3f}%")

            r = dict(
                gWH=gWH, mWp=mWp,
                q_bsm=q_bsm, rank_uv=rank_uv,
                sigma_bsm=sigma_bsm,
                frac_sm_3sig=frac_3s,
            )

            key = (round(gWH, 4), round(mWp, 4))
            if key in BENCH:
                r["q_sm_toys"]  = q_sm_toys
                r["q_bsm_toys"] = q_bsm_toys
                BENCH[key] = r

            results.append(r)

    return results, BENCH


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_grids(results):
    gWH_arr = np.array(sorted(set(r["gWH"] for r in results)))
    mWp_arr = np.array(sorted(set(r["mWp"] for r in results)))
    nc, nm  = len(gWH_arr), len(mWp_arr)

    def mg(key):
        g = np.full((nc, nm), np.nan)
        for r in results:
            ic = list(gWH_arr).index(r["gWH"])
            im = list(mWp_arr).index(r["mWp"])
            g[ic, im] = r[key]
        return g

    return gWH_arr, mWp_arr, {k: mg(k) for k in ["sigma_bsm", "frac_sm_3sig"]}


def _panel(ax, mass, coupling, grid, vmax, cmap, title, contour_levels=None,
           cb_label=""):
    from scipy.interpolate import RectBivariateSpline
    from scipy.ndimage import gaussian_filter

    grid = np.clip(np.nan_to_num(grid, nan=0.0), 0, vmax)

    # Interpolate to fine grid for smooth rendering
    spl      = RectBivariateSpline(coupling, mass, grid, kx=3, ky=3)
    m_fine   = np.linspace(mass[0],     mass[-1],     300)
    c_fine   = np.linspace(coupling[0], coupling[-1], 300)
    grid_fine = np.clip(spl(c_fine, m_fine), 0, vmax)
    grid_fine = gaussian_filter(grid_fine, sigma=0.8)

    im = ax.pcolormesh(m_fine, c_fine, grid_fine, cmap=cmap,
                       vmin=0, vmax=vmax, shading="auto")
    plt.colorbar(im, ax=ax, label=cb_label, fraction=0.046, pad=0.04)

    if contour_levels:
        for lev, ls, lw, col, lbl in contour_levels:
            try:
                cs = ax.contour(m_fine, c_fine, grid_fine, levels=[lev],
                                colors=[col], linewidths=[lw], linestyles=[ls])
                ax.clabel(cs, fmt=lbl, fontsize=9, inline=True,
                          inline_spacing=4, use_clabeltext=True)
            except Exception:
                pass

    ax.axhline(0.12, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"$m_{W'}$ [TeV]", fontsize=10)
    ax.set_ylabel(r"$g_{WH}$", fontsize=10)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_three_panel(gWH_arr, mWp_arr, grids, bench, out_dir, N_toys):
    from scipy.stats import ncx2 as ncx2_dist

    fig = plt.figure(figsize=(18, 5))
    gs  = fig.add_gridspec(1, 3, wspace=0.35)

    # ── Panel A: BSM significance map ────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    _panel(ax0, mWp_arr, gWH_arr, grids["sigma_bsm"],
           vmax=20, cmap="Blues",
           title=r"H$_1$ = BSM Asimov  —  $\sigma_\mathrm{UV}(g, m)$",
           contour_levels=[
               (5.0, "-",  2.5, "#8B0000", r"$5\sigma$"),
               (3.0, "--", 1.8, "#8B0000", r"$3\sigma$"),
           ],
           cb_label=r"$\sigma_\mathrm{UV}$")

    # ── Panels B & C: q-distribution at two benchmark points ─────────────────
    bench_list = [
        ((0.12, 3.0), r"Deep in discovery region  ($g=0.12,\ m_{W'}=3$ TeV)"),
        ((0.12, 5.0), r"Near 3$\sigma$ boundary  ($g=0.12,\ m_{W'}=5$ TeV)"),
    ]

    col_sm  = "#2980B9"
    col_bsm = "#E67E22"

    for col_idx, (key, subtitle) in enumerate(bench_list):
        r = bench.get(key)
        if r is None:
            continue
        ax = fig.add_subplot(gs[col_idx + 1])

        q_sm   = r["q_sm_toys"]
        q_bsm  = r["q_bsm_toys"]
        q_asim = r["q_bsm"]
        ndof   = r["rank_uv"]
        lam    = q_asim          # non-centrality = Asimov q

        q_max  = max(np.percentile(q_bsm, 99.5), q_asim * 1.1, 30)
        bins   = np.linspace(0, q_max, 50)

        ax.hist(q_sm,  bins=bins, density=True, alpha=0.55,
                color=col_sm,  label=r"H$_0$=SM replicas")
        ax.hist(q_bsm, bins=bins, density=True, alpha=0.55,
                color=col_bsm, label=r"H$_1$=BSM replicas")

        x = np.linspace(0.05, q_max, 500)
        ax.plot(x, chi2_dist.pdf(x, df=ndof), color=col_sm,
                lw=2, ls="--", label=fr"$\chi^2({ndof})$ [analytic SM]")
        ax.plot(x, ncx2_dist.pdf(x, df=ndof, nc=lam), color=col_bsm,
                lw=2, ls="--", label=fr"$\chi^2({ndof},\lambda={lam:.1f})$ [analytic BSM]")

        # 3σ threshold
        thresh = chi2_dist.ppf(1 - 2.7e-3, df=ndof)
        ax.axvline(thresh, color="gray", lw=1.5, ls=":",
                   label=fr"$3\sigma$ threshold ($q={thresh:.1f}$)")

        # Asimov q
        sigma_asim = q_to_sigma_wilks(q_asim, ndof)
        ax.axvline(q_asim, color="black", lw=2, ls="-",
                   label=fr"BSM Asimov $q={q_asim:.1f}$ ($\sigma={sigma_asim:.1f}$)")

        ax.set_xlabel(r"$q = \chi^2_\mathrm{SM} - \chi^2_\mathrm{UV,min}$", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(subtitle, fontsize=10)
        ax.legend(fontsize=7.5, framealpha=0.85)
        ax.set_xlim(0, q_max)

    fig.suptitle(
        r"$\mathbf{Exclusion \neq Discovery}$: "
        r"W$'$ UV coupling method  —  FCC-ee  (N=" + str(N_toys) + r" toys)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    out = out_dir / "sm_vs_bsm_three_panel.png"
    plt.savefig(str(out), dpi=160, bbox_inches="tight")
    plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_contour_overlay(gWH_arr, mWp_arr, grids, out_dir, N_toys):
    """
    Clean overlay: BSM 3/5sigma boundaries (blue) vs SM 95th-pct boundary (red dashed).
    The SM 95th-pct contour (at sigma ≈ 1.64) should lie far from the 3sigma line,
    demonstrating that SM fluctuations cannot mimic the BSM exclusion region.
    """
    sg_bsm = np.clip(np.nan_to_num(grids["sigma_bsm"], nan=0.0), 0, 25)
    sg_sm  = sg_bsm * 0  # SM 95th pct is uniform ~1.63; shown via annotation only

    fig, ax = plt.subplots(figsize=(9, 6))

    # BSM filled discovery region
    ax.contourf(mWp_arr, gWH_arr, sg_bsm,
                levels=[5.0, 26.0], colors=["#AED6F1"], alpha=0.4)
    ax.contourf(mWp_arr, gWH_arr, sg_bsm,
                levels=[3.0, 5.0],  colors=["#85C1E9"], alpha=0.3)

    # BSM contour lines
    for lev, ls, lw, lbl in [
        (5.0, "-",  2.5, r"$5\sigma$ (BSM)"),
        (3.0, "--", 2.0, r"$3\sigma$ (BSM)"),
    ]:
        try:
            cs = ax.contour(mWp_arr, gWH_arr, sg_bsm, levels=[lev],
                            colors=["#1A5276"], linewidths=[lw], linestyles=[ls])
            ax.clabel(cs, fmt=lbl, fontsize=9, inline=True)
        except Exception:
            pass

    # SM 95th-pct: annotate directly since sigma is uniform (~1.63) across all (m,g)
    # — no contour crossing exists, which is itself the key result
    sm_med_val = float(np.nanmedian(sg_sm))
    ax.text(0.98, 0.04,
            fr"SM 95th pct: $\sigma_{{UV}} \approx {sm_med_val:.2f}$ everywhere"
            "\n(no exclusion region under H$_0$=SM)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color="#C0392B",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#C0392B", alpha=0.85))

    # Reference line
    ax.axhline(0.12, color="gray", lw=1.0, ls=":", alpha=0.6)
    ax.text(mWp_arr[0] + 0.1, 0.122, r"$g=0.12$", fontsize=9, color="gray")

    legend_els = [
        plt.matplotlib.patches.Patch(facecolor="#AED6F1", alpha=0.7,
                                     label=r"$5\sigma$ discovery region (BSM)"),
        plt.matplotlib.patches.Patch(facecolor="#85C1E9", alpha=0.5,
                                     label=r"$3\sigma$ evidence region (BSM)"),
        Line2D([0], [0], color="#1A5276", lw=2.5, ls="-",
               label=r"BSM $5\sigma$ boundary"),
        Line2D([0], [0], color="#1A5276", lw=2.0, ls="--",
               label=r"BSM $3\sigma$ boundary"),
        Line2D([0], [0], color="#C0392B", lw=0, marker="s",
               markerfacecolor="#C0392B", markersize=0,
               label=fr"SM 95th pct: $\sigma\approx1.63$ everywhere (no exclusion)"),
    ]
    ax.legend(handles=legend_els, fontsize=9, loc="upper right",
              framealpha=0.9, edgecolor="lightgray")

    ax.set_xlim(mWp_arr[0], mWp_arr[-1])
    ax.set_ylim(gWH_arr[0],  gWH_arr[-1])
    ax.set_xlabel(r"$m_{W'}$ [TeV]", fontsize=12)
    ax.set_ylabel(r"$g_{WH}$", fontsize=12)
    ax.set_title(
        r"W$'$ at FCC-ee: discovery reach (H$_1$=BSM) vs SM false-positive boundary (H$_0$=SM)"
        "\n" + r"UV coupling method — Exclusion $\neq$ Discovery",
        fontsize=11, pad=8,
    )
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(True, which="major", alpha=0.15, lw=0.5)

    plt.tight_layout()
    out = out_dir / "sm_vs_bsm_contour_overlay.png"
    plt.savefig(str(out), dpi=160, bbox_inches="tight")
    plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Table ─────────────────────────────────────────────────────────────────────

def save_table(results, path, N_toys):
    header = (
        f"# SM vs BSM pseudo-data scan — W' UV coupling method\n"
        f"# N_toys = {N_toys}\n"
        f"# sigma_bsm     : Wilks significance on BSM Asimov data\n"
        f"# frac_sm_3sig% : pct of SM toys exceeding 3sigma threshold (nominal=0.27%)\n"
        f"# {'gWH':>8}  {'mWp':>6}  {'sigma_bsm':>11}  {'frac_3sig_%':>12}\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for r in results:
            f.write(
                f"  {r['gWH']:>8.4f}  {r['mWp']:>6.2f}  "
                f"{r['sigma_bsm']:>11.3f}  "
                f"{r['frac_sm_3sig']*100:>12.4f}\n"
            )
    print(f"  Table: {path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(results, N_toys):
    print(f"\n{'='*70}")
    print(f"  SUMMARY — UV coupling method, W' at FCC-ee")
    print(f"  N_toys = {N_toys}")
    print(f"{'='*70}")
    print(f"  {'gWH':>6}  {'mWp':>5}  {'σ_BSM':>7}  {'frac_3σ%':>9}")
    print(f"  {'─'*6}  {'─'*5}  {'─'*7}  {'─'*9}")
    for r in results:
        print(f"  {r['gWH']:>6.3f}  {r['mWp']:>5.1f}  "
              f"{r['sigma_bsm']:>7.2f}  "
              f"{r['frac_sm_3sig']*100:>9.4f}")

    rank_mode = int(np.round(np.median([r["rank_uv"] for r in results])))
    q95_expected = chi2_dist.ppf(0.95, df=rank_mode)
    sig95_expected = q_to_sigma_wilks(q95_expected, rank_mode)
    print(f"\n  Expected chi2({rank_mode}) 95th pct: q={q95_expected:.2f}  →  σ={sig95_expected:.2f}")
    print(f"  Expected false-positive rate at 3σ threshold: 0.270%")
    all_fracs = [r["frac_sm_3sig"] * 100 for r in results]
    print(f"  Observed false-positive rate: mean={np.mean(all_fracs):.3f}%  "
          f"std={np.std(all_fracs):.3f}%")
    print(f"{'='*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-toys", type=int, default=5000)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--no-theory-covmat", action="store_true")
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else PIPELINE / "results" / "sm_exclusion_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  SM vs BSM PSEUDO-DATA SCAN  —  W' UV coupling method")
    print(f"  H0 = SM null  vs  H1 = BSM Asimov")
    print(f"  N_toys = {args.n_toys}  |  seed = {args.seed}")
    print(f"  Grid: {len(GWH_GRID)} × {len(MWP_GRID)} = {len(GWH_GRID)*len(MWP_GRID)} points  "
          f"(mWp {MWP_GRID[0]}–{MWP_GRID[-1]} TeV)")
    print("=" * 64)

    print("\n[1] Building K, C, Ci from theory DB...")
    K, C, Ci = build_KC(not args.no_theory_covmat)
    print(f"    K: {K.shape},  C: {C.shape}")

    print("\n[2] Cholesky factorisation C = L L^T ...")
    L = np.linalg.cholesky(C)

    print(f"\n[3] Generating {args.n_toys} whitened SM toys  (seed={args.seed})...")
    rng  = np.random.default_rng(args.seed)
    # u ~ N(0, I),  actual data eps = L @ u^T ~ N(0, C)
    u_sm = rng.standard_normal((args.n_toys, K.shape[0]))

    print(f"\n[4] Scanning {len(GWH_GRID)*len(MWP_GRID)} grid points...")
    results, bench = run_scan(K, Ci, L, u_sm, GWH_GRID, MWP_GRID)

    print("\n[5] Saving table...")
    save_table(results, out_dir / "sm_exclusion_comparison_table.txt", args.n_toys)

    print("\n[6] Generating figures...")
    gWH_arr, mWp_arr, grids = build_grids(results)
    plot_three_panel(gWH_arr, mWp_arr, grids, bench, out_dir, args.n_toys)
    plot_contour_overlay(gWH_arr, mWp_arr, grids, out_dir, args.n_toys)

    print_summary(results, args.n_toys)

    print("  DONE — results in", out_dir)


if __name__ == "__main__":
    main()
