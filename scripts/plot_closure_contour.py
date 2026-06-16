"""
scripts/plot_closure_contour.py

Closure metrics M1 and M2 contour plots in (coupling, mass) BSM parameter space.

Supports two models:
  --model comphiggs    : comphiggs_grho*_mrho*_l1closure  dirs
  --model wprime       : wprime_gwh*_mwp*_l1closure       dirs  (default)

M1: KS p-value testing chi2_min distribution against chi2(ndof)  -> target >0.05
M2: mean posterior pull (g_fit - g_truth)/sigma_g across replicas -> target ~0

Usage:
    python scripts/plot_closure_contour.py --model wprime
    python scripts/plot_closure_contour.py --model comphiggs
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import griddata

PIPELINE = Path(__file__).parent.parent
RESULTS  = PIPELINE / "results"
OUTDIR   = RESULTS / "bsm_closure_metrics"
OUTDIR.mkdir(exist_ok=True)

NDOF = 72  # 73 FCCee data points - 1 free UV param


# ── tag parsers ───────────────────────────────────────────────────────────────

def parse_tag_comphiggs(tag):
    # comphiggs_grho050_mrho500_l1closure  -> g_rho=0.50, m_rho=50 TeV
    parts = tag.replace("_l1closure", "").split("_")
    g = float(parts[1].replace("grho", "")) / 100.0
    m = float(parts[2].replace("mrho", "")) / 10.0
    return g, m


def parse_tag_wprime(tag):
    # wprime_gwh012_mwp030_l1closure  -> gWH=0.12, mWp=3.0 TeV
    parts = tag.replace("_l1closure", "").split("_")
    g = float(parts[1].replace("gwh", "")) / 100.0
    m = float(parts[2].replace("mwp", "")) / 10.0
    return g, m


# ── per-point metrics ─────────────────────────────────────────────────────────

def load_point(closure_dir, g_truth, uv_key, fold=False):
    """
    Return (M1_ks_pvalue, M2_mean_pull, M2_std_pull) for one BSM point.

    fold=True: fold samples to |g| before computing pull (Z2-symmetric models).
    """
    fits_dir  = closure_dir / "fits"
    chi2_mins = []
    pulls     = []

    for rep_dir in sorted(fits_dir.iterdir()):
        fpath = rep_dir / "fit_results.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            d = json.load(f)

        max_ll = d.get("max_loglikelihood")
        if max_ll is not None:
            chi2_mins.append(-2.0 * max_ll)

        samples = d.get("samples", {}).get(uv_key)
        if samples is None:
            continue
        s = np.array(samples)
        if fold:
            s = np.abs(s)
        g_fit  = s.mean()
        g_sig  = s.std()
        if g_sig > 0:
            pulls.append((g_fit - g_truth) / g_sig)

    if not chi2_mins or not pulls:
        return np.nan, np.nan, np.nan

    chi2_arr = np.array(chi2_mins)
    pvals    = stats.chi2.sf(chi2_arr, df=NDOF)
    ks_p     = stats.kstest(pvals, "uniform").pvalue

    pulls_arr = np.array(pulls)
    return ks_p, float(pulls_arr.mean()), float(pulls_arr.std())


# ── plotting ──────────────────────────────────────────────────────────────────

def make_contour(points, model_name, xlabel, ylabel, coupling_label, mass_label):
    pts     = np.array(points)
    g_vals  = pts[:, 0]
    m_vals  = pts[:, 1]
    m1_vals = pts[:, 2]
    m2_vals = np.abs(pts[:, 3])

    # interpolation grid
    g_grid = np.linspace(g_vals.min(), g_vals.max(), 200)
    m_grid = np.linspace(m_vals.min(), m_vals.max(), 200)
    GG, MM = np.meshgrid(g_grid, m_grid)

    M1_grid = griddata((g_vals, m_vals), m1_vals, (GG, MM), method="cubic")
    M2_grid = griddata((g_vals, m_vals), m2_vals, (GG, MM), method="cubic")
    M1_grid = np.clip(M1_grid, 0, 1)
    M2_grid = np.clip(M2_grid, 0, None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{model_name} — L1 Closure Metrics in BSM Parameter Space",
                 fontsize=14, fontweight="bold")

    # ---- M1 panel ----
    ax = axes[0]
    cf1 = ax.contourf(GG, MM, M1_grid, levels=np.linspace(0, 1, 21),
                      cmap="RdYlGn", alpha=0.85)
    fig.colorbar(cf1, ax=ax, label="KS p-value")
    cs1 = ax.contour(GG, MM, M1_grid, levels=[0.05],
                     colors="black", linewidths=2, linestyles="--")
    ax.clabel(cs1, fmt={0.05: "p=0.05"}, fontsize=10)
    cs1b = ax.contour(GG, MM, M1_grid, levels=[0.10, 0.20, 0.50],
                      colors="black", linewidths=0.7, linestyles=":")
    ax.clabel(cs1b, fmt="%.2f", fontsize=8)
    ax.scatter(g_vals, m_vals, c=m1_vals, cmap="RdYlGn", vmin=0, vmax=1,
               s=80, edgecolors="black", linewidths=0.8, zorder=5)
    ax.set_xlabel(coupling_label, fontsize=13)
    ax.set_ylabel(mass_label, fontsize=13)
    ax.set_title("Metric 1 — Data-space closure\n(KS p-value, target > 0.05)", fontsize=11)

    # ---- M2 panel ----
    ax = axes[1]
    vmax2 = max(1.5, float(np.nanmax(M2_grid)))
    cf2 = ax.contourf(GG, MM, M2_grid, levels=np.linspace(0, vmax2, 21),
                      cmap="RdYlGn_r", alpha=0.85)
    fig.colorbar(cf2, ax=ax,
                 label=r"$|\langle (g_{\rm fit}-g_{\rm true})/\sigma_g \rangle|$")
    cs2 = ax.contour(GG, MM, M2_grid, levels=[0.5],
                     colors="black", linewidths=2, linestyles="--")
    ax.clabel(cs2, fmt={0.5: "|pull|=0.5"}, fontsize=10)
    cs2b = ax.contour(GG, MM, M2_grid, levels=[0.25, 1.0],
                      colors="black", linewidths=0.7, linestyles=":")
    ax.clabel(cs2b, fmt="%.2f", fontsize=8)
    ax.scatter(g_vals, m_vals, c=m2_vals, cmap="RdYlGn_r", vmin=0, vmax=vmax2,
               s=80, edgecolors="black", linewidths=0.8, zorder=5)
    ax.set_xlabel(coupling_label, fontsize=13)
    ax.set_ylabel(mass_label, fontsize=13)
    ax.set_title("Metric 2 — UV param-space closure\n"
                 r"($|\langle\rm pull\rangle|$, target < 0.5)", fontsize=11)

    for ax in axes:
        ax.tick_params(labelsize=11)

    plt.tight_layout()
    return fig


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["wprime", "comphiggs"], default="wprime")
    args = ap.parse_args()

    if args.model == "wprime":
        pattern      = "wprime_gwh*_mwp*_l1closure"
        parse_tag    = parse_tag_wprime
        uv_key       = "gWH"
        fold         = True   # all ops ~ gWH^2, Z2 symmetric
        coupling_lbl = r"$g_{WH}$"
        mass_lbl     = r"$m_{W'}$  [TeV]"
        model_name   = "W' (gWLf=gWqf=gWH/3)"
        out_stem     = "closure_contour_wprime"
    else:
        pattern      = "comphiggs_grho*_mrho*_l1closure"
        parse_tag    = parse_tag_comphiggs
        uv_key       = "g_rho"
        fold         = False
        coupling_lbl = r"$g_\rho$"
        mass_lbl     = r"$m_\rho$  [TeV]"
        model_name   = "Composite Higgs"
        out_stem     = "closure_contour_comphiggs"

    points = []
    for d in sorted(RESULTS.glob(pattern)):
        tag = d.name
        try:
            g, m = parse_tag(tag)
        except Exception as e:
            print(f"  skip {tag}: {e}")
            continue

        m1, m2_mean, m2_std = load_point(d, g, uv_key, fold=fold)
        status = "✓" if (not np.isnan(m1) and m1 > 0.05 and abs(m2_mean) < 0.5) else "✗"
        print(f"  {tag:50s}  g={g:.2f}  m={m:6.1f} TeV  "
              f"M1={m1:.3f}  pull={m2_mean:+.3f}±{m2_std:.3f}  {status}")
        points.append((g, m, m1, m2_mean, m2_std))

    if len(points) < 4:
        print(f"\nERROR: only {len(points)} points found — need at least 4 for contour")
        sys.exit(1)

    print(f"\nPlotting {len(points)} points...")
    fig = make_contour(points, model_name, coupling_lbl, mass_lbl,
                       coupling_lbl, mass_lbl)

    out_png = OUTDIR / f"{out_stem}.png"
    out_pdf = OUTDIR / f"{out_stem}.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
