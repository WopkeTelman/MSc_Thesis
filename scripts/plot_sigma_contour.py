"""
scripts/plot_sigma_contour.py

BSM detection significance contour in (gWH, mWp) parameter space.
Reads PCA sigma values from existing pipeline summary.txt files.

For missing grid points, run first:
    python run_pipeline.py --model wprime --gWH X --mWp Y --no-ns --no-report --skip-existing

Usage:
    python scripts/plot_sigma_contour.py [--model wprime]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RBFInterpolator

PIPELINE = Path(__file__).parent.parent
RESULTS  = PIPELINE / "results"
OUTDIR   = RESULTS / "bsm_closure_metrics"
OUTDIR.mkdir(exist_ok=True)


# ── tag parsers ───────────────────────────────────────────────────────────────

def parse_wprime_tag(tag):
    # wprime_gwh012_mwp030  ->  gWH=0.12, mWp=3.0 TeV
    m = re.match(r"wprime_gwh(\d+)_mwp(\d+)$", tag)
    if not m:
        return None
    g = float(m.group(1)) / 100.0
    mass = float(m.group(2)) / 10.0
    return g, mass


def read_sigma(summary_path, k=2):
    """Extract PCA k=<k> sigma from summary.txt. Returns None if not found."""
    try:
        txt = summary_path.read_text()
        # match line like "  PCA k=2                   21.78      2    4.28  "
        pat = re.compile(rf"PCA k={k}\s+[\d.]+\s+{k}\s+([\d.]+)")
        m = pat.search(txt)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["wprime"], default="wprime")
    ap.add_argument("--pca-k", type=int, default=2,
                    help="PCA k to use for sigma (default: 2)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Output directory (default: results/bsm_closure_metrics)")
    args = ap.parse_args()
    global OUTDIR
    if args.out_dir:
        OUTDIR = Path(args.out_dir)
        OUTDIR.mkdir(parents=True, exist_ok=True)

    # skip dirs with these substrings
    SKIP = {"freeops", "newdata", "allg", "glf", "gwhonly", "uva",
            "quad", "27c", "scan", "l1", "constrained", "unig", "test",
            "universal", "u2d"}

    points = []  # (gWH, mWp, sigma)
    for d in sorted(RESULTS.glob("wprime_gwh*_mwp*")):
        tag = d.name
        # skip non-standard dirs
        if any(s in tag for s in SKIP):
            continue
        parsed = parse_wprime_tag(tag)
        if parsed is None:
            continue
        g, m = parsed

        summary = d / "summary.txt"
        if not summary.exists():
            print(f"  [skip] {tag}: no summary.txt")
            continue

        sigma = read_sigma(summary, k=args.pca_k)
        if sigma is None:
            print(f"  [skip] {tag}: PCA k={args.pca_k} not found in summary")
            continue

        print(f"  {tag:35s}  gWH={g:.3f}  mWp={m:5.1f} TeV  sigma={sigma:.2f}")
        points.append((g, m, sigma))

    if len(points) < 4:
        print(f"\nERROR: only {len(points)} points. Run more pipeline --no-ns points first.")
        sys.exit(1)

    pts    = np.array(points)
    g_vals = pts[:, 0]
    m_vals = pts[:, 1]
    s_vals = np.clip(pts[:, 2], 0, 30)  # cap at 30σ for display

    print(f"\n{len(points)} points found.")

    # ── Interpolation grid ────────────────────────────────────────────────────
    # Use log scale for both axes (physics scales roughly as g²/m²)
    log_g = np.log10(g_vals)
    log_m = np.log10(m_vals)

    log_g_grid = np.linspace(log_g.min(), log_g.max(), 300)
    log_m_grid = np.linspace(log_m.min(), log_m.max(), 300)
    LG, LM     = np.meshgrid(log_g_grid, log_m_grid)

    # RBF interpolation (more robust than cubic griddata for scattered points)
    xy     = np.column_stack([log_g, log_m])
    xy_grd = np.column_stack([LG.ravel(), LM.ravel()])
    try:
        rbf    = RBFInterpolator(xy, s_vals, kernel="thin_plate_spline", smoothing=0.0)
        S_grid = rbf(xy_grd).reshape(LG.shape)
    except Exception:
        S_grid = griddata((log_g, log_m), s_vals, (LG, LM), method="cubic")

    S_grid = np.clip(S_grid, 0, 30)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    # filled contour
    lvls = np.linspace(0, min(30, float(S_grid.max())), 31)
    cf   = ax.contourf(10**LG, 10**LM, S_grid, levels=lvls, cmap="plasma", alpha=0.85)
    cb   = fig.colorbar(cf, ax=ax, label=r"BSM significance  $[\sigma]$  (PCA k=2)")

    # significance contours
    sig_levels = [1, 2, 3, 5, 7, 10, 15, 20]
    sig_levels = [s for s in sig_levels if s <= float(S_grid.max())]
    if sig_levels:
        cs = ax.contour(10**LG, 10**LM, S_grid, levels=sig_levels,
                        colors="white", linewidths=1.0, alpha=0.8)
        ax.clabel(cs, fmt="%dσ", fontsize=9, inline=True)

    # scatter actual data points
    sc = ax.scatter(g_vals, m_vals, c=s_vals, cmap="plasma",
                    vmin=0, vmax=min(30, float(S_grid.max())),
                    s=80, edgecolors="black", linewidths=0.8, zorder=5)

    # annotate each point with its sigma
    for g, m, s in points:
        ax.annotate(f"{s:.1f}σ", (g, m),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=7, color="white", fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$g_{WH}$", fontsize=13)
    ax.set_ylabel(r"$m_{W'}$  [TeV]", fontsize=13)
    ax.set_title(r"W' BSM detection significance in $(g_{WH},\, m_{W'})$ space"
                 "\n" + r"(gWLf = gWqf = gWH/3;  PCA k=2 analytic)",
                 fontsize=11)

    plt.tight_layout()

    out_png = OUTDIR / "sigma_contour_wprime.png"
    out_pdf = OUTDIR / "sigma_contour_wprime.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
