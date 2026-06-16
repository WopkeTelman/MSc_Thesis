"""
Replot discovery_region_extended with a clean title and no in-plot labels.
Computes an 80x100 analytical grid (same as run_pipeline) so contours are smooth.

Usage:
    python scripts/replot_discovery_clean.py \
        --pca   results/wprime_gwh020_mwp050/pca \
        --gWH   0.20 --mWp 5.0 \
        --out   results/wprime_gwh020_mwp050_scan/plots/discovery_region_extended.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import norm as norm_dist, chi2 as chi2_dist
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.wprime import WPrimeModel


def q_to_sigma(q, ndof):
    p = chi2_dist.sf(float(q), df=ndof)
    p = max(p, 1e-300)
    return float(norm_dist.isf(p / 2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca",  required=True, help="path to pca directory")
    ap.add_argument("--gWH",  type=float, required=True)
    ap.add_argument("--mWp",  type=float, required=True)
    ap.add_argument("--out",  required=True)
    args = ap.parse_args()

    pca = Path(args.pca)
    K   = np.load(pca / "K_fit.npy")
    Ci  = np.load(pca / "C_inv.npy")

    g0, m0 = args.gWH, args.mWp
    gWLf   = g0 / 3.0
    model_ref = WPrimeModel(gWH=g0, gWLf11=gWLf, gWLf22=gWLf,
                            gWLf33=gWLf, gWqf33=gWLf, mWp=m0)
    ops    = model_ref.OPERATORS
    ndof_uv = len(model_ref.uv_param_names())

    F       = K.T @ Ci @ K
    eigs    = np.linalg.eigvalsh(F)
    ndof_full = int(np.sum(eigs > 1e-8 * eigs.max()))

    c_ref  = np.array([model_ref.eft_coefficients().get(op, 0.0) for op in ops])
    q_ref  = float((K @ c_ref) @ Ci @ (K @ c_ref))

    p_5s     = 2.0 * norm_dist.sf(5.0)
    q_5s_uv  = chi2_dist.ppf(1.0 - p_5s, df=ndof_uv)

    gWH_max = 0.60
    mWp_max = m0 * (gWH_max / g0) * (q_ref / q_5s_uv) ** 0.25 * 1.15
    mWp_min = max(0.3, m0 * 0.03)

    coupling_arr = np.linspace(0.001, gWH_max, 80)
    mass_arr     = np.linspace(mWp_min, mWp_max, 100)

    sigma_uv   = np.zeros((80, 100))
    sigma_full = np.zeros((80, 100))

    for ic, g in enumerate(coupling_arr):
        lf = g / 3.0
        for im, m in enumerate(mass_arr):
            model = WPrimeModel(gWH=g, gWLf11=lf, gWLf22=lf,
                                gWLf33=lf, gWqf33=lf, mWp=m)
            c = np.array([model.eft_coefficients().get(op, 0.0) for op in ops])
            q = float((K @ c) @ Ci @ (K @ c))
            sigma_uv[ic, im]   = q_to_sigma(q, ndof_uv)
            sigma_full[ic, im] = q_to_sigma(q, ndof_full)

    sigma_uv   = np.clip(sigma_uv,   0, 50)
    sigma_full = np.clip(sigma_full, 0, 50)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.contourf(mass_arr, coupling_arr, sigma_uv,
                levels=[5.0, 1e4], colors=["#BDD7EE"], alpha=0.7)
    ax.contourf(mass_arr, coupling_arr, sigma_uv,
                levels=[0, 3.0],   colors=["#DCDCDC"], alpha=0.6)
    ax.contourf(mass_arr, coupling_arr, sigma_uv,
                levels=[3.0, 5.0], colors=["#E8F4FD"], alpha=0.5)

    cs5 = ax.contour(mass_arr, coupling_arr, sigma_uv,
                     levels=[5.0], colors=["#1f77b4"], linewidths=[2.5])
    ax.clabel(cs5, fmt=r"$5\sigma$", fontsize=11, inline=True, inline_spacing=8)

    cs3 = ax.contour(mass_arr, coupling_arr, sigma_uv,
                     levels=[3.0], colors=["#1f77b4"], linewidths=[1.5],
                     linestyles=["--"])
    ax.clabel(cs3, fmt=r"$3\sigma$", fontsize=11, inline=True, inline_spacing=8)

    cs5f = ax.contour(mass_arr, coupling_arr, sigma_full,
                      levels=[5.0], colors=["#ff7f0e"], linewidths=[1.5],
                      linestyles=[":"])
    ax.clabel(cs5f, fmt=r"$5\sigma$ (SMEFT)", fontsize=9, inline=True)

    ax.set_xlabel(r"$m_{W'}$ [TeV]", fontsize=14)
    ax.set_ylabel(r"$|g_{WH}|$",     fontsize=14)
    ax.set_title(r"FCC-ee  W'",      fontsize=13)
    ax.set_xlim(mWp_min, mWp_max)
    ax.set_ylim(0, gWH_max)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=11)
    ax.grid(True, which="major", alpha=0.15, lw=0.5)

    handles = [
        Line2D([0],[0], color="#1f77b4", lw=2.5,        label=r"UV: $5\sigma$"),
        Line2D([0],[0], color="#1f77b4", lw=1.5, ls="--", label=r"UV: $3\sigma$"),
        Line2D([0],[0], color="#ff7f0e", lw=1.5, ls=":",  label=r"Full SMEFT: $5\sigma$"),
    ]
    ax.legend(handles=handles, fontsize=10, loc="upper left",
              framealpha=0.9, edgecolor="gray")

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
