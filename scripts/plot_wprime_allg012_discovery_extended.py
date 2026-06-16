#!/usr/bin/env python3
"""
plot_wprime_allg012_discovery_extended.py

Analytic discovery region for W' with universal coupling (all g = gWH).
Extended grid so the 5sigma contour spans corner to corner.

Saves to results/wprime_gwh012_allg012_mwp010_scan/plots/discovery_region_extended.png
"""

import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.stats import chi2 as chi2_dist, norm as norm_dist
from pathlib import Path

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(Path(__file__).parent))

from models.wprime import WPrimeModel
from scripts.run_pipeline import _build_K_Ci


def q_to_sigma(q, ndof):
    if q <= 0:
        return 0.0
    p = chi2_dist.sf(q, df=ndof)
    return float(norm_dist.isf(p / 2)) if p > 0 else 99.0


def main():
    # Grid bounds: chosen so 5sigma contour goes corner to corner
    # At gWH=0.36, 5sigma hits mWp ~ 4.3 TeV → set mWp_max=4.5
    gWH_min = 0.001
    gWH_max = 0.36
    mWp_min = 0.3
    mWp_max = 4.5
    ng, nm  = 80, 100

    ops = WPrimeModel().OPERATORS

    print(f"Building K, Ci  (ops: {ops})")
    K, Ci = _build_K_Ci(ops)

    # SMEFT rank for ndof_full
    F = K.T @ Ci @ K
    ndof_full = int(np.sum(np.linalg.eigvalsh(F) > 1e-8 * np.linalg.eigvalsh(F).max()))
    ndof_uv   = 5   # gWH, gWLf11, gWLf22, gWLf33, gWqf33
    print(f"  K: {K.shape}   Ci: {Ci.shape}   SMEFT rank: {ndof_full}   UV ndof: {ndof_uv}")

    coupling_grid = np.linspace(gWH_min, gWH_max, ng)
    mass_grid     = np.linspace(mWp_min, mWp_max, nm)

    print(f"Computing {ng}x{nm} grid ...")
    sigma_uv   = np.zeros((ng, nm))
    sigma_full = np.zeros((ng, nm))

    for ic, gWH in enumerate(coupling_grid):
        for im, mWp in enumerate(mass_grid):
            model   = WPrimeModel(gWH=gWH, gWLf11=gWH, gWLf22=gWH,
                                  gWLf33=gWH, gWqf33=gWH, mWp=mWp)
            c_truth = np.array([model.eft_coefficients().get(op, 0.0) for op in ops])
            Kc      = K @ c_truth
            q       = float(Kc @ Ci @ Kc)
            sigma_uv[ic, im]   = q_to_sigma(q, ndof_uv)
            sigma_full[ic, im] = q_to_sigma(q, ndof_full)

    sigma_uv   = np.clip(sigma_uv,   0, 50)
    sigma_full = np.clip(sigma_full, 0, 50)

    mass_arr     = np.array(mass_grid)
    coupling_arr = np.array(coupling_grid)

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
    ax.set_title(r"FCC-ee  discovery reach  (W'$,\ g_{\rm Lf}=g_{\rm qf}=g_{WH}$)",
                 fontsize=13)
    ax.set_xlim(mWp_min, mWp_max)
    ax.set_ylim(0, gWH_max)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=11)
    ax.grid(True, which="major", alpha=0.15, lw=0.5)

    ax.text(0.97, 0.97, r"Discoverable ($>5\sigma$)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color="#1f77b4", style="italic")
    ax.text(0.97, 0.03, "Not discoverable",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color="gray", style="italic")

    handles = [
        Line2D([0],[0], color="#1f77b4", lw=2.5, label=r"UV: $5\sigma$ discovery"),
        Line2D([0],[0], color="#1f77b4", lw=1.5, ls="--", label=r"UV: $3\sigma$ evidence"),
        Line2D([0],[0], color="#ff7f0e", lw=1.5, ls=":", label=r"Full SMEFT: $5\sigma$"),
    ]
    ax.legend(handles=handles, fontsize=10, loc="upper left",
              framealpha=0.9, edgecolor="gray")

    plt.tight_layout()

    out_dir = PIPELINE / "results" / "wprime_gwh012_allg012_mwp010_scan" / "plots"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "discovery_region_extended.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
