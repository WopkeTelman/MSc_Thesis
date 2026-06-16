#!/usr/bin/env python3
"""
replot_discovery.py

Re-plot a discovery region from an existing discovery_table.txt,
with optional mass/coupling range cuts.

Usage:
    python replot_discovery.py --tag wprime_gwh012_mwp010_scan --mmax 3.0
    python replot_discovery.py --tag zprime_gzh012_mzp010_scan --mmax 5.0
"""

import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

PIPELINE = Path(__file__).parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag",   required=True, help="Results tag (e.g. wprime_gwh012_mwp010_scan)")
    p.add_argument("--mmin",  type=float, default=None)
    p.add_argument("--mmax",  type=float, default=None)
    p.add_argument("--gmin",  type=float, default=None)
    p.add_argument("--gmax",  type=float, default=None)
    args = p.parse_args()

    table_path = PIPELINE / "results" / args.tag / "discovery_table.txt"
    if not table_path.exists():
        print(f"ERROR: {table_path} not found")
        return

    # Parse table
    rows = []
    with open(table_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            rows.append({
                "g":          float(vals[0]),
                "m":          float(vals[1]),
                "sigma_uv":   float(vals[5]),
                "sigma_full": float(vals[4]),
            })

    # Apply cuts
    mmin = args.mmin or min(r["m"] for r in rows)
    mmax = args.mmax or max(r["m"] for r in rows)
    gmin = args.gmin or min(r["g"] for r in rows)
    gmax = args.gmax or max(r["g"] for r in rows)

    rows = [r for r in rows if mmin <= r["m"] <= mmax and gmin <= r["g"] <= gmax]

    coupling_grid = sorted(set(r["g"] for r in rows))
    mass_grid     = sorted(set(r["m"] for r in rows))
    nc, nm = len(coupling_grid), len(mass_grid)

    sigma_uv   = np.zeros((nc, nm))
    sigma_full = np.zeros((nc, nm))
    for r in rows:
        ic = coupling_grid.index(r["g"])
        im = mass_grid.index(r["m"])
        sigma_uv[ic, im]   = min(r["sigma_uv"],   50.0)
        sigma_full[ic, im] = min(r["sigma_full"],  50.0)

    mass_arr     = np.array(mass_grid)
    coupling_arr = np.array(coupling_grid)

    fig, ax = plt.subplots(figsize=(8, 6))

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

    model_label = args.tag.replace("wprime", "W'").replace("zprime", "Z'")
    ax.set_xlabel(r"$m$ [TeV]", fontsize=14)
    ax.set_ylabel(r"$|g|$", fontsize=14)
    ax.set_title(f"FCC-ee  discovery reach  ({model_label})", fontsize=13)
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
    ax.legend(handles=handles, fontsize=10, loc="upper left", framealpha=0.9, edgecolor="gray")

    plt.tight_layout()

    out_dir = PIPELINE / "results" / args.tag / "plots"
    out_dir.mkdir(exist_ok=True)
    suffix = f"_m{int(mmax*10):03d}" if args.mmax else ""
    out_path = out_dir / f"discovery_region{suffix}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
