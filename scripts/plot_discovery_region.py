#!/usr/bin/env python3
"""
plot_discovery_region.py

Analytically compute and plot the BSM discovery region in (mass, coupling) space
for W' or Z' models, using the precomputed K matrix and C^{-1} from the PCA step.

No smefit calls. Runs in seconds on a fine grid.

Physics:
    For injected signal c_truth(g, m), the profile likelihood test statistic is
        q(g, m) = (K · c)^T · C^{-1} · (K · c)
    which equals chi2_SM at the truth point.  Significance = Gaussian sigma
    converted from the chi2(ndof) p-value.

Usage:
    python plot_discovery_region.py --model wprime
    python plot_discovery_region.py --model zprime
    python plot_discovery_region.py --model wprime --ng 60 --nm 80
"""

import os, sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import chi2 as chi2_dist, norm as norm_dist
from pathlib import Path

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(Path(__file__).parent))


def q_to_sigma(q, k):
    if q <= 0:
        return 0.0
    p = chi2_dist.sf(q, df=k)
    return float(norm_dist.isf(p / 2)) if p > 0 else 99.0


def compute_grid(model_cls, base_params, coupling_key, mass_key,
                 coupling_grid, mass_grid, K, Ci, ops):
    """
    Analytically compute significance on a (coupling, mass) grid.
    Returns sigma_uv array of shape (n_coupling, n_mass).
    """
    nc = len(coupling_grid)
    nm = len(mass_grid)
    sigma = np.zeros((nc, nm))

    for ic, g_val in enumerate(coupling_grid):
        for im, m_val in enumerate(mass_grid):
            params = {**base_params, coupling_key: g_val, mass_key: m_val}
            # Scale all fermion couplings with gWH for W' so signal ∝ g²/m²
            if "gWLf11" in params:
                ratio = g_val / base_params.get(coupling_key, g_val) if base_params.get(coupling_key, g_val) != 0 else 1.0
                for fkey in ["gWLf11", "gWLf22", "gWLf33", "gWqf33"]:
                    if fkey in params:
                        params[fkey] = base_params[fkey] * ratio

            model   = model_cls(**params)
            c_truth = np.array([model.eft_coefficients().get(op, 0.0) for op in ops])
            Kc      = K @ c_truth
            q       = float(Kc @ Ci @ Kc)
            ndof    = len(model.uv_param_names()) if hasattr(model, "uv_param_names") else 2
            sigma[ic, im] = q_to_sigma(q, ndof)

    return sigma


def make_plot(sigma_uv, coupling_grid, mass_grid, coupling_key, mass_key,
              tag, out_path, sigma_full=None):

    mass_arr     = np.array(mass_grid)
    coupling_arr = np.array(coupling_grid)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Filled regions
    ax.contourf(mass_arr, coupling_arr, sigma_uv,
                levels=[5.0, 1e4], colors=["#BDD7EE"], alpha=0.7)
    ax.contourf(mass_arr, coupling_arr, sigma_uv,
                levels=[0, 3.0],   colors=["#DCDCDC"], alpha=0.6)
    ax.contourf(mass_arr, coupling_arr, sigma_uv,
                levels=[3.0, 5.0], colors=["#E8F4FD"], alpha=0.5)

    # Contour lines
    cs5 = ax.contour(mass_arr, coupling_arr, sigma_uv,
                     levels=[5.0], colors=["#1f77b4"], linewidths=[2.5])
    ax.clabel(cs5, fmt=r"$5\sigma$", fontsize=11, inline=True, inline_spacing=8)

    cs3 = ax.contour(mass_arr, coupling_arr, sigma_uv,
                     levels=[3.0], colors=["#1f77b4"], linewidths=[1.5],
                     linestyles=["--"])
    ax.clabel(cs3, fmt=r"$3\sigma$", fontsize=11, inline=True, inline_spacing=8)

    if sigma_full is not None:
        csf = ax.contour(mass_arr, coupling_arr, sigma_full,
                         levels=[5.0], colors=["#ff7f0e"], linewidths=[1.5],
                         linestyles=[":"])
        ax.clabel(csf, fmt=r"$5\sigma$ SMEFT", fontsize=9, inline=True)

    # Axes
    model_label = tag.replace("wprime", "W'").replace("zprime", "Z'")
    ax.set_xlabel(f"$m$ [TeV]", fontsize=14)
    ax.set_ylabel(f"$|g|$", fontsize=14)
    ax.set_title(f"FCC-ee  discovery reach  ({model_label})", fontsize=13)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=11)
    ax.grid(True, which="major", alpha=0.15, lw=0.5)

    # Region labels
    ax.text(0.97, 0.97, "Discoverable  ($>5\\sigma$)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color="#1f77b4", style="italic")
    ax.text(0.97, 0.03, "Not discoverable",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color="gray", style="italic")

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color="#1f77b4", lw=2.5,
               label=r"UV coupling fit: $5\sigma$ discovery"),
        Line2D([0],[0], color="#1f77b4", lw=1.5, ls="--",
               label=r"UV coupling fit: $3\sigma$ evidence"),
    ]
    if sigma_full is not None:
        handles.append(Line2D([0],[0], color="#ff7f0e", lw=1.5, ls=":",
                               label=r"Full SMEFT: $5\sigma$"))
    ax.legend(handles=handles, fontsize=10, loc="upper left",
              framealpha=0.9, edgecolor="gray")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["wprime", "zprime"], required=True)
    p.add_argument("--tag",   default=None)
    p.add_argument("--ng",    type=int, default=50, help="Number of coupling points")
    p.add_argument("--nm",    type=int, default=60, help="Number of mass points")
    p.add_argument("--gmin",  type=float, default=None)
    p.add_argument("--gmax",  type=float, default=None)
    p.add_argument("--mmin",  type=float, default=None)
    p.add_argument("--mmax",  type=float, default=None)
    # W' params
    p.add_argument("--gWH",   type=float, default=0.12)
    p.add_argument("--gWLf",  type=float, default=None)
    p.add_argument("--gWqf",  type=float, default=None)
    p.add_argument("--mWp",   type=float, default=1.0)
    # Z' params
    p.add_argument("--gZH",   type=float, default=0.12)
    p.add_argument("--gZl",   type=float, default=0.04)
    p.add_argument("--mZp",   type=float, default=1.0)
    args = p.parse_args()

    if args.model == "wprime":
        from models.wprime import WPrimeModel
        gWLf        = args.gWLf if args.gWLf is not None else args.gWH / 3
        gWqf        = args.gWqf if args.gWqf is not None else args.gWH / 3
        base_params = {"gWH": args.gWH, "gWLf11": gWLf, "gWLf22": gWLf,
                       "gWLf33": gWLf, "gWqf33": gWqf, "mWp": args.mWp}
        model_cls   = WPrimeModel
        coupling_key, mass_key = "gWH", "mWp"
        tag         = args.tag or f"wprime_gwh{int(args.gWH*100):03d}_mwp{int(args.mWp*10):03d}"
        gmin = args.gmin or 0.02
        gmax = args.gmax or 0.40
        mmin = args.mmin or 0.3
        mmax = args.mmax or 5.0
        pca_tag = tag

    elif args.model == "zprime":
        from models.zprime import ZPrimeModel
        base_params = {"gZH": args.gZH, "gZl": args.gZl, "mZp": args.mZp}
        model_cls   = ZPrimeModel
        coupling_key, mass_key = "gZH", "mZp"
        tag         = args.tag or f"zprime_gzh{int(args.gZH*100):03d}_mzp{int(args.mZp*10):03d}"
        gmin = args.gmin or 0.02
        gmax = args.gmax or 0.40
        mmin = args.mmin or 0.3
        mmax = args.mmax or 5.0
        pca_tag = tag

    # Load precomputed K and Ci from PCA step
    pca_dir = PIPELINE / "results" / pca_tag / "pca"
    if not (pca_dir / "K_fit.npy").exists():
        print(f"ERROR: PCA data not found at {pca_dir}")
        print(f"Run the full pipeline first:  python run_pipeline.py --model {args.model} ...")
        sys.exit(1)

    K   = np.load(pca_dir / "K_fit.npy")
    Ci  = np.load(pca_dir / "C_inv.npy")
    ops = model_cls(**base_params).OPERATORS
    print(f"Loaded K {K.shape}, Ci {Ci.shape}  from {pca_dir}")

    # Fine grids
    coupling_grid = list(np.linspace(gmin, gmax, args.ng))
    mass_grid     = list(np.linspace(mmin, mmax, args.nm))

    print(f"Computing {args.ng} x {args.nm} = {args.ng * args.nm} grid points ...")
    sigma_uv = compute_grid(model_cls, base_params, coupling_key, mass_key,
                            coupling_grid, mass_grid, K, Ci, ops)

    # Also compute Full SMEFT significance (ndof = rank of F)
    import numpy.linalg as la
    F    = K.T @ Ci @ K
    rank = la.matrix_rank(F)
    Fpinv = la.pinv(F)

    sigma_full = np.zeros_like(sigma_uv)
    for ic, g_val in enumerate(coupling_grid):
        for im, m_val in enumerate(mass_grid):
            params = {**base_params, coupling_key: g_val, mass_key: m_val}
            if "gWLf11" in params:
                ratio = g_val / base_params.get(coupling_key, g_val) if base_params.get(coupling_key, g_val) != 0 else 1.0
                for fkey in ["gWLf11", "gWLf22", "gWLf33", "gWqf33"]:
                    if fkey in params:
                        params[fkey] = base_params[fkey] * ratio
            model   = model_cls(**params)
            c_truth = np.array([model.eft_coefficients().get(op, 0.0) for op in ops])
            g_vec   = K.T @ Ci @ (K @ c_truth)
            q_full  = float(g_vec @ Fpinv @ g_vec)
            sigma_full[ic, im] = q_to_sigma(q_full, rank)

    # Output
    out_dir = PIPELINE / "results" / pca_tag / "plots"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / f"discovery_region_analytic_m{int(mmax*10):03d}.png"

    make_plot(sigma_uv, coupling_grid, mass_grid,
              coupling_key, mass_key, tag, str(out_path),
              sigma_full=sigma_full)

    # Print the 5sigma threshold mass at each coupling
    print(f"\n5sigma discovery threshold (UV method):")
    print(f"  {'|g|':>6}  {'m_threshold [TeV]':>20}")
    print(f"  {'-'*30}")
    mass_arr     = np.array(mass_grid)
    coupling_arr = np.array(coupling_grid)
    for ic, g_val in enumerate(coupling_arr):
        row = sigma_uv[ic, :]
        # find where sigma crosses 5
        above = row >= 5.0
        if above.all():
            print(f"  {g_val:>6.3f}  all masses discoverable")
        elif not above.any():
            print(f"  {g_val:>6.3f}  not discoverable")
        else:
            # last mass that is discoverable
            idx = np.where(above)[0][-1]
            m_thresh = mass_arr[idx]
            print(f"  {g_val:>6.3f}  m < {m_thresh:.2f} TeV")


if __name__ == "__main__":
    main()
