#!/usr/bin/env python3
"""
scripts/plot_sigma_vs_pca_k.py

Plot discovery significance σ as a function of the number of PCA modes k.

Each PCA mode is a linear combination of Wilson coefficients defined by the
eigenvectors of the Fisher matrix F = K^T C^{-1} K.  In the k-mode subspace the
profile-likelihood test statistic is:

    q_k = Σ_{i=1}^{k} (v_i · g)² / λ_i

where v_i are the PCA eigenvectors, λ_i the eigenvalues, and g = K^T C^{-1} Δd
is the signal vector projected onto operator space.  Significance is obtained via
Wilks' theorem: σ_k = Φ⁻¹(1 − p(χ²(k) > q_k)).

This plot answers the supervisor's question: "At which k does the likelihood stop
improving?"  Answer: σ is maximised at k=1 (the leading PCA direction carries
~90% of signal with only 1 dof), and decreases monotonically thereafter because
each additional mode adds one degree of freedom but contributes negligible signal.

Usage
-----
    python plot_sigma_vs_pca_k.py                    # three standard W' benchmarks
    python plot_sigma_vs_pca_k.py --out /some/dir    # custom output directory
    python plot_sigma_vs_pca_k.py --zprime           # include Z' benchmark

Outputs
-------
    <out>/sigma_vs_pca_k.pdf
    <out>/sigma_vs_pca_k.png
    <out>/sigma_vs_pca_k_table.txt
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm as norm_dist, chi2 as chi2_dist

# ---------------------------------------------------------------------------
RESULTS = Path(__file__).parent.parent / "results"
SCRIPTS = Path(__file__).parent

# ---------------------------------------------------------------------------

def compute_sigma_vs_k(pca_dir):
    """
    Load PCA arrays from pca_dir and return (k_arr, sigma_arr, q_arr, frac_arr, q_full).

    k_arr    : 1-based mode count
    sigma_arr: Wilks sigma at each k
    q_arr    : cumulative q_k at each k
    frac_arr : fraction of q_full recovered at each k (%)
    q_full   : q using all modes (= chi2_SM from profile likelihood)
    """
    p = Path(pca_dir)
    K  = np.load(p / "K_fit.npy")          # (n_dat, n_op)
    Ci = np.load(p / "C_inv.npy")          # (n_dat, n_dat)
    dd = np.load(p / "data_delta.npy")     # (n_dat,)
    ev = np.load(p / "eigenvectors.npy")   # (n_op, n_op), columns = eigenvectors
    el = np.load(p / "eigenvalues.npy")    # (n_op,) eigenvalues descending

    F      = K.T @ Ci @ K                       # (n_op, n_op)
    g      = K.T @ Ci @ dd                      # (n_op,)
    q_full = float(g @ np.linalg.pinv(F, rcond=1e-10) @ g)
    proj   = ev.T @ g                            # projections onto each PC

    n_op = len(el)
    k_arr    = np.arange(1, n_op + 1)
    sigma_arr = np.zeros(n_op)
    q_arr     = np.zeros(n_op)
    frac_arr  = np.zeros(n_op)
    q_cum = 0.0
    for i in range(n_op):
        contrib = proj[i] ** 2 / el[i] if el[i] > 1e-10 else 0.0
        q_cum  += contrib
        q_arr[i]    = q_cum
        frac_arr[i] = 100.0 * q_cum / q_full if q_full > 0 else 0.0
        p_val = chi2_dist.sf(q_cum, i + 1)
        sigma_arr[i] = float(norm_dist.isf(p_val)) if p_val > 1e-300 else 99.0

    return k_arr, sigma_arr, q_arr, frac_arr, q_full


def load_summary_sigmas(summary_path):
    """Return dict method->sigma from a summary.txt file (float or None)."""
    vals = {}
    try:
        with open(summary_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("Best 1-op"):
                    vals["1op"]  = float(line.split()[3])
                elif line.startswith("Full SMEFT"):
                    vals["full"] = float(line.split()[3])
                elif line.startswith("UV coupling"):
                    try:
                        vals["uv"] = float(line.split()[3])
                    except ValueError:
                        vals["uv"] = None
    except FileNotFoundError:
        pass
    return vals


def plot(benchmarks, out_dir, title_suffix=""):
    """
    benchmarks : list of dicts with keys:
        label   (str)  — legend label
        pca_dir (Path) — path to pca/ directory
        summary (Path) — path to summary.txt (optional, for reference lines)
        color   (str)  — matplotlib colour
        ls      (str)  — linestyle
    out_dir : Path — output directory
    """
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax_sig, ax_frac = axes

    table_rows = []

    rank_marked = False
    for bm in benchmarks:
        k_arr, sigma_arr, q_arr, frac_arr, q_full = compute_sigma_vs_k(bm["pca_dir"])
        ref = load_summary_sigmas(bm["summary"]) if bm.get("summary") else {}

        lbl  = bm["label"]
        col  = bm["color"]
        ls   = bm.get("ls", "-")

        # Determine rank (non-degenerate modes)
        el   = np.load(Path(bm["pca_dir"]) / "eigenvalues.npy")
        rank = int(np.sum(el > 1e-4))

        # Plot only up to rank+1 to keep the graph readable
        k_plot = k_arr[:rank]
        s_plot = sigma_arr[:rank]
        f_plot = frac_arr[:rank]

        # --- sigma vs k ---
        ax_sig.plot(k_plot, s_plot, color=col, ls=ls, lw=2.0, marker="o",
                    markersize=4, label=lbl)
        # mark k=1 with a larger dot
        ax_sig.scatter([1], [sigma_arr[0]], color=col, s=100, zorder=6)
        # UV coupling reference (dashed horizontal, labelled once)
        if "uv" in ref and ref["uv"] is not None:
            ax_sig.axhline(ref["uv"], color=col, ls="--", lw=1.2, alpha=0.55)

        # --- fraction vs k ---
        ax_frac.plot(k_plot, f_plot, color=col, ls=ls, lw=2.0, marker="o",
                     markersize=4, label=lbl)
        ax_frac.scatter([1], [frac_arr[0]], color=col, s=100, zorder=6)

        # Mark full-SMEFT boundary (rank) once with a grey vertical line
        if not rank_marked:
            ax_sig.axvline(rank, color="gray", ls="-.", lw=1.2, alpha=0.5,
                           label=f"Full SMEFT dof ($k$={rank})")
            ax_frac.axvline(rank, color="gray", ls="-.", lw=1.2, alpha=0.5,
                            label=f"Full SMEFT dof ($k$={rank})")
            rank_marked = True

        # Table
        table_rows.append(f"\n# {lbl}  (q_full={q_full:.3f})")
        table_rows.append(f"  {'k':>3}  {'q_k':>10}  {'sigma':>8}  {'frac%':>7}")
        for i in range(len(k_arr)):
            table_rows.append(
                f"  {k_arr[i]:3d}  {q_arr[i]:10.4f}  {sigma_arr[i]:8.4f}  {frac_arr[i]:6.2f}%"
            )

    # --- formatting sigma panel ---
    ax_sig.axhline(5.0, color="gray", ls="--", lw=1.0, alpha=0.7, label="5σ threshold")
    ax_sig.axhline(3.0, color="gray", ls=":",  lw=1.0, alpha=0.7, label="3σ threshold")
    ax_sig.set_xlabel("Number of PCA modes $k$", fontsize=13)
    ax_sig.set_ylabel("Discovery significance $\\sigma$ [Wilks]", fontsize=13)
    ax_sig.set_title(f"Significance vs PCA dimensionality{title_suffix}", fontsize=12)
    ax_sig.legend(fontsize=9, loc="upper right")
    ax_sig.set_xlim(0.5, None)
    ax_sig.set_ylim(bottom=0)
    ax_sig.grid(True, alpha=0.3)

    # --- formatting fraction panel ---
    ax_frac.axhline(95, color="gray", ls="--", lw=1.0, alpha=0.7, label="95%")
    ax_frac.axhline(99, color="gray", ls=":",  lw=1.0, alpha=0.7, label="99%")
    ax_frac.set_xlabel("Number of PCA modes $k$", fontsize=13)
    ax_frac.set_ylabel("Fraction of $q_{\\rm full}$ recovered [%]", fontsize=13)
    ax_frac.set_title(f"Signal recovery vs PCA dimensionality{title_suffix}", fontsize=12)
    ax_frac.legend(fontsize=9, loc="lower right")
    ax_frac.set_xlim(0.5, None)
    ax_frac.set_ylim(80, 101)
    ax_frac.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"sigma_vs_pca_k.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir}/sigma_vs_pca_k.{{pdf,png}}")

    # Write table
    tbl_path = out_dir / "sigma_vs_pca_k_table.txt"
    with open(tbl_path, "w") as f:
        f.write("# Significance vs PCA k — W' model at FCC-ee\n")
        f.write("# Wilks: q_k ~ chi2(k) approximation\n")
        f.write("# Dotted reference lines in sigma plot = Full SMEFT significance\n")
        f.write("\n".join(table_rows))
    print(f"  Saved: {tbl_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None,
                    help="Output directory (default: results/sigma_vs_pca_k/)")
    ap.add_argument("--zprime", action="store_true",
                    help="Add Z' benchmark panel")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else RESULTS / "sigma_vs_pca_k"

    # Standard three W' benchmarks
    w_benchmarks = [
        {
            "label":   r"W$'$: $g_{WH}=0.12$, $m=1$ TeV",
            "pca_dir": RESULTS / "wprime_gwh012_mwp010" / "pca",
            "summary": RESULTS / "wprime_gwh012_mwp010" / "summary.txt",
            "color":   "#1f77b4",
            "ls":      "-",
        },
        {
            "label":   r"W$'$: $g_{WH}=1.0$, $m=10$ TeV",
            "pca_dir": RESULTS / "wprime_gwh100_mwp100" / "pca",
            "summary": RESULTS / "wprime_gwh100_mwp100" / "summary.txt",
            "color":   "#ff7f0e",
            "ls":      "-",
        },
        {
            "label":   r"W$'$: $g_{WH}=0.08$, $m=1$ TeV",
            "pca_dir": RESULTS / "wprime_gwh008_mwp010" / "pca",
            "summary": RESULTS / "wprime_gwh008_mwp010" / "summary.txt",
            "color":   "#2ca02c",
            "ls":      "-",
        },
    ]

    if args.zprime:
        zp_pca = RESULTS / "zprime_gzh012_mzp010" / "pca"
        if zp_pca.exists():
            w_benchmarks.append({
                "label":   r"Z$'$: $g_{ZH}=0.12$, $m=1$ TeV",
                "pca_dir": zp_pca,
                "summary": RESULTS / "zprime_gzh012_mzp010" / "summary.txt",
                "color":   "#9467bd",
                "ls":      "--",
            })
        else:
            print(f"  Warning: Z' pca directory not found: {zp_pca}")

    print(f"\nGenerating sigma vs PCA k plot → {out_dir}")
    plot(w_benchmarks, out_dir)

    # Print summary to stdout
    # k=rank = number of non-degenerate eigenmodes (= actual Full SMEFT dof)
    print("\n  Summary at k=1, 2, 4, rank (= Full SMEFT dof):")
    print(f"  {'Benchmark':<40} {'k=1':>7} {'k=2':>7} {'k=4':>7} {'k=rank':>8}")
    for bm in w_benchmarks:
        k_arr, sigma_arr, q_arr, _, qf = compute_sigma_vs_k(bm["pca_dir"])
        el = np.load(Path(bm["pca_dir"]) / "eigenvalues.npy")
        rank = int(np.sum(el > 1e-4))  # gap at ~1e-9: PC14=2.7e-2, PC15=3.4e-9
        n = len(sigma_arr)
        s1 = sigma_arr[0]
        s2 = sigma_arr[1] if n >= 2 else float("nan")
        s4 = sigma_arr[3] if n >= 4 else float("nan")
        sr = sigma_arr[rank - 1] if rank <= n else float("nan")
        print(f"  {bm['label']:<40} {s1:7.3f} {s2:7.3f} {s4:7.3f} {sr:8.3f}  (rank={rank})")


if __name__ == "__main__":
    main()
