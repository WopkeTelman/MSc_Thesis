"""
L1 pull distribution for the BSM closure test.

Generates N_rep Gaussian noise replicas δ^(r) = δ_BSM + ε^(r), ε ~ N(0,C),
and fits each analytically using the linear Fisher-matrix approximation:

    c^fit(r) = F^{-1} K^T C^{-1} δ^(r),   F = K^T C^{-1} K

Pull for replica r, operator i:
    P_i^(r) = (c_i^fit(r) - c_i^truth) / σ_i^fit

where σ_i^fit = sqrt(F^{-1}_{ii}) is identical for all replicas (posterior
width under flat prior, Gaussian approximation).

Under the Gaussian model this distribution is provably N(0,1).
L0 (single noiseless dataset) gives std≈0 (trivially).
L1 replicas should give std≈1 if the fit is well-calibrated.

Usage:
    python scripts/l1_pull_distribution.py
    python scripts/l1_pull_distribution.py --model wprime_constrained --gWH 0.12 --mWp 3.0 --nrep 500
    python scripts/l1_pull_distribution.py --mWp 4.0 --nrep 1000
"""
import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as norm_dist
from pathlib import Path

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))

from models.wprime_constrained import WPrimeConstrainedModel
from models.zprime import ZPrimeModel


def load_pca_cache(pca_dir):
    pca_dir = Path(pca_dir)
    K  = np.load(pca_dir / "K_fit.npy")
    Ci = np.load(pca_dir / "C_inv.npy")
    return K, Ci


def run_l1_pulls(K, Ci, c_truth_vec, n_rep, seed=42):
    """
    Returns pulls array of shape (n_rep, n_ops).

    c_truth_vec : (n_ops,) truth coefficients ordered as K columns.
    """
    rng = np.random.default_rng(seed)
    n_data, n_ops = K.shape

    F     = K.T @ Ci @ K
    F_inv = np.linalg.pinv(F)              # pinv handles near-singular F safely
    sigma = np.sqrt(np.abs(np.diag(F_inv)))  # posterior width, same for all replicas

    # covariance for noise generation: C = Ci^{-1}
    C = np.linalg.inv(Ci)
    L = np.linalg.cholesky(C)              # C = L L^T

    # F^{-1} K^T Ci  (n_ops × n_data) — maps noise to coefficient error
    A = F_inv @ K.T @ Ci

    # generate all replicas at once: noise shape (n_data, n_rep)
    noise = L @ rng.standard_normal((n_data, n_rep))

    # coefficient errors: (n_ops, n_rep) → transpose to (n_rep, n_ops)
    dc = (A @ noise).T

    # pulls
    pulls = dc / sigma[np.newaxis, :]     # (n_rep, n_ops)
    return pulls, sigma


def print_l1_summary(pulls, ops):
    pull_vals = pulls.flatten()
    print(f"\nL1 pull summary  (n_rep={pulls.shape[0]}, n_ops={len(ops)}, total={len(pull_vals)} pulls)")
    print(f"  Mean : {pull_vals.mean():+.4f}  (should be  0)")
    print(f"  Std  : {pull_vals.std(ddof=1):.4f}  (should be  1)")
    print(f"  |P|>1: {(np.abs(pull_vals)>1).sum()}/{len(pull_vals)}"
          f"  (expect {int(0.317*len(pull_vals))})")
    print(f"  |P|>2: {(np.abs(pull_vals)>2).sum()}/{len(pull_vals)}"
          f"  (expect {int(0.046*len(pull_vals))})")
    print()
    print(f"{'Operator':<14} {'mean_P':>9} {'std_P':>9}")
    print("-" * 36)
    for i, op in enumerate(ops):
        p = pulls[:, i]
        print(f"{op:<14} {p.mean():>9.3f} {p.std(ddof=1):>9.3f}")


def plot_l1_pulls(pulls, ops, out_path, tag):
    pull_vals = pulls.flatten()
    per_op_mean = pulls.mean(axis=0)
    per_op_std  = pulls.std(axis=0, ddof=1)
    n_ops = len(ops)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # ── left: per-operator mean pull ──────────────────────────────────────────
    ax = axes[0]
    colors = ["tomato" if abs(m) > 2*s/np.sqrt(pulls.shape[0]) * 2 else "steelblue"
              for m, s in zip(per_op_mean, per_op_std)]
    err = per_op_std / np.sqrt(pulls.shape[0])
    ax.barh(range(n_ops), per_op_mean, xerr=err, color=colors,
            alpha=0.8, edgecolor="white", capsize=3)
    ax.axvline(0, color="black", lw=1.0)
    ax.axvline( 0.1, color="gray", lw=0.8, ls="--")
    ax.axvline(-0.1, color="gray", lw=0.8, ls="--")
    ax.set_yticks(range(n_ops))
    ax.set_yticklabels(ops, fontsize=10)
    ax.set_xlabel(r"$\langle P_i \rangle$  (mean pull per operator)", fontsize=11)
    ax.set_title("Per-operator mean pull", fontsize=12)
    ax.set_xlim(-0.5, 0.5)

    # ── middle: per-operator pull width ───────────────────────────────────────
    ax = axes[1]
    colors2 = ["tomato" if abs(s - 1) > 0.2 else "steelblue" for s in per_op_std]
    ax.barh(range(n_ops), per_op_std, color=colors2, alpha=0.8, edgecolor="white")
    ax.axvline(1.0, color="black", lw=1.2, label=r"$\sigma=1$ target")
    ax.axvline(0.8, color="gray", lw=0.8, ls="--")
    ax.axvline(1.2, color="gray", lw=0.8, ls="--")
    ax.set_yticks(range(n_ops))
    ax.set_yticklabels(ops, fontsize=10)
    ax.set_xlabel(r"$\mathrm{std}(P_i)$  (pull width per operator)", fontsize=11)
    ax.set_title("Per-operator pull width", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 2)

    # ── panel 3: combined histogram vs N(0,1) ────────────────────────────────
    ax = axes[2]
    bins = np.linspace(-4, 4, 20)
    ax.hist(pull_vals, bins=bins, density=True, alpha=0.7,
            color="steelblue", edgecolor="white",
            label=f"L1 pulls (n={len(pull_vals)})")
    x = np.linspace(-4, 4, 300)
    ax.plot(x, norm_dist.pdf(x), "r-", lw=2, label=r"$\mathcal{N}(0,1)$")
    ax.axvline(pull_vals.mean(), color="steelblue", lw=2, ls="--",
               label=fr"Mean={pull_vals.mean():.3f}")
    ax.set_xlabel(r"Pull $P_i^{(r)}$", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    mu, sig = pull_vals.mean(), pull_vals.std(ddof=1)
    ax.set_title(fr"Combined L1 pulls  ($\mu={mu:.3f},\ \sigma={sig:.3f}$)",
                 fontsize=12)
    ax.legend(fontsize=10)

    # ── panel 4: total chi2 per replica vs chi2(n_ops) ───────────────────────
    # Q_r = sum_i P_{i,r}^2 = total chi2 of the fit for replica r.
    # Under a correctly calibrated Gaussian model: Q_r ~ chi2(n_ops).
    # Standardised: Z_r = (Q_r - n_ops) / sqrt(2*n_ops) ~ N(0,1).
    ax = axes[3]
    Q_rep = (pulls ** 2).sum(axis=1)          # (n_rep,)  ~ chi2(n_ops)
    Z_rep = (Q_rep - n_ops) / np.sqrt(2 * n_ops)  # standardised ~ N(0,1)

    ax.hist(Z_rep, bins=25, density=True, alpha=0.75,
            color="mediumseagreen", edgecolor="white",
            label=fr"$Z_r = (Q_r - n_{{ops}})/\sqrt{{2n_{{ops}}}}$  (n={len(Z_rep)})")
    x2 = np.linspace(Z_rep.min() - 1, Z_rep.max() + 1, 300)
    ax.plot(x2, norm_dist.pdf(x2), "r-", lw=2, label=r"$\mathcal{N}(0,1)$ expected")
    ax.axvline(Z_rep.mean(), color="green", lw=2, ls="--",
               label=fr"Mean = {Z_rep.mean():.3f}")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(
        r"Standardised total $\chi^2$:  $Z_r = (Q_r - n_\mathrm{ops})\,/\,\sqrt{2\,n_\mathrm{ops}}$",
        fontsize=10)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        fr"Total fit $\chi^2$ per replica  ($n_{{ops}}={n_ops}$)"
        f"\n" + fr"$\langle Q_r\rangle = {Q_rep.mean():.1f}$, "
        fr"$\langle Z_r\rangle = {Z_rep.mean():.3f}$, "
        fr"$\sigma(Z_r) = {Z_rep.std(ddof=1):.3f}$",
        fontsize=10)
    ax.legend(fontsize=9)

    fig.suptitle(f"L1 BSM closure test — {tag}", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{out_path}.png", bbox_inches="tight", dpi=150)
    plt.savefig(f"{out_path}.pdf", bbox_inches="tight")
    print(f"Plot saved: {out_path}.png")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="wprime_constrained",
                   choices=["wprime_constrained", "zprime"])
    p.add_argument("--gWH",  type=float, default=0.12)
    p.add_argument("--mWp",  type=float, default=3.0)
    p.add_argument("--gZH",  type=float, default=0.12)
    p.add_argument("--mZp",  type=float, default=1.0)
    p.add_argument("--nrep", type=int,   default=500)
    p.add_argument("--seed", type=int,   default=42)
    args = p.parse_args()

    if args.model == "wprime_constrained":
        gWLf  = args.gWH / 3
        model = WPrimeConstrainedModel(gWH=args.gWH, gWLf11=gWLf, gWLf22=gWLf,
                                       gWLf33=gWLf, gWqf33=gWLf, mWp=args.mWp)
        tag     = f"W' (gWH={args.gWH}, mWp={args.mWp} TeV)"
        run_tag = f"wprime_constrained_gwh{int(args.gWH*100):03d}_mwp{int(args.mWp*10):03d}"
    else:
        model   = ZPrimeModel(gZH=args.gZH, gZl=args.gZH/3, mZp=args.mZp)
        tag     = f"Z' (gZH={args.gZH}, mZp={args.mZp} TeV)"
        run_tag = f"zprime_gzh{int(args.gZH*100):03d}_mzp{int(args.mZp*10):03d}"

    pca_dir  = PIPELINE / "results" / run_tag / "pca"
    out_dir  = PIPELINE / "results" / run_tag / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model   : {model}")
    print(f"Run tag : {run_tag}")
    print(f"N_rep   : {args.nrep}")

    from run_pipeline import _build_K_Ci
    ops   = model.OPERATORS
    K, Ci = _build_K_Ci(ops)   # always rebuild fresh — K_fit.npy may be stale
    truth = model.eft_coefficients()

    # truth vector aligned with K columns (model.OPERATORS order)
    c_truth_vec = np.array([truth.get(op, 0.0) for op in ops])

    print(f"K shape : {K.shape}  (n_data × n_ops)")
    print(f"Operators: {ops}")

    pulls, sigma = run_l1_pulls(K, Ci, c_truth_vec, n_rep=args.nrep, seed=args.seed)

    print_l1_summary(pulls, ops)

    plot_l1_pulls(pulls, ops,
                  str(out_dir / "l1_pull_distribution"), tag)


if __name__ == "__main__":
    main()
