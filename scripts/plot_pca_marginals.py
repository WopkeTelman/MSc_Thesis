"""
scripts/plot_pca_marginals.py

Plot marginal histograms of the SMEFT posterior projected onto the top-k PCA eigenvectors.

For each PCA mode e_i:
    alpha_i = e_i^T c_posterior   (projection of each replica onto mode i)

The truth value alpha_i^truth = e_i^T c_truth is overlaid as a vertical line.

Usage:
    python scripts/plot_pca_marginals.py --tag wprime_gwh050_mwp150 --k 5
    python scripts/plot_pca_marginals.py --tag wprime_gwh050_mwp150 --k 5 --model wprime --gWH 0.5 --mWp 15
"""
import sys, argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "scripts"))


def get_model(args):
    if args.model == "wprime_constrained":
        from models.wprime_constrained import WPrimeConstrainedModel
        gWLf = args.gWLf or args.gWH / 3
        return WPrimeConstrainedModel(gWH=args.gWH, gWLf11=gWLf, gWLf22=gWLf,
                                      gWLf33=gWLf, gWqf33=gWLf, mWp=args.mWp)
    if args.model == "wprime_universal":
        from models.wprime_universal import WPrimeUniversalModel
        return WPrimeUniversalModel(gWH=args.gWH, mWp=args.mWp)
    from models.wprime import WPrimeModel
    gWLf = args.gWLf or args.gWH / 3
    return WPrimeModel(gWH=args.gWH, gWLf11=gWLf, gWLf22=gWLf,
                       gWLf33=gWLf, gWqf33=gWLf, mWp=args.mWp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag",   required=True)
    p.add_argument("--k",     type=int, default=5, help="Number of PCA modes to show")
    p.add_argument("--model", default="wprime",
                   choices=["wprime", "wprime_constrained", "wprime_universal"])
    p.add_argument("--gWH",  type=float, default=0.5)
    p.add_argument("--gWLf", type=float, default=None)
    p.add_argument("--mWp",  type=float, default=15.0)
    p.add_argument("--fit",  default=None,
                   help="Fit ID to use (default: <tag>_BSMclosure_SMEFT)")
    p.add_argument("--bins", type=int, default=50)
    args = p.parse_args()

    tag     = args.tag
    k       = args.k
    res_dir = PIPELINE / "results" / tag
    pca_dir = res_dir / "pca"
    fit_id  = args.fit or f"{tag}_BSMclosure_SMEFT"
    fit_dir = res_dir / "fits" / fit_id

    # ── Load PCA eigenvectors ──────────────────────────────────────────────────
    evecs = np.load(pca_dir / "eigenvectors.npy")   # (n_ops, n_modes), col = mode
    evals = np.load(pca_dir / "eigenvalues.npy")    # (n_modes,), descending

    # ── Load SMEFT posterior ───────────────────────────────────────────────────
    res    = json.load(open(fit_dir / "fit_results.json"))
    df     = pd.DataFrame(res["samples"])            # (n_replicas, n_ops)

    # Align operator order to PCA (eigenvectors were built with K's op order)
    # K was built with model.OPERATORS — detect from K shape
    K_shape = np.load(pca_dir / "K_fit.npy").shape
    n_ops   = K_shape[1]

    # Infer operator list: try constrained first, else full
    if n_ops == 16:
        from models.wprime_constrained import WPrimeConstrainedModel
        ops = WPrimeConstrainedModel.OPERATORS
    else:
        from models.wprime import WPrimeModel
        ops = WPrimeModel.OPERATORS

    # Subset posterior to these operators (in correct order)
    missing = [op for op in ops if op not in df.columns]
    if missing:
        print(f"  Warning: operators missing from posterior, filling with 0: {missing}")
    C = np.array([df[op].values if op in df.columns else np.zeros(len(df)) for op in ops]).T
    # C: (n_replicas, n_ops)

    # ── Truth point ───────────────────────────────────────────────────────────
    model   = get_model(args)
    c_truth = np.array([model.eft_coefficients().get(op, 0.0) for op in ops])

    # ── Project onto top-k PCA modes ──────────────────────────────────────────
    # alpha_i = e_i^T c  for each replica
    alphas_post  = C   @ evecs[:, :k]       # (n_replicas, k)
    alphas_truth = c_truth @ evecs[:, :k]   # (k,)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, k, figsize=(3.5 * k, 4), sharey=False)
    if k == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        a = alphas_post[:, i]
        truth = alphas_truth[i]

        ax.hist(a, bins=args.bins, color="#1f77b4", alpha=0.75, density=True,
                edgecolor="white", linewidth=0.3)
        ax.axvline(truth, color="red", lw=2.0, ls="--", label=f"truth = {truth:.3e}")
        ax.axvline(np.mean(a), color="orange", lw=1.5, ls=":", label=f"mean = {np.mean(a):.3e}")

        pull = (np.mean(a) - truth) / np.std(a) if np.std(a) > 0 else 0
        ax.set_title(
            fr"PCA mode $e_{i+1}$  ($\lambda={evals[i]:.2e}$)"
            f"\npull = {pull:+.2f}$\sigma$",
            fontsize=9
        )
        ax.set_xlabel(fr"$\alpha_{i+1} = e_{i+1}^\top c$", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)

    fig.suptitle(
        fr"PCA marginals (top {k} modes) — {tag}"
        f"\n{model}",
        fontsize=10, y=1.01
    )
    plt.tight_layout()

    out_dir = res_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        path = out_dir / f"pca_marginals_k{k}.{ext}"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
