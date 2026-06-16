"""
Pull distribution for the BSM closure test.

For each Wilson coefficient c_i, computes:
    P_i = (c_i^fit - c_i^truth) / sigma_i^fit

where c_i^fit = posterior mean, sigma_i^fit = posterior std (from NS samples),
and c_i^truth = EFT coefficient predicted by the UV model at the injected point.

If the closure test is successful and uncertainties are correctly estimated,
the pull distribution should be a Gaussian with mean=0 and width=1.

Usage:
    python scripts/pull_distribution.py
    python scripts/pull_distribution.py --model wprime_constrained --gWH 0.12 --mWp 3.0
    python scripts/pull_distribution.py --latex
"""
import sys, os, argparse, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as norm_dist
from pathlib import Path

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))

from models.wprime_constrained import WPrimeConstrainedModel
from models.zprime import ZPrimeModel


def load_fit_results(fit_dir, fit_id):
    path = Path(fit_dir) / fit_id / "fit_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Fit results not found: {path}")
    return json.load(open(path))


def compute_pulls(fit_results, truth_coeffs):
    """
    Returns dict: op -> (pull, c_fit, sigma_fit, c_truth)
    Only operators present in both fit and truth are included.
    """
    samples = fit_results["samples"]
    pulls = {}
    for op, s in samples.items():
        s = np.array(s)
        c_fit    = float(np.mean(s))
        sigma_fit = float(np.std(s, ddof=1))
        if sigma_fit == 0:
            continue
        c_truth = truth_coeffs.get(op, 0.0)
        pull = (c_fit - c_truth) / sigma_fit
        pulls[op] = {
            "pull":    pull,
            "c_fit":   c_fit,
            "sigma":   sigma_fit,
            "c_truth": c_truth,
        }
    return pulls


def print_table(pulls, latex=False):
    if latex:
        print(r"\begin{table}[h]")
        print(r"\centering")
        print(r"\renewcommand{\arraystretch}{1.2}")
        print(r"\begin{tabular}{lcccc}")
        print(r"\hline\hline")
        print(r"Operator & $c_i^\mathrm{truth}$ & $c_i^\mathrm{fit}$ & $\sigma_i^\mathrm{fit}$ & $P_i$ \\")
        print(r"\hline")
        for op, v in pulls.items():
            print(f"{op} & {v['c_truth']:.4e} & {v['c_fit']:.4e} & "
                  f"{v['sigma']:.4e} & {v['pull']:.3f} \\\\")
        print(r"\hline\end{tabular}")
        print(r"\caption{Pull distribution for BSM closure test. "
              r"$P_i = (c_i^\mathrm{fit} - c_i^\mathrm{truth})/\sigma_i^\mathrm{fit}$ "
              r"should follow $\mathcal{N}(0,1)$ if the closure test is successful.}")
        print(r"\label{tab:pulls}")
        print(r"\end{table}")
    else:
        print(f"{'Operator':<14} {'c_truth':>12} {'c_fit':>12} {'sigma_fit':>12} {'Pull':>8}")
        print("-" * 62)
        for op, v in pulls.items():
            print(f"{op:<14} {v['c_truth']:>12.4e} {v['c_fit']:>12.4e} "
                  f"{v['sigma']:>12.4e} {v['pull']:>8.3f}")
        pull_vals = np.array([v["pull"] for v in pulls.values()])
        print("-" * 62)
        print(f"{'Mean':>52} {pull_vals.mean():>8.3f}")
        print(f"{'Std':>52} {pull_vals.std(ddof=1):>8.3f}")


def plot_pulls(pulls, out_path, tag):
    pull_vals = np.array([v["pull"] for v in pulls.values()])
    ops       = list(pulls.keys())
    n         = len(pull_vals)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── left: pull per operator ───────────────────────────────────────────────
    ax = axes[0]
    colors = ["tomato" if abs(p) > 2 else "steelblue" for p in pull_vals]
    ax.barh(range(n), pull_vals, color=colors, alpha=0.8, edgecolor="white")
    ax.axvline(0,  color="black", lw=1.0)
    ax.axvline( 1, color="gray",  lw=1.0, ls="--", label=r"$\pm 1\sigma$")
    ax.axvline(-1, color="gray",  lw=1.0, ls="--")
    ax.axvline( 2, color="gray",  lw=0.7, ls=":",  label=r"$\pm 2\sigma$")
    ax.axvline(-2, color="gray",  lw=0.7, ls=":")
    ax.set_yticks(range(n))
    ax.set_yticklabels(ops, fontsize=11)
    ax.set_xlabel(r"Pull $P_i = (c_i^\mathrm{fit} - c_i^\mathrm{truth})\,/\,\sigma_i^\mathrm{fit}$",
                  fontsize=12)
    ax.set_title("Pull per operator", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(-4, 4)

    # ── right: pull histogram vs N(0,1) ───────────────────────────────────────
    ax = axes[1]
    bins = np.linspace(-4, 4, 16)
    ax.hist(pull_vals, bins=bins, density=True, alpha=0.7,
            color="steelblue", edgecolor="white", label=f"Pulls (n={n})")
    x = np.linspace(-4, 4, 200)
    ax.plot(x, norm_dist.pdf(x), "r-", lw=2, label=r"$\mathcal{N}(0,1)$")
    ax.axvline(pull_vals.mean(), color="steelblue", lw=2, ls="--",
               label=fr"Mean = {pull_vals.mean():.2f}")
    ax.set_xlabel(r"Pull $P_i$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(fr"Pull distribution  ($\mu={pull_vals.mean():.2f},\ \sigma={pull_vals.std(ddof=1):.2f}$)",
                 fontsize=13)
    ax.legend(fontsize=10)

    fig.suptitle(f"BSM closure test — {tag}", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{out_path}.png", bbox_inches="tight", dpi=150)
    plt.savefig(f"{out_path}.pdf", bbox_inches="tight")
    print(f"\nPlot saved: {out_path}.png")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="wprime_constrained",
                   choices=["wprime_constrained", "zprime"])
    p.add_argument("--gWH",   type=float, default=0.12)
    p.add_argument("--mWp",   type=float, default=1.0)
    p.add_argument("--gZH",   type=float, default=0.12)
    p.add_argument("--mZp",   type=float, default=1.0)
    p.add_argument("--latex", action="store_true")
    args = p.parse_args()

    if args.model == "wprime_constrained":
        gWLf  = args.gWH / 3
        model = WPrimeConstrainedModel(gWH=args.gWH, gWLf11=gWLf, gWLf22=gWLf,
                                       gWLf33=gWLf, gWqf33=gWLf, mWp=args.mWp)
        tag   = f"W' (gWH={args.gWH}, mWp={args.mWp} TeV)"
        run_tag = f"wprime_constrained_gwh{int(args.gWH*100):03d}_mwp{int(args.mWp*10):03d}"
    else:
        model = ZPrimeModel(gZH=args.gZH, gZl=args.gZH/3, mZp=args.mZp)
        tag   = f"Z' (gZH={args.gZH}, mZp={args.mZp} TeV)"
        run_tag = f"zprime_gzh{int(args.gZH*100):03d}_mzp{int(args.mZp*10):03d}"

    results_dir = PIPELINE / "results" / run_tag
    fit_dir     = results_dir / "fits"
    fit_id      = f"{run_tag}_BSMclosure_SMEFT"
    out_dir     = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model  : {model}")
    print(f"Fit    : {fit_id}")

    fit_results  = load_fit_results(fit_dir, fit_id)
    truth_coeffs = model.eft_coefficients()

    print(f"Ops    : {list(fit_results['samples'].keys())}")
    print(f"Samples: {len(next(iter(fit_results['samples'].values())))}")
    print(f"Truth  : {len(truth_coeffs)} non-zero coefficients")
    print()

    pulls = compute_pulls(fit_results, truth_coeffs)

    print_table(pulls, latex=args.latex)

    pull_vals = np.array([v["pull"] for v in pulls.values()])
    print(f"\nPull summary:")
    print(f"  Mean : {pull_vals.mean():.3f}  (should be ~0)")
    print(f"  Std  : {pull_vals.std(ddof=1):.3f}  (should be ~1)")
    print(f"  |P|>1: {sum(abs(p)>1 for p in pull_vals)}/{len(pull_vals)}"
          f"  (expect ~32%)")
    print(f"  |P|>2: {sum(abs(p)>2 for p in pull_vals)}/{len(pull_vals)}"
          f"  (expect ~5%)")

    plot_pulls(pulls, str(out_dir / "pull_distribution"), tag)


if __name__ == "__main__":
    main()
