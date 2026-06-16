"""
scripts/l1_pull_distribution_v2.py

L1 pull distribution using actual NS fit results — not the analytic approximation.

For each L1 replica r (a full NS fit on noisy BSM pseudo-data):
    P_i^(r) = (mean_r(c_i) - c_i^truth) / std_r(c_i)

where mean_r and std_r are taken from the NS posterior samples of that replica.
This uses real fit results, so deviations from N(0,1) reflect genuine issues:
non-Gaussian posteriors, prior effects, insufficient sampling, etc.

Compare with l1_pull_distribution.py (v1) which uses the analytic Gaussian
approximation — that is provably N(0,1) by construction and tests nothing real.

Usage:
    python scripts/l1_pull_distribution_v2.py \\
        --l1-dir results/wprime_l1_gwh012_mwp030 \\
        --model wprime_constrained \\
        --gWH 0.12 --mWp 3.0
"""
import sys, argparse, json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm as norm_dist
from pathlib import Path

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))


def get_model(args):
    if args.model == "wprime_constrained":
        from models.wprime_constrained import WPrimeConstrainedModel
        gWLf = args.gWH / 3
        return WPrimeConstrainedModel(gWH=args.gWH, gWLf11=gWLf, gWLf22=gWLf,
                                      gWLf33=gWLf, gWqf33=gWLf, mWp=args.mWp)
    if args.model == "wprime":
        from models.wprime import WPrimeModel
        gWLf = args.gWH / 3
        return WPrimeModel(gWH=args.gWH, gWLf11=gWLf, gWLf22=gWLf,
                           gWLf33=gWLf, gWqf33=gWLf, mWp=args.mWp)
    if args.model == "zprime_constrained":
        from models.zprime_constrained import ZPrimeConstrainedModel
        return ZPrimeConstrainedModel(gZH=args.gZH, gZl=args.gZl, mZp=args.mZp)
    raise ValueError(f"Unknown model: {args.model}")


def load_ns_replicas(l1_dir, ops):
    """Load NS posterior from all l1rep* fit folders.

    Returns
    -------
    pulls : (n_rep, n_ops)  per-replica per-operator pulls
    n_loaded : int          number of replicas successfully loaded
    """
    l1_dir = Path(l1_dir)
    fit_dir = l1_dir / "fits"
    rep_dirs = sorted(fit_dir.glob("*l1rep*"))

    pulls_list = []
    skipped = 0

    for rep in rep_dirs:
        res_path = rep / "fit_results.json"
        if not res_path.exists():
            skipped += 1
            continue
        res = json.load(open(res_path))
        df  = pd.DataFrame(res["samples"])

        # Pull per operator: (posterior_mean - truth) / posterior_std
        row = []
        for op in ops:
            if op not in df.columns:
                row.append(np.nan)
                continue
            s = df[op].values
            mu, sig = s.mean(), s.std(ddof=1)
            row.append(mu / sig if sig > 0 else np.nan)  # truth = 0 for SM-pull check
        pulls_list.append(row)

    if not pulls_list:
        raise RuntimeError(f"No completed NS fits found in {fit_dir}")

    print(f"  Loaded {len(pulls_list)} replicas  (skipped {skipped})")
    return np.array(pulls_list)   # (n_rep, n_ops)


def load_ns_replicas_with_truth(l1_dir, ops, c_truth):
    """Like load_ns_replicas but uses actual BSM truth for the pull numerator."""
    l1_dir = Path(l1_dir)
    fit_dir = l1_dir / "fits"
    rep_dirs = sorted(fit_dir.glob("*l1rep*"))

    pulls_list = []
    skipped = 0

    for rep in rep_dirs:
        res_path = rep / "fit_results.json"
        if not res_path.exists():
            skipped += 1
            continue
        res = json.load(open(res_path))
        df  = pd.DataFrame(res["samples"])

        row = []
        for i, op in enumerate(ops):
            truth = c_truth[i]
            if op not in df.columns:
                row.append(np.nan)
                continue
            s   = df[op].values
            mu  = s.mean()
            sig = s.std(ddof=1)
            row.append((mu - truth) / sig if sig > 0 else np.nan)
        pulls_list.append(row)

    if not pulls_list:
        raise RuntimeError(f"No completed NS fits found in {fit_dir}")

    print(f"  Loaded {len(pulls_list)} replicas  (skipped {skipped})")
    return np.array(pulls_list)   # (n_rep, n_ops)


def plot_pulls(pulls, ops, out_path, tag):
    pulls    = np.where(np.isfinite(pulls), pulls, np.nan)
    pull_vals = pulls[np.isfinite(pulls)].ravel()
    n_rep, n_ops = pulls.shape

    per_op_mean = np.nanmean(pulls, axis=0)
    per_op_std  = np.nanstd(pulls, axis=0, ddof=1)

    fig, axes = plt.subplots(1, 4, figsize=(22, max(5, n_ops * 0.32 + 2)))

    # ── panel 1: per-operator mean pull ──────────────────────────────────────
    ax = axes[0]
    err = per_op_std / np.sqrt(n_rep)
    colors = ["tomato" if abs(m) > 2 * e else "steelblue"
              for m, e in zip(per_op_mean, err)]
    ax.barh(range(n_ops), per_op_mean, xerr=err, color=colors,
            alpha=0.8, edgecolor="white", capsize=3)
    ax.axvline(0,    color="black", lw=1.0)
    ax.axvline( 0.1, color="gray",  lw=0.8, ls="--")
    ax.axvline(-0.1, color="gray",  lw=0.8, ls="--")
    ax.set_yticks(range(n_ops))
    ax.set_yticklabels(ops, fontsize=9)
    ax.set_xlabel(r"$\langle P_i \rangle$  (mean pull per operator)", fontsize=11)
    ax.set_title("Per-operator mean pull\n(NS posterior)", fontsize=11)
    ax.set_xlim(-1.0, 1.0)

    # ── panel 2: per-operator pull width ─────────────────────────────────────
    ax = axes[1]
    colors2 = ["tomato" if abs(s - 1) > 0.2 else "steelblue" for s in per_op_std]
    ax.barh(range(n_ops), per_op_std, color=colors2, alpha=0.8, edgecolor="white")
    ax.axvline(1.0, color="black", lw=1.2, label=r"$\sigma=1$ target")
    ax.axvline(0.8, color="gray",  lw=0.8, ls="--")
    ax.axvline(1.2, color="gray",  lw=0.8, ls="--")
    ax.set_yticks(range(n_ops))
    ax.set_yticklabels(ops, fontsize=9)
    ax.set_xlabel(r"$\mathrm{std}(P_i)$  (pull width per operator)", fontsize=11)
    ax.set_title("Per-operator pull width\n(NS posterior)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 2)

    # ── panel 3: combined histogram vs N(0,1) ────────────────────────────────
    ax = axes[2]
    bins = np.linspace(-4, 4, 22)
    ax.hist(pull_vals, bins=bins, density=True, alpha=0.7,
            color="steelblue", edgecolor="white",
            label=f"NS pulls  (n={len(pull_vals)})")
    x = np.linspace(-4, 4, 300)
    ax.plot(x, norm_dist.pdf(x), "r-", lw=2, label=r"$\mathcal{N}(0,1)$")
    mu, sig = pull_vals.mean(), pull_vals.std(ddof=1)
    ax.axvline(mu, color="steelblue", lw=2, ls="--", label=f"Mean = {mu:.3f}")
    ax.set_xlabel(r"Pull  $P_i^{(r)} = (\bar{c}_i^{(r)} - c_i^\mathrm{truth})\,/\,\sigma_i^{(r)}$",
                  fontsize=10)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(fr"Combined NS pulls  ($\mu={mu:.3f},\ \sigma={sig:.3f}$)", fontsize=11)
    ax.legend(fontsize=9)

    # ── panel 4: total chi2 per replica ──────────────────────────────────────
    ax = axes[3]
    Q_rep = np.nansum(pulls ** 2, axis=1)        # (n_rep,)
    Z_rep = (Q_rep - n_ops) / np.sqrt(2 * n_ops) # standardised ~ N(0,1)

    ax.hist(Z_rep, bins=18, density=True, alpha=0.75,
            color="mediumseagreen", edgecolor="white",
            label=fr"$Z_r$  (n={len(Z_rep)})")
    x2 = np.linspace(Z_rep.min() - 1, Z_rep.max() + 1, 300)
    ax.plot(x2, norm_dist.pdf(x2), "r-", lw=2, label=r"$\mathcal{N}(0,1)$ expected")
    ax.axvline(Z_rep.mean(), color="green", lw=2, ls="--",
               label=f"Mean = {Z_rep.mean():.3f}")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(
        r"Standardised total $\chi^2$:  $Z_r = (Q_r - n_\mathrm{ops})\,/\,\sqrt{2\,n_\mathrm{ops}}$",
        fontsize=10)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        fr"Total fit $\chi^2$ per replica  ($n_\mathrm{{ops}}={n_ops}$)"
        f"\n"
        fr"$\langle Q_r\rangle={Q_rep.mean():.1f}$,  "
        fr"$\langle Z_r\rangle={Z_rep.mean():.3f}$,  "
        fr"$\sigma(Z_r)={Z_rep.std(ddof=1):.3f}$",
        fontsize=10)
    ax.legend(fontsize=9)

    fig.suptitle(f"L1 BSM closure test (NS) — {tag}", fontsize=13, y=1.01)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = f"{out_path}.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--l1-dir", default=None,
                   help="Path to L1 results dir (default: auto from model/params)")
    p.add_argument("--model",  default="wprime_constrained",
                   choices=["wprime_constrained", "wprime", "zprime_constrained"])
    p.add_argument("--gWH",  type=float, default=0.12)
    p.add_argument("--mWp",  type=float, default=3.0)
    p.add_argument("--gZH",   type=float, default=0.12)
    p.add_argument("--gZl",   type=float, default=0.04)
    p.add_argument("--mZp",   type=float, default=1.0)
    p.add_argument("--use-sm", action="store_true",
                   help="Use c_SM=0 as reference instead of c_truth (shows discovery pull)")
    args = p.parse_args()

    model = get_model(args)
    ops   = model.OPERATORS
    truth = model.eft_coefficients()
    c_truth = np.array([truth.get(op, 0.0) for op in ops])

    if args.l1_dir:
        l1_dir  = Path(args.l1_dir)
        run_tag = l1_dir.name
    else:
        if "wprime" in args.model:
            run_tag = f"wprime_l1_gwh{int(args.gWH*100):03d}_mwp{int(args.mWp*10):03d}"
        else:
            run_tag = f"zprime_l1_gzh{int(args.gZH*100):03d}_mzp{int(args.mZp*10):03d}"
        l1_dir  = PIPELINE / "results" / run_tag

    out_dir = l1_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(model)

    print(f"Model  : {model}")
    print(f"L1 dir : {l1_dir}")
    print(f"Ops    : {ops}")

    c_ref = np.zeros(len(ops)) if args.use_sm else c_truth
    ref_label = "SM (c=0)" if args.use_sm else "BSM truth"
    suffix    = "_sm" if args.use_sm else ""
    print(f"Reference: {ref_label}")

    pulls = load_ns_replicas_with_truth(l1_dir, ops, c_ref)
    print(f"Pulls shape: {pulls.shape}  (n_rep × n_ops)")

    plot_pulls(pulls, ops, str(out_dir / f"l1_pull_distribution_v2{suffix}"), tag)


if __name__ == "__main__":
    main()
