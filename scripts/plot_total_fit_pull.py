"""
scripts/plot_total_fit_pull.py

Total-fit pull across all L1 NS replicas, computed in UV coupling space.

The NS fit has 5 free UV coupling parameters (gWH, gWLf11, gWLf22, gWLf33, gWqf33).
The EFT operator columns in fit_results.json are derived from those 5 parameters via
matching relations — they are NOT independent DoFs.

Correct total pull per replica:
    P_{k,r} = (mean_r(g_k) - g_k^truth) / std_r(g_k)   for each free UV parameter k
    Q_r     = sum_k P_{k,r}^2   ~  chi2(n_uv)  if well-calibrated

Three-panel output (suited for ~50 replicas):
    1. QQ-plot: observed Q_r quantiles vs chi2(n_uv) theoretical quantiles
    2. Ranked Z_r profile: each replica as a dot, sorted, with ±1σ / ±2σ bands
    3. ECDF of Q_r vs chi2(n_uv) CDF

Usage:
    python scripts/plot_total_fit_pull.py --gWH 0.12 --mWp 3.0
    python scripts/plot_total_fit_pull.py --l1-dir results/wprime_l1_gwh012_mwp030 --gWH 0.12 --mWp 3.0
"""
import sys, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2_dist, norm as norm_dist
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--l1-dir", default=None)
    p.add_argument("--model", default="wprime_constrained",
                   choices=["wprime_constrained", "wprime", "zprime_constrained"])
    p.add_argument("--gWH", type=float, default=0.12)
    p.add_argument("--mWp", type=float, default=3.0)
    p.add_argument("--gZH", type=float, default=0.12)
    p.add_argument("--gZl", type=float, default=0.04)
    p.add_argument("--mZp", type=float, default=7.0)
    args = p.parse_args()

    model     = get_model(args)
    uv_params = model.uv_param_names()
    uv_truth  = model.uv_truth()
    n_uv      = len(uv_params)

    if args.l1_dir:
        l1_dir = Path(args.l1_dir)
    else:
        if "wprime" in args.model:
            tag = f"wprime_l1_gwh{int(args.gWH*100):03d}_mwp{int(args.mWp*10):03d}"
        else:
            tag = f"zprime_l1_gzh{int(args.gZH*100):03d}_mzp{int(args.mZp*10):03d}"
        l1_dir = PIPELINE / "results" / tag

    fit_dir = l1_dir / "fits"
    out_dir = l1_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model      : {model}")
    print(f"UV params  : {uv_params}  (n_uv={n_uv})")
    print(f"UV truth   : {uv_truth}")
    print(f"L1 dir     : {l1_dir}")

    # ── compute Q_r in UV coupling space for each replica ────────────────────
    Q_vals      = []
    per_param   = {k: [] for k in uv_params}   # per-param pull, for diagnostics

    for rep in sorted(fit_dir.glob("*l1rep*")):
        res_path = rep / "fit_results.json"
        if not res_path.exists():
            continue
        r  = json.load(open(res_path))
        df = pd.DataFrame(r["samples"])

        # Use free_parameters from the fit to guard against wrong-model runs
        free = r.get("free_parameters", uv_params)
        active = [k for k in uv_params if k in df.columns and k in free]
        if not active:
            print(f"  WARN: {rep.name} has no matching UV columns — skipping")
            continue

        Q_r = 0.0
        for k in active:
            s   = df[k].values
            mu  = s.mean()
            sig = s.std(ddof=1)
            if sig > 0:
                pull = (mu - uv_truth[k]) / sig
                Q_r += pull ** 2
                per_param[k].append(pull)
        Q_vals.append(Q_r)

    Q     = np.array(Q_vals)
    n_rep = len(Q)
    print(f"\nReplicas loaded : {n_rep}")
    print(f"Q_r  mean={Q.mean():.3f}  std={Q.std(ddof=1):.3f}")
    print(f"Expected under chi2({n_uv}): mean={n_uv}, std={np.sqrt(2*n_uv):.3f}")
    print()
    for k in uv_params:
        pulls_k = np.array(per_param[k])
        if len(pulls_k):
            print(f"  {k:12s}: mean={pulls_k.mean():+.3f}  std={pulls_k.std(ddof=1):.3f}")

    Z = (Q - n_uv) / np.sqrt(2 * n_uv)   # standardised: ~ N(0,1) if well-calibrated

    # ── plot: 3 panels ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle(
        fr"Total UV-coupling fit pull — {model}   ({n_rep} L1 NS replicas, $n_\mathrm{{UV}}={n_uv}$)",
        fontsize=12, y=1.02)

    # ── panel 1: QQ-plot of Q_r vs chi2(n_uv) ────────────────────────────────
    ax = axes[0]
    probs    = (np.arange(1, n_rep + 1) - 0.5) / n_rep   # Hazen plotting positions
    q_theory = chi2_dist.ppf(probs, df=n_uv)
    q_obs    = np.sort(Q)
    ax.scatter(q_theory, q_obs, s=28, color="steelblue", zorder=3, label="Replicas")
    lim = max(q_theory[-1], q_obs[-1]) * 1.05
    ax.plot([0, lim], [0, lim], "r-", lw=1.5, label="y = x  (perfect)")
    # 95% pointwise confidence band from order-statistic variance
    se = np.sqrt(probs * (1 - probs) / n_rep) * chi2_dist.pdf(q_theory, df=n_uv) ** -1
    ax.fill_between(q_theory, q_obs - 1.96 * se, q_obs + 1.96 * se,
                    alpha=0.15, color="steelblue", label="95% band")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(fr"$\chi^2({n_uv})$ theoretical quantiles", fontsize=11)
    ax.set_ylabel(r"Observed $Q_r$ quantiles", fontsize=11)
    ax.set_title(fr"QQ-plot: $Q_r$ vs $\chi^2({n_uv})$", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    # ── panel 2: ranked Z_r profile ───────────────────────────────────────────
    ax = axes[1]
    z_sorted = np.sort(Z)
    ranks    = np.arange(1, n_rep + 1)
    ax.axhspan(-1, 1, alpha=0.12, color="green",  label=r"$\pm 1\sigma$")
    ax.axhspan(-2, 2, alpha=0.07, color="orange", label=r"$\pm 2\sigma$")
    ax.axhline(0, color="black", lw=0.8)
    ax.scatter(ranks, z_sorted, s=30, color="steelblue", zorder=3,
               label=fr"$Z_r$ (sorted), $\langle Z_r\rangle={Z.mean():.3f}$")
    ax.set_xlabel("Replica rank (sorted by $Z_r$)", fontsize=11)
    ax.set_ylabel(r"$Z_r = (Q_r - n_\mathrm{UV})\,/\,\sqrt{2\,n_\mathrm{UV}}$", fontsize=11)
    ax.set_title(fr"Ranked $Z_r$ profile  ($\sigma={Z.std(ddof=1):.3f}$, expected 1.0)",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25)

    # ── panel 3: ECDF of Q_r vs chi2(n_uv) CDF ───────────────────────────────
    ax = axes[2]
    q_sorted = np.sort(Q)
    ecdf     = np.arange(1, n_rep + 1) / n_rep
    x_th     = np.linspace(0, q_sorted[-1] * 1.1, 400)
    ax.step(q_sorted, ecdf, where="post", color="steelblue", lw=2,
            label=f"Empirical CDF  (n={n_rep})")
    ax.plot(x_th, chi2_dist.cdf(x_th, df=n_uv), "r-", lw=2,
            label=fr"$\chi^2({n_uv})$ CDF")
    # KS band (Kolmogorov–Smirnov 95%)
    eps = 1.36 / np.sqrt(n_rep)
    ax.fill_between(x_th,
                    np.clip(chi2_dist.cdf(x_th, df=n_uv) - eps, 0, 1),
                    np.clip(chi2_dist.cdf(x_th, df=n_uv) + eps, 0, 1),
                    alpha=0.15, color="red", label="KS 95% band")
    ax.set_xlabel(
        r"$Q_r = \sum_k \left(\frac{\bar{g}_k^{(r)} - g_k^\mathrm{truth}}{\sigma_k^{(r)}}\right)^2$",
        fontsize=11)
    ax.set_ylabel("Cumulative probability", fontsize=11)
    ax.set_title(r"ECDF of $Q_r$", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = out_dir / "total_fit_pull.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
