"""
scripts/metric1_delta_chi2.py

Plots the delta_chi2 = chi2_SM - chi2_UV_min distribution across L1 replicas.

chi2_SM^(i)     = delta_i^T C^{-1} delta_i   (residual at SM best-fit, c=0)
chi2_UV_min^(i) = -2 * max_loglikelihood      (from NS fit_results.json)
delta_chi2^(i)  = chi2_SM^(i) - chi2_UV_min^(i)

Under the BSM hypothesis, delta_chi2 ~ non-central chi2(k=n_UV, lambda)
where lambda = c_truth^T F c_truth is the analytic non-centrality.

Usage:
    python scripts/metric1_delta_chi2.py --tag wprime_l1_gwh020_mwp050 --gWH 0.20 --mWp 5.0
    python scripts/metric1_delta_chi2.py --tag wprime_l1_gwh012_mwp030 --gWH 0.12 --mWp 3.0
    python scripts/metric1_delta_chi2.py --tag wprime_l1_gwh030_mwp080 --gWH 0.30 --mWp 8.0
"""

import sys, argparse, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ncx2, chi2 as chi2_dist, norm as norm_dist

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE / "scripts"))
sys.path.insert(0, str(PIPELINE))

from run_pipeline import _build_K_Ci, _build_delta, DS_NAMES
from models.wprime_constrained import WPrimeConstrainedModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--gWH", type=float, required=True)
    p.add_argument("--mWp", type=float, required=True)
    args = p.parse_args()

    res_dir  = PIPELINE / "results" / args.tag
    fits_dir = res_dir / "fits"
    proj_dir = res_dir / "projections"

    # ── truth model ───────────────────────────────────────────────────────────
    g  = args.gWH
    gf = g / 3.0
    model  = WPrimeConstrainedModel(
        gWH=g, gWLf11=gf, gWLf22=gf, gWLf33=gf, gWqf33=gf, mWp=args.mWp
    )
    ops    = model.OPERATORS
    c_truth = np.array([model.eft_coefficients().get(op, 0.0) for op in ops])

    K, Ci = _build_K_Ci(ops)
    F     = K.T @ Ci @ K
    lam   = float(c_truth @ F @ c_truth)   # analytic non-centrality
    n_uv  = 5                               # free UV parameters in NS fit

    # ── per-replica delta chi2 ────────────────────────────────────────────────
    rep_dirs  = sorted(fits_dir.iterdir())
    delta_chi2_vals = []

    for rd in rep_dirs:
        fj = rd / "fit_results.json"
        if not fj.exists():
            continue
        rep_id   = rd.name.split("l1rep")[-1]          # e.g. "000"
        proj_rep = proj_dir / f"rep{rep_id}"
        if not proj_rep.exists():
            continue

        ll       = json.load(open(fj)).get("max_loglikelihood")
        if ll is None:
            continue
        chi2_uv  = -2.0 * ll

        delta_i  = _build_delta(str(proj_rep))
        chi2_sm  = float(delta_i @ Ci @ delta_i)

        delta_chi2_vals.append(chi2_sm - chi2_uv)

    delta_chi2_vals = np.array(delta_chi2_vals)
    n = len(delta_chi2_vals)
    print(f"Loaded {n} replicas")
    print(f"  analytic lambda = {lam:.2f}   (sqrt(lambda) = {np.sqrt(lam):.2f})")
    print(f"  delta_chi2: mean={delta_chi2_vals.mean():.2f}  "
          f"std={delta_chi2_vals.std():.2f}  "
          f"expected mean={lam + n_uv:.2f}")

    # discovery threshold: delta_chi2 > chi2_threshold for 5-sigma
    # One-sided: p = norm.sf(5) -> chi2.isf(2*norm.sf(5), df=n_uv)
    p5 = 2 * norm_dist.sf(5)
    thr_5s = chi2_dist.isf(p5, df=n_uv)
    frac_5s = float(np.mean(delta_chi2_vals > thr_5s))
    print(f"  5-sigma threshold (ndof={n_uv}): {thr_5s:.1f}")
    print(f"  Fraction of replicas above 5-sigma: {frac_5s*100:.1f}%")

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    x_max = max(delta_chi2_vals.max() * 1.1, lam + n_uv + 3 * np.sqrt(2*(n_uv + 2*lam)))
    x_max = min(x_max, lam * 3 + 50)
    xg = np.linspace(0, x_max, 500)

    ax.hist(delta_chi2_vals, bins=15, density=True,
            color="#1f77b4", alpha=0.7, label=f"NS replicas (n={n})")

    # non-central chi2(n_uv, lambda) — expected under BSM
    ax.plot(xg, ncx2.pdf(xg, df=n_uv, nc=lam), "r-", lw=2,
            label=fr"$\chi^2_{{nc}}(k={n_uv},\,\lambda={lam:.1f})$  [expected]")

    # central chi2(n_uv) — Wilks under H0
    ax.plot(xg, chi2_dist.pdf(xg, df=n_uv), "k--", lw=1.5, alpha=0.6,
            label=fr"$\chi^2(k={n_uv})$  [SM null]")

    # 5-sigma threshold
    ax.axvline(thr_5s, color="darkorange", lw=1.8, ls=":",
               label=fr"$5\sigma$ threshold $= {thr_5s:.1f}$")

    # median of replicas
    ax.axvline(np.median(delta_chi2_vals), color="#1f77b4", lw=1.4, ls="--",
               label=f"median $= {np.median(delta_chi2_vals):.1f}$")

    ax.set_xlabel(r"$\Delta\chi^2 = \chi^2_\mathrm{SM} - \chi^2_\mathrm{UV,min}$",
                  fontsize=12)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        fr"Metric 1 — $\Delta\chi^2$ distribution  (W', $g_{{WH}}={g}$, "
        fr"$g_{{Wf}}={gf:.3f}$, $m_{{W'}}={args.mWp}$ TeV)"
        "\n"
        fr"$\lambda={lam:.1f}$,  $\sqrt{{\lambda}}={np.sqrt(lam):.1f}$,  "
        fr"{frac_5s*100:.0f}% of replicas exceed $5\sigma$",
        fontsize=10
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, x_max)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        out = res_dir / f"metric1_delta_chi2.{ext}"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
