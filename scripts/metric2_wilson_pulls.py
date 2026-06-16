"""
scripts/metric2_wilson_pulls.py

Metric 2 in Wilson coefficient space (supervisor's definition):
    P_i = (c_i^fit - c_i^truth) / delta_c_i^fit

Uses the existing NS fit samples from the L1 closure run.

Usage:
    python scripts/metric2_wilson_pulls.py --tag wprime_l1_gwh020_mwp050 --gWH 0.20 --mWp 5.0
    python scripts/metric2_wilson_pulls.py --tag wprime_l1_gwh012_mwp030 --gWH 0.12 --mWp 3.0
    python scripts/metric2_wilson_pulls.py --tag wprime_l1_gwh030_mwp080 --gWH 0.30 --mWp 8.0
"""

import sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.stats import norm, ks_1samp, uniform

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE / "scripts"))
sys.path.insert(0, str(PIPELINE))

from models.wprime_constrained import WPrimeConstrainedModel

UV_KEYS = {"gWH", "gWLf11", "gWLf22", "gWLf33", "gWqf33", "mWp_TeV"}

OP_LABELS = {
    "O3pQ3":   r"$\mathcal{O}_{3pQ3}$",
    "O3pl1":   r"$\mathcal{O}_{3pl1}$",
    "O3pl2":   r"$\mathcal{O}_{3pl2}$",
    "O3pl3":   r"$\mathcal{O}_{3pl3}$",
    "OQQ1":    r"$\mathcal{O}_{QQ}^{1}$",
    "OQQ8":    r"$\mathcal{O}_{QQ}^{8}$",
    "OQl13":   r"$\mathcal{O}_{Ql}^{13}$",
    "OQl1M":   r"$\mathcal{O}_{Ql}^{1(-)}$",
    "OQl23":   r"$\mathcal{O}_{Ql}^{23}$",
    "OQl2M":   r"$\mathcal{O}_{Ql}^{2(-)}$",
    "OQl33":   r"$\mathcal{O}_{Ql}^{33}$",
    "OQl3M":   r"$\mathcal{O}_{Ql}^{3(-)}$",
    "Obp":     r"$\mathcal{O}_{bW}$",
    "Oll1111": r"$\mathcal{O}_{ll}^{1111}$",
    "Oll1122": r"$\mathcal{O}_{ll}^{1122}$",
    "Oll1133": r"$\mathcal{O}_{ll}^{1133}$",
    "Oll1221": r"$\mathcal{O}_{ll}^{1221}$",
    "Oll1331": r"$\mathcal{O}_{ll}^{1331}$",
    "Oll2222": r"$\mathcal{O}_{ll}^{2222}$",
    "Oll2233": r"$\mathcal{O}_{ll}^{2233}$",
    "Oll2332": r"$\mathcal{O}_{ll}^{2332}$",
    "Oll3333": r"$\mathcal{O}_{ll}^{3333}$",
    "Op":      r"$\mathcal{O}_{\varphi}$",
    "OpBox":   r"$\mathcal{O}_{\varphi,\mathrm{box}}$",
    "OpQM":    r"$\mathcal{O}_{\varphi Q}^{(-)}$",
    "Otap":    r"$\mathcal{O}_{t\varphi}$",
    "Otp":     r"$\mathcal{O}_{tW}$",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag",  required=True, help="e.g. wprime_l1_gwh020_mwp050")
    p.add_argument("--gWH",  type=float, required=True)
    p.add_argument("--mWp",  type=float, required=True)
    args = p.parse_args()

    res_dir = PIPELINE / "results" / args.tag
    fits_dir = res_dir / "fits"

    # truth Wilson coefficients — use all 27 operators from the fit samples
    g = args.gWH
    gf = g / 3.0
    model = WPrimeConstrainedModel(
        gWH=g, gWLf11=gf, gWLf22=gf, gWLf33=gf, gWqf33=gf, mWp=args.mWp
    )
    c_truth_dict = model.eft_coefficients()

    # read operator list from first available fit
    first_fit = next(
        (d / "fit_results.json")
        for d in sorted(fits_dir.iterdir())
        if (d / "fit_results.json").exists()
    )
    all_keys = list(json.load(open(first_fit))["samples"].keys())
    ops = [k for k in all_keys if k not in UV_KEYS]
    c_truth = np.array([c_truth_dict.get(op, 0.0) for op in ops])

    # collect pulls from each replica
    rep_dirs = sorted(fits_dir.glob(f"{args.tag.replace('_l1_', '_')}_l1rep*"))
    if not rep_dirs:
        # fallback: any subdir with fit_results.json
        rep_dirs = sorted([d for d in fits_dir.iterdir() if (d / "fit_results.json").exists()])

    pulls = {op: [] for op in ops}
    n_loaded = 0

    for rd in rep_dirs:
        fj = rd / "fit_results.json"
        if not fj.exists():
            continue
        data = json.load(open(fj))
        samples = data["samples"]
        for i, op in enumerate(ops):
            if op not in samples:
                continue
            s = np.array(samples[op])
            mu  = np.mean(s)
            sig = np.std(s, ddof=1)
            if sig > 0:
                pulls[op].append((mu - c_truth[i]) / sig)
        n_loaded += 1

    print(f"Loaded {n_loaded} replicas from {fits_dir}")

    # summary stats
    print(f"\n{'Operator':<12}  {'n_pulls':>7}  {'mean':>7}  {'std':>7}  {'pass?':>6}")
    print("-" * 50)
    for op in ops:
        arr = np.array(pulls[op])
        if len(arr) == 0:
            continue
        m, s = np.mean(arr), np.std(arr, ddof=1)
        ok = abs(m) < 0.5 and abs(s - 1.0) < 0.3
        print(f"{op:<12}  {len(arr):>7}  {m:>7.3f}  {s:>7.3f}  {'✓' if ok else '✗':>6}")

    # ── bar summary plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        f"Metric 2 — Wilson coefficient pull summary (NS, n={n_loaded} replicas)\n"
        f"W' truth: $g_{{WH}}={g}$, $g_{{Wf}}={gf:.3f}$, $m_{{W'}}={args.mWp}$ TeV",
        fontsize=11
    )

    x = np.arange(len(ops))
    labels = [OP_LABELS.get(op, op) for op in ops]

    means = [np.mean(pulls[op]) if pulls[op] else 0.0 for op in ops]
    stds  = [np.std(pulls[op], ddof=1) if len(pulls[op]) > 1 else 0.0 for op in ops]
    colors_m = ["#d62728" if abs(m) >= 0.5 else "#2ca02c" for m in means]
    colors_s = ["#d62728" if abs(s - 1.0) >= 0.3 else "#2ca02c" for s in stds]

    ax0 = axes[0]
    ax0.bar(x, means, color=colors_m, zorder=3)
    ax0.axhline(0,    color="red",    lw=1.5, ls="--", label="target: 0")
    ax0.axhline(+0.5, color="orange", lw=0.8, ls=":")
    ax0.axhline(-0.5, color="orange", lw=0.8, ls=":")
    ax0.set_ylabel(r"$\langle P \rangle$ (bias)", fontsize=10)
    ax0.set_title(f"Wilson coefficient pull mean  (NS, n={n_loaded} replicas)", fontsize=10)
    ax0.legend(fontsize=9, loc="upper right")
    ax0.grid(axis="y", alpha=0.3)

    ax1 = axes[1]
    ax1.bar(x, stds, color=colors_s, zorder=3)
    ax1.axhline(1.0, color="red",    lw=1.5, ls="--", label="target: 1")
    ax1.axhline(1.3, color="orange", lw=0.8, ls=":")
    ax1.axhline(0.7, color="orange", lw=0.8, ls=":")
    ax1.set_ylabel(r"$\sigma(P)$ (calibration)", fontsize=10)
    ax1.set_title("Wilson coefficient pull width  (NS)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = res_dir / "metric2_wilson_pull_bars.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")

    # ── per-operator pull histograms ──────────────────────────────────────────
    ncols = 5
    nrows = (len(ops) + ncols - 1) // ncols
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows))
    fig2.suptitle(
        f"Metric 2 — Wilson coefficient pull histograms (NS, n={n_loaded})\n"
        f"W' truth: $g_{{WH}}={g}$, $g_{{Wf}}={gf:.3f}$, $m_{{W'}}={args.mWp}$ TeV",
        fontsize=11
    )
    xg = np.linspace(-4, 4, 200)

    for i, op in enumerate(ops):
        ax = axes2.flat[i]
        arr = np.array(pulls[op])
        if len(arr) > 1:
            ax.hist(arr, bins=12, density=True, color="#5599dd", alpha=0.7)
            ax.plot(xg, norm.pdf(xg), "r-", lw=1.5, label=r"$\mathcal{N}(0,1)$")
            ax.axvline(np.mean(arr), color="k", lw=1.2, ls="--")
        ax.set_title(OP_LABELS.get(op, op), fontsize=9)
        ax.set_xlim(-4, 4)
        ax.set_xlabel("Pull", fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(len(ops), len(axes2.flat)):
        axes2.flat[j].axis("off")

    plt.tight_layout()
    out2 = res_dir / "metric2_wilson_pull_histograms.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
