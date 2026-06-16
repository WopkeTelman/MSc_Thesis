"""
Replot the UV Fisher heatmap with clean formatting.

- No title
- Legend (Linear / Quadratic) below the heatmap
- Energy groups in specified order (left to right)
- Clean group labels (no brackets)
- Correct 49-dataset grouping

Usage:
    python scripts/replot_fisher_heatmap.py \
        --tag  zprime_constrained_gzh012_mzp075 \
        --model zprime_constrained \
        --gZH  0.12 --gZl 0.04 --mZp 7.5 \
        --out  results/zprime_constrained_gzh012_mzp075/reports/\
Report_zprime_constrained_gzh012_mzp075_UVcoup/meta/\
fisher_heatmap_zprime_constrained_gzh012_mzp075_UVcoup.png
"""
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import yaml
from pathlib import Path

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE / "scripts"))
sys.path.insert(0, str(PIPELINE))
from run_pipeline import DATA_INFO, DATASETS, _build_K_Ci, SM_DATA, THEORY


def get_model(args):
    if args.model == "zprime_constrained":
        from models.zprime_constrained import ZPrimeConstrainedModel
        return ZPrimeConstrainedModel(gZH=args.gZH, gZl=args.gZl, mZp=args.mZp)
    if args.model == "zprime":
        from models.zprime import ZPrimeModel
        return ZPrimeModel(gZH=args.gZH, gZl=args.gZl, mZp=args.mZp)
    if args.model == "wprime_constrained":
        from models.wprime_constrained import WPrimeConstrainedModel
        gWLf = args.gWLf if args.gWLf else args.gWH / 3
        return WPrimeConstrainedModel(gWH=args.gWH, gWLf11=gWLf, gWLf22=gWLf,
                                      gWLf33=gWLf, gWqf33=gWLf, mWp=args.mWp)
    if args.model == "wprime_universal":
        from models.wprime_universal import WPrimeUniversalModel
        return WPrimeUniversalModel(gWH=args.gWH, mWp=args.mWp)
    raise ValueError(f"Unknown model: {args.model}")


def compute_fisher(tag, model):
    ops = model.OPERATORS

    # Rebuild K and Ci together from the shared SM data — same source _build_K_Ci uses,
    # so the block structure is guaranteed consistent.  The cached K_fit.npy / C_inv.npy
    # can be stale (different operator set or dataset sizes), so we don't use them.
    K, Ci = _build_K_Ci(ops)

    # Numerical Jacobian at truth point
    eps       = 1e-6
    truth     = model.uv_truth()
    uv_params = model.uv_param_names()
    J         = np.zeros((len(ops), len(uv_params)))
    cls       = type(model)
    for j, par in enumerate(uv_params):
        kp = {k: float(v) for k, v in truth.items()}
        km = {**kp}
        kp[par] += eps
        km[par] -= eps
        cp = np.array([cls(**kp).eft_coefficients().get(op, 0.) for op in ops])
        cm = np.array([cls(**km).eft_coefficients().get(op, 0.) for op in ops])
        J[:, j] = (cp - cm) / (2 * eps)

    KJ = K @ J  # (n_obs, n_uv)

    # Dataset row ranges — use the same SM data files that _build_K_Ci reads,
    # so the row boundaries match K and Ci exactly.
    import os, json as _json
    row = 0
    ds_rows = {}
    for d in DATASETS:
        name = d["name"]
        sm_path = f"{SM_DATA}/{name}.yaml"
        th_path = f"{THEORY}/{name}.json"
        if not os.path.exists(sm_path) or not os.path.exists(th_path):
            continue
        sm = yaml.safe_load(open(sm_path))
        dc = sm["data_central"]
        n  = len(dc) if isinstance(dc, list) else 1
        ds_rows[name] = (row, row + n)
        row += n

    # Energy groups — use full DATA_INFO
    groups = {
        "91 GeV":  DATA_INFO["91 GeV (Z-pole)"],
        "161 GeV": DATA_INFO["161 GeV (WW thr.)"],
        "240 GeV": DATA_INFO["240 GeV"],
        "365 GeV": DATA_INFO["365 GeV"],
    }

    group_order = ["91 GeV", "161 GeV", "240 GeV", "365 GeV"]
    n_uv = len(uv_params)
    n_g  = len(group_order)
    F    = np.zeros((n_uv, n_g))

    for ig, grp in enumerate(group_order):
        for ds, _ in groups[grp]:
            if ds not in ds_rows:
                continue
            r0, r1 = ds_rows[ds]
            KJ_d = KJ[r0:r1, :]
            Ci_d = Ci[r0:r1, r0:r1]
            F[:, ig] += np.diag(KJ_d.T @ Ci_d @ KJ_d)

    # Normalise each UV param row to 100 %
    row_sums = F.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.
    F_norm = 100. * F / row_sums

    return F_norm, uv_params, group_order


UV_LATEX = {
    "gWH":    r"$g_{WH}$",
    "gWLf11": r"$g_{WLf,11}$",
    "gWLf22": r"$g_{WLf,22}$",
    "gWLf33": r"$g_{WLf,33}$",
    "gWqf33": r"$g_{Wqf,33}$",
    "gZH":    r"$g_{ZH}$",
    "gZl":    r"$g_{Zl}$",
}


def make_plot(F_norm, uv_params, group_order, out_path):
    n_uv = len(uv_params)
    n_g  = len(group_order)

    cmap_base = plt.get_cmap("Blues")
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "blues_trunc", cmap_base(np.linspace(0, 0.85, 100)))
    bounds = np.arange(0, 110, 10)
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(max(5, n_g * 1.4), max(2.5, n_uv * 1.2 + 0.8)))

    im = ax.imshow(F_norm, aspect="auto", cmap=cmap, norm=norm,
                   origin="upper")

    # Cell text
    for i in range(n_uv):
        for j in range(n_g):
            val = F_norm[i, j]
            color = "white" if val > 60 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")

    # Axes
    ax.set_xticks(np.arange(n_g))
    ax.set_xticklabels(group_order, fontsize=12)
    ax.set_yticks(np.arange(n_uv))
    ax.set_yticklabels([UV_LATEX.get(p, p) for p in uv_params], fontsize=13)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(which="major", top=False, bottom=False, left=False)

    # Minor grid
    ax.set_xticks(np.arange(n_g) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_uv) - 0.5, minor=True)
    ax.tick_params(which="minor", bottom=False, top=False, left=False)
    ax.grid(visible=True, which="minor", color="white", linewidth=1.5)

    # Colorbar below
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                        pad=0.08, fraction=0.05, aspect=30)
    cbar.set_label("Normalised Fisher information [%]", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # "Linear" legend text below (no quadratic for analytic method)
    ax.annotate("Linear EFT, evaluated at UV truth point",
                xy=(0.5, -0.22), xycoords="axes fraction",
                ha="center", va="top", fontsize=9, color="gray",
                style="italic")

    ax.set_title("")   # no title
    plt.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out,              dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag",   required=True)
    ap.add_argument("--model", required=True,
                    choices=["zprime_constrained","zprime",
                             "wprime_constrained","wprime_universal"])
    ap.add_argument("--gZH",  type=float, default=0.12)
    ap.add_argument("--gZl",  type=float, default=0.04)
    ap.add_argument("--mZp",  type=float, default=7.0)
    ap.add_argument("--gWH",  type=float, default=0.20)
    ap.add_argument("--gWLf", type=float, default=None)
    ap.add_argument("--mWp",  type=float, default=3.0)
    ap.add_argument("--out",  required=True)
    args = ap.parse_args()

    model = get_model(args)
    F_norm, uv_params, group_order = compute_fisher(args.tag, model)
    make_plot(F_norm, uv_params, group_order, args.out)


if __name__ == "__main__":
    main()
