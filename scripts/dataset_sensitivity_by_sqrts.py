"""
scripts/dataset_sensitivity_by_sqrts.py

Supervisor task: re-cluster datasets by √s and show which energy stage
drives the W' and Z' discovery sensitivity.

For each dataset block b we compute:
  - chi2_b  = delta_b^T C_b^{-1} delta_b        (signal chi2 contribution)
  - Fgrad_b = K_b^T C_b^{-1} K_b c_truth        (full Fisher-gradient per op per block)

where delta_b = K_b c_truth is the BSM signal in block b.

The right panel shows the Fisher gradient (Fc)_i decomposed by √s block, using
the FULL Fisher matrix (including off-diagonal correlations) evaluated at c_truth.
This replaces the old diagonal-only F_diag_ii = (K^T C^{-1} K)_ii which ignores
operator correlations.

Results grouped into four √s bins:
  91 GeV  : Z-pole (FCCee_Zdata, FCCee_Wwidth, FCCee_Brw)
  161 GeV : WW threshold (FCCee_ww_161GeV)
  240 GeV : All datasets with 240 in name
  365 GeV : All datasets with 365 in name

Two-panel output per model:
  Left  : BSM signal chi2 fraction by √s (stacked bar, grouped by √s)
  Right : Fisher gradient |(F_b c)_i| by operator, stacked by √s group

Usage:
    python scripts/dataset_sensitivity_by_sqrts.py
    python scripts/dataset_sensitivity_by_sqrts.py --gstar 0.18 --mWp 10.0 --mZp 10.0
"""

import sys, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import os, yaml, json

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))

from models.wprime_constrained import WPrimeConstrainedModel
from models.zprime_constrained  import ZPrimeConstrainedModel

DB      = "/data/theorie/wtelman/smefit_database"
SM_DATA = f"{DB}/commondata_projections_L0"
THEORY  = f"{DB}/theory"

# Same dataset list as run_pipeline.py (51 datasets)
DATASETS = [
    "FCCee_Wwidth", "FCCee_Zdata",
    "FCCee_ww_161GeV", "FCCee_ww_240GeV", "FCCee_ww_365GeV",
    "FCCee_Brw",
    "FCCee_Rb_240GeV", "FCCee_Rc_240GeV", "FCCee_Rmu_240GeV", "FCCee_Rtau_240GeV",
    "FCCee_ee_240GeV", "FCCee_ee_Afb_240GeV",
    "FCCee_bb_Afb_240GeV", "FCCee_cc_Afb_240GeV",
    "FCCee_mumu_Afb_240GeV", "FCCee_tautau_Afb_240GeV", "FCCee_sigmaHad_240GeV",
    "FCCee_Rb_365GeV", "FCCee_Rc_365GeV", "FCCee_Rmu_365GeV", "FCCee_Rtau_365GeV",
    "FCCee_ee_365GeV", "FCCee_ee_Afb_365GeV",
    "FCCee_bb_Afb_365GeV", "FCCee_cc_Afb_365GeV",
    "FCCee_mumu_Afb_365GeV", "FCCee_tautau_Afb_365GeV", "FCCee_sigmaHad_365GeV",
    "FCCee_zh_240GeV", "FCCee_zh_WW_240GeV", "FCCee_zh_ZZ_240GeV",
    "FCCee_zh_aZ_240GeV", "FCCee_zh_aa_240GeV", "FCCee_zh_tautau_240GeV",
    "FCCee_zh_mumu_240GeV", "FCCee_240_H_HADR",
    "FCCee_zh_365GeV", "FCCee_zh_WW_365GeV", "FCCee_zh_ZZ_365GeV",
    "FCCee_zh_aZ_365GeV", "FCCee_zh_aa_365GeV", "FCCee_zh_tautau_365GeV",
    "FCCee_zh_mumu_365GeV", "FCCee_365_H_HADR",
    "FCCee_vvh_WW_365GeV", "FCCee_vvh_ZZ_365GeV", "FCCee_vvh_aZ_365GeV",
    "FCCee_vvh_aa_365GeV", "FCCee_vvh_tautau_365GeV",
]

SQRTS_GROUPS = {
    "91 GeV":  {"FCCee_Wwidth", "FCCee_Zdata", "FCCee_Brw"},
    "161 GeV": {"FCCee_ww_161GeV"},
    "240 GeV": set(),   # filled below
    "365 GeV": set(),
}
for ds in DATASETS:
    if "240" in ds:
        SQRTS_GROUPS["240 GeV"].add(ds)
    elif "365" in ds:
        SQRTS_GROUPS["365 GeV"].add(ds)

GROUP_ORDER  = ["91 GeV", "161 GeV", "240 GeV", "365 GeV"]
GROUP_COLORS = {"91 GeV": "#2196F3", "161 GeV": "#FF9800",
                "240 GeV": "#4CAF50", "365 GeV": "#9C27B0"}


def sqrts_of(ds):
    for g, members in SQRTS_GROUPS.items():
        if ds in members:
            return g
    return "other"


def build_per_dataset(ops, use_theory_cov=True):
    """
    Returns list of dicts, one per dataset that loads successfully:
        ds, n_obs, K_ds (n_obs x n_ops), Ci_ds (n_obs x n_obs)
    """
    result = []
    for ds in DATASETS:
        sm_path = f"{SM_DATA}/{ds}.yaml"
        th_path = f"{THEORY}/{ds}.json"
        if not os.path.exists(sm_path) or not os.path.exists(th_path):
            continue
        sm = yaml.safe_load(open(sm_path))
        th = json.load(open(th_path))
        lo = th.get("LO", {})
        dc = sm["data_central"]
        dc = [dc] if not isinstance(dc, list) else list(dc)
        n  = len(dc)

        stat = sm.get("statistical_error", [0.0] * n)
        stat = list(stat) if isinstance(stat, list) else [stat] * n
        C = np.diag([float(e)**2 if float(e) > 0 else (0.01*abs(float(d)))**2
                     for e, d in zip(stat, dc)])
        sys_mat = sm.get("systematics", None)
        if sys_mat is not None:
            S = np.array(sys_mat, dtype=float)
            if S.ndim == 2 and S.shape[0] == n:
                C += S @ S.T
        if use_theory_cov:
            th_cov = th.get("theory_cov", None)
            if th_cov is not None:
                T = np.array(th_cov, dtype=float)
                if T.shape == (n, n):
                    C += T

        K_ds = np.zeros((n, len(ops)))
        for j, op in enumerate(ops):
            k = lo.get(op, 0.0)
            if isinstance(k, list):
                K_ds[:, j] = [float(k[i]) if i < len(k) else 0.0 for i in range(n)]
            else:
                K_ds[:, j] = float(k)

        try:
            Ci_ds = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            Ci_ds = np.diag(1.0 / np.maximum(np.diag(C), 1e-30))

        result.append({"ds": ds, "n": n, "K": K_ds, "Ci": Ci_ds})
    return result


def compute_contributions(model, ds_list):
    ops = model.OPERATORS
    c   = np.array([model.eft_coefficients().get(op, 0.0) for op in ops])

    per_ds = []
    for item in ds_list:
        delta_i  = item["K"] @ c                        # BSM signal in this block
        chi2_i   = float(delta_i @ item["Ci"] @ delta_i)
        # Full Fisher gradient contribution from this dataset block:
        #   (F_b c)_i = (K_b^T C_b^{-1} K_b c)_i = (K_b^T C_b^{-1} delta_b)_i
        # Using the full matrix product correctly accounts for operator correlations.
        Fgrad_i  = item["K"].T @ item["Ci"] @ delta_i   # shape (n_ops,)
        per_ds.append({
            "ds":     item["ds"],
            "group":  sqrts_of(item["ds"]),
            "chi2":   chi2_i,
            "Fgrad":  Fgrad_i,   # shape (n_ops,) — full Fisher gradient contribution
        })
    return ops, c, per_ds


def aggregate_by_group(per_ds, ops):
    agg = {g: {"chi2": 0.0, "Fgrad": np.zeros(len(ops))} for g in GROUP_ORDER}
    for item in per_ds:
        g = item["group"]
        if g in agg:
            agg[g]["chi2"]  += item["chi2"]
            agg[g]["Fgrad"] += item["Fgrad"]
    return agg


def plot_model(model, model_label, ds_list, ax_chi2, ax_fisher, ops_latex=None):
    ops, c, per_ds = compute_contributions(model, ds_list)
    agg = aggregate_by_group(per_ds, ops)

    total_chi2  = sum(a["chi2"]  for a in agg.values())
    total_Fgrad = sum(a["Fgrad"] for a in agg.values())   # shape (n_ops,)
    total_Fgrad_abs = np.abs(total_Fgrad)                 # for normalisation

    # --- BSM signal chi2 stacked bar ---
    left = 0.0
    for g in GROUP_ORDER:
        frac = agg[g]["chi2"] / total_chi2 * 100 if total_chi2 > 0 else 0
        ax_chi2.barh(model_label, frac, left=left,
                     color=GROUP_COLORS[g], edgecolor="white", height=0.55)
        if frac > 3:
            ax_chi2.text(left + frac/2, model_label, f"{frac:.0f}%",
                         ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        left += frac

    # --- Full Fisher-gradient |(F_b c)_i| stacked bar per operator ---
    # Each bar shows which √s block drives the Fisher gradient for that operator,
    # accounting for inter-operator correlations via the full off-diagonal F matrix.
    x      = np.arange(len(ops))
    width  = 0.65
    bottom = np.zeros(len(ops))
    for g in GROUP_ORDER:
        vals     = np.abs(agg[g]["Fgrad"])
        norm_vals = np.where(total_Fgrad_abs > 0, vals / total_Fgrad_abs * 100, 0)
        ax_fisher.bar(x, norm_vals, width, bottom=bottom,
                      color=GROUP_COLORS[g], edgecolor="white", linewidth=0.4)
        bottom += norm_vals

    labels = [ops_latex.get(op, op) if ops_latex else op for op in ops]
    ax_fisher.set_xticks(x)
    ax_fisher.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_fisher.set_ylim(0, 105)
    ax_fisher.set_ylabel(r"% of Fisher gradient $|(F_b c)_i|$", fontsize=9)
    ax_fisher.set_title(f"{model_label} — Fisher-gradient sensitivity by √s", fontsize=10)
    ax_fisher.axhline(100, color="k", lw=0.5, ls="--")
    ax_fisher.grid(axis="y", alpha=0.3)

    print(f"\n{model_label}  chi²_BSM = {total_chi2:.2f}")
    for g in GROUP_ORDER:
        pct = agg[g]["chi2"] / total_chi2 * 100 if total_chi2 > 0 else 0
        print(f"  {g:12s}: chi2={agg[g]['chi2']:9.2f}  ({pct:.1f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gstar",  type=float, default=0.18)
    p.add_argument("--mWp",    type=float, default=10.0)
    p.add_argument("--mZp",    type=float, default=10.0)
    args = p.parse_args()

    g   = args.gstar
    mWp = args.mWp
    mZp = args.mZp

    wp_model = WPrimeConstrainedModel(
        gWH=g, gWLf11=g, gWLf22=g, gWLf33=g, gWqf33=g, mWp=mWp
    )
    zp_model = ZPrimeConstrainedModel(gZH=g, gZl=g, mZp=mZp)

    ops_latex = {
        "O3pQ3": r"$\mathcal{O}_{Hq}^{(3)}$", "O3pl1": r"$\mathcal{O}_{Hl1}^{(3)}$",
        "O3pl2": r"$\mathcal{O}_{Hl2}^{(3)}$", "O3pl3": r"$\mathcal{O}_{Hl3}^{(3)}$",
        "OQl13": r"$\mathcal{O}_{Ql}^{13}$",   "OQl1M": r"$\mathcal{O}_{Ql}^{1-}$",
        "Obp":   r"$\mathcal{O}_{HB}$",         "Oll1111": r"$\mathcal{O}_{ll}^{1111}$",
        "Oll1122": r"$\mathcal{O}_{ll}^{1122}$","Oll1221": r"$\mathcal{O}_{ll}^{1221}$",
        "Oll1331": r"$\mathcal{O}_{ll}^{1331}$","OpBox": r"$\mathcal{O}_{H,\mathrm{box}}$",
        "OpQM":  r"$\mathcal{O}_{HQ}^{(-)}$",  "Otap":  r"$\mathcal{O}_{Ht}$",
        "Otp":   r"$\mathcal{O}_{HW}$",         "OpD":   r"$\mathcal{O}_{HD}$",
        "Opl1":  r"$\mathcal{O}_{Hl1}^{(1)}$", "Opl2":  r"$\mathcal{O}_{Hl2}^{(1)}$",
        "Opl3":  r"$\mathcal{O}_{Hl3}^{(1)}$",
    }

    # Build per-dataset blocks once (covers all operators for both models)
    all_ops = list(dict.fromkeys(wp_model.OPERATORS + zp_model.OPERATORS))
    ds_list_wp = build_per_dataset(wp_model.OPERATORS)
    ds_list_zp = build_per_dataset(zp_model.OPERATORS)

    fig = plt.figure(figsize=(18, 11))
    gs  = fig.add_gridspec(3, 2, height_ratios=[1, 4, 4],
                            hspace=0.55, wspace=0.35)

    # Legend row
    ax_leg = fig.add_subplot(gs[0, :])
    ax_leg.axis("off")
    patches = [mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in GROUP_ORDER]
    ax_leg.legend(handles=patches, loc="center", ncol=4, fontsize=11,
                  title="FCC-ee energy stage", title_fontsize=11)

    # chi2 rows
    ax_chi2_wp = fig.add_subplot(gs[1, 0])
    ax_chi2_zp = fig.add_subplot(gs[2, 0])
    ax_fish_wp = fig.add_subplot(gs[1, 1])
    ax_fish_zp = fig.add_subplot(gs[2, 1])

    wp_label = fr"W' ($g^*={g}$, $m_{{W'}}={mWp:.0f}$ TeV)"
    zp_label = fr"Z' ($g^*={g}$, $m_{{Z'}}={mZp:.0f}$ TeV)"

    for ax in [ax_chi2_wp, ax_chi2_zp]:
        ax.set_xlabel(r"Fraction of total BSM signal $\chi^2$ [%]", fontsize=9)
        ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(left=False)

    ax_chi2_wp.set_title(r"BSM signal $\chi^2 = s^T C^{-1} s$ by $\sqrt{s}$", fontsize=10)
    ax_chi2_zp.set_title(r"BSM signal $\chi^2 = s^T C^{-1} s$ by $\sqrt{s}$", fontsize=10)

    plot_model(wp_model, wp_label, ds_list_wp, ax_chi2_wp, ax_fish_wp, ops_latex)
    plot_model(zp_model, zp_label, ds_list_zp, ax_chi2_zp, ax_fish_zp, ops_latex)

    out_dir = PIPELINE / "results" / "dataset_sensitivity_sqrts_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"gstar{int(g*100):03d}_mwp{int(mWp*10):03d}_mzp{int(mZp*10):03d}"
    for ext in ["png", "pdf"]:
        path = out_dir / f"sqrts_sensitivity_fullFisher_{tag}.{ext}"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()

    # ── Numerical table ────────────────────────────────────────────────────────
    tbl = out_dir / f"sqrts_sensitivity_fullFisher_{tag}.txt"
    with open(tbl, "w") as f:
        for model, label, ds_list in [
            (wp_model, "W'", ds_list_wp),
            (zp_model, "Z'", ds_list_zp),
        ]:
            ops_m, c_m, per_ds = compute_contributions(model, ds_list)
            agg = aggregate_by_group(per_ds, ops_m)
            total = sum(a["chi2"] for a in agg.values())
            mass = mWp if label == "W'" else mZp
            f.write(f"# {label}  g*={g}  m={mass:.1f} TeV\n")
            f.write(f"# {'group':<12} {'chi2':>10} {'frac_%':>8}\n")
            for g_name in GROUP_ORDER:
                pct = agg[g_name]["chi2"] / total * 100 if total > 0 else 0
                f.write(f"  {g_name:<12} {agg[g_name]['chi2']:>10.2f} {pct:>8.2f}\n")
            f.write(f"  {'TOTAL':<12} {total:>10.2f} {'100.00':>8}\n\n")
    print(f"Table:  {tbl}")


if __name__ == "__main__":
    main()
