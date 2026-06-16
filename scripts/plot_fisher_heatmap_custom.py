"""
Regenerate a Fisher heatmap with improvements over the smefit default:
  - Short energy-stage labels (e.g. "91 GeV")
  - Columns sorted by ascending center-of-mass energy
  - No title
  - Dynamic colormap: normalised to the actual data maximum so relative
    differences are visible even when all values are small

Usage:
    python scripts/plot_fisher_heatmap_custom.py \\
        --rc   <report_yaml>   \\
        --fit  <fit_id>        \\   (default: first entry in result_IDs)
        --out  <output_stem>   \\   (no extension; .png and .pdf are written)
        --energy-groups            (auto-group datasets by √s instead of yaml grouping)

Examples:
    # allg012 UVcoup run (default)
    python scripts/plot_fisher_heatmap_custom.py

    # constrained run, SMEFT fit, energy grouping, dynamic colour scale
    python scripts/plot_fisher_heatmap_custom.py \\
        --rc results/wprime_constrained_gwh012_mwp010/runcards/Report_wprime_constrained_gwh012_mwp010.yaml \\
        --fit wprime_constrained_gwh012_mwp010_BSMclosure_SMEFT \\
        --out results/wprime_constrained_gwh012_mwp010/reports/Report_wprime_constrained_gwh012_mwp010/meta/fisher_heatmap_custom \\
        --energy-groups
"""
import sys, os, re, argparse, copy
import yaml
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon
from pathlib import Path

sys.path.insert(0, "/data/theorie/wtelman/smefit_release/src")
from smefit.fit_manager import FitManager
from smefit.analyze.fisher import FisherCalculator


def _truth_linear_corrections(coefficients, datasets, truth_vals, eps=1e-6):
    """Jacobian of SMEFT coefficients w.r.t. UV free parameters at the truth point.

    Standard impose_constrain sets all-but-one UV param to 0, so mixed operators
    (e.g. c ~ gWH * gWLf / m²) contribute zero.  This version perturbs around
    truth_vals, giving the correct ∂c/∂g_α for every operator including mixed ones.

    Returns array of shape (n_free_params, n_dat), same format as new_LinearCorrections.
    """
    temp = copy.deepcopy(coefficients)
    free_params = temp.free_parameters.index
    truth = np.array([truth_vals.get(str(p), 0.0) for p in free_params])

    rows = []
    for idx in range(len(free_params)):
        pp = truth.copy(); pp[idx] += eps
        pm = truth.copy(); pm[idx] -= eps

        temp.set_free_parameters(pp)
        temp.set_constraints()
        c_p = temp.value.copy()

        temp.set_free_parameters(pm)
        temp.set_constraints()
        c_m = temp.value.copy()

        jac_col = (c_p - c_m) / (2.0 * eps)
        rows.append(jac_col @ datasets.LinearCorrections.T)

    return np.array(rows)

# ── defaults ──────────────────────────────────────────────────────────────────
_BASE = Path("/data/theorie/wtelman/wprime_explicit_uv_closure_vs_sm/fullpipeline")
DEFAULT_RC  = str(_BASE / "results/wprime_gwh012_allg012_mwp010/runcards"
                         "/Report_wprime_gwh012_allg012_mwp010_UVcoup.yaml")
DEFAULT_OUT = str(_BASE / "results/wprime_gwh012_allg012_mwp010/reports"
                         "/Report_wprime_gwh012_allg012_mwp010_UVcoup/meta"
                         "/fisher_heatmap_custom")

# Canonical energy-based grouping (applied when --energy-groups is set)
_ENERGY_GROUPS = {
    "91 GeV": [
        "FCCee_Zdata", "FCCee_Wwidth",
    ],
    "161 GeV": [
        "FCCee_ww_161GeV",
    ],
    "240 GeV": [
        "FCCee_ww_240GeV", "FCCee_Rb_240GeV", "FCCee_Rc_240GeV",
        "FCCee_Rmu_240GeV", "FCCee_Rtau_240GeV", "FCCee_bb_Afb_240GeV",
        "FCCee_cc_Afb_240GeV", "FCCee_mumu_Afb_240GeV", "FCCee_sigmaHad_240GeV",
        "FCCee_tautau_Afb_240GeV", "FCCee_zh_240GeV", "FCCee_zh_WW_240GeV",
        "FCCee_zh_ZZ_240GeV", "FCCee_zh_aZ_240GeV", "FCCee_zh_aa_240GeV",
        "FCCee_zh_tautau_240GeV",
    ],
    "365 GeV": [
        "FCCee_ww_365GeV", "FCCee_Rb_365GeV", "FCCee_Rc_365GeV",
        "FCCee_Rmu_365GeV", "FCCee_Rtau_365GeV", "FCCee_bb_Afb_365GeV",
        "FCCee_cc_Afb_365GeV", "FCCee_mumu_Afb_365GeV", "FCCee_sigmaHad_365GeV",
        "FCCee_tautau_Afb_365GeV", "FCCee_zh_365GeV", "FCCee_zh_WW_365GeV",
        "FCCee_zh_ZZ_365GeV", "FCCee_zh_aa_365GeV", "FCCee_zh_tautau_365GeV",
    ],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def short_label(name: str) -> str:
    """Strip parenthetical qualifiers: '161 GeV (WW thr.)' → '161 GeV'."""
    return re.sub(r"\s*\(.*?\)", "", name).strip()


def _energy_gev(label: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*GeV", label)
    return float(m.group(1)) if m else 0.0


def sort_by_energy(labels: list) -> list:
    return sorted(labels, key=_energy_gev)


def build_data_info(raw_dict: dict, fit_datasets: list) -> pd.Series:
    out = {}
    for group, entries in raw_dict.items():
        out[group] = {name: label for name, label in entries if name in fit_datasets}
        if not out[group]:
            out.pop(group)
    if not out:
        return pd.DataFrame().stack()
    return pd.DataFrame(out).stack().swaplevel()


def build_energy_data_info(fit_datasets: list) -> pd.Series:
    """Build data_info grouped by center-of-mass energy from the canonical map."""
    raw = {
        group: [(ds, ds) for ds in datasets if ds in fit_datasets]
        for group, datasets in _ENERGY_GROUPS.items()
    }
    return build_data_info({g: v for g, v in raw.items() if v}, fit_datasets)


def build_coeff_info(raw_dict: dict, fit_coeffs: list) -> pd.Series:
    out = {}
    for group, entries in raw_dict.items():
        out[group] = {name: label for name, label in entries if name in fit_coeffs}
        if not out[group]:
            out.pop(group)
    if not out:
        return pd.DataFrame().stack()
    return pd.DataFrame(out).stack().swaplevel()


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_heatmap_custom(lin_df, quad_df, coeff_info, fig_name, sort_order=None):
    if sort_order is not None:
        lin_df = lin_df.loc[sort_order]
        if quad_df is not None:
            quad_df = quad_df.loc[sort_order]

    x_labels = [short_label(g) for g in lin_df.index]
    col_order = coeff_info.index.get_level_values(1)
    lin_df    = lin_df[col_order]
    if quad_df is not None:
        quad_df = quad_df[col_order]
    y_labels  = list(coeff_info.values)

    # ── dynamic colour scale: 0 → actual max across all panels
    all_vals = lin_df.values.copy()
    if quad_df is not None:
        all_vals = np.concatenate([all_vals.ravel(), quad_df.values.ravel()])
    vmax = float(np.nanmax(all_vals))
    # round up to a nice number for the colour bar
    magnitude = 10 ** np.floor(np.log10(vmax)) if vmax > 0 else 1
    vmax_nice = np.ceil(vmax / magnitude) * magnitude

    cmap_full = plt.get_cmap("Blues")
    cmap = colors.LinearSegmentedColormap.from_list(
        "trunc_blues", cmap_full(np.linspace(0, 0.8, 256))
    )
    cnorm = colors.Normalize(vmin=0, vmax=vmax_nice)

    has_quad = quad_df is not None
    # wider figure when many operators (tall grid)
    nC = lin_df.shape[1]
    nE = lin_df.shape[0]
    cell = 0.75
    fig_h = max(5, nC * cell + 1.5)
    fig_w = (nE * cell + 2.5) * (2 if has_quad else 1) + (1.5 if has_quad else 0)
    fig = plt.figure(figsize=(fig_w, fig_h))

    def _draw_panel(ax, df):
        nE_p, nC_p = df.shape
        for ei in range(nE_p):
            for ci in range(nC_p):
                val = float(df.iloc[ei, ci])
                y   = nC_p - 1 - ci
                rect = Polygon(
                    [[ei-0.5, y-0.5], [ei+0.5, y-0.5],
                     [ei+0.5, y+0.5], [ei-0.5, y+0.5]],
                    closed=True, ec="grey", color=cmap(cnorm(val))
                )
                ax.add_patch(rect)
                if val > 0:
                    ax.text(ei, y, f"{val:.1f}", va="center", ha="center",
                            fontsize=9)

        ax.set_xlim(-0.5, nE_p - 0.5)
        ax.set_ylim(-0.5, nC_p - 0.5)
        ax.set_aspect("equal", adjustable="box")

        xticks = np.arange(nE_p)
        ax.set_xticks(xticks, labels=x_labels, rotation=90, fontsize=13)
        ax.xaxis.set_ticks_position("top")
        ax.set_xticks(xticks - 0.5, minor=True)

        yticks = np.arange(nC_p)
        ax.set_yticks(yticks, labels=y_labels[::-1], fontsize=13)
        ax.set_yticks(yticks - 0.5, minor=True)

        ax.tick_params(which="major", top=False, bottom=False, left=False)
        ax.tick_params(which="minor", bottom=False)
        ax.grid(visible=True, which="minor", alpha=0.2)

        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.5)
        cb  = fig.colorbar(mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap), cax=cax)
        cb.set_label(r"${\rm Normalized\ Value}$", fontsize=14, labelpad=20,
                     rotation=270)

    if has_quad:
        ax1 = fig.add_subplot(121)
        _draw_panel(ax1, lin_df)
        ax1.set_title(r"$\rm Linear$",    fontsize=16, y=-0.05)
        ax2 = fig.add_subplot(122)
        _draw_panel(ax2, quad_df)
        ax2.set_title(r"$\rm Quadratic$", fontsize=16, y=-0.05)
    else:
        ax = fig.add_subplot(111)
        _draw_panel(ax, lin_df)
        ax.set_title(r"$\rm Linear$", fontsize=16, y=-0.04)

    fig.subplots_adjust(top=0.85, bottom=0.04, left=0.15, right=0.92,
                        wspace=0.7)
    plt.savefig(f"{fig_name}.pdf", bbox_inches="tight")
    plt.savefig(f"{fig_name}.png", bbox_inches="tight", dpi=150)
    print(f"  Saved: {fig_name}.png  (vmax={vmax_nice:.1f})")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rc",            default=DEFAULT_RC,  help="Report yaml path")
    p.add_argument("--fit",           default=None,        help="Fit ID (default: first in result_IDs)")
    p.add_argument("--out",           default=DEFAULT_OUT, help="Output path stem (no extension)")
    p.add_argument("--energy-groups", action="store_true", help="Group datasets by √s instead of yaml grouping")
    p.add_argument("--truth-point",   nargs="*", metavar="PARAM=VALUE",
                   help="Evaluate Jacobian at this UV truth point, e.g. "
                        "gWH=0.12 gWLf11=0.04 gWLf22=0.04 gWLf33=0.04 gWqf33=0.04 mWp_TeV=1.0 "
                        "(default: origin, i.e. standard impose_constrain behaviour)")
    args = p.parse_args()

    with open(args.rc) as f:
        rc = yaml.safe_load(f)

    fit_name = args.fit if args.fit else rc["result_IDs"][0]
    fit_path = Path(rc.get("result_path",
                           str(Path(args.rc).parent.parent / "fits")))

    print(f"Loading fit: {fit_name}")
    fit = FitManager(str(fit_path), fit_name)
    fit.load_results()
    fit.load_datasets()

    fit_datasets = [d["name"] for d in fit.config["datasets"]]
    fit_coeffs   = list(fit.config["coefficients"].keys())

    if args.energy_groups:
        data_info = build_energy_data_info(fit_datasets)
        print("  Using energy-stage grouping (91/161/240/365 GeV)")
    else:
        data_info = build_data_info(
            {g: [(e[0], e[1]) for e in v] for g, v in rc["data_info"].items()},
            fit_datasets,
        )

    coeff_info = build_coeff_info(
        {g: [(e[0], e[1]) for e in v] for g, v in rc["coeff_info"].items()},
        fit_coeffs,
    )

    fisher_norm = rc.get("fisher", {}).get("norm", "coeff")
    fisher_log  = rc.get("fisher", {}).get("log",  False)
    compute_quad = fit.config.get("use_quad", False)

    fisher = FisherCalculator(fit.coefficients, fit.datasets, compute_quad)

    if args.truth_point:
        truth_vals = {}
        for item in args.truth_point:
            k, v = item.split("=")
            truth_vals[k.strip()] = float(v.strip())
        print(f"  Overriding Jacobian at truth point: {truth_vals}")
        fisher.new_LinearCorrections = _truth_linear_corrections(
            fit.coefficients, fit.datasets, truth_vals
        )

    fisher.compute_linear()
    fisher.lin_fisher    = fisher.normalize(fisher.lin_fisher, norm=fisher_norm, log=fisher_log)
    fisher.summary_table = fisher.groupby_data(fisher.lin_fisher, data_info,
                                               norm=fisher_norm, log=fisher_log)

    quad_summary = None
    if compute_quad:
        fisher.compute_quadratic(fit.results["samples"], fit.smeft_predictions)
        fisher.quad_fisher     = fisher.normalize(fisher.quad_fisher, norm=fisher_norm, log=fisher_log)
        fisher.summary_HOtable = fisher.groupby_data(fisher.quad_fisher, data_info,
                                                     norm=fisher_norm, log=fisher_log)
        quad_summary = fisher.summary_HOtable

    sort_order = sort_by_energy(list(fisher.summary_table.index))
    print(f"  Column order: {sort_order}")

    plot_heatmap_custom(
        fisher.summary_table, quad_summary,
        coeff_info, args.out,
        sort_order=sort_order,
    )


if __name__ == "__main__":
    main()
