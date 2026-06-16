"""
bsm_closure_metrics.py
----------------------
Two quantitative closure-test metrics validating the BSM analysis pipeline,
in analogy with PDF closure tests (see Rojo et al.).

Metric 1 — data-space p-value
  For each of N_rep L1 BSM pseudo-datasets (BSM signal + Gaussian noise),
  compute the test statistic q = chi2_SM - chi2_UV_min. Convert to a p-value
  under the SM null hypothesis. The observed distribution of p-values must
  match the analytic prediction given the known non-centrality lambda.

Metric 2 — Wilson coefficient pulls
  For each L1 BSM pseudo-dataset, compute the analytic SMEFT MLE c_i^fit
  and the posterior width sigma_i from the Fisher matrix. Compute the pull
      P_i^(r) = (c_i^{fit,(r)} - c_i^{truth}) / sigma_i
  If the closure test is successful, the pooled pull distribution (over all
  replicas and operators) should be Gaussian with mean=0 and width=1.
  A NS-fit validation column uses the existing Asimov NS posterior for comparison.

Run for 5 model points: W' at mWp = 3, 4, 5 TeV; Z' at mZp = 7 TeV;
Composite Higgs at g_rho=2.0, m_rho=100 TeV (5σ sweet spot).

Usage:
    python scripts/bsm_closure_metrics.py [--nrep 2000] [--seed 42]
"""
import sys, argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2_dist, ncx2 as ncx2_dist, norm as norm_dist
from pathlib import Path

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))

from models.wprime_constrained import WPrimeConstrainedModel
from models.zprime_constrained  import ZPrimeConstrainedModel
from models.zprime              import ZPrimeModel
from models.comphiggs           import CompHiggsModel
from scripts.run_pipeline       import _build_K_Ci


# ── model-point registry ─────────────────────────────────────────────────────

def make_points():
    """Return list of (label, run_tag, model) for all closure-test points."""
    f = 1.0 / 3.0
    return [
        (r"W$'$, $m=1.0$ TeV",
         "wprime_constrained_gwh012_mwp010",
         WPrimeConstrainedModel(gWH=0.12, gWLf11=0.12*f, gWLf22=0.12*f,
                                gWLf33=0.12*f, gWqf33=0.12*f, mWp=1.0)),
        (r"W$'$, $m=1.25$ TeV",
         "wprime_constrained_gwh012_mwp012",
         WPrimeConstrainedModel(gWH=0.12, gWLf11=0.12*f, gWLf22=0.12*f,
                                gWLf33=0.12*f, gWqf33=0.12*f, mWp=1.25)),
        (r"W$'$, $m=1.5$ TeV",
         "wprime_constrained_gwh012_mwp015",
         WPrimeConstrainedModel(gWH=0.12, gWLf11=0.12*f, gWLf22=0.12*f,
                                gWLf33=0.12*f, gWqf33=0.12*f, mWp=1.5)),
        (r"Z$'$, $m=5$ TeV",
         "zprime_gzh012_mzp050",
         ZPrimeModel(gZH=0.12, gZl=0.12*f, mZp=5.0)),
        (r"Z$'$, $m=7$ TeV",
         "zprime_gzh012_mzp070",
         ZPrimeModel(gZH=0.12, gZl=0.12*f, mZp=7.0)),
        (r"Z$'$, $m=9$ TeV",
         "zprime_gzh012_mzp090",
         ZPrimeModel(gZH=0.12, gZl=0.12*f, mZp=9.0)),
        (r"Z$'$, $m=11$ TeV",
         "zprime_gzh012_mzp110",
         ZPrimeModel(gZH=0.12, gZl=0.12*f, mZp=11.0)),
        (r"CHM, $m_\rho=100$ TeV",
         "comphiggs_grho200_mrho1000",
         CompHiggsModel(g_rho=2.0, m_rho=100.0)),
    ]


# ── UV Jacobian via numerical differentiation ─────────────────────────────────

def uv_jacobian(model, ops, eps=1e-5):
    """dc_i/dtheta_j  (n_ops × n_UV) at the truth point."""
    params = model.uv_param_names()
    J = np.zeros((len(ops), len(params)))
    for j, pname in enumerate(params):
        import dataclasses
        kwargs = dataclasses.asdict(model)
        kw_p = {**kwargs, pname: kwargs[pname] + eps}
        kw_m = {**kwargs, pname: kwargs[pname] - eps}
        mp = type(model)(**kw_p)
        mm = type(model)(**kw_m)
        cp = np.array([mp.eft_coefficients().get(op, 0.0) for op in ops])
        cm = np.array([mm.eft_coefficients().get(op, 0.0) for op in ops])
        J[:, j] = (cp - cm) / (2 * eps)
    return J   # (n_ops, n_UV)


# ── per-point computation ─────────────────────────────────────────────────────

def compute_point(run_tag, model, n_rep, seed):
    """
    Load data and run both metrics for one model point.

    Returns a dict with keys:
      label, n_ops, n_uv,
      lambda_smeft, lambda_uv,
      q_uv_bsm, q_uv_sm,       # (n_rep,)
      p_uv_bsm, p_uv_sm,
      pulls_l1,                  # (n_rep * n_ops,) analytic L1 pulls
      ns_pull_vals,              # (n_ops,)   NS Asimov pull
      ns_ops,                    # operator names
      sigma_ratio_ns,            # sigma_NS / sigma_analytic per operator
    """
    # Build K and Ci fresh from the theory database — guarantees dimension
    # consistency regardless of what version of the model stored K_fit.npy.
    ops   = model.OPERATORS
    K, Ci = _build_K_Ci(ops)                      # (n_data, n_ops), (n_data, n_data)

    # covariance and Cholesky for noise generation
    C = np.linalg.inv(Ci)
    L = np.linalg.cholesky(C)

    n_ops = len(ops)
    n_uv  = len(model.uv_param_names())

    truth   = model.eft_coefficients()
    c_truth = np.array([truth.get(op, 0.0) for op in ops])

    # L0 delta: BSM signal in observable space (exact, no stored projections needed)
    delta = K @ c_truth                            # (n_data,)

    # SMEFT Fisher matrix (pinv handles rank-deficient F when n_ops > effective rank)
    F     = K.T @ Ci @ K                          # (n_ops, n_ops)
    F_inv = np.linalg.pinv(F)
    sigma_i = np.sqrt(np.abs(np.diag(F_inv)))     # analytic posterior widths

    # SMEFT truth as seen by the linear fit
    # (F^{-1} K^T Ci delta): what the fit exactly recovers from the signal
    c_truth_fit = F_inv @ (K.T @ Ci @ delta)

    # UV Jacobian and Fisher matrix
    J_uv  = uv_jacobian(model, ops)               # (n_ops, n_uv)
    G     = K @ J_uv                               # (n_data, n_uv)
    F_uv  = G.T @ Ci @ G                           # (n_uv, n_uv)
    F_uv_inv = np.linalg.pinv(F_uv)

    # non-centrality parameters (Asimov signal strength)
    h_truth    = G.T @ Ci @ delta
    lambda_uv  = float(h_truth @ F_uv_inv @ h_truth)
    lambda_smeft = float(delta @ Ci @ delta)       # = chi2_SM

    # ── generate L1 replicas ─────────────────────────────────────────────────
    rng   = np.random.default_rng(seed)
    noise = L @ rng.standard_normal((len(delta), n_rep))   # (n_data, n_rep)
    data  = delta[:, np.newaxis] + noise                   # (n_data, n_rep)

    # also generate SM null replicas (no signal)
    noise_sm = L @ rng.standard_normal((len(delta), n_rep))
    data_sm  = noise_sm

    # UV q-statistics
    h_bsm = G.T @ Ci @ data                               # (n_uv, n_rep)
    h_sm  = G.T @ Ci @ data_sm
    q_uv_bsm = np.einsum('ir,ij,jr->r', h_bsm, F_uv_inv, h_bsm)
    q_uv_sm  = np.einsum('ir,ij,jr->r', h_sm,  F_uv_inv, h_sm)

    p_uv_bsm = chi2_dist.sf(q_uv_bsm, df=n_uv)
    p_uv_sm  = chi2_dist.sf(q_uv_sm,  df=n_uv)

    # ── analytic SMEFT L1 pulls ───────────────────────────────────────────────
    # c_fit^(r) = F^{-1} K^T Ci data^(r)
    # pull     = (c_fit_i^(r) - c_truth_fit_i) / sigma_i
    g_bsm   = K.T @ Ci @ data                             # (n_ops, n_rep)
    c_fit   = F_inv @ g_bsm                               # (n_ops, n_rep)
    # Mask operators with zero posterior width (degenerate directions in data space)
    constrained = sigma_i > 1e-6 * sigma_i.max()
    pulls_l1 = ((c_fit[constrained] - c_truth_fit[constrained, np.newaxis])
                / sigma_i[constrained, np.newaxis]).ravel()  # (n_rep * n_constrained,)

    # ── NS Asimov pull (validation) ──────────────────────────────────────────
    ns_pull_vals = np.array([])
    ns_ops       = []
    sigma_ratio_ns = np.array([])

    fits_dir = PIPELINE / "results" / run_tag / "fits"
    json_glob = list(fits_dir.glob("*BSMclosure_SMEFT/fit_results.json"))
    if json_glob:
        res = json.load(open(json_glob[0]))
        samples = res["samples"]
        ns_pull_list = []
        sigma_ratio_list = []
        for i, op in enumerate(ops):
            if op not in samples:
                continue
            s = np.array(samples[op])
            c_fit_ns  = float(np.mean(s))
            sigma_ns  = float(np.std(s, ddof=1))
            if sigma_ns <= 0:
                continue
            pull = (c_fit_ns - c_truth_fit[i]) / sigma_i[i]
            ns_pull_list.append((op, pull))
            sigma_ratio_list.append(sigma_ns / sigma_i[i])
        ns_pull_vals   = np.array([v for _, v in ns_pull_list])
        ns_ops         = [o for o, _ in ns_pull_list]
        sigma_ratio_ns = np.array(sigma_ratio_list)

    return dict(
        n_ops=n_ops, n_uv=n_uv,
        lambda_smeft=lambda_smeft, lambda_uv=lambda_uv,
        q_uv_bsm=q_uv_bsm, q_uv_sm=q_uv_sm,
        p_uv_bsm=p_uv_bsm, p_uv_sm=p_uv_sm,
        pulls_l1=pulls_l1,
        ns_pull_vals=ns_pull_vals, ns_ops=ns_ops,
        sigma_ratio_ns=sigma_ratio_ns,
        c_truth_model=c_truth, c_truth_fit=c_truth_fit,
        sigma_i=sigma_i, ops=ops,
    )


# ── plotting ──────────────────────────────────────────────────────────────────

def q_to_sigma(q, k):
    p = chi2_dist.sf(float(q), df=k)
    return float(norm_dist.isf(p / 2)) if p > 0 else 99.0


def plot_metric1(points, results, out_path):
    """
    n-row × 2-col figure (one row per model point).
    Left:  q-statistic distribution (BSM L1 vs SM null vs analytic curves).
    Right: power curve — observed vs analytic ncx2 prediction.
    """
    n = len(points)
    fig, axes = plt.subplots(n, 2, figsize=(13, n * 4))

    col_bsm, col_sm = "#1f77b4", "#d62728"

    for row, (label, _rt, _model) in enumerate(points):
        R   = results[row]
        nu  = R["n_uv"]
        lam = R["lambda_uv"]

        q_bsm = R["q_uv_bsm"]
        q_sm  = R["q_uv_sm"]
        p_bsm = R["p_uv_bsm"]

        # -- left: q distribution --------------------------------------------
        ax = axes[row, 0]
        q_hi = max(float(np.percentile(q_bsm, 99.5)), ncx2_dist.ppf(0.999, nu, lam))
        bins = np.linspace(0, q_hi, 60)
        ax.hist(q_bsm, bins=bins, density=True, alpha=0.55, color=col_bsm,
                label=r"BSM $\ell_1$ replicas")
        ax.hist(q_sm,  bins=bins, density=True, alpha=0.55, color=col_sm,
                label=r"SM null replicas")
        xc = np.linspace(0, q_hi, 400)
        ax.plot(xc, chi2_dist.pdf(xc, df=nu), "r-", lw=2,
                label=fr"$\chi^2({nu})$  [SM null]")
        ax.plot(xc, ncx2_dist.pdf(xc, df=nu, nc=lam), "b--", lw=2,
                label=fr"noncen $\chi^2({nu},\lambda\!=\!{lam:.1f})$")
        thr3 = chi2_dist.ppf(1 - 2.7e-3, df=nu)
        thr5 = chi2_dist.ppf(1 - 5.7e-7, df=nu)
        ax.axvline(thr3, color="gray", lw=1.2, ls="--", alpha=0.8,
                   label=r"$3\sigma$ threshold")
        ax.axvline(thr5, color="gray", lw=0.8, ls=":",  alpha=0.8,
                   label=r"$5\sigma$ threshold")
        pwr3 = (p_bsm < 2.7e-3).mean() * 100
        pwr5 = (p_bsm < 5.7e-7).mean() * 100
        ax.text(0.97, 0.95,
                fr"Power $3\sigma$: {pwr3:.0f}%"+"\n"+fr"Power $5\sigma$: {pwr5:.0f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
        ax.set_xlabel(r"$q = \chi^2_{\rm SM} - \chi^2_{\rm UV,min}$", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(label, fontsize=11, fontweight="bold")
        if row == 0:
            ax.legend(fontsize=8, loc="upper right")

        # -- right: power curve  ----------------------------------------------
        # Power = fraction of BSM L1 replicas whose p-value falls below the
        # significance threshold alpha.  Plotted against the threshold expressed
        # in Gaussian sigma.  Observed (empirical) vs analytic ncx2 prediction.
        # The SM null curve should hug the diagonal (type-I error = alpha).
        ax = axes[row, 1]

        sig_max = min(8.0, max(5.5, q_to_sigma(float(np.median(q_bsm)), nu) + 2))
        sigmas  = np.linspace(0, sig_max, 400)
        # q-threshold corresponding to each sigma value (one-sided chi2 test)
        q_thrs  = chi2_dist.ppf(norm_dist.cdf(sigmas), df=nu)

        # observed power from L1 BSM replicas
        power_obs = np.array([(q_bsm > t).mean() for t in q_thrs])
        # analytic expected power from ncx2(nu, lam)
        power_exp = ncx2_dist.sf(q_thrs, df=nu, nc=lam)
        # SM null type-I error rate (should track 1 - norm_cdf(sigma) = norm_sf)
        power_sm_obs = np.array([(q_sm > t).mean() for t in q_thrs])
        power_sm_exp = norm_dist.sf(sigmas)

        ax.plot(sigmas, power_obs, color=col_bsm, lw=2.2,
                label=r"BSM $\ell_1$ (observed)")
        ax.plot(sigmas, power_exp, color=col_bsm, lw=1.5, ls="--",
                label=r"BSM analytic (noncen $\chi^2$)")
        ax.plot(sigmas, power_sm_obs, color=col_sm, lw=1.5, alpha=0.8,
                label=r"SM null (observed)")
        ax.plot(sigmas, power_sm_exp, color=col_sm, lw=1.0, ls="--",
                alpha=0.6, label=r"SM null (expected $\equiv\alpha$)")

        for sig_mark, ls_m in [(3.0, "--"), (5.0, ":")]:
            if sig_mark <= sig_max:
                ax.axvline(sig_mark, color="gray", lw=1.0, ls=ls_m, alpha=0.7)

        # annotate power at 3σ and 5σ
        for sig_mark in [3.0, 5.0]:
            if sig_mark <= sig_max:
                idx = np.searchsorted(sigmas, sig_mark)
                pwr = float(power_obs[idx]) * 100
                ax.annotate(fr"{pwr:.0f}%",
                            xy=(sig_mark, float(power_obs[idx])),
                            xytext=(sig_mark + 0.15, float(power_obs[idx]) + 0.04),
                            fontsize=9, color=col_bsm)

        ax.set_xlim(0, sig_max)
        ax.set_ylim(-0.03, 1.08)
        ax.set_xlabel(r"Significance threshold  $\sigma$", fontsize=11)
        ax.set_ylabel(r"Power = $P(p < \alpha \mid \mathrm{BSM})$", fontsize=11)
        ax.set_title(label + r"  —  power curve", fontsize=11)
        if row == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(r"Metric 1: data-space closure test  "
                 r"(left: $q$ distribution;  right: power curve — observed vs analytic)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{out_path}.png", bbox_inches="tight", dpi=150)
    plt.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}.png")


def plot_metric2(points, results, out_path):
    """n-row × 2-col: L1 pull histogram (left) and NS per-operator pulls (right)."""
    n = len(points)
    fig, axes = plt.subplots(n, 2, figsize=(13, n * 4))

    for row, (label, _rt, _model) in enumerate(points):
        R = results[row]
        pulls = R["pulls_l1"]

        # -- left: pooled L1 pull histogram -----------------------------------
        ax = axes[row, 0]
        bins = np.linspace(-4, 4, 24)
        ax.hist(pulls, bins=bins, density=True, alpha=0.65,
                color="#2196F3", edgecolor="white",
                label=fr"L1 pulls  (n={len(pulls):,})")
        x = np.linspace(-4, 4, 300)
        ax.plot(x, norm_dist.pdf(x), "r-", lw=2.5, label=r"$\mathcal{N}(0,1)$")
        mu  = float(pulls.mean())
        sig = float(pulls.std(ddof=1))
        ax.axvline(mu, color="#2196F3", lw=1.5, ls="--",
                   label=fr"Mean $={mu:+.3f}$")
        ax.text(0.97, 0.95,
                fr"$\mu={mu:+.3f}$" + "\n" + fr"$\sigma={sig:.3f}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
        ax.set_xlim(-4.5, 4.5)
        ax.set_xlabel(r"Pull  $P_i = (c_i^{\rm fit} - c_i^{\rm truth})\,/\,\sigma_i$",
                      fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(label + "  —  analytic L1 pulls", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")

        # -- right: NS Asimov per-operator pulls ------------------------------
        ax = axes[row, 1]
        ns_pulls = R["ns_pull_vals"]
        ns_ops   = R["ns_ops"]
        sig_ratio = R["sigma_ratio_ns"]
        n_ns = len(ns_pulls)

        if n_ns > 0:
            colors = ["tomato" if abs(p) > 2 else "steelblue" for p in ns_pulls]
            ax.barh(range(n_ns), ns_pulls, color=colors, alpha=0.8,
                    edgecolor="white", height=0.7)
            ax.axvline(0,  color="black", lw=0.8)
            ax.axvline( 1, color="gray",  lw=1.0, ls="--", label=r"$\pm1\sigma$")
            ax.axvline(-1, color="gray",  lw=1.0, ls="--")
            ax.axvline( 2, color="gray",  lw=0.6, ls=":",  label=r"$\pm2\sigma$")
            ax.axvline(-2, color="gray",  lw=0.6, ls=":")
            ax.set_yticks(range(n_ns))
            ax.set_yticklabels(ns_ops, fontsize=8)
            ax.set_xlim(-4, 4)
            # sigma ratio text
            ax.text(0.02, 0.02,
                    fr"NS $\sigma$ / analytic: "
                    fr"{float(sig_ratio.mean()):.3f} $\pm$ {float(sig_ratio.std(ddof=1)):.3f}",
                    transform=ax.transAxes, va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
        else:
            ax.text(0.5, 0.5, "NS fit not found", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="gray")

        ax.set_xlabel(r"Pull  $P_i$", fontsize=11)
        ax.set_title(label + "  —  NS Asimov pulls", fontsize=11)
        if row == 0 and n_ns > 0:
            ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(r"Metric 2: Wilson-coefficient pull distribution  "
                 r"($P_i = (c_i^{\rm fit}-c_i^{\rm truth})/\sigma_i$)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{out_path}.png", bbox_inches="tight", dpi=150)
    plt.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}.png")


def print_summary(points, results):
    print("\n" + "=" * 76)
    print(f"{'Model':<26} {'n_UV':>5} {'lambda_UV':>10} {'sigma_med':>10} "
          f"{'pow3σ':>7} {'pow5σ':>7} {'pull_mu':>8} {'pull_sig':>9}")
    print("-" * 76)
    for (label, _, _), R in zip(points, results):
        nu  = R["n_uv"]
        lam = R["lambda_uv"]
        q_bsm  = R["q_uv_bsm"]
        p_bsm  = R["p_uv_bsm"]
        sig_med = q_to_sigma(float(np.median(q_bsm)), nu)
        pow3 = (p_bsm < 2.7e-3).mean() * 100
        pow5 = (p_bsm < 5.7e-7).mean() * 100
        pull_mu  = float(R["pulls_l1"].mean())
        pull_sig = float(R["pulls_l1"].std(ddof=1))
        print(f"{label:<26} {nu:>5} {lam:>10.2f} {sig_med:>10.2f} "
              f"{pow3:>7.1f} {pow5:>7.1f} {pull_mu:>+8.3f} {pull_sig:>9.3f}")
    print("=" * 76)
    print(r"Expected for closed CT: pull_mu ~ 0, pull_sig ~ 1")
    print()


def save_summary_table(points, results, out_path):
    with open(out_path, "w") as f:
        f.write("# BSM closure test metrics\n")
        f.write(f"# Columns: model  n_UV  lambda_SMEFT  lambda_UV  "
                f"sigma_Asimov  power_3sigma(%)  power_5sigma(%)  "
                f"pull_mean  pull_std  ns_sigma_ratio_mean\n")
        for (label, _, _), R in zip(points, results):
            nu   = R["n_uv"]
            lam_s = R["lambda_smeft"]
            lam_u = R["lambda_uv"]
            q_as  = float(lam_u)   # Asimov q = lambda_UV
            sig_as = q_to_sigma(q_as, nu)
            pow3  = (R["p_uv_bsm"] < 2.7e-3).mean() * 100
            pow5  = (R["p_uv_bsm"] < 5.7e-7).mean() * 100
            pull_mu  = float(R["pulls_l1"].mean())
            pull_sig = float(R["pulls_l1"].std(ddof=1))
            ns_r = float(R["sigma_ratio_ns"].mean()) if len(R["sigma_ratio_ns"]) else float("nan")
            tag = label.replace("$", "").replace("'", "p").replace(",", "").replace(" ", "_")
            f.write(f"  {tag:<25} {nu:>4} {lam_s:>12.4f} {lam_u:>10.4f} "
                    f"{sig_as:>12.3f} {pow3:>15.1f} {pow5:>15.1f} "
                    f"{pull_mu:>10.4f} {pull_sig:>9.4f} {ns_r:>20.4f}\n")
    print(f"Saved: {out_path}")


def plot_summary(points, results, out_dir):
    """
    Two compact summary figures for the paper:
      (a) All four power curves on one panel — shows the progression from
          deep discovery (3 TeV) through the boundary (4 TeV) to below
          threshold (5 TeV), plus Z'.
      (b) Pull-width bar chart — the single-number closure verdict: σ_pull
          per model point, with a horizontal reference at 1.0.
    """
    _palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f"]
    colours = _palette[:len(points)]
    short_labels = [lbl for lbl, _, _ in points]

    # ── (a) combined power curve ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, ((label, _rt, _model), R, col, slabel) in enumerate(
            zip(points, results, colours, short_labels)):
        nu  = R["n_uv"]
        lam = R["lambda_uv"]
        q_bsm = R["q_uv_bsm"]

        sig_max = 7.5
        sigmas  = np.linspace(0, sig_max, 500)
        q_thrs  = chi2_dist.ppf(norm_dist.cdf(sigmas), df=nu)
        power_obs = np.array([(q_bsm > t).mean() for t in q_thrs])
        power_exp = ncx2_dist.sf(q_thrs, df=nu, nc=lam)

        ax.plot(sigmas, power_obs, color=col, lw=2.2, label=slabel)
        ax.plot(sigmas, power_exp, color=col, lw=1.2, ls="--", alpha=0.55)

    # SM null reference (same for all — just the alpha line)
    sigmas_ref = np.linspace(0, 7.5, 500)
    ax.plot(sigmas_ref, norm_dist.sf(sigmas_ref), color="gray", lw=1.0,
            ls=":", label=r"SM null ($\equiv\alpha$)")

    for sig_mark, ls_m, lbl in [(3.0, "--", r"$3\sigma$"),
                                  (5.0, ":",  r"$5\sigma$")]:
        ax.axvline(sig_mark, color="gray", lw=1.0, ls=ls_m, alpha=0.6)
        ax.text(sig_mark + 0.05, 1.02, lbl, fontsize=9, color="gray",
                transform=ax.get_xaxis_transform(), ha="left")

    ax.set_xlim(0, 7.5)
    ax.set_ylim(-0.03, 1.10)
    ax.set_xlabel(r"Significance threshold  $\sigma$", fontsize=12)
    ax.set_ylabel(r"Power = $P(p < \alpha \mid \mathrm{BSM})$", fontsize=12)
    ax.set_title(r"BSM closure test — discovery power at each model point",
                 fontsize=12)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    plt.tight_layout()
    out_a = str(out_dir / "summary_power_curves")
    plt.savefig(f"{out_a}.png", bbox_inches="tight", dpi=150)
    plt.savefig(f"{out_a}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_a}.png")

    # ── (b) pull-width bar chart ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))

    pull_sigs = [float(R["pulls_l1"].std(ddof=1)) for R in results]
    pull_mus  = [float(R["pulls_l1"].mean())       for R in results]
    x = np.arange(len(short_labels))

    bars = ax.bar(x, pull_sigs, color=colours, alpha=0.75, edgecolor="white",
                  width=0.55, zorder=3)
    ax.axhline(1.0, color="black", lw=1.8, ls="--", zorder=4,
               label=r"Target: $\sigma_{\rm pull} = 1$")
    ax.axhline(0.9, color="gray",  lw=0.8, ls=":",  zorder=3, alpha=0.6)
    ax.axhline(1.1, color="gray",  lw=0.8, ls=":",  zorder=3, alpha=0.6)

    for xi, (sig, mu, col) in enumerate(zip(pull_sigs, pull_mus, colours)):
        ax.text(xi, sig + 0.012, fr"$\sigma={sig:.3f}$"+"\n"+fr"$\mu={mu:+.3f}$",
                ha="center", va="bottom", fontsize=9, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=11)
    ax.set_ylim(0.8, 1.25)
    ax.set_ylabel(r"Pull width  $\sigma_{\rm pull}$", fontsize=12)
    ax.set_title(r"Metric 2 — Wilson coefficient pull width  "
                 r"(closed CT $\Rightarrow$ width $= 1$)", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out_b = str(out_dir / "summary_pull_width")
    plt.savefig(f"{out_b}.png", bbox_inches="tight", dpi=150)
    plt.savefig(f"{out_b}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_b}.png")


def plot_pvalue_dist(points, results, out_path):
    """
    Supervisor's data-space metric: distribution of p-values from L1 BSM
    pseudo-data, compared to the expected distribution given the known
    underlying law (non-central chi2 with known lambda).

    Shown as an ECDF on a log p-value axis so the small-p region (where
    the BSM signal lives) is spread out and the observed vs expected
    comparison is visible.

      Observed  : ECDF of p = chi2_sf(q^(r), n_UV) over L1 BSM replicas
      Expected  : ECDF of p drawn from ncx2(n_UV, lambda) -> chi2_sf
      SM null   : should be the diagonal F(p) = p  (uniform distribution)

    Closure passes when observed coincides with expected.
    """
    import math
    n = len(points)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 4.5))
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)

    col_obs = "#1f77b4"
    col_exp = "black"
    col_sm  = "#d62728"

    for idx, ((label, _rt, _model), R) in enumerate(zip(points, results)):
        ax  = axes[idx]
        nu  = R["n_uv"]
        lam = R["lambda_uv"]

        p_bsm = np.sort(R["p_uv_bsm"])          # observed, L1 BSM replicas
        p_sm  = np.sort(R["p_uv_sm"])            # SM null replicas

        # expected p-value distribution from ncx2(nu, lam) -> chi2_sf
        rng_e = np.random.default_rng(0)
        q_exp = ncx2_dist.rvs(df=nu, nc=lam, size=50_000, random_state=0)
        p_exp = np.sort(chi2_dist.sf(q_exp, df=nu))

        n_obs = len(p_bsm)
        n_exp = len(p_exp)
        n_sm  = len(p_sm)

        ecdf_obs = np.arange(1, n_obs + 1) / n_obs
        ecdf_exp = np.arange(1, n_exp + 1) / n_exp
        ecdf_sm  = np.arange(1, n_sm  + 1) / n_sm

        # subsample expected (50k points) for plotting
        stride = max(1, n_exp // 1000)

        ax.plot(p_exp[::stride], ecdf_exp[::stride],
                color=col_exp, lw=1.5, ls="--",
                label=fr"Expected (noncen $\chi^2$, $\lambda={lam:.1f}$)")
        ax.plot(p_bsm, ecdf_obs,
                color=col_obs, lw=2.0,
                label=r"Observed (L1 BSM replicas)")
        ax.plot(p_sm, ecdf_sm,
                color=col_sm, lw=1.5, alpha=0.8,
                label=r"SM null (should be diagonal)")

        # diagonal reference F(p) = p  (uniform distribution)
        p_ref = np.logspace(-8, 0, 500)
        ax.plot(p_ref, p_ref, color="gray", lw=0.8, ls=":",
                label=r"Uniform $F(p)=p$")

        # mark 3σ and 5σ thresholds
        for p_thr, lbl in [(2.7e-3, r"$3\sigma$"), (5.7e-7, r"$5\sigma$")]:
            ax.axvline(p_thr, color="gray", lw=1.0, ls="--", alpha=0.6)
            ax.text(p_thr * 1.4, 0.04, lbl, fontsize=8, color="gray",
                    rotation=90, va="bottom")

        ax.set_xscale("log")
        ax.set_xlim(1e-8, 1.0)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel(r"$p$-value  (SM null hypothesis)", fontsize=11)
        ax.set_ylabel(r"Cumulative fraction $F(p)$", fontsize=11)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")

        # annotate fraction below 3σ and 5σ
        frac3 = (p_bsm < 2.7e-3).mean() * 100
        frac5 = (p_bsm < 5.7e-7).mean() * 100
        ax.text(0.97, 0.35,
                fr"$p<3\sigma$: {frac3:.0f}% of L1 replicas" + "\n"
                fr"$p<5\sigma$: {frac5:.0f}% of L1 replicas",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    fig.suptitle(r"Data-space metric: ECDF of $p$-values from L1 BSM pseudo-data"
                 "\n"
                 r"Observed (blue) vs expected given known underlying law (black dashed)"
                 r" — closure passes when they coincide",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{out_path}.png", bbox_inches="tight", dpi=150)
    plt.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nrep", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else PIPELINE / "results" / "bsm_closure_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    points = make_points()

    print(f"\nBSM closure-test metrics  (N_rep={args.nrep}, seed={args.seed})")
    print(f"Output: {out_dir}\n")

    results = []
    for label, run_tag, model in points:
        print(f"  Computing: {label} ...", end="", flush=True)
        R = compute_point(run_tag, model, n_rep=args.nrep, seed=args.seed)
        results.append(R)
        nu  = R["n_uv"]
        lam = R["lambda_uv"]
        pow5 = (R["p_uv_bsm"] < 5.7e-7).mean() * 100
        pull_sig = float(R["pulls_l1"].std(ddof=1))
        print(f"  lambda_UV={lam:.2f}  pow5σ={pow5:.0f}%  pull_sigma={pull_sig:.3f}")

    print_summary(points, results)

    plot_metric1(points, results, str(out_dir / "metric1_pvalue"))
    plot_metric2(points, results, str(out_dir / "metric2_pulls"))
    plot_summary(points, results, out_dir)
    plot_pvalue_dist(points, results, str(out_dir / "metric1_pvalue_dist"))
    save_summary_table(points, results, str(out_dir / "closure_summary.txt"))

    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
