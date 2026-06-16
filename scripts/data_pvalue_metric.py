"""
Data-space p-value metric for the BSM closure test.

For each L1 replica r (BSM signal + Gaussian noise), computes the test
statistic q^(r) = chi2_SM^(r) - chi2_min_fit^(r) and converts it to a
p-value under the SM null hypothesis.

Two fits are compared:
  - Full SMEFT (n_ops free coefficients, ndof = n_ops)
  - UV coupling (n_UV free parameters,  ndof = n_UV)

Under H0 (SM is true):  q ~ chi2(ndof)   → p-values uniform on [0,1]
Under H1 (BSM is true): q ~ chi2(ndof, λ) → p-values concentrated near 0

"Expected" p-value distribution is the analytic prediction given the known
non-centrality λ = q_UV_truth. "Observed" is the empirical distribution
from N_rep L1 replicas. Agreement validates the closure test.

Usage:
    python scripts/data_pvalue_metric.py
    python scripts/data_pvalue_metric.py --model wprime_constrained --gWH 0.12 --mWp 3.0 --nrep 2000
    python scripts/data_pvalue_metric.py --mWp 4.0 --nrep 2000
"""
import sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2_dist, ncx2 as ncx2_dist, norm as norm_dist
from pathlib import Path

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))

from models.wprime_constrained import WPrimeConstrainedModel
from models.zprime_constrained  import ZPrimeConstrainedModel


def q_to_sigma(q, k):
    if q <= 0:
        return 0.0
    p = chi2_dist.sf(float(q), df=k)
    return float(norm_dist.isf(p / 2)) if p > 0 else 99.0


def uv_jacobian(model, ops, eps=1e-4):
    """
    Numerical Jacobian dc/dθ  (n_ops × n_UV_params) at the truth point.
    Columns correspond to model.uv_param_names().
    """
    param_names = model.uv_param_names()
    c0 = np.array([model.eft_coefficients().get(op, 0.0) for op in ops])
    J  = np.zeros((len(ops), len(param_names)))

    for j, pname in enumerate(param_names):
        m_plus  = _perturb_model(model, pname, +eps)
        m_minus = _perturb_model(model, pname, -eps)
        c_plus  = np.array([m_plus.eft_coefficients().get(op, 0.0) for op in ops])
        c_minus = np.array([m_minus.eft_coefficients().get(op, 0.0) for op in ops])
        J[:, j] = (c_plus - c_minus) / (2 * eps)

    return J


def _perturb_model(model, param_name, delta):
    """Return a copy of model with one UV parameter shifted by delta."""
    import dataclasses
    kwargs = dataclasses.asdict(model)
    kwargs[param_name] = kwargs[param_name] + delta
    return type(model)(**kwargs)


def run_replicas(K, Ci, delta, c_truth, J_uv, n_rep, seed=42):
    """
    Generate n_rep L1 BSM replicas and compute q statistics for each.

    Returns dict with arrays of shape (n_rep,):
        q_smeft_bsm  : SMEFT fit q under BSM replicas
        q_uv_bsm     : UV coupling fit q under BSM replicas
        q_smeft_sm   : SMEFT fit q under SM null replicas (no signal)
        q_uv_sm      : UV coupling fit q under SM null replicas
    """
    rng = np.random.default_rng(seed)
    n_data, n_ops = K.shape
    n_uv = J_uv.shape[1]

    F     = K.T @ Ci @ K
    F_inv = np.linalg.inv(F)

    # UV Fisher matrix in UV-parameter space
    G     = K @ J_uv              # (n_data × n_uv) signal directions
    F_uv  = G.T @ Ci @ G         # (n_uv × n_uv)
    F_uv_inv = np.linalg.inv(F_uv)

    # Covariance for noise generation
    C = np.linalg.inv(Ci)
    L = np.linalg.cholesky(C)

    # Precompute truth chi2 components
    s_truth = K @ c_truth                    # signal in data space

    results = {}
    for label, include_signal in [("bsm", True), ("sm", False)]:
        noise = L @ rng.standard_normal((n_data, n_rep))  # (n_data, n_rep)

        if include_signal:
            data = s_truth[:, np.newaxis] + noise          # (n_data, n_rep)
        else:
            data = noise

        chi2_sm = np.einsum('ir,ij,jr->r', data, Ci, data)  # (n_rep,)

        # SMEFT fit: q = g^T F^{-1} g, g = K^T Ci data
        g_smeft = K.T @ Ci @ data                            # (n_ops, n_rep)
        q_smeft = np.einsum('ir,ij,jr->r', g_smeft, F_inv, g_smeft)

        # UV coupling fit: q = h^T F_uv^{-1} h, h = G^T Ci data
        h_uv = G.T @ Ci @ data                              # (n_uv, n_rep)
        q_uv = np.einsum('ir,ij,jr->r', h_uv, F_uv_inv, h_uv)

        results[f"q_smeft_{label}"] = q_smeft
        results[f"q_uv_{label}"]    = q_uv
        results[f"chi2_sm_{label}"] = chi2_sm

    return results


def pvalues(q_arr, ndof):
    """Convert q statistics to p-values under chi2(ndof) null."""
    return chi2_dist.sf(q_arr, df=ndof)


def plot_results(results, n_ops, n_uv, lambda_smeft, lambda_uv, out_path, tag):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    colors = {"bsm": "C0", "sm": "C3"}
    labels = {"bsm": "BSM replicas (L1)", "sm": "SM null replicas (L1)"}

    for row, (method, ndof, lam) in enumerate([
            ("smeft", n_ops, lambda_smeft),
            ("uv",    n_uv,  lambda_uv)]):

        method_label = "Full SMEFT" if method == "smeft" else "UV coupling"
        ndof_label   = f"ndof={ndof}"

        q_bsm = results[f"q_{method}_bsm"]
        q_sm  = results[f"q_{method}_sm"]
        p_bsm = pvalues(q_bsm, ndof)
        p_sm  = pvalues(q_sm,  ndof)

        # ── col 0: q-statistic distribution ──────────────────────────────────
        ax = axes[row, 0]
        q_max = max(np.percentile(q_bsm, 99), chi2_dist.ppf(0.999, ndof) * 3)
        bins  = np.linspace(0, q_max, 50)
        ax.hist(q_bsm, bins=bins, density=True, alpha=0.6,
                color=colors["bsm"], label=labels["bsm"])
        ax.hist(q_sm, bins=bins, density=True, alpha=0.6,
                color=colors["sm"], label=labels["sm"])
        x = np.linspace(0, q_max, 500)
        ax.plot(x, chi2_dist.pdf(x, df=ndof), "k-", lw=2,
                label=fr"$\chi^2({ndof})$ [SM null]")
        ax.plot(x, ncx2_dist.pdf(x, df=ndof, nc=lam), "k--", lw=2,
                label=fr"$\chi^2({ndof},\lambda={lam:.1f})$ [BSM expected]")
        ax.axvline(np.median(q_bsm), color=colors["bsm"], lw=1.5, ls=":")
        ax.set_xlabel(r"$q$", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{method_label} — $q$ distribution  ({ndof_label})", fontsize=11)
        ax.legend(fontsize=8)

        # ── col 1: p-value distribution ───────────────────────────────────────
        ax = axes[row, 1]
        pbins = np.linspace(0, 1, 21)
        ax.hist(p_bsm, bins=pbins, density=True, alpha=0.6,
                color=colors["bsm"], label=labels["bsm"])
        ax.hist(p_sm, bins=pbins, density=True, alpha=0.6,
                color=colors["sm"], label=labels["sm"])
        ax.axhline(1.0, color="black", lw=1.5, ls="--", label="Uniform [SM expected]")

        # expected p-value density under H1 (analytic)
        p_grid = np.linspace(0.001, 0.999, 200)
        q_grid = chi2_dist.ppf(1 - p_grid, df=ndof)
        # density dp/dq × dq/dp, but simpler: use kde of expected non-central chi2
        # approximated by histogram of draws
        q_h1_draws = ncx2_dist.rvs(df=ndof, nc=lam, size=50000,
                                    random_state=0)
        p_h1_draws = chi2_dist.sf(q_h1_draws, df=ndof)
        ax.hist(p_h1_draws, bins=pbins, density=True, histtype="step",
                color="black", lw=2, label=fr"Expected BSM ($\lambda={lam:.1f}$)")
        ax.set_xlabel(r"$p$-value", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{method_label} — $p$-value distribution", fontsize=11)
        ax.legend(fontsize=8)

        # ── col 2: power curve P(p < alpha | BSM true) ───────────────────────
        ax = axes[row, 2]
        alphas = np.linspace(0, 1, 200)
        power_obs      = np.array([(p_bsm < a).mean() for a in alphas])
        power_expected = np.array([(p_h1_draws < a).mean() for a in alphas])
        ax.plot(alphas, power_obs,      color=colors["bsm"], lw=2,
                label="Observed power (L1 replicas)")
        ax.plot(alphas, power_expected, color="black", lw=2, ls="--",
                label=fr"Expected power ($\lambda={lam:.1f}$)")
        ax.plot(alphas, alphas, color="gray", lw=1, ls=":",
                label="No-power baseline")
        ax.axvline(chi2_dist.sf(chi2_dist.ppf(1 - 2.87e-7, ndof), ndof),
                   color="gray", lw=1, ls="--", alpha=0.5)
        ax.set_xlabel(r"$\alpha$ (significance threshold)", fontsize=11)
        ax.set_ylabel(r"Power = $P(p < \alpha \mid \mathrm{BSM})$", fontsize=11)
        ax.set_title(f"{method_label} — power curve", fontsize=11)
        ax.legend(fontsize=8)

        # print summary
        print(f"\n{method_label} ({ndof_label}, λ={lam:.2f}):")
        print(f"  BSM replicas — median q = {np.median(q_bsm):.2f}, "
              f"median σ = {q_to_sigma(np.median(q_bsm), ndof):.2f}")
        print(f"  SM  replicas — median q = {np.median(q_sm):.2f}  "
              f"(expect {ndof * (1 - 2/(9*ndof)):.1f})")
        print(f"  p < 5σ threshold: {(p_bsm < chi2_dist.sf(chi2_dist.ppf(1-2.87e-7, ndof), ndof)).mean()*100:.1f}% of BSM replicas")

    fig.suptitle(f"Data-space p-value metric — {tag}", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{out_path}.png", bbox_inches="tight", dpi=150)
    plt.savefig(f"{out_path}.pdf", bbox_inches="tight")
    print(f"\nPlot saved: {out_path}.png")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="wprime_constrained",
                   choices=["wprime_constrained", "zprime_constrained"])
    p.add_argument("--gWH",  type=float, default=0.12)
    p.add_argument("--mWp",  type=float, default=3.0)
    p.add_argument("--gZH",  type=float, default=0.12)
    p.add_argument("--mZp",  type=float, default=7.0)
    p.add_argument("--nrep", type=int,   default=2000)
    p.add_argument("--seed", type=int,   default=42)
    args = p.parse_args()

    if args.model == "wprime_constrained":
        gWLf  = args.gWH / 3
        model = WPrimeConstrainedModel(gWH=args.gWH, gWLf11=gWLf, gWLf22=gWLf,
                                       gWLf33=gWLf, gWqf33=gWLf, mWp=args.mWp)
        tag     = f"W' (gWH={args.gWH}, mWp={args.mWp} TeV)"
        run_tag = f"wprime_constrained_gwh{int(args.gWH*100):03d}_mwp{int(args.mWp*10):03d}"
    else:
        model   = ZPrimeConstrainedModel(gZH=args.gZH, gZl=args.gZH/3, mZp=args.mZp)
        tag     = f"Z' (gZH={args.gZH}, mZp={args.mZp} TeV)"
        run_tag = f"zprime_constrained_gzh{int(args.gZH*100):03d}_mzp{int(args.mZp*10):03d}"

    pca_dir = PIPELINE / "results" / run_tag / "pca"
    out_dir = PIPELINE / "results" / run_tag / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model   : {model}")
    print(f"Run tag : {run_tag}")
    print(f"N_rep   : {args.nrep}")

    K     = np.load(pca_dir / "K_fit.npy")
    Ci    = np.load(pca_dir / "C_inv.npy")
    delta = np.load(pca_dir / "data_delta.npy")   # BSM - SM signal

    ops     = model.OPERATORS
    n_ops   = len(ops)
    truth   = model.eft_coefficients()
    c_truth = np.array([truth.get(op, 0.0) for op in ops])
    J_uv    = uv_jacobian(model, ops)
    n_uv    = J_uv.shape[1]

    # non-centrality parameters (expected signal strength)
    F      = K.T @ Ci @ K
    G      = K @ J_uv
    F_uv   = G.T @ Ci @ G
    lambda_smeft = float(c_truth @ F @ c_truth)
    lambda_uv    = float((J_uv @ np.linalg.inv(F_uv) @ J_uv.T @ F @ c_truth) @
                         F @ c_truth)

    # simpler: lambda_uv = q_UV = delta^T Ci G F_uv^{-1} G^T Ci delta at truth
    h_truth      = G.T @ Ci @ delta
    lambda_uv    = float(h_truth @ np.linalg.inv(F_uv) @ h_truth)
    lambda_smeft = float(delta @ Ci @ delta)        # chi2_SM = total signal

    print(f"Non-centrality λ_SMEFT = {lambda_smeft:.2f}  "
          f"(σ = {q_to_sigma(lambda_smeft, n_ops):.2f}  at ndof={n_ops})")
    print(f"Non-centrality λ_UV    = {lambda_uv:.2f}  "
          f"(σ = {q_to_sigma(lambda_uv, n_uv):.2f}  at ndof={n_uv})")

    results = run_replicas(K, Ci, delta, c_truth, J_uv,
                           n_rep=args.nrep, seed=args.seed)

    plot_results(results, n_ops, n_uv, lambda_smeft, lambda_uv,
                 str(out_dir / "data_pvalue_metric"), tag)


if __name__ == "__main__":
    main()
