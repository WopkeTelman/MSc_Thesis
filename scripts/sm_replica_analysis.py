"""
SM replica analysis: exclusion vs discovery significance.

Generates two replica ensembles:
  1. SM null replicas:  delta^(r) ~ N(0, C)
     → empirical null distribution of q; validates chi2(k) approximation
  2. BSM signal replicas: delta^(r) = delta_BSM + N(0, C)
     → realistic discovery significance (median and 1-sigma band)

Compares both against the Asimov (noise-free) result from run_pipeline.

Usage:
    python scripts/sm_replica_analysis.py
    python scripts/sm_replica_analysis.py --nreplicas 2000 --nmodes 4
    python scripts/sm_replica_analysis.py --model zprime
"""
import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import chi2 as chi2_dist, norm as norm_dist

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "scripts"))

from run_pipeline import _build_K_Ci, _build_delta, _point_significance, _q_to_sigma, CFG
from models.wprime import WPrimeModel
from models.zprime import ZPrimeModel


# ── helpers ───────────────────────────────────────────────────────────────────

def invert_block_diagonal(Ci):
    """Invert Ci block by block via Cholesky for numerical stability."""
    return np.linalg.inv(Ci)


def draw_sm_replicas(C, n):
    """Draw n replicas from N(0, C) using Cholesky decomposition."""
    L = np.linalg.cholesky(C)
    z = np.random.randn(C.shape[0], n)   # (n_obs, n)
    return (L @ z).T                      # (n, n_obs)


def compute_q_ensemble(deltas, K, Ci, F_inv):
    """
    Compute q = g^T F^{-1} g for each row of deltas (shape n_rep x n_obs).
    F_inv = pinv(K^T Ci K) precomputed.
    """
    G = (K.T @ Ci @ deltas.T).T          # (n_rep, n_ops)
    return np.einsum('ri,ij,rj->r', G, F_inv, G)


def _to_sigma(q, k):
    return _q_to_sigma(float(q), k)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="wprime",
                   choices=["wprime", "zprime"], help="Model to analyse")
    p.add_argument("--nreplicas",   type=int, default=1000,
                   help="Number of replicas per ensemble")
    p.add_argument("--nmodes",      type=int, default=4,
                   help="Number of PCA modes to use for discovery significance")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--out",         default=None,
                   help="Output directory for plots (default: results/<model>/plots)")
    args = p.parse_args()

    np.random.seed(args.seed)

    # ── model + projections ───────────────────────────────────────────────────
    if args.model == "wprime":
        model    = WPrimeModel(gWH=0.12, gWLf11=0.04, gWLf22=0.04,
                               gWLf33=0.04, gWqf33=0.04, mWp=1.0)
        proj_dir = str(PIPELINE / "projections" / "wprime_gwh012_mwp010")
        tag      = "W' (gWH=0.12, mWp=1 TeV)"
    else:
        model    = ZPrimeModel(gZH=0.12, gZl=0.04, mZp=1.0)
        proj_dir = str(PIPELINE / "projections" / "zprime_gzh012_mzp010")
        tag      = "Z' (gZH=0.12, mZp=1 TeV)"

    out_dir = Path(args.out) if args.out else \
              PIPELINE / "results" / f"{'wprime_gwh012_mwp010' if args.model=='wprime' else 'zprime_gzh012_mzp010'}" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    ops      = model.OPERATORS
    K, Ci    = _build_K_Ci(ops)
    delta_bsm = _build_delta(proj_dir)

    print(f"Model : {tag}")
    print(f"Ops   : {len(ops)}  |  Obs: {K.shape[0]}  |  Replicas: {args.nreplicas}")

    # ── Asimov significance (no noise) ───────────────────────────────────────
    s_asimov = _point_significance(model, delta_bsm, K, Ci, ops)
    F        = s_asimov["F"]
    rank     = s_asimov["rank"]
    F_inv    = np.linalg.pinv(F)

    # PCA k=nmodes q value (Asimov)
    evals, evecs = s_asimov["evals"], s_asimov["evecs"]
    g_asimov     = s_asimov["g"]
    q_pca_asimov = sum(
        (evecs[:, i] @ g_asimov)**2 / max(evals[i], 1e-30)
        for i in range(min(args.nmodes, len(evals)))
    )

    print(f"\n{'─'*55}")
    print(f"  Asimov significance (no noise, single pseudo-data)")
    print(f"{'─'*55}")
    print(f"  Best 1-op  : q = {s_asimov['best_q']:.2f}  →  {s_asimov['sigma_best1']:.2f}σ  ({s_asimov['best_op']})")
    print(f"  PCA k={args.nmodes}    : q = {q_pca_asimov:.2f}  →  {_to_sigma(q_pca_asimov, args.nmodes):.2f}σ")
    print(f"  Full SMEFT : q = {s_asimov['q_full']:.2f}  →  {s_asimov['sigma_full']:.2f}σ  (rank={rank})")
    print(f"  UV truth   : q = {s_asimov['q_uv']:.2f}  →  {s_asimov['sigma_uv']:.2f}σ  (ndof={s_asimov['ndof_uv']})")

    # ── Covariance matrix ─────────────────────────────────────────────────────
    print(f"\nInverting covariance C ({K.shape[0]}×{K.shape[0]})...")
    C = invert_block_diagonal(Ci)

    # ── SM null replicas ──────────────────────────────────────────────────────
    print(f"Drawing {args.nreplicas} SM null replicas...")
    deltas_sm = draw_sm_replicas(C, args.nreplicas)
    q_sm_full  = compute_q_ensemble(deltas_sm, K, Ci, F_inv)

    # best-1op q for SM replicas (diagonal of F)
    F_diag = np.diag(F)
    G_sm   = (K.T @ Ci @ deltas_sm.T).T   # (n_rep, n_ops)
    q_sm_best1 = np.max(G_sm**2 / np.maximum(F_diag, 1e-30), axis=1)

    # PCA k=nmodes for SM replicas
    evec_mat = evecs[:, :args.nmodes]                          # (n_ops, k)
    proj_sm  = G_sm @ evec_mat                                 # (n_rep, k)
    q_sm_pca = np.sum(proj_sm**2 / np.maximum(evals[:args.nmodes], 1e-30), axis=1)

    print(f"  SM null:  <q_full> = {q_sm_full.mean():.2f}  (expected chi2({rank}) mean = {rank})")
    print(f"  SM null:  <q_pca{args.nmodes}> = {q_sm_pca.mean():.2f}  (expected chi2({args.nmodes}) mean = {args.nmodes})")

    # empirical p-values
    p_emp_full  = np.mean(q_sm_full  >= s_asimov["q_full"])
    p_emp_best1 = np.mean(q_sm_best1 >= s_asimov["best_q"])
    p_emp_pca   = np.mean(q_sm_pca   >= q_pca_asimov)

    print(f"\n{'─'*55}")
    print(f"  Empirical p-values  (fraction of SM replicas ≥ q_BSM)")
    print(f"{'─'*55}")
    print(f"  Best 1-op  : p_emp = {p_emp_best1:.4f}  →  {_to_sigma(s_asimov['best_q'],1):.2f}σ (analytic)")
    print(f"  PCA k={args.nmodes}    : p_emp = {p_emp_pca:.4f}  →  {_to_sigma(q_pca_asimov, args.nmodes):.2f}σ (analytic)")
    print(f"  Full SMEFT : p_emp = {p_emp_full:.4f}  →  {s_asimov['sigma_full']:.2f}σ (analytic)")

    # ── BSM signal replicas ───────────────────────────────────────────────────
    print(f"\nDrawing {args.nreplicas} BSM signal replicas (δ_BSM + noise)...")
    noise         = draw_sm_replicas(C, args.nreplicas)
    deltas_bsm_r  = delta_bsm[np.newaxis, :] + noise

    q_bsm_full  = compute_q_ensemble(deltas_bsm_r, K, Ci, F_inv)
    G_bsm       = (K.T @ Ci @ deltas_bsm_r.T).T
    q_bsm_best1 = np.max(G_bsm**2 / np.maximum(F_diag, 1e-30), axis=1)
    proj_bsm    = G_bsm @ evec_mat
    q_bsm_pca   = np.sum(proj_bsm**2 / np.maximum(evals[:args.nmodes], 1e-30), axis=1)

    sigma_bsm_full  = np.array([_to_sigma(q, rank)         for q in q_bsm_full])
    sigma_bsm_best1 = np.array([_to_sigma(q, 1)            for q in q_bsm_best1])
    sigma_bsm_pca   = np.array([_to_sigma(q, args.nmodes)  for q in q_bsm_pca])

    print(f"\n{'─'*55}")
    print(f"  Discovery significance over BSM replicas (median [16%, 84%])")
    print(f"{'─'*55}")
    for name, arr in [("Best 1-op", sigma_bsm_best1),
                      (f"PCA k={args.nmodes}",   sigma_bsm_pca),
                      ("Full SMEFT", sigma_bsm_full)]:
        lo, med, hi = np.percentile(arr, [16, 50, 84])
        print(f"  {name:<12}: {med:.2f}σ  [{lo:.2f}, {hi:.2f}]")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    configs = [
        ("Best 1-op",     q_sm_best1, q_bsm_best1, s_asimov["best_q"],  1),
        (f"PCA k={args.nmodes}", q_sm_pca,   q_bsm_pca,   q_pca_asimov,          args.nmodes),
        ("Full SMEFT",    q_sm_full,  q_bsm_full,  s_asimov["q_full"],   rank),
    ]

    for col, (name, q_null, q_bsm_r, q_asim, dof) in enumerate(configs):
        ax = fig.add_subplot(gs[col])

        # histogram range
        qmax = max(np.percentile(q_bsm_r, 99.5), q_asim * 1.05)
        bins = np.linspace(0, qmax, 60)

        ax.hist(q_null,  bins=bins, density=True, alpha=0.55,
                color="steelblue",  label="SM null")
        ax.hist(q_bsm_r, bins=bins, density=True, alpha=0.55,
                color="darkorange", label="BSM signal")

        # analytic chi2(dof) for null
        x = np.linspace(1e-3, qmax, 300)
        ax.plot(x, chi2_dist.pdf(x, dof), "b--", lw=1.5,
                label=fr"$\chi^2({dof})$")

        ax.axvline(q_asim, color="red", lw=2, ls="-", label=f"Asimov q={q_asim:.1f}")

        ax.set_xlabel(r"$q$", fontsize=13)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(name, fontsize=13)
        ax.legend(fontsize=9, framealpha=0.7)

    fig.suptitle(f"SM replica analysis — {tag}", fontsize=13, y=1.02)
    out_path = out_dir / "sm_replica_analysis.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.savefig(str(out_path).replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nPlot saved: {out_path}")
    plt.close()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  SUMMARY: {tag}")
    print(f"{'═'*65}")
    print(f"  {'Method':<14}  {'Asimov σ':>9}  {'Median σ':>9}  {'[16%,84%]':>14}  {'p_emp':>8}")
    print(f"  {'─'*14}  {'─'*9}  {'─'*9}  {'─'*14}  {'─'*8}")
    rows = [
        ("Best 1-op",    s_asimov["sigma_best1"], sigma_bsm_best1, p_emp_best1),
        (f"PCA k={args.nmodes}",  _to_sigma(q_pca_asimov, args.nmodes), sigma_bsm_pca, p_emp_pca),
        ("Full SMEFT",   s_asimov["sigma_full"],  sigma_bsm_full,  p_emp_full),
    ]
    for name, sig_asim, sig_arr, p_emp in rows:
        lo, med, hi = np.percentile(sig_arr, [16, 50, 84])
        print(f"  {name:<14}  {sig_asim:>9.2f}  {med:>9.2f}  [{lo:.2f}, {hi:.2f}]  {p_emp:>8.4f}")
    print(f"{'═'*65}")


if __name__ == "__main__":
    main()
