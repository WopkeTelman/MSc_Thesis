"""
W' vs Z' model discrimination via likelihood ratio.

Given data injected by model A (W' or Z'), computes how well model B (Z' or W')
can fit the same data. The discrimination statistic is:

    q_disc = q_A - q_B(A data)
           = χ²_SM(A data) × (1 - cos²θ)

where cos²θ is the squared cosine similarity between A and B signal directions
in the C^{-1}-weighted data space:

    cos²θ = (s_A^T C^{-1} s_B)² / (s_A^T C^{-1} s_A)(s_B^T C^{-1} s_B)

s_X = K c_X is the signal vector in data space for model X.

Key insight: cos²θ is independent of mass (both signals scale as 1/m²).
The discrimination significance therefore scales as q_A^{1/2} ∝ g²/m².

For W' (5 UV params) we use the full 3D Z' signal subspace (spanned by
the OpD, Opl, and Oll quadratic components) for the best-case Z' fit.

Usage:
    python scripts/model_discrimination.py
    python scripts/model_discrimination.py --gWH 0.12 --gZH 0.12
    python scripts/model_discrimination.py --mWp-ref 3.0 --mZp-ref 3.0
"""
import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2_dist, norm as norm_dist
from pathlib import Path

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))

from models.wprime_constrained import WPrimeConstrainedModel
from models.zprime_constrained import ZPrimeConstrainedModel


def q_to_sigma(q, k=1):
    if q <= 0:
        return 0.0
    p = chi2_dist.sf(float(q), df=k)
    return float(norm_dist.isf(p / 2)) if p > 0 else 99.0


def build_K_Ci_combined(ops_W, ops_Z):
    """
    Build K (n_data × n_combined_ops) and Ci (n_data × n_data)
    for the union of W' and Z' operators, by importing the pipeline's
    cached builder.
    """
    import importlib.util, types
    spec = importlib.util.spec_from_file_location(
        "run_pipeline",
        str(PIPELINE / "scripts" / "run_pipeline.py")
    )
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)

    combined_ops = list(dict.fromkeys(ops_W + ops_Z))   # union, preserve order
    K, Ci = rp._build_K_Ci(combined_ops)
    return K, Ci, combined_ops


def signal_vector(K, Ci, combined_ops, model_coeffs, model_ops):
    """Project model EFT coefficients onto the combined operator K matrix."""
    c_vec = np.array([model_coeffs.get(op, 0.0) for op in combined_ops])
    return K @ c_vec


def zprime_signal_subspace(K, Ci, combined_ops, mZp=1.0):
    """
    Build the Z' signal subspace basis in data space.

    Z' EFT coefficients are bilinear in UV couplings:
        c = A·gZH² + B·gZH·gZl + C·gZl²

    Returns G_Z' = (n_data × 3) matrix whose columns span the Z' signal space.
    Also returns Gram-Schmidt orthonormal basis in Ci metric.
    """
    def s_at(gZH, gZl):
        m = ZPrimeConstrainedModel(gZH=gZH, gZl=gZl, mZp=mZp)
        return signal_vector(K, Ci, combined_ops, m.eft_coefficients(), m.OPERATORS)

    # Extract the three quadratic components (at mZp=1, renormalized later)
    e = 0.1
    sA  = s_at(e, 0.0) / e**2               # OpD direction (∝ gZH²)
    sC  = s_at(0.0, e) / e**2               # Oll direction (∝ gZl²)
    sAC = (s_at(e, e) / e**2 - sA - sC)     # cross-term Opl (∝ gZH·gZl)

    G = np.column_stack([sA, sAC, sC])       # (n_data, 3)

    # Gram-Schmidt orthonormalization in Ci metric
    Q = np.zeros_like(G)
    for j in range(3):
        v = G[:, j].copy()
        for i in range(j):
            v -= (Q[:, i] @ Ci @ G[:, j]) * Q[:, i]
        norm = np.sqrt(v @ Ci @ v)
        if norm < 1e-12:
            continue
        Q[:, j] = v / norm

    return G, Q


def compute_discrimination(s_A, s_B_subspace_Q, Ci):
    """
    Given:
        s_A : signal vector of model A in data space
        s_B_subspace_Q : orthonormal basis (n_data × k) for model B signal subspace
        Ci : inverse covariance

    Returns:
        q_A    : total A signal (chi2_SM under A data)
        q_proj : A signal explained by B subspace
        q_disc : unexplained residual  q_A - q_proj
        cos2   : squared cosine (for 1-column B only, else None)
    """
    q_A = float(s_A @ Ci @ s_A)
    # project s_A onto B subspace: sum (e_k^T Ci s_A)^2
    q_proj = float(sum(float(q @ Ci @ s_A)**2 for q in s_B_subspace_Q.T))
    q_disc = q_A - q_proj
    return q_A, q_proj, q_disc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gstar",   type=float, default=0.18,
                   help="Universal coupling g* (applied to all W' and Z' couplings)")
    p.add_argument("--mWp-ref", type=float, default=10.0,
                   help="Reference W' mass (TeV) for computing cos²θ")
    p.add_argument("--mZp-ref", type=float, default=10.0,
                   help="Reference Z' mass (TeV) for Z' subspace")
    p.add_argument("--mass-max", type=float, default=15.0)
    p.add_argument("--nm",      type=int,   default=60,
                   help="Number of mass points in scan")
    p.add_argument("--out-dir", type=str,   default=None,
                   help="Output directory (default: results/model_discrimination)")
    args = p.parse_args()

    gWH  = args.gstar
    gWLf = args.gstar
    gZH  = args.gstar
    gZl  = args.gstar

    print("Building K, Ci for combined W'+Z' operator set...")
    mW_ref = getattr(args, "mWp_ref")
    mZ_ref = getattr(args, "mZp_ref")

    m_W_ref = WPrimeConstrainedModel(gWH=gWH, gWLf11=gWLf, gWLf22=gWLf,
                                     gWLf33=gWLf, gWqf33=gWLf, mWp=mW_ref)
    m_Z_ref = ZPrimeConstrainedModel(gZH=gZH, gZl=gZl, mZp=mZ_ref)

    K, Ci, combined_ops = build_K_Ci_combined(m_W_ref.OPERATORS, m_Z_ref.OPERATORS)
    print(f"  Combined operators ({len(combined_ops)}): {combined_ops}")
    print(f"  K shape: {K.shape}")

    # ── reference signal directions ───────────────────────────────────────────
    s_W_ref = signal_vector(K, Ci, combined_ops,
                            m_W_ref.eft_coefficients(), m_W_ref.OPERATORS)
    _, Q_Z  = zprime_signal_subspace(K, Ci, combined_ops, mZp=mZ_ref)

    # single Z' direction (fixed coupling ratio, for cos²θ display)
    s_Z_ref = signal_vector(K, Ci, combined_ops,
                            m_Z_ref.eft_coefficients(), m_Z_ref.OPERATORS)

    # cos²θ between W' and Z' (1D, fixed coupling ratio)
    qW = float(s_W_ref @ Ci @ s_W_ref)
    qZ = float(s_Z_ref @ Ci @ s_Z_ref)
    qWZ = float(s_W_ref @ Ci @ s_Z_ref)
    cos2_1d = qWZ**2 / (qW * qZ) if (qW > 0 and qZ > 0) else 0.0

    # cos²θ for the full Z' subspace (best-case Z' fit)
    _, _, q_disc_ref, = compute_discrimination(s_W_ref, Q_Z, Ci)
    cos2_3d = 1 - q_disc_ref / qW if qW > 0 else 0.0

    print(f"\nSignal overlap at reference masses "
          f"(mWp={mW_ref} TeV, mZp={mZ_ref} TeV):")
    print(f"  cos²θ  (1D, fixed coupling ratio): {cos2_1d:.4f}")
    print(f"  cos²θ  (3D Z' subspace):           {cos2_3d:.4f}")
    print(f"  q_W' (total signal):               {qW:.2f}")
    print(f"  q_disc (unexplained by best Z'):   {q_disc_ref:.2f}")
    print(f"  σ_disc = {q_to_sigma(q_disc_ref):.2f}σ")

    # ── mass scan: W' injected, fit with Z' (best 3D subspace) ───────────────
    masses   = np.linspace(0.5, args.mass_max, args.nm)
    sig_W    = []   # W' discovery significance (chi2_SM under W' data)
    sig_disc = []   # discrimination significance (W' vs Z')

    # Signal scales as 1/m^2, so q scales as 1/m^4
    # Use reference direction (mass-independent) × (m_ref/m)^4 scaling
    qW_ref_norm = float(s_W_ref @ Ci @ s_W_ref)

    for m in masses:
        scale = (mW_ref / m) ** 4
        qW_m  = qW_ref_norm * scale
        # q_disc scales identically (same cos²θ independent of mass)
        qd_m  = q_disc_ref * scale
        sig_W.append(q_to_sigma(qW_m))
        sig_disc.append(q_to_sigma(qd_m))

    sig_W    = np.array(sig_W)
    sig_disc = np.array(sig_disc)

    # ── print table ───────────────────────────────────────────────────────────
    header = f"\n{'mWp (TeV)':>10} {'sig_W':>10} {'sig_disc':>10}  note"
    print(header)
    print("-" * 46)
    key_masses = [1, 2, 3, 4, 5, 6]
    for m in key_masses:
        if m > args.mass_max:
            continue
        scale = (mW_ref / m) ** 4
        qW_m  = qW_ref_norm * scale
        qd_m  = q_disc_ref * scale
        sw    = q_to_sigma(qW_m)
        sd    = q_to_sigma(qd_m)
        note  = ("✓ discriminable" if sd >= 5 else
                 ("evidence" if sd >= 3 else "indistinguishable"))
        print(f"{m:>10.1f} {sw:>10.2f} {sd:>10.2f}  {note}")

    # ── build proper 5D W' UV subspace ───────────────────────────────────────
    import dataclasses
    n_UV_W = len(m_W_ref.uv_param_names())
    eps_j  = 1e-4
    J_W_full = np.zeros((len(combined_ops), n_UV_W))
    for j, pname in enumerate(m_W_ref.uv_param_names()):
        kw  = dataclasses.asdict(m_W_ref)
        m_p = WPrimeConstrainedModel(**{**kw, pname: kw[pname] + eps_j})
        m_m = WPrimeConstrainedModel(**{**kw, pname: kw[pname] - eps_j})
        c_p = np.array([m_p.eft_coefficients().get(op, 0.0) for op in combined_ops])
        c_m = np.array([m_m.eft_coefficients().get(op, 0.0) for op in combined_ops])
        J_W_full[:, j] = (c_p - c_m) / (2 * eps_j)

    G_W5     = K @ J_W_full
    F_W5     = G_W5.T @ Ci @ G_W5
    F_W5_inv = np.linalg.inv(F_W5)

    n_UV_Z   = 3  # Q_Z has 3 orthonormal columns
    F_Z3     = Q_Z.T @ Ci @ Q_Z
    F_Z3_inv = np.linalg.inv(F_Z3 + 1e-12 * np.eye(n_UV_Z))

    # cross-leakage: how much of each model's signal falls in the other's subspace
    s_W    = signal_vector(K, Ci, combined_ops,
                           m_W_ref.eft_coefficients(), m_W_ref.OPERATORS)
    s_Z    = signal_vector(K, Ci, combined_ops,
                           m_Z_ref.eft_coefficients(), m_Z_ref.OPERATORS)
    lam_W_ref = float(s_W @ Ci @ s_W)
    lam_Z_ref = float(s_Z @ Ci @ s_Z)

    h_Z_on_W  = Q_Z.T @ Ci @ s_W
    lam_ZonW  = float(h_Z_on_W @ F_Z3_inv @ h_Z_on_W)   # Z' absorbs this of W'
    h_W_on_Z  = G_W5.T @ Ci @ s_Z
    lam_WonZ  = float(h_W_on_Z @ F_W5_inv @ h_W_on_Z)   # W' absorbs this of Z'

    cos2_ZonW = lam_ZonW / lam_W_ref   # fraction W' signal absorbed by Z'
    cos2_WonZ = lam_WonZ / lam_Z_ref   # fraction Z' signal absorbed by W'

    print(f"\nCross-leakage (5D W' subspace, 3D Z' subspace):")
    print(f"  Z' absorbs {cos2_ZonW*100:.1f}% of W' signal  (lam_ZonW={lam_ZonW:.2f})")
    print(f"  W' absorbs {cos2_WonZ*100:.1f}% of Z' signal  (lam_WonZ={lam_WonZ:.2f})")

    # ── MC replicas at reference point ────────────────────────────────────────
    print("\nGenerating L1 replicas for scatter plot...")
    n_rep  = 4000
    rng    = np.random.default_rng(42)
    n_data = K.shape[0]
    C_full = np.linalg.inv(Ci)
    L_chol = np.linalg.cholesky(C_full)

    def q_fit(data_mat, G_sub, F_inv):
        h = G_sub.T @ Ci @ data_mat
        return np.einsum('ir,ij,jr->r', h, F_inv, h)

    scatter = {}
    for label, sig in [("wp", s_W), ("zp", s_Z)]:
        noise = L_chol @ rng.standard_normal((n_data, n_rep))
        d = sig[:, None] + noise
        scatter[f"qW_{label}"] = q_fit(d, G_W5, F_W5_inv)
        scatter[f"qZ_{label}"] = q_fit(d, Q_Z,  F_Z3_inv)

    def disc_score(qW, qZ):
        pW = chi2_dist.sf(np.maximum(qW, 0), df=n_UV_W)
        pZ = chi2_dist.sf(np.maximum(qZ, 0), df=n_UV_Z)
        return np.log(pZ + 1e-300) - np.log(pW + 1e-300)

    acc_wp = (disc_score(scatter["qW_wp"], scatter["qZ_wp"]) > 0).mean()
    acc_zp = (disc_score(scatter["qW_zp"], scatter["qZ_zp"]) < 0).mean()

    # ── panel 3: P(correct ID) vs sigma — correct ncx2 with leakage ──────────
    print("Computing P(correct ID) curves...")
    scales = np.linspace(0.0, 3.0, 120)
    n_mc   = 40000
    p_id_W, p_id_Z = [], []

    for f in scales:
        lam_W_s = f**2 * lam_W_ref
        lam_Z_s = f**2 * lam_Z_ref
        # W' truth: W' fits well, Z' absorbs only leakage fraction
        qW_s = np.random.noncentral_chisquare(n_UV_W, max(lam_W_s, 1e-9), n_mc)
        qZ_s = np.random.noncentral_chisquare(n_UV_Z, max(cos2_ZonW * lam_W_s, 1e-9), n_mc)
        p_id_W.append((disc_score(qW_s, qZ_s) > 0).mean())
        # Z' truth: Z' fits well, W' absorbs only leakage fraction
        qZ_s2 = np.random.noncentral_chisquare(n_UV_Z, max(lam_Z_s, 1e-9), n_mc)
        qW_s2 = np.random.noncentral_chisquare(n_UV_W, max(cos2_WonZ * lam_Z_s, 1e-9), n_mc)
        p_id_Z.append((disc_score(qW_s2, qZ_s2) < 0).mean())

    sigma_W_axis = np.array([q_to_sigma(f**2 * lam_W_ref) for f in scales])
    sigma_Z_axis = np.array([q_to_sigma(f**2 * lam_Z_ref) for f in scales])

    # ── plot ──────────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir) if args.out_dir else PIPELINE / "results" / "model_discrimination"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"model_discrimination_gstar{int(gWH*100):03d}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── panel 0: 2D scatter — hexbin for density ─────────────────────────────
    ax = axes[0]
    eps = 0.05

    # use hexbin for density instead of individual dots
    xW = np.log10(scatter["qW_wp"] + eps);  yW = np.log10(scatter["qZ_wp"] + eps)
    xZ = np.log10(scatter["qW_zp"] + eps);  yZ = np.log10(scatter["qZ_zp"] + eps)

    x_all = np.concatenate([xW, xZ])
    y_all = np.concatenate([yW, yZ])
    xlo, xhi = x_all.min() - 0.1, x_all.max() + 0.1
    ylo, yhi = y_all.min() - 0.1, y_all.max() + 0.1

    ax.hexbin(xW, yW, gridsize=30, cmap="Blues",  mincnt=1, alpha=0.8,
              extent=[xlo, xhi, ylo, yhi], zorder=2)
    ax.hexbin(xZ, yZ, gridsize=30, cmap="Reds",   mincnt=1, alpha=0.8,
              extent=[xlo, xhi, ylo, yhi], zorder=3)

    # decision boundary: equal p-value
    q_grid  = np.logspace(np.log10(max(10**xlo, 0.01)), xhi, 400)
    p_grid  = chi2_dist.sf(q_grid, df=n_UV_W)
    q_bound = chi2_dist.isf(np.clip(p_grid, 1e-300, 1), df=n_UV_Z)
    ok      = (q_bound > 0) & np.isfinite(q_bound) & (np.log10(q_bound) >= ylo)
    ax.plot(np.log10(q_grid[ok]), np.log10(q_bound[ok]),
            "k-", lw=2, label=r"$p_{W'}=p_{Z'}$  (decision boundary)", zorder=5)

    # label regions
    ax.text(0.75, 0.12, "W' wins", transform=ax.transAxes,
            color="#1f77b4", fontsize=11, fontweight="bold", ha="center")
    ax.text(0.20, 0.88, "Z' wins", transform=ax.transAxes,
            color="#d62728", fontsize=11, fontweight="bold", ha="center")

    # colour patches for legend
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#1f77b4", label=fr"W' data  ({acc_wp*100:.0f}% correct)"),
        Patch(facecolor="#d62728", label=fr"Z' data  ({acc_zp*100:.0f}% correct)"),
        plt.Line2D([0],[0], color="black", lw=2, label=r"$p_{W'}=p_{Z'}$"),
    ]
    ax.legend(handles=legend_els, fontsize=9, loc="center right")
    ax.set_xlabel(r"$\log_{10}\,q_{W'}$  (W' fit quality)", fontsize=11)
    ax.set_ylabel(r"$\log_{10}\,q_{Z'}$  (Z' fit quality)", fontsize=11)
    ax.set_title(fr"L1 replica scatter  ($g^*={gWH},\ m=10$ TeV)",
                 fontsize=11)
    ax.set_xlim(xlo, xhi);  ax.set_ylim(ylo, yhi)
    ax.grid(True, alpha=0.3)

    # ── panel 1: discovery + discrimination significance vs mass ──────────────
    ax = axes[1]
    ax.plot(masses, sig_W,    color="#1f77b4", lw=2.5,
            label=r"$\sigma_{W'}$  (W' discovery)")
    ax.plot(masses, sig_disc, color="darkorange", lw=2, ls="--",
            label=fr"$\sigma_\mathrm{{disc}}$  (W' $\neq$ Z', "
                  fr"$\cos^2\theta={cos2_3d:.2f}$)")
    ax.fill_between(masses, sig_disc, sig_W, alpha=0.15, color="darkorange",
                    label="Discovered but not yet discriminated")
    ax.axhline(5, color="black", lw=1.2, ls=":",  label=r"$5\sigma$")
    ax.axhline(3, color="black", lw=1,   ls="--", alpha=0.5, label=r"$3\sigma$")

    m_disc5 = np.interp(5.0, sig_disc[::-1], masses[::-1]) if sig_disc[0] > 5 else None
    m_disc3 = np.interp(3.0, sig_disc[::-1], masses[::-1]) if sig_disc[0] > 3 else None
    if m_disc5:
        ax.axvline(m_disc5, color="darkorange", lw=1.2, ls=":", alpha=0.8)
        ax.text(m_disc5 + 0.2, 5.4, fr"$m={m_disc5:.1f}$ TeV",
                color="darkorange", fontsize=9)
    if m_disc3:
        ax.axvline(m_disc3, color="darkorange", lw=1.2, ls=":", alpha=0.5)

    ax.set_xlabel(r"$m_{W'}$ (TeV)", fontsize=12)
    ax.set_ylabel(r"Significance ($\sigma$)", fontsize=12)
    ax.set_title(fr"Discovery vs discrimination reach  ($g^*={gWH}$)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(masses[0], masses[-1])
    ax.set_ylim(0, min(sig_W.max() * 1.05, 30))
    ax.grid(True, alpha=0.3)

    # ── panel 2: P(correct ID) vs discovery significance ─────────────────────
    ax = axes[2]
    p_id_W = np.array(p_id_W)
    p_id_Z = np.array(p_id_Z)

    ax.plot(sigma_W_axis, p_id_W, color="#1f77b4", lw=2.5,
            label=r"W' data: $P(\hat{H}=W')$")
    ax.plot(sigma_Z_axis, p_id_Z, color="#d62728", lw=2.5,
            label=r"Z' data: $P(\hat{H}=Z')$")
    ax.axhline(0.95, color="black", lw=1.2, ls="--", label="95% threshold")
    ax.axhline(0.50, color="grey",  lw=0.8, ls=":")

    for arr, axis, col in [(p_id_W, sigma_W_axis, "#1f77b4"),
                           (p_id_Z, sigma_Z_axis, "#d62728")]:
        idx = np.where(arr >= 0.95)[0]
        if len(idx):
            s95 = axis[idx[0]]
            ax.axvline(s95, color=col, lw=1.2, ls=":", alpha=0.8)
            ax.text(s95 + 0.15, 0.40, fr"$\sigma={s95:.1f}$",
                    color=col, fontsize=9)

    ax.set_xlabel(r"Signal significance $\sigma$  (of injected model)", fontsize=12)
    ax.set_ylabel(r"$P(\mathrm{correct\ model\ ID})$", fontsize=12)
    ax.set_title("When can you tell W' from Z'?\n"
                 fr"(leakage: Z' absorbs {cos2_ZonW*100:.0f}% of W', "
                 fr"W' absorbs {cos2_WonZ*100:.0f}% of Z')",
                 fontsize=10)
    ax.set_xlim(0, min(sigma_W_axis.max(), 12))
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        fr"W' vs Z' model discrimination at FCC-ee  "
        fr"($g^*={gWH}$,  $m_{{W'}}=m_{{Z'}}={mW_ref}$ TeV  reference)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(f"{out_path}.png", bbox_inches="tight", dpi=150)
    plt.savefig(f"{out_path}.pdf", bbox_inches="tight")
    print(f"\nPlot saved: {out_path}.png")
    plt.close()


if __name__ == "__main__":
    main()
