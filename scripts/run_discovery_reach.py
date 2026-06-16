#!/usr/bin/env python3
"""
run_discovery_reach.py

True discovery reach pipeline for the W' BSM analysis at FCC-ee.

Uses analytic toy MC to establish the null distribution of the LRT statistic
q under H0=SM, bypassing the Wilks chi2 approximation.

No smefit calls are made. All computation is pure matrix algebra.

Usage:
    python run_discovery_reach.py [--n-toys 10000] [--seed 42] [--no-theory-covmat]

Output: results/discovery_reach/
    plots/   -- discovery region plots, null distribution histograms
    tables/  -- discovery tables (.txt)

Significance convention (discovery):
    p = P(q > q_BSM | H0=SM)   [one-sided]
    sigma = Phi^{-1}(1 - p)

Compare with run_pipeline.py which uses two-sided (exclusion convention).
"""

import os, sys, argparse, json, yaml
import numpy as np
from pathlib import Path
from scipy.stats import chi2 as chi2_dist, norm as norm_dist
from scipy.linalg import block_diag

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))

DB      = "/data/theorie/wtelman/smefit_database"
SM_DATA = f"{DB}/commondata_projections_L0"
THEORY  = f"{DB}/theory"

DS_NAMES = [
    "FCCee_ww_161GeV", "FCCee_ww_240GeV", "FCCee_ww_365GeV",
    "FCCee_Rb_240GeV", "FCCee_Rc_240GeV", "FCCee_Rmu_240GeV", "FCCee_Rtau_240GeV",
    "FCCee_bb_Afb_240GeV", "FCCee_cc_Afb_240GeV", "FCCee_mumu_Afb_240GeV",
    "FCCee_sigmaHad_240GeV", "FCCee_tautau_Afb_240GeV",
    "FCCee_Rb_365GeV", "FCCee_Rc_365GeV", "FCCee_Rmu_365GeV", "FCCee_Rtau_365GeV",
    "FCCee_bb_Afb_365GeV", "FCCee_cc_Afb_365GeV", "FCCee_mumu_Afb_365GeV",
    "FCCee_sigmaHad_365GeV", "FCCee_tautau_Afb_365GeV",
    "FCCee_Wwidth", "FCCee_Zdata",
    "FCCee_zh_240GeV", "FCCee_zh_WW_240GeV", "FCCee_zh_ZZ_240GeV",
    "FCCee_zh_aZ_240GeV", "FCCee_zh_aa_240GeV", "FCCee_zh_tautau_240GeV",
    "FCCee_zh_365GeV", "FCCee_zh_WW_365GeV", "FCCee_zh_ZZ_365GeV",
    "FCCee_zh_aa_365GeV", "FCCee_zh_tautau_365GeV",
]

# Exact ordering from WPrimeModel.OPERATORS
OPERATORS = [
    "O3pQ3", "O3pl1", "O3pl2", "O3pl3", "OQQ1", "OQQ8", "OQl13", "OQl1M",
    "OQl33", "OQl3M", "Obp", "Oll1111", "Oll1122", "Oll1133", "Oll1221",
    "Oll1331", "Oll2222", "Oll2233", "Oll2332", "Oll3333", "Op", "OpBox",
    "OpQM", "Otap", "Otp"
]

# Scan grid
GWH_GRID = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.33, 0.36]
MWP_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
FERMION_RATIO = 1.0 / 3.0

# Representative points for null distribution diagnostic plots
DIAG_POINTS = [(0.12, 1.0), (0.12, 2.0), (0.24, 1.0), (0.06, 0.5)]


# ── EFT coefficients (analytic, mirrors WPrimeModel.eft_coefficients) ──────────

def eft_coefficients(gWH, gWLf11, gWLf22, gWLf33, gWqf33, mWp):
    """Analytic W' matching relations. Returns dict {op: value}."""
    g, lf11, lf22, lf33, qf33, m2 = gWH, gWLf11, gWLf22, gWLf33, gWqf33, mWp**2
    return {
        "O3pQ3":   (-0.25 * g * qf33) / m2,
        "O3pl1":   (-0.25 * g * lf11) / m2,
        "O3pl2":   (-0.25 * g * lf22) / m2,
        "O3pl3":   (-0.25 * g * lf33) / m2,
        "OQQ1":    (0.08333333333333333 * qf33**2) / m2,
        "OQQ8":    (-1.0 * qf33**2) / m2,
        "OQl13":   (-1.0 * lf11 * qf33) / m2,
        "OQl1M":   (1.0  * lf11 * qf33) / m2,
        "OQl33":   (-1.0 * lf33 * qf33) / m2,
        "OQl3M":   (1.0  * lf33 * qf33) / m2,
        "Obp":     (-0.006002165432052166 * g**2) / m2,
        "Oll1111": (-0.125 * lf11**2) / m2,
        "Oll1122": (0.125  * lf11 * lf22) / m2,
        "Oll1133": (0.125  * lf11 * lf33) / m2,
        "Oll1221": (-0.25  * lf11 * lf22) / m2,
        "Oll1331": (0.25   * lf11 * lf33) / m2,
        "Oll2222": (0.125  * lf22**2) / m2,
        "Oll2233": (0.125  * lf22 * lf33) / m2,
        "Oll2332": (0.25   * lf22 * lf33) / m2,
        "Oll3333": (-0.125 * lf33**2) / m2,
        "Op":      (-0.12938347743146458 * g**2) / m2,
        "OpBox":   (-0.375 * g**2) / m2,
        "OpQM":    (0.25   * g * qf33) / m2,
        "Otap":    (-0.0025559460452279554 * g**2) / m2,
        "Otp":     (-0.2480703588615627   * g**2) / m2,
    }


def eft_coefficients_vec(gWH, gWLf11, gWLf22, gWLf33, gWqf33, mWp):
    """Return c as numpy array ordered by OPERATORS."""
    c_dict = eft_coefficients(gWH, gWLf11, gWLf22, gWLf33, gWqf33, mWp)
    return np.array([c_dict.get(op, 0.0) for op in OPERATORS])


# ── Analytic Jacobian dc/d(UV params) ──────────────────────────────────────────

def wprime_jacobian(gWH, gWLf11, gWLf22, gWLf33, gWqf33, mWp):
    """
    Jacobian J (25 x 5) of eft_coefficients w.r.t. UV params.

    theta = [gWH(0), gWLf11(1), gWLf22(2), gWLf33(3), gWqf33(4)]
    J[i, j] = d(c_i) / d(theta_j)

    Row ordering follows OPERATORS exactly.
    """
    g, lf11, lf22, lf33, qf33, m2 = gWH, gWLf11, gWLf22, gWLf33, gWqf33, mWp**2
    J = np.zeros((25, 5))

    # i=0  O3pQ3:  -0.25*g*qf33/m2
    J[0, 0] = -0.25 * qf33 / m2;       J[0, 4] = -0.25 * g / m2
    # i=1  O3pl1:  -0.25*g*lf11/m2
    J[1, 0] = -0.25 * lf11 / m2;       J[1, 1] = -0.25 * g / m2
    # i=2  O3pl2:  -0.25*g*lf22/m2
    J[2, 0] = -0.25 * lf22 / m2;       J[2, 2] = -0.25 * g / m2
    # i=3  O3pl3:  -0.25*g*lf33/m2
    J[3, 0] = -0.25 * lf33 / m2;       J[3, 3] = -0.25 * g / m2
    # i=4  OQQ1:   0.08333*qf33^2/m2
    J[4, 4] = 2 * 0.08333333333333333 * qf33 / m2
    # i=5  OQQ8:   -1.0*qf33^2/m2
    J[5, 4] = -2.0 * qf33 / m2
    # i=6  OQl13:  -lf11*qf33/m2
    J[6, 1] = -qf33 / m2;              J[6, 4] = -lf11 / m2
    # i=7  OQl1M:  +lf11*qf33/m2
    J[7, 1] =  qf33 / m2;              J[7, 4] =  lf11 / m2
    # i=8  OQl33:  -lf33*qf33/m2
    J[8, 3] = -qf33 / m2;              J[8, 4] = -lf33 / m2
    # i=9  OQl3M:  +lf33*qf33/m2
    J[9, 3] =  qf33 / m2;              J[9, 4] =  lf33 / m2
    # i=10 Obp:    -2*0.006002*g/m2
    J[10, 0] = -2 * 0.006002165432052166 * g / m2
    # i=11 Oll1111: -0.25*lf11/m2
    J[11, 1] = -0.25 * lf11 / m2
    # i=12 Oll1122: 0.125*lf22/m2, 0.125*lf11/m2
    J[12, 1] = 0.125 * lf22 / m2;     J[12, 2] = 0.125 * lf11 / m2
    # i=13 Oll1133: 0.125*lf33/m2, 0.125*lf11/m2
    J[13, 1] = 0.125 * lf33 / m2;     J[13, 3] = 0.125 * lf11 / m2
    # i=14 Oll1221: -0.25*lf22/m2, -0.25*lf11/m2
    J[14, 1] = -0.25 * lf22 / m2;     J[14, 2] = -0.25 * lf11 / m2
    # i=15 Oll1331: 0.25*lf33/m2, 0.25*lf11/m2
    J[15, 1] =  0.25 * lf33 / m2;     J[15, 3] =  0.25 * lf11 / m2
    # i=16 Oll2222: 0.25*lf22/m2
    J[16, 2] = 0.25 * lf22 / m2
    # i=17 Oll2233: 0.125*lf33/m2, 0.125*lf22/m2
    J[17, 2] = 0.125 * lf33 / m2;     J[17, 3] = 0.125 * lf22 / m2
    # i=18 Oll2332: 0.25*lf33/m2, 0.25*lf22/m2
    J[18, 2] =  0.25 * lf33 / m2;     J[18, 3] =  0.25 * lf22 / m2
    # i=19 Oll3333: -0.25*lf33/m2
    J[19, 3] = -0.25 * lf33 / m2
    # i=20 Op:      -2*0.12938*g/m2
    J[20, 0] = -2 * 0.12938347743146458 * g / m2
    # i=21 OpBox:   -2*0.375*g/m2
    J[21, 0] = -2 * 0.375 * g / m2
    # i=22 OpQM:    0.25*qf33/m2, 0.25*g/m2
    J[22, 0] = 0.25 * qf33 / m2;      J[22, 4] = 0.25 * g / m2
    # i=23 Otap:    -2*0.002556*g/m2
    J[23, 0] = -2 * 0.0025559460452279554 * g / m2
    # i=24 Otp:     -2*0.24807*g/m2
    J[24, 0] = -2 * 0.2480703588615627 * g / m2

    return J


# ── Build K and C from theory DB ───────────────────────────────────────────────

def build_KC(use_theory_covmat=True):
    """
    Build:
        K  (n_obs x 25) theory matrix
        C  (n_obs x n_obs) full covariance (block diagonal, returned as dense)
        Ci (n_obs x n_obs) inverse covariance

    Mirrors _build_K_Ci in run_pipeline.py but also returns C for Cholesky.
    """
    K_blocks, C_blocks, Ci_blocks = [], [], []

    for ds in DS_NAMES:
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
        C_ds = np.diag([float(e)**2 if float(e) > 0 else (0.01 * abs(float(d)))**2
                        for e, d in zip(stat, dc)])

        sys_mat = sm.get("systematics", None)
        if sys_mat is not None:
            S = np.array(sys_mat, dtype=float)
            if S.ndim == 2 and S.shape[0] == n:
                C_ds += S @ S.T

        if use_theory_covmat:
            th_cov = th.get("theory_cov", None)
            if th_cov is not None:
                T = np.array(th_cov, dtype=float)
                if T.shape == (n, n):
                    C_ds += T

        K_ds = np.zeros((n, len(OPERATORS)))
        for j, op in enumerate(OPERATORS):
            k = lo.get(op, 0.0)
            if isinstance(k, list):
                K_ds[:, j] = [float(k[i]) if i < len(k) else 0.0 for i in range(n)]
            else:
                K_ds[:, j] = float(k)

        K_blocks.append(K_ds)
        C_blocks.append(C_ds)
        try:
            Ci_blocks.append(np.linalg.inv(C_ds))
        except np.linalg.LinAlgError:
            Ci_blocks.append(np.diag(1.0 / np.maximum(np.diag(C_ds), 1e-30)))

    K  = np.vstack(K_blocks)
    C  = block_diag(*C_blocks)
    Ci = block_diag(*Ci_blocks)
    return K, C, Ci


# ── Basis construction ─────────────────────────────────────────────────────────

def build_range_basis(A, tol=1e-8):
    """
    Economy SVD of A to get orthonormal range basis U_r.
    rank = number of singular values > tol * s_max.
    Returns U_r (n_obs x rank), rank.
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    rank = int(np.sum(s > tol * s[0]))
    return U[:, :rank], rank


def build_1op_basis(K, L, op_name="OQl13"):
    """
    Build whitened range basis for single-operator subset.
    K_1op = K[:, idx] as column vector.
    A_1op = L^{-1} K_1op.
    Returns (U_r shape (n,1), rank=1).
    """
    idx    = OPERATORS.index(op_name)
    K_1op  = K[:, idx:idx+1]
    A_1op  = np.linalg.solve(L, K_1op)
    U_r, rank = build_range_basis(A_1op)
    return U_r, rank


def build_full_smeft_basis(K, L):
    """
    Build whitened range basis for full 25-operator SMEFT.
    A = L^{-1} K  (n x 25).
    Rank = 14 for this FCC-ee setup.
    Returns (U_r shape (n, 14), rank=14).
    """
    A     = np.linalg.solve(L, K)
    U_r, rank = build_range_basis(A)
    return U_r, rank


def build_uv_basis(K, L, gWH, gWLf11, gWLf22, gWLf33, gWqf33, mWp):
    """
    Build whitened range basis for UV-constrained 5-parameter subspace.
    K_eff_UV = K @ J(theta)  (n x 5).
    A_UV = L^{-1} K_eff_UV.
    Returns (U_r shape (n, rank_uv), rank_uv <= 5).
    """
    J       = wprime_jacobian(gWH, gWLf11, gWLf22, gWLf33, gWqf33, mWp)
    K_eff   = K @ J
    A_eff   = np.linalg.solve(L, K_eff)
    U_r, rank = build_range_basis(A_eff)
    return U_r, rank


# ── Toy generation and test statistics ─────────────────────────────────────────

def generate_sm_toys(N_toys, seed=42):
    """
    Generate N_toys whitened standard normal vectors u ~ N(0, I).
    eps = L @ u^T ~ N(0, C).
    Returns u of shape (N_toys, n_obs).
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_toys, L_GLOBAL.shape[0]))


def compute_q_toys(u, U_r):
    """
    Compute q_i = ||U_r^T u_i||^2 for all toys.
    Under H0=SM: q ~ chi2(rank) exactly (linear Gaussian).

    Parameters
    ----------
    u   : (N_toys, n_obs) whitened toy vectors
    U_r : (n_obs, rank) orthonormal range basis

    Returns
    -------
    q_toys : (N_toys,) array
    """
    proj = u @ U_r        # (N_toys, rank)
    return np.sum(proj**2, axis=1)


# ── BSM Asimov test statistic ───────────────────────────────────────────────────

def compute_q_bsm(gWH, gWLf, gWqf, mWp, F, U_r_1op, U_r_full):
    """
    Compute q_BSM for the Asimov dataset (BSM signal, no noise).

    For Asimov: delta = K c_truth, so:
        q_full = c_truth^T F c_truth   (= delta^T Ci delta = chi2_SM)
        q_1op  = c_OQl13^2 * F[idx, idx]
        q_uv   = c_truth^T F c_truth   (same as full, UV truth = zero residual)

    All returned in a dict.
    """
    c = eft_coefficients_vec(gWH, gWLf, gWLf, gWLf, gWqf, mWp)

    q_full = float(c @ F @ c)

    idx   = OPERATORS.index("OQl13")
    # q_1op = (g_idx)^2 / F[idx,idx]  where g_idx = F[idx,:] @ c (score at Asimov truth)
    # Equivalent to ||U_r_1op^T a_BSM||^2 in the whitened space
    g_idx = float(F[idx, :] @ c)
    q_1op = g_idx**2 / float(F[idx, idx])

    # UV: same as full for Asimov (truth achieves zero residual)
    q_uv = q_full

    return {"full": q_full, "1op": q_1op, "uv": q_uv}


# ── Significance conversion ────────────────────────────────────────────────────

def q_to_sigma_wilks(q, ndof):
    """Wilks approximation: q ~ chi2(ndof). One-sided discovery convention."""
    if q <= 0 or ndof <= 0:
        return 0.0
    p = chi2_dist.sf(q, df=ndof)
    if p <= 0:
        return 99.0
    return float(norm_dist.isf(p))   # one-sided: sigma = Phi^{-1}(1-p)


def q_to_sigma_toys(q_bsm, q_toys):
    """
    Toy-corrected significance using empirical null distribution.
    p = fraction of SM toys with q > q_BSM.
    sigma = Phi^{-1}(1-p)  [one-sided discovery].
    """
    N      = len(q_toys)
    n_above = int(np.sum(q_toys > q_bsm))
    if n_above == 0:
        return float("inf")   # all toys below q_BSM: extrapolate with Wilks
    p = n_above / N
    if p >= 1.0:
        return 0.0
    sigma = float(norm_dist.isf(p))
    return max(0.0, sigma)   # clamp: discovery significance is always >= 0


# ── Main scan ──────────────────────────────────────────────────────────────────

def run_scan(K, C, Ci, N_toys, seed, gWH_grid, mWp_grid, fermion_ratio):
    """
    Full discovery reach scan over (gWH, mWp) grid.

    Returns list of result dicts, one per grid point.
    """
    global L_GLOBAL

    F = K.T @ Ci @ K
    L = np.linalg.cholesky(C)
    L_GLOBAL = L   # make available to generate_sm_toys

    rank_full = int(np.sum(np.linalg.eigvalsh(F) > 1e-8 * np.linalg.eigvalsh(F).max()))

    print(f"\n  Fisher rank: {rank_full}")
    print(f"  Generating {N_toys} SM toy datasets (seed={seed})...")
    u = generate_sm_toys(N_toys, seed)

    print("  Building constant operator bases...")
    U_r_1op,  rank_1op  = build_1op_basis(K, L, "OQl13")
    U_r_full, rank_full  = build_full_smeft_basis(K, L)
    print(f"    1-op  basis: rank={rank_1op}")
    print(f"    Full SMEFT:  rank={rank_full}")

    print("  Precomputing constant toy q-values...")
    q_toys_1op  = compute_q_toys(u, U_r_1op)
    q_toys_full = compute_q_toys(u, U_r_full)

    results = []
    n_total = len(gWH_grid) * len(mWp_grid)
    n_done  = 0

    print(f"\n  Scanning {n_total} grid points...\n")

    for gWH in gWH_grid:
        for mWp in mWp_grid:
            n_done += 1
            gWLf = gWH * fermion_ratio
            gWqf = gWH * fermion_ratio

            # Whitened signal vector a = L^{-1} K c_truth
            c = eft_coefficients_vec(gWH, gWLf, gWLf, gWLf, gWqf, mWp)
            a = np.linalg.solve(L, K @ c)   # (n_obs,) whitened signal

            # H1 pseudo-data toys: shift SM toys by signal
            # u_h1_i = u_i + a  →  q_H1_i = ||U_r^T (u_i + a)||^2
            # Under H1: q ~ non-central chi2(rank, lambda=||U_r^T a||^2)
            u_h1 = u + a[np.newaxis, :]     # (N_toys, n_obs)

            # UV basis (model-point-dependent — J depends on couplings)
            U_r_uv, rank_uv = build_uv_basis(K, L, gWH, gWLf, gWLf, gWLf, gWqf, mWp)
            q_toys_uv       = compute_q_toys(u, U_r_uv)

            # BSM Asimov q values (noiseless, for Wilks reference)
            q_bsm = compute_q_bsm(gWH, gWLf, gWqf, mWp, F, U_r_1op, U_r_full)

            # H1 toy q distributions (signal present)
            q_h1_1op  = compute_q_toys(u_h1, U_r_1op)
            q_h1_full = compute_q_toys(u_h1, U_r_full)
            q_h1_uv   = compute_q_toys(u_h1, U_r_uv)

            # Median q under H1 → compare to H0 null → median expected significance
            s_h1_1op  = q_to_sigma_toys(float(np.median(q_h1_1op)),  q_toys_1op)
            s_h1_full = q_to_sigma_toys(float(np.median(q_h1_full)), q_toys_full)
            s_h1_uv   = q_to_sigma_toys(float(np.median(q_h1_uv)),   q_toys_uv)
            if not np.isfinite(s_h1_1op):  s_h1_1op  = q_to_sigma_wilks(float(np.median(q_h1_1op)),  rank_1op)
            if not np.isfinite(s_h1_full): s_h1_full = q_to_sigma_wilks(float(np.median(q_h1_full)), rank_full)
            if not np.isfinite(s_h1_uv):   s_h1_uv   = q_to_sigma_wilks(float(np.median(q_h1_uv)),   rank_uv)

            # Wilks significances (one-sided, Asimov)
            s_wk_1op  = q_to_sigma_wilks(q_bsm["1op"],  ndof=rank_1op)
            s_wk_full = q_to_sigma_wilks(q_bsm["full"], ndof=rank_full)
            s_wk_uv   = q_to_sigma_wilks(q_bsm["uv"],   ndof=rank_uv)

            # Asimov toy-corrected significances (H0 null, Asimov q)
            s_toy_1op  = q_to_sigma_toys(q_bsm["1op"],  q_toys_1op)
            s_toy_full = q_to_sigma_toys(q_bsm["full"], q_toys_full)
            s_toy_uv   = q_to_sigma_toys(q_bsm["uv"],   q_toys_uv)

            # Cap inf at Wilks value for plotting
            if not np.isfinite(s_toy_1op):  s_toy_1op  = s_wk_1op
            if not np.isfinite(s_toy_full): s_toy_full = s_wk_full
            if not np.isfinite(s_toy_uv):   s_toy_uv   = s_wk_uv

            disc = ("DISCOVERY" if s_h1_uv >= 5.0 else
                    "evidence"  if s_h1_uv >= 3.0 else "")
            print(f"  [{n_done:3d}/{n_total}] gWH={gWH:.3f}  mWp={mWp:.2f} TeV  "
                  f"q={q_bsm['full']:7.2f}  "
                  f"sigma_H1(1op)={s_h1_1op:.2f}  "
                  f"sigma_H1(full)={s_h1_full:.2f}  "
                  f"sigma_H1(UV)={s_h1_uv:.2f}  {disc}")

            results.append(dict(
                gWH=gWH, mWp=mWp,
                q_bsm_full=q_bsm["full"], q_bsm_1op=q_bsm["1op"],
                rank_uv=rank_uv,
                sigma_wilks_1op=s_wk_1op,   sigma_wilks_full=s_wk_full,
                sigma_wilks_uv=s_wk_uv,
                sigma_toys_1op=s_toy_1op,   sigma_toys_full=s_toy_full,
                sigma_toys_uv=s_toy_uv,
                sigma_h1_1op=s_h1_1op,      sigma_h1_full=s_h1_full,
                sigma_h1_uv=s_h1_uv,
                # Store toys for diagnostic plots at selected points
                q_toys_1op=q_toys_1op if (gWH, mWp) in DIAG_POINTS else None,
                q_toys_full=q_toys_full if (gWH, mWp) in DIAG_POINTS else None,
                q_toys_uv=q_toys_uv if (gWH, mWp) in DIAG_POINTS else None,
                q_h1_uv=q_h1_uv if (gWH, mWp) in DIAG_POINTS else None,
            ))

    return results, u, U_r_1op, U_r_full, F, rank_1op, rank_full


# ── Plotting ───────────────────────────────────────────────────────────────────

def _contour_panel(ax, mass_arr, coupling_arr, sigma_grid, title, n_sigma=[3.0, 5.0]):
    """Draw a single discovery-region contour panel."""
    import matplotlib.ticker as ticker

    grid = np.nan_to_num(np.clip(sigma_grid, 0, 20), nan=0.0)

    ax.contourf(mass_arr, coupling_arr, grid,
                levels=[5.0, 21.0], colors=["#ADD8E6"], alpha=0.55)
    ax.contourf(mass_arr, coupling_arr, grid,
                levels=[0, 5.0], colors=["#DCDCDC"], alpha=0.55)

    for lev, ls, lw, lbl in [(5.0, "-",  2.5, r"$5\sigma$"),
                              (3.0, "--", 1.5, r"$3\sigma$")]:
        try:
            cs = ax.contour(mass_arr, coupling_arr, grid,
                            levels=[lev], colors=["#1f77b4"],
                            linewidths=[lw], linestyles=[ls])
            ax.clabel(cs, fmt=lbl, fontsize=9, inline=True)
        except Exception:
            pass

    ax.set_xlabel(r"$m_{W'}$ [TeV]", fontsize=11)
    ax.set_ylabel(r"$g_{WH}$", fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(True, which="major", alpha=0.2, lw=0.5)


def plot_discovery_region(results, plt_dir, N_toys, rank_full=14):
    """
    2x3 grid figure: rows = Wilks/Toy-corrected, cols = 1op/Full/UV.
    Plus a comparison figure with all 5sigma contours overlaid.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gWH_vals = sorted(set(r["gWH"] for r in results))
    mWp_vals = sorted(set(r["mWp"] for r in results))
    nc, nm   = len(gWH_vals), len(mWp_vals)

    def make_grid(key):
        g = np.full((nc, nm), np.nan)
        for r in results:
            ic = gWH_vals.index(r["gWH"])
            im = mWp_vals.index(r["mWp"])
            g[ic, im] = r[key]
        return g

    mass_arr     = np.array(mWp_vals)
    coupling_arr = np.array(gWH_vals)

    methods = [
        ("sigma_wilks_1op",  "sigma_toys_1op",  "sigma_h1_1op",  "1-op: OQl13 (ndof=1)"),
        ("sigma_wilks_full", "sigma_toys_full", "sigma_h1_full", f"Full SMEFT (rank={rank_full})"),
        ("sigma_wilks_uv",   "sigma_toys_uv",   "sigma_h1_uv",   "UV-profiled (5 params)"),
    ]

    # Figure 1: 3x3 grid — rows: Wilks / Asimov toy / H1 pseudo-data
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    for col, (wk_key, toy_key, h1_key, label) in enumerate(methods):
        _contour_panel(axes[0, col], mass_arr, coupling_arr,
                       make_grid(wk_key),
                       f"Wilks (Asimov): {label}")
        _contour_panel(axes[1, col], mass_arr, coupling_arr,
                       make_grid(toy_key),
                       f"Asimov + H0 toys: {label}")
        _contour_panel(axes[2, col], mass_arr, coupling_arr,
                       make_grid(h1_key),
                       f"H1 pseudo-data (median): {label}")

    fig.suptitle(r"$W'$ discovery reach at FCC-ee — Wilks vs Asimov vs H1 pseudo-data",
                 fontsize=13)
    plt.tight_layout()
    out = f"{plt_dir}/discovery_region_all_methods.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    # Figure 2: comparison — all 5sigma contours overlaid (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors  = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    lstyles = ["-", "--", ":"]
    labels  = ["1-op (OQl13)", "Full SMEFT", "UV-profiled"]

    for ax, (row_wk, row_toy), title in [
        (axes[0], [("sigma_wilks_1op","sigma_wilks_full","sigma_wilks_uv"), None],
                   "Wilks (Asimov)"),
        (axes[1], [("sigma_toys_1op","sigma_toys_full","sigma_toys_uv"), None],
                   f"Asimov + H0 toys (N={N_toys})"),
        (axes[2], [("sigma_h1_1op","sigma_h1_full","sigma_h1_uv"), None],
                   f"H1 pseudo-data median (N={N_toys})"),
    ]:
        import matplotlib.ticker as ticker
        keys = row_wk   # tuple of sigma key names for this panel (Wilks or Toy)
        # Explicitly set axis limits first so contour failures don't leave default [0,1] axes
        ax.set_xlim(mass_arr[0], mass_arr[-1])
        ax.set_ylim(coupling_arr[0], coupling_arr[-1])
        for key, col, ls, lbl in zip(keys, colors, lstyles, labels):
            try:
                # Clip to [0, 50] to avoid -inf/-large values breaking contour
                # np.clip(-inf, 0, 50) == 0, then nan_to_num cleans any NaN
                grid_clipped = np.nan_to_num(np.clip(make_grid(key), 0, 50), nan=0.0)
                cs = ax.contour(mass_arr, coupling_arr, grid_clipped,
                                levels=[5.0], colors=[col], linewidths=[2.0],
                                linestyles=[ls])
                ax.clabel(cs, fmt=f"5σ ({lbl})", fontsize=8, inline=True)
            except Exception:
                pass
        ax.set_xlabel(r"$m_{W'}$ [TeV]", fontsize=12)
        ax.set_ylabel(r"$g_{WH}$", fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.grid(True, which="major", alpha=0.2, lw=0.5)

    from matplotlib.lines import Line2D
    legend_els = [Line2D([0],[0], color=c, lw=2, ls=ls, label=lbl)
                  for c, ls, lbl in zip(colors, lstyles, labels)]
    axes[2].legend(handles=legend_els, fontsize=9, loc="upper right")
    fig.suptitle(r"$W'$ discovery reach: 5$\sigma$ contours by method", fontsize=12)
    plt.tight_layout()
    out2 = f"{plt_dir}/discovery_region_comparison.png"
    plt.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out2}")


def plot_null_distributions(results, u, U_r_1op, U_r_full, K, L, plt_dir, N_toys):
    """
    For each diagnostic (gWH, mWp) point:
    Plot null distribution histogram + q_BSM marker + chi2(ndof) curve.
    3 subpanels: 1op / Full SMEFT / UV.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import chi2 as chi2_dist

    F = K.T @ (np.linalg.solve(L.T, np.linalg.solve(L, K)))

    for gWH, mWp in DIAG_POINTS:
        matching = [r for r in results if r["gWH"] == gWH and r["mWp"] == mWp]
        if not matching:
            continue
        r = matching[0]

        gWLf = gWH * FERMION_RATIO
        gWqf = gWH * FERMION_RATIO

        q_bsm_full = r["q_bsm_full"]
        q_bsm_1op  = r["q_bsm_1op"]

        U_r_uv, rank_uv = build_uv_basis(K, L, gWH, gWLf, gWLf, gWLf, gWqf, mWp)
        q_toys_1op  = compute_q_toys(u, U_r_1op)
        q_toys_full = compute_q_toys(u, U_r_full)
        q_toys_uv   = compute_q_toys(u, U_r_uv)

        rank_1op  = U_r_1op.shape[1]
        rank_full = U_r_full.shape[1]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, q_toys, q_bsm, ndof, label in [
            (axes[0], q_toys_1op,  q_bsm_1op,  rank_1op,  "1-op: OQl13"),
            (axes[1], q_toys_full, q_bsm_full, rank_full, "Full SMEFT"),
            (axes[2], q_toys_uv,   q_bsm_full, rank_uv,   "UV-profiled"),
        ]:
            bins = np.linspace(0, max(q_toys.max(), q_bsm) * 1.1, 60)
            ax.hist(q_toys, bins=bins, density=True, alpha=0.6,
                    color="#4C72B0", label=f"SM toys (N={N_toys})")

            # chi2(ndof) theory curve
            x = np.linspace(0, bins[-1], 300)
            ax.plot(x, chi2_dist.pdf(x, df=ndof), "k--", lw=1.5,
                    label=fr"$\chi^2({ndof})$ Wilks")

            # q_BSM line
            ax.axvline(q_bsm, color="crimson", lw=2.0, ls="-",
                       label=f"q_BSM = {q_bsm:.1f}")

            # Shade p-value region
            n_above = int(np.sum(q_toys > q_bsm))
            p_val   = n_above / N_toys
            ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1],
                             q_bsm, bins[-1], alpha=0.2, color="crimson",
                             label=f"p = {p_val:.4f}")

            sig_toy   = q_to_sigma_toys(q_bsm, q_toys)
            sig_wilks = q_to_sigma_wilks(q_bsm, ndof)
            ax.set_xlabel("q", fontsize=11)
            ax.set_ylabel("density", fontsize=11)
            ax.set_title(f"{label}\n"
                         f"σ(toys)={sig_toy:.2f}  σ(Wilks)={sig_wilks:.2f}",
                         fontsize=10)
            ax.legend(fontsize=8)

        fig.suptitle(f"Null distribution: gWH={gWH}, mWp={mWp} TeV", fontsize=12)
        plt.tight_layout()
        tag = f"gWH{int(gWH*100):03d}_mWp{int(mWp*10):03d}"
        out = f"{plt_dir}/null_dist_{tag}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


def plot_sigma_vs_mass(results, plt_dir):
    """
    For each gWH slice: sigma vs mWp for all methods + Wilks vs toys.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gWH_vals = sorted(set(r["gWH"] for r in results))

    for gWH in gWH_vals:
        pts = sorted([r for r in results if r["gWH"] == gWH], key=lambda x: x["mWp"])
        if not pts:
            continue
        mWp_vals = [r["mWp"] for r in pts]

        fig, ax = plt.subplots(figsize=(9, 5))

        styles = [
            ("sigma_wilks_1op",  "#1f77b4", "-",  "Wilks 1-op"),
            ("sigma_toys_1op",   "#1f77b4", "--", "Toy 1-op"),
            ("sigma_wilks_full", "#ff7f0e", "-",  "Wilks Full SMEFT"),
            ("sigma_toys_full",  "#ff7f0e", "--", "Toy Full SMEFT"),
            ("sigma_wilks_uv",   "#2ca02c", "-",  "Wilks UV"),
            ("sigma_toys_uv",    "#2ca02c", "--", "Toy UV"),
        ]
        for key, col, ls, lbl in styles:
            vals = [r[key] for r in pts]
            ax.plot(mWp_vals, vals, color=col, ls=ls, lw=1.8, label=lbl)

        ax.axhline(5.0, color="gold",   lw=2,   ls="--", label=r"5$\sigma$")
        ax.axhline(3.0, color="orange", lw=1.5, ls=":",  label=r"3$\sigma$")
        ax.set_xlabel(r"$m_{W'}$ [TeV]", fontsize=12)
        ax.set_ylabel(r"Significance $\sigma$", fontsize=12)
        ax.set_title(fr"Discovery significance vs mass — $g_{{WH}}={gWH}$", fontsize=11)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = f"{plt_dir}/sigma_vs_mass_gWH{int(gWH*100):03d}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()


def save_table(results, path, N_toys, seed, use_theory_covmat):
    header = (
        f"# True discovery reach: W' model at FCC-ee\n"
        f"# Toy MC null distribution: N_toys={N_toys}, seed={seed}\n"
        f"# Theory covmat: {'on' if use_theory_covmat else 'off'}\n"
        f"# sigma convention: one-sided discovery (Phi^{{-1}}(1-p))\n"
        f"# Wilks: q ~ chi2(ndof) approximation\n"
        f"# Toys:  empirical null from SM toy datasets\n"
        f"# H1:    median q under H1=W' pseudo-data, compared to H0 null\n"
        f"# {'gWH':>8}  {'mWp':>6}  {'q_bsm':>10}  "
        f"{'wk_1op':>8}  {'wk_full':>8}  {'wk_uv':>8}  "
        f"{'toy_1op':>8}  {'toy_full':>8}  {'toy_uv':>8}  "
        f"{'H1_1op':>8}  {'H1_full':>8}  {'H1_uv':>8}\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for r in results:
            f.write(
                f"  {r['gWH']:>8.4f}  {r['mWp']:>6.2f}  {r['q_bsm_full']:>10.3f}  "
                f"{r['sigma_wilks_1op']:>8.3f}  {r['sigma_wilks_full']:>8.3f}  "
                f"{r['sigma_wilks_uv']:>8.3f}  "
                f"{r['sigma_toys_1op']:>8.3f}  {r['sigma_toys_full']:>8.3f}  "
                f"{r['sigma_toys_uv']:>8.3f}  "
                f"{r['sigma_h1_1op']:>8.3f}  {r['sigma_h1_full']:>8.3f}  "
                f"{r['sigma_h1_uv']:>8.3f}\n"
            )
    print(f"  Saved: {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

L_GLOBAL = None  # set by run_scan, used by generate_sm_toys

def main():
    p = argparse.ArgumentParser(
        description="True discovery reach for W' at FCC-ee (toy MC null distribution)"
    )
    p.add_argument("--n-toys",           type=int,   default=10000)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--no-theory-covmat", action="store_true")
    p.add_argument("--fermion-ratio",    type=float, default=FERMION_RATIO,
                   help="gWLf/gWH = gWqf/gWH (default 1/3)")
    p.add_argument("--out-dir",          type=str,   default=None)
    args = p.parse_args()

    use_theory_covmat = not args.no_theory_covmat

    out_base = args.out_dir or str(PIPELINE / "results" / "discovery_reach")
    plt_dir  = f"{out_base}/plots"
    tbl_dir  = f"{out_base}/tables"
    for d in [plt_dir, tbl_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 64)
    print("  TRUE DISCOVERY REACH PIPELINE  —  W' at FCC-ee")
    print(f"  N_toys        = {args.n_toys}  |  seed = {args.seed}")
    print(f"  Theory covmat : {'on' if use_theory_covmat else 'off (--no-theory-covmat)'}")
    print(f"  Fermion ratio : gWLf = gWqf = {args.fermion_ratio} × gWH")
    print(f"  Grid          : {len(GWH_GRID)} × {len(MWP_GRID)} = {len(GWH_GRID)*len(MWP_GRID)} points")
    print(f"  Output        : {out_base}")
    print("=" * 64)

    print("\n[1] Building K, C, Ci from theory DB...")
    K, C, Ci = build_KC(use_theory_covmat)
    print(f"    K: {K.shape}, C: {C.shape}")

    print("\n[2/7] Running scan + generating toys...")
    results, u, U_r_1op, U_r_full, F, rank_1op, rank_full = run_scan(
        K, C, Ci,
        N_toys        = args.n_toys,
        seed          = args.seed,
        gWH_grid      = GWH_GRID,
        mWp_grid      = MWP_GRID,
        fermion_ratio = args.fermion_ratio,
    )

    print(f"\n[8] Saving table...")
    save_table(results, f"{tbl_dir}/discovery_reach_table.txt",
               args.n_toys, args.seed, use_theory_covmat)

    print(f"\n[9] Generating plots...")
    plot_discovery_region(results, plt_dir, args.n_toys, rank_full)
    plot_null_distributions(results, u, U_r_1op, U_r_full, K, L_GLOBAL, plt_dir, args.n_toys)
    plot_sigma_vs_mass(results, plt_dir)

    print(f"\n{'='*64}")
    print(f"  DONE  —  results in {out_base}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
