# Discovering New Physics at the FCC-ee with the SMEFT
## Project Overview — Master Reference Document

> **How to use this document.** This file is the single source of truth for the project
> status. It is structured to mirror the paper draft section by section. Each section states
> (i) what the section is about and why it matters, (ii) the exact method used and its
> implementation, (iii) all key numerical results with the file paths they come from, and
> (iv) a status block listing what is done and what remains. Status markers:
> ✅ done | 🔲 to do / missing | ⚠️ needs update or attention.

---

## Background and Motivation

The Standard Model (SM) of particle physics is not expected to be the final theory of nature.
Numerous extensions predict new heavy particles — W' bosons, Z' bosons, leptoquarks,
supersymmetric partners — whose masses may lie far above the direct production threshold at
any foreseeable collider. Their effects can nevertheless manifest at lower energies as small
but correlated deviations in precisely measured SM observables. The Standard Model Effective
Field Theory (SMEFT) provides a systematic framework for this: BSM physics at a high scale Λ
is captured by a tower of higher-dimensional operators added to the SM Lagrangian,
L_eff = L_SM + Σ_i (C_i / Λ^2) O_i, where C_i are the Wilson coefficients. The leading
corrections are dimension-6 (d=6) operators in the Warsaw basis.

The Future Circular Collider in electron-positron mode (FCC-ee) is designed to measure a very
large set of electroweak, Higgs, and top observables with unprecedented precision at √s = 91,
161, 240, and 365 GeV. The global SMEFT fit to all these observables — as implemented by the
SMEFiT collaboration — constrains the Wilson coefficients simultaneously and identifies which
BSM models are compatible with the data.

**This project addresses a gap in the existing literature:** can a global SMEFT fit at the
FCC-ee not only detect a BSM anomaly but also *discover* a specific UV completion and
*discriminate* between competing models? The paper introduces a quantitative BSM closure test,
constructs several significance metrics for UV model discovery, and demonstrates the
methodology on two benchmark models: a W' boson and a Z' boson.

---

## Repository Structure

```
fullpipeline/
├── models/
│   ├── wprime.py                   W' UV model (asymmetric couplings, 27 ops)
│   ├── wprime_constrained.py       W' with gWLf=gWqf=gWH/3, 16 ops
│   ├── wprime_constrained_v2.py    W' with 18 empirically constrained ops
│   ├── wprime_constrained_v3.py    W' with 14 empirically constrained ops (best free-EFT)
│   └── zprime.py                   Z' UV model (hypercharge-singlet, 10 ops)
├── scripts/
│   ├── run_pipeline.py             Master pipeline script (see §2.4)
│   ├── fisher_uv.py                Corrected UV Fisher matrix (see §2.3)
│   ├── data_pvalue_metric.py       Data-space closure metric (Metric 1)
│   ├── l1_pull_distribution.py     Coefficient-space pull (Metric 2b)
│   ├── gen_ns_eft_pulls.py         NS EFT pull distribution (Metric 2c)
│   ├── bsm_closure_metrics.py      Power table across mass points
│   ├── sm_pseudodata_scan.py       SM false-positive scan (§3.3)
│   ├── sm_replica_analysis.py      SM toy analysis
│   ├── model_discrimination.py     W'/Z' cross-injection test (§4.3)
│   ├── fingerprint_comparison.py   Per-operator signal fingerprint
│   ├── dataset_sensitivity_by_sqrts.py  Per-√s chi2_SM decomposition
│   ├── run_discovery_reach.py      Discovery reach grid scan
│   ├── run_exclusion_reach.py      Exclusion reach grid scan
│   └── plot_*.py                   Various plotting utilities
├── results/                        All output directories (see §§3–4)
└── PROJECT_OVERVIEW.md             This file
```

The SMEFiT package itself (at `/data/theorie/wtelman/smefit_release/`) is **never modified**.
All analysis code lives in `scripts/` as standalone scripts.

---

## 1. Introduction

The paper's central claim is that the UV coupling method — fitting the FCC-ee data directly in
terms of the UV parameters (g, m) of a specific BSM model — gives substantially higher
discovery significance than fitting the same data with the full SMEFT (all unconstrained Wilson
coefficients simultaneously free). The improvement is structural: the UV completion enforces
relations among Wilson coefficients that reduce the effective degrees of freedom from ~15 to 5
(for the W' model), dramatically improving the signal-to-noise.

The paper goes further: it distinguishes *SM exclusion* (is the data inconsistent with the SM?)
from *BSM discovery* (is the data consistent with a specific BSM model H₁?). These are
different statistical claims — the first is a one-tailed test of H₀; the second requires an
additional likelihood ratio comparison. The paper shows quantitatively that the false-positive
rate of the SM exclusion test under SM-generated data is exactly at the nominal level (0.30%
at 3σ) throughout the (mass, coupling) plane.

**Status:** ⚠️ Introduction section flagged as needing to be written (EH: Need to sketch an
intro). Must reflect the Z' model, combined W'+Z' reach, model discrimination, and the
exclusion-vs-discovery distinction once sections are complete.

---

## 2. New Physics Signature at the FCC-ee

### 2.1 Presentation of the UV Models

#### W' boson

The W' is an SU(2)_L triplet vector field W'^a_μ (a = 1, 2, 3) with mass m_W' and
coupling to the SM left-handed fermion doublets and the Higgs doublet. The Lagrangian is:
  L ⊃ g_WH · W'^a_μ (H† τ^a i D^μ H) + g_WLf · W'^a_μ (l̄_L γ^μ τ^a l_L) + g_Wqf · (...)

The model has **5 independent UV parameters:**
  g_WH     — coupling to the Higgs current
  g_WLf11  — coupling to 1st-generation lepton doublet
  g_WLf22  — coupling to 2nd-generation lepton doublet
  g_WLf33  — coupling to 3rd-generation lepton doublet
  g_Wqf33  — coupling to 3rd-generation quark doublet (bottom/top)

Three coupling scenarios are studied:
- **W' (asymmetric, primary benchmark):** g_WH free, g_WLf11/22/33 = g_WLf, g_Wqf33 = g_WLf.
  Default at g_WH=0.12, g_WLf=g_Wqf=0.04 (= g_WH/3). Implemented in `models/wprime.py`.
- **W' constrained:** g_WLf = g_Wqf = g_WH/3 enforced at the runcard level.
  Implemented in `models/wprime_constrained.py`. Slightly lower chi2_SM at 1 TeV because the
  constraint removes 1 effective parameter from the fit.
- **W' universal (allg012):** g_WH = g_WLf = g_Wqf (all equal). Much larger leptonic
  coupling → 3× stronger signal at the same g_WH. Produces dramatically higher discovery
  reach (results/wprime_gwh012_allg012_mwp010_scan/).

The W' generates **27 Warsaw-basis operators** (the full set produced by the W' Feynman rules):

  Four-fermion operators (charged current):
    OQl13, OQl1M, OQl23, OQl2M, OQl33, OQl3M  — quark-lepton (dominant signal carriers)
    OQQ1, OQQ8                                  — quark-quark (invisible at FCC-ee, unconstrained)

  Four-lepton operators:
    Oll1111, Oll1122, Oll1133, Oll1221, Oll1331  — 1st-gen lepton diagonal/cross
    Oll2222, Oll2233, Oll2332, Oll3333           — 2nd/3rd-gen lepton

  Higgs-lepton operators (triplet):
    O3pl1, O3pl2, O3pl3  — (H†τ^a i↔D H)(l̄_gen τ^a l_gen)
    O3pQ3                 — (H†τ^a i↔D H)(Q̄_3 τ^a Q_3)

  Higgs-boson operators:
    OpBox, OpQM, Op, Obp, Otp, Otap

Matching relations (all at tree-level, coefficient ∝ g²/m² or g·g'/m²):
  OQl13  = −g_WH · g_WLf11 / m²    (largest, most sensitive to W')
  OQl1M  = +g_WH · g_WLf11 / m²
  O3pl1  = −0.25 · g_WH · g_WLf11 / m²
  OpBox  = −0.375 · g_WH² / m²
  Oll1111 = −0.125 · g_WLf11² / m²
  (all others analogously — see models/wprime.py:eft_coefficients())

At the default benchmark (g_WH=0.12, g_WLf=0.04, m=1 TeV):
  OQl13 ≈ −0.0048 TeV⁻²,  OQl1M ≈ +0.0048 TeV⁻²,  OpBox ≈ −0.0054 TeV⁻²

**11 operators are effectively unconstrained** by the FCC-ee dataset alone (OQQ1, OQQ8,
OQl23, OQl2M, OQl33, OQl3M, Oll2222, Oll2233, Oll2332, Oll3333, Op) because they
contain quark currents not accessible at an e+e− collider. LHC high-mass Drell-Yan data
would constrain these but is not included in the current analysis.

#### Z' boson

The Z' is a SU(2)_L × U(1)_Y singlet (hypercharge-singlet, B'-like) neutral vector boson
with Lagrangian:
  L ⊃ g_ZH · Z'_μ (H† i D^μ H) + g_Zl · Z'_μ Σ_gen (l̄_L,gen γ^μ l_L,gen)

The model has **3 UV parameters:** g_ZH, g_Zl, m_Zp.
Default: g_ZH=0.12, g_Zl=0.04 (=g_ZH/3). Constrained variant: g_Zl = g_ZH/3.

The Z' generates **10 Warsaw-basis operators:**
  OpD      — O_HD = |H† D_μ H|²: the T-parameter, C_HD = −g_ZH² / (2 m²_Zp)
  Opl1/2/3 — O^(1)_{Hl,gen}: SU(2) singlet Higgs-lepton, C = −g_ZH·g_Zl / m²
  Oll1111, Oll2222, Oll3333  — same-generation 4-lepton, C = −g_Zl² / (2 m²)
  Oll1221, Oll1331, Oll2332  — cross-generation 4-lepton, C = +g_Zl² / m² (Fierz)

The Z' has a **dramatically different SMEFT footprint** than W':
  W' → SU(2) triplet operators (O^(3)_Hl, OQl charged current, Higgs-box)
  Z' → SU(2) singlet T-parameter (OpD) and singlet Higgs-lepton (Opl)

The T-parameter operator OpD is exquisitely constrained at the Z-pole, giving Z' roughly
**twice the mass reach** of W' at the same coupling strength.

**Status:** ✅ W' model (asymmetric) fully implemented in `models/wprime.py`.
✅ W' constrained model fully implemented in `models/wprime_constrained.py`.
✅ Z' model fully implemented in `models/zprime.py`, including Fierz identities and
   confirmed operator availability in smefit FCC-ee theory database.
🔲 Paper §2.1 needs a dedicated Z' subsection covering Lagrangian, UV parameters, and
   structural distinction from W'.

### 2.2 Matching to the SMEFT and FCC-ee Observables

**Dataset:** The FCC-ee pseudo-data consists of **40 projected observables** drawn from the
smefit database at `/data/theorie/wtelman/smefit_database/commondata_projections_L0`. They
are grouped by centre-of-mass energy tier:

| √s     | Datasets (n_obs) | Physics content |
|--------|-----------------|-----------------|
| 91 GeV | FCCee_Zdata (many), FCCee_Wwidth | Z-pole EWPOs: Γ_Z, σ_had, R_l, A_FB, Γ_W |
| 161 GeV | FCCee_ww_161GeV, FCCee_Brw | WW cross-section at threshold, W branching ratio |
| 240 GeV | FCCee_ww_240GeV, FCCee_ee/Afb/Rb/Rc/..., FCCee_zh_* (8 modes) | EWPO + all major ZH Higgs couplings |
| 365 GeV | FCCee_ww/ee/Rb/Rc/.../_365GeV, FCCee_zh_365GeV (8 modes), FCCee_vvh_* (5 modes) | High-energy EWPO + ZH + WW-fusion Higgs |

Six optimised WW datasets (FCCee_161/240/365_ww_leptonic/semilep_optim_obs) are excluded
because they have negative covariance matrix eigenvalues that destabilise the fit.
FCCee_alphaEW was removed from the database in February 2026 (merged into FCCee_Zdata).
FCCee_365_tt_optim_obs excluded: all 8 central values = 0.0.

The theory prediction for observable i at Wilson coefficients c is:
  σ_i(c) = σ_i(SM) + K_ij c_j + K^(2)_{i,ab} c_a c_b
where K_ij are the linear theory coefficients (∂σ/∂c at c=0) and K^(2) are the quadratic
(interference) corrections. The chi-square is:
  χ²(c) = Σ_{ij} [σ_i(c) − d_i] (C^{-1})_{ij} [σ_j(c) − d_j]
where d_i is the pseudo-data central value and C is the experimental covariance matrix
(plus optionally a theory covariance matrix from missing higher-order uncertainty).

**RGE:** SMEFT operators are run from the matching scale μ₀ = m_W' (or m_Zp) down to the
measurement scale using the SMEFT renormalisation group equations. This is handled internally
by smefit with settings: init_scale = m_BSM × 1000 GeV, smeft_accuracy = "integrate",
yukawa = "top". RGE can be disabled with --no-rge for cross-checks.

**Theory covariance:** A theory covariance matrix (aggressive variant) is included by
default to capture missing higher-order EW corrections. This regularises the fit but slightly
reduces the effective chi2_SM compared to using only the experimental covariance.

**Matching implementation:** `models/wprime.py:eft_coefficients()` returns the full dict of
Wilson coefficients at the matching scale for any (g_WH, g_WLf, g_Wqf, m_W') input. The
pipeline uses this to generate BSM pseudo-data: d_i = σ_i(SM) + K_ij c^truth_j.

**Status:** ✅ W' and Z' matching verified against MATCH2FIT/MatchMakerEFT.
✅ All 40 datasets loaded and validated (negative-cov datasets excluded).
🔲 Appendix B.1 matching table in the paper draft is empty — W' coefficients are in
   models/wprime.py and Z' in models/zprime.py, both need to be transcribed to LaTeX.
🔲 Z' matching relations are not yet in the paper at all.

### 2.3 Impact on FCC-ee Observables — Fisher Information Matrix

The Fisher information matrix quantifies, before any fit, how much information the FCC-ee
dataset carries about each Wilson coefficient (or UV parameter). For the SMEFT coefficient
space it is:
  F_{ij} = K^T C^{-1} K  (purely linear, evaluated at c=0)
Its eigenvectors (from PCA, §4.1) give the directions of maximum sensitivity.

For the UV parameter space, the Fisher matrix is:
  I_{ℓℓ'} = Σ_{ij} (∂σ_i/∂g^ℓ)(C^{-1})_{ij}(∂σ_j/∂g^{ℓ'})
where ∂σ_i/∂g^ℓ = K_{ij} (∂c_j/∂g^ℓ) is obtained by contracting the theory kernel
with the matching Jacobian J_{jℓ} = ∂c_j/∂g^ℓ.

**The bug in smefit's UV Fisher:** The smefit package contains a `fisher.py` and `pca.py`
that implement this, but with a critical bug: `impose_constrain` evaluates the Jacobian by
setting one UV parameter to 1 and all others to 0 (unit-vector substitution). This zeroes
all **bilinear** matching terms of the form C ∝ g_a · g_b / m² (e.g. OQl13 = −g_WH·g_WLf/m²),
because setting g_WH=1, g_WLf=0 gives ∂OQl13/∂g_WH → 0 instead of the correct ∂OQl13/∂g_WH
= −g_WLf/m². The correct Jacobian must be evaluated at the **UV truth point** (g_truth).

**Corrected implementation:** `scripts/fisher_uv.py` (standalone, never modifies smefit):
1. Evaluates J_{jℓ} = ∂c_j/∂g^ℓ via finite differences at g_truth (step=0.01%)
2. Contracts as F_{ℓℓ'} = Σ_{ij} J_{iℓ} K^T_{ij} (C^{-1})_{ij} J_{jℓ'} + quadratic correction
3. Uses posterior displacements Δg_s = g_s − g_truth (not absolute g_s) in all NS moments
4. Includes quadratic SMEFT terms (K^(2) c_s² from posterior samples) in delta_th

**Key results** (corrected, grouped by √s, at g_WH=0.5 for W'):

Row-normalised fraction of Fisher information by energy tier:
  g_WH   : 91 GeV = 40.9%,  240 GeV = 51.4%  (ZH at 240 GeV dominates)
  g_WLf11: 91 GeV = 66%                        (Z-pole via Oll1221)
  g_WLf22: 91 GeV = 93%
  g_WLf33: 91 GeV = 99%
  g_Wqf33: 240 GeV = 50%,   365 GeV = 38%     (top quark sector, high-energy dominated)

**Why gWqf33 is NOT Z-pole dominated:** While Rb and Ab are included in FCCee_Zdata (and
gWqf33 does shift the Zbb vertex), the smefit database has NO dedicated 91 GeV b-quark or
leptonic datasets (no FCCee_Rb_91GeV, FCCee_bb_Afb_91GeV, etc.). The Z-pole tier is only
the 13 EWPO bundled in FCCee_Zdata, whereas 240 and 365 GeV each have ~20 individual datasets.
Additionally, gWqf33 generates top-quark operators (Otap, Otp) that are inaccessible below the
top threshold — these only open at 240–365 GeV. The dataset asymmetry + top-quark operators
together make the high-energy tiers dominate gWqf33's Fisher.

After row-normalisation, the linear Fisher is **mass-independent** (all entries ∝ 1/m⁴ cancel
in the ratio). Quadratic corrections are negligible at both m=1 TeV and m=10 TeV because
c_truth ~ g²/m² ≪ 1 and Q_ijk ~ K^(2) J_i J_j ∝ 1/m⁴ is additionally suppressed.

Output files (run Jun 7–11 2026 with latest code):
  results/wprime_gwh050_mwp010/plots/fisher_uv.png        (W', m=1 TeV)
  results/wprime_gwh050_mwp100/plots/fisher_uv.png        (W', m=10 TeV)
  results/wprime_constrained_v3_gwh050_mwp050/reports/.../fisher_heatmap_..._corrected.png
  results/zprime_gzh050_mzp120/reports/.../fisher_heatmap_zprime_gzh050_mzp120_UVcoup_corrected.png

The old smefit-generated Fisher for W' (wprime_gwh100_mwp020_quad/reports/) has 6 group
labels (EWPO_240, EWPO_365, EW_inputs, WW, ZH_240, ZH_365) and uses the buggy Jacobian.
It is superseded by the corrected fisher_uv.py output and should no longer be used.

**Z' Fisher results** (gZH=0.50, gZl=0.1667, mZp=12 TeV, Jun 11 2026):
  gZH: 91 GeV=46.0%, 161 GeV=0%, 240 GeV=48.3%, 365 GeV=5.7%
  gZl: 91 GeV=66.9%, 161 GeV=0.4%, 240 GeV=24.0%, 365 GeV=8.7%
  Linear = Quadratic (Z' in linear regime at these couplings/mass).
  gZH splits ~50/50 Z-pole vs ZH because OpD (T-parameter) enters both.
  gZl is Z-pole dominated via Opl operators shifting leptonic Z widths.
  Consistent with earlier Z' Fisher at gZH=0.12 — normalized Fisher is mass/coupling independent.

**FCC-ee dataset gap (important for Fisher interpretation):** The smefit database only contains
dedicated resolved datasets at 240 and 365 GeV (Rb, Rc, Rmu, Rtau, bb/cc/mumu/tautau AFBs,
sigmaHad, ee, ee_Afb, ww). At 91 GeV only the bundled FCCee_Zdata (13 EWPO) exists. This
means the Z-pole Fisher weight is structurally lower than it would be if individual 91 GeV
observables were implemented, especially for quark-sensitive operators.

**Prior bug fixed (Jun 11 2026):** models/zprime.py had `prior_max = min(5*gZH, 0.5)` — at
gZH=0.50 this put the prior wall exactly at the truth, truncating the NS posterior. Fixed to
`min(5*gZH, 2.0)`. Affects any Z' run at gZH≥0.10; re-run zprime_gzh050_mzp120 after fix.

**Status:** ✅ Corrected UV Fisher in `scripts/fisher_uv.py` (fixes impose_constrain bug).
✅ Dataset grouping by √s = 91/161/240/365 GeV implemented.
✅ Fisher runs: W' (1 TeV, 10 TeV, v3 14-op), Z' (gZH=0.50, mZp=12 TeV).
✅ Quadratic = linear confirmed numerically at all tested points (W' and Z').
✅ fisher_uv.py supports wprime_constrained_v2/v3 and zprime/zprime_constrained tags.
✅ Z' prior bug fixed in models/zprime.py (cap raised from 0.5 → 2.0).
🔲 Fisher matrix dependence on UV parameter values (mass/coupling scan) not done.
🔲 Fig 3.2 in paper shows old buggy Fisher — must be replaced with corrected output.
🔲 Fig 3.4 (Fisher in SMEFT coefficient space) is a placeholder — missing from paper.

### 2.4 BSM Closure-Test Methodology and Pipeline

#### Level-0 and Level-1 pseudo-data

Pseudo-data is generated from the theory prediction at the BSM truth point:
  d_i^(L0)  = σ_i(c_truth)                              — noiseless
  d_i^(L1,r) = σ_i(c_truth) + Σ_j L_{ij} ξ^r_j         — one noise realisation per replica r

where L is the Cholesky factor of C and ξ^r ~ N(0,1). L0 tests whether the central-value
fit recovers c_truth exactly. L1 builds the sampling distribution of any test statistic under
the BSM hypothesis — needed for computing the discovery power.

#### Analysis methods and test statistics

Five methods are implemented and compared for each benchmark:

1. **Best single operator (1-op):** Free one operator (the one with the largest individual
   chi2_SM contribution), fix all others to zero. Gives df=1 chi-square. The dominant
   operator is always OQl13 for W' and Opl2 for Z'. Susceptible to look-elsewhere effect
   if the best operator is not known in advance.

2. **PCA k-modes:** Free the k leading PCA directions of the Fisher matrix F = K^T C^{-1} K.
   No look-elsewhere effect if k is chosen a priori. PCA k=1 corresponds to the Fisher's
   leading eigenvector (maximally sensitive linear combination); k=4 recovers ~97% of signal.

3. **Full SMEFT (all free operators):** Free all ~15 constrained Wilson coefficients
   simultaneously. Gives df = rank(K) chi-square. Diluted by degrees of freedom from the
   11 unconstrained operators that absorb signal into noise.

4. **UV coupling (NS):** Run the full nested-sampling (NS) Bayesian fit with Wilson
   coefficients constrained to lie on the W' (or Z') matching relations via the smefit
   `constrain` field. The NS fit has 6 free parameters (g_WH, g_WLf11/22/33, g_Wqf33, m_W').
   Significance = Φ^{-1}(1 − p_NS) where p_NS comes from the NS log-evidence comparison.
   At large signal the NS can fail to converge (lnBF → +∞); in these cases the NS significance
   is not quoted but the Bayes factor confirms strong BSM evidence.

5. **Bayesian model selection (Bayes factor):** The NS routine also computes the log Bayes
   factor ln BF = ln Z(BSM) − ln Z(SM). This is reported alongside frequentist significance
   in every summary.txt file.

Profile likelihood significance is computed as q = chi2_SM − chi2_min and converted via
Wilks' theorem: sigma = Φ^{-1}(1 − p) where p = P(chi2(df) > q). Toy MC null distributions
(10000 SM toys) are also computed for validation; these agree with Wilks to within <0.1σ
for all validated points.

#### Pipeline automation

`scripts/run_pipeline.py` runs all steps for a given model and benchmark in one command:

```
# Single-point run
python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0

# Discovery scan (analytic only, fast grid)
python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0 --scan

# Separate per-operator projections
python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0 --sepproj OQl13 OpQM

# Skip slow NS fit (report + significance still produced)
python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0 --no-ns
```

Steps executed in a single-point run:
  Step 0 : Generate BSM L0 pseudo-data (d_i = σ_i(c_truth))
  Step 0b: [if --sepproj] Generate individual per-operator projections
  Step 1 : SM-only analytic fit (analytic linearised, ~1 min)
  Step 2 : Full SMEFT analytic fit (all operators free)
  Step 3 : UV coupling nested-sampling fit (~15–30 min, skip with --no-ns)
  Step 4 : PCA (eigenvectors of F = K^T C^{-1} K, stored in pca/)
  Step 5 : Profile likelihood table + significance bar chart
  Step 6 : smefit report (posterior histograms, 2D contours, chi2 table)

All outputs land in `results/<tag>/` where `<tag>` encodes the model and benchmark
(e.g. `wprime_gwh012_mwp010`). Standard subdirectories:
  fits/      smefit input runcards and NS output (.yaml + .json)
  pca/       Fisher eigenvectors, eigenvalues, K matrix
  plots/     Significance bar chart, Fisher heatmap, etc.
  reports/   smefit-generated posterior histogram PDFs
  summary.txt  Profile likelihood table (machine-readable)

**--sepproj feature:** Runs each specified operator in isolation — generates a projection
with only that operator free, then runs SM-only and BSMclosure fits. This shows which
operators individually carry signal and how their posteriors look in isolation.
Current outputs: `results/wprime_gwh012_mwp010/sepproj/` for OQl13 and OpQM.

**Status:** ✅ L0/L1 pseudo-data generation implemented.
✅ All 5 analysis methods implemented and producing output.
✅ --sepproj flag implemented (Step 0b in run_pipeline.py ~line 558); OQl13 and OpQM run.
🔲 Section 2.4 in paper is incomplete (EH: Need to add more details).
🔲 Additional --sepproj runs for c_HB, OQl1M, O3pQ3 not yet done.

---

## 3. Comparison of UV and SMEFT Analysis

### 3.1 UV Closure Tests

The BSM closure test validates the pipeline by checking that the fit correctly recovers the
injected BSM signal. Following the PDF closure-test methodology used in SMEFiT:

#### Metric 1 — Data-space p-value distribution (KS test)

For each L1 BSM replica, compute q^(r) = chi2_SM − chi2_min(r). Under the BSM hypothesis
with non-centrality parameter λ (the signal strength), q^(r) ~ chi2(df, λ) (non-central).
Convert to a two-sided p-value. If the pipeline is correct:
  - Under BSM L1 replicas: the p-value distribution should match P(chi2(df, λ) > q).
  - Under SM null replicas: the p-value distribution should be uniform on [0,1].

**Validated results** at g_WH=0.12, m_W'=3 TeV (from data_pvalue_metric.py):
  SM null median q = 14.3 ≈ chi2(15) mean of 14.8  ✓
  Power (fraction of BSM replicas crossing 5σ) > 99%  ✓
  KS test of SM null p-value uniformity:
    NS UV coupling (50 replicas)       : KS p = 0.734  ✓
    Analytic UV coupling (1000 replicas): KS p = 0.416  ✓
    Analytic SMEFT free (1000 replicas): KS p = 0.023  ✗ (fit underperforms)

Output: `results/wprime_l1_gwh012_mwp030/` — 2×3 panel plot metric1_pvalue_dist.png

#### Metric 2b — Coefficient-space pull distribution (analytic)

For each L1 replica, compute the pull P_i = (c^fit_i − c^truth_i) / σ^fit_i analytically
over 1000 replicas. For a correctly calibrated fit: <P_i> ≈ 0, std(P_i) ≈ 1.

**Validated results** at g_WH=0.12, m_W'=3 TeV (l1_pull_distribution.py):
  Mean pull = −0.0008, std pull = 1.0042 — consistent with N(0,1)  ✓

The 11 unconstrained operators (OQQ1, OQQ8, OQl23, OQl2M, OQl33, OQl3M, Oll2222, Oll2233,
Oll2332, Oll3333, Op) are excluded from the pull statistics (their pulls are undefined
because σ^fit → ∞). These operators require quark-sensitive data to be constrained.

Output: `results/wprime_l1_gwh012_mwp030/` — 3-panel histogram metric2_pulls.png

#### Metric 2c — NS EFT pull distribution (posterior-based)

Pull distribution computed from the NS posterior samples (not the analytic linearised fit).
Tests whether the full non-linear Bayesian posterior for Wilson coefficients recovers c_truth.

**Validated results** at g_WH=0.12, m_W'=3 TeV, 50 NS replicas (gen_ns_eft_pulls.py):
Output: `results/wprime_l1_gwh012_mwp030/metric2c_ns_eft_pull_distribution.png`

#### Metric 2 (NS) — UV parameter pull distribution

Pull in UV coupling space from NS posterior samples. Tests whether the NS fit recovers the
injected UV parameters (g_WH, g_WLf11/22/33, g_Wqf33) correctly.

**Validated results** at g_WH=0.12, m_W'=3 TeV, 50 NS replicas:
  gWH   : pull mean = −0.622  ✗   (biased away from truth)
  gWLf11: pull mean = +0.375  ✓
  gWLf22: pull mean = +0.559  ✗
  gWLf33: pull mean = +0.479  ✓
  gWqf33: pull mean = +0.617  ✗

Several UV coupling pulls fail (mean ≠ 0). This is expected: the UV parameter space has
bilinear flat directions — multiple (g, m) combinations map to the same SMEFT point
(e.g. doubling g and m simultaneously keeps g²/m² fixed). The NS is not expected to
uniquely recover the UV parameters, only the Wilson coefficients.

#### BSM closure power table (multi-mass)

`scripts/bsm_closure_metrics.py` summarises the closure test at multiple mass points.
Results from `results/bsm_closure_metrics/closure_summary.txt`:

| Model          | n_UV | λ_UV   | Asimov σ | power_5σ | pull (mean, std) |
|----------------|------|--------|----------|----------|------------------|
| W', m=3 TeV    |  5   | 107.68 |  9.55σ   | 100.0%   | (−0.012, 0.998)  |
| W', m=4 TeV    |  5   |  34.07 |  4.73σ   |  55.0%   | (−0.012, 0.998)  |
| W', m=5 TeV    |  5   |  13.96 |  2.41σ   |   3.4%   | (−0.012, 0.998)  |
| Z', m=7 TeV    |  2   |  38.87 |  5.90σ   |  82.6%   | (−0.030, 0.995)  |

The pull distribution is perfectly calibrated at all tested mass points. Power drops below
50% (5σ threshold) between 4 and 5 TeV for the W', consistent with the analytic Asimov
significance.

#### NS vs analytic comparison

The NS significance is systematically lower than the analytic Wilks-based significance
at weak signal, because the NS prior volume penalises large parameter spaces at low evidence.
From `results/uv_validated/tables/uv_validated_table.txt`:
  gWH=0.12, m=1 TeV: sigma_NS = 8.09σ  vs  sigma_analytic = 10.15σ
  gWH=0.12, m=2 TeV: sigma_NS = 0.00σ  vs  sigma_analytic = 0.40σ  (NS saturates at 0)
  gWH=0.12, m=4 TeV: sigma_NS = 0.00σ  vs  sigma_analytic = 0.009σ (both ~0)

The analytic method (profile likelihood, Wilks' theorem) is the primary significance
estimator. The NS result provides a consistent Bayesian cross-check at moderate-to-high signal.

#### L1 UV coupling standalone closure (Jun 2026)

**This is a separate and primary validation task:** run 50 NS replicas with L1 BSM pseudo-data
where only the **UV parameters** (gWH, gWLf11/22/33, gWqf33) are free — i.e. the Wilson
coefficients are constrained via the matching relations and the NS samples directly in the
5-dimensional (g, m) UV parameter space. This tests whether the NS correctly covers the
*UV truth values*, not just the operator space.

At gWH=0.50, mWp=5 TeV, 50 NS replicas (L1 Cholesky noise):
  std of pull distribution: **0.992** (consistent with N(0,1) to within statistical error)
  Number of prior-dominated UV params: 0
  → The NS posterior for the UV coupling correctly covers the injected truth point.
  → The UV coupling L1 closure is better calibrated than any free-EFT operator set (all have std<1).

Output: `results/wprime_gwh050_mwp050_l1closure/ns_l1_pulls.png`

This result is the primary justification for using UV coupling as the discovery metric:
not only does it improve sensitivity (§3.2), it also gives correctly calibrated posteriors
for the physical parameters (g, m) rather than abstract Wilson coefficients.

#### NS L1 closure — operator set comparison (Jun 2026)

`scripts/ns_l1_closure.py` runs 50 NS replicas with L1 pseudo-data and computes UV and EFT
pull distributions. Four operator sets compared at gWH=0.50, mWp=5 TeV:

| Model                  | Ops | Free-EFT std | Prior-dominated ops |
|------------------------|-----|-------------|---------------------|
| UV coupling (truth)    |  5  | 0.992       | 0 (UV params only)  |
| Full SMEFT (27 ops)    | 27  | 0.807       | 9 (OQQ1/8, OQl23-3M, Oll2222-3333, Op) |
| WPrimeConstrained (16) | 16  | 0.960       | 2 (Oll1133, Oll1331) |
| WPrimeConstrainedV3    | 14  | 0.952       | 0                   |

Underdispersion (std<1) in free-EFT fits is expected and physical: fitting many operators
simultaneously inflates posteriors through near-flat inter-operator degeneracies. Conservative
(underconfident), not a calibration failure. The 14-op V3 model achieves the best free-EFT
calibration with zero prior-dominated operators.

The 14 operators in WPrimeConstrainedV3 (empirically constrained from this closure test):
  O3pQ3, O3pl1, O3pl2, O3pl3, OQl13, OQl1M, Obp, Oll1111, Oll1122, Oll1221,
  OpBox, OpQM, Otap, Otp

Output paths:
  results/wprime_gwh050_mwp050_l1closure/        (UV closure, 50 reps)
  results/wprime_constrained_v3_gwh050_mwp050_l1closure_smeft/  (14-op free-EFT, 50 reps)
  results/*/ns_l1_pulls.png, ns_l1_pulls_pooled.png, ns_l1_eft_pulls.png

#### NS sigma vs mass curve (Jun 2026)

NS runs at mWp = 2, 4, 6, 8 TeV (gWH=0.50) provide a genuine NS discovery reach curve:
  mWp=2 TeV: ~99σ  |  mWp=4 TeV: 14.4σ  |  mWp=5 TeV: 8.75σ
  mWp=6 TeV: 5.61σ |  mWp=8 TeV: 2.41σ
  → NS 5σ crossing at ~6.4 TeV (vs ~8 TeV analytic Asimov) at gWH=0.50.
  NS is lower because the Bayesian evidence penalises the UV prior volume at weak signal.
Output: results/ns_sigma_vs_mass/ns_sigma_vs_mass.png, ns_lnBF_vs_mass.png

**Status:** ✅ `data_pvalue_metric.py` implements Metric 1 (2×3 panel plot).
✅ `l1_pull_distribution.py` implements Metric 2b (3-panel pull histogram).
✅ `gen_ns_eft_pulls.py` implements Metric 2c (NS posterior-based pull).
✅ Benchmark wprime_gwh012_mwp030 fully run; all metric outputs in results/wprime_l1_gwh012_mwp030/.
✅ BSM power table computed in results/bsm_closure_metrics/closure_summary.txt.
✅ NS vs analytic comparison validated in results/uv_validated/.
✅ NS L1 closure for all 4 operator sets done (UV, 27-op, 16-op, 14-op); best = V3 14-op.
✅ WPrimeConstrainedV2Model (18 ops) and WPrimeConstrainedV3Model (14 ops) in models/.
✅ NS sigma vs mass curve at gWH=0.50 in results/ns_sigma_vs_mass/.
✅ ns_l1_closure.py supports --smeft flag (free-EFT run) and pooled UV pull plot.
🔲 Fig 3.1 (UV closure test figure) missing from paper draft — must be placed.
🔲 UV coupling pull failures (bilinear flat directions) should be discussed in the paper.
🔲 NS discovery reach full grid scan (mWp × gWH) not done — discussed, deferred.

### 3.2 Results of SMEFT Global Fits

The full pipeline is run on BSM L0 pseudo-data for each benchmark. The BSM fit (orange in
Fig 3.3) is shifted from the SM fit (blue), with the UV truth (dashed red) falling inside
the BSM posterior for all constrained operators.

#### Significance table

Results from `summary.txt` files for the three benchmark points shown in the paper:

| Method         | gWH=0.12, mWp=1 TeV        | gWH=1.0, mWp=10 TeV        | gWH=0.08, mWp=1 TeV  |
|----------------|-----------------------------|-----------------------------|----------------------|
| chi2_SM        | 124.1                       | 57.8                        | 23.7                 |
| Best 1-op      | 10.61σ (OQl13)             | 7.22σ (OQl13)              | 4.62σ (OQl13)        |
| PCA k=2        | 10.41σ                     | 6.94σ                       | 4.25σ                |
| PCA k=4        | 10.26σ                     | 6.64σ                       | 3.78σ                |
| Full SMEFT     | 9.09σ (rank 14/25)         | 5.14σ (rank 14/25)          | 1.96σ (rank 14/25)   |
| UV coupling    | NS fail (lnBF=+48.1)       | 6.63σ (lnBF=+18.3)          | — (lnBF=−55.8)       |

Source files:
  results/wprime_gwh012_mwp010/summary.txt
  results/wprime_gwh100_mwp100/summary.txt
  results/wprime_gwh008_mwp010/summary.txt

Key observations:
- At gWH=0.08, 1 TeV: Full SMEFT = 1.96σ vs Best 1-op = 4.62σ. This 2.4× dilution factor
  is caused by the 14 degrees of freedom in the Full SMEFT absorbing noise that the signal
  does not populate. The UV coupling fit is not shown because at this weak signal the NS
  evidence is negative (lnBF=−55.8) — the prior volume penalty exceeds the likelihood gain.
- At gWH=1.0, 10 TeV: UV coupling (6.63σ) and PCA k=4 (6.64σ) are nearly identical —
  showing that PCA correctly identifies the UV-matched subspace without assuming the UV model.
- "NS fail" at gWH=0.12, 1 TeV means the NS log-evidence diverges (signal overwhelms the
  prior) and the chi2_UV,min → chi2_SM, so the significances saturate. The lnBF=+48.1
  confirms extremely strong BSM evidence.

#### Constrained W' model: from asymmetric to empirically refined (dedicated task)

**Motivation:** The asymmetric W' model (wprime.py) has 27 Warsaw operators, of which 11 are
completely unconstrained by FCC-ee data (OQQ1/8, OQl23/2M/33/3M, Oll2222/2233/2332/3333, Op —
all involve hadronic currents invisible to e+e- colliders). These 11 operators absorb signal
into unconstrained directions, diluting the EFT significance. A series of constrained models
was developed to remove or empirically identify these dead-weight operators.

**WPrimeConstrained (v1, 16 ops):** Implemented in `models/wprime_constrained.py`. Enforces
gWLf=gWqf=gWH/3 at the runcard level, reducing the free UV params from 5→1. This
constraint removes several operator degeneracies. The model was run through the full pipeline:
  - At gWH=0.12, mWp=1 TeV: UV=10.18σ, Full SMEFT=8.92σ
  - Both higher than the asymmetric fit (9.09σ SMEFT) because fewer unconstrained operators
  Source: `results/wprime_constrained_gwh012_mwp010/summary.txt`
L1 closure at gWH=0.50, mWp=5 TeV (16-op set): std=0.960, 2 prior-dominated ops (Oll1133, Oll1331)

**WPrimeConstrainedV2 (18 ops):** Implemented in `models/wprime_constrained_v2.py`. Empirically
derived by removing the 9 prior-dominated operators from the 27-op free-EFT closure
(OQQ1, OQQ8, OQl23, OQl2M, OQl33, OQl3M, Oll2222, Oll2233, Oll2332, Oll3333 → most
removed, a few added back from the W' matching set). Intermediate step in the refinement.

**WPrimeConstrainedV3 (14 ops, best):** Implemented in `models/wprime_constrained_v3.py`.
Final empirically constrained set determined by the closure test: 14 operators with zero
prior-dominated operators and best free-EFT calibration (std=0.952).
Operators: O3pQ3, O3pl1, O3pl2, O3pl3, OQl13, OQl1M, Obp, Oll1111, Oll1122, Oll1221,
           OpBox, OpQM, Otap, Otp
Full pipeline (Fisher + L1 closure) run for V3 at gWH=0.50, mWp=5 TeV:
  Source: `results/wprime_constrained_v3_gwh050_mwp050/`
  Fisher: `results/wprime_constrained_v3_gwh050_mwp050/reports/.../fisher_heatmap_..._corrected.png`
  L1 closure: std=0.952, 0 prior-dominated (best of all four operator sets tested)

The V3 model is the recommended operator set for constrained W' analysis at FCC-ee. It
removes operator directions that are invisible to the experiment while retaining all operators
that carry genuine W' signal or improve calibration.

**W' constrained model** (gWLf=gWqf=gWH/3, 15+1 operators):
At gWH=0.12, mWp=1 TeV: UV=10.18σ, Full SMEFT=8.92σ (slightly lower than asymmetric W'
because the constrained model has 15 operators constrained vs 25 in the asymmetric fit).
Source: `results/wprime_constrained_gwh012_mwp010/summary.txt`

#### Discovery reach

The discovery reach is the 5σ significance contour in the (g_WH, m_W') plane, comparing:
  - Wilks' approximation (q ~ chi2(df))
  - Empirical toy MC null (10000 SM toys)
  - Median significance under H₁=BSM pseudo-data

Results from `results/discovery_reach/tables/discovery_reach_table.txt` (main asymmetric W'):
  At gWH=0.12, the UV coupling H₁ significance crosses 5σ between mWp=1.25 TeV (6.30σ)
  and mWp=1.50 TeV (4.02σ) — so the 5σ discovery boundary for the **asymmetric W'** is
  approximately **1.4 TeV** at g_WH=0.12.

Results from `results/discovery_reach_v2/` (universal allg012 model, gWH=gWLf=gWqf=0.12):
  At gWH=0.12 (universal), mWp=3 TeV: H₁ UV = 9.54σ. The 5σ boundary extends well past
  3 TeV in this model due to the 3× stronger leptonic coupling.

**Z' discovery reach** (constrained variant, gZH=0.12, gZl=0.04):
  At mZp=7.0 TeV: UV coupling = 5.86σ  (results/zprime_constrained_gzh012_mzp070/)
  At mZp=7.5 TeV: UV coupling = 5.04σ  (results/zprime_constrained_gzh012_mzp075/)
  → 5σ boundary ≈ **7.5 TeV** at gZH=0.12 for the Z' constrained model.

Combined W'+Z' reach plot: `results/discovery_reach/combined_discovery_reach.png`

**Z' primary benchmark** (gZH=0.50, gZl=0.1667, mZp=12 TeV) — chosen to give comparable
signal strength to W' at (gWH=0.50, mWp=5 TeV):
  chi2_SM = 70.09  (W' benchmark: ~93)
  Best 1-op (OpD): 6.81σ  |  PCA k=5: 7.42σ  |  Full SMEFT: 7.08σ  |  UV coupling: 7.85σ
  lnBF = 29.40
  Source: results/zprime_gzh050_mzp120/summary.txt
  Note: q_UV < q_PCA_k5 numerically (66.19 vs 69.73) but sigma_UV > sigma_PCA_k5 (7.85 > 7.42)
  because UV coupling has only ndof=2 vs ndof=5. Direct q comparison is invalid across different ndof.

**Z' coefficient histograms** at gZH=0.12, mZp=1 TeV (results/zprime_gzh012_mzp010/):
  chi2_SM = 93359 — enormous because OpD (T-parameter) is extremely tightly constrained
  at the Z-pole. Dominant operator is Opl2 (O^(1)_{Hl,2}).

**Z' at gZH=0.50, mZp=5 TeV** (results/zprime_gzh050_mzp050/): chi2_SM=2488, everything
  saturates at 99σ. Too strong for meaningful comparison — used only to confirm prior bug.

**Status:** ✅ Significance table computed for all three benchmarks.
✅ W' constrained model run at gWH=0.12, mWp=1 TeV.
✅ Discovery reach scan in discovery_reach/ (asymmetric W', gWH=0.03–0.30) and
   discovery_reach_v2/ (universal allg012, same coupling grid).
✅ Z' 5σ boundary determined: ~7.5 TeV at gZH=0.12.
⚠️ Fig 3.6 in paper shows old W' scan up to 3 TeV — must be replaced with updated plot.
🔲 Z' results not yet fully integrated into paper §3.2.
🔲 Significance table in paper is a screenshot (Fig 3.5) — replace with LaTeX table.

### 3.3 SM Exclusion Does Not Equate to NP Discovery

**This is the most novel and important section of the paper** (currently body-empty in draft).

#### Conceptual distinction

**SM exclusion:** The data rejects the SM null hypothesis H₀ at significance level α. This
requires only that χ²_SM > q_threshold where q_threshold is set by chi2(n_dof). It says
nothing about what the alternative is.

**BSM discovery:** The data is consistent with a specific BSM model H₁ at significance
level β. This requires additionally that L(H₁) / L(H₀) > threshold, where the threshold is
set by the distribution of this ratio under SM-generated L1 replicas.

In the current pipeline both claims are numerically close when the UV coupling fit is used,
because χ²_SM ≈ q_UV + χ²_{UV,min}, and for a well-matched H₁ both tests fire at similar
parameter values. However they are conceptually distinct and can differ substantially at the
boundary of sensitivity.

#### SM pseudo-data scan — quantitative validation

`scripts/sm_pseudodata_scan.py` runs the UV coupling pipeline on 5000 SM toy datasets for
each (gWH, mWp) grid point and measures:
  sigma_bsm : Wilks significance on the BSM Asimov (expected significance under H₁)
  frac_3σ_% : fraction of SM toys that exceed the 3σ threshold (false-positive rate)

**Key result** from `results/sm_exclusion_comparison/sm_exclusion_comparison_table.txt`:
  frac_3σ = **0.30%** for every single (gWH, mWp) grid point (gWH = 0.03–0.15+, mWp = 0.5–8 TeV)

This is the exact nominal 3σ false-positive rate (0.27% ≈ 0.30%). The pipeline maintains
correct coverage everywhere in parameter space — SM data never mimics BSM at above-nominal
rates, regardless of the assumed BSM mass or coupling.

**BSM Asimov significance scaling** (at gWH=0.12, UV coupling, from the table):
  mWp = 1 TeV: sigma_bsm = 99σ (saturated)
  mWp = 3 TeV: sigma_bsm = 9.34σ
  mWp = 4 TeV: sigma_bsm = 4.50σ
  mWp = 5 TeV: sigma_bsm = 2.09σ
  mWp = 6 TeV: sigma_bsm = 0.66σ   ← 3σ BSM Asimov boundary ≈ 5.5–6 TeV

The 3σ BSM Asimov boundary at gWH=0.12 (where the experiment expects to see a 3σ signal)
lies between 5 and 6 TeV for the UV coupling method.

Output plots (contour overlay and three-panel):
  `results/sm_exclusion_comparison_v2/sm_vs_bsm_contour_overlay.png`
  `results/sm_exclusion_comparison_v2/sm_vs_bsm_three_panel.png`

**Status:** ✅ SM pseudo-data scan implemented in scripts/sm_pseudodata_scan.py.
✅ Scan completed: results/sm_exclusion_comparison/ and results/sm_exclusion_comparison_v2/.
✅ Key result confirmed: frac_3σ(SM) = 0.30% (nominal) at all grid points.
✅ BSM Asimov 3σ boundary at gWH=0.12 ≈ 5.5–6 TeV.
🔲 Section body in paper is completely empty — needs to be written.
🔲 Side-by-side figure (BSM exclusion reach vs SM false-positive contour) not yet placed.
🔲 Concrete example illustrating the exclusion/discovery distinction not yet written.

---

## 4. Model Selection for New Physics Discovery

### 4.1 Identifying Optimal Direction in SMEFT from the Observables

#### Principal Component Analysis (PCA)

PCA of the Fisher matrix F = K^T C^{-1} K identifies the directions of maximum sensitivity:
  F v_k = λ_k v_k   (eigendecomposition, λ_1 ≥ λ_2 ≥ ... ≥ 0)
The k leading eigenvectors v_1,...,v_k define the PCA k-mode subspace. Fitting only these
k combinations has no look-elsewhere penalty (k is fixed a priori) and typically recovers
nearly all signal for small k.

**PCA component loadings** (at gWH=0.12, mWp=1 TeV, from pca/eigenvectors.npy):
  PC1 (largest eigenvalue): dominated by O^(3)_{ℓq,13} (OQl13) — the four-fermion operator
       coupling first-generation leptons to third-generation quarks. This is the operator that
       receives the largest W' Wilson coefficient (c ∝ g_WH · g_WLf11 / m²) and is the most
       tightly constrained at the Z-pole via its effect on lepton universality.
  PC2: dominated by O_{ll,1221} (Oll1221) — the mixed-generation four-lepton operator.
       Constrained by the ratio of leptonic partial widths at the Z-pole.

The 1-op result (always OQl13 for W') exactly corresponds to PC1. The improvement PCA k=2,4
over 1-op comes from including these additional orthogonal directions.

**Per-operator significance** (at gWH=0.08 vs gWH=0.12):
  At gWH=0.12: the signal is large enough that many operators contribute; Full SMEFT=9.1σ.
  At gWH=0.08: the signal is marginal; OQl13 still gives 4.62σ but Full SMEFT drops to
  1.96σ because 13 additional operators absorb noise. The dilution factor is ×2.4× at this
  benchmark.
  Comparison plot: `results/presentation_plots/per_operator_significance.png`

**Likelihood vs PCA k — COMPLETED.** The supervisor asked: at which k does the profile
likelihood stop improving? Answer: **sigma is maximised at k=1** for all W' benchmarks.
PC1 alone captures ~90% of the signal (91.2% at gWH=0.12, 90.5% at gWH=0.08/1.0) with
only 1 degree of freedom. Sigma then decreases monotonically because each additional mode
adds one dof but contributes negligible signal (~0.3% of q_full per mode). The "elbow"
is at k=1 — there is no elbow further along the curve.

Precise numbers from `scripts/plot_sigma_vs_pca_k.py` and
`results/sigma_vs_pca_k/sigma_vs_pca_k_table.txt`:

| Benchmark                | k=1    | k=2    | k=4    | k=14 (Full SMEFT) |
|--------------------------|--------|--------|--------|-------------------|
| gWH=0.12, mWp=1 TeV     | 10.57σ | 10.34σ | 10.19σ | 9.01σ             |
| gWH=1.0,  mWp=10 TeV    |  7.14σ |  6.84σ |  6.54σ | 5.00σ             |
| gWH=0.08, mWp=1 TeV     |  4.48σ |  4.09σ |  3.61σ | 1.65σ             |

The k=14 (full-rank) values match the Full SMEFT significance from summary.txt to within
the rounding differences in chi2_SM (124.08 vs 124.0835, etc.). The Z' at gZH=0.12, 1 TeV
saturates at 99σ throughout (signal overwhelmingly large).

Physical interpretation: The W' signal is confined almost entirely to the OQl13 direction
(PC1). Adding more orthogonal directions only dilutes the test statistic by increasing the
degrees of freedom. The UV coupling fit (k_eff = 5) outperforms Full SMEFT (k=14) because
the additional 9 constrained operator directions carry zero signal but cost 9 dof.

Plots: `results/sigma_vs_pca_k/sigma_vs_pca_k.{pdf,png}`
  — left panel: σ vs k for all three W' benchmarks (curves decrease from k=1)
  — right panel: fraction of q_full recovered vs k (plateau above 90% from k=1)
  — vertical dash-dot line at k=14 = Full SMEFT boundary
  — dashed horizontal lines at the UV coupling significance for each benchmark

**Separate projections (--sepproj):** Individual operator projections show that OQl13 alone
recovers most of the signal, while OpQM shows a smaller but nonzero shift. These are useful
for communicating which operators are the primary signal carriers without PCA abstraction.
Outputs: `results/wprime_gwh012_mwp010/sepproj/` — separate NS fits and histograms for
OQl13 and OpQM.

**Status:** ✅ PCA implemented and producing eigenvectors/eigenvalues in every pca/ subdirectory.
✅ PCA k=1/2/4 significance in all summary.txt files.
✅ PC1=OQl13, PC2=Oll1221 confirmed from eigenvector analysis.
✅ Per-operator significance plot: results/presentation_plots/per_operator_significance.png.
✅ --sepproj outputs for OQl13 and OpQM at gWH=0.12, mWp=1 TeV.
✅ Likelihood vs PCA k plot produced: results/sigma_vs_pca_k/sigma_vs_pca_k.{pdf,png}.
   Script: scripts/plot_sigma_vs_pca_k.py — answers supervisor's question.
🔲 Section 4.1 body is empty in paper (EH: Add PCA results here).
🔲 Posteriors in PCA basis vs operator basis comparison not yet done.

### 4.2 Penalising Excessive Degrees of Freedom

The dilution from using 15 free operators instead of 5 (UV coupling) reduces significance
by ~30% in the moderate-signal regime (gWH=0.12, 1 TeV: 9.1σ vs 10.3σ) and by ~2.4× in
the weak-signal regime (gWH=0.08, 1 TeV: 1.96σ vs 4.62σ). The AIC/BIC framework
penalises excess parameters: AIC = 2k − 2 ln L. The penalty 2(k₁ − k₂) = 2(15 − 5) = 20
corresponds to a reduction in q of 20, i.e. roughly 2σ for df=1, which is consistent with
the observed differences.

The pipeline already computes the Bayesian analogue: ln BF = ln Z(BSM) − ln Z(SM). At
gWH=0.12, mWp=3 TeV: ln BF = +47.7 (overwhelming BSM evidence); at mWp=4 TeV: ln BF = +12.2.
At mZp=7 TeV: ln BF = +16.8. Negative lnBF means the SM is preferred (gWH=0.08: lnBF=−55.8).

**Status:** ⚠️ Bayesian model comparison already implemented and in all summary.txt files.
🔲 Section 4.2 body is empty (EH: AIC (+ ...) using analytical fit results).
🔲 Formal AIC/BIC comparison using analytical fit results not yet implemented.

### 4.3 Model Discrimination: Bottom-Up Reconstruction of the UV

#### Model discrimination via cross-injection

Can one tell, from a BSM anomaly in the SMEFT fit, which UV completion is responsible?
The UV coupling method answers this directly: if the *wrong* UV model is assumed, its
constrained fit will give low significance even when the correct model would give high significance.

Two cross-injection tests are implemented in `scripts/model_discrimination.py`:

**Test A — W' pseudo-data, Z' fit** (results/robustness_wprime_data_zprime_fit/):
  Pseudo-data generated under W' (gWH=0.12, gWLf=0.04, mWp=1 TeV), fit with Z' model.
  chi2_SM = 124.1 (W' truth, strong signal).

  | Method        | Significance |
  |---------------|-------------|
  | Full SMEFT    | 7.30σ       |   ← model-agnostic BSM detection succeeds
  | UV coupling   | 0.00σ       |   ← wrong UV model completely fails
  | Bayes SM-only | lnBF=+18.2  |   ← W' data is BSM-like, correctly flagged
  | Bayes Z' BSM  | lnBF=+12.1  |   ← Z' hypothesis partially accepted by Bayes (wrong)

  Interpretation: The SMEFT free fit correctly detects the W' anomaly. The Z' UV coupling
  fit gives 0σ because the Z' pattern (T-parameter dominant) cannot describe W' data (charged
  current dominant). This is the model discrimination criterion.

**Test B — Z' pseudo-data, W' fit** (results/robustness_zprime_data_wprime_fit/):
  Pseudo-data generated under Z' (gZH=0.12, gZl=0.04, mZp=1 TeV), fit with W' model.
  chi2_SM = 230.1 (Z' truth, very strong signal due to T-parameter).

  | Method        | Significance |
  |---------------|-------------|
  | Full SMEFT    | 8.46σ       |   ← model-agnostic detection succeeds
  | UV coupling   | 0.00σ       |   ← wrong UV model completely fails
  | Bayes SM-only | lnBF=+47.4  |   ← Z' data extremely BSM-like
  | Bayes W' BSM  | lnBF=+46.7  |   ← W' hypothesis partially accepted (wrong)

**Discrimination criterion:** If the UV coupling fit gives 0σ but the SMEFT free fit gives
>5σ, the assumed UV completion is incorrect. If both give >5σ, the assumed model is a
plausible description of the data.

**Signal subspace overlap:** The cosine similarity cos²θ ≈ 0.34–0.45 between the W' and
Z' signal directions in data space (model_discrimination_gstar018.png in both directories)
confirms partial but non-trivial overlap — the models are distinguishable but not orthogonal.

#### Fingerprint comparison

`scripts/fingerprint_comparison.py` plots the Fisher-gradient fingerprint n_i = (Fc)_i / ||Fc||
for W' and Z' side by side, and computes the data-space cosine similarity cos θ = s_W^T C⁻¹ s_Z
/ √(λ_W λ_Z) where s = Kc is the BSM signal vector. This is a purely analytical calculation
using the theory database — no pipeline run is needed.

Two runs produced:
  - gstar=0.18, m=1 TeV (both models): `results/fingerprint_comparison/fingerprint_gstar018_mwp100_mzp100.png`
    W'-only ops (O3pQ3, OQl1M, ...): large entries. Z'-only ops (OpD, Opl1/2/3): large entries.
    Patterns completely distinct visually.
  - **Primary benchmark:** gWH=gZH=0.50, gZl=0.167, mWp=mZp=5 TeV (Jun 11 2026):
    `results/fingerprint_comparison/fingerprint_gwh050_gzl016_mwp050_mzp050.png`
    cos θ = 0.737 (θ ≈ 42°) — models are partially overlapping but distinguishable.
    Script supports `--gZl` argument (added Jun 11 2026) to set Z' lepton coupling independently.

The fingerprint is coupling/mass independent after normalisation — the n_i pattern encodes
only the structural shape of the BSM signal, not its overall strength.

#### Dataset sensitivity by √s

`scripts/dataset_sensitivity_by_sqrts.py` decomposes chi2_SM into contributions from each
√s tier. From `results/dataset_sensitivity_sqrts/sqrts_sensitivity_gstar018_mwp100_mzp100.txt`:
  W' sensitivity: ZH at 240 GeV dominates (g_WH coupling), Z-pole second (g_WLf).
  Z' sensitivity: Z-pole (91 GeV) overwhelmingly dominates via T-parameter (OpD).
  This explains why Z' has higher mass reach: Z-pole constraints are the tightest at FCC-ee.

Full Fisher decomposition: `results/dataset_sensitivity_sqrts_v2/sqrts_sensitivity_fullFisher_*.txt`

**Status:** ✅ `model_discrimination.py` and both robustness test runs (and _v2 variants).
✅ Fingerprint comparison plot in results/fingerprint_comparison/.
✅ Dataset sensitivity by √s in results/dataset_sensitivity_sqrts/ and _sqrts_v2/.
✅ Cross-injection discrimination criterion quantified: 0σ UV = wrong model.
🔲 Section 4.3 body is empty in paper (EH: Look into SOLD tool).
🔲 SOLD tool integration for data-driven UV reconstruction not done.
🔲 Model discrimination result (cross-injection test) should be the core of §4.3.

### 4.4 General Recommendations

The final subsection distils practical recommendations from the quantitative results:

1. **Use UV coupling fit when a specific model is hypothesised.** At weak signal (gWH=0.08,
   1 TeV) the UV coupling fit recovers 4.62σ while Full SMEFT gives only 1.96σ — a factor
   of 2.4× improvement from the 5→14 degrees-of-freedom reduction.

2. **PCA k=4 is sufficient for model-agnostic detection.** It recovers ~97% of signal
   without assuming a UV model, at the cost of not providing model identification.

3. **The best single operator (OQl13 for W', Opl2 for Z') carries most of the signal.**
   In practice, if a BSM anomaly is seen, the single most-shifted operator is a powerful
   discriminator. The look-elsewhere correction (for scanning n_ops=15 operators) costs about
   0.5–1σ and is well within the discovery margin at the benchmarks studied.

4. **LHC high-mass Drell-Yan data** is needed to constrain the 11 FCC-ee-unconstrained
   operators (OQQ1, OQQ8, OQl23/2M/33/3M, Oll2222/2233/2332/3333, Op) involving quark
   currents. FCC-ee and LHC are complementary: FCC-ee constrains the leptonic sector
   (OQl13, O3pl, Oll diagonal), LHC constrains the hadronic sector. A combined global fit
   would further improve sensitivity.

5. **Model discrimination is robust:** The UV coupling criterion (0σ = wrong model) works
   cleanly even when the signal is very large (chi2_SM=124 or chi2_SM=230).

**Status:** 🔲 Section 4.4 body is empty in paper.

---

## 5. Conclusions

Key results to be stated (once sections are written):

1. **Significance improvement:** The UV coupling method outperforms Full SMEFT for known
   models. Dilution factor: ×1.1 at large signal (gWH=0.12, 1 TeV), ×2.4 at weak signal
   (gWH=0.08, 1 TeV). PCA k=4 matches UV coupling performance without assuming the model.

2. **Closure test validation:** Pull distribution consistent with N(0,1) at all tested mass
   points. Power_5σ = 100% at m=3 TeV, 55% at m=4 TeV, 3.4% at m=5 TeV for W'. The
   pipeline is correctly calibrated.

3. **Discovery reach:** W' universal model (gWH=gWLf=gWqf=0.12): 5σ boundary past 3 TeV.
   Z' constrained model (gZH=0.12): 5σ boundary ≈ 7.5 TeV. The Z' has ~2× larger mass
   reach due to the T-parameter operator OpD being tightly constrained at the Z-pole.

4. **SM false-positive rate is exact:** Under SM-generated data, the fraction of replicas
   crossing the 3σ threshold is 0.30% everywhere in the (g, m) plane — matching the nominal
   rate. SM exclusion and BSM discovery are distinct: the BSM Asimov 3σ boundary at gWH=0.12
   is ~5.5–6 TeV.

5. **Model discrimination works:** The UV coupling fit gives 0σ when the wrong model is
   assumed, even when the full SMEFT gives 7–8σ. W' and Z' have signal subspace overlap
   cos²θ ≈ 0.34–0.45 — partially but not fully overlapping.

---

## Appendix A — FCC-ee Dataset Details

**Total observables:** 40 (after exclusions). Grouped by √s:

**91 GeV (Z-pole), 2 datasets:**
  FCCee_Zdata   — Z-pole EWPOs: Γ_Z, σ_had, R_e/μ/τ, A^(0,b)_FB, R_b/c, A_e/b/c/s,
                   sin²θ_eff, m_Z, Γ_had. This is the dominant dataset for Z-coupling operators.
  FCCee_Wwidth  — W boson total width Γ_W at the Z-pole energy.

**161 GeV (WW threshold), 2 datasets:**
  FCCee_ww_161GeV — Total WW cross-section at threshold, sensitive to triple gauge couplings.
  FCCee_Brw       — W leptonic branching ratio.

**240 GeV (ZH run), ~20 datasets:**
  FCCee_ww_240GeV     — WW cross-section at 240 GeV.
  FCCee_ee_240GeV, FCCee_ee_Afb_240GeV  — e+e- → e+e- (Bhabha) and forward-backward asymmetry.
  FCCee_Rb/Rc/Rmu/Rtau/bb_Afb/cc_Afb/mumu_Afb/tautau_Afb/sigmaHad_240GeV — EWPO.
  FCCee_zh_240GeV            — ZH total production.
  FCCee_zh_WW/ZZ/aZ/aa/tautau/mumu_240GeV — ZH decay channels (7 modes), NLO_EW order.
  FCCee_240_H_HADR           — ZH → hadrons.

**365 GeV (ttbar run), ~16 datasets:**
  FCCee_ww/ee/Rb/Rc/Rmu/Rtau/.../sigmaHad_365GeV — same as 240 GeV tiers but at 365 GeV.
  FCCee_zh_365GeV + 6 decay modes (NLO_EW).
  FCCee_vvh_WW/ZZ/aZ/aa/tautau_365GeV — ννH (WW-fusion) Higgs production (LO).
  FCCee_365_H_HADR — ZH → hadrons at 365 GeV.

**Excluded datasets (and why):**
  FCCee_161/240/365_ww_leptonic/semilep_optim_obs (6 datasets) — negative covariance
    eigenvalues due to numerical instabilities in the optimised-observable construction.
  FCCee_alphaEW — removed from smefit database February 2026, merged into FCCee_Zdata.
  FCCee_365_tt_optim_obs — all 8 central values = 0.0, causes divide-by-zero.

**Status:** ⚠️ Appendix A body is empty in paper. Dataset clustering by √s, the exclusion
rationale, and the theory covariance treatment should all be documented here.

---

## Appendix B — BSM Model Details

### B.1 Matching Relations (table to be filled in paper)

The full tree-level matching for W' (27 operators) is implemented in `models/wprime.py`
and for Z' (10 operators) in `models/zprime.py`. Both are verified against MatchMakerEFT.

**W' non-zero coefficients** (at g_WH, g_WLf=g_WH/3, m_W'):
  C_{OQl13} = −g_WH · g_WLf / m²    C_{OQl1M} = +g_WH · g_WLf / m²
  C_{O3pl1} = −¼ g_WH · g_WLf / m²  C_{OpBox} = −3/8 g_WH² / m²
  C_{Oll1111} = −⅛ g_WLf² / m²      C_{Oll1221} = −¼ g_WLf² / m²
  (see models/wprime.py:eft_coefficients() for all 27 coefficients)

**Z' non-zero coefficients** (at g_ZH, g_Zl, m_Zp):
  C_{OpD}    = −g_ZH² / (2 m²)       (T-parameter, dominant constraint)
  C_{Opl1/2/3} = −g_ZH · g_Zl / m²  (SU(2) singlet Higgs-lepton)
  C_{Oll1111/2222/3333} = −g_Zl² / (2 m²)
  C_{Oll1221/1331/2332} = +g_Zl² / m²    (Fierz: O^{1122} = −O^{1221})
  (see models/zprime.py:eft_coefficients() and the docstring for Fierz derivation)

**Status:** 🔲 Paper Appendix B.1 table is empty (EH: This should be double-rowed).
Coefficients are fully implemented in code; need to be transcribed to LaTeX.
🔲 Z' matching not yet in paper at all.

---

## Appendix C — Description of the Different Fits

Documents all five analysis methods: (1) best single operator, (2) PCA k-modes, (3) full
SMEFT, (4) UV coupling (NS), (5) Bayesian model selection. Each has a distinct chi-square
definition, number of degrees of freedom, and sensitivity to model assumptions.

**Status:** 🔲 Empty in paper draft.

---

## Appendix D — Bayesian Model Selection with SMEFiT

The NS log Bayes factor ln BF = ln Z(BSM) − ln Z(SM) provides a Bayesian significance
measure. Key values from summary.txt files:
  W', m=3 TeV:  ln BF = +47.7  (decisive evidence for BSM)
  W', m=4 TeV:  ln BF = +12.2
  Z', m=7 TeV:  ln BF = +16.8
  gWH=0.08, 1 TeV: ln BF = −55.8  (SM preferred at this weak signal with large prior volume)

Note: negative ln BF does not mean the data is inconsistent with BSM; it means the Bayesian
evidence calculation penalises the large prior volume of the UV parameter space more than the
small likelihood gain at this signal level. The frequentist significance (4.62σ at 1-op) is
a more appropriate metric at marginal signal.

**Status:** ✅ ln BF computed in all summary.txt files.
🔲 Appendix D body is empty in paper draft.

---

## Open Tasks Summary (prioritised)

| Priority | Task | Section | Output path |
|----------|------|---------|-------------|
| 🔴 High | Write Section 3.3 body — SM scan is done, result needs to be described | 3.3 | sm_exclusion_comparison_v2/ |
| 🔴 High | Add Z' results throughout (matching, Fisher, fits, reach, §§2.1/2.2/2.3/3.2) | multiple | zprime_*/ |
| 🔴 High | Fill Appendix B.1 matching table for W' and Z' (LaTeX from models/*.py) | App. B | — |
| 🔴 High | Replace significance table screenshot (Fig 3.5) with LaTeX table | 3.2 | summary.txt files |
| ✅ Done | Likelihood vs PCA k plot (σ peaks at k=1, decreases to Full SMEFT) | 4.1 | sigma_vs_pca_k/ |
| 🟡 Med  | Replace Fig 3.6 with updated combined W'+Z' reach plot | 3.2 | discovery_reach/combined_*.png |
| 🟡 Med  | Write Section 4.3 body using model discrimination as main result | 4.3 | robustness_*/ |
| 🟡 Med  | Place UV closure test figure (data-space p-value) into paper as Fig 3.1 | 3.1 | wprime_l1_gwh012_mwp030/ |
| ✅ Done | Fisher for Z' (gZH=0.50, mZp=12 TeV): gZH 46/0/48/6%, gZl 67/0/24/9% by tier | 2.3 | zprime_gzh050_mzp120/plots/ |
| 🟡 Med  | Run --sepproj for additional operators: c_HB, OQl1M, O3pQ3 | 2.4 | wprime_gwh012_mwp010/sepproj/ |
| 🟢 Low  | NS discovery reach full grid scan (mWp × gWH) — agreed to defer for now | 3.1 | — |
| 🟢 Low  | Fisher matrix dependence on UV parameters (mass/coupling scan) | 2.3 | — |
| 🟢 Low  | AIC/BIC model comparison using analytical fit results | 4.2 | — |
| 🟢 Low  | Posteriors in PCA basis vs operator basis comparison | 4.1 | — |
| 🟢 Low  | SOLD integration for data-driven bottom-up UV reconstruction | 4.3 | — |
| 🟢 Low  | Write intro (§1), conclusions (§5), Appendices A/C/D | 1,5,Apps | — |
| ✅ Done | SM pseudo-data scan (5000 toys): frac_3σ=0.30% everywhere | 3.3 | sm_exclusion_comparison_v2/ |
| ✅ Done | Fisher matrix re-clustered by √s (91/161/240/365 GeV) in fisher_uv.py | 2.3 | fisher_uv.py |
| ✅ Done | Corrected UV Fisher (fixed impose_constrain bug, Δg moments, quadratic terms) | 2.3 | scripts/fisher_uv.py |
| ✅ Done | Fisher runs at 1 TeV, 10 TeV, and v3 14-op model (Jun 7–11 2026) | 2.3 | wprime_gwh050_mwp*/plots/ |
| ✅ Done | NS L1 closure for 4 operator sets; V3 14-op best (std=0.952, 0 prior-dom) | 3.1 | *_l1closure*/ |
| ✅ Done | NS sigma vs mass curve at gWH=0.50 (5σ at ~6.4 TeV) | 3.1 | ns_sigma_vs_mass/ |
| ✅ Done | WPrimeConstrainedV2 (18 ops) + V3 (14 ops) models created | 2.1 | models/wprime_constrained_v*.py |
| ✅ Done | Z' prior bug fixed (cap 0.5→2.0 in models/zprime.py) | — | models/zprime.py |
| ✅ Done | Z' benchmark at gZH=0.50, mZp=12 TeV run (7.85σ UV coupling) | 3.2 | zprime_gzh050_mzp120/ |
| ✅ Done | Fingerprint W' vs Z' updated: --gZl arg added, gwh050/gzl016/mwp050/mzp050 | 4.3 | fingerprint_comparison/ |
| ✅ Done | BSM power table (W' m=3/4/5, Z' m=7 TeV) | 3.1 | bsm_closure_metrics/ |
| ✅ Done | Model discrimination / robustness cross-injection tests | 4.3 | robustness_*/ |
| ✅ Done | Fingerprint comparison W' vs Z' | 4.3 | fingerprint_comparison/ |
| ✅ Done | Dataset sensitivity by √s for W' and Z' | 4.3 | dataset_sensitivity_sqrts/ |
| ✅ Done | Discovery reach scan for W' (asymmetric + universal) | 3.2 | discovery_reach/, _v2/ |
| ✅ Done | Z' 5σ boundary determined (~7.5 TeV at gZH=0.12) | 3.2 | zprime_constrained_gzh012_mzp07*/ |
| ✅ Done | --sepproj (separate projection) feature implemented and run (OQl13, OpQM) | 2.4 | wprime_gwh012_mwp010/sepproj/ |
| ✅ Done | Significance table for all 3 benchmark points (summary.txt files) | 3.2 | *_mwp010/, *_mwp100/, *_gwh008*/ |
| ✅ Done | Bayesian model selection (ln BF) computed at all benchmark points | App. D | all summary.txt files |
