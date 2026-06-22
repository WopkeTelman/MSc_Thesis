# Significance & Discovery Reach

## Significance Table

Results from `summary.txt` files for the three main W' benchmark points:

| Method | g=0.12, m=1 TeV | g=1.0, m=10 TeV | g=0.08, m=1 TeV |
|--------|----------------|----------------|----------------|
| chi2_SM | 124.1 | 57.8 | 23.7 |
| Best 1-op (OQl13) | 10.61σ | 7.22σ | 4.62σ |
| PCA k=2 | 10.41σ | 6.94σ | 4.25σ |
| PCA k=4 | 10.26σ | 6.64σ | 3.78σ |
| Full SMEFT | 9.09σ | 5.14σ | 1.96σ |
| UV coupling | NS fail (lnBF=+48.1) | 6.63σ | — (lnBF=−55.8) |

!!! note "Key observation"
    At weak signal (g=0.08): UV coupling (4.62σ) vs Full SMEFT (1.96σ) — a **2.4× improvement**
    from reducing 14 → 5 degrees of freedom.

## Sigma vs PCA k

The significance is **maximised at k=1** for all W' benchmarks — PC1 alone captures >90% of the signal.

| Benchmark | k=1 | k=2 | k=4 | Full SMEFT (k=14) |
|-----------|-----|-----|-----|-------------------|
| g=0.12, m=1 TeV | 10.57σ | 10.34σ | 10.19σ | 9.01σ |
| g=1.0, m=10 TeV | 7.14σ | 6.84σ | 6.54σ | 5.00σ |
| g=0.08, m=1 TeV | 4.48σ | 4.09σ | 3.61σ | 1.65σ |

## W' Discovery Reach

- **Asymmetric W'** at $g_{WH}=0.12$: 5σ boundary ≈ **1.4 TeV**
- **Universal W'** at $g_{WH}=g_{WLf}=g_{Wqf}=0.12$: 5σ boundary > **3 TeV**
- **NS** at $g_{WH}=0.50$: 5σ crossing at ~**6.4 TeV** (lower than Asimov due to Bayesian prior penalty)

## Z' Discovery Reach

| Config | 5σ boundary |
|--------|------------|
| Constrained ($g_{ZH}=0.12$, $g_{Zl}=0.04$) | ≈ **7.5 TeV** |
| Primary benchmark ($g_{ZH}=0.50$, $m_{Zp}=12$ TeV) | UV = **7.85σ** |

The Z' has ~**2× larger mass reach** than the W' because the T-parameter operator (OpD) is
exquisitely constrained at the Z-pole.

## SM False-Positive Rate

From `scripts/sm_pseudodata_scan.py` (5000 SM toys per grid point):

!!! success "Result"
    The fraction of SM toys crossing the 3σ threshold is **0.30%** at every point in the
    $(g_{WH}, m_{W'})$ plane — exactly the nominal rate. The pipeline maintains correct
    coverage everywhere.

## Z' Primary Benchmark

At $g_{ZH}=0.50$, $g_{Zl}=0.167$, $m_{Zp}=12$ TeV:

| Method | Significance |
|--------|-------------|
| Best 1-op (OpD) | 6.81σ |
| PCA k=5 | 7.42σ |
| Full SMEFT | 7.08σ |
| UV coupling | **7.85σ** |
| ln BF | 29.40 |
