# Discovering New Physics at the FCC-ee with the SMEFT

**MSc Thesis — Wopke Telman (Physics and Astronomy, University of Amsterdam)**

> Can a global SMEFT fit at the FCC-ee not only detect a BSM anomaly but also *discover* a specific UV completion and *discriminate* between competing models?

---

## What this project does

This repository contains the full analysis pipeline for my master's thesis, which studies how the [Future Circular Collider in electron-positron mode (FCC-ee)](https://fcc.web.cern.ch) can be used to discover new heavy particles — specifically a **W' boson** and a **Z' boson** — through precision measurements interpreted within the Standard Model Effective Field Theory (SMEFT).

The key idea: instead of fitting all SMEFT Wilson coefficients freely (which dilutes the signal), we fit directly in the UV parameter space of a specific BSM model. This *UV coupling method* dramatically improves discovery sensitivity by reducing the effective degrees of freedom from ~15 to 5.

## Key results

| Result | Value |
|--------|-------|
| W' 5σ discovery reach (universal, g=0.12) | > 3 TeV |
| Z' 5σ discovery reach (g_ZH=0.12) | ≈ 7.5 TeV |
| Significance improvement (UV vs full SMEFT, weak signal) | ×2.4 |
| SM false-positive rate at 3σ | 0.30% (nominal) at all tested (g, m) |
| UV closure pull std | 0.992 — consistent with N(0,1) |

## Repository structure

```
├── models/           UV model definitions (W', Z', constrained variants)
├── scripts/          Analysis scripts (pipeline, Fisher, closure tests, plots)
├── PIPELINE.md       Step-by-step pipeline usage guide
└── PROJECT_OVERVIEW.md  Full project status and results reference
```

## Quick start

```bash
conda activate smefit-dev
cd scripts/

# W' boson at g_WH=0.12, m=1 TeV
python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0

# Z' boson at g_ZH=0.12, g_Zl=0.04, m=1 TeV
python run_pipeline.py --model zprime --gZH 0.12 --gZl 0.04 --mZp 1.0

# Fast run (skip nested sampling)
python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0 --no-ns

# Discovery reach scan over (coupling, mass) grid
python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0 --scan
```

See [PIPELINE.md](PIPELINE.md) for the full guide and [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for the complete results reference.

## Physics background

The SMEFT extends the SM Lagrangian with higher-dimensional operators:

$$\mathcal{L}_\mathrm{eff} = \mathcal{L}_\mathrm{SM} + \sum_i \frac{C_i}{\Lambda^2} \mathcal{O}_i$$

New heavy particles at scale Λ imprint correlated deviations in the Wilson coefficients C_i. The FCC-ee will measure ~40 electroweak, Higgs, and WW observables with unprecedented precision at √s = 91, 161, 240, and 365 GeV. A global SMEFT fit to all these observables simultaneously constrains the C_i and can reveal which BSM models are compatible with the data.

The two benchmark models studied are:

- **W' boson** — SU(2)_L triplet, generates 27 Warsaw-basis operators. Primary operators: OQl13 (charged-current quark-lepton), O3pl (Higgs-lepton triplet), OpBox.
- **Z' boson** — U(1) singlet (B'-like), generates 10 operators. Primary operator: OpD (T-parameter), which is extremely tightly constrained at the Z-pole — giving Z' roughly 2× the mass reach of W'.

## Supervisor

[Juan Rojo](https://www.jrojo.info), VU Amsterdam / Nikhef
