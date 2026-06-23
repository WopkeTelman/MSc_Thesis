# Background & Motivation

The Standard Model of particle physics is not expected to be the final theory of nature.
Numerous extensions predict new heavy particles — W' bosons, Z' bosons, leptoquarks,
supersymmetric partners — whose masses may lie far above the direct production threshold
at any foreseeable collider. Their effects can nevertheless manifest at lower energies as
small but correlated deviations in precisely measured SM observables.

## The SMEFT Framework

The Standard Model Effective Field Theory (SMEFT) provides a systematic framework:
BSM physics at a high scale Λ is captured by a tower of higher-dimensional operators
added to the SM Lagrangian:

$$\mathcal{L}_{\mathrm{eff}} = \mathcal{L}_{\mathrm{SM}} + \sum_i \frac{C_i}{\Lambda^2} \mathcal{O}_i$$

where $C_i$ are the Wilson coefficients and $\mathcal{O}_i$ are dimension-6 operators in the Warsaw basis.

## The FCC-ee

The Future Circular Collider in electron-positron mode (FCC-ee) is designed to measure
a very large set of electroweak, Higgs, and top observables with unprecedented precision
at $\sqrt{s}$ = 91, 161, 240, and 365 GeV. The global SMEFT fit to all these observables
constrains the Wilson coefficients simultaneously.

## The Central Question

!!! question "This thesis"
    Can a global SMEFT fit at the FCC-ee not only **detect** a BSM anomaly but also
    *discover* a specific UV completion and *discriminate* between competing models?

The paper introduces a quantitative BSM closure test, constructs several significance
metrics for UV model discovery, and demonstrates the methodology on two benchmark models:
a **W' boson** and a **Z' boson**.

## Key Idea: UV Coupling Fit

Instead of fitting all Wilson coefficients freely (~15 free parameters), we fit directly
in the UV parameter space of a specific BSM model (5 parameters for W', 2 for Z').
This enforces relations between coefficients via tree-level matching, dramatically
reducing the effective degrees of freedom and improving the signal-to-noise ratio.

| Method | Free params | Significance (g=0.12, m=1 TeV) |
|--------|-------------|-------------------------------|
| Full SMEFT | 14 | 9.1σ |
| PCA k=4 | 4 | 10.2σ |
| UV coupling | 5 | > 10σ |
| Full SMEFT (weak signal, g=0.08) | 14 | 1.96σ |
| UV coupling (weak signal, g=0.08) | 5 | 4.62σ |

At weak signal the UV coupling method gives **2.4× higher significance** than the full SMEFT.
