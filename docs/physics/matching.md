# SMEFT Matching & FCC-ee Observables

## Theory Prediction

The theory prediction for observable $i$ at Wilson coefficients $c$ is:

$$\sigma_i(c) = \sigma_i(\mathrm{SM}) + K_{ij} c_j + K^{(2)}_{i,ab} c_a c_b$$

where $K_{ij}$ are the linear theory coefficients ($\partial\sigma/\partial c$ at $c=0$)
and $K^{(2)}$ are the quadratic corrections. The chi-square is:

$$\chi^2(c) = \sum_{ij} [\sigma_i(c) - d_i] (C^{-1})_{ij} [\sigma_j(c) - d_j]$$

## Fisher Information Matrix

The Fisher information matrix quantifies sensitivity before any fit:

$$F_{ij} = K^T C^{-1} K$$

Its eigenvectors (PCA) give the directions of maximum sensitivity in Wilson coefficient space.
For the UV parameter space the Fisher is contracted with the matching Jacobian:

$$\mathcal{I}_{\ell\ell'} = \sum_{ij} \frac{\partial\sigma_i}{\partial g^\ell} (C^{-1})_{ij} \frac{\partial\sigma_j}{\partial g^{\ell'}}$$

!!! warning "Corrected UV Fisher"
    The smefit package contains a bug in `impose_constrain`: it evaluates the Jacobian
    by setting one UV parameter to 1 and all others to 0, which zeroes all bilinear
    matching terms (e.g. $C_{OQl13} \propto g_{WH} \cdot g_{WLf}$).
    The corrected implementation is in `scripts/fisher_uv.py`, which evaluates the
    Jacobian via finite differences at the UV truth point.

## FCC-ee Dataset Summary

**40 observables** across four energy tiers:

| √s | Datasets | Physics |
|----|----------|---------|
| 91 GeV | FCCee_Zdata, FCCee_Wwidth | Z-pole EWPOs: Γ_Z, σ_had, R_l, A_FB |
| 161 GeV | FCCee_ww_161GeV, FCCee_Brw | WW threshold, W branching ratio |
| 240 GeV | FCCee_ww/zh/ee/Rb/Rc/... | ZH Higgs + EWPO (~20 datasets) |
| 365 GeV | FCCee_ww/zh/vvh/ee/Rb/... | High-energy EWPO + ZH + WW-fusion (~16 datasets) |

## RGE Settings

All fits use SMEFT RGE from the matching scale down to the observable scale:

```yaml
init_scale: 1000.0 GeV   # matching scale
obs_scale:  dynamic       # run to each dataset's energy
smeft_accuracy: integrate # exact numerical integration
yukawa: top               # include top Yukawa
```
