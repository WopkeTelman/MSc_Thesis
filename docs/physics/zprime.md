# Z' Boson Model

The Z' is a SU(2)$_L \times$ U(1)$_Y$ singlet (hypercharge-singlet, B'-like) neutral
vector boson.

## Lagrangian

$$\mathcal{L} \supset g_{ZH} \cdot Z'_\mu (H^\dagger i D^\mu H)
+ g_{Zl} \cdot Z'_\mu \sum_\mathrm{gen} (\bar{l}_{L,\mathrm{gen}} \gamma^\mu l_{L,\mathrm{gen}})$$

## UV Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| $g_{ZH}$ | Coupling to Higgs current | 0.12 |
| $g_{Zl}$ | Coupling to lepton doublets | 0.04 |
| $m_{Zp}$ | Z' mass | 1.0 TeV |

## SMEFT Operators Generated

The Z' generates **10 Warsaw-basis operators** at tree level:

| Operator | Matching relation | Role |
|----------|-----------------|------|
| OpD | $-g_{ZH}^2 / (2 m^2)$ | **T-parameter** — dominant constraint |
| Opl1/2/3 | $-g_{ZH} \cdot g_{Zl} / m^2$ | SU(2) singlet Higgs-lepton |
| Oll1111/2222/3333 | $-g_{Zl}^2 / (2 m^2)$ | Same-generation 4-lepton |
| Oll1221/1331/2332 | $+g_{Zl}^2 / m^2$ | Cross-generation 4-lepton (Fierz) |

## Why Z' has larger mass reach than W'

The T-parameter operator **OpD** is exquisitely constrained at the Z-pole, giving Z' roughly
**twice the mass reach** of W' at the same coupling strength. The Z-pole measurements at 91 GeV
directly probe OpD via its effect on the $\rho$-parameter.

!!! success "Z' vs W' structural difference"
    - **W'** → SU(2) triplet operators (charged current, Higgs-box)
    - **Z'** → SU(2) singlet T-parameter (OpD) and singlet Higgs-lepton (Opl)

    These patterns are *completely distinct* — enabling clean model discrimination.

## Discovery Reach

- **Z' constrained** at $g_{ZH}=0.12$, $g_{Zl}=0.04$: 5σ boundary ≈ **7.5 TeV**
- **Z' primary benchmark** ($g_{ZH}=0.50$, $m_{Zp}=12$ TeV): UV coupling = **7.85σ**

## Code

| File | Class |
|------|-------|
| `models/zprime.py` | `ZPrimeModel` |
| `models/zprime_constrained.py` | `ZPrimeConstrainedModel` |
