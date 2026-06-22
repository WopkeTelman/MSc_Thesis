# W' Boson Model

The W' is an SU(2)$_L$ triplet vector field $W'^a_\mu$ with mass $m_{W'}$ and
coupling to the SM left-handed fermion doublets and the Higgs doublet.

## Lagrangian

$$\mathcal{L} \supset g_{WH} \cdot W'^a_\mu (H^\dagger \tau^a i D^\mu H)
+ g_{WLf} \cdot W'^a_\mu (\bar{l}_L \gamma^\mu \tau^a l_L) + \ldots$$

## UV Parameters

| Parameter | Description | Default value |
|-----------|-------------|---------------|
| $g_{WH}$ | Coupling to Higgs current | 0.12 |
| $g_{WLf11}$ | Coupling to 1st-gen lepton doublet | 0.04 |
| $g_{WLf22}$ | Coupling to 2nd-gen lepton doublet | 0.04 |
| $g_{WLf33}$ | Coupling to 3rd-gen lepton doublet | 0.04 |
| $g_{Wqf33}$ | Coupling to 3rd-gen quark doublet | 0.04 |

## Coupling Scenarios

- **Asymmetric (primary):** $g_{WH}$ free, $g_{WLf11/22/33} = g_{WLf}$, $g_{Wqf33} = g_{WLf}$. Default: $g_{WH}=0.12$, $g_{WLf}=0.04$.
- **Constrained:** $g_{WLf} = g_{Wqf} = g_{WH}/3$ enforced. Implemented in `models/wprime_constrained.py`.
- **Universal (allg012):** $g_{WH} = g_{WLf} = g_{Wqf}$. Much stronger leptonic coupling → 3× larger signal.

## SMEFT Operators Generated

The W' generates **27 Warsaw-basis operators** at tree level. The dominant signal carriers:

| Operator | Matching relation | Role |
|----------|-----------------|------|
| OQl13 | $-g_{WH} \cdot g_{WLf11} / m^2$ | Dominant — largest Wilson coefficient |
| OQl1M | $+g_{WH} \cdot g_{WLf11} / m^2$ | Charged current |
| O3pl1/2/3 | $-\frac{1}{4} g_{WH} \cdot g_{WLf} / m^2$ | Higgs-lepton triplet |
| OpBox | $-\frac{3}{8} g_{WH}^2 / m^2$ | Higgs-box |

!!! warning "FCC-ee blind spots"
    11 of the 27 operators are **unconstrained** by FCC-ee data alone (OQQ1, OQQ8, OQl23/2M/33/3M, Oll2222/2233/2332/3333, Op) because they involve quark currents invisible to $e^+e^-$ colliders. LHC Drell-Yan data would constrain these.

## Model Variants in Code

| File | Class | Description |
|------|-------|-------------|
| `models/wprime.py` | `WPrimeModel` | Full asymmetric, 27 operators |
| `models/wprime_constrained.py` | `WPrimeConstrainedModel` | $g_{WLf}=g_{Wqf}=g_{WH}/3$, 16 ops |
| `models/wprime_constrained_v3.py` | `WPrimeConstrainedV3Model` | 14 empirically constrained ops (best) |

## Discovery Reach

- **Asymmetric W'** at $g_{WH}=0.12$: 5σ boundary ≈ **1.4 TeV**
- **Universal W'** at $g_{WH}=g_{WLf}=g_{Wqf}=0.12$: 5σ boundary > **3 TeV**
- **NS significance** at $g_{WH}=0.50$: 5σ crossing at ~**6.4 TeV**
