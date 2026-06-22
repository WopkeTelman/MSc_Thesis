# Model Discrimination

Can one tell, from a BSM anomaly in the SMEFT fit, *which* UV completion is responsible?
The UV coupling method answers this directly: if the **wrong** UV model is assumed, its
constrained fit gives ~0σ even when the correct model gives high significance.

## Cross-Injection Tests

Script: `scripts/model_discrimination.py`

### Test A — W' data, Z' fit

Pseudo-data injected under W' ($g_{WH}=0.12$, $m_{Wp}=1$ TeV), fit with Z' model.

| Method | Significance |
|--------|-------------|
| Full SMEFT | 7.30σ — BSM detected ✓ |
| UV coupling (Z' model) | **0.00σ** — wrong model ✗ |

The Z' pattern (T-parameter dominant) cannot describe W' data (charged current dominant).

### Test B — Z' data, W' fit

Pseudo-data injected under Z' ($g_{ZH}=0.12$, $m_{Zp}=1$ TeV), fit with W' model.

| Method | Significance |
|--------|-------------|
| Full SMEFT | 8.46σ — BSM detected ✓ |
| UV coupling (W' model) | **0.00σ** — wrong model ✗ |

## Discrimination Criterion

!!! tip "Rule of thumb"
    - UV coupling = 0σ **and** SMEFT free > 5σ → assumed UV completion is **incorrect**
    - Both > 5σ → assumed model is a **plausible** description of the data

## Signal Subspace Overlap

The cosine similarity between the W' and Z' signal directions in data space:

$$\cos^2\theta \approx 0.34\text{–}0.45$$

at $g^* = 0.18$, $m=1$ TeV for both models. The models are distinguishable but not orthogonal.

At the primary benchmarks ($g_{WH}=0.50$, $g_{Zl}=0.167$, $m=5$ TeV):
$$\cos\theta = 0.737 \quad (\theta \approx 42°)$$

## Fingerprint Comparison

Script: `scripts/fingerprint_comparison.py`

The Fisher-gradient fingerprint $n_i = (Fc)_i / \|Fc\|$ shows which observables each model
is most sensitive to:

- **W'** → large entries in O3pQ3, OQl1M, ... (charged-current operators)
- **Z'** → large entries in OpD, Opl1/2/3 (T-parameter, singlet Higgs-lepton)

The patterns are **completely distinct** visually, making the two models easily separable
even without the UV coupling fit.

Output: `results/fingerprint_comparison/fingerprint_gwh050_gzl016_mwp050_mzp050.png`
