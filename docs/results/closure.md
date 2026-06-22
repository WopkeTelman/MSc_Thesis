# Closure Tests

The BSM closure test validates the pipeline by checking that the fit correctly recovers
the injected BSM signal. Following the PDF closure-test methodology used in SMEFiT.

## Pseudo-data Generation

- **Level-0 (L0):** $d_i^{(L0)} = \sigma_i(c_\mathrm{truth})$ — noiseless, tests central-value recovery
- **Level-1 (L1):** $d_i^{(L1,r)} = \sigma_i(c_\mathrm{truth}) + \sum_j L_{ij} \xi^r_j$ — one noise realisation per replica

## Metric 1 — Data-space p-value (KS test)

Script: `scripts/data_pvalue_metric.py`

For each L1 BSM replica, compute $q^{(r)} = \chi^2_\mathrm{SM} - \chi^2_\mathrm{min}(r)$.
Under the BSM hypothesis the SM null p-value distribution should be **uniform on [0,1]**.

| Method | KS test p-value | Pass? |
|--------|----------------|-------|
| NS UV coupling (50 replicas) | 0.734 | ✓ |
| Analytic UV coupling (1000 replicas) | 0.416 | ✓ |
| Analytic SMEFT free (1000 replicas) | 0.023 | ✗ |

## Metric 2b — Coefficient-space pull (analytic)

Script: `scripts/l1_pull_distribution.py`

Pull $P_i = (c^\mathrm{fit}_i - c^\mathrm{truth}_i) / \sigma^\mathrm{fit}_i$ over 1000 replicas.
For a correctly calibrated fit: $\langle P_i \rangle \approx 0$, $\mathrm{std}(P_i) \approx 1$.

**Result at g=0.12, m=3 TeV:** mean = −0.0008, std = 1.0042 ✓

## NS L1 Closure — Operator Set Comparison

At $g_{WH}=0.50$, $m_{Wp}=5$ TeV, 50 NS replicas:

| Model | Ops | Free-EFT std | Prior-dominated ops |
|-------|-----|-------------|---------------------|
| UV coupling (truth) | 5 | **0.992** | 0 |
| Full SMEFT (27 ops) | 27 | 0.807 | 9 |
| WPrimeConstrained (16) | 16 | 0.960 | 2 |
| WPrimeConstrainedV3 (14) | 14 | 0.952 | 0 |

!!! success "Best result"
    The UV coupling fit achieves the best-calibrated posteriors (std=0.992, 0 prior-dominated parameters). This is the primary justification for using UV coupling as the discovery metric.

## BSM Power Table

| Model | λ_UV | Asimov σ | power_5σ |
|-------|------|----------|----------|
| W', m=3 TeV | 107.68 | 9.55σ | 100.0% |
| W', m=4 TeV | 34.07 | 4.73σ | 55.0% |
| W', m=5 TeV | 13.96 | 2.41σ | 3.4% |
| Z', m=7 TeV | 38.87 | 5.90σ | 82.6% |

Power drops below 50% (5σ threshold) between 4 and 5 TeV for the W'.
