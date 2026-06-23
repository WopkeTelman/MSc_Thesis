# Scripts & Runcards

Complete reference of all analysis scripts and model files.

## Main Entry Point

### `scripts/run_pipeline.py`

The master pipeline script. Runs all steps for a given model and benchmark from a single command.

```bash
# Standard run (all steps)
python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0

# Skip nested sampling (~15 min saved)
python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0 --no-ns

# Resume an interrupted run
python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0 --skip-existing

# Discovery reach scan over (coupling × mass) grid
python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0 --scan

# Custom output tag
python run_pipeline.py --model zprime --gZH 0.12 --gZl 0.04 --mZp 7.5 --tag zprime_5sigma

# Per-operator projections
python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0 --sepproj OQl13 OpQM
```

**Model flags:**

| Model | Flags |
|-------|-------|
| `wprime` | `--gWH`, `--gWLf`, `--gWqf`, `--mWp` |
| `zprime` | `--gZH`, `--gZl`, `--mZp` |

---

## Analysis Scripts

### Significance & Discovery Reach

| Script | Description |
|--------|-------------|
| `run_discovery_reach.py` | Grid scan over (coupling, mass); produces discovery region heatmap |
| `run_exclusion_reach.py` | Exclusion reach scan |
| `run_scan_validated.py` | Validated scan using analytic profile likelihood |
| `run_scan_l1_point.py` | L1 scan at a single parameter point |
| `run_uv_validated.py` | UV coupling significance table across mass points |
| `sm_pseudodata_scan.py` | 5000 SM toy datasets per grid point — measures false-positive rate |
| `sm_replica_analysis.py` | SM toy analysis |

### Fisher Information

| Script | Description |
|--------|-------------|
| `fisher_uv.py` | Corrected UV Fisher matrix (fixes `impose_constrain` bug in smefit); groups by √s tier |
| `replot_fisher_heatmap.py` | Re-render Fisher heatmap from saved data |
| `plot_fisher_heatmap_custom.py` | Custom Fisher heatmap styling |

### Closure Tests

| Script | Description |
|--------|-------------|
| `ns_l1_closure.py` | 50 NS replicas with L1 pseudo-data; UV and EFT pull distributions; supports `--smeft` flag |
| `run_l1_replica.py` | Single L1 replica NS run |
| `fold_uv_samples.py` | Fold UV posterior samples for pull computation |
| `data_pvalue_metric.py` | Metric 1: data-space p-value distribution (KS test) |
| `l1_pull_distribution.py` | Metric 2b: coefficient-space pull distribution (analytic, 1000 replicas) |
| `l1_pull_distribution_v2.py` | Updated version of Metric 2b |
| `gen_ns_eft_pulls.py` | Metric 2c: NS posterior-based EFT pull distribution |
| `bsm_closure_metrics.py` | Power table across multiple mass points |
| `metric1_delta_chi2.py` | Delta chi² metric computation |
| `metric2_wilson_pulls.py` | Wilson coefficient pull metric |

### Model Discrimination

| Script | Description |
|--------|-------------|
| `model_discrimination.py` | Cross-injection tests (W' data → Z' fit and vice versa) |
| `fingerprint_comparison.py` | Fisher-gradient fingerprint of W' vs Z' side by side |
| `dataset_sensitivity_by_sqrts.py` | Decomposes chi²_SM by √s tier (91/161/240/365 GeV) |

### PCA

| Script | Description |
|--------|-------------|
| `pca_table.py` | PCA component loadings table |
| `regen_pca.py` | Recompute PCA from saved K matrix |

### Plotting

| Script | Description |
|--------|-------------|
| `plot_sigma_vs_pca_k.py` | σ vs PCA k curve — answers "at which k does sigma peak?" |
| `plot_discovery_region.py` | Discovery region heatmap in (coupling, mass) plane |
| `plot_wprime_allg012_discovery_extended.py` | Extended W' universal model reach plot |
| `plot_zprime_discovery_extended.py` | Extended Z' reach plot |
| `combined_reach_plot.py` | Combined W' + Z' discovery reach overlay |
| `plot_closure_contour.py` | Closure test contour plot |
| `closure_contour.py` | Closure contour helper |
| `plot_sigma_contour.py` | Significance contour in parameter space |
| `plot_scan_l1_from_table.py` | Plot L1 scan results from table |
| `plot_total_fit_pull.py` | Total fit pull distribution plot |
| `plot_pca_marginals.py` | PCA marginal distributions |
| `replot_discovery.py` | Re-render discovery plot from saved table |
| `replot_discovery_clean.py` | Clean version of discovery plot |
| `pull_distribution.py` | Pull distribution plotting utility |
| `test_significance.py` | Significance test utility |

### Projections

| Script | Description |
|--------|-------------|
| `generate_projections.py` | Generate BSM pseudo-data projections for all datasets |
| `run_ns_discovery.py` | NS-based discovery run |
| `setup_l1_gwh050_mwp120.py` | Setup script for specific L1 benchmark |
| `wprime_l1_analysis.py` | W' L1 analysis helper |

---

## Model Files

All models live in `models/`. Each implements `eft_coefficients()`, `uv_coeff_block()`, `uv_param_names()`, and `uv_truth()`.

| File | Class | Description |
|------|-------|-------------|
| `models/wprime.py` | `WPrimeModel` | Full asymmetric W', 27 Warsaw operators, 5 UV params |
| `models/wprime_constrained.py` | `WPrimeConstrainedModel` | $g_{WLf}=g_{Wqf}=g_{WH}/3$, 16 operators |
| `models/wprime_constrained_v2.py` | `WPrimeConstrainedV2Model` | 18 empirically constrained operators |
| `models/wprime_constrained_v3.py` | `WPrimeConstrainedV3Model` | **14 operators — best free-EFT calibration** (std=0.952, 0 prior-dominated) |
| `models/wprime_1g.py` | — | W' single-coupling variant |
| `models/wprime_universal.py` | — | Universal W': $g_{WH}=g_{WLf}=g_{Wqf}$ |
| `models/zprime.py` | `ZPrimeModel` | Z' hypercharge-singlet, 10 operators, 2 UV params |
| `models/zprime_constrained.py` | `ZPrimeConstrainedModel` | Z' with $g_{Zl}=g_{ZH}/3$ |
| `models/comphiggs.py` | — | Composite Higgs model |

---

## Output Structure

Every run produces output under `results/<tag>/`:

```
results/<tag>/
├── summary.txt                   ← significance table (read this first)
├── plots/
│   └── significance_summary.png
├── fits/
│   ├── <tag>_SMonly/             ← SM-only baseline fit
│   ├── <tag>_BSMclosure_SMEFT/  ← full SMEFT fit on BSM data
│   └── <tag>_UVcoup/            ← UV coupling NS fit
├── pca/
│   ├── eigenvalues.npy
│   └── eigenvectors.npy
└── reports/Report_<tag>/
    └── meta/
        ├── coefficient_histo.png
        └── chi2_table.*
```
