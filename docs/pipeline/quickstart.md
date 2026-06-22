# Quick Start

## Setup

```bash
conda activate smefit-dev
cd /data/theorie/wtelman/wprime_explicit_uv_closure_vs_sm/fullpipeline/scripts
```

## Run the pipeline

=== "W' boson"

    ```bash
    python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0
    ```

=== "Z' boson"

    ```bash
    python run_pipeline.py --model zprime --gZH 0.12 --gZl 0.04 --mZp 1.0
    ```

=== "Fast (no NS)"

    ```bash
    python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0 --no-ns
    ```

=== "Discovery scan"

    ```bash
    python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0 --scan
    ```

## What the pipeline produces

```
results/<tag>/
├── summary.txt                  ← significance table (read this first)
├── plots/significance_summary.png
├── fits/                        ← smefit fit outputs
├── pca/                         ← Fisher eigenvectors
└── reports/Report_<tag>/        ← posterior histograms, chi2 table
```

## Interpreting significance

| σ | Conclusion |
|---|------------|
| < 3 | No evidence |
| 3 – 5 | Evidence (not discovery) |
| ≥ 5 | **5σ discovery** |

## Common flags

| Flag | Effect |
|------|--------|
| `--no-ns` | Skip nested sampling (~15 min saved) |
| `--skip-existing` | Resume interrupted run |
| `--scan` | Grid scan over (coupling, mass) |
| `--tag <name>` | Custom output directory name |
| `--sepproj OQl13` | Run individual operator projections |
