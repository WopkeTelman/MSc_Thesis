# FCC-ee BSM Discovery Pipeline

**Script:** `fullpipeline/scripts/run_pipeline.py`

A fully self-contained pipeline that takes a UV model (W' or Z' boson), generates
BSM pseudo-data for FCC-ee, runs SMEFT fits, computes discovery significance, and
produces a full smefit report — all from a single command.

---

## What you need to run it

### 1. Conda environment

```bash
conda activate smefit-dev
```

The `smefit-dev` environment must have `smefit`, `numpy`, `scipy`, `matplotlib`,
`pyyaml`, and `jax` installed.

### 2. The smefit database

Located at `/data/theorie/wtelman/smefit_database/`. It provides:
- `commondata_projections_L0/` — SM pseudo-data central values and uncertainties for all FCC-ee datasets
- `theory/` — LO theory matrix files (`FCCee_*.json`) containing `K_j[i]` (linearised EFT coefficients per observable per operator)

This database is already present on the cluster and does **not** need to be modified.

### 3. Nothing else — model selection is done on the command line

There is **no YAML file to prepare**. You select which model to study by passing
`--model wprime` or `--model zprime` on the command line, together with the
physical parameter values:

```bash
# Study a Z' with gZH=0.12, gZl=0.04, mZp=1 TeV
python run_pipeline.py --model zprime --gZH 0.12 --gZl 0.04 --mZp 1.0

# Study a W' with gWH=0.12, mWp=1 TeV
python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0
```

The script reads `--model` and instantiates the corresponding Python class:

| `--model` | Python class | Parameter flags |
|---|---|---|
| `zprime` | `ZPrimeModel(gZH, gZl, mZp)` | `--gZH`, `--gZl`, `--mZp` |
| `wprime` | `WPrimeModel(gWH, gWLf, gWqf, mWp)` | `--gWH`, `--gWLf`, `--gWqf`, `--mWp` |

That model object carries all the physics: which Warsaw-basis operators are
generated at tree level, what the Wilson coefficients are at the matching scale,
and how they are expressed in terms of UV couplings for the NS fit. All smefit
runcards are then built automatically inside the script from that object.
No files need to be written or edited by hand.

Each model lives in `fullpipeline/models/` as a plain Python dataclass:

| File | Class | Parameters |
|---|---|---|
| `models/wprime.py` | `WPrimeModel` | `gWH`, `gWLf11/22/33`, `gWqf33`, `mWp` |
| `models/zprime.py` | `ZPrimeModel` | `gZH`, `gZl`, `mZp` |

A model class must implement:
- `OPERATORS` — list of Warsaw-basis operator names generated at tree level
- `eft_coefficients()` — returns `{op: value}` dict of Wilson coefficients at the matching scale (units: TeV⁻²)
- `uv_coeff_block()` — returns the smefit `coefficients` block for the NS runcard, expressing each EFT operator as a function of UV couplings
- `uv_param_names()` — list of free UV parameters, e.g. `["gZH", "gZl"]`
- `uv_truth()` — dict of injected UV parameter values (for closure)

---

## How to run

Navigate to the `fullpipeline/scripts/` directory (or run from anywhere with the full path):

```bash
cd /data/theorie/wtelman/wprime_explicit_uv_closure_vs_sm/fullpipeline/scripts
conda activate smefit-dev
```

### Standard single-point run (full pipeline including NS)

```bash
# Z' boson
python run_pipeline.py --model zprime --gZH 0.12 --gZl 0.04 --mZp 1.0

# W' boson
python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0
```

This runs all 7 steps.

### Without the NS fit (fast)

```bash
python run_pipeline.py --model zprime --gZH 0.12 --gZl 0.04 --mZp 1.0 --no-ns
```

Skips Step 3 (nested sampling). All analytic significance estimates, PCA,
and the smefit report still run. Use this for quick parameter exploration.

### Resume an interrupted run

```bash
python run_pipeline.py --model zprime --gZH 0.12 --gZl 0.04 --mZp 1.0 --skip-existing
```

Skips any fit step that already has a `fit_results.json`. Useful if the NS
fit completed but the pipeline was interrupted before Step 5/6.

### Discovery scan over (coupling, mass) grid

```bash
python run_pipeline.py --model zprime --gZH 0.12 --gZl 0.04 --mZp 1.0 --scan
python run_pipeline.py --model wprime --gWH 0.12 --mWp 1.0 --scan
```

Loops over a 2D grid of coupling × mass values using only the analytic
profile likelihood (no NS, fast). Produces a discovery region heatmap.
Default grids: coupling ∈ {0.05, 0.07, 0.09, 0.11, 0.12, 0.15},
mass ∈ {0.5, 1.0, 1.5, 2.0, 3.0} TeV.

### Custom run tag

```bash
python run_pipeline.py --model zprime --gZH 0.08 --gZl 0.03 --mZp 2.0 --tag zprime_weak_highm
```

All outputs go under `results/zprime_weak_highm/`. If `--tag` is omitted,
the tag is auto-generated from the parameter values, e.g.
`zprime_gzh012_mzp010` for gZH=0.12, mZp=1.0 TeV.

---

## What the pipeline does, step by step

### Step 0 — Generate BSM pseudo-data

**Method:** Injects a BSM signal into SM central values using the RGE-evolved
theory matrix from the smefit database:

```
data_BSM[i] = data_SM[i] + Σ_j  K_eff_j[i] · c_j
```

where `K_eff_j[i]` is the LO theory coefficient for operator j in observable i,
evolved by SMEFT RGE from the matching scale (1 TeV) to the observable scale
(e.g. 91 GeV for Z-pole, 240 GeV for WW threshold). The RGE evolution is
performed by loading smefit's `load_datasets` with the same RGE configuration
(`init_scale: 1000 GeV, obs_scale: dynamic`) as the fits — ensuring the
pseudo-data is exactly self-consistent with smefit's internal predictions.

> Without this RGE consistency, OpD (which runs ~13% from 1 TeV to 91 GeV)
> would show a ~1σ artificial offset in the closure histograms.

**Output:** `projections/<tag>/*.yaml` — BSM central values for all 35 FCC-ee datasets.

Skipped automatically if the directory is non-empty. To regenerate, delete
`projections/<tag>/`.

---

### Step 1 — SM-only baseline fit (analytic)

**What it tests:** What would smefit find if the data were pure SM?

Runs smefit in analytic mode (`smefit A`) on the **SM data** (not the BSM
projections) with flat priors `[-1, 1]` on all operators. The posterior is
centred at zero — this is the null hypothesis.

**Why:** Provides the reference posterior against which the BSM fit is compared
in the report histograms. If you see the SM-only posterior (blue) and the
SMEFT posterior (orange) clearly separated, that is a signature of a BSM
signal.

**Output:** `results/<tag>/fits/<tag>_SMonly/fit_results.json`

---

### Step 2 — Full SMEFT fit (analytic)

**What it tests:** Can smefit recover the injected BSM signal treating all
operators as independent?

Runs smefit in analytic mode on the **BSM pseudo-data** with priors set to
±20× the injected truth value for each operator (minimum ±0.01, maximum ±1.0).
This ensures the posterior is well-resolved around the true value without
being artificially narrow.

**Why:** This is the main closure test. If the fit recovers all Wilson
coefficients centred on the injected truth, the RGE-consistent projections and
theory matrix are correct.

**Output:** `results/<tag>/fits/<tag>_BSMclosure_SMEFT/fit_results.json`

---

### Step 3 — UV coupling NS fit (nested sampling)

**What it tests:** Can smefit recover the injected BSM signal treating the UV
couplings (e.g. gZH, gZl) as the fundamental parameters, with the operator
coefficients expressed as correlated functions of those couplings via
tree-level matching?

Runs smefit in nested sampling mode (`smefit NS`) on the **BSM pseudo-data**
with the UV constrain block from `model.uv_coeff_block()`. Each EFT operator
is expressed as e.g.:

```
C_HD = -gZH² / mZp²
C_Hl = +gZH · gZl / mZp²
```

The NS fit explores the joint posterior over `(gZH, gZl)` and computes the
Bayesian evidence `log Z`.

**Skip with:** `--no-ns`
**Runtime:** ~15–30 minutes
**Output:** `results/<tag>/fits/<tag>_UVcoup/fit_results.json`

---

### Step 4 — PCA (Fisher eigenvectors)

Builds the Fisher information matrix:

```
F = K^T · C⁻¹ · K
```

where `K` is the (n_obs × n_ops) theory matrix from the database (not
RGE-evolved here — this is a direct DB read used for diagnostic decomposition)
and `C⁻¹` is the full inverse covariance:

```
C = diag(stat²) + S·Sᵀ + C_theory
```

(stat errors, systematic correlations, theory covariance, block-diagonal across datasets).

Computes eigenvalues and eigenvectors and saves them for Step 5.

**Output:** `results/<tag>/pca/*.npy`

---

### Step 5 — Profile likelihood table + significance bar chart

Computes discovery significance for the injected signal under multiple
hypotheses. The test statistic is:

```
q = χ²_SM − χ²_best-fit
```

which follows a χ²(k) distribution under the SM null, where k is the number
of free parameters. Significance = Gaussian sigma converted from the p-value.

| Method | Description |
|---|---|
| **Best 1-op** | Best single-operator profile likelihood. Tests if one operator alone explains the signal. |
| **PCA k=2** | Top-2 Fisher eigenvectors. Often poor for Z' (signal not aligned with top-2 modes). |
| **PCA k=4** | Top-4 eigenvectors. Usually captures the signal well. |
| **Full SMEFT** | All operators simultaneously. Maximum sensitivity. |
| **UV coupling** | Profile likelihood at the injected UV truth coefficients (χ²_SM − χ²_truth). Upper bound on what NS can recover. |

Bayesian evidence from the analytic and NS fits is also reported as `ln(BF)`
relative to the SM null.

**How to interpret:**
- σ < 3: no evidence for new physics
- 3 ≤ σ < 5: evidence (suggestive)
- σ ≥ 5: **discovery**

For the Z' model at gZH=0.12, gZl=0.04, mZp=1.0 TeV (FCC-ee projections):
- Best 1-op (OpD): ~12σ
- Full SMEFT: ~14σ
- UV coupling: ~15σ → clear 5σ discovery

**Output:**
- `results/<tag>/summary.txt` — numerical table
- `results/<tag>/plots/significance_summary.png` — bar chart with 3σ/5σ reference lines

---

### Step 6 — smefit report

Calls `smefit R` to generate a full HTML/PDF report comparing the SM-only fit
and the SMEFT BSM fit.

**Contents:**
- **Posterior histograms** — for each operator, overlays SM-only (blue) vs
  SMEFT (orange) posteriors with the UV truth injected value (red dashed line).
  A well-centred orange histogram with the dashed line in the middle confirms
  the closure test passes.
- **Scatter plot** — confidence level bounds (68%/95%) per operator across both fits.
- **Chi² table** — per-dataset chi² contributions for both fits.
- **Correlation matrix** — operator correlations in the SMEFT fit.
- **PCA decomposition** — Fisher information breakdown by operator group.

> The UV NS fit is **not** included in the report because its free parameters
> are UV couplings (`gZH`, `gZl`), not EFT operators — incompatible with the
> operator-level plots. Its significance is in the Step 5 summary instead.

**Output:** `results/<tag>/reports/Report_<tag>/`

---

## Output directory structure

```
fullpipeline/
├── projections/
│   └── <tag>/
│       ├── FCCee_ww_161GeV.yaml        (BSM pseudo-data)
│       ├── FCCee_Zdata.yaml
│       └── ...  (35 datasets total)
│
└── results/
    └── <tag>/
        ├── summary.txt                 (Step 5 significance table)
        ├── runcards/
        │   ├── <tag>_SMonly.yaml
        │   ├── <tag>_BSMclosure_SMEFT.yaml
        │   ├── <tag>_UVcoup.yaml
        │   └── Report_<tag>.yaml
        ├── fits/
        │   ├── <tag>_SMonly/
        │   │   └── fit_results.json
        │   ├── <tag>_BSMclosure_SMEFT/
        │   │   ├── fit_results.json
        │   │   └── rge_matrix.pkl
        │   └── <tag>_UVcoup/
        │       ├── fit_results.json
        │       └── rge_matrix.pkl
        ├── pca/
        │   ├── K_fit.npy
        │   ├── C_inv.npy
        │   ├── data_delta.npy
        │   ├── eigenvalues.npy
        │   └── eigenvectors.npy
        ├── plots/
        │   └── significance_summary.png
        └── reports/
            └── Report_<tag>/
                └── meta/
                    ├── coefficient_histo.png
                    ├── scatter.png
                    ├── chi2_table.*
                    └── ...
```

For a scan run, outputs go under `results/<tag>_scan/`:
```
results/<tag>_scan/
├── discovery_table.txt
└── plots/
    └── discovery_region.png
```

---

## FCC-ee datasets included

The pipeline uses 35 FCC-ee datasets at the WW threshold (161, 240, 365 GeV),
Z pole, Higgs (ZH), electroweak precision observables:

| Group | Datasets |
|---|---|
| WW threshold | `FCCee_ww_161GeV`, `_240GeV`, `_365GeV` |
| Hadronic Z | `FCCee_Rb/Rc/Rmu/Rtau/bb_Afb/cc_Afb/mumu_Afb/tautau_Afb/sigmaHad` at 240/365 GeV |
| Z pole | `FCCee_Zdata` (12 observables), `FCCee_Wwidth`, `FCCee_alphaEW` |
| ZH Higgs | `FCCee_zh_240/365GeV`, `_WW/ZZ/aZ/aa/tautau_240/365GeV` |

All at LO EFT accuracy except Higgs datasets (NLO_EW).

---

## RGE settings

All fits use:
```
init_scale: 1000.0 GeV     (matching scale = 1 TeV)
obs_scale:  dynamic         (run to each dataset's own energy scale)
smeft_accuracy: integrate   (exact numerical integration of RGE)
yukawa: top                 (include top Yukawa in ADM)
adm_QCD: False              (QCD ADM off)
```

The projection generation uses the same settings so that the injected signal
is exactly self-consistent with the fit predictions.

---

## How to add a new BSM model

The **only physics input you need to write** is a Python model file.
Everything else (runcards, fits, report, significance) is generated automatically.

### Step 1 — Write the model file

Create `fullpipeline/models/mynewmodel.py`. Use `zprime.py` as a template.
The file must define a Python dataclass with these five elements:

```python
from dataclasses import dataclass
from typing import Dict

@dataclass
class MyNewModel:
    g:   float = 0.1    # UV coupling(s)
    mX:  float = 1.0    # mass in TeV

    # 1. Which Warsaw-basis operators does this particle generate at tree level?
    #    These must exist as keys in the smefit theory DB (smefit_database/theory/*.json)
    OPERATORS = ["OpD", "Opl1", ...]

    # 2. Tree-level matching: Wilson coefficients at scale mX (units TeV^-2)
    def eft_coefficients(self) -> Dict[str, float]:
        return {
            "OpD":  -self.g**2 / self.mX**2,
            "Opl1": +self.g**2 / self.mX**2,
            ...
        }

    # 3. UV constrain block for the NS runcard
    #    Expresses each operator as a function of UV couplings
    #    Format: C_op = prefactor * param^power * ...
    def uv_coeff_block(self) -> Dict:
        return {
            "OpD":  {"constrain": [{"g": [-1.0, 2], "mX_TeV": [1.0, -2]}]},
            "Opl1": {"constrain": [{"g": [1.0,  2], "mX_TeV": [1.0, -2]}]},
            ...
            "g":     {"min": 0.0, "max": 0.5},
            "mX_TeV": {"constrain": True, "value": float(self.mX)},
        }

    # 4. Names of the free UV parameters in the NS fit
    def uv_param_names(self):
        return ["g"]

    # 5. Injected truth values (for closure test overlay in report)
    def uv_truth(self) -> Dict[str, float]:
        return {"g": self.g}
```

The matching relations and operator list come from your UV Lagrangian via the
standard SMEFT matching dictionaries (e.g. de Blas et al., arXiv:1706.03171).
The operators must be present as keys in the `LO` block of the smefit theory
files in `smefit_database/theory/FCCee_*.json` — check this first.

### Step 2 — Register it

Add to `fullpipeline/models/__init__.py`:
```python
from .mynewmodel import MyNewModel
```

### Step 3 — Add CLI support in `run_pipeline.py`

In `parse_args()`, add the parameter flags:
```python
p.add_argument("--g",  type=float, default=0.1)
p.add_argument("--mX", type=float, default=1.0)
```

In the `if __name__ == "__main__"` block, add a branch:
```python
elif args.model == "mynewmodel":
    model = MyNewModel(g=args.g, mX=args.mX)
    base_params = {"g": args.g, "mX": args.mX}
    tag = args.tag or f"mynewmodel_g{int(args.g*100):03d}_mX{int(args.mX*10):03d}"
    coupling_key, mass_key = "g", "mX"
    model_cls = MyNewModel
    coupling_grid = [0.05, 0.07, 0.09, 0.11, 0.12, 0.15]
    mass_grid     = [0.5, 1.0, 1.5, 2.0, 3.0]
    scan_params   = {**base_params}
```

Also add `"mynewmodel"` to the `choices=` list in `--model`.

### Step 4 — Run

```bash
python run_pipeline.py --model mynewmodel --g 0.1 --mX 1.0
```

That is all. The pipeline generates projections, fits, significance, and report
automatically from the model object.

---

### Summary of what you write vs what is automatic

| What | Written by you | Automatic |
|---|---|---|
| Tree-level matching relations | `eft_coefficients()` in `.py` | — |
| Operator list | `OPERATORS` in `.py` | — |
| UV constrain block | `uv_coeff_block()` in `.py` | — |
| CLI flags | 3 lines in `run_pipeline.py` | — |
| smefit runcards (A, NS, R) | — | built by pipeline |
| BSM pseudo-data | — | generated by pipeline |
| Fits (SM-only, SMEFT, UV NS) | — | run by pipeline |
| Significance table + plot | — | computed by pipeline |
| smefit HTML report | — | generated by pipeline |

---

## Interpreting the results

### The closure histograms (`coefficient_histo.png`)

Each panel shows the posterior distribution of one EFT operator:
- **Blue (SM-only):** Fit to SM data. Should be centred at zero with width set
  by the experimental precision.
- **Orange (SMEFT):** Fit to BSM pseudo-data. Should be shifted away from zero
  and centred on the red dashed line.
- **Red dashed line:** Injected truth value `c_j = model.eft_coefficients()[op]`.

A good closure test has the orange peak sitting exactly on the red line.
The separation between the blue and orange distributions is the discovery power
for that operator.

### The significance table (`summary.txt`)

```
Method                        q   ndof    sigma  note
--------------------------------------------------------------
Best 1-op                144.37      1   12.02σ  OpD
PCA k=2                    7.31      2    2.23σ
PCA k=4                  158.59      4   12.02σ
Full SMEFT               230.13      7   14.25σ  rank=7/8
UV coupling              226.70      2   14.86σ  lnBF=107.43
```

- **Best 1-op:** Which single operator drives the signal? Here OpD (the Higgs-D
  operator from Z'-Higgs coupling) with 12σ — this is the dominant mode.
- **PCA k=2 = 2.2σ vs PCA k=4 = 12σ:** The Z' signal is not in the top-2 Fisher
  eigenvectors; you need at least 4. This means a naive 2-mode compression would
  destroy 90% of the discovery power.
- **Full SMEFT = 14σ:** Maximum EFT sensitivity with all operators free.
- **UV coupling = 15σ:** Slightly higher because the UV matching constraints
  correlate operators (fewer degrees of freedom), making the fit more powerful.
- **lnBF = 107:** Strong Bayesian evidence for the BSM hypothesis over SM.

### Conclusion threshold

| sigma | Conclusion |
|---|---|
| < 3 | No evidence. Signal below FCC-ee sensitivity at this coupling/mass. |
| 3 – 5 | Evidence for new physics, but not discovery-level. |
| ≥ 5 | **5σ discovery.** FCC-ee can discover this UV model. |

For the Z' at gZH=0.12, mZp=1 TeV, all methods exceed 5σ comfortably.
To find the discovery threshold, run the scan:
```bash
python run_pipeline.py --model zprime --gZH 0.12 --gZl 0.04 --mZp 1.0 --scan
```
The resulting `discovery_region.png` shows the 5σ contour in (coupling, mass) space.
