# Adding a New BSM Model

The pipeline is model-agnostic. The **only physics you need to write** is a Python model file — everything else (runcards, fits, significance, report) is generated automatically.

## Step 1 — Write the model file

Create `fullpipeline/models/mynewmodel.py`. Use `models/zprime.py` as a template.

```python
from dataclasses import dataclass
from typing import Dict

@dataclass
class MyNewModel:
    g:   float = 0.1    # UV coupling
    mX:  float = 1.0    # mass in TeV

    OPERATORS = ["OpD", "Opl1", ...]   # Warsaw-basis operators generated at tree level

    def eft_coefficients(self) -> Dict[str, float]:
        """Tree-level matching: Wilson coefficients at scale mX (units TeV^-2)."""
        return {
            "OpD":  -self.g**2 / self.mX**2,
            "Opl1": +self.g**2 / self.mX**2,
        }

    def uv_coeff_block(self) -> Dict:
        """UV constrain block for the NS runcard."""
        return {
            "OpD":  {"constrain": [{"g": [-1.0, 2], "mX_TeV": [1.0, -2]}]},
            "Opl1": {"constrain": [{"g": [1.0,  2], "mX_TeV": [1.0, -2]}]},
            "g":     {"min": 0.0, "max": 0.5},
            "mX_TeV": {"constrain": True, "value": float(self.mX)},
        }

    def uv_param_names(self):
        return ["g"]

    def uv_truth(self) -> Dict[str, float]:
        return {"g": self.g}
```

## Step 2 — Register it

Add to `models/__init__.py`:
```python
from .mynewmodel import MyNewModel
```

## Step 3 — Add CLI support

In `scripts/run_pipeline.py`, add parameter flags in `parse_args()` and a branch in the main block. See the existing `wprime` / `zprime` branches as examples.

## Step 4 — Run

```bash
python run_pipeline.py --model mynewmodel --g 0.1 --mX 1.0
```

## What's automatic vs what you write

| Task | Your job | Automatic |
|------|----------|-----------|
| Tree-level matching | `eft_coefficients()` | — |
| Operator list | `OPERATORS` | — |
| UV constrain block | `uv_coeff_block()` | — |
| CLI flags | 3 lines in `run_pipeline.py` | — |
| smefit runcards | — | ✓ |
| BSM pseudo-data | — | ✓ |
| Fits + significance | — | ✓ |
| HTML report | — | ✓ |
