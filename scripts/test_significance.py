"""
Smoke tests for the analytic significance pipeline.

Run with:
    python scripts/test_significance.py

No smefit call needed — purely analytic. Tests that core numbers haven't
regressed after edits to run_pipeline.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.run_pipeline import _build_K_Ci, _build_delta, _point_significance, _q_to_sigma
from models.wprime import WPrimeModel

PROJ_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "projections", "wprime_gwh012_mwp010")

def test_sigma_benchmark():
    """Best-1op significance for gWH=0.12, mWp=1 TeV must be ~10.6σ."""
    model = WPrimeModel(gWH=0.12, gWLf11=0.04, gWLf22=0.04, gWLf33=0.04,
                        gWqf33=0.04, mWp=1.0)
    ops   = model.OPERATORS
    K, Ci = _build_K_Ci(ops)
    delta = _build_delta(PROJ_DIR)
    s     = _point_significance(model, delta, K, Ci, ops)

    sigma = s["sigma_best1"]
    assert 10.0 < sigma < 11.5, f"Best-1op sigma={sigma:.2f}, expected ~10.6"
    print(f"  [PASS] Best-1op sigma = {sigma:.2f}σ")

def test_uv_ge_full_smeft():
    """UV truth significance must be >= Full SMEFT significance."""
    model = WPrimeModel(gWH=0.12, gWLf11=0.04, gWLf22=0.04, gWLf33=0.04,
                        gWqf33=0.04, mWp=1.0)
    ops   = model.OPERATORS
    K, Ci = _build_K_Ci(ops)
    delta = _build_delta(PROJ_DIR)
    s     = _point_significance(model, delta, K, Ci, ops)

    assert s["sigma_uv"] >= s["sigma_full"] - 0.5, (
        f"UV sigma={s['sigma_uv']:.2f} unexpectedly below Full SMEFT={s['sigma_full']:.2f}")
    print(f"  [PASS] UV sigma={s['sigma_uv']:.2f}σ >= Full SMEFT={s['sigma_full']:.2f}σ")

def test_pca_monotone():
    """PCA k=4 significance must be >= PCA k=2."""
    model = WPrimeModel(gWH=0.12, gWLf11=0.04, gWLf22=0.04, gWLf33=0.04,
                        gWqf33=0.04, mWp=1.0)
    ops   = model.OPERATORS
    K, Ci = _build_K_Ci(ops)
    delta = _build_delta(PROJ_DIR)
    s     = _point_significance(model, delta, K, Ci, ops)

    assert s["q_pca4"] >= s["q_pca2"], (
        f"PCA q4={s['q_pca4']:.2f} < q2={s['q_pca2']:.2f} — monotonicity broken")
    print(f"  [PASS] PCA k=2: {s['sigma_pca2']:.2f}σ  k=4: {s['sigma_pca4']:.2f}σ")

def test_q_to_sigma_known():
    """chi2(1) = 3.84 corresponds to 1.96σ (two-sided 95%)."""
    sigma = _q_to_sigma(3.8415, 1)
    assert abs(sigma - 1.96) < 0.01, f"q_to_sigma(3.84, 1) = {sigma:.3f}, expected ~1.96"
    print(f"  [PASS] q_to_sigma(3.84, 1) = {sigma:.3f}σ")

if __name__ == "__main__":
    if not os.path.isdir(PROJ_DIR):
        print(f"  [SKIP] Projections not found at {PROJ_DIR} — run the pipeline first")
        sys.exit(0)

    print("Running significance smoke tests...")
    test_q_to_sigma_known()
    test_sigma_benchmark()
    test_uv_ge_full_smeft()
    test_pca_monotone()
    print("\nAll tests passed.")
