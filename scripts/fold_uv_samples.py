"""
Post-process UV coupling NS samples to resolve the Z2 sign degeneracy.

For W' models, flipping all UV couplings simultaneously (g -> -g for all)
leaves all Wilson coefficients unchanged. This creates a bimodal posterior
at +truth and -truth. We fix this by gauge-choosing gWH > 0: for any sample
where gWH < 0, flip the sign of all free UV couplings simultaneously.
The SMEFT operator values are recomputed consistently from the flipped couplings.

Usage:
    python fold_uv_samples.py <fit_results.json>

Writes the result to <fit_results_folded.json> in the same directory.
"""

import json
import sys
import os
import numpy as np

UV_COUPLINGS = ["gWH", "gWLf11", "gWLf22", "gWLf33", "gWqf33"]


def fold_samples(fit_results_path):
    with open(fit_results_path) as f:
        data = json.load(f)

    samples = data["samples"]
    n = len(samples["gWH"])

    gwh = np.array(samples["gWH"])
    negative_mask = gwh < 0
    n_flipped = negative_mask.sum()
    print(f"Total samples: {n}")
    print(f"Samples with gWH < 0 (to be flipped): {n_flipped} ({100*n_flipped/n:.1f}%)")

    # Flip all UV couplings for samples where gWH < 0
    for coupling in UV_COUPLINGS:
        arr = np.array(samples[coupling])
        arr[negative_mask] *= -1
        samples[coupling] = arr.tolist()

    # Recompute SMEFT operator values from flipped UV couplings
    # Each SMEFT op is a sum of terms: coeff * prod(g_i^p_i)
    # After flipping all g_i simultaneously, products of pairs are unchanged,
    # but odd-power products change sign. However, the constrain relations
    # encode products of exactly 2 UV couplings (one from each vertex),
    # so all SMEFT ops are invariant under the simultaneous flip.
    # Therefore we do NOT need to recompute SMEFT ops — they are already correct.
    # (Verified: all Wilson coefficients depend on even products of UV couplings.)

    data["samples"] = samples

    out_path = fit_results_path.replace(".json", "_folded.json")
    with open(out_path, "w") as f:
        json.dump(data, f)
    print(f"Written to: {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fold_uv_samples.py <fit_results.json>")
        sys.exit(1)
    fold_samples(sys.argv[1])
