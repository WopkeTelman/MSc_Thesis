#!/usr/bin/env python3
"""
Regenerate PCA arrays for specified tags without re-running NS fits.

Usage:
    python scripts/regen_pca.py wprime_gwh050_mwp010 wprime_gwh050_mwp100
"""
import re
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE / "scripts"))

from run_pipeline import _run_pca, PIPELINE  # noqa: E402
RESULTS = PIPELINE / "results"

TAG_PATTERNS = [
    (r"^wprime_gwh(\d+)_mwp(\d+)$",             "wprime",             "gWH", "mWp"),
    (r"^wprime_constrained_gwh(\d+)_mwp(\d+)$", "wprime_constrained", "gWH", "mWp"),
    (r"^wprime_universal_gwh(\d+)_mwp(\d+)$",   "wprime_universal",   "gWH", "mWp"),
    (r"^zprime_gzh(\d+)_mzp(\d+)$",             "zprime",             "gZH", "mZp"),
]

def _model_from_tag(tag):
    for pat, mname, ck, mk in TAG_PATTERNS:
        m = re.match(pat, tag)
        if m:
            g_val = int(m.group(1)) / 100
            m_val = int(m.group(2)) / 10
            if mname == "wprime":
                from models.wprime import WPrimeModel
                return WPrimeModel(gWH=g_val, gWLf11=g_val/3, gWLf22=g_val/3,
                                   gWLf33=g_val/3, gWqf33=g_val/3, mWp=m_val)
            if mname == "wprime_constrained":
                from models.wprime_constrained import WPrimeConstrainedModel
                return WPrimeConstrainedModel(gWH=g_val, gWLf11=g_val/3, gWLf22=g_val/3,
                                              gWLf33=g_val/3, gWqf33=g_val/3, mWp=m_val)
            if mname == "wprime_universal":
                from models.wprime_universal import WPrimeUniversalModel
                return WPrimeUniversalModel(gWH=g_val, mWp=m_val)
            if mname == "zprime":
                from models.zprime import ZPrimeModel
                return ZPrimeModel(gZH=g_val, gZl=g_val/3, mZp=m_val)
    raise ValueError(f"Cannot parse tag: {tag!r}")


def main():
    tags = sys.argv[1:]
    if not tags:
        print("Usage: python regen_pca.py <tag> [<tag> ...]")
        sys.exit(1)

    for tag in tags:
        model    = _model_from_tag(tag)
        proj_dir = str(PIPELINE / "projections" / tag)
        pca_dir  = str(RESULTS / tag / "pca")

        if not Path(proj_dir).exists():
            print(f"  [{tag}] projections dir not found at {proj_dir} — skipping")
            continue

        print(f"\n[{tag}] Regenerating PCA  (model={model}) ...")
        _run_pca(model, proj_dir, pca_dir)
        print(f"  Done → {pca_dir}")


if __name__ == "__main__":
    main()
