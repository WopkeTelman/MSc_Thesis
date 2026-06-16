#!/usr/bin/env python3
"""
scripts/run_l1_replica.py

Per-condor-job script for L1 closure test.
Called by run_l1_ns.sh with the replica index (= HTCondor $(Process)).

For replica r:
  1. Loads L0 BSM pseudo-data from projections/l0/
  2. Draws L1 noise: d_rep = d_L0 + L_chol @ z,  z ~ N(0,I),  seed = r
  3. Writes noisy dataset yamls to projections/rep{r:03d}/
  4. Writes NS runcard to runcards/{tag}_l1rep{r:03d}.yaml
  5. Runs smefit NS on that runcard

Usage (by condor, not by hand):
    python run_l1_replica.py --rep 0 --out-dir /path/to/results/wprime_l1_gwh050_mwp120
"""
import sys, os, re, shutil, yaml, subprocess, argparse
import numpy as np
from pathlib import Path

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))

from scripts.run_pipeline import SMEFIT, ENV, DS_NAMES, DATASETS

TAG = "wprime_l1_gwh050_mwp120"


def _replace_data_central(src_text: str, new_vals) -> str:
    """Patch data_central values in a dataset yaml string."""
    scalar = re.compile(r'^(data_central:\s*)[\d.eE+\-]+(\s*)$', re.MULTILINE)
    if scalar.search(src_text):
        return scalar.sub(
            lambda m: f"{m.group(1)}{repr(float(new_vals[0]))}{m.group(2)}", src_text
        )
    hdr = re.compile(r'^data_central:\s*\n', re.MULTILINE)
    m = hdr.search(src_text)
    if m is None:
        raise ValueError("data_central block not found")
    start = m.end()
    item  = re.compile(r'^- [\d.eE+\-]+\n', re.MULTILINE)
    new_lines = [f"- {repr(float(v))}\n" for v in new_vals]
    out, pos, k = src_text[:start], start, 0
    for im in item.finditer(src_text, start):
        if im.start() != pos:
            break
        out += new_lines[k]; pos = im.end(); k += 1
        if k == len(new_lines):
            break
    return out + src_text[pos:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rep",     type=int, required=True, help="Replica index (0-49)")
    p.add_argument("--out-dir", type=Path,
                   default=PIPELINE / "results" / TAG)
    args = p.parse_args()

    r       = args.rep
    out_dir = Path(args.out_dir)
    rep_id  = f"{TAG}_l1rep{r:03d}"

    fits_dir = out_dir / "fits"
    done     = fits_dir / rep_id / "fit_results.json"
    if done.exists():
        print(f"[rep{r:03d}] Already done — skipping.")
        return

    l0_dir   = out_dir / "projections" / "l0"
    rep_dir  = out_dir / "projections" / f"rep{r:03d}"
    rc_dir   = out_dir / "runcards"

    # ── 1. Load Cholesky + slice map ─────────────────────────────────────────
    L     = np.load(out_dir / "projections" / "cov_chol.npy")
    slices_arr = np.load(out_dir / "projections" / "cov_slices.npy", allow_pickle=True)
    slices = {row[0]: (int(row[1]), int(row[2])) for row in slices_arr}

    # ── 2. Draw L1 noise ─────────────────────────────────────────────────────
    rng   = np.random.default_rng(seed=r)
    noise = L @ rng.standard_normal(L.shape[0])

    # ── 3. Write noisy projection yamls ──────────────────────────────────────
    rep_dir.mkdir(parents=True, exist_ok=True)

    # copy all files from L0 first (keeps metadata, stat/sys blocks intact)
    for f in l0_dir.iterdir():
        shutil.copy2(f, rep_dir / f.name)

    # patch data_central for each dataset
    for ds in DATASETS:
        name    = ds["name"]
        src     = rep_dir / f"{name}.yaml"
        if not src.exists():
            continue
        s, e    = slices[name]
        txt     = src.read_text("utf-8")
        src.write_text(_replace_data_central(txt, noise[s:e] +
                       _read_data_central(l0_dir / f"{name}.yaml")), "utf-8")

    print(f"[rep{r:03d}] L1 projection written to {rep_dir}")

    # ── 4. Write NS runcard ───────────────────────────────────────────────────
    template_path = rc_dir / "base_ns_template.yaml"
    rc = yaml.safe_load(template_path.read_text())
    rc["result_ID"] = rep_id
    rc["data_path"] = str(rep_dir.resolve())

    rc_path = rc_dir / f"{rep_id}.yaml"
    with open(rc_path, "w") as f:
        yaml.dump(rc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"[rep{r:03d}] Runcard: {rc_path}")

    # ── 5. Run smefit NS ──────────────────────────────────────────────────────
    print(f"[rep{r:03d}] Running smefit NS ...")
    result = subprocess.run([SMEFIT, "NS", str(rc_path)], env=ENV)
    if result.returncode != 0:
        print(f"[rep{r:03d}] ERROR: smefit NS exited {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"[rep{r:03d}] Done.")


def _read_data_central(path: Path):
    """Return data_central values as a numpy array from a dataset yaml."""
    sm = yaml.safe_load(path.read_text())
    dc = sm["data_central"]
    if not isinstance(dc, list):
        dc = [dc]
    return np.array([float(v) for v in dc])


if __name__ == "__main__":
    main()
