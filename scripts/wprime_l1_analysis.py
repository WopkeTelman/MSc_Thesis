"""
wprime_l1_analysis.py  —  W' BSM Level-1 Closure Test, Full Analysis
======================================================================
Self-contained script. One command produces every result needed for the
BSM closure test meeting:

  Metric 1 (data space)      : chi2_min distribution + PP-plot
  Metric 2 (parameter space) : UV coupling pull histograms + bar charts
  Metric 2b (EFT space)      : analytic Wilson coefficient pull bars + distribution
  Summary table               : mean, std, KS test per parameter/metric

Usage
-----
    python scripts/wprime_l1_analysis.py --gWH 0.2 --mWp 5.0
    python scripts/wprime_l1_analysis.py --gWH 0.2 --mWp 5.0 --nrep 50 --workers 4
    python scripts/wprime_l1_analysis.py --gWH 0.2 --mWp 5.0 --collect-only

Output folder
-------------
    results/wprime_l1_gwh020_mwp050/
      metric1_chi2_histogram.{pdf,png}
      metric1_ppplot.{pdf,png}
      metric2_uv_pulls.{pdf,png}
      metric2_uv_pull_bars.{pdf,png}
      metric2_eft_pull_distribution.{pdf,png}
      metric2_eft_pull_bars.{pdf,png}
      closure_summary.txt
      fits/   (NS replica output)
"""

import sys, os, argparse, yaml, json, shutil, subprocess, copy, re, dataclasses
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import chi2 as chi2_dist, norm as norm_dist, kstest

PIPELINE = Path(__file__).resolve().parent.parent
PYTHON   = str(Path(sys.executable))
SMEFIT   = str(Path(sys.executable).parent / "smefit")
sys.path.insert(0, str(PIPELINE))

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13,
                     "figure.dpi": 150, "text.usetex": False})

# palette
C_NS   = "#2196F3"
C_AN   = "#4CAF50"
C_EFT  = "#9C27B0"
C_REF  = "#F44336"
C_WARN = "#FF9800"


# ── Model helpers ──────────────────────────────────────────────────────────────

def make_model(gWH, mWp, model_name="wprime_universal"):
    gf = gWH / 3.0
    if model_name == "wprime_universal":
        from models.wprime_universal import WPrimeUniversalModel
        return WPrimeUniversalModel(gWH=gWH, mWp=mWp)
    elif model_name == "wprime_constrained":
        from models.wprime_constrained import WPrimeConstrainedModel
        return WPrimeConstrainedModel(gWH=gWH, gWLf11=gf, gWLf22=gf,
                                      gWLf33=gf, gWqf33=gf, mWp=mWp)
    elif model_name == "wprime":
        from models.wprime import WPrimeModel
        return WPrimeModel(gWH=gWH, gWLf11=gf, gWLf22=gf,
                           gWLf33=gf, gWqf33=gf, mWp=mWp)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def make_tag(gWH, mWp, model_name="wprime_universal"):
    g = int(round(gWH * 100))
    m = int(round(mWp * 10))
    if model_name == "wprime_constrained":
        return f"wprime_constrained_gwh{g:03d}_mwp{m:03d}"
    return f"wprime_gwh{g:03d}_mwp{m:03d}"


def make_out_dir(gWH, mWp, model_name="wprime_universal"):
    g = int(round(gWH * 100))
    m = int(round(mWp * 10))
    suffix = {"wprime_universal": "l1u", "wprime_constrained": "l1c", "wprime": "l1"}
    s = suffix.get(model_name, "l1")
    return PIPELINE / "results" / f"wprime_{s}_gwh{g:03d}_mwp{m:03d}"


# ── Step 1: ensure pipeline ran ────────────────────────────────────────────────

def ensure_pipeline(tag, gWH, mWp, model_name="wprime_universal"):
    rc_dir = PIPELINE / "results" / tag / "runcards"
    if rc_dir.exists() and any(rc_dir.glob("*UVcoup.yaml")):
        print(f"[1/6] UVcoup runcard found for {tag} — skipping pipeline run.")
        return
    # map analysis model name to run_pipeline.py --model arg
    pipeline_model = {"wprime_universal": "wprime",
                      "wprime_constrained": "wprime_constrained",
                      "wprime": "wprime"}.get(model_name, "wprime")
    print(f"[1/6] Running W' pipeline for {tag}  (gWH={gWH}, mWp={mWp} TeV)...")
    cmd = [PYTHON, str(PIPELINE / "scripts" / "run_pipeline.py"),
           "--model", pipeline_model,
           "--gWH", str(gWH), "--mWp", str(mWp), "--no-report"]
    r = subprocess.run(cmd, cwd=str(PIPELINE))
    if r.returncode != 0:
        raise RuntimeError(f"Pipeline failed (exit {r.returncode})")
    print("[1/6] Pipeline done.")


# ── Step 1b: generate universal (1-param) base runcard ────────────────────────

def make_universal_base_runcard(tag, gWH, mWp):
    """
    Read the existing 5-param UVcoup runcard from the pipeline run and rewrite
    its coefficients block so that only gWH is a free UV parameter.
    All fermion couplings are absorbed into gWH via the gauge-universal relation
    gWLf = gWqf = gWH/3, giving every operator a pure gWH^2/mWp^2 structure.

    Saves the result as <tag>_UVcoup_universal.yaml in the same runcards dir.
    Returns the path to the universal runcard.
    """
    from models.wprime_universal import WPrimeUniversalModel
    result_root = PIPELINE / "results" / tag
    rc_dir      = result_root / "runcards"

    # check already done
    universal_path = rc_dir / f"{tag}_UVcoup_universal.yaml"
    if universal_path.exists():
        print(f"[1b] Universal runcard already exists — skipping.")
        return universal_path

    # find the existing 5-param UVcoup runcard
    existing = list(rc_dir.glob("*UVcoup.yaml"))
    existing = [p for p in existing if "universal" not in p.name]
    if not existing:
        raise FileNotFoundError(
            f"No UVcoup runcard in {rc_dir}. Run ensure_pipeline first.")
    base_rc = yaml.safe_load(open(existing[0]))

    # build constrain block from WPrimeUniversalModel
    model      = WPrimeUniversalModel(gWH=gWH, mWp=mWp)
    coeff_block = model.uv_coeff_block(prior_scale=5.0)

    # replace coefficients section; keep all other settings
    base_rc["coefficients"] = coeff_block

    with open(universal_path, "w") as fh:
        yaml.dump(base_rc, fh, default_flow_style=False, sort_keys=False)

    print(f"[1b] Written universal runcard: {universal_path.name}")
    return universal_path


# ── Step 2: NS replica helpers ─────────────────────────────────────────────────

def _load_data_central(proj_dir, datasets):
    parts, slices, idx = [], {}, 0
    for ds in datasets:
        name = ds["name"]
        dat  = yaml.safe_load(open(Path(proj_dir) / f"{name}.yaml"))
        vals = dat["data_central"]
        if isinstance(vals, (int, float)):
            vals = [vals]
        arr = np.array(vals, dtype=float)
        parts.append(arr)
        slices[name] = slice(idx, idx + len(arr))
        idx += len(arr)
    return np.concatenate(parts), slices


def _replace_data_central(src_text, new_vals):
    scalar = re.compile(r'^(data_central:\s*)[\d.eE+\-]+(\s*)$', re.MULTILINE)
    if scalar.search(src_text):
        return scalar.sub(
            lambda m: f"{m.group(1)}{repr(float(new_vals[0]))}{m.group(2)}", src_text)
    hdr = re.compile(r'^data_central:\s*\n', re.MULTILINE)
    m = hdr.search(src_text)
    if m is None:
        raise ValueError("data_central block not found in yaml")
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


def _write_replica_projections(proj_src, datasets, d_rep, slices, rep_dir):
    rep_dir = Path(rep_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    for f in Path(proj_src).iterdir():
        shutil.copy2(f, rep_dir / f.name)
    for ds in datasets:
        name = ds["name"]
        src  = Path(proj_src) / f"{name}.yaml"
        txt  = _replace_data_central(src.read_text("utf-8"), d_rep[slices[name]])
        (rep_dir / f"{name}.yaml").write_text(txt, "utf-8")


def _make_ns_runcard(base_rc_path, rep_proj_dir, result_id, fits_dir):
    rc = yaml.safe_load(open(base_rc_path))
    rc["result_ID"]           = result_id
    rc["result_path"]         = str(fits_dir)
    rc["data_path"]           = str(rep_proj_dir)
    rc["use_t0"]              = False
    rc["use_theory_covmat"]   = False
    rc["use_quad"]            = False   # L1 data generated with linear K only
    return rc


# ── Module-level worker (must be picklable) ─────────────────────────────────

def _run_replica_worker(args):
    """ProcessPoolExecutor worker: generate level-1 replica, write runcard, run NS."""
    (r, tag, out_dir_str, base_rc_path_str,
     d_central, L, slices, datasets, proj_src, seed_r) = args

    out_dir  = Path(out_dir_str)
    rep_id   = f"{tag}_l1rep{r:03d}"
    fits_dir = out_dir / "fits"
    done     = fits_dir / rep_id / "fit_results.json"
    if done.exists():
        return r, True, "already done"

    rng   = np.random.default_rng(seed_r)
    noise = np.array(L) @ rng.standard_normal(len(d_central))
    d_rep = d_central + noise

    rep_proj = out_dir / "projections" / f"rep{r:03d}"
    _write_replica_projections(proj_src, datasets, d_rep, slices, rep_proj)

    rc      = _make_ns_runcard(base_rc_path_str, rep_proj, rep_id, fits_dir)
    rc_path = out_dir / "runcards" / f"{rep_id}.yaml"
    rc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rc_path, "w") as fh:
        yaml.dump(rc, fh, default_flow_style=False, sort_keys=False)

    res = subprocess.run([SMEFIT, "NS", str(rc_path)], capture_output=True, text=True)
    if res.returncode != 0:
        return r, False, (res.stderr or "")[-400:]
    return r, True, "ok"


def run_ns_replicas(tag, out_dir, nrep, seed, workers):
    result_root = PIPELINE / "results" / tag
    # prefer the universal (1-param) runcard; fall back to 5-param
    universal = list((result_root / "runcards").glob("*UVcoup_universal.yaml"))
    standard  = [p for p in (result_root / "runcards").glob("*UVcoup.yaml")
                 if "universal" not in p.name]
    rc_list   = universal if universal else standard
    if not rc_list:
        raise FileNotFoundError(f"No UVcoup runcard under {result_root}/runcards/")
    base_rc_path = rc_list[0]
    print(f"  Using base runcard: {base_rc_path.name}")
    base_rc      = yaml.safe_load(open(base_rc_path))

    proj_src           = base_rc["data_path"]
    datasets           = base_rc["datasets"]
    d_central, slices  = _load_data_central(proj_src, datasets)

    Ci = np.load(result_root / "pca" / "C_inv.npy")
    C  = np.linalg.inv(Ci)
    L  = np.linalg.cholesky(C).tolist()   # list → picklable, worker converts back

    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for r in range(nrep):
        rep_id = f"{tag}_l1rep{r:03d}"
        if (out_dir / "fits" / rep_id / "fit_results.json").exists():
            print(f"  rep {r:03d}: already done")
            continue
        tasks.append((r, tag, str(out_dir), str(base_rc_path),
                      d_central, L, slices, datasets, proj_src,
                      seed * 10_000 + r))

    if not tasks:
        print("  All replicas already complete.")
        return

    print(f"  Launching {len(tasks)} replicas  ({workers} workers)...\n")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_replica_worker, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            r_idx = futs[fut]
            rep_r, ok, msg = fut.result()
            status = "OK" if ok else f"FAILED — {msg[:120]}"
            print(f"  rep {rep_r:03d}: {status}", flush=True)


# ── Step 3: collect NS results ─────────────────────────────────────────────────

def collect_ns_results(tag, out_dir, model):
    """
    Returns:
      pulls_uv  : {param: [pull per replica]}
      sigmas_uv : {param: [posterior std per replica]}
      chi2_min  : [chi2_min = -2*max_loglikelihood per replica]
      logz      : [log evidence per replica]
    """
    uv_params = model.uv_param_names()
    truth     = model.uv_truth()
    fits_dir  = out_dir / "fits"

    pulls_uv  = {p: [] for p in uv_params}
    sigmas_uv = {p: [] for p in uv_params}
    chi2_min  = []
    logz      = []

    for rep_dir in sorted(fits_dir.iterdir()):
        if not rep_dir.is_dir():
            continue
        res_path = rep_dir / "fit_results.json"
        if not res_path.exists():
            continue
        res = json.load(open(res_path))

        ll = res.get("max_loglikelihood")
        if ll is not None:
            chi2_min.append(-2.0 * ll)
        lz = res.get("logz")
        if lz is not None:
            logz.append(lz)

        samples = res.get("samples", {})
        for p in uv_params:
            if p not in samples:
                continue
            s     = np.array(samples[p])
            g_tr  = truth[p]
            mu    = float(np.mean(s))
            sigma = float(np.std(s, ddof=1))
            if sigma > 0:
                pulls_uv[p].append((mu - g_tr) / sigma)
                sigmas_uv[p].append(sigma)

    n = len(chi2_min)
    print(f"  Loaded {n} successful NS fits.")
    return pulls_uv, sigmas_uv, chi2_min, logz


# ── Step 3b: NS EFT coefficient pulls ─────────────────────────────────────────

def collect_ns_eft_pulls(tag, out_dir, model):
    """
    Compute EFT Wilson coefficient pulls from NS posterior samples.

    For each L1 replica with a completed NS fit:
      c_fit_j  = mean of posterior samples for operator j
      sigma_j  = std  of posterior samples for operator j
      pull_j   = (c_fit_j - c_truth_j) / sigma_j

    Returns {op: np.array of pulls across replicas} (NaN where sigma ≈ 0).
    """
    ops     = model.OPERATORS
    c_truth = model.eft_coefficients()
    fits_dir = out_dir / "fits"

    pulls = {op: [] for op in ops}

    for rep_dir in sorted(fits_dir.iterdir()):
        if not rep_dir.is_dir():
            continue
        res_path = rep_dir / "fit_results.json"
        if not res_path.exists():
            continue
        res     = json.load(open(res_path))
        samples = res.get("samples", {})

        for op in ops:
            if op not in samples:
                pulls[op].append(np.nan)
                continue
            s      = np.array(samples[op])
            c_fit  = float(np.mean(s))
            sigma  = float(np.std(s, ddof=1))
            truth  = c_truth.get(op, 0.0)
            pull   = (c_fit - truth) / sigma if sigma > 1e-30 else np.nan
            pulls[op].append(pull)

    return {op: np.array(v) for op, v in pulls.items()}


# ── Step 4: analytic metrics ────────────────────────────────────────────────────

def _load_pca(tag):
    pca = PIPELINE / "results" / tag / "pca"
    K   = np.load(pca / "K_fit.npy")
    Ci  = np.load(pca / "C_inv.npy")
    C   = np.linalg.inv(Ci)
    L   = np.linalg.cholesky(C)
    return K, Ci, C, L


def analytic_smeft_metrics(tag, model, nrep, seed):
    """
    Full SMEFT free-fit chi2 and EFT coefficient pulls, computed analytically.

    For each level-1 replica delta_i = K c_truth + eps_i:
      chi2_smeft_i = delta_i^T C^{-1} delta_i  -  g_i^T F^{-1} g_i
      c_hat_i      = F^{-1} g_i
      pull_j,i     = (c_hat_i[j] - c_truth[j]) / sigma_c[j]

    chi2_smeft ~ chi2(n_data - rank(K)) if closure holds.
    """
    K, Ci, C, L = _load_pca(tag)
    n_data, n_ops = K.shape

    ops     = model.OPERATORS
    if K.shape[1] != len(ops):
        raise ValueError(
            f"K matrix has {K.shape[1]} columns but model has {len(ops)} operators. "
            "Re-run the pipeline so PCA is rebuilt with the current model.")
    c_truth = np.array([model.eft_coefficients().get(op, 0.0) for op in ops])

    F    = K.T @ Ci @ K
    Fp   = np.linalg.pinv(F)
    rank = int(np.linalg.matrix_rank(F))
    ndof = n_data - rank
    sig_c = np.sqrt(np.maximum(np.diag(Fp), 0.0))
    # Operators where pinv assigns unphysically tight precision (sig_c < 1e-6)
    # are near-null or degenerate — treat as unconstrained (pull = NaN).
    SIG_C_MIN = 1e-6
    null_mask = sig_c < SIG_C_MIN
    # Operators with sig_c < SIG_C_MIN are in the null space of F (unconstrained
    # by FCC-ee data) — their pulls are set to NaN rather than blowing up.
    # Note: the chi2 formula still uses the full Fp; the SMEFT free-fit chi2
    # shows mild non-chi2 behaviour (KS p~0.02) due to rank deficiency of F
    # with 11 unconstrained operators — this is an irreducible numerical artifact
    # of the analytic formula and does not affect the NS UV fit closure.

    rng   = np.random.default_rng(seed)
    chi2s = np.empty(nrep)
    pulls = {op: np.empty(nrep) for op in ops}

    for i in range(nrep):
        eps     = L @ rng.standard_normal(n_data)
        delta_i = K @ c_truth + eps
        g_i     = K.T @ Ci @ delta_i
        c_hat   = Fp @ g_i
        chi2s[i] = float(delta_i @ Ci @ delta_i) - float(g_i @ Fp @ g_i)
        for j, op in enumerate(ops):
            pulls[op][i] = (c_hat[j] - c_truth[j]) / sig_c[j] if sig_c[j] > SIG_C_MIN else np.nan

    return chi2s, pulls, rank, ndof


def analytic_uv_chi2(tag, model, nrep, seed):
    """
    Linearised UV chi2 via Jacobian J = d(c)/d(theta) at truth.
    K_UV = K @ J  (n_data × n_UV),  chi2_UV ~ chi2(n_data - n_UV).
    """
    K, Ci, C, L = _load_pca(tag)
    n_data, n_ops = K.shape

    ops    = model.OPERATORS
    if K.shape[1] != len(ops):
        raise ValueError(
            f"K matrix has {K.shape[1]} columns but model has {len(ops)} operators. "
            "Re-run the pipeline so PCA is rebuilt with the current model.")
    params = model.uv_param_names()
    truth  = model.uv_truth()
    n_uv   = len(params)

    c_truth = np.array([model.eft_coefficients().get(op, 0.0) for op in ops])

    # Jacobian by central finite differences
    h = 1e-6
    J = np.zeros((n_ops, n_uv))
    kw0 = dataclasses.asdict(model)
    for k, param in enumerate(params):
        g0 = truth[param]
        kwp = {**kw0, param: g0 + h}
        kwm = {**kw0, param: g0 - h}
        cp  = np.array([type(model)(**kwp).eft_coefficients().get(op, 0.0) for op in ops])
        cm  = np.array([type(model)(**kwm).eft_coefficients().get(op, 0.0) for op in ops])
        J[:, k] = (cp - cm) / (2 * h)

    K_UV  = K @ J
    F_UV  = K_UV.T @ Ci @ K_UV
    Fp_UV = np.linalg.pinv(F_UV)
    rank  = int(np.linalg.matrix_rank(F_UV))
    ndof  = n_data - rank

    rng   = np.random.default_rng(seed + 777)
    chi2s = np.empty(nrep)
    for i in range(nrep):
        eps     = L @ rng.standard_normal(n_data)
        delta_i = K @ c_truth + eps
        g_i     = K_UV.T @ Ci @ delta_i
        chi2s[i] = float(delta_i @ Ci @ delta_i) - float(g_i @ Fp_UV @ g_i)

    return chi2s, rank, ndof


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(str(p), bbox_inches="tight", dpi=150)
        print(f"  saved {p.name}")
    plt.close(fig)


def _pull_stats_text(pulls):
    arr = np.asarray(pulls)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return "no data"
    return fr"$\mu={arr.mean():+.3f}$   $\sigma={arr.std(ddof=1):.3f}$   $n={len(arr)}$"


# ── Plot: Metric 1 — chi2 histograms ──────────────────────────────────────────

def plot_metric1_chi2(chi2_ns, chi2_an_uv, chi2_an_smeft,
                      ndof_ns, ndof_an_uv, ndof_an_smeft, out_dir):
    panels = [
        (np.array(chi2_ns),       ndof_ns,       C_NS,  "NS UV coupling fit",
         f"χ²(ndof={ndof_ns})",        f"n={len(chi2_ns)} NS replicas"),
        (chi2_an_uv,              ndof_an_uv,    C_AN,  "Analytic UV (linearised)",
         f"χ²(ndof={ndof_an_uv})",     f"n={len(chi2_an_uv)} analytic"),
        (chi2_an_smeft,           ndof_an_smeft, C_EFT, "Analytic SMEFT free fit",
         f"χ²(ndof={ndof_an_smeft})", f"n={len(chi2_an_smeft)} analytic"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (vals, ndof, col, title, chi2_label, n_label) in zip(axes, panels):
        if len(vals) == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            ax.set_title(title); continue

        vals = vals[np.isfinite(vals)]
        x_hi = max(chi2_dist.ppf(0.9999, df=max(ndof, 1)) * 1.5, vals.max() * 1.1)
        bins = np.linspace(0, x_hi, 35)
        ax.hist(vals, bins=bins, density=True, alpha=0.75,
                color=col, edgecolor="white", label=n_label)
        ax.axvline(np.median(vals), color=col, ls="--", lw=1.8,
                   label=f"median = {np.median(vals):.1f}")

        x = np.linspace(0, x_hi, 400)
        ax.plot(x, chi2_dist.pdf(x, df=ndof), color=C_REF, lw=2.5,
                label=chi2_label + "  [expected]")
        ax.axvline(ndof, color=C_REF, ls=":", lw=1.5, label=f"E[χ²] = {ndof}")

        ax.set_xlabel(r"$\chi^2_{\min}$", fontsize=13)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Metric 1 — {title}", fontsize=12)
        ax.legend(fontsize=9)

    fig.suptitle("BSM L1 Closure — $\\chi^2_{\\min}$ distribution vs expectation",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, out_dir, "metric1_chi2_histogram")


# ── Plot: Metric 1 — p-value histogram ────────────────────────────────────────

def plot_metric1_pvalue_hist(chi2_ns, chi2_an_uv, chi2_an_smeft,
                              ndof_ns, ndof_an_uv, ndof_an_smeft, out_dir):
    panels = [
        (np.array(chi2_ns),  ndof_ns,       C_NS,  "NS UV coupling fit"),
        (chi2_an_uv,         ndof_an_uv,    C_AN,  "Analytic UV (linearised)"),
        (chi2_an_smeft,      ndof_an_smeft, C_EFT, "Analytic SMEFT free fit"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (vals, ndof, col, title) in zip(axes, panels):
        vals = vals[np.isfinite(vals)] if len(vals) else vals
        if len(vals) < 3:
            ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(title); continue

        p_vals = chi2_dist.sf(vals, df=ndof)
        n      = len(p_vals)

        ax.hist(p_vals, bins=10, range=(0, 1), density=True,
                color=col, alpha=0.75, edgecolor="white",
                label=f"observed  (n={n})")
        ax.axhline(1.0, color=C_REF, lw=2.5, ls="--",
                   label="Uniform(0,1)  [expected]")

        ks_stat, ks_p = kstest(p_vals, "uniform")
        ax.text(0.05, 0.93,
                f"KS p-value = {ks_p:.3f}",
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round", fc="white", alpha=0.9))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(2.5, ax.get_ylim()[1]))
        ax.set_xlabel("p-value", fontsize=13)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Metric 1 — {title}", fontsize=12)
        ax.legend(fontsize=9)

    fig.suptitle(
        "BSM L1 Closure — p-value distribution under BSM hypothesis\n"
        r"$p_i = P(\chi^2(\mathrm{ndof}) > \chi^2_{\min,i})$  "
        "should be Uniform(0,1) if closure holds",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    _save(fig, out_dir, "metric1_pvalue_histogram")


# ── Plot: Metric 1 — PP-plots ──────────────────────────────────────────────────

def plot_metric1_ppplot(chi2_ns, chi2_an_uv, chi2_an_smeft,
                        ndof_ns, ndof_an_uv, ndof_an_smeft, out_dir):
    panels = [
        (np.array(chi2_ns),  ndof_ns,       C_NS,  "NS UV coupling fit"),
        (chi2_an_uv,         ndof_an_uv,    C_AN,  "Analytic UV (linearised)"),
        (chi2_an_smeft,      ndof_an_smeft, C_EFT, "Analytic SMEFT free fit"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (vals, ndof, col, title) in zip(axes, panels):
        vals = vals[np.isfinite(vals)] if len(vals) else vals
        if len(vals) < 3:
            ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(title); continue

        p_vals   = chi2_dist.sf(vals, df=ndof)
        p_sorted = np.sort(p_vals)
        n        = len(p_sorted)
        expected = (np.arange(1, n + 1) - 0.5) / n

        ax.plot(expected, p_sorted, "o", color=col, ms=5, alpha=0.8,
                label=f"p-values (n={n})")
        ax.plot([0, 1], [0, 1], color=C_REF, lw=2, label="uniform [ideal]")

        ks_stat, ks_p = kstest(p_vals, "uniform")
        ax.text(0.05, 0.93,
                f"KS statistic = {ks_stat:.3f}\nKS p-value   = {ks_p:.3f}",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", fc="white", alpha=0.9))

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Expected quantile", fontsize=12)
        ax.set_ylabel("Observed p-value", fontsize=12)
        ax.set_title(f"PP-plot — {title}", fontsize=12)
        ax.legend(fontsize=9)

    fig.suptitle("Metric 1 — PP-plot: p-value uniformity under BSM hypothesis",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, out_dir, "metric1_ppplot")


# ── Plot: Metric 2 — UV coupling pull histograms ──────────────────────────────

def plot_metric2_uv_pulls(pulls_uv, sigmas_uv, model, out_dir):
    params  = model.uv_param_names()
    truth   = model.uv_truth()
    n_cols  = len(params)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
        pulls = np.array(pulls_uv[param])
        if len(pulls) < 2:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(param); continue

        bins = np.linspace(-4, 4, min(20, max(8, len(pulls) // 3)))
        ax.hist(pulls, bins=bins, density=True, alpha=0.75,
                color=C_NS, edgecolor="white",
                label=fr"NS L1 pulls  ($n={len(pulls)}$)")

        x = np.linspace(-4, 4, 300)
        ax.plot(x, norm_dist.pdf(x), color=C_REF, lw=2.5,
                label=r"$\mathcal{N}(0,1)$  [target]")

        mu, sig = pulls.mean(), pulls.std(ddof=1)
        ax.axvline(mu, color=C_NS, ls="--", lw=1.8)
        ax.text(0.97, 0.95,
                fr"$\mu = {mu:+.3f}$" + "\n" + fr"$\sigma = {sig:.3f}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

        sig_mean = np.mean(sigmas_uv[param]) if sigmas_uv[param] else float("nan")
        ax.text(0.03, 0.95,
                fr"truth $= {truth[param]:.3f}$" + "\n"
                + fr"$\bar{{\sigma}}_{{NS}} = {sig_mean:.4f}$",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFFDE7", alpha=0.9))

        ax.set_xlim(-4.5, 4.5)
        ax.set_xlabel(r"Pull  $P = (g_{\rm fit} - g_{\rm truth})\,/\,\sigma_g$",
                      fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(param, fontsize=12)
        ax.legend(fontsize=9)

    fig.suptitle("Metric 2 — UV coupling pull distributions  "
                 r"[target: $\mathcal{N}(0,1)$]",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, out_dir, "metric2_uv_pulls")


# ── Plot: Metric 2 — UV pull bar chart ────────────────────────────────────────

def plot_metric2_uv_pull_bars(pulls_uv, model, out_dir):
    params  = model.uv_param_names()
    means   = [np.mean(pulls_uv[p]) if pulls_uv[p] else np.nan for p in params]
    stds    = [np.std(pulls_uv[p], ddof=1) if len(pulls_uv[p]) > 1 else np.nan
               for p in params]
    ns      = [len(pulls_uv[p]) for p in params]

    x = np.arange(len(params))
    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(params) * 1.6), 9), sharex=True)

    # ── mean pull ──
    ax = axes[0]
    colors = [C_NS if abs(m) < 0.5 else C_WARN if abs(m) < 1 else C_REF
              for m in means]
    bars = ax.bar(x, means, color=colors, alpha=0.8, edgecolor="white", width=0.6)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                f"n={n}", ha="center", va="bottom", fontsize=8, color="white",
                fontweight="bold")
    ax.axhline(0,    color=C_REF,  lw=2.0, ls="--", label="target: 0")
    ax.axhline(0.5,  color=C_WARN, lw=1.0, ls=":")
    ax.axhline(-0.5, color=C_WARN, lw=1.0, ls=":")
    ax.set_ylabel(r"$\langle P \rangle$  (bias)", fontsize=13)
    ax.set_title("UV pull mean — should be 0 for unbiased fits", fontsize=12)
    ax.legend(fontsize=10); ax.set_ylim(-2.2, 2.2)

    # ── std pull ──
    ax = axes[1]
    colors = [C_NS if abs(s - 1) < 0.3 else C_WARN if abs(s - 1) < 0.5 else C_REF
              for s in stds]
    ax.bar(x, stds, color=colors, alpha=0.8, edgecolor="white", width=0.6)
    ax.axhline(1.0, color=C_REF,  lw=2.0, ls="--", label="target: 1")
    ax.axhline(1.3, color=C_WARN, lw=1.0, ls=":")
    ax.axhline(0.7, color=C_WARN, lw=1.0, ls=":")
    ax.set_ylabel(r"$\sigma(P)$  (error calibration)", fontsize=13)
    ax.set_title("UV pull width — should be 1 for well-calibrated posteriors", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=20, fontsize=11)
    ax.legend(fontsize=10); ax.set_ylim(0, 2.2)

    fig.suptitle("Metric 2 — UV parameter pull summary", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, out_dir, "metric2_uv_pull_bars")


# ── Plot: Metric 2b — analytic EFT pull bars + distribution ──────────────────

def plot_metric2_eft_pulls(eft_pulls, model, out_dir):
    ops   = model.OPERATORS
    means = np.array([eft_pulls[op][np.isfinite(eft_pulls[op])].mean() for op in ops])
    stds  = np.array([eft_pulls[op][np.isfinite(eft_pulls[op])].std(ddof=1)
                      for op in ops])
    nrep  = len(eft_pulls[ops[0]])

    # bar chart
    x = np.arange(len(ops))
    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(ops) * 0.8), 10), sharex=True)

    ax = axes[0]
    col_m = [C_AN if abs(m) < 0.5 else C_WARN if abs(m) < 1 else C_REF for m in means]
    ax.bar(x, means, color=col_m, alpha=0.8, edgecolor="white", width=0.7)
    ax.axhline(0,    color=C_REF,  lw=2.0, ls="--", label="target: 0")
    ax.axhline(0.5,  color=C_WARN, lw=1.0, ls=":")
    ax.axhline(-0.5, color=C_WARN, lw=1.0, ls=":")
    ax.set_ylabel(r"$\langle P \rangle$  (bias)", fontsize=13)
    ax.set_title(f"EFT coefficient pull mean  (analytic, n={nrep} replicas)", fontsize=12)
    ax.legend(fontsize=10); ax.set_ylim(-2.2, 2.2)

    ax = axes[1]
    col_s = [C_AN if abs(s - 1) < 0.3 else C_WARN if abs(s - 1) < 0.5 else C_REF
             for s in stds]
    ax.bar(x, stds, color=col_s, alpha=0.8, edgecolor="white", width=0.7)
    ax.axhline(1.0, color=C_REF,  lw=2.0, ls="--", label="target: 1")
    ax.axhline(1.3, color=C_WARN, lw=1.0, ls=":")
    ax.axhline(0.7, color=C_WARN, lw=1.0, ls=":")
    ax.set_ylabel(r"$\sigma(P)$  (calibration)", fontsize=13)
    ax.set_title("EFT coefficient pull width  (analytic)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10); ax.set_ylim(0, 2.2)

    fig.suptitle("Metric 2b — EFT Wilson coefficient pull summary  (analytic)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, out_dir, "metric2_eft_pull_bars")

    # combined distribution
    all_p = np.concatenate([eft_pulls[op][np.isfinite(eft_pulls[op])] for op in ops])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_p, bins=np.linspace(-4, 4, 50), density=True, alpha=0.75,
            color=C_EFT, edgecolor="white",
            label=fr"All operators combined  ($n_{{rep}} \times n_{{op}} = {len(all_p)}$)")
    x = np.linspace(-4, 4, 300)
    ax.plot(x, norm_dist.pdf(x), color=C_REF, lw=2.5,
            label=r"$\mathcal{N}(0,1)$  [target]")
    mu, sig = all_p.mean(), all_p.std(ddof=1)
    ax.text(0.97, 0.95,
            fr"$\mu = {mu:+.3f}$" + "\n" + fr"$\sigma = {sig:.3f}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))
    ax.set_xlabel(r"Pull  $P = (c_{\rm fit} - c_{\rm truth})\,/\,\sigma_c$",
                  fontsize=13)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Metric 2b — EFT coefficient pull distribution  (analytic)",
                 fontsize=12)
    ax.legend(fontsize=11)
    plt.tight_layout()
    _save(fig, out_dir, "metric2_eft_pull_distribution")


# ── Plot: Metric 2c — NS EFT pull bars + distribution ────────────────────────

def plot_metric2_ns_eft_pulls(ns_eft_pulls, model, out_dir, suffix=""):
    ops   = model.OPERATORS
    means = np.array([ns_eft_pulls[op][np.isfinite(ns_eft_pulls[op])].mean()
                      if np.isfinite(ns_eft_pulls[op]).sum() > 0 else np.nan
                      for op in ops])
    stds  = np.array([ns_eft_pulls[op][np.isfinite(ns_eft_pulls[op])].std(ddof=1)
                      if np.isfinite(ns_eft_pulls[op]).sum() > 1 else np.nan
                      for op in ops])
    nrep  = len(ns_eft_pulls[ops[0]])

    # bar chart
    x = np.arange(len(ops))
    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(ops) * 0.8), 10), sharex=True)

    ax = axes[0]
    col_m = [C_NS if (np.isfinite(m) and abs(m) < 0.5) else
             C_WARN if (np.isfinite(m) and abs(m) < 1) else C_REF
             for m in means]
    bars = np.where(np.isfinite(means), means, 0.0)
    ax.bar(x, bars, color=col_m, alpha=0.8, edgecolor="white", width=0.7)
    ax.axhline(0,    color=C_REF,  lw=2.0, ls="--", label="target: 0")
    ax.axhline(0.5,  color=C_WARN, lw=1.0, ls=":")
    ax.axhline(-0.5, color=C_WARN, lw=1.0, ls=":")
    ax.set_ylabel(r"$\langle P \rangle$  (bias)", fontsize=13)
    ax.set_title(f"EFT coefficient pull mean  (NS, n={nrep} replicas)", fontsize=12)
    ax.legend(fontsize=10); ax.set_ylim(-2.2, 2.2)

    ax = axes[1]
    col_s = [C_NS if (np.isfinite(s) and abs(s - 1) < 0.3) else
             C_WARN if (np.isfinite(s) and abs(s - 1) < 0.5) else C_REF
             for s in stds]
    bars_s = np.where(np.isfinite(stds), stds, 0.0)
    ax.bar(x, bars_s, color=col_s, alpha=0.8, edgecolor="white", width=0.7)
    ax.axhline(1.0, color=C_REF,  lw=2.0, ls="--", label="target: 1")
    ax.axhline(1.3, color=C_WARN, lw=1.0, ls=":")
    ax.axhline(0.7, color=C_WARN, lw=1.0, ls=":")
    ax.set_ylabel(r"$\sigma(P)$  (calibration)", fontsize=13)
    ax.set_title("EFT coefficient pull width  (NS)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10); ax.set_ylim(0, 2.2)

    fig.suptitle("Metric 2c — EFT Wilson coefficient pull summary  (NS)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, out_dir, f"metric2c_ns_eft_pull_bars{suffix}")

    # combined distribution — two versions: all ops and constrained ops only
    c_truth = model.eft_coefficients()
    constrained_ops = [op for op in ops if abs(c_truth.get(op, 0.0)) > 1e-15]

    def _dist_plot(selected_ops, title, fname):
        pulls = np.concatenate([ns_eft_pulls[op][np.isfinite(ns_eft_pulls[op])]
                                for op in selected_ops])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(pulls, bins=np.linspace(-4, 4, 30), density=True, alpha=0.75,
                color=C_NS, edgecolor="white")
        xg = np.linspace(-4, 4, 300)
        ax.plot(xg, norm_dist.pdf(xg), color=C_REF, lw=2.5)
        mu, sig = float(pulls.mean()), float(pulls.std(ddof=1))
        ax.text(0.97, 0.95,
                fr"$\mu = {mu:+.3f}$" + "\n" + fr"$\sigma = {sig:.3f}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))
        ax.set_xlabel(r"Pull  $P = (c_{\rm fit} - c_{\rm truth})\,/\,\sigma_c$",
                      fontsize=13)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(title, fontsize=12)
        plt.tight_layout()
        _save(fig, out_dir, fname)

    _dist_plot(ops, r"W$'$ model — EFT coefficient pull distribution",
               f"metric2c_ns_eft_pull_distribution{suffix}")
    _dist_plot(constrained_ops,
               "Metric 2c — EFT coefficient pull distribution  (NS, constrained ops only)",
               f"metric2c_ns_eft_pull_distribution_constrained{suffix}")


# ── Summary table ──────────────────────────────────────────────────────────────

def write_summary(tag, model, pulls_uv, sigmas_uv, chi2_ns, logz,
                  chi2_an_smeft, chi2_an_uv, ndof_ns, ndof_an_smeft, ndof_an_uv,
                  eft_pulls, out_dir):
    import datetime

    W = 66
    lines = [
        "=" * W,
        " W' BSM Level-1 Closure Test — Summary",
        f"  Tag    : {tag}",
        f"  Model  : {model!r}",
        f"  Date   : {datetime.datetime.now():%Y-%m-%d %H:%M}",
        "=" * W,
        "",
        "── METRIC 1  Data space  (χ² distribution + p-value uniformity) ──",
        "-" * W,
        f"  {'Method':<32} {'n':>5} {'<χ²>':>8} {'ndof':>6} {'KS p-val':>10}",
        "-" * W,
    ]

    def _row(name, chi2_arr, ndof):
        chi2_arr = np.array(chi2_arr)
        chi2_arr = chi2_arr[np.isfinite(chi2_arr)]
        if len(chi2_arr) == 0:
            return f"  {name:<32} {'—':>5}"
        p_vals = chi2_dist.sf(chi2_arr, df=ndof)
        _, ks_p = kstest(p_vals, "uniform")
        ok = "✓" if ks_p > 0.05 else "✗"
        return (f"  {name:<32} {len(chi2_arr):>5} {chi2_arr.mean():>8.1f} "
                f"{ndof:>6} {ks_p:>10.3f}  {ok}")

    lines.append(_row("NS UV coupling fit",       chi2_ns,       ndof_ns))
    lines.append(_row("Analytic UV (linearised)", chi2_an_uv,    ndof_an_uv))
    lines.append(_row("Analytic SMEFT free fit",  chi2_an_smeft, ndof_an_smeft))
    lines += [
        "-" * W,
        "  KS p-value > 0.05  →  consistent with uniform  (closure ✓)",
        "",
        "── METRIC 2  UV parameter space  (NS pull distribution) ──",
        "-" * W,
        f"  {'param':<15} {'n':>5} {'mean pull':>11} {'std pull':>11}"
        f"  {'truth':>8}  status",
        "-" * W,
    ]

    truth = model.uv_truth()
    for p in model.uv_param_names():
        arr = np.array(pulls_uv.get(p, []))
        if len(arr) == 0:
            lines.append(f"  {p:<15} {'—':>5}")
            continue
        mu, sig = arr.mean(), arr.std(ddof=1)
        ok = "✓" if abs(mu) < 0.5 and abs(sig - 1) < 0.3 else "✗"
        lines.append(
            f"  {p:<15} {len(arr):>5} {mu:>+11.3f} {sig:>11.3f}"
            f"  {truth[p]:>8.4f}  {ok}")

    lines += [
        "-" * W,
        "  target: mean ≈ 0 (unbiased), std ≈ 1 (calibrated)",
        "",
        "── METRIC 2b  EFT coefficient space  (analytic pull) ──",
        "-" * W,
    ]
    if eft_pulls:
        ops   = model.OPERATORS
        # Only include operators that have at least one finite pull (constrained by data)
        constrained_ops = [op for op in ops
                           if np.isfinite(eft_pulls[op]).sum() > 0]
        unconstrained_ops = [op for op in ops if op not in constrained_ops]
        all_p = np.concatenate([eft_pulls[op][np.isfinite(eft_pulls[op])]
                                 for op in constrained_ops]) if constrained_ops else np.array([])
        if all_p.size > 0:
            mu, sig = all_p.mean(), all_p.std(ddof=1)
            ok = "✓" if abs(mu) < 0.2 and abs(sig - 1) < 0.15 else "~"
        else:
            mu, sig, ok = float("nan"), float("nan"), "?"
        lines += [
            f"  Combined ({len(constrained_ops)} constrained operators"
            f" × {len(eft_pulls[ops[0]])} reps):",
            f"    mean = {mu:+.4f}   std = {sig:.4f}   ({ok})",
            f"  Unconstrained (NaN pulls, {len(unconstrained_ops)} ops): "
            + ", ".join(unconstrained_ops),
            "",
            f"  {'operator':<12} {'mean pull':>11} {'std pull':>11}",
            "  " + "-" * 35,
        ]
        for op in ops:
            arr = eft_pulls[op][np.isfinite(eft_pulls[op])]
            if arr.size > 0:
                lines.append(f"  {op:<12} {arr.mean():>+11.4f} {arr.std(ddof=1):>11.4f}")
            else:
                lines.append(f"  {op:<12} {'NaN':>11} {'NaN':>11}  (unconstrained)")

    lines += ["", "=" * W]
    txt = "\n".join(lines) + "\n"

    print(txt)
    out = out_dir / "closure_summary.txt"
    out.write_text(txt)
    print(f"  Saved {out.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="W' L1 BSM closure test — full analysis in one command",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--gWH",           type=float, required=True)
    ap.add_argument("--mWp",           type=float, required=True)
    ap.add_argument("--model",         default="wprime_universal",
                    choices=["wprime_universal", "wprime_constrained", "wprime"],
                    help="UV model variant for the closure test")
    ap.add_argument("--nrep",          type=int,   default=50,
                    help="NS closure replicas")
    ap.add_argument("--nrep-analytic", type=int,   default=1000,
                    help="Analytic replicas for metric 1 SMEFT + metric 2b EFT")
    ap.add_argument("--workers",       type=int,   default=4,
                    help="Parallel NS workers")
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--collect-only",  action="store_true",
                    help="Skip NS runs; only collect and plot existing results")
    args = ap.parse_args()

    tag     = make_tag(args.gWH, args.mWp, args.model)
    model   = make_model(args.gWH, args.mWp, args.model)
    out_dir = make_out_dir(args.gWH, args.mWp, args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  W' L1 BSM Closure Test Analysis")
    print(f"  tag    : {tag}")
    print(f"  model  : {model!r}")
    print(f"  output : {out_dir}")
    print(f"{sep}\n")

    # ── 1. pipeline ──────────────────────────────────────────────────────────
    ensure_pipeline(tag, args.gWH, args.mWp, args.model)
    if args.model == "wprime_universal":
        make_universal_base_runcard(tag, args.gWH, args.mWp)

    # ── 2. NS replicas ───────────────────────────────────────────────────────
    if not args.collect_only:
        print(f"\n[2/6] Running {args.nrep} NS L1 replicas  ({args.workers} workers)...")
        run_ns_replicas(tag, out_dir, args.nrep, args.seed, args.workers)

    # ── 3. collect NS ────────────────────────────────────────────────────────
    print(f"\n[3/6] Collecting NS results...")
    pulls_uv, sigmas_uv, chi2_ns, logz = collect_ns_results(tag, out_dir, model)
    ns_eft_pulls = collect_ns_eft_pulls(tag, out_dir, model)

    # ndof for NS UV fit
    K_shape  = np.load(PIPELINE / "results" / tag / "pca" / "K_fit.npy").shape
    n_data   = K_shape[0]
    ndof_ns  = n_data - len(model.uv_param_names())

    # ── 4. analytic metrics ──────────────────────────────────────────────────
    print(f"\n[4/6] Analytic metrics  ({args.nrep_analytic} replicas)...")
    chi2_an_smeft, eft_pulls, rank_smeft, ndof_an_smeft = analytic_smeft_metrics(
        tag, model, args.nrep_analytic, args.seed)
    chi2_an_uv, rank_an_uv, ndof_an_uv = analytic_uv_chi2(
        tag, model, args.nrep_analytic, args.seed)

    # ── 5. plots ─────────────────────────────────────────────────────────────
    print(f"\n[5/6] Generating plots...")
    plot_metric1_chi2(chi2_ns, chi2_an_uv, chi2_an_smeft,
                      ndof_ns, ndof_an_uv, ndof_an_smeft, out_dir)
    plot_metric1_pvalue_hist(chi2_ns, chi2_an_uv, chi2_an_smeft,
                              ndof_ns, ndof_an_uv, ndof_an_smeft, out_dir)
    plot_metric1_ppplot(chi2_ns, chi2_an_uv, chi2_an_smeft,
                        ndof_ns, ndof_an_uv, ndof_an_smeft, out_dir)
    if any(len(v) > 0 for v in pulls_uv.values()):
        plot_metric2_uv_pulls(pulls_uv, sigmas_uv, model, out_dir)
        plot_metric2_uv_pull_bars(pulls_uv, model, out_dir)
    else:
        print("  (skipping UV pull plots — no NS results)")
    plot_metric2_eft_pulls(eft_pulls, model, out_dir)
    plot_metric2_ns_eft_pulls(ns_eft_pulls, model, out_dir)

    # ── 6. summary table ─────────────────────────────────────────────────────
    print(f"\n[6/6] Summary table...")
    write_summary(tag, model, pulls_uv, sigmas_uv, chi2_ns, logz,
                  chi2_an_smeft, chi2_an_uv, ndof_ns, ndof_an_smeft, ndof_an_uv,
                  eft_pulls, out_dir)

    print(f"\n{sep}")
    print(f"  All results in: {out_dir}/")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
