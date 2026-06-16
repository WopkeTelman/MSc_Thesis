"""
closure_contour.py
------------------
2D NS-based closure metric scanner for the Composite Higgs Model.

Grid (defaults):
    g_rho  in [0.5, 1.0, 1.5, 2.0, 3.0]
    m_rho  in [20, 30, 50, 75, 100]  TeV
    = 25 grid points, each with --nrep NS replicas (default 50)

For each point:
  1. Run run_pipeline.py --model comphiggs --g_rho X --m_rho Y
                         --no-ns --skip-existing --no-report
     (generates BSM pseudo-data + PCA; skipped if cache already present)
  2. Run ns_l1_closure.py --tag TAG --nrep N
     (genuine NS L1 closure test: pull P=(g_fit-g_truth)/σ_g)

Collect σ_pull and μ_pull from every point and write two 2D colour maps.

Usage:
    # sequential (safe, slow — ~2.5 h per row on a single core)
    python scripts/closure_contour.py

    # 5 parallel workers  (~6 h total)
    python scripts/closure_contour.py --n-workers 5

    # subset for testing
    python scripts/closure_contour.py --g-rho 2.0 --m-rho 50 100 --nrep 10

    # dry-run: print grid, write runcards, no NS
    python scripts/closure_contour.py --dry-run

    # collect + plot whatever already finished
    python scripts/closure_contour.py --collect-only
"""
import sys, os, argparse, json, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))

# default grid
G_RHO_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]
M_RHO_GRID = [20.0, 30.0, 50.0, 75.0, 100.0]

# use the same interpreter that launched this script
PYTHON = sys.executable

RESULTS_DIR = PIPELINE / "results"


# ── helpers ───────────────────────────────────────────────────────────────────

def make_tag(g_rho: float, m_rho: float) -> str:
    return f"comphiggs_grho{int(round(g_rho * 100)):03d}_mrho{int(round(m_rho * 10)):03d}"


def pipeline_done(tag: str) -> bool:
    """True only when the full pipeline (including NS) has run — UVcoup runcard must exist."""
    rc_dir = RESULTS_DIR / tag / "runcards"
    return any(rc_dir.glob("*UVcoup.yaml")) if rc_dir.exists() else False


def closure_n_done(tag: str) -> int:
    fits_dir = RESULTS_DIR / f"{tag}_l1closure" / "fits"
    if not fits_dir.exists():
        return 0
    return sum(
        1 for d in fits_dir.iterdir()
        if d.is_dir() and (d / "fit_results.json").exists()
    )


def run_pipeline_step(tag: str, g_rho: float, m_rho: float) -> tuple[int, str]:
    cmd = [
        PYTHON,
        str(PIPELINE / "scripts" / "run_pipeline.py"),
        "--model",   "comphiggs",
        "--g_rho",   str(g_rho),
        "--m_rho",   str(m_rho),
        "--skip-existing",
        "--no-report",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PIPELINE))
    return r.returncode, r.stderr


def run_l1_closure(tag: str, nrep: int, seed: int) -> tuple[int, str]:
    cmd = [
        PYTHON,
        str(PIPELINE / "scripts" / "ns_l1_closure.py"),
        "--tag",  tag,
        "--nrep", str(nrep),
        "--seed", str(seed),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PIPELINE))
    return r.returncode, r.stderr


def collect_point(tag: str) -> dict | None:
    """
    Read fit_results.json files from the l1closure fits directory and compute
    the pull statistics for the primary UV parameter (g_rho for CHM).
    Returns None if no successful fits exist.
    """
    fits_dir = RESULTS_DIR / f"{tag}_l1closure" / "fits"
    if not fits_dir.exists():
        return None

    # decode model from tag:  comphiggs_grho{G*100:03d}_mrho{M*10:04d}
    parts = tag.split("_")
    g_rho = float(parts[1].replace("grho", "")) / 100.0
    m_rho = float(parts[2].replace("mrho", "")) / 10.0

    # CHM: single UV param g_rho, truth value = g_rho, no Z2 folding (Otap linear)
    uv_param = "g_rho"
    g_truth  = g_rho

    pulls = []
    for rep_dir in sorted(fits_dir.iterdir()):
        if not rep_dir.is_dir():
            continue
        path = rep_dir / "fit_results.json"
        if not path.exists():
            continue
        try:
            res     = json.load(open(path))
            samples = np.array(res.get("samples", {}).get(uv_param, []))
        except Exception:
            continue
        if len(samples) < 2:
            continue
        mu    = float(np.mean(samples))
        sigma = float(np.std(samples, ddof=1))
        if sigma > 0:
            pulls.append((mu - g_truth) / sigma)

    if not pulls:
        return None
    arr = np.array(pulls)
    return {"n": len(arr), "mu_pull": float(arr.mean()), "sigma_pull": float(arr.std(ddof=1))}


# ── grid-point worker ─────────────────────────────────────────────────────────

def _worker(args_tuple):
    """
    Run pipeline + closure for one (g_rho, m_rho) grid point.
    Designed to be called inside ProcessPoolExecutor.
    Returns (tag, g_rho, m_rho, result_dict_or_None).
    """
    g_rho, m_rho, nrep, seed, dry_run = args_tuple
    tag = make_tag(g_rho, m_rho)
    label = f"[g={g_rho:.1f} m={m_rho:.0f}]"

    # Step 1: run full pipeline (includes NS) if UVcoup runcard not yet present
    if not pipeline_done(tag):
        print(f"  {label} running full pipeline ...", flush=True)
        if not dry_run:
            ret, err = run_pipeline_step(tag, g_rho, m_rho)
            if ret != 0:
                print(f"  {label} pipeline FAILED (exit {ret})\n{err[-600:]}", flush=True)
                return tag, g_rho, m_rho, None
    else:
        print(f"  {label} pipeline already done, skipping", flush=True)

    # Step 2: run L1 NS closure test
    n_done = closure_n_done(tag)
    if n_done >= nrep:
        print(f"  {label} closure already complete ({n_done} reps), skipping", flush=True)
    elif dry_run:
        print(f"  {label} dry-run: would run {nrep - n_done} NS replicas", flush=True)
    else:
        print(f"  {label} running NS L1 closure ({n_done} done, targeting {nrep}) ...",
              flush=True)
        ret, err = run_l1_closure(tag, nrep, seed)
        if ret != 0:
            print(f"  {label} closure FAILED (exit {ret})\n{err[-600:]}", flush=True)
            # collect whatever finished before the failure
        else:
            print(f"  {label} closure done", flush=True)

    if dry_run:
        return tag, g_rho, m_rho, None

    res = collect_point(tag)
    if res:
        print(f"  {label} μ={res['mu_pull']:+.3f}  σ={res['sigma_pull']:.3f}  n={res['n']}",
              flush=True)
    else:
        print(f"  {label} no results to collect", flush=True)
    return tag, g_rho, m_rho, res


# ── plotting ──────────────────────────────────────────────────────────────────

def _panel(ax, g_arr, m_arr, data, cmap, vmin, vmax, cbar_label, fmt, n_map):
    """Draw one pcolormesh panel with annotated cell values."""
    ng, nm = len(g_arr), len(m_arr)

    # uniform integer grid — avoids distortion from non-uniform physical spacing
    GI, MI = np.meshgrid(np.arange(ng), np.arange(nm), indexing="ij")
    # edges for pcolormesh (cells centred on integer indices)
    ge = np.arange(-0.5, ng)
    me = np.arange(-0.5, nm)
    GE, ME = np.meshgrid(ge, me, indexing="ij")

    masked = np.ma.masked_invalid(data)
    im = ax.pcolormesh(GE, ME, masked, cmap=cmap,
                       vmin=vmin, vmax=vmax, shading="flat")

    for gi in range(ng):
        for mi in range(nm):
            v = data[gi, mi]
            if np.isnan(v):
                ax.text(gi, mi, "–", ha="center", va="center",
                        fontsize=10, color="gray")
                continue
            mid = 0.5 * (vmin + vmax)
            col = "white" if abs(v - mid) > 0.35 * (vmax - vmin) else "k"
            n   = n_map[gi, mi]
            ax.text(gi, mi, f"{fmt.format(v)}\n(n={n})",
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", color=col)

    cb = plt.colorbar(im, ax=ax)
    cb.set_label(cbar_label, fontsize=11)
    ax.set_xticks(range(ng))
    ax.set_xticklabels([f"{g:.1f}" for g in g_arr], fontsize=10)
    ax.set_yticks(range(nm))
    ax.set_yticklabels([f"{m:.0f}" for m in m_arr], fontsize=10)
    ax.set_xlabel(r"$g_\rho$", fontsize=13)
    ax.set_ylabel(r"$m_\rho$ [TeV]", fontsize=13)
    return im


def plot_contours(results, g_grid, m_grid, out_dir, nrep):
    g_arr = np.array(sorted(g_grid))
    m_arr = np.array(sorted(m_grid))
    ng, nm = len(g_arr), len(m_arr)

    mu_map    = np.full((ng, nm), np.nan)
    sigma_map = np.full((ng, nm), np.nan)
    n_map     = np.zeros((ng, nm), dtype=int)

    g_list = list(g_arr)
    m_list = list(m_arr)
    for tag, g, m, res in results:
        if res is None:
            continue
        gi = g_list.index(g)
        mi = m_list.index(m)
        mu_map[gi, mi]    = res["mu_pull"]
        sigma_map[gi, mi] = res["sigma_pull"]
        n_map[gi, mi]     = res["n"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 1 + nm * 1.4))

    # ── panel 1: σ_pull ──
    sigma_range   = np.nanmax(np.abs(sigma_map - 1.0)) if not np.all(np.isnan(sigma_map)) else 0.5
    sigma_half    = max(sigma_range, 0.3)
    _panel(axes[0], g_arr, m_arr, sigma_map,
           cmap="RdYlGn_r",
           vmin=max(0.0, 1.0 - sigma_half), vmax=1.0 + sigma_half,
           cbar_label=r"$\sigma_{\rm pull}$",
           fmt="{:.2f}", n_map=n_map)
    axes[0].set_title(r"Pull width  $\sigma_{\rm pull}$  (target: $\mathbf{1.00}$)",
                      fontsize=12, pad=8)

    # ── panel 2: μ_pull ──
    mu_half = max(np.nanmax(np.abs(mu_map)) if not np.all(np.isnan(mu_map)) else 0.3, 0.2)
    _panel(axes[1], g_arr, m_arr, mu_map,
           cmap="RdBu_r",
           vmin=-mu_half, vmax=mu_half,
           cbar_label=r"$\mu_{\rm pull}$",
           fmt="{:+.2f}", n_map=n_map)
    axes[1].set_title(r"Pull mean  $\mu_{\rm pull}$  (target: $\mathbf{0.00}$)",
                      fontsize=12, pad=8)

    fig.suptitle(
        fr"CHM NS-based L1 closure scan  ($n_{{\rm rep}}={nrep}$ per point)"
        "\n"
        r"Genuine closure: $\sigma_{\rm pull}\approx1.0$,  $\mu_{\rm pull}\approx0.0$",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    out = out_dir / "closure_contour"
    plt.savefig(f"{out}.png", bbox_inches="tight", dpi=150)
    plt.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}.png  /  {out}.pdf")


# ── summary table ─────────────────────────────────────────────────────────────

def print_table(results, g_grid, m_grid):
    g_arr = sorted(g_grid)
    m_arr = sorted(m_grid)
    # index
    idx = {(g, m): res for _, g, m, res in results}

    header = f"{'g_rho':>8}" + "".join(f"  m={m:.0f} TeV" for m in m_arr)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for g in g_arr:
        row = f"{g:>8.1f}"
        for m in m_arr:
            res = idx.get((g, m))
            if res is None:
                row += f"  {'—':>10}"
            else:
                row += f"  μ={res['mu_pull']:+.2f} σ={res['sigma_pull']:.2f}"
        print(row)
    print("=" * len(header))
    print("Expected for genuine closure: μ ≈ 0.00,  σ ≈ 1.00\n")


def save_json(results, out_dir):
    out = {}
    for tag, g, m, res in results:
        out[tag] = {
            "g_rho": g,
            "m_rho": m,
            **(res or {"n": 0, "mu_pull": None, "sigma_pull": None}),
        }
    path = out_dir / "closure_contour_summary.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"Saved: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="2D NS closure scan for Composite Higgs Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--nrep",         type=int, default=50,
                    help="NS replicas per grid point")
    ap.add_argument("--seed",         type=int, default=1000,
                    help="Base RNG seed (point i gets seed+i)")
    ap.add_argument("--n-workers",    type=int, default=1,
                    help="Parallel grid-point workers")
    ap.add_argument("--dry-run",      action="store_true",
                    help="Print grid and write runcards; do not run NS")
    ap.add_argument("--collect-only", action="store_true",
                    help="Skip all fitting; collect existing results and plot")
    ap.add_argument("--g-rho", nargs="+", type=float, default=None,
                    metavar="G",
                    help="Override g_rho grid (default: 0.5 1.0 1.5 2.0 3.0)")
    ap.add_argument("--m-rho", nargs="+", type=float, default=None,
                    metavar="M",
                    help="Override m_rho grid in TeV (default: 20 30 50 75 100)")
    args = ap.parse_args()

    g_grid = args.g_rho or G_RHO_GRID
    m_grid = args.m_rho or M_RHO_GRID

    out_dir = PIPELINE / "results" / "closure_contour"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_points = [(g, m) for g in sorted(g_grid) for m in sorted(m_grid)]
    n_pts = len(grid_points)

    print(f"CHM 2D NS closure scan — {n_pts} grid points")
    print(f"  g_rho grid : {sorted(g_grid)}")
    print(f"  m_rho grid : {sorted(m_grid)} TeV")
    print(f"  n_rep      : {args.nrep}")
    print(f"  seed base  : {args.seed}")
    print(f"  n_workers  : {args.n_workers}")
    print(f"  output dir : {out_dir}")
    print()

    # preview grid with Wilson coefficients
    from models.comphiggs import CompHiggsModel
    print(f"  {'tag':<35} {'OpBox':>10} {'Otap':>10} {'O3pQ3':>10} {'OpQM':>10}")
    print("  " + "-" * 75)
    for g, m in grid_points:
        tag  = make_tag(g, m)
        mdl  = CompHiggsModel(g_rho=g, m_rho=m)
        cs   = mdl.eft_coefficients()
        print(f"  {tag:<35} {cs['OpBox']:>10.5f} {cs['Otap']:>10.5f} "
              f"{cs['O3pQ3']:>10.5f} {cs['OpQM']:>10.5f}")
    print()

    if args.collect_only:
        results = []
        for g, m in grid_points:
            tag = make_tag(g, m)
            res = collect_point(tag)
            if res:
                print(f"  {tag}: μ={res['mu_pull']:+.3f}  σ={res['sigma_pull']:.3f}  n={res['n']}")
            else:
                print(f"  {tag}: no results")
            results.append((tag, g, m, res))
    else:
        worker_args = [
            (g, m, args.nrep, args.seed + i, args.dry_run)
            for i, (g, m) in enumerate(grid_points)
        ]
        results = []

        if args.n_workers > 1:
            print(f"Running {n_pts} grid points with {args.n_workers} parallel workers ...\n")
            with ProcessPoolExecutor(max_workers=args.n_workers) as exe:
                futures = {exe.submit(_worker, wa): wa for wa in worker_args}
                for fut in as_completed(futures):
                    tag, g, m, res = fut.result()
                    results.append((tag, g, m, res))
        else:
            print(f"Running {n_pts} grid points sequentially ...\n")
            for wa in worker_args:
                tag, g, m, res = _worker(wa)
                results.append((tag, g, m, res))

    if not args.dry_run:
        print_table(results, g_grid, m_grid)
        plot_contours(results, g_grid, m_grid, out_dir, args.nrep)
        save_json(results, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
