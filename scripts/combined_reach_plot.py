"""
Combined W' + Z' discovery reach plot.

Plots the 5sigma UV coupling discovery contour for both models
on a single (mass, coupling) figure, using the analytic scan tables.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from pathlib import Path

PIPELINE = Path(__file__).parent.parent

XLIM = (1.0, 13.0)   # TeV
YLIM = (0.03, 0.22)  # coupling


def load_scan(path):
    data = np.loadtxt(path, comments="#")
    return {
        "coupling": data[:, 0],
        "mass":     data[:, 1],
        "sigma_uv": data[:, 6] if data.shape[1] > 6 else data[:, 5],
    }


def build_grid(scan, clip_sigma=12.0, m_extend=None):
    """
    Build interpolated sigma grid on a fine (coupling, mass) mesh.

    If m_extend > scan mass max, appends extra mass points using 1/m^2
    power-law extrapolation calibrated to the last scan column.
    """
    couplings = np.unique(scan["coupling"])
    masses    = np.unique(scan["mass"])
    sigma_grid = np.clip(
        scan["sigma_uv"].reshape(len(couplings), len(masses)), 0, clip_sigma)

    if m_extend is not None and m_extend > masses[-1]:
        m_last  = masses[-1]
        s_last  = sigma_grid[:, -1]          # sigma at last scan mass, all g
        m_extra = np.linspace(m_last, m_extend, 8)[1:]   # skip m_last itself
        # sigma ∝ 1/m^2 at fixed coupling
        s_extra = s_last[:, None] * (m_last / m_extra[None, :]) ** 2
        s_extra = np.clip(s_extra, 0, clip_sigma)
        masses     = np.concatenate([masses, m_extra])
        sigma_grid = np.hstack([sigma_grid, s_extra])

    sigma_grid = gaussian_filter(sigma_grid, sigma=0.7)
    interp = RegularGridInterpolator(
        (couplings, masses), sigma_grid,
        method="linear", bounds_error=False, fill_value=0.0)

    c_fine = np.linspace(couplings.min(), couplings.max(), 600)
    m_fine = np.linspace(masses.min(),    masses.max(),    600)
    C, M   = np.meshgrid(c_fine, m_fine, indexing="ij")
    S      = interp((C, M))
    return C, M, S


def mass_reach_at_g(scan, g_ref=0.12):
    mask   = np.abs(scan["coupling"] - g_ref) < 0.01
    masses = scan["mass"][mask]
    sigmas = scan["sigma_uv"][mask]
    above  = masses[sigmas >= 5]
    return above.max() if len(above) > 0 else float("nan")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None, help="Output directory")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else PIPELINE / "results" / "discovery_reach"
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_W = load_scan(PIPELINE / "results" /
                       "wprime_constrained_gwh012_mwp030_scan" /
                       "discovery_table.txt")
    scan_Z = load_scan(PIPELINE / "results" /
                       "zprime_constrained_gzh012_mzp070_scan" /
                       "discovery_table.txt")

    # W' scan only reaches 9 TeV; extend to 13 TeV via 1/m² extrapolation
    C_W, M_W, S_W = build_grid(scan_W, m_extend=XLIM[1])
    # Z' scan reaches 21 TeV; crop at 13 TeV via xlim, no extension needed
    C_Z, M_Z, S_Z = build_grid(scan_Z)

    fig, ax = plt.subplots(figsize=(10, 6))

    col_W = "#1f77b4"   # blue
    col_Z = "#d62728"   # red

    # ── filled discovery regions (5σ < sigma < clip) ─────────────────────────
    kw_fill = dict(levels=[5, 13], extend="neither")
    ax.contourf(M_W, C_W, S_W, colors=[col_W], alpha=0.12, **kw_fill)
    ax.contourf(M_Z, C_Z, S_Z, colors=[col_Z], alpha=0.12, **kw_fill)

    # ── 5σ discovery boundary ────────────────────────────────────────────────
    ax.contour(M_W, C_W, S_W, levels=[5], colors=[col_W], linewidths=2.5)
    ax.contour(M_Z, C_Z, S_Z, levels=[5], colors=[col_Z], linewidths=2.5)

    # ── 3σ evidence boundary (dashed) ───────────────────────────────────────
    ax.contour(M_W, C_W, S_W, levels=[3], colors=[col_W],
               linewidths=1.4, linestyles=["--"], alpha=0.55)
    ax.contour(M_Z, C_Z, S_Z, levels=[3], colors=[col_Z],
               linewidths=1.4, linestyles=["--"], alpha=0.55)

    # ── legend ───────────────────────────────────────────────────────────────
    legend_elements = [
        Patch(facecolor=col_W, alpha=0.3,
              label=r"W$'$ discovery region ($5\sigma$)"),
        Line2D([0], [0], color=col_W, lw=2.5,
               label=r"W$'$  $5\sigma$ boundary"),
        Patch(facecolor=col_Z, alpha=0.3,
              label=r"Z$'$ discovery region ($5\sigma$)"),
        Line2D([0], [0], color=col_Z, lw=2.5,
               label=r"Z$'$  $5\sigma$ boundary"),
        Line2D([0], [0], color="gray", lw=1.4, ls="--",
               label=r"$3\sigma$ evidence boundaries"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper right",
              framealpha=0.9, edgecolor="lightgray")

    # ── axes ─────────────────────────────────────────────────────────────────
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel(r"Heavy boson mass  (TeV)", fontsize=13)
    ax.set_ylabel(r"UV coupling  $g_{WH}$  or  $g_{ZH}$", fontsize=13)
    ax.set_title(
        r"FCC-ee discovery reach: W$'$ and Z$'$  (UV coupling method, 50 datasets)",
        fontsize=12, pad=8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax.tick_params(which="minor", length=3)

    plt.tight_layout()
    out = str(out_dir / "combined_discovery_reach")
    plt.savefig(f"{out}.png", bbox_inches="tight", dpi=150)
    plt.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"Saved: {out}.png")
    plt.close()

    print("\n5σ UV coupling mass reach at g=0.12:")
    for model, scan in [("W'", scan_W), ("Z'", scan_Z)]:
        m_reach = mass_reach_at_g(scan)
        print(f"  {model}: {m_reach:.1f} TeV  (from scan grid)")


if __name__ == "__main__":
    main()
