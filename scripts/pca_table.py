"""
PCA direction table for a given W' model point.

For each PCA mode k it reports:
  - eigenvalue lambda_k  (absolute Fisher information in that direction)
  - signal fraction f_k  = (e_k^T g)^2 / lambda_k  (q contribution of mode k)
  - cumulative q_k and sigma_k
  - operator composition: top-3 components of e_k shown as percentages

Usage:
    python scripts/pca_table.py                  # gWH=0.12 allg=0.12 mWp=1 TeV
    python scripts/pca_table.py --nmodes 8       # show 8 modes
    python scripts/pca_table.py --latex          # print LaTeX table

"""
import sys, os, argparse
import numpy as np
from pathlib import Path
from scipy.stats import chi2 as chi2_dist, norm as norm_dist

PIPELINE = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "scripts"))

from run_pipeline import _build_K_Ci, _build_delta, _q_to_sigma, CFG
from models.wprime import WPrimeModel

PROJ_DIR = str(PIPELINE / "projections" / "wprime_gwh012_allg012_mwp010")

# LaTeX names for operators (for pretty printing)
OP_LATEX = {
    "OpBox":   r"$\mathcal{O}_{\Box\varphi}$",
    "OpQM":    r"$\mathcal{O}_{\varphi Q}^{(-)}$",
    "O3pQ3":   r"$\mathcal{O}_{\varphi Q}^{(3)}$",
    "O3pl1":   r"$\mathcal{O}_{\varphi l_1}^{(3)}$",
    "O3pl2":   r"$\mathcal{O}_{\varphi l_2}^{(3)}$",
    "O3pl3":   r"$\mathcal{O}_{\varphi l_3}^{(3)}$",
    "OQQ1":    r"$\mathcal{O}_{QQ}^{(1)}$",
    "OQQ8":    r"$\mathcal{O}_{QQ}^{(8)}$",
    "Oll1221": r"$\mathcal{O}_{ll,1221}$",
    "Oll1111": r"$\mathcal{O}_{ll,1111}$",
    "Obp":     r"$\mathcal{O}_{\varphi b}$",
    "Otp":     r"$\mathcal{O}_{\varphi t}$",
    "Otap":    r"$\mathcal{O}_{\varphi\tilde{t}}$",
    "Op":      r"$\mathcal{O}_{\varphi}$",
    "OQl13":   r"$\mathcal{O}_{Ql,13}$",
    "OQl1M":   r"$\mathcal{O}_{Ql,1M}$",
    "Oll1122": r"$\mathcal{O}_{ll,1122}$",
    "Oll1133": r"$\mathcal{O}_{ll,1133}$",
    "Oll1331": r"$\mathcal{O}_{ll,1331}$",
    "Oll2222": r"$\mathcal{O}_{ll,2222}$",
    "Oll2332": r"$\mathcal{O}_{ll,2332}$",
    "Oll2233": r"$\mathcal{O}_{ll,2233}$",
    "Oll3333": r"$\mathcal{O}_{ll,3333}$",
    "OQl3M":   r"$\mathcal{O}_{Ql,3M}$",
    "OQl33":   r"$\mathcal{O}_{Ql,33}$",
}


def op_label(name: str, latex: bool) -> str:
    if latex:
        return OP_LATEX.get(name, name)
    return name


def compute_pca_table(model, proj_dir, n_modes=10):
    ops   = model.OPERATORS
    K, Ci = _build_K_Ci(ops)
    delta = _build_delta(proj_dir)

    F = K.T @ Ci @ K
    g = K.T @ Ci @ delta

    evals, evecs = np.linalg.eigh(F)
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]   # columns are eigenvectors

    # signal projections per mode
    proj  = np.array([(evecs[:, k] @ g)**2 / max(evals[k], 1e-30)
                      for k in range(len(evals))])

    rows = []
    q_cum = 0.0
    for k in range(min(n_modes, len(evals))):
        q_k    = proj[k]
        q_cum += q_k
        sigma_k = _q_to_sigma(q_cum, k + 1)

        # eigenvector composition: |e_{k,i}|^2 as % of total
        vec    = evecs[:, k]
        comp   = vec**2 * 100.0          # percentage contribution of each op
        order  = np.argsort(comp)[::-1]  # descending
        top3   = [(ops[i], comp[i]) for i in order[:3] if comp[i] > 0.5]

        rows.append({
            "k":       k + 1,
            "lambda":  evals[k],
            "q_k":     q_k,
            "q_cum":   q_cum,
            "sigma":   sigma_k,
            "top3":    top3,
        })

    return rows, ops, evecs, evals, g


def print_text_table(rows):
    header = f"{'k':>3}  {'λ_k':>10}  {'q_k':>8}  {'q_cum':>8}  {'σ(k)':>7}  Top operators"
    print(header)
    print("-" * 85)
    for r in rows:
        top = "  ".join(f"{op}({pct:.0f}%)" for op, pct in r["top3"])
        print(f"{r['k']:>3}  {r['lambda']:>10.3f}  {r['q_k']:>8.3f}  "
              f"{r['q_cum']:>8.3f}  {r['sigma']:>7.2f}σ  {top}")


def print_latex_table(rows):
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\renewcommand{\arraystretch}{1.2}")
    print(r"\begin{tabular}{ccccl}")
    print(r"\hline")
    print(r"\hline")
    print(r"Mode $k$ & $\lambda_k$ & $q_k$ & $\sigma_{\rm cum}(k)$ & "
          r"Dominant operators \\")
    print(r"\hline")
    for r in rows:
        top = ", ".join(
            f"{OP_LATEX.get(op, op)} ({pct:.0f}\\%)"
            for op, pct in r["top3"]
        )
        print(f"{r['k']} & {r['lambda']:.2f} & {r['q_k']:.2f} & "
              f"{r['sigma']:.2f} & {top} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{PCA modes of the Fisher information matrix $F = K^T C^{-1} K$ "
          r"for the W$'$ model with $g_{WH}=0.12$, $m_{W'}=1\,\text{TeV}$. "
          r"$\lambda_k$ is the $k$-th eigenvalue (descending), $q_k = "
          r"(\mathbf{e}_k^T \mathbf{g})^2/\lambda_k$ is the signal "
          r"contribution of mode $k$, $\sigma_{\rm cum}(k)$ is the "
          r"cumulative significance summing modes $1\ldots k$, and the "
          r"dominant operators show the squared eigenvector components "
          r"exceeding 0.5\%.}")
    print(r"\label{tab:pca_modes}")
    print(r"\end{table}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nmodes", type=int, default=10, help="Number of modes to show")
    p.add_argument("--latex",  action="store_true",  help="Output LaTeX table")
    p.add_argument("--proj",   default=PROJ_DIR,     help="Projection directory")
    args = p.parse_args()

    model = WPrimeModel(gWH=0.12, gWLf11=0.04, gWLf22=0.04,
                        gWLf33=0.04, gWqf33=0.04, mWp=1.0)

    print(f"Model : {model}")
    print(f"Ops   : {len(model.OPERATORS)} operators")
    print(f"Proj  : {args.proj}\n")

    rows, ops, evecs, evals, g = compute_pca_table(model, args.proj, args.nmodes)

    if args.latex:
        print_latex_table(rows)
    else:
        print_text_table(rows)

    # also print the full cumulative significance at the maximum
    q_total = sum(r["q_k"] for r in rows)
    sigma_total = _q_to_sigma(q_total, len(rows))
    print(f"\nCumulative q (k={len(rows)}): {q_total:.3f}  →  σ = {sigma_total:.2f}")


if __name__ == "__main__":
    main()
