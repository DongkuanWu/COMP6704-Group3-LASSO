"""Reproducible ISTA/FISTA/ADMM comparison for the same LASSO objective.

Run: python run_experiments.py
Only NumPy and the Python standard library are required.
"""
from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import numpy as np

from solvers import MAX_ITER, TOL, admm, fista, ista, make_data

ROOT = Path(__file__).resolve().parents[1] / 'runs' / 'pilot'
SEEDS = range(10)
CASES = {
    "baseline": (96, 192, 10, 0.02, 0.05),
    "fewer_measurements": (64, 192, 10, 0.02, 0.05),
    "less_sparse": (96, 192, 24, 0.02, 0.05),
    "more_noise": (96, 192, 10, 0.08, 0.05),
    "lower_lambda": (96, 192, 10, 0.02, 0.01),
    "higher_lambda": (96, 192, 10, 0.02, 0.15),
}


def write_csv(path, rows, names):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader(); writer.writerows(rows)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    raw = []; traces = []
    for case, (m, n, s, sigma, lam_factor) in CASES.items():
        for seed in SEEDS:
            A, b, truth, lam, L, scale = make_data(m, n, s, sigma, lam_factor, seed)
            for name, method in (("ISTA", ista), ("FISTA", fista), ("ADMM", admm)):
                x, trace, setup = method(A, b, lam, L, scale)
                final = trace[-1]
                raw.append(dict(case=case, seed=seed, algorithm=name, m=m, n=n,
                                sparsity=s, noise_sigma=sigma, lambda_factor=lam_factor, lambda_value=lam,
                                iterations=final[0], seconds=final[1], setup_seconds=setup,
                                objective=final[2], pg_residual=final[3],
                                relative_reconstruction_error=float(np.linalg.norm(x-truth)/np.linalg.norm(truth)),
                                converged=bool(final[3] <= TOL and (name != "ADMM" or
                                     (final[4] <= TOL and final[5] <= TOL)))))
                for point in trace:
                    traces.append(dict(case=case, seed=seed, algorithm=name, iteration=point[0],
                                       seconds=point[1], objective=point[2], pg_residual=point[3],
                                       primal_residual=point[4] if name == "ADMM" else "",
                                       dual_residual=point[5] if name == "ADMM" else ""))
                print(case, seed, name, final[0], round(final[1], 4), f"res={final[3]:.2e}", flush=True)
    write_csv(ROOT/'results_raw.csv', raw, list(raw[0]))
    write_csv(ROOT/'traces.csv', traces, list(traces[0]))
    summary = []
    for case in CASES:
        for name in ("ISTA", "FISTA", "ADMM"):
            a = [r for r in raw if r['case'] == case and r['algorithm'] == name]
            summary.append(dict(case=case, algorithm=name, runs=len(a),
                converged=sum(r['converged'] for r in a),
                mean_seconds=np.mean([r['seconds'] for r in a]),
                sd_seconds=np.std([r['seconds'] for r in a], ddof=1),
                mean_iterations=np.mean([r['iterations'] for r in a]),
                mean_objective=np.mean([r['objective'] for r in a]),
                mean_pg_residual=np.mean([r['pg_residual'] for r in a]),
                mean_relative_reconstruction_error=np.mean([r['relative_reconstruction_error'] for r in a])))
    write_csv(ROOT/'results_summary.csv', summary, list(summary[0]))
    (ROOT/'run_metadata.json').write_text(json.dumps(dict(python=platform.python_version(),
        numpy=np.__version__, platform=platform.platform(), tolerance=TOL,
        max_iterations=MAX_ITER, seeds=list(SEEDS), cases=CASES,
        lambda_rule='lambda_factor * ||A.T @ b||_infinity', admm_rho=1.0), indent=2), encoding='utf-8')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()

