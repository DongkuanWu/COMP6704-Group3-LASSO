"""Extended paired study: 20 data seeds, five timing repeats per algorithm.

Run after installing NumPy: python extended_experiments.py
"""
import csv
import json
import platform
from pathlib import Path

import numpy as np

from solvers import make_data, ista, fista, admm, TOL

ROOT = Path(__file__).resolve().parents[1] / 'runs' / 'extended'
SEEDS = range(20)
REPEATS = 5
CASES = {
    'baseline': (96, 192, 10, .02, .05),
    'fewer_measurements': (64, 192, 10, .02, .05),
    'more_measurements': (128, 192, 10, .02, .05),
    'more_sparse': (96, 192, 5, .02, .05),
    'less_sparse': (96, 192, 24, .02, .05),
    'no_noise': (96, 192, 10, 0, .05),
    'more_noise': (96, 192, 10, .08, .05),
    'lower_lambda': (96, 192, 10, .02, .01),
    'higher_lambda': (96, 192, 10, .02, .15),
    'larger_problem': (192, 384, 20, .02, .05),
}
METHODS = {'ISTA': ista, 'FISTA': fista, 'ADMM': admm}


def save_csv(name, rows):
    with (ROOT/name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def boot_ci(values, rng, B=2000):
    values = np.asarray(values, dtype=float)
    samples = rng.choice(values, size=(B, len(values)), replace=True)
    return np.quantile(samples.mean(axis=1), [.025, .975])


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    all_runs = []; per_seed = []; convergence = []
    for case, config in CASES.items():
        m, n, s, sigma, factor = config
        for seed in SEEDS:
            A, b, truth, lam, L, scale = make_data(*config, seed)
            jobs = [(name, rep) for rep in range(REPEATS) for name in METHODS]
            np.random.default_rng(100000 + seed).shuffle(jobs)
            buckets = {name: [] for name in METHODS}
            for name, rep in jobs:
                x, trace, setup = METHODS[name](A, b, lam, L, scale)
                last = trace[-1]
                converged = last[3] <= TOL and (name != 'ADMM' or
                             (last[4] <= TOL and last[5] <= TOL))
                row = dict(case=case, seed=seed, repeat=rep, algorithm=name,
                    m=m, n=n, sparsity=s, noise_sigma=sigma, lambda_factor=factor,
                    lambda_value=lam, iterations=last[0], seconds=last[1],
                    setup_seconds=setup, objective=last[2], pg_residual=last[3],
                    relative_reconstruction_error=float(np.linalg.norm(x-truth)/np.linalg.norm(truth)),
                    converged=converged)
                all_runs.append(row); buckets[name].append(row)
                if seed == 0 and rep == 0:
                    for p in trace:
                        convergence.append(dict(case=case, algorithm=name, iteration=p[0],
                            seconds=p[1], objective=p[2], pg_residual=p[3],
                            primal_residual=p[4] if name == 'ADMM' else '',
                            dual_residual=p[5] if name == 'ADMM' else ''))
            for name, rows in buckets.items():
                first = rows[0]
                assert all(r['converged'] for r in rows)
                assert len({r['iterations'] for r in rows}) == 1
                per_seed.append(dict(case=case, seed=seed, algorithm=name,
                    median_seconds=float(np.median([r['seconds'] for r in rows])),
                    median_setup_seconds=float(np.median([r['setup_seconds'] for r in rows])),
                    within_seed_min_seconds=min(r['seconds'] for r in rows),
                    within_seed_max_seconds=max(r['seconds'] for r in rows),
                    iterations=first['iterations'], objective=first['objective'],
                    pg_residual=first['pg_residual'],
                    relative_reconstruction_error=first['relative_reconstruction_error']))
        print('finished', case, flush=True)
    save_csv('extended_all_runs.csv', all_runs)
    save_csv('extended_per_seed.csv', per_seed)
    save_csv('extended_traces.csv', convergence)
    rng = np.random.default_rng(20261030)
    summary = []
    for case in CASES:
        for name in METHODS:
            a = [r for r in per_seed if r['case'] == case and r['algorithm'] == name]
            sec = [r['median_seconds'] for r in a]
            err = [r['relative_reconstruction_error'] for r in a]
            low_t, high_t = boot_ci(sec, rng)
            low_e, high_e = boot_ci(err, rng)
            summary.append(dict(case=case, algorithm=name, seeds=len(a), repeats=REPEATS,
                mean_seed_median_seconds=float(np.mean(sec)), time_ci95_low=float(low_t),
                time_ci95_high=float(high_t), mean_iterations=float(np.mean([r['iterations'] for r in a])),
                mean_reconstruction_error=float(np.mean(err)), error_ci95_low=float(low_e),
                error_ci95_high=float(high_e), mean_pg_residual=float(np.mean([r['pg_residual'] for r in a]))))
    save_csv('extended_summary.csv', summary)
    paired = []
    for case in CASES:
        by = {(r['seed'],r['algorithm']): r for r in per_seed if r['case']==case}
        for a,b in [('FISTA','ISTA'),('ADMM','ISTA'),('ADMM','FISTA')]:
            diffs = [by[(seed,a)]['median_seconds']-by[(seed,b)]['median_seconds'] for seed in SEEDS]
            lo,hi = boot_ci(diffs,rng)
            paired.append(dict(case=case, algorithm_a=a, algorithm_b=b,
                mean_time_difference_seconds=float(np.mean(diffs)), ci95_low=float(lo), ci95_high=float(hi)))
    save_csv('extended_paired_comparisons.csv', paired)
    (ROOT/'extended_metadata.json').write_text(json.dumps(dict(python=platform.python_version(),
        numpy=np.__version__, platform=platform.platform(), cases=CASES, seeds=list(SEEDS),
        repeats=REPEATS, tolerance=TOL, bootstrap_resamples=2000,
        statistic='mean across 20 seeds of within-seed median of five timings',
        timing_order='shuffled within each case and seed',
        note='bootstrap intervals reflect instance variation; not hardware generalization'), indent=2), encoding='utf-8')
    print('DONE',len(all_runs),flush=True)


if __name__=='__main__':
    main()

