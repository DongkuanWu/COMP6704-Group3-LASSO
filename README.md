# Group 3 · Convex Optimization for Sparse Signal Reconstruction

**Course:** COMP6704 — Advanced Topics in Optimization, The Hong Kong Polytechnic University.

**Repository:** https://github.com/DongkuanWu/COMP6704-Group3-LASSO

This repository compares ISTA, FISTA and ADMM on the same LASSO sparse signal reconstruction problem. We evaluate objective values, reconstruction error, iteration counts and elapsed time. Archived raw measurements are included so that the reported statistics can be checked before rerunning the solvers.

**Expanded technical report:** [LaTeX source](reports/COMP6704.tex), including eight analytical figures, with [the accompanying style file](reports/neurips_2026.sty). From `reports/`, run `pdflatex COMP6704.tex` twice. Alternatively, upload both files to Overleaf and select `COMP6704.tex` as the main document. Plot coordinates are embedded in the source; underlying measurements are in `data/reference/extended/`.

**Earlier HTML report:** [Self-contained technical report](reports/technical-report.html), with four figures and the complete ten-configuration comparison table. Download and open it in a browser to read or print it. Its text, figures and styles are bundled in one file.

## 1. Quick start

Clone the repository and open a terminal in its root directory:

```bash
git clone https://github.com/DongkuanWu/COMP6704-Group3-LASSO.git
cd COMP6704-Group3-LASSO
```

The archived experiments used Windows 11, Python 3.12.14 and NumPy 2.3.5. The requirements file pins NumPy. The experiments run on the CPU and require neither a GPU nor an external dataset. Python 3.12 is the verified Python version.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools/verify.py
.\.venv\Scripts\python.exe experiments/extended_experiments.py
.\.venv\Scripts\python.exe tools/verify.py runs/extended
```

### macOS / Linux

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tools/verify.py
.venv/bin/python experiments/extended_experiments.py
.venv/bin/python tools/verify.py runs/extended
```

Verification of the reference archive should report 3,000 runs, 600 instance/solver groups and 30 configuration/solver summaries. The maximum within-instance objective spread is approximately `2.299e-06`. Timings and their intervals will change on another machine. The verifier recomputes statistics from the dataset being checked.

## 2. Repository structure and entry points

```text
experiments/
  solvers.py                 Data generation, objectives, residuals and three solvers
  run_experiments.py         Pilot: 6 configurations × 10 instances × 3 solvers
  extended_experiments.py    Main: 10 configurations × 20 instances × 3 solvers × 5 repeats
data/reference/
  pilot/                    180 pilot runs, summaries, trajectories and metadata
  extended/                 3,000 main runs and the statistics used in the report
tools/
  verify.py                 Recompute grouped statistics and bootstrap intervals
  build_report.py           Build the earlier HTML report and four SVG figures
  report_template.html      English HTML report text and layout
reports/
  COMP6704.tex               Expanded report with eight figures
  neurips_2026.sty           LaTeX style used by the expanded report
  technical-report.html     Self-contained earlier report
  figures/                  Standalone figures for slides or documents
requirements.txt            Pinned NumPy version
runs/                       Generated local results; ignored by Git
```

With the virtual environment activated, the main commands are:

```bash
python experiments/run_experiments.py
python experiments/extended_experiments.py
python tools/verify.py runs/extended
python tools/build_report.py
```

The first two commands write to `runs/pilot/` and `runs/extended/`, respectively. Repeating a command updates that directory; copy previous outputs elsewhere to retain multiple executions. The HTML builder always reads `data/reference/extended/`, keeping its figures aligned with the archived presentation results. It uses only the Python standard library and can rebuild its figures offline. It does not regenerate the expanded LaTeX report.

## 3. Problem, inputs and outputs

The observation model is

$$b = Ax^\star + \varepsilon,$$

where $A\in\mathbb{R}^{m\times n}$, $b\in\mathbb{R}^m$, and the unknown signal $x^\star\in\mathbb{R}^n$ has $s$ nonzero coefficients. All solvers start from zero and solve

$$\min_{x\in\mathbb{R}^n}\;\frac12\lVert Ax-b\rVert_2^2+\lambda\lVert x\rVert_1,$$

with $\lambda=\alpha\lVert A^\top b\rVert_\infty$, $L=\lVert A\rVert_2^2$, and proximal-gradient step size $1/L$.

The true signal is used only to generate observations and evaluate reconstruction error. Solver inputs are $A$, $b$, $\lambda$, $L$ and the residual normalization factor. Each solver returns an estimated vector, a trajectory recorded at stopping checkpoints, and a separately recorded matrix preparation time. ADMM returns its split variable $z$. The setup field is zero for ISTA and FISTA; it specifically measures ADMM's inverse preparation stage.

| Method | Update | Computational implications |
|---|---|---|
| ISTA | Gradient step of size $1/L$, followed by soft thresholding at $\lambda/L$ | Each gradient evaluation requires two dense matrix-vector products |
| FISTA | Proximal-gradient step at an extrapolated point, followed by a momentum update | Extra vector operations can reduce the number of updates needed |
| ADMM | Split with $x=z$, then update $x$, $z$ and scaled dual variable $u$ | Fixed $\rho=1$; an explicit inverse is computed once per run and reused |

The shared stopping metric is the infinity norm of the proximal-gradient mapping, normalized by $\max(1,\lVert A^\top b\rVert_\infty)$. Its threshold is $10^{-4}$. ADMM must also satisfy its normalized primal and dual residual thresholds. Checks occur at iteration 1, every tenth iteration, and the limit of 1,200 iterations. Reported iteration counts therefore reflect this checkpoint spacing.

## 4. Data generation and configurations

Each instance draws a standard Gaussian matrix and normalizes its columns to unit Euclidean norm. The signal support contains $s$ positions sampled without replacement. Active coefficients have random signs and magnitudes uniformly distributed on $[0.8,1.2]$. Additive noise has standard deviation $\sigma$. Each configuration uses seeds 0–19; all solvers share the same input for a given seed.

| Configuration | m | n | s | σ | α |
|---|---:|---:|---:|---:|---:|
| baseline | 96 | 192 | 10 | 0.02 | 0.05 |
| fewer_measurements | 64 | 192 | 10 | 0.02 | 0.05 |
| more_measurements | 128 | 192 | 10 | 0.02 | 0.05 |
| more_sparse | 96 | 192 | 5 | 0.02 | 0.05 |
| less_sparse | 96 | 192 | 24 | 0.02 | 0.05 |
| no_noise | 96 | 192 | 10 | 0 | 0.05 |
| more_noise | 96 | 192 | 10 | 0.08 | 0.05 |
| lower_lambda | 96 | 192 | 10 | 0.02 | 0.01 |
| higher_lambda | 96 | 192 | 10 | 0.02 | 0.15 |
| larger_problem | 192 | 384 | 20 | 0.02 | 0.05 |

To extend the settings, edit `CASES` in the main experiment script. Each tuple has the order `(m, n, s, sigma, alpha)`. The size comparison increases $m$, $n$ and $s$ together, preserving sampling and sparsity ratios. Reusing a seed across different dimensions changes the sequence of random draws. The regularization comparison holds each seed's data fixed and changes only $\alpha$.

## 5. Interpreting millisecond-level timings

1. Within each instance, shuffle the execution order of all three solvers and their five repetitions. Record elapsed time using `time.perf_counter()`.
2. Take the median of five timings for each configuration/seed/solver combination, then average those medians across twenty seeds. The 3,000 runs cover 200 configuration/seed instances.
3. For paired comparisons, subtract the two solver medians on each shared instance. Bootstrap these twenty paired differences with 2,000 resamples and report the 2.5th and 97.5th percentiles of the resampled means. Negative FISTA-minus-ISTA differences favor FISTA.
4. Data generation and calculation of $L$ are outside the timer. ISTA/FISTA form $A^\top b$ before timing; ADMM includes $A^\top b$ and explicit inverse preparation. The intervals describe variation in this environment and synthetic instance distribution. No dedicated warm-up or CPU affinity control was used, and the exploratory configuration-level intervals have no multiple-comparison correction.

Historical metadata records the operating system, Python and NumPy versions. CPU model, BLAS thread count and background load were not recorded in the original experiment. Record these details and retain thread settings when collecting new timings.

## 6. Output data dictionary

| Main experiment file | Unit of observation | Main fields and units |
|---|---|---|
| `extended_all_runs.csv` | One execution; 3,000 rows | Key: `case`, `seed`, `algorithm`, `repeat`. `seconds` and `setup_seconds` are in seconds. Endpoint fields include `iterations`, `objective`, `pg_residual`, `relative_reconstruction_error` and `converged`. |
| `extended_per_seed.csv` | One configuration/seed/solver group; 600 rows | `median_seconds`, minimum/maximum repeat time, `median_setup_seconds` and final quality metrics. |
| `extended_summary.csv` | One configuration/solver group; 30 rows | Mean of twenty instance timing medians and its interval; mean iterations; mean reconstruction error and its interval. |
| `extended_paired_comparisons.csv` | Three ordered solver comparisons per configuration; 30 rows | Mean time difference `algorithm_a` minus `algorithm_b` and its 95% interval, in seconds. |
| `extended_traces.csv` | One checkpoint for seed 0 and repeat label 0 in each configuration/solver combination | `iteration`, `seconds`, `objective`, `pg_residual`; ADMM also records `primal_residual` and `dual_residual`. |
| `extended_metadata.json` | Run environment and experimental settings | Configurations, seeds, repeats, tolerance, bootstrap resample count and execution-order protocol. |

Reconstruction error is relative Euclidean error: a CSV value of `0.0846` means 8.46%. Plots and reports multiply seconds by 1,000 to display milliseconds. Setup-time and total-time medians can select different repetitions. The report's additive ADMM decomposition instead uses components from the same run selected by its median total time. Repeat label 0 is assigned before the execution order is shuffled.

The pilot archive contains `results_raw.csv`, `results_summary.csv`, `traces.csv` and `run_metadata.json`. It times each solver once per instance and preserves the initial stage of the study.

## 7. Reference results and scope

| Baseline metric | ISTA | FISTA | ADMM |
|---|---:|---:|---:|
| Mean iteration count | 148.0 | 95.0 | 40.5 |
| Mean of instance median times (ms) | 2.12 | 1.74 | 48.83 |
| Timing 95% interval (ms) | [1.84, 2.45] | [1.56, 1.97] | [29.70, 71.97] |
| Mean relative reconstruction error | 8.47% | 8.46% | 8.46% |

FISTA has a lower mean time than ISTA in nine of the ten shared configurations. With 128 measurements, ISTA is faster by 0.172 ms on average. All 3,000 runs meet their respective stopping conditions. The maximum within-instance spread in final objective values is approximately $2.30\times10^{-6}$. The reports examine timing decomposition, paired intervals, measurement conditions, noise and regularization sensitivity.

The FISTA presentation also contains a separate supplementary study of 2,000 runs. Its complete source and per-run measurements were unavailable in the assembled materials. This repository provides the runnable FISTA implementation used in the shared three-solver study; all three-way comparisons in the reports use that shared archive.

## 8. Reproducibility and extensions

`tools/verify.py` checks combination coverage, convergence status and within-instance timing medians. It reconstructs the summary and paired-comparison means and bootstrap intervals from individual runs. Floating-point libraries can produce small numerical differences; execution times also depend on hardware and system load.

Useful extensions include comparing ADMM's explicit inverse with Cholesky-based solves on paired instances, measuring cold setup separately from reuse across right-hand sides, and conducting a noise sweep with fixed $\lambda$. The latter would separate noise sensitivity from the current data-dependent regularization rule.

## 9. Contributors

Project contributors:

- [jiajieliu.pku@gmail.com](mailto:jiajieliu.pku@gmail.com)
- [wenzhao.dong@connect.polyu.hk](mailto:wenzhao.dong@connect.polyu.hk)
- [dwzh1677@gmail.com](mailto:dwzh1677@gmail.com)

## 10. Report update

The expanded LaTeX report includes LASSO optimality conditions, the critical regularization threshold, a soft-thresholding derivation, ISTA/FISTA objective-gap bounds, dense implementation costs, and an ADMM time-decomposition table. Remaining time after setup includes initialization and convergence checks; it is not an isolated iterative-loop benchmark. The report retains the archived total-time comparison and distinguishes observed results from untested factorization-reuse scenarios.
