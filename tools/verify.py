"""Recompute the shared study's key statistics from individual runs."""
import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def verify(folder):
    def rows(name):
        with (folder / f"extended_{name}.csv").open(encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    metadata = json.loads((folder / "extended_metadata.json").read_text(encoding="utf-8"))
    cases = metadata["cases"]
    seeds = metadata["seeds"]
    names = ("ISTA", "FISTA", "ADMM")
    raw = rows("all_runs")
    expected = set(itertools.product(cases, seeds, names, range(metadata["repeats"])))
    observed = {(r["case"], int(r["seed"]), r["algorithm"], int(r["repeat"])) for r in raw}
    assert len(raw) == len(expected) and observed == expected, "Missing or duplicate runs"
    times = defaultdict(list)
    objectives = defaultdict(set)
    for r in raw:
        assert r["converged"] == "True" and float(r["pg_residual"]) <= metadata["tolerance"]
        times[r["case"], int(r["seed"]), r["algorithm"]].append(float(r["seconds"]))
        objectives[r["case"], r["seed"]].add(float(r["objective"]))

    medians = {key: np.median(values) for key, values in times.items()}
    per_seed = rows("per_seed")
    assert len(per_seed) == len(medians)
    for r in per_seed:
        key = (r["case"], int(r["seed"]), r["algorithm"])
        np.testing.assert_allclose(medians[key], float(r["median_seconds"]), rtol=1e-10)

    summaries = rows("summary")
    assert len(summaries) == len(cases) * len(names)
    rng = np.random.default_rng(20261030)
    def check_interval(values, record, mean_key, low_key, high_key):
        means = rng.choice(values, size=(metadata["bootstrap_resamples"], len(values))).mean(axis=1)
        bounds = np.quantile(means, [.025, .975])
        actual = [np.mean(values), *bounds]
        saved = [float(record[k]) for k in (mean_key, low_key, high_key)]
        np.testing.assert_allclose(saved, actual, rtol=1e-10, atol=1e-12)

    for r in summaries:
        case, alg = r["case"], r["algorithm"]
        check_interval([medians[case, s, alg] for s in seeds], r,
                       "mean_seed_median_seconds", "time_ci95_low", "time_ci95_high")
        errors = [float(v["relative_reconstruction_error"]) for v in per_seed
                  if v["case"] == case and v["algorithm"] == alg]
        check_interval(errors, r, "mean_reconstruction_error", "error_ci95_low", "error_ci95_high")

    for r in rows("paired_comparisons"):
        diffs = [medians[r["case"], s, r["algorithm_a"]] - medians[r["case"], s, r["algorithm_b"]] for s in seeds]
        check_interval(diffs, r, "mean_time_difference_seconds", "ci95_low", "ci95_high")
    spread = max(max(group) - min(group) for group in objectives.values())
    print(f"Verified {len(raw)} runs, {len(medians)} instance/solver groups and {len(summaries)} summaries.")
    print(f"Maximum objective spread within one instance: {spread:.4g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="?", default=ROOT / "data/reference/extended")
    verify(parser.parse_args().folder)

