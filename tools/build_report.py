"""Build standalone HTML and editable SVG figures from the archived shared study."""
import csv
import math
from pathlib import Path
from html import escape
from statistics import mean, median
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/reference/extended"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def records(suffix):
    with (DATA / f"extended_{suffix}.csv").open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def label(x, y, value, anchor="start", size=13):
    return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-size="{size}">{escape(str(value))}</text>'


def line(x1, y1, x2, y2, color="#d5dce2", width=1):
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"/>'


def svg(parts, height, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}" '
            f'viewBox="0 0 820 {height}" font-family="Arial,sans-serif" fill="#203344">'
            f'<title>{escape(title)}</title><rect width="100%" height="100%" fill="white"/>'
            + "".join(parts) + "</svg>")


def interval_plot(rows, ticks, axis, title, digits=2):
    """One dot and percentile interval per row. Axis values are in the displayed units."""
    lower = min(ticks[0], *(r[2] for r in rows))
    upper = max(ticks[-1], *(r[3] for r in rows))
    def x(value):
        return 175 + (value - lower) / (upper - lower) * 470
    bottom = 42 + 32 * len(rows)
    parts = []
    for tick in ticks:
        parts += [line(x(tick), 22, x(tick), bottom, "#8796a0" if tick == 0 else "#e4e9ec"),
                  label(x(tick), bottom+19, tick, "middle")]
    for j, (name, value, lo, hi) in enumerate(rows):
        y = 38 + 32*j
        parts += [label(155, y+4, name, "end"), line(x(lo), y, x(hi), y, "#176b70", 2),
                  line(x(lo), y-4, x(lo), y+4, "#176b70"), line(x(hi), y-4, x(hi), y+4, "#176b70"),
                  f'<circle cx="{x(value):.2f}" cy="{y}" r="4" fill="#a92940"/>',
                  label(665, y+4, f"{value:.{digits}f}", size=13)]
    parts.append(label(410, bottom+47, axis, "middle", 15))
    return svg(parts, bottom+60, title)


def convergence():
    traces = [r for r in records("traces") if r["case"] == "baseline"]
    right = max(int(r["iteration"]) for r in traces)
    floor = math.floor(min(math.log10(float(r["pg_residual"])) for r in traces))
    def x(k): return 88 + 674*k/right
    def y(v): return 38 + 232*math.log10(v)/floor
    parts = []
    for k in range(0, right + 1, 25):
        parts += [line(x(k), 38, x(k), 270), label(x(k), 293, k, "middle")]
    for p in range(floor, 1):
        parts += [line(88, y(10**p), 762, y(10**p), "#c35243" if p == -4 else "#d5dce2"),
                  label(77, y(10**p)+4, f"1e{p}", "end")]
    for j,(name,color) in enumerate([("ISTA","#285a96"),("FISTA","#a92940"),("ADMM","#14766a")]):
        pts = " ".join(f'{x(int(r["iteration"])):.2f},{y(float(r["pg_residual"])):.2f}' for r in traces if r["algorithm"] == name)
        parts += [f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.4"/>',
                  line(390+115*j,15,415+115*j,15,color,3),label(421+115*j,20,name)]
    parts += [label(83,20,"Normalized residual"), label(410,321,"Iteration","middle")]
    return svg(parts,338,"Baseline convergence on seed zero")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    summaries = records("summary")
    labels = dict(baseline="Baseline", fewer_measurements="64 measurements", more_measurements="128 measurements",
                  more_sparse="5 nonzeros", less_sparse="24 nonzeros", no_noise="No noise", more_noise="Higher noise",
                  lower_lambda="Weaker penalty", higher_lambda="Stronger penalty", larger_problem="Double dimensions")
    def points(rows, key, lo, hi, factor, naming):
        return [(naming(r), *(float(r[k])*factor for k in (key,lo,hi))) for r in rows]
    baseline = [r for r in summaries if r["case"] == "baseline"]
    paired = [r for r in records("paired_comparisons") if r["algorithm_a"] == "FISTA" and r["algorithm_b"] == "ISTA"]
    fista = [r for r in summaries if r["algorithm"] == "FISTA"]
    figures = {
        "convergence": convergence(),
        "baseline": interval_plot(points(baseline,"mean_seed_median_seconds","time_ci95_low","time_ci95_high",1000,lambda r:r["algorithm"]),[0,20,40,60,80],"Elapsed time (ms)","Baseline timing with 95% bootstrap intervals"),
        "paired": interval_plot(points(paired,"mean_time_difference_seconds","ci95_low","ci95_high",1000,lambda r:labels[r["case"]]),[-6,-5,-4,-3,-2,-1,0,1],"FISTA minus ISTA (ms)","Paired timing differences"),
        "error": interval_plot(points(fista,"mean_reconstruction_error","error_ci95_low","error_ci95_high",100,lambda r:labels[r["case"]]),[0,5,10,15,20,25,30],"Mean relative reconstruction error (%)","Recovery error with 95% bootstrap intervals")
    }
    source = (ROOT / "tools/report_template.html").read_text(encoding="utf8")
    for name, graphic in figures.items():
        (FIGURES / f"{name}.svg").write_text(graphic, encoding="utf8")
        source = source.replace("{{"+name+"}}", graphic)
    groups = {(r["case"],r["algorithm"]):r for r in summaries}
    table = []
    for case, name in labels.items():
        row = [name]
        for alg in ("ISTA","FISTA","ADMM"):
            r = groups[case,alg]
            row += [f'{1000*float(r["mean_seed_median_seconds"]):.2f}',f'{float(r["mean_iterations"]):.1f}']
        row.append(f'{100*float(groups[case,"FISTA"]["mean_reconstruction_error"]):.2f}')
        table.append("<tr>"+"".join(f"<td>{v}</td>" for v in row)+"</tr>")
    source = source.replace("{{table}}","".join(table))
    by_seed = defaultdict(list)
    for r in records("all_runs"):
        if r["case"]=="baseline" and r["algorithm"]=="ADMM":
            by_seed[r["seed"]].append(r)
    central = [sorted(group,key=lambda r:float(r["seconds"]))[len(group)//2] for group in by_seed.values()]
    share = mean(float(r["setup_seconds"]) for r in central)/mean(float(r["seconds"]) for r in central)
    source = source.replace("{{share}}",f"{100*share:.1f}%")
    (REPORTS/"technical-report.html").write_text(source,encoding="utf8")
    print("Published four figures and a self-contained report.")


if __name__ == "__main__":
    main()

