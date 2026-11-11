"""Extra report figures and a shortened robustness table, built only from stored data and the stored test
outputs (results/weekly_returns.csv, results/tables/robustness.tex). It does not re-run or change the test.
Usage: python diagnostics/report_figures.py"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.config import OUT, ROOT, TABLE  # noqa: E402
from pipeline.data import load_universe  # noqa: E402
from pipeline.news import attach_scores, load_headlines  # noqa: E402
from pipeline.outputs import COLORS, DATA_BLUE, REFERENCE, plt, save  # noqa: E402

DROP_ROWS = ("Costs 0 bp per trade", "Costs 5 bp per trade")   # shown in the cost figure instead


def robustness_main():
    lines = (TABLE / "robustness.tex").read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if not line.startswith(DROP_ROWS)]
    (TABLE / "robustness_main.tex").write_text("\n".join(kept) + "\n", encoding="utf-8")


def uncertainty_main():
    """Plain-language version of the bootstrap table: each difference with its 95% range and a yes/no verdict."""
    import json
    boot = json.loads((OUT / "analysis.json").read_text())["bootstrap"]
    names = {"Combined": "Price + sentiment", "Price": "price only", "Sentiment": "Sentiment-driven", "EW": "equal weight",
             "MinVar": "minimum variance", "Index": "MSCI World"}
    order = ["Combined|Price", "Sentiment|EW", "Sentiment|MinVar", "Sentiment|Index", "Combined|EW", "Price|EW"]

    def cell(v, scale, digits):
        d, (lo, hi) = v["difference"] * scale, (x * scale for x in v["ci"])
        return f"{d:+.{digits}f} ({lo:+.{digits}f} to {hi:+.{digits}f})"

    rows = []
    for key in order:
        v = boot[key]
        x, y = key.split("|")
        significant = any(v[m]["ci"][0] > 0 or v[m]["ci"][1] < 0 for m in ("cagr", "sharpe", "max_drawdown"))
        rows.append(f"{names[x][0].upper() + names[x][1:]} vs {names[y]} & {cell(v['cagr'], 100, 2)} & {cell(v['sharpe'], 1, 2)} & "
                    f"{cell(v['max_drawdown'], 100, 1)} & {'Yes' if significant else 'No'} \\\\")
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Comparison (A vs B) & Return (pp a year) & Sharpe ratio & Drawdown (pp) & Significant? \\",
             r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    (TABLE / "uncertainty_main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def performance_main():
    """Performance table with the alpha t-statistic to two decimals."""
    import json
    alpha = json.loads((OUT / "analysis.json").read_text())["index_alpha"]
    labels = {"Sentiment-driven": "Sentiment", "Price + sentiment": "Combined", "Price only": "Price",
              "Minimum variance": "MinVar", "Equal weight": "EW"}
    out = []
    for line in (TABLE / "performance.tex").read_text(encoding="utf-8").splitlines():
        name = line.split(" & ")[0]
        if name in labels:
            a = alpha[labels[name]]
            cells = line.split(" & ")
            cells[-2] = f"{a['alpha'] * 100:+.2f} ({a['alpha_t']:+.2f})"
            line = " & ".join(cells)
        out.append(line)
    (TABLE / "performance_main.tex").write_text("\n".join(out) + "\n", encoding="utf-8")


def universe_main():
    """Universe table with full company names (the frozen output truncates them)."""
    import csv
    names = {r["ticker"]: r["name"] for r in csv.DictReader((ROOT / "data" / "universe.csv").open(encoding="utf-8"))}
    out = []
    for line in (TABLE / "universe.tex").read_text(encoding="utf-8").splitlines():
        cells = line.split(" & ")
        if len(cells) == 6:
            for k in (0, 3):
                ticker = cells[k].replace("\\_", "_").strip()
                if ticker in names:
                    cells[k + 1] = names[ticker].replace("&", "\\&")
            line = " & ".join(cells)
        out.append(line)
    (TABLE / "universe_main.tex").write_text("\n".join(out) + "\n", encoding="utf-8")


def allocation_mechanism():
    """Left: the measured skill that scales each signal's views. Right: where the return differences come from."""
    f = pd.read_csv(OUT / "weekly_returns.csv", parse_dates=["week"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.9, 2.5), gridspec_kw={"width_ratios": [1.25, 1]})
    for key, label in (("Price", "Price only"), ("Combined", "Price + sentiment"), ("Sentiment", "Sentiment-driven")):
        left.plot(f.week, f[f"skill_{key}"], color=COLORS[key], label=label, lw=1.3)
    left.set_ylabel("Measured skill (trailing IC)")
    left.legend(fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=0.8)
    pairs = (("Combined", "Price", "Price + sentiment\nvs price only"), ("Sentiment", "EW", "Sentiment-driven\nvs equal weight"))
    macros = {}
    parts = ("Before costs", "Cost saving", "After costs")
    width = 0.26
    for j, (x, y, _) in enumerate(pairs):
        gross = (f[f"gross_{x}"] - f[f"gross_{y}"]).mean() * 52 * 100
        saving = (f[f"cost_{y}"] - f[f"cost_{x}"]).mean() * 52 * 100
        net = (f[x] - f[y]).mean() * 52 * 100
        macros.update({f"Gross{x}{y}": f"{gross:+.2f}", f"Saving{x}{y}": f"{saving:+.2f}", f"Net{x}{y}": f"{net:+.2f}"})
        for k, (value, colour) in enumerate(zip((gross, saving, net), (REFERENCE, DATA_BLUE, "#3D3D3D"))):
            bar = right.bar(j + (k - 1) * width, value, width, color=colour, label=parts[k] if j == 0 else None)
            right.annotate(f"{value:+.2f}", (bar[0].get_x() + width / 2, value), ha="center",
                           va="bottom" if value >= 0 else "top", fontsize=6.5, xytext=(0, 2 if value >= 0 else -2),
                           textcoords="offset points")
    right.axhline(0, color="black", lw=0.8)
    right.set_xticks(range(len(pairs)), [p[2] for p in pairs], fontsize=7)
    right.set_ylabel("Mean difference (pp a year)")
    right.legend(fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=0.6)
    right.margins(y=0.2)
    fig.tight_layout()
    save(fig, "allocation_mechanism")
    zero = {k: f"{(f[f'skill_{k}'] == 0).mean() * 100:.0f}" for k in ("Price", "Combined", "Sentiment")}
    macros.update({f"ZeroSkill{k}": v + r"\%" for k, v in zero.items()})
    write_macros(macros)


def write_macros(new):
    path = TABLE / "extra_numbers.tex"
    old = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            name, value = line[len("\\newcommand{\\"):].split("}{", 1)
            old[name] = value[:-1]
    old.update(new)
    path.write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in old.items()) + "\n", encoding="utf-8")


def tone():
    """Data understanding: the distribution of headline tone and its average by region and year."""
    universe = load_universe()
    region = {f["ticker"]: f["region"] for f in universe}
    records = attach_scores(load_headlines(universe, {}), {})
    s = np.array([r["tone"] for r in records])
    by = defaultdict(list)
    for r in records:
        by[(region[r["ticker"]], int(r["date"][:4]))].append(r["tone"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.9, 2.4), gridspec_kw={"width_ratios": [1, 1.25]})
    left.hist(s, bins=40, color=DATA_BLUE)
    left.set_xlabel("Headline tone $s=p^{+}-p^{-}$")
    left.set_ylabel("Headlines")
    left.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    for name, colour in (("United States", COLORS["EW"]), ("Euro area", COLORS["Sentiment"])):
        years = sorted(y for (g, y) in by if g == name and y >= 2011)
        right.plot(years, [np.mean(by[(name, y)]) for y in years], marker="o", ms=3, color=colour, label=name)
    right.axhline(0, color="black", lw=0.8)
    right.set_ylabel("Mean headline tone")
    right.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "tone")
    share = {k: float(np.mean(v)) for k, v in (("positive", s > 0.5), ("negative", s < -0.5))}
    write_macros({"TonePositiveShare": f"{share['positive'] * 100:.0f}\\%", "ToneNegativeShare": f"{share['negative'] * 100:.0f}\\%"})
    print(f"tone: mean {s.mean():+.3f}, positive {share['positive']:.1%}, negative {share['negative']:.1%}")


def decision_sides():
    """Post-test diagnostic: the before-cost return gain of price + sentiment over price only, split into the price
    portfolio's buy decisions (overweight vs equal weight) and sell decisions (underweight), from the stored weights."""
    from pipeline.run import cached
    ctx = cached("context", lambda: None)
    index = {ctx["decisions"][i]["week"]: i for i in ctx["weeks"]}
    w = pd.read_csv(OUT / "weights.csv")
    w = w[w.strategy.isin(["Price", "Combined"])].pivot_table(index=["week", "ticker"], columns="strategy",
                                                                values="weight").reset_index()
    rows = []
    for week, g in w.groupby("week"):
        r = np.array([ctx["returns"][index[week]][t] for t in g.ticker])
        rel, n = r - r.mean(), len(g)
        price, change = g.Price.values, g.Combined.values - g.Price.values
        rec = {}
        for side, mask, sign in (("Buy", price > 1 / n + 1e-9, 1), ("Sell", price < 1 / n - 1e-9, -1)):
            rec[f"{side}Gain"] = change[mask] @ rel[mask]
            resized = mask & (np.abs(change) > 1e-4)
            larger = np.sign(change[resized]) * sign > 0          # sentiment made the bet larger
            right = np.sign(rel[resized]) * sign > 0              # the stock moved the decision's way
            rec[f"{side}Good"] = np.mean(larger == right) if resized.any() else np.nan
            rec[f"{side}LargerRight"] = right[larger].mean() if larger.any() else np.nan
        rows.append(rec)
    f = pd.DataFrame(rows)

    def hac_t(x, lags=13):
        x = np.asarray(x, float)
        e, T = x - x.mean(), len(x)
        v = e @ e / T + sum(2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / T for k in range(1, lags + 1))
        return x.mean() / np.sqrt(v / T)

    macros = {}
    for side in ("Buy", "Sell"):
        macros.update({f"Side{side}Gain": f"{f[f'{side}Gain'].mean() * 52 * 100:+.2f}",
                       f"Side{side}GainT": f"{hac_t(f[f'{side}Gain']):.2f}",
                       f"Side{side}Good": f"{f[f'{side}Good'].mean() * 100:.1f}\\%",
                       f"Side{side}LargerRight": f"{f[f'{side}LargerRight'].mean() * 100:.1f}\\%"})
    write_macros(macros)
    print(macros)


if __name__ == "__main__":
    robustness_main()
    uncertainty_main()
    performance_main()
    universe_main()
    allocation_mechanism()
    decision_sides()
    if "--skip-tone" not in sys.argv:
        tone()
