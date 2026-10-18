"""Figures, LaTeX tables and the numbers quoted in the report (results/tables/numbers.tex)."""
import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, PercentFormatter, ScalarFormatter  # noqa: E402

from .config import FIG, LABELS, OUT, PARAMS, TABLE  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#B7BDC0",
    "axes.grid": True, "grid.color": "#E4E7E8", "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 1.5})
COLORS = {"Sentiment": "#2F7613", "Combined": "#8A6300", "Price": "#BB3A2E", "MinVar": "#64A8D1",
          "EW": "#243C64", "Index": "#D9A300"}
MODEL_COLORS = {"Ridge": "#7A7A7A", "Factor": "#8A6300", "ARIMA": "#64A8D1", "RandomForest": "#2F7613", "GBM": "#243C64",
                "LSTM": "#BB3A2E"}
DATA_BLUE, REFERENCE = "#36566F", "#A3B6C9"
SEQ = LinearSegmentedColormap.from_list("magnitude", ["#F2F4F5", "#BACCD7", DATA_BLUE])
DIV = LinearSegmentedColormap.from_list(
    "signed", [(0, "#BB3A2E"), (0.42, "#BF8179"), (0.5, "#F4F3EF"), (0.58, "#8EA77F"), (1, "#2F7613")])
PLAIN = {k: v.replace("\\&", "&") for k, v in LABELS.items()}
ORDER = ["Sentiment", "Combined", "Price", "MinVar", "EW", "Index"]
MODEL_NAMES = {"Ridge": "Ridge regression", "Factor": "Factor composite", "ARIMA": "ARIMA", "RandomForest": "Random forest",
               "GBM": "Gradient boosting", "LSTM": "LSTM"}
ILLUSTRATION = "BA"      # Boeing: the gauge example figure (chosen for its well-known 2019-2020 news shocks)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


def latex_table(name, headers, rows, alignment=None):
    alignment = alignment or "l" + "r" * (len(headers) - 1)
    lines = [r"\begin{tabular}{" + alignment + "}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    lines += [(r"\midrule" if row == "MIDRULE" else " & ".join(map(str, row)) + r" \\") for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLE / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pct(x, digits=2, sign=False):
    return f"{x * 100:{'+' if sign else ''}.{digits}f}"


def tex(s):
    return str(s).replace("&", r"\&").replace("%", r"\%")


def make_outputs(a, frame, weights, universe, prices, news, records, g, active, decisions):
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(frame.week)
    sel = a["selection"]
    m, macros = a["metrics"], {}
    test_weeks = [i for i, d in enumerate(decisions) if PARAMS["evaluation_start"] <= d["week"] <= PARAMS["evaluation_end"]]

    # ------------------------------------------------------------------ data
    au = a["audit"]
    stages = {"Retrieved items": au["headlines_retrieved"], "Unique per firm": au["headlines_unique"],
              "Within sample dates": au["headlines_in_range"], "Names the issuer": au["headlines_issuer"],
              "Not a market roundup": au["headlines_accepted"]}
    fig, ax = plt.subplots(figsize=(6.7, 2))
    ax.barh(list(stages), list(stages.values()), color=DATA_BLUE, height=0.55)
    for i, value in enumerate(stages.values()):
        ax.text(value * 1.01, i, f"{value:,}", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(stages.values()) * 1.18)
    ax.grid(False)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    save(fig, "retention")

    sector = {f["ticker"]: f["sector"] for f in universe}
    counts = Counter((sector[r["ticker"]], r["date"][:4]) for r in records)
    sectors = sorted({s for s, _ in counts})
    years = [str(y) for y in range(2011, 2026)]
    matrix = np.array([[counts.get((s, y), 0) for y in years] for s in sectors], dtype=float)
    per_firm = Counter(sector.values())
    matrix = matrix / np.array([per_firm[s] for s in sectors])[:, None] / 52
    fig, ax = plt.subplots(figsize=(6.7, 2.6))
    image = ax.imshow(matrix, cmap=SEQ, aspect="auto")
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5, color="white" if v > matrix.max() * 0.6 else "black")
    ax.set_xticks(range(len(years)), years, fontsize=7)
    ax.set_yticks(range(len(sectors)), [s.replace("Telecommunication Services", "Telecom services") for s in sectors], fontsize=7)
    ax.grid(False)
    ax.tick_params(length=0)
    bar = fig.colorbar(image, ax=ax, shrink=0.9, label="Headlines per firm-week")
    bar.outline.set_visible(False)
    save(fig, "coverage")

    # Gauge example.
    idx = [i for i, d in enumerate(decisions) if "2017-01-01" <= d["week"] <= "2021-12-31"]
    w = sel["sector_weight"]
    series = np.array([news[i][ILLUSTRATION][f"gauge_{w:g}"] for i in idx])
    count = np.array([news[i][ILLUSTRATION]["news_count"] for i in idx])
    when = pd.to_datetime([decisions[i]["week"] for i in idx])
    fig, axes = plt.subplots(2, 1, figsize=(6.7, 3.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    axes[0].plot(when, series, color=COLORS["Sentiment"], lw=1.2)
    for level, style in ((PARAMS["entry_band"], "-"), (PARAMS["exit_band"], ":")):
        for s in (1, -1):
            axes[0].axhline(s * level, color=REFERENCE, lw=0.8, ls=style)
    axes[0].fill_between(when, series, 0, where=np.abs(series) >= PARAMS["entry_band"], color=COLORS["Sentiment"], alpha=0.25)
    axes[0].set_ylabel("Sentiment gauge (s.d.)")
    axes[1].bar(when, count, width=6, color=DATA_BLUE)
    axes[1].set_ylabel("Headlines")
    save(fig, "gauge_example")

    # ------------------------------------------------------------------ validation
    cand = sel["price_candidates"]
    names = PARAMS["price_models"]
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.7), gridspec_kw={"width_ratios": [1.1, 1.4]})
    vals = [cand[n]["mean"] for n in names]
    axes[0].bar(range(len(names)), vals, yerr=[1.96 * cand[n]["se"] for n in names], capsize=2,
                color=[MODEL_COLORS[n] for n in names], error_kw={"lw": 0.8, "ecolor": "#555555"})
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(range(len(names)), [MODEL_NAMES[n].replace(" ", "\n") for n in names], fontsize=7)
    axes[0].set_ylabel("Mean weekly rank IC")
    for n in names:
        weeks = pd.to_datetime([decisions[i]["week"] for i, _ in cand[n]["weekly"]])
        axes[1].plot(weeks, np.cumsum([v for _, v in cand[n]["weekly"]]), color=MODEL_COLORS[n], label=MODEL_NAMES[n],
                     lw=1.8 if n == sel["price_model"] else 1.0)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("Cumulative rank IC")
    axes[1].xaxis.set_major_locator(matplotlib.dates.YearLocator())
    axes[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    axes[1].legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    save(fig, "model_selection")
    rows = [[(r"\textbf{" + MODEL_NAMES[n] + "}") if n == sel["price_model"] else MODEL_NAMES[n],
             f"{cand[n]['mean']:+.4f}", f"{cand[n]['t']:.2f}", pct(cand[n]["positive_share"], 1)] for n in names]
    rows.append("MIDRULE")
    gc = sel["gauge_candidates"]
    for key, label in (("0", "Sentiment gauge, firm only"), ("0.5", "Sentiment gauge, firm + sector")):
        if key in gc:
            bold = float(key) == sel["sector_weight"]
            rows.append([(r"\textbf{" + label + "}") if bold else label, f"{gc[key]['mean']:+.4f}",
                         f"{gc[key]['t']:.2f}", pct(gc[key]["positive_share"], 1)])
    latex_table("model_selection", ["Model", "Mean rank IC", "HAC $t$", "Weeks IC $>0$ (\\%)"], rows)

    # ------------------------------------------------------------------ sentiment and prices
    es = a["event_study"]
    days = np.array(es["days"])
    fig, ax = plt.subplots(figsize=(6.7, 2.6))
    for sign, label, color in ((1, "Positive sentiment events", COLORS["Sentiment"]),
                               (-1, "Negative sentiment events", COLORS["Price"])):
        e = es[str(sign)] if str(sign) in es else es[sign]
        if not e["mean"]:
            continue
        mean, se = np.array(e["mean"]), np.array(e["se"])
        ax.plot(days, mean * 100, color=color, label=f"{label} ($n$={e['n']:,})")
        ax.fill_between(days, (mean - 1.96 * se) * 100, (mean + 1.96 * se) * 100, color=color, alpha=0.15, lw=0)
    ax.axvline(0, color=REFERENCE, lw=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Trading days relative to the first close at which the signal can be traded")
    ax.set_ylabel("Cumulative abnormal return (%)")
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "event_study")

    fmc = a["fama_macbeth"]
    reg_labels = {"firm": "Firm sentiment gauge", "sector": "Sector sentiment gauge", "momentum_12_1": "Momentum 12-1",
                  "reversal_4w": "Reversal (1 month)", "volatility_60d": "Volatility (60 days)",
                  "dollar_volume": "Dollar volume", "beta_52w": "Beta (52 weeks)"}
    cols = [("sentiment", "Sentiment only"), ("controls", "With price controls"), ("United States", "U.S."),
            ("Euro area", "Euro area")]
    rows = []
    for k, label in reg_labels.items():
        row = [label]
        for key, _ in cols:
            v = fmc[key].get(k)
            row.append(f"{v['mean'] * 1e4:+.1f} ({v['t']:.2f})" if v else "")
        rows.append(row)
    rows.append(["Mean $R^2$"] + [f"{fmc[key]['r2']:.3f}" for key, _ in cols])
    latex_table("fama_macbeth", ["Regressor"] + [c for _, c in cols], rows, alignment="lcccc")

    # ------------------------------------------------------------------ forecasts
    ic = a["ic_summary"]
    latex_table("forecasts", ["Signal", "Mean rank IC", "HAC $t$", "Weeks IC $>0$ (\\%)"],
                [[MODEL_NAMES[sel["price_model"]] + " (price only)", f"{ic['Price']['mean']:+.4f}", f"{ic['Price']['t']:.2f}",
                  pct(ic["Price"]["positive_share"], 1)],
                 ["Price + sentiment correction", f"{ic['Combined']['mean']:+.4f}", f"{ic['Combined']['t']:.2f}",
                  pct(ic["Combined"]["positive_share"], 1)],
                 ["Sentiment gauge", f"{ic['Sentiment']['mean']:+.4f}", f"{ic['Sentiment']['t']:.2f}",
                  pct(ic["Sentiment"]["positive_share"], 1)],
                 ["Correction $-$ price only", f"{ic['Combined-Price']['mean']:+.4f}", f"{ic['Combined-Price']['t']:.2f}", "--"]])
    ov = a["overlay"]
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    keys = ["agree", "neutral", "disagree"]
    ax.bar(range(3), [ov[k]["mean"] * 1e4 for k in keys], yerr=[1.96 * ov[k]["se"] * 1e4 for k in keys], capsize=3,
           color=[COLORS["Sentiment"], REFERENCE, COLORS["Price"]], error_kw={"lw": 0.8})
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(3), [f"Sentiment agrees\n({pct(ov['agree']['share'], 0)}%)", f"Neutral\n({pct(ov['neutral']['share'], 0)}%)",
                             f"Sentiment disagrees\n({pct(ov['disagree']['share'], 0)}%)"], fontsize=7)
    ax.set_ylabel("Next-week return in the\nprice signal's direction (bp)")
    save(fig, "overlay")

    fig, ax = plt.subplots(figsize=(6.7, 2.5))
    for j, (key, label) in enumerate((("Sentiment", "Sentiment gauge"), ("Combined", "Price + sentiment"), ("Price", "Price only"))):
        series = a["decay"][key]
        k = np.array([d["horizon"] for d in series]) + (j - 1) * 0.12
        ax.errorbar(k, [d["mean"] for d in series], yerr=[1.96 * d["se"] for d in series], color=COLORS[key],
                    marker="o", ms=3, capsize=2, lw=1.2, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(1, PARAMS["decay_horizons"] + 1))
    ax.set_xlabel("Week after the decision in which the return is measured")
    ax.set_ylabel("Mean rank IC")
    ax.legend(fontsize=7)
    save(fig, "decay")

    # ------------------------------------------------------------------ performance
    fig, ax = plt.subplots(figsize=(6.7, 3.0))
    for name in ORDER:
        ax.plot(dates, np.cumprod(1 + frame[name]), color=COLORS[name], label=PLAIN[name],
                lw=1.9 if name in ("Sentiment", "Combined", "Price") else 1.0)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 4, 5]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(dates.iloc[0], dates.iloc[-1])
    ax.set_ylabel("Growth of USD 1 after costs")
    ax.legend(ncol=3, fontsize=7, loc="upper left")
    save(fig, "wealth")
    fig, ax = plt.subplots(figsize=(6.7, 2.3))
    for name in ORDER:
        wealth = np.cumprod(1 + frame[name].to_numpy())
        ax.plot(dates, wealth / np.maximum.accumulate(np.r_[1, wealth])[1:] - 1, color=COLORS[name], label=PLAIN[name],
                lw=1.6 if name in ("Sentiment", "Combined", "Price") else 0.9)
    ax.set_ylabel("Drawdown from peak")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(ncol=3, fontsize=7, loc="lower left")
    save(fig, "drawdown")
    yrs = sorted(dates.dt.year.unique())
    matrix = np.array([[np.prod(1 + frame[n][dates.dt.year == y]) - 1 for y in yrs] for n in ORDER])
    fig, ax = plt.subplots(figsize=(6.7, 2.6))
    ax.imshow(matrix, cmap=DIV, norm=TwoSlopeNorm(0, -0.35, 0.35), aspect="auto")
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(yrs)), yrs, fontsize=8)
    ax.set_yticks(range(len(ORDER)), [PLAIN[n] for n in ORDER], fontsize=8)
    ax.grid(False)
    ax.tick_params(length=0)
    save(fig, "annual")
    rob = a["robustness"]
    grid = PARAMS["cost_grid_bps"]
    table = {bps: (rob["Base case"] if bps == PARAMS["base_cost_bps"] else rob[f"Costs {bps:g} bp per trade"]) for bps in grid}
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.6))
    for name in ["Sentiment", "Combined", "Price", "MinVar", "EW"]:
        for ax, key in zip(axes, ("cagr", "sharpe")):
            ax.plot(grid, [table[b][name][key] for b in grid], color=COLORS[name], marker="o", ms=3, label=PLAIN[name])
    for ax, label in zip(axes, ("CAGR", "Sharpe ratio")):
        ax.axvline(PARAMS["base_cost_bps"], color=REFERENCE, lw=0.8)
        ax.set_xticks(grid)
        ax.set_xlabel("Cost per purchase or sale (bp)")
        ax.set_ylabel(label)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    save(fig, "costs")
    held = weights[weights.strategy == "Sentiment"].copy()
    held["sector"] = held.ticker.map(sector)
    by_sector = held.groupby(["week", "sector"]).weight.sum().unstack().fillna(0)
    ew = weights[weights.strategy == "EW"].copy()
    ew["sector"] = ew.ticker.map(sector)
    ew_sector = ew.groupby(["week", "sector"]).weight.sum().unstack().fillna(0)
    active_w = (by_sector - ew_sector).mean().sort_values()
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    ax.barh([s.replace("Telecommunication Services", "Telecom services") for s in active_w.index], active_w.values * 100,
            color=[COLORS["Sentiment"] if v > 0 else COLORS["Price"] for v in active_w.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Average weight relative to equal weighting (percentage points)")
    save(fig, "sector_tilts")

    latex_table("performance", ["Strategy", "CAGR", "Vol.", "Sharpe", "Sortino", "Max DD", "Turnover", "Costs",
                                r"$\alpha$ ($t$)", r"$\beta$"],
                [[LABELS[n], pct(m[n]["cagr"]), pct(m[n]["volatility"]), f"{m[n]['sharpe']:.2f}", f"{m[n]['sortino']:.2f}",
                  pct(m[n]["max_drawdown"], 1), pct(m[n]["turnover"], 1) if "turnover" in m[n] else "--",
                  pct(m[n]["costs"]) if "costs" in m[n] else "--",
                  f"{pct(a['index_alpha'][n]['alpha'], 2, True)} ({a['index_alpha'][n]['alpha_t']:+.1f})" if n in a["index_alpha"] else "--",
                  f"{a['index_alpha'][n]['beta']:.2f}" if n in a["index_alpha"] else "--"] for n in ORDER])
    latex_table("uncertainty", ["Comparison", "$\\Delta$CAGR (pp)", "95\\% interval", "$\\Delta$Sharpe", "95\\% interval",
                                "$\\Delta$Max DD (pp)", "95\\% interval"],
                [[f"{LABELS[x]} vs {LABELS[y] if y == 'Index' else LABELS[y].lower()}",
                  pct(v["cagr"]["difference"], 2, True), f"[{pct(v['cagr']['ci'][0])}, {pct(v['cagr']['ci'][1])}]",
                  f"{v['sharpe']['difference']:+.3f}", f"[{v['sharpe']['ci'][0]:.3f}, {v['sharpe']['ci'][1]:.3f}]",
                  pct(v["max_drawdown"]["difference"], 1, True),
                  f"[{pct(v['max_drawdown']['ci'][0], 1)}, {pct(v['max_drawdown']['ci'][1], 1)}]"]
                 for (x, y), v in ((tuple(k.split("|")), v) for k, v in a["bootstrap"].items())])
    cols = ["Sentiment", "Combined", "Price", "MinVar", "EW"]
    latex_table("robustness", ["Scenario"] + [LABELS[n] for n in cols],
                [[tex(label)] + [f"{pct(v[n]['cagr'])} / {v[n]['sharpe']:.2f}" if n in v else "--" for n in cols]
                 for label, v in rob.items()], alignment="l" + "c" * len(cols))
    firms = sorted(universe, key=lambda r: (r["region"], r["sector"], r["ticker"]))
    per = Counter(r["ticker"] for r in records)
    half = (len(firms) + 1) // 2
    left, right = firms[:half], firms[half:]
    rows = []
    for k in range(half):
        row = []
        for side in (left, right):
            if k < len(side):
                f = side[k]
                row += [tex(f["ticker"]), tex(f["name"][:26]), f"{per[f['ticker']]:,}"]
            else:
                row += ["", "", ""]
        rows.append(row)
    latex_table("universe", ["Ticker", "Company", "Headlines", "Ticker", "Company", "Headlines"], rows, alignment="llrllr")

    # ------------------------------------------------------------------ numbers in the text
    macros["Weeks"] = str(len(frame))
    macros["Firms"] = str(len(universe))
    macros["FirmsUS"] = str(sum(f["region"] == "United States" for f in universe))
    macros["FirmsEU"] = str(sum(f["region"] == "Euro area" for f in universe))
    macros["Retrieved"] = f"{au['headlines_retrieved']:,}"
    macros["Headlines"] = f"{au['headlines_accepted']:,}"
    macros["Roundups"] = f"{au['headlines_roundup']:,}"
    nw = [news[i][t]["news_count"] for i in test_weeks for t in news[i]]
    macros["NoNewsShare"] = pct(np.mean(np.array(nw) == 0), 1) + r"\%"
    macros["MedianHeadlinesWeek"] = f"{np.median(nw):.0f}"
    act = [s for i in test_weeks for s in active.get(i, {}).values()]
    macros["ActiveShare"] = pct(np.mean(np.array(act) != 0), 1) + r"\%"
    macros["SelectedPrice"] = MODEL_NAMES[sel["price_model"]]
    macros["SelectedGauge"] = "firm and sector" if sel["sector_weight"] > 0 else "firm-only"
    macros["Direction"] = "continuation" if sel["direction"] > 0 else "reversal"
    for n in names:
        macros[f"Val{n}IC"] = f"{cand[n]['mean']:+.4f}"
        macros[f"Val{n}T"] = f"{cand[n]['t']:.2f}"
    for key, tag in (("0", "Firm"), ("0.5", "FirmSector")):
        if key in gc:
            macros[f"ValGauge{tag}IC"] = f"{gc[key]['mean']:+.4f}"
            macros[f"ValGauge{tag}T"] = f"{gc[key]['t']:.2f}"
    for k, tag in (("Price", "Price"), ("Combined", "Combined"), ("Sentiment", "Sentiment"), ("Combined-Price", "Diff")):
        macros[f"IC{tag}"] = f"{ic[k]['mean']:+.4f}"
        macros[f"IC{tag}T"] = f"{ic[k]['t']:.2f}"
    for k in ("agree", "neutral", "disagree"):
        macros[f"Overlay{k.capitalize()}"] = f"{ov[k]['mean'] * 1e4:+.1f}"
        macros[f"Overlay{k.capitalize()}T"] = f"{ov[k]['t']:.2f}"
        macros[f"Overlay{k.capitalize()}Share"] = pct(ov[k]["share"], 0) + r"\%"
    for sign, tag in (("1", "Pos"), ("-1", "Neg")):
        e = es[sign] if sign in es else es[int(sign)]
        d = list(days)
        macros[f"Event{tag}N"] = f"{e['n']:,}"
        if e["mean"]:
            mean = np.array(e["mean"])
            pre, post5, post20, post40 = mean[d.index(0)], mean[d.index(5)], mean[d.index(20)], mean[-1]
            se = np.array(e["se"])
            macros[f"Event{tag}Pre"] = pct(pre, 2, True) + r"\%"
            macros[f"Event{tag}Post"] = pct(post20 - pre, 2, True) + r"\%"
            macros[f"Event{tag}PostFive"] = pct(post5 - pre, 2, True) + r"\%"
            macros[f"Event{tag}PostForty"] = pct(post40 - pre, 2, True) + r"\%"
    for key, tag in (("sentiment", "Only"), ("controls", "Ctrl"), ("United States", "US"), ("Euro area", "EU")):
        for reg, rtag in (("firm", "Firm"), ("sector", "Sector")):
            v = fmc[key][reg]
            macros[f"FM{tag}{rtag}"] = f"{v['mean'] * 1e4:+.1f}"
            macros[f"FM{tag}{rtag}T"] = f"{v['t']:.2f}"
    for n in ORDER:
        v = m[n]
        macros.update({f"{n}CAGR": pct(v["cagr"]) + r"\%", f"{n}Vol": pct(v["volatility"]) + r"\%",
                       f"{n}Sharpe": f"{v['sharpe']:.2f}", f"{n}Sortino": f"{v['sortino']:.2f}",
                       f"{n}DD": pct(v["max_drawdown"], 1) + r"\%"})
        if "turnover" in v:
            macros.update({f"{n}Turnover": pct(v["turnover"], 1) + r"\%", f"{n}Costs": pct(v["costs"]) + r"\%",
                           f"{n}Holdings": f"{v['holdings']:.0f}", f"{n}Views": f"{v['views']:.0f}",
                           f"{n}Skill": f"{v['skill']:.3f}"})
        if n in a["index_alpha"]:
            ia = a["index_alpha"][n]
            macros.update({f"{n}Alpha": pct(ia["alpha"], 2, True) + r"\%", f"{n}AlphaT": f"{ia['alpha_t']:.2f}",
                           f"{n}Beta": f"{ia['beta']:.2f}"})
    for key, v in a["bootstrap"].items():
        tag = "Boot" + key.replace("|", "")
        macros.update({tag + "CAGR": pct(v["cagr"]["difference"], 2, True), tag + "CAGRLow": pct(v["cagr"]["ci"][0]),
                       tag + "CAGRHigh": pct(v["cagr"]["ci"][1]), tag + "Sharpe": f"{v['sharpe']['difference']:+.3f}",
                       tag + "SharpeLow": f"{v['sharpe']['ci'][0]:.3f}", tag + "SharpeHigh": f"{v['sharpe']['ci'][1]:.3f}",
                       tag + "DD": pct(v["max_drawdown"]["difference"], 1, True),
                       tag + "DDLow": pct(v["max_drawdown"]["ci"][0], 1), tag + "DDHigh": pct(v["max_drawdown"]["ci"][1], 1)})
    for key, v in a["conditions"].items():
        tag = key.replace("|", "")
        for part in ("up", "down"):
            macros[f"{part.capitalize()}{tag}"] = f"{v[part]['mean'] * 1e4:+.1f}"
            macros[f"{part.capitalize()}{tag}T"] = f"{v[part]['t']:.2f}"
    for key, v in a["break_even_bps"].items():
        macros["BreakEven" + key.replace("|", "")] = "--" if v is None else f"{v:.1f}"
    for n, v in a["deflated_sharpe"].items():
        macros[f"DSR{n}"] = f"{v:.2f}"
    for k, word in ((0, "One"), (1, "Two"), (3, "Four"), (7, "Eight")):
        for key in ("Sentiment", "Price", "Combined"):
            d = a["decay"][key][k]
            macros[f"Decay{key}{word}"] = f"{d['mean']:+.4f}"
            macros[f"Decay{key}{word}T"] = f"{d['t']:.2f}"
    for label, v in rob.items():
        words = "Zero One Two Three Four Five Six Seven Eight Nine".split()
        tag = "".join(words[int(ch)] if ch.isdigit() else ch for ch in label.title() if ch.isalnum())
        for n, mm in v.items():
            if n in ("Sentiment", "Combined", "Price", "MinVar", "EW"):
                macros[f"Rob{tag}{n}CAGR"] = pct(mm["cagr"]) + r"\%"
                macros[f"Rob{tag}{n}Sharpe"] = f"{mm['sharpe']:.2f}"
    (TABLE / "numbers.tex").write_text(
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items() if k.isalpha()) + "\n", encoding="utf-8")
    bad = [k for k in macros if not k.isalpha()]
    if bad:
        print("Macros with non-letter names skipped:", bad)
