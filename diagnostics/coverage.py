"""Post-test diagnostic of news coverage, added after the test in response to supervisor feedback.
It uses the frozen signals and optimiser (results/frozen.json) and changes no frozen choice.
1. How often Google News's limit of 100 items per firm-month query binds.
2. Signal quality and contribution to the sentiment-driven portfolio for high- and low-coverage firms.
   Coverage is a firm's mean number of accepted headlines per week over the last 52 weeks (known at the
   decision); firms at or above the weekly cross-sectional median are "high coverage".
3. The sentiment-driven portfolio with views restricted by coverage.
4. Whether weeks without news behave as neutral (next-week relative return of stock-weeks without news).
Writes results/coverage.json, results/tables/coverage.tex and macros in results/tables/extra_numbers.tex.
Usage: python diagnostics/coverage.py   (about 10 minutes; --tables-only rebuilds the outputs)"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import forecast as fc  # noqa: E402
from pipeline.config import DATA, OUT, PARAMS, TABLE  # noqa: E402
from pipeline.evaluate import fama_macbeth, hac, paired_bootstrap, scalar_metrics  # noqa: E402
from pipeline.portfolio import simulate  # noqa: E402
from pipeline.run import (build_context, cached, finish_frame, market_inputs, news_inputs, signals,  # noqa: E402
                          test_indices)
from diagnostics.report_figures import write_macros  # noqa: E402
from project import unpack_scores  # noqa: E402


def cap_statistics(universe):
    """Share of firm-month queries that returned the maximum of 100 items."""
    region = {f["ticker"]: f["region"] for f in universe}
    rows = []
    for path in sorted((DATA / "headlines").glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if r["ticker"] in region:
                rows.append((r["ticker"], int(r["from"][:4]), int(r["n"])))
    capped = np.array([n >= 100 for _, _, n in rows])
    test = np.array([2015 <= y <= 2025 for _, y, _ in rows])
    by_firm = defaultdict(list)
    for (t, y, n), c in zip(rows, capped):
        if 2015 <= y <= 2025:
            by_firm[t].append(c)
    firm_share = {t: float(np.mean(v)) for t, v in by_firm.items()}
    out = {"firm_months": len(rows), "capped_share": float(capped.mean()),
           "capped_share_test": float(capped[test].mean()),
           "items_in_capped_months": float(sum(n for (_, _, n), c in zip(rows, capped) if c) / sum(n for *_, n in rows)),
           "firms_capped_half_of_test_months": int(sum(v >= 0.5 for v in firm_share.values())),
           "firms_never_capped_in_test": int(sum(v == 0 for v in firm_share.values())),
           "firms": len(firm_share)}
    for reg in sorted(set(region.values())):
        mask = np.array([region[t] == reg for t, _, _ in rows]) & test
        out[f"capped_share_test_{reg}"] = float(capped[mask].mean())
    out["capped_share_by_year"] = {y: float(capped[np.array([yy == y for _, yy, _ in rows])].mean())
                                   for y in sorted({y for _, y, _ in rows})}
    return out


def coverage_groups(news, idx):
    """Trailing mean headlines per week (last 52 weeks including the decision week) and a high/low split
    at the weekly cross-sectional median."""
    window = PARAMS["baseline_weeks"]
    coverage, group = {}, {}
    for i in idx:
        cov = {t: float(np.mean([news[j][t]["news_count"] for j in range(max(0, i - window + 1), i + 1)]))
               for t in news[i]}
        median = float(np.median(list(cov.values())))
        coverage[i] = cov
        group[i] = {t: "high" if v >= median else "low" for t, v in cov.items()}
    return coverage, group


def main():
    frozen = json.loads((OUT / "frozen.json").read_text())
    stored = json.loads((OUT / "analysis.json").read_text())["metrics"]
    universe, clock, decisions, prices = market_inputs()
    panel = fc.Panel(prices)
    _, news = news_inputs(universe, clock, decisions, {})
    price_fc = cached(f"forecast_{frozen['price_model']}",
                      lambda: fc.price_forecasts(panel, frozen["price_model"], "2008", "2025"))
    price_cal = fc.calibrate(panel, price_fc, [str(y) for y in range(2009, 2026)])
    g, active, combined, _ = signals(panel, news, price_fc, price_cal, frozen["sector_weight"], frozen["direction"],
                                     frozen["correction_variant"])
    ctx = cached("context", lambda: build_context(prices, panel))
    idx = test_indices(decisions)
    result = {"caps": cap_statistics(universe)}
    print(json.dumps(result["caps"], indent=1), flush=True)

    # Base case: check that this environment reproduces the stored test result.
    spec = {"kind": "views", "signal": g, "active": active}
    base, weights = simulate(ctx, {"Sentiment": spec}, keep_weights=True)
    base = finish_frame(base)
    rf = base.risk_free.to_numpy()
    check = {n: scalar_metrics(base[n].to_numpy(), rf) for n in ("Sentiment", "EW")}
    result["reproduction_gap"] = max(abs(check[n][k] - stored[n][k]) for n in check for k in ("cagr", "sharpe"))
    print("Reproduction gap vs stored metrics:", result["reproduction_gap"], flush=True)

    coverage, group = coverage_groups(news, range(PARAMS["baseline_weeks"], len(news)))
    result["median_coverage"] = float(np.median([np.median(list(coverage[i].values())) for i in idx]))

    # Signal quality by coverage group.
    quality = {}
    for name in ("high", "low"):
        ics, view_ret, view_share = [], [], []
        for i in idx:
            rel = panel.relative[i]
            names = [t for t in rel if t in g.get(i, {}) and group[i].get(t) == name]
            if len(names) >= 20:
                x = np.array([g[i][t] for t in names])
                if x.std() > 1e-12:
                    ics.append(fc.weekly_ic(panel, {i: {t: g[i][t] for t in names}}, [i])[0][1])
            on = [t for t in names if active.get(i, {}).get(t, 0) != 0]
            all_on = [t for t in rel if active.get(i, {}).get(t, 0) != 0]
            if all_on:
                view_share.append(len(on) / len(all_on))
            if on:
                view_ret.append(float(np.mean([active[i][t] * rel[t] for t in on])))
        quality[name] = {"ic": hac(ics), "view_return": hac(view_ret), "view_share": float(np.mean(view_share))}
    result["quality"] = quality

    # Contribution of each group to the sentiment-driven portfolio's return relative to equal weight (before costs).
    w = weights.pivot_table(index=["week", "ticker"], columns="strategy", values="weight").reset_index()
    week_index = {decisions[i]["week"]: i for i in idx}
    contrib = {"high": [], "low": []}
    for week, rows in w.groupby("week"):
        i = week_index[week]
        part = {"high": 0.0, "low": 0.0}
        for t, ws, we in zip(rows.ticker, rows.Sentiment, rows.EW):
            part[group[i].get(t, "low")] += (ws - we) * ctx["returns"][i][t]
        for k in part:
            contrib[k].append(part[k])
    result["contribution"] = {k: {**hac(v), "annual_pp": float(np.mean(v) * 52 * 100)} for k, v in contrib.items()}

    # Views restricted by coverage (one setting changed at a time, as in the robustness tests).
    variants = {"Views only for high-coverage firms": lambda i, t: group[i].get(t) == "high",
                "Views only for low-coverage firms": lambda i, t: group[i].get(t) == "low",
                "No views below 1 headline a week": lambda i, t: coverage[i].get(t, 0.0) >= 1.0}
    result["variants"] = {"Base case": {**check["Sentiment"], "views": float(base["views_Sentiment"].mean())}}
    for label, keep in variants.items():
        act = {i: {t: (s if i in group and keep(i, t) else 0) for t, s in states.items()} for i, states in active.items()}
        frame, _ = simulate(ctx, {"Sentiment": {"kind": "views", "signal": g, "active": act}})
        frame = finish_frame(frame)
        m = scalar_metrics(frame["Sentiment"].to_numpy(), rf)
        boot = paired_bootstrap(frame["Sentiment"].to_numpy(), base["Sentiment"].to_numpy(), rf)
        result["variants"][label] = {**m, "views": float(frame["views_Sentiment"].mean()),
                                     "sharpe_vs_base_ci": boot["sharpe"]["ci"]}
        print(label, {k: round(v, 4) for k, v in m.items()}, flush=True)

    # Price + sentiment: the sentiment correction applied only to some firms; the others keep the price forecast.
    base_c, _ = simulate(ctx, {"Combined": {"kind": "views", "signal": combined, "active": None}})
    base_c = finish_frame(base_c)
    check_c = scalar_metrics(base_c["Combined"].to_numpy(), rf)
    result["reproduction_gap_combined"] = max(abs(check_c[k] - stored["Combined"][k]) for k in ("cagr", "sharpe"))
    result["combined_variants"] = {"Base case": check_c}
    for label, keep in (("Correction only for high-coverage firms", variants["Views only for high-coverage firms"]),
                        ("Correction only for low-coverage firms", variants["Views only for low-coverage firms"]),
                        ("No correction below 1 headline a week", variants["No views below 1 headline a week"])):
        comb = {i: {t: (v if i not in group or keep(i, t) else price_cal[i][t]) for t, v in week.items()}
                for i, week in combined.items()}
        frame, _ = simulate(ctx, {"Combined": {"kind": "views", "signal": comb, "active": None}})
        frame = finish_frame(frame)
        m = scalar_metrics(frame["Combined"].to_numpy(), rf)
        boot = paired_bootstrap(frame["Combined"].to_numpy(), base_c["Combined"].to_numpy(), rf)
        result["combined_variants"][label] = {**m, "sharpe_vs_base_ci": boot["sharpe"]["ci"]}
        print(label, {k: round(v, 4) for k, v in m.items()}, flush=True)

    # Fama-MacBeth by coverage group, with the price controls of the main regressions.
    controls = ("momentum_12_1", "reversal_4w", "volatility_60d", "dollar_volume", "beta_52w")
    result["fama_macbeth"] = {}
    for name in ("high", "low"):
        rows = [(i, {t: {"y": rel, "firm": news[i][t]["gauge_0"], "sector": news[i][t]["gauge_1"],
                         **{k: prices.features[i][t][k] for k in controls}}
                     for t, rel in panel.relative[i].items() if group[i].get(t) == name})
                for i in idx]
        result["fama_macbeth"][name] = fama_macbeth(rows, ["firm", "sector", *controls])

    # Is no news neutral? Next-week relative return of stock-weeks without a headline in the decision week.
    no_news, no_news_low = [], []
    for i in idx:
        rel = panel.relative[i]
        quiet = [t for t in rel if t in news[i] and news[i][t]["news_count"] == 0]
        if quiet:
            no_news.append(float(np.mean([rel[t] for t in quiet])))
        quiet_low = [t for t in quiet if group[i].get(t) == "low"]
        if quiet_low:
            no_news_low.append(float(np.mean([rel[t] for t in quiet_low])))
    result["no_news"] = {"all": hac(no_news), "low_coverage": hac(no_news_low)}

    (OUT / "coverage.json").write_text(json.dumps(result, indent=1, default=str))
    write_outputs(result)


def write_outputs(r):
    q, c, v, cv, fm = r["quality"], r["contribution"], r["variants"], r["combined_variants"], r["fama_macbeth"]
    pct = lambda x: f"{100 * x:.0f}\\%"   # noqa: E731
    lines = [r"\begin{tabular}{lcc}", r"\toprule", r" & High coverage & Low coverage \\", r"\midrule",
             f"Share of sentiment views & {pct(q['high']['view_share'])} & {pct(q['low']['view_share'])} \\\\",
             f"Gauge rank IC ($t$) & {q['high']['ic']['mean']:+.4f} ({q['high']['ic']['t']:.2f}) & "
             f"{q['low']['ic']['mean']:+.4f} ({q['low']['ic']['t']:.2f}) \\\\",
             f"Return in view direction, bp/week ($t$) & {1e4 * q['high']['view_return']['mean']:+.1f} "
             f"({q['high']['view_return']['t']:.2f}) & {1e4 * q['low']['view_return']['mean']:+.1f} "
             f"({q['low']['view_return']['t']:.2f}) \\\\",
             f"Contribution vs equal weight, pp/year ($t$) & {c['high']['annual_pp']:+.2f} ({c['high']['t']:.2f}) & "
             f"{c['low']['annual_pp']:+.2f} ({c['low']['t']:.2f}) \\\\",
             f"Firm gauge, Fama--MacBeth, bp ($t$) & {1e4 * fm['high']['firm']['mean']:+.1f} ({fm['high']['firm']['t']:.2f}) & "
             f"{1e4 * fm['low']['firm']['mean']:+.1f} ({fm['low']['firm']['t']:.2f}) \\\\",
             r"\midrule", r"\multicolumn{3}{l}{\emph{Sentiment-driven portfolio with views restricted by coverage}}\\",
             r"Variant & CAGR (\%) / Sharpe & Views per week \\", r"\midrule"]
    for label, m in v.items():
        lines.append(f"{label} & {100 * m['cagr']:.2f} / {m['sharpe']:.2f} & {m['views']:.1f} \\\\")
    lines += [r"\midrule", r"\multicolumn{3}{l}{\emph{Price + sentiment portfolio with the correction restricted by coverage}}\\",
              r"\midrule"]
    for label, m in cv.items():
        lines.append(f"{label} & {100 * m['cagr']:.2f} / {m['sharpe']:.2f} & -- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLE / "coverage.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    caps = r["caps"]
    fmt = lambda x: f"{x:.2f}"   # noqa: E731
    macros = {"CapShareTest": pct(caps["capped_share_test"]),
              "CapShareUS": pct(caps["capped_share_test_United States"]),
              "CapShareEU": pct(caps["capped_share_test_Euro area"]),
              "CapItemsShare": pct(caps["items_in_capped_months"]),
              "CapFirmsHalf": str(caps["firms_capped_half_of_test_months"]),
              "CapFirmsNever": str(caps["firms_never_capped_in_test"]),
              "CovViewShareHigh": pct(q["high"]["view_share"]),
              "CovContribHigh": f"{c['high']['annual_pp']:+.2f}", "CovContribHighT": f"{c['high']['t']:.2f}",
              "CovContribLow": f"{c['low']['annual_pp']:+.2f}", "CovContribLowT": f"{c['low']['t']:.2f}",
              "CovViewRetHigh": f"{1e4 * q['high']['view_return']['mean']:+.1f}",
              "CovViewRetLow": f"{1e4 * q['low']['view_return']['mean']:+.1f}",
              "NoNewsRet": f"{1e4 * r['no_news']['all']['mean']:+.1f}", "NoNewsRetT": fmt(r['no_news']['all']['t']),
              "NoNewsRetLow": f"{1e4 * r['no_news']['low_coverage']['mean']:+.1f}",
              "NoNewsRetLowT": fmt(r['no_news']['low_coverage']['t'])}
    years = {str(k): v for k, v in caps["capped_share_by_year"].items()}
    macros.update({"CapShareFirstTest": pct(years["2015"]), "CapShareLastTest": pct(years["2025"]),
                   "CovMedian": f"{r['median_coverage']:.1f}"})
    for key, label in (("High", "Views only for high-coverage firms"), ("Low", "Views only for low-coverage firms"),
                       ("Min", "No views below 1 headline a week")):
        macros[f"Cov{key}Sharpe"] = f"{v[label]['sharpe']:.2f}"
        macros[f"Cov{key}CAGR"] = f"{100 * v[label]['cagr']:.2f}\\%"
    macros.update({"CovCombHighSharpe": f"{cv['Correction only for high-coverage firms']['sharpe']:.2f}",
                   "CovCombLowSharpe": f"{cv['Correction only for low-coverage firms']['sharpe']:.2f}",
                   "CovCombMinSharpe": f"{cv['No correction below 1 headline a week']['sharpe']:.2f}",
                   "CovFMHigh": f"{1e4 * fm['high']['firm']['mean']:+.1f}", "CovFMHighT": f"{fm['high']['firm']['t']:.2f}",
                   "CovFMLow": f"{1e4 * fm['low']['firm']['mean']:+.1f}", "CovFMLowT": f"{fm['low']['firm']['t']:.2f}"})
    write_macros(macros)
    print("\n".join(lines))
    print(json.dumps(macros, indent=1))


if __name__ == "__main__":
    if "--tables-only" in sys.argv:      # rebuild the table and macros from results/coverage.json
        write_outputs(json.loads((OUT / "coverage.json").read_text()))
    else:
        unpack_scores()
        main()
