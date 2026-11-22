"""Post-test check added after the second examiner's feedback: which news lies behind the sentiment signals,
and whether the signals behave differently in asset-heavy and asset-light sectors.
1. Event types: every test headline is given one type by keyword rules (first match in the order of TYPES).
   A signal is the week a view opens (the event study in results/analysis.json); its type is the most common
   type among the firm's headlines that week ("Other" only if none has a type). The event study is repeated
   by type and reproduces the report's figures for all signals.
2. Asset intensity: the event study is repeated for asset-heavy sectors (energy, materials, industrials,
   utilities, telecoms), asset-light sectors (information technology, health care, consumer goods) and
   financials. Sector is a proxy, since free point-in-time balance sheets are not available.
Uses the frozen gauge and changes no frozen choice.
Writes results/events.json, results/tables/events.tex and macros in extra_numbers.tex.
Usage: python diagnostics/events.py   (about 5 minutes)"""
import json
import re
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.config import OUT, PARAMS, TABLE  # noqa: E402
from pipeline.evaluate import event_study  # noqa: E402
from pipeline.run import daily_abnormal, market_inputs, news_inputs, test_indices  # noqa: E402
from pipeline.sentiment import view_states  # noqa: E402
from project import unpack_scores  # noqa: E402
from diagnostics.report_figures import write_macros  # noqa: E402

TYPES = [   # (name, pattern); a headline takes the first type whose pattern it matches
    ("Earnings and guidance", r"earnings|profits?|revenues?|\bsales\b|results|quarter|\bq[1-4]\b|\beps\b|guidance|"
                              r"outlook|forecasts?|\bbeats?\b|miss(es|ed)?\b|dividend|buyback|margins?"),
    ("Deals and acquisitions", r"acqui\w*|merger|merge[sd]?\b|takeover|\bbids?\b|buyout|\bdeals?\b|\bstake\b|"
                               r"divest\w*|spin-?off|joint venture|\bbuys?\b|\bbought\b|purchas\w*|\bsells?\b|"
                               r"\bsold\b|\bexit\w*\b"),
    ("Legal, regulatory and political", r"lawsuits?|\bsuits?\b|\bsue[sd]?\b|suing|court|judge|settle\w*|\bfined?\b|"
                                        r"\bfines\b|probe|investigat\w*|regulat\w*|antitrust|\bsec\b|\bftc\b|"
                                        r"\bdoj\b|\bban(s|ned)?\b|recall\w*|scandal|fraud|penalt\w*|watchdog|"
                                        r"emissions|protest\w*|accus\w*|allegations?|tax(es)?\b|tariffs?|"
                                        r"sanctions?|government|congress|senators?|ministers?|\beu\b"),
    ("Analyst ratings", r"upgrade\w*|downgrade\w*|price target|\brating\b|analysts?|outperform|underperform|"
                        r"overweight|underweight"),
    ("Management and staff", r"\bceo\b|chief|executive|chairman|\bcfo\b|president|resign\w*|steps? down|"
                             r"appoint\w*|hires?\b|hiring|\bfire[sd]?\b|layoffs?|lay(ing)? off|job cuts|\bjobs\b|"
                             r"workers|employees|staff|union|strikes?\b|recruit\w*"),
    ("Products and operations", r"launch\w*|unveil\w*|introduc\w*|products?|\bdrugs?\b|vaccine|\bfda\b|approv\w*|"
                                r"\btrials?\b|patients|plants?\b|factory|production|orders?\b|contracts?\b|"
                                r"deliver\w*|outage|\bmodels?\b|partner\w*|collaborat\w*|teams? up|agreement|"
                                r"\bopen(s|ing)?\b|stores?\b|branch|expan\w*|invest(s|ing|ment)\b|\bcloses?\b|"
                                r"closing|shut\w*|devices?|\bapp\b|software|cloud|\bai\b|chips?|cars?\b|trucks?|"
                                r"aircraft|planes?|jets?|prices?\b|customers"),
    ("Share price and commentary", r"shares?\b|stocks?\b|rall\w*|jump\w*|soar\w*|surg\w*|slump\w*|plung\w*|"
                                   r"tumbl\w*|\bfall(s|ing)?\b|\bfell\b|drop\w*|sink\w*|record high|all-time high|"
                                   r"\brises?\b|\brose\b|climb\w*|\bgains?\b|\blosses\b|market value|valuation|"
                                   r"investors?|\?\s*$|^(why|how|what|where|should|is|can|will)\b"),
]
OTHER, NONE = "Other", "No firm headline"
GROUPS = {"Asset-heavy": {"Energy", "Materials", "Industrials", "Utilities", "Telecommunication Services"},
          "Asset-light": {"Information Technology", "Health Care", "Consumer Discretionary", "Consumer Staples"},
          "Financials": {"Financials"}}
COMPILED = [(name, re.compile(p, re.I)) for name, p in TYPES]


def event_type(title):
    return next((name for name, p in COMPILED if p.search(title)), OTHER)


def summary(events, daily, window):
    """Cumulative abnormal return before (day -10 to 0) and after (day 0 to 20 and to 40) the signal, as in
    the report's event study, with the t statistic of the after-signal returns across events."""
    lo, hi = window
    es = event_study(events, daily, window)
    out = {}
    for sign in (1, -1):
        paths = []
        for t, date, s in events:
            if s != sign:
                continue
            dates, abnormal = daily[t]
            k = bisect_left(dates, date)
            if k >= len(dates) or dates[k] != date or k + lo < 1 or k + hi >= len(dates):
                continue
            seg = abnormal[k + lo:k + hi + 1]
            if not np.isnan(seg).any():
                paths.append(np.cumsum(seg))
        p = np.array(paths)
        assert len(p) == es[sign]["n"]
        if len(p) < 2:
            out[str(sign)] = {"n": len(p)}
            continue
        d0, d20 = -lo, 20 - lo
        post = p[:, d20] - p[:, d0]
        out[str(sign)] = {"n": len(p), "pre": float(p[:, d0].mean()), "post20": float(post.mean()),
                          "post20_t": float(post.mean() / (post.std(ddof=1) / np.sqrt(len(post)))),
                          "post40": float((p[:, -1] - p[:, d0]).mean())}
    return out


def main():
    unpack_scores()
    frozen = json.loads((OUT / "frozen.json").read_text())
    universe, clock, decisions, prices = market_inputs()
    records, news = news_inputs(universe, clock, decisions, {})
    firms = {f["ticker"]: f for f in universe}
    idx = test_indices(decisions)
    test = set(idx)

    # headlines per (decision week, firm), timed exactly as in pipeline/sentiment.py
    cutoffs = [d["cutoff"] for d in decisions]
    weekly, available = defaultdict(list), {}
    for r in records:
        key = (firms[r["ticker"]]["exchange"], r["date"])
        if key not in available:
            available[key] = clock.news_available(*key)
        if available[key] is None:
            continue
        i = bisect_right(cutoffs, available[key])
        if i in test:
            weekly[(i, r["ticker"])].append(event_type(r["title"]))
    all_types = Counter(t for v in weekly.values() for t in v)

    # signals: the weeks a view opens, exactly as in pipeline/run.py
    w = frozen["sector_weight"]
    gauge = [{t: news[i][t][f"gauge_{w:g}"] for t in news[i]} for i in range(len(news))]
    states = view_states(gauge, PARAMS["entry_band"], PARAMS["exit_band"])
    events, previous = [], {}
    for i, week in enumerate(states):
        for t, s in week.items():
            if s != 0 and previous.get(t, 0) == 0 and i in test and prices.targets[i][t]["realised_return"] is not None:
                events.append((t, prices.targets[i][t]["execution_date"], s, i))
            previous[t] = s

    def dominant(i, t):
        types = weekly.get((i, t), [])
        if not types:
            return NONE
        counts = Counter(x for x in types if x != OTHER)   # "Other" only when no headline has a type
        if not counts:
            return OTHER
        order = [n for n, _ in TYPES]
        return max(counts, key=lambda k: (counts[k], -order.index(k)))

    daily = daily_abnormal(prices, universe)
    window = PARAMS["event_window_days"]
    base = summary([(t, d, s) for t, d, s, _ in events], daily, window)
    signal_types = Counter(t for t_, _, s, i in events for t in weekly.get((i, t_), []))
    by_type = {}
    for name in [n for n, _ in TYPES] + [OTHER, NONE]:
        sub = [(t, d, s) for t, d, s, i in events if dominant(i, t) == name]
        by_type[name] = {"share_all": all_types[name] / sum(all_types.values()) if name != NONE else None,
                         "share_signal_weeks": signal_types[name] / sum(signal_types.values()) if name != NONE else None,
                         **summary(sub, daily, window)}
    by_group = {}
    for g, sectors in GROUPS.items():
        sub = [(t, d, s) for t, d, s, _ in events if firms[t]["sector"] in sectors]
        by_group[g] = {"firms": sum(f["sector"] in sectors for f in universe if f["ticker"] in daily),
                       **summary(sub, daily, window)}
    out = {"base": base, "headlines": sum(all_types.values()),
           "signals": len(events), "by_type": by_type, "by_group": by_group}
    (OUT / "events.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    write_outputs(out)
    print(json.dumps(out, indent=1))


def write_outputs(out):
    pct = lambda x: "" if x is None else f"{100 * x:.1f}"   # noqa: E731

    def car(r, k):
        if k not in r:
            return "--"
        text = f"{100 * r[k]:+.2f}"
        t = r["post20_t"] if abs(r["post20_t"]) >= 0.05 else 0.0
        return text + (f" ({t:.1f})" if k == "post20" else "")

    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Headlines (\%)} & \multicolumn{3}{c}{Positive signals} "
             r"& \multicolumn{3}{c}{Negative signals}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
             r" & All & Signal weeks & $n$ & Before & After ($t$) & $n$ & Before & After ($t$)\\", r"\midrule"]

    def row(name, r, shares=True):
        p, n = r["1"], r["-1"]
        a = f"{pct(r.get('share_all'))} & {pct(r.get('share_signal_weeks'))}" if shares else " & "
        return (f"{name} & {a} & {p['n']:,} & {car(p, 'pre')} & {car(p, 'post20')} & "
                f"{n['n']:,} & {car(n, 'pre')} & {car(n, 'post20')}\\\\")
    lines += [row(name, r) for name, r in out["by_type"].items()]
    lines += [r"\midrule", row("All signals", out["base"], False), r"\midrule"]
    lines += [row(f"{g} ({r['firms']} firms)", r, False) for g, r in out["by_group"].items()]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLE / "events.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    t, g = out["by_type"], out["by_group"]
    sign = lambda x: f"{100 * x:+.2f}" + r"\%"   # noqa: E731
    write_macros({"EvOtherShare": f"{100 * t[OTHER]['share_all']:.0f}" + r"\%",
                  "EvEarnPosPre": sign(t["Earnings and guidance"]["1"]["pre"]),
                  "EvEarnPosPost": sign(t["Earnings and guidance"]["1"]["post20"]),
                  "EvLegalNegPre": sign(t["Legal, regulatory and political"]["-1"]["pre"]),
                  "EvFinPosPost": sign(g["Financials"]["1"]["post20"]),
                  "EvFinPosPostT": f"{g['Financials']['1']['post20_t']:.2f}",
                  "EvFinNegPre": sign(g["Financials"]["-1"]["pre"]),
                  "EvFinNegPost": sign(g["Financials"]["-1"]["post20"]),
                  "EvFinNegPostT": f"{g['Financials']['-1']['post20_t']:.2f}"})


if __name__ == "__main__":
    main()
