"""Post-test check of the Google News archive against time-stamped headlines (FNSPID; Dong et al., 2024),
added after supervisor feedback on retrospective retrieval. US firms only: FNSPID covers S&P 500 stocks.
FNSPID merges Nasdaq.com news with an external archive (mainly Benzinga) and dates each headline by its
publisher's own archive, without search ranking or a cap per query; it was itself collected in 2023 and gives
the time of day for almost no headline. data_collection/fetch_fnspid.py fetches and filters it into data/fnspid/.

1. Timing: for headlines found in both sources (same firm, same normalised title), how Google's date compares
   with the time-stamped publication date.
2. Ex-post selection: if a search run in 2026 favours stories that later proved important, Google's weekly
   headline count should rise with the stock's FUTURE absolute returns once the real-time news flow (FNSPID
   count) and current and past absolute returns are controlled for. Fama-MacBeth regressions of firm-demeaned
   log(1 + headlines) on standardised regressors, Newey-West t-statistics, test weeks 2015-2023.
Writes results/fnspid_check.json, results/tables/fnspid.tex and macros in results/tables/extra_numbers.tex.
It uses the frozen data preparation and changes no frozen choice.
Usage: python diagnostics/fnspid_check.py"""
import csv
import datetime as dt
import gzip
import json
import re
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import forecast as fc  # noqa: E402
from pipeline.config import DATA, OUT, PARAMS, TABLE  # noqa: E402
from pipeline.evaluate import fama_macbeth  # noqa: E402
from pipeline.news import IssuerFilter, load_headlines  # noqa: E402
from pipeline.run import market_inputs, news_inputs, test_indices  # noqa: E402
from diagnostics.report_figures import write_macros  # noqa: E402
from project import unpack_scores  # noqa: E402

LAST_WEEK = "2023-12-25"          # FNSPID ends in December 2023


def norm(title):
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def load_fnspid(universe):
    """Time-stamped headlines of the study's US firms, de-duplicated and passed through the same issuer
    and market-roundup filters as the Google headlines."""
    issuer = IssuerFilter(universe)
    tickers = {f["ticker"] for f in universe}
    seen, out = set(), []
    for path in sorted((DATA / "fnspid").glob("*.csv.gz")):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            for r in csv.DictReader(stream):
                t, title = r["ticker"], r["title"].strip()
                key = (t, norm(title))
                if t not in tickers or not title or key in seen:
                    continue
                seen.add(key)
                if not issuer.accept(t, title) or len(issuer.firms_named(title)) >= PARAMS["roundup_min_firms"]:
                    continue
                # Dates without a time of day count from the end of the day (UTC), so no headline enters early.
                stamp = dt.datetime.strptime(r["timestamp"][:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
                if stamp.time() == dt.time(0, 0):
                    stamp += dt.timedelta(hours=23, minutes=59)
                out.append({"ticker": t, "time": stamp, "title": title})
    return out


def timing(google, wire):
    stamps = defaultdict(list)
    for r in wire:
        stamps[(r["ticker"], norm(r["title"]))].append(r["time"].date())
    gaps = []
    for r in google:
        key = (r["ticker"], norm(r["title"]))
        if key in stamps and len(key[1]) >= 25:
            gaps.append((dt.date.fromisoformat(r["date"]) - min(stamps[key])).days)
    gaps = np.array(gaps)
    return {"matched": int(len(gaps)), "same_day": float(np.mean(gaps == 0)),
            "google_later": float(np.mean(gaps > 0)), "google_earlier": float(np.mean(gaps < 0)),
            "google_earlier_1d": float(np.mean(gaps == -1)), "google_earlier_2d_plus": float(np.mean(gaps <= -2)),
            "median_gap_days": float(np.median(gaps)) if len(gaps) else float("nan")}


def selection(panel, news, wire, decisions, idx, us):
    cutoffs = [d["cutoff"] for d in decisions]
    wire_count = Counter()
    for r in wire:
        i = bisect_right(cutoffs, r["time"])
        if i < len(cutoffs):
            wire_count[(i, r["ticker"])] += 1
    rel = panel.relative
    raw = []
    for i in idx:
        if decisions[i]["week"] > LAST_WEEK:
            continue
        for t in us:
            window = [rel[j].get(t) for j in range(i - 5, i + 4)]
            if t not in news[i] or any(v is None for v in window):
                continue
            a = np.abs(window)
            raw.append((i, t, {"google": np.log1p(news[i][t]["news_count"]), "wire": np.log1p(wire_count[(i, t)]),
                               "past": a[:4].mean(), "current": a[4], "future": a[5:].mean()}))
    # Firm fixed effects: subtract each firm's mean of every variable.
    keys = ("google", "wire", "past", "current", "future")
    means = {t: {k: np.mean([v[k] for _, tt, v in raw if tt == t]) for k in keys} for t in us}
    weeks = defaultdict(dict)
    for i, t, v in raw:
        weeks[i][t] = {k: v[k] - means[t][k] for k in keys}

    def run(y, regressors):
        rows = [(i, {t: {"y": v[y], **{k: v[k] for k in regressors}} for t, v in d.items()}) for i, d in sorted(weeks.items())]
        return fama_macbeth(rows, regressors)
    return {"observations": len(raw), "weeks": len(weeks),
            "wire_on_returns": run("wire", ["past", "current", "future"]),
            "google_on_returns": run("google", ["past", "current", "future"]),
            "google_controlled": run("google", ["wire", "past", "current", "future"])}


def main():
    universe, clock, decisions, prices = market_inputs()
    us_firms = [f for f in universe if f["region"] == "United States"]
    us = [f["ticker"] for f in us_firms]
    panel = fc.Panel(prices)
    _, news = news_inputs(universe, clock, decisions, {})
    google = [r for r in load_headlines(universe, {}) if r["ticker"] in set(us)]
    wire = load_fnspid(us_firms)
    by_year = Counter(r["time"].year for r in wire)
    result = {"wire_headlines": len(wire),               "wire_by_year": dict(sorted(by_year.items())),
              "timing": timing(google, wire),
              "selection": selection(panel, news, wire, decisions, test_indices(decisions), us)}
    (OUT / "fnspid_check.json").write_text(json.dumps(result, indent=1, default=str))
    write_outputs(result)


def write_outputs(r):
    s, tm = r["selection"], r["timing"]
    names = {"wire": "FNSPID headlines (log)", "past": "$|r|$, weeks $-4$ to $-1$",
             "current": "$|r|$, week 0", "future": "$|r|$, weeks $+1$ to $+4$"}

    def cell(model, k):
        v = s[model].get(k)
        return f"{100 * v['mean']:+.1f} ({v['t']:.2f})" if v else "--"
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r" & \multicolumn{1}{c}{FNSPID} & \multicolumn{2}{c}{Google News} \\",
             r"Regressor & (1) & (2) & (3) \\", r"\midrule"]
    for k in ("wire", "past", "current", "future"):
        lines.append(f"{names[k]} & {cell('wire_on_returns', k)} & {cell('google_on_returns', k)} & "
                     f"{cell('google_controlled', k)} \\\\")
    lines += [r"\midrule", f"Firm-weeks / weeks & \\multicolumn{{3}}{{c}}{{{s['observations']:,} / {s['weeks']}}} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    (TABLE / "fnspid.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pct = lambda x: f"{100 * x:.0f}\\%"   # noqa: E731
    f = s["google_controlled"]["future"]
    macros = {"FnMatched": f"{tm['matched']:,}", "FnSameDay": pct(tm["same_day"]), "FnLater": pct(tm["google_later"]),
              "FnEarlier": pct(tm["google_earlier"]), "FnEarlierOne": pct(tm["google_earlier_1d"]),
              "FnWire": f"{r['wire_headlines']:,}", "FnFuture": f"{100 * f['mean']:+.1f}", "FnFutureT": f"{f['t']:.2f}",
              "FnWireFuture": f"{100 * s['wire_on_returns']['future']['mean']:+.1f}",
              "FnWireFutureT": f"{s['wire_on_returns']['future']['t']:.2f}"}
    write_macros(macros)
    print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    if "--tables-only" in sys.argv:
        write_outputs(json.loads((OUT / "fnspid_check.json").read_text()))
    else:
        unpack_scores()
        main()
