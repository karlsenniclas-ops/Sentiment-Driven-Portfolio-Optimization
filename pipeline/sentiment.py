"""Sentiment architecture: headline tone -> firm-week and sector-week tone -> abnormal tone
against the firm's (sector's) own previous year -> smoothed sentiment gauge."""
from bisect import bisect_right
from collections import defaultdict

import numpy as np

from .config import PARAMS


def assign_weeks(records, firms, clock, decisions, extra_sessions=0):
    """A headline enters the first decision whose cutoff is after its availability time."""
    cutoffs = [d["cutoff"] for d in decisions]
    cache, buckets = {}, defaultdict(list)
    for r in records:
        key = (firms[r["ticker"]]["exchange"], r["date"])
        if key not in cache:
            cache[key] = clock.news_available(*key, extra_sessions)
        if cache[key] is None:
            continue
        index = bisect_right(cutoffs, cache[key])
        if index < len(cutoffs):
            buckets[(index, r["ticker"])].append((r["tone"], cache[key]))
    return buckets


def _abnormal(history, index, mean_tone, n):
    """(mean tone - own 52-week mean) / own 52-week s.d., shrunk by n / (n + k)."""
    k = PARAMS["shrinkage_headlines"]
    base = [m for i, m in history if index - PARAMS["baseline_weeks"] <= i < index]
    if n == 0 or len(base) < PARAMS["baseline_min_news_weeks"]:
        return 0.0
    sd = float(np.std(base, ddof=1))
    if sd < 1e-9:
        return 0.0
    return float(np.clip((mean_tone - np.mean(base)) / sd, -4, 4) * n / (n + k))


def news_panel(records, universe, clock, decisions, extra_sessions=0, half_life=None):
    """Weekly sentiment per firm: headline count, shrunk tone, abnormal firm and sector tone,
    and the gauge for each candidate sector weight."""
    firms = {f["ticker"]: f for f in universe}
    buckets = assign_weeks(records, firms, clock, decisions, extra_sessions)
    sectors = defaultdict(list)
    for t, f in firms.items():
        sectors[f["sector"]].append(t)
    k = PARAMS["shrinkage_headlines"]
    lam = 1 - 0.5 ** (1 / (half_life or PARAMS["gauge_half_life_weeks"]))
    firm_hist, sector_hist = defaultdict(list), defaultdict(list)
    weights = sorted(set(PARAMS["sector_weight_grid"]) | {0.0, 1.0})   # 1.0 = sector-only, for analysis
    meters = {w: defaultdict(float) for w in weights}
    panel = []
    for index in range(len(decisions)):
        week, sector_z = {}, {}
        for sector, members in sectors.items():
            tones = [tone for t in members for tone, _ in buckets.get((index, t), [])]
            n = len(tones)
            mean = float(np.mean(tones)) if n else 0.0
            sector_z[sector] = _abnormal(sector_hist[sector], index, mean, n)
            if n:
                sector_hist[sector].append((index, mean))
        for t, f in firms.items():
            items = buckets.get((index, t), [])
            n = len(items)
            mean = float(np.mean([tone for tone, _ in items])) if n else 0.0
            z_firm = _abnormal(firm_hist[t], index, mean, n)
            if n:
                firm_hist[t].append((index, mean))
            row = {"news_count": n, "tone": mean * n / (n + k), "raw_tone": mean if n else np.nan,
                   "abnormal_firm": z_firm, "abnormal_sector": sector_z[f["sector"]],
                   "latest_news": max((a for _, a in items), default=None)}
            for w, meter in meters.items():
                meter[t] = (1 - lam) * meter[t] + lam * ((1 - w) * z_firm + w * sector_z[f["sector"]])
                row[f"meter_{w:g}"] = meter[t]
            week[t] = row
        panel.append(week)
    # Express each gauge in standard deviations: the pooled standard deviation of the meter across
    # all firms over the previous 52 weeks (causal, and robust to the growth of news coverage).
    window = PARAMS["baseline_weeks"]
    for w in weights:
        key = f"meter_{w:g}"
        sums = np.array([[sum(r[key] for r in week.values()), sum(r[key] ** 2 for r in week.values()), len(week)]
                         for week in panel])
        for i, week in enumerate(panel):
            lo = max(0, i - window)
            total, squares, count = sums[lo:i].sum(axis=0) if i > lo else (0.0, 0.0, 0)
            sd = np.sqrt(max(squares / count - (total / count) ** 2, 0.0)) if i - lo >= 26 and count else 0.0
            for row in week.values():
                row[f"gauge_{w:g}"] = row[key] / sd if sd > 1e-12 else 0.0
    return panel


def view_states(gauges, entry, exit_):
    """Hysteresis rule: a view opens when |gauge| >= entry and stays open, with the sign at entry,
    until |gauge| falls below exit or the sign flips. Returns +1 / -1 / 0 per week and stock."""
    state, out = {}, []
    for week in gauges:
        current = {}
        for t, g in week.items():
            s = state.get(t, 0)
            if s == 0 and abs(g) >= entry:
                s = int(np.sign(g))
            elif s != 0 and (abs(g) < exit_ or np.sign(g) != s):
                s = int(np.sign(g)) if abs(g) >= entry else 0
            state[t] = s
            current[t] = s
        out.append(current)
    return out
