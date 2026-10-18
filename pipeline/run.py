"""Shared inputs and the single walk-forward test run."""
import datetime as dt
import json
import pickle

import numpy as np
import pandas as pd

from . import forecast as fc
from .config import OUT, PARAMS
from .data import ExchangeClock, PricePanel, decision_calendar, load_universe, RiskFreeSeries
from .evaluate import (event_study, fama_macbeth, hac, index_regression, paired_bootstrap, scalar_metrics,
                       deflated_sharpe)
from .news import attach_scores, load_headlines
from .portfolio import ledoit_wolf, simulate
from .sentiment import news_panel, view_states

CACHE = OUT / "cache"


def cached(name, build):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    value = build()
    path.write_bytes(pickle.dumps(value))
    return value


def market_inputs():
    def build():
        universe = load_universe()
        decisions = decision_calendar()
        return universe, decisions, PricePanel(universe, ExchangeClock(), decisions)
    universe, decisions, prices = cached("prices", build)
    return universe, ExchangeClock(), decisions, prices


def news_inputs(universe, clock, decisions, audit, extra_sessions=0, half_life=None):
    records = attach_scores(load_headlines(universe, audit), audit)
    return records, news_panel(records, universe, clock, decisions, extra_sessions, half_life)


# --------------------------------------------------------------------------- context
def build_context(prices, panel):
    decisions = prices.decisions
    returns, stamp = {}, {}
    for i, targets in enumerate(prices.targets):
        rows = {t: r["realised_return"] for t, r in targets.items()
                if t in prices.features[i] and r["realised_return"] is not None}
        if rows:
            returns[i] = rows
            stamp[i] = max(targets[t]["outcome_available"] for t in rows)
    rf = RiskFreeSeries()
    clock = ExchangeClock()
    index_close = prices.index
    weeks = [i for i, d in enumerate(decisions)
             if PARAMS["evaluation_start"] <= d["week"] <= PARAMS["evaluation_end"]]
    investable, covariance, index_return, risk_free = {}, {}, {}, {}
    matured_order = sorted(stamp, key=lambda j: stamp[j])
    for i in weeks:
        cutoff = decisions[i]["cutoff"]
        past = [j for j in matured_order if stamp[j] < cutoff][-PARAMS["covariance_weeks"]:]
        names = sorted(t for t in returns.get(i, {}) if sum(t in returns[j] for j in past) >= 40)
        matrix = np.array([[returns[j].get(t, np.nan) for t in names] for j in past])
        means = np.nanmean(matrix, axis=1, keepdims=True)
        matrix = np.where(np.isnan(matrix), means, matrix)
        investable[i], covariance[i] = names, ledoit_wolf(matrix)
        risk_free[i] = rf.weekly(cutoff)
        start_day, _ = clock.execution("XNYS", decisions[i]["decision_week"])
        end_day, _ = clock.execution("XNYS", (dt.date.fromisoformat(decisions[i]["decision_week"])
                                              + dt.timedelta(days=7)).isoformat())
        index_return[i] = float(index_close.loc[end_day] / index_close.loc[start_day] - 1)
    return {"decisions": decisions, "weeks": weeks, "returns": returns, "outcome_stamp": stamp,
            "investable": investable, "covariance": covariance, "risk_free": risk_free,
            "index_return": index_return}


CORRECTION_YEARS = [str(y) for y in range(2012, 2026)]


def signals(panel, news, price_fc, price_cal, weight, direction, variant, entry=None, exit_=None):
    """Sentiment gauge (signed so that positive means higher expected return), view states for
    the sentiment-driven portfolio, and the price + sentiment forecast: calibrated price forecast plus the sentiment correction."""
    gauge = [{t: direction * news[i][t][f"gauge_{weight:g}"] for t in news[i]} for i in range(len(news))]
    states = view_states(gauge, entry or PARAMS["entry_band"], exit_ or PARAMS["exit_band"])
    feats = panel.prices.features
    g = {i: {t: gauge[i][t] for t in feats[i]} for i in range(len(news)) if feats[i]}
    active = {i: {t: states[i][t] for t in feats[i]} for i in range(len(news)) if feats[i]}
    combined, coefficients = fc.sentiment_correction(panel, news, price_cal, price_fc, CORRECTION_YEARS, variant)
    return g, active, combined, coefficients


def strategy_specs(g, active, combined, price_fc):
    return {"Sentiment": {"kind": "views", "signal": g, "active": active},
            "Combined": {"kind": "views", "signal": combined, "active": None},
            "Price": {"kind": "views", "signal": dict(price_fc), "active": None},
            "MinVar": {"kind": "minvar"}}


def finish_frame(frame):
    frame = frame.copy()
    frame.loc[frame.index[0], "Index"] -= PARAMS["base_cost_bps"] / 1e4   # entry into the ETF
    return frame


def metrics_table(frame, names):
    rf = frame.risk_free.to_numpy()
    out = {}
    for name in names:
        m = scalar_metrics(frame[name].to_numpy(), rf)
        if f"turnover_{name}" in frame:
            m.update(turnover=float(frame[f"turnover_{name}"].mean()), costs=float(frame[f"cost_{name}"].mean() * 52),
                     holdings=float(frame[f"holdings_{name}"].mean()), views=float(frame[f"views_{name}"].mean()),
                     skill=float(frame[f"skill_{name}"].mean()))
        out[name] = m
    return out


# --------------------------------------------------------------------------- analyses
def test_indices(decisions):
    return [i for i, d in enumerate(decisions) if PARAMS["evaluation_start"] <= d["week"] <= PARAMS["evaluation_end"]]


def overlay_diagnostics(panel, price_fc, g, idx):
    """Return in the direction of the price signal when sentiment agrees, is neutral or disagrees."""
    groups = {"agree": [], "neutral": [], "disagree": []}
    shares = {k: [] for k in groups}
    for i in idx:
        f = price_fc.get(i, {})
        rel = panel.relative[i]
        names = [t for t in f if t in rel and t in g.get(i, {})]
        if len(names) < 20:
            continue
        z = fc.zscore({t: f[t] for t in names})
        bucket = {k: [] for k in groups}
        for t in names:
            agreement = np.sign(z[t]) * g[i][t]
            key = "agree" if agreement >= 0.5 else "disagree" if agreement <= -0.5 else "neutral"
            bucket[key].append(np.sign(z[t]) * rel[t])
        for k, v in bucket.items():
            shares[k].append(len(v) / len(names))
            if v:
                groups[k].append(float(np.mean(v)))
    return {k: {**hac(v), "share": float(np.mean(shares[k]))} for k, v in groups.items()}


def decay(panel, signal, idx):
    out = []
    for k in range(1, PARAMS["decay_horizons"] + 1):
        ics = []
        for i in idx:
            j = i + k - 1
            if j >= len(panel.relative):
                continue
            f = signal.get(i, {})
            names = [t for t in f if t in panel.relative[j]]
            if len(names) < 20:
                continue
            x = np.array([f[t] for t in names])
            if x.std() < 1e-12:
                continue
            from scipy.stats import spearmanr
            ics.append(spearmanr(x, [panel.relative[j][t] for t in names]).statistic)
        out.append({"horizon": k, **hac(ics)})
    return out


def daily_abnormal(prices, universe):
    """Daily USD returns minus the equal-weighted return of the same region's stocks."""
    region = {f["ticker"]: f["region"] for f in universe}
    frames = {}
    for t, points in prices.stock.items():
        days = [p[0] for p in points]
        _, usd, _ = prices.daily[t]
        if len(usd) != len(days):        # rows without an FX rate were skipped
            days = days[len(days) - len(usd):]
        frames[t] = pd.Series(usd, index=days).pct_change()
    table = pd.DataFrame(frames).sort_index()
    out = {}
    for reg in set(region.values()):
        cols = [t for t in table if region[t] == reg]
        market = table[cols].mean(axis=1)
        for t in cols:
            series = (table[t] - market).dropna()
            out[t] = (list(series.index), series.to_numpy())
    return out


def run_test(frozen):
    t0 = dt.datetime.now()
    model, weight, direction = frozen["price_model"], frozen["sector_weight"], frozen["direction"]
    variant = frozen["correction_variant"]
    universe, clock, decisions, prices = market_inputs()
    panel = fc.Panel(prices)
    audit = {}
    records, news = news_inputs(universe, clock, decisions, audit)
    print(f"Inputs ready ({len(records):,} headlines); forecasting with {model}", flush=True)
    price_fc = cached(f"forecast_{model}", lambda: fc.price_forecasts(panel, model, "2008", "2025"))
    price_cal = fc.calibrate(panel, price_fc, [str(y) for y in range(2009, 2026)])
    g, active, combined, coefficients = signals(panel, news, price_fc, price_cal, weight, direction, variant)
    ctx = cached("context", lambda: build_context(prices, panel))
    specs = strategy_specs(g, active, combined, price_fc)
    print("Simulating base case", flush=True)
    frame, weights = simulate(ctx, specs, keep_weights=True)
    frame = finish_frame(frame)
    names = ["Sentiment", "Combined", "Price", "MinVar", "EW", "Index"]
    idx = test_indices(decisions)
    a = {"selection": json.loads((OUT / "selection.json").read_text()), "frozen": frozen, "audit": audit}
    a["metrics"] = metrics_table(frame, names)
    rf = frame.risk_free.to_numpy()
    pairs = [("Combined", "Price"), ("Sentiment", "EW"), ("Sentiment", "MinVar"), ("Sentiment", "Index"),
             ("Combined", "EW"), ("Price", "EW")]
    a["bootstrap"] = {f"{x}|{y}": paired_bootstrap(frame[x].to_numpy(), frame[y].to_numpy(), rf) for x, y in pairs}
    index_excess = frame["Index"].to_numpy() - rf
    a["index_alpha"] = {n: index_regression(frame[n].to_numpy() - rf, index_excess) for n in names if n != "Index"}
    sharpe = [a["metrics"][n]["sharpe"] for n in ("Sentiment", "Combined", "Price", "MinVar")]
    a["deflated_sharpe"] = {n: deflated_sharpe(frame[n].to_numpy(), rf, 4, float(np.var(sharpe, ddof=1)))
                            for n in ("Sentiment", "Combined", "Price")}
    up = frame["Index"].to_numpy() > 0
    a["conditions"] = {}
    for x, y in (("Combined", "Price"), ("Sentiment", "EW"), ("Sentiment", "MinVar")):
        d = frame[x].to_numpy() - frame[y].to_numpy()
        a["conditions"][f"{x}|{y}"] = {"up": hac(d[up]), "down": hac(d[~up])}
    # Break-even cost: per-trade cost (bp) at which the net return gap to the comparison is zero.
    a["break_even_bps"] = {}
    for x, y in (("Sentiment", "EW"), ("Combined", "Price"), ("Price", "EW")):
        gap = frame[f"gross_{x}"].mean() - frame[f"gross_{y}"].mean()
        traded = 2 * (frame[f"turnover_{x}"].mean() - frame[f"turnover_{y}"].mean())
        # Defined only when x earns more before costs and trades more than y.
        a["break_even_bps"][f"{x}|{y}"] = float(gap / traded * 1e4) if traded > 0 and gap > 0 else None
    # Forecast quality in the test period.
    a["ic"] = {"Price": fc.weekly_ic(panel, price_fc, idx), "Combined": fc.weekly_ic(panel, combined, idx),
               "Sentiment": fc.weekly_ic(panel, g, idx)}
    diff = []
    p_ic, c_ic = dict(a["ic"]["Price"]), dict(a["ic"]["Combined"])
    for i in p_ic:
        if i in c_ic:
            diff.append(c_ic[i] - p_ic[i])
    a["ic_summary"] = {k: {**hac([v for _, v in s]), "positive_share": float(np.mean([v > 0 for _, v in s]))}
                       for k, s in a["ic"].items()}
    a["ic_summary"]["Combined-Price"] = hac(diff)
    a["overlay"] = overlay_diagnostics(panel, price_fc, g, idx)
    a["correction"] = coefficients
    a["decay"] = {"Sentiment": decay(panel, g, idx), "Price": decay(panel, price_fc, idx),
                  "Combined": decay(panel, combined, idx)}
    # Does sentiment move prices beyond the price information? Fama-MacBeth on the test period.
    fm_rows = []
    for i in idx:
        data = {}
        for t, rel in panel.relative[i].items():
            f = prices.features[i][t]
            data[t] = {"y": rel, "firm": news[i][t]["gauge_0"], "sector": news[i][t]["gauge_1"],
                       **{k: f[k] for k in ("momentum_12_1", "reversal_4w", "volatility_60d", "dollar_volume", "beta_52w")}}
        fm_rows.append((i, data))
    a["fama_macbeth"] = {"sentiment": fama_macbeth(fm_rows, ["firm", "sector"]),
                         "controls": fama_macbeth(fm_rows, ["firm", "sector", "momentum_12_1", "reversal_4w",
                                                             "volatility_60d", "dollar_volume", "beta_52w"])}
    region = {f["ticker"]: f["region"] for f in universe}
    for reg in sorted(set(region.values())):
        sub = [(i, {t: v for t, v in d.items() if region[t] == reg}) for i, d in fm_rows]
        a["fama_macbeth"][reg] = fama_macbeth(sub, ["firm", "sector", "momentum_12_1", "reversal_4w",
                                                    "volatility_60d", "dollar_volume", "beta_52w"])
    # Event study around sentiment triggers (raw gauge, before the sign chosen on validation).
    raw_gauge = [{t: news[i][t][f"gauge_{weight:g}"] for t in news[i]} for i in range(len(news))]
    states = view_states(raw_gauge, PARAMS["entry_band"], PARAMS["exit_band"])
    events, previous = [], {}
    for i, week in enumerate(states):
        for t, s in week.items():
            if s != 0 and previous.get(t, 0) == 0 and i in idx and prices.targets[i][t]["realised_return"] is not None:
                events.append((t, prices.targets[i][t]["execution_date"], s))
            previous[t] = s
    a["event_study"] = event_study(events, daily_abnormal(prices, universe), PARAMS["event_window_days"])
    print(f"Base analyses done ({dt.datetime.now() - t0}); robustness", flush=True)
    # Robustness: costs, position cap, bands, half-life, news delay, sub-periods.
    rob = {"Base case": a["metrics"]}
    for bps in PARAMS["cost_grid_bps"]:
        if bps != PARAMS["base_cost_bps"]:
            f_, _ = simulate(ctx, specs, cost_bps=bps)
            rob[f"Costs {bps:g} bp per trade"] = metrics_table(finish_frame(f_), names[:5])
    f_, _ = simulate(ctx, specs, cap=0.10)
    rob["Position limit 10%"] = metrics_table(finish_frame(f_), names[:5])
    for entry in (1.0, 2.0):
        g2, act2, _, _ = signals(panel, news, price_fc, price_cal, weight, direction, variant, entry=entry)
        f_, _ = simulate(ctx, {"Sentiment": {"kind": "views", "signal": g2, "active": act2}})
        rob[f"Entry band {entry:g} s.d."] = metrics_table(finish_frame(f_), ["Sentiment"])
    for hl in (1, 4):
        _, news_hl = news_inputs(universe, clock, decisions, {}, half_life=hl)
        g2, act2, comb2, _ = signals(panel, news_hl, price_fc, price_cal, weight, direction, variant)
        f_, _ = simulate(ctx, {"Sentiment": {"kind": "views", "signal": g2, "active": act2},
                               "Combined": {"kind": "views", "signal": comb2, "active": None}})
        rob[f"Gauge half-life {hl} week{'s' if hl > 1 else ''}"] = metrics_table(finish_frame(f_), ["Sentiment", "Combined"])
    _, news_delay = news_inputs(universe, clock, decisions, {}, extra_sessions=1)
    g2, act2, comb2, _ = signals(panel, news_delay, price_fc, price_cal, weight, direction, variant)
    f_, _ = simulate(ctx, {"Sentiment": {"kind": "views", "signal": g2, "active": act2},
                           "Combined": {"kind": "views", "signal": comb2, "active": None}})
    rob["News one session later"] = metrics_table(finish_frame(f_), ["Sentiment", "Combined"])
    for label, (s, e) in PARAMS["subperiods"].items():
        mask = ((frame.week >= s) & (frame.week <= e)).to_numpy()
        rob[label] = {n: scalar_metrics(frame[n].to_numpy()[mask], rf[mask]) for n in names}
    a["robustness"] = rob
    a["runtime"] = str(dt.datetime.now() - t0)
    OUT.mkdir(exist_ok=True)
    frame.to_csv(OUT / "weekly_returns.csv", index=False)
    weights.to_csv(OUT / "weights.csv", index=False)
    (OUT / "analysis.json").write_text(json.dumps(a, indent=1, default=str))
    from .outputs import make_outputs
    make_outputs(a, frame, weights, universe, prices, news, records, g, active, decisions)
    print("Test run complete", a["runtime"], flush=True)
