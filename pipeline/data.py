"""Calendars, risk-free rate, prices in U.S. dollars, weekly price features and targets."""
import csv
import datetime as dt
import math
from bisect import bisect_left, bisect_right

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from .config import DATA, PARAMS

UTC = dt.timezone.utc


def load_universe():
    with (DATA / "universe.csv").open(newline="", encoding="utf-8") as stream:
        return [r for r in csv.DictReader(stream) if r["included"] == "1"]


class ExchangeClock:
    def __init__(self):
        self._schedules = {}

    def schedule(self, exchange):
        if exchange not in self._schedules:
            s = xcals.get_calendar(exchange, start=PARAMS["calendar_start"], end=PARAMS["calendar_end"]).schedule
            self._schedules[exchange] = {
                "dates": [x.date().isoformat() for x in s.index],
                "opens": [x.to_pydatetime().astimezone(UTC) for x in s["open"]],
                "closes": [x.to_pydatetime().astimezone(UTC) for x in s["close"]],
            }
        return self._schedules[exchange]

    def news_available(self, exchange, date, extra_sessions=0):
        """Headlines carry a date only, so a headline counts from the first exchange opening after that date."""
        end_of_date = dt.datetime.combine(dt.date.fromisoformat(date), dt.time.max, UTC)
        opens = self.schedule(exchange)["opens"]
        index = bisect_right(opens, end_of_date) + extra_sessions
        return opens[index] if index < len(opens) else None

    def execution(self, exchange, decision_week):
        """First session on or after the Friday of the decision week."""
        friday = dt.date.fromisoformat(decision_week) + dt.timedelta(days=4)
        s = self.schedule(exchange)
        index = bisect_left(s["dates"], friday.isoformat())
        return s["dates"][index], s["closes"][index]

    def close_for_date(self, exchange, date):
        s = self.schedule(exchange)
        index = bisect_left(s["dates"], date)
        if index == len(s["dates"]) or s["dates"][index] != date:
            raise ValueError(f"{date} is not a session on {exchange}")
        return s["closes"][index]


def decision_calendar():
    week = dt.date.fromisoformat(PARAMS["first_decision_week"])
    last = dt.date.fromisoformat(PARAMS["last_decision_week"])
    decisions = []
    while week <= last:
        day = week + dt.timedelta(days=PARAMS["cutoff_weekday"])
        decisions.append({"decision_week": week.isoformat(),
                          "cutoff": dt.datetime.combine(day, dt.time(PARAMS["cutoff_hour_utc"]), UTC),
                          "week": (week + dt.timedelta(days=7)).isoformat()})  # Monday of the holding week
        week += dt.timedelta(days=7)
    return decisions


class RiskFreeSeries:
    """FRED DGS3MO; the last observation strictly before the decision date."""

    def __init__(self):
        self.dates, self.values = [], []
        with (DATA / "us_risk_free_dgs3mo.csv").open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row.get("DGS3MO", ".") not in ("", "."):
                    self.dates.append(dt.date.fromisoformat(row["observation_date"]))
                    self.values.append(float(row["DGS3MO"]) / 100)

    def weekly(self, cutoff):
        index = bisect_left(self.dates, cutoff.date()) - 1
        return (1 + self.values[max(index, 0)]) ** (1 / 52) - 1


def _series(path, columns):
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str).str[:10]
    return frame[columns].apply(pd.to_numeric, errors="coerce").dropna()


def rank_normalise(rows, keys):
    """Map each input to [-1, 1] by its rank across stocks in the same week."""
    tickers = list(rows)
    out = {t: {} for t in tickers}
    for key in keys:
        values = pd.Series([rows[t][key] for t in tickers], dtype=float)
        ranks = values.rank(method="average").to_numpy()
        scaled = 2 * (ranks - 1) / (len(values) - 1) - 1 if len(values) > 1 else np.zeros(len(values))
        for t, v in zip(tickers, scaled):
            out[t][key] = float(v)
    return out


class PricePanel:
    """Prices in U.S. dollars as they were known at each cutoff, weekly features and targets."""

    def __init__(self, universe, clock, decisions):
        self.clock, self.decisions = clock, decisions
        self.firms = {r["ticker"]: r for r in universe}
        lag = dt.timedelta(minutes=PARAMS["price_publication_lag_minutes"])
        self.fx = {}
        for pair in ("EURUSD", "GBPUSD"):
            s = _series(DATA / "fx" / f"{pair}.csv", ["Close"])["Close"]
            # A daily FX close is treated as known from midnight UTC after its date.
            self.fx[pair] = ([d for d in s.index], s.to_numpy(),
                             [dt.datetime.combine(dt.date.fromisoformat(d) + dt.timedelta(days=1), dt.time(), UTC)
                              for d in s.index])
        self.stock, self.invalid_sessions = {}, []
        for ticker, firm in self.firms.items():
            s = _series(DATA / "prices" / f"{ticker}.csv", ["Adj Close"])["Adj Close"]
            v = _series(DATA / "prices" / f"{ticker}_volume.csv", ["Close", "Volume"])
            points = []
            for day, value in s.items():
                try:
                    stamp = clock.close_for_date(firm["exchange"], day) + lag
                except ValueError:
                    self.invalid_sessions.append((ticker, day))
                    continue
                close, shares = (v.loc[day, "Close"], v.loc[day, "Volume"]) if day in v.index else (np.nan, np.nan)
                points.append((day, float(value), stamp, float(close * shares) if shares > 0 else np.nan))
            self.stock[ticker] = points
        self.index = _series(DATA / "prices" / "URTH.csv", ["Adj Close"])["Adj Close"]
        self.daily = {t: self._daily_usd(t) for t in self.firms}
        self._stamps = {t: [p[2] for p in pts] for t, pts in self.stock.items()}
        self._days = {t: [p[0] for p in pts] for t, pts in self.stock.items()}
        self.signals = [{t: self._signal(t, d["cutoff"]) for t in self.firms} for d in decisions]
        self.features = self._features()
        self.targets = [{t: self.target(t, d["decision_week"]) for t in self.firms} for d in decisions]

    # -- currency -------------------------------------------------------------------------
    def _pair(self, ticker):
        currency = self.firms[ticker]["currency"]
        return {"USD": (None, 1.0), "EUR": ("EURUSD", 1.0), "GBp": ("GBPUSD", 0.01)}[currency]

    def _fx_on_or_before(self, pair, day):
        days, values, _ = self.fx[pair]
        j = bisect_right(days, day) - 1
        return (days[j], values[j]) if j >= 0 else (None, None)

    def _fx_known_at(self, pair, moment):
        days, values, stamps = self.fx[pair]
        j = bisect_left(stamps, moment) - 1
        return (values[j], stamps[j]) if j >= 0 else (None, None)

    def _daily_usd(self, ticker):
        """Daily USD prices and dollar volume; FX from the previous FX date (known at that close)."""
        pair, scale = self._pair(ticker)
        available, price, dollar = [], [], []
        for day, value, stamp, traded in self.stock[ticker]:
            rate = 1.0
            if pair:
                prev = (dt.date.fromisoformat(day) - dt.timedelta(days=1)).isoformat()
                _, rate = self._fx_on_or_before(pair, prev)
                if rate is None:
                    continue
            available.append(stamp)
            price.append(value * scale * rate)
            dollar.append(traded * scale * rate)
        return available, np.array(price), np.array(dollar)

    # -- signals and outcomes ---------------------------------------------------------------
    def _signal(self, ticker, cutoff):
        """Latest close published before the cutoff, converted with the FX rate known then."""
        i = bisect_left(self._stamps[ticker], cutoff) - 1
        if i < 0:
            return None
        day, price, available, _ = self.stock[ticker][i]
        pair, scale = self._pair(ticker)
        if pair:
            rate, fx_stamp = self._fx_known_at(pair, cutoff)
            if rate is None:
                return None
            price, available = price * scale * rate, max(available, fx_stamp)
        return {"value": price, "available": available, "age_days": (cutoff.date() - dt.date.fromisoformat(day)).days}

    def valuation(self, ticker, date, close):
        """USD value at a session close, using that date's FX close (known at midnight UTC after it)."""
        i = bisect_right(self._days[ticker], date) - 1
        if i < 0:
            return None
        day, price, available, _ = self.stock[ticker][i]
        pair, scale = self._pair(ticker)
        fx_day = date
        if pair:
            fx_day, rate = self._fx_on_or_before(pair, date)
            if rate is None:
                return None
            price = price * scale * rate
            available = max(available, dt.datetime.combine(dt.date.fromisoformat(fx_day) + dt.timedelta(days=1),
                                                           dt.time(), UTC))
        lag = dt.timedelta(minutes=PARAMS["price_publication_lag_minutes"])
        return {"value": price, "available": max(available, close + lag), "stale": day != date or fx_day != date}

    def target(self, ticker, week):
        next_week = (dt.date.fromisoformat(week) + dt.timedelta(days=7)).isoformat()
        exchange = self.firms[ticker]["exchange"]
        start_day, start = self.clock.execution(exchange, week)
        end_day, end = self.clock.execution(exchange, next_week)
        first, last = self.valuation(ticker, start_day, start), self.valuation(ticker, end_day, end)
        last_price_day = self.stock[ticker][-1][0] if self.stock[ticker] else ""
        complete = (first is not None and last is not None and end_day <= PARAMS["observation_through_date"]
                    and end_day <= last_price_day)
        return {"execution_date": start_day, "execution_at": start,
                "realised_return": last["value"] / first["value"] - 1 if complete else None,
                "outcome_available": max(first["available"], last["available"]) if complete else None,
                "stale_quote": bool(complete and (first["stale"] or last["stale"])),
                "zero_horizon": start == end}

    def _features(self):
        """Standard cross-sectional return predictors from prices and trading activity."""
        panel, history = [], {t: [] for t in self.firms}
        market = []
        for decision, snapshot in zip(self.decisions, self.signals):
            weekly = []
            for ticker, point in snapshot.items():
                h = history[ticker]
                h.append(point["value"] if point else None)
                if len(h) >= 2 and h[-1] is not None and h[-2] is not None:
                    weekly.append(h[-1] / h[-2] - 1)
            market.append(float(np.mean(weekly)) if weekly else np.nan)
            mkt = np.array(market[-52:])
            row = {}
            for ticker, point in snapshot.items():
                values = history[ticker][-53:]
                if (point is None or point["age_days"] > PARAMS["max_price_age_days"] or len(values) < 53
                        or any(v is None for v in values) or len(mkt) < 52 or np.isnan(mkt).any()):
                    continue
                available, price, dollar = self.daily[ticker]
                m = bisect_left(available, decision["cutoff"])
                if m < 121:
                    continue
                daily_returns = price[m - 120:m][1:] / price[m - 120:m][:-1] - 1
                recent_dollar, long_dollar = dollar[m - 20:m], dollar[m - 120:m]
                if np.isnan(recent_dollar).mean() > 0.5:
                    continue
                v = np.array(values, dtype=float)
                weekly = v[1:] / v[:-1] - 1
                last20 = daily_returns[-20:]
                valid = ~np.isnan(recent_dollar)
                row[ticker] = {
                    "momentum_12_1": float(v[-5] / v[0] - 1),
                    "reversal_4w": float(-(v[-1] / v[-5] - 1)),
                    "reversal_1w": float(-weekly[-1]),
                    "volatility_60d": float(np.std(daily_returns[-60:], ddof=1)),
                    "max_return_20d": float(last20.max()),
                    "beta_52w": float(np.cov(weekly, mkt, ddof=1)[0, 1] / np.var(mkt, ddof=1)),
                    "high_52w": float(v[-1] / v[1:].max() - 1),
                    "dollar_volume": float(np.log(np.nanmean(recent_dollar))),
                    "illiquidity": float(np.mean(np.abs(last20[valid]) / recent_dollar[valid])),
                    "abnormal_volume": float(np.log(np.nanmean(recent_dollar) / np.nanmean(long_dollar))),
                    "weekly_returns": weekly.tolist(),
                }
            panel.append(row)
        return panel
