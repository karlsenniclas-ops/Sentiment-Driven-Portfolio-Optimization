"""Performance metrics, uncertainty and the tests of whether sentiment moves prices."""
import math
from bisect import bisect_left

import numpy as np
from scipy.stats import norm

from .config import PARAMS


def path_metrics(r, rf):
    """Vectorised over rows (bootstrap paths) when r is 2-D."""
    r, rf = np.atleast_2d(r), np.atleast_2d(rf)
    wealth = np.cumprod(1 + r, axis=1)
    peak = np.maximum.accumulate(np.column_stack([np.ones(len(r)), wealth]), axis=1)[:, 1:]
    excess = r - rf
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2, axis=1))
    return {"cagr": wealth[:, -1] ** (52 / r.shape[1]) - 1,
            "volatility": r.std(axis=1, ddof=1) * math.sqrt(52),
            "sharpe": excess.mean(axis=1) / excess.std(axis=1, ddof=1) * math.sqrt(52),
            "sortino": excess.mean(axis=1) / downside * math.sqrt(52),
            "max_drawdown": (wealth / peak - 1).min(axis=1)}


def scalar_metrics(r, rf):
    return {k: float(v[0]) for k, v in path_metrics(r, rf).items()}


def hac(values):
    """Mean with a Newey-West standard error."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    lags, n = PARAMS["hac_lags"], len(values)
    if n < 3:
        return {"mean": float("nan"), "se": float("nan"), "t": float("nan"), "n": n}
    c = values - values.mean()
    variance = c @ c
    for lag in range(1, lags + 1):
        variance += 2 * (1 - lag / (lags + 1)) * (c[lag:] @ c[:-lag])
    se = math.sqrt(max(variance, 0)) / n
    return {"mean": float(values.mean()), "se": se, "t": float(values.mean() / se) if se else 0.0, "n": n}


def paired_bootstrap(a, b, rf):
    """Circular block bootstrap of differences in CAGR, Sharpe ratio and maximum drawdown."""
    rng = np.random.default_rng(PARAMS["bootstrap_seed"])
    n, block, draws = len(a), PARAMS["bootstrap_block_weeks"], PARAMS["bootstrap_resamples"]
    idx = (rng.integers(0, n, (draws, math.ceil(n / block), 1)) + np.arange(block)).reshape(draws, -1)[:, :n] % n
    ma, mb = path_metrics(a[idx], rf[idx]), path_metrics(b[idx], rf[idx])
    oa, ob = scalar_metrics(a, rf), scalar_metrics(b, rf)
    return {k: {"difference": oa[k] - ob[k], "ci": np.percentile(ma[k] - mb[k], [2.5, 97.5]).tolist(),
                "p_positive": float(np.mean(ma[k] - mb[k] > 0))}
            for k in ("cagr", "sharpe", "max_drawdown")}


def index_regression(y, x):
    """Annualised alpha and beta against the index, Newey-West t-statistic for alpha."""
    design = np.column_stack([np.ones(len(y)), x])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    scores = design * (y - design @ coef)[:, None]
    meat = scores.T @ scores
    for lag in range(1, PARAMS["hac_lags"] + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (PARAMS["hac_lags"] + 1)) * (cross + cross.T)
    bread = np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(bread @ meat @ bread))
    return {"alpha": float(coef[0] * 52), "alpha_t": float(coef[0] / se[0]), "beta": float(coef[1])}


def deflated_sharpe(r, rf, trials, sharpe_variance):
    """Probability that the true Sharpe ratio exceeds the best expected among `trials` unskilled
    strategies (Bailey and Lopez de Prado, 2014). Weekly units inside, annualised input variance."""
    excess = np.asarray(r) - np.asarray(rf)
    t = len(excess)
    sr = excess.mean() / excess.std(ddof=1)
    skew = float(((excess - excess.mean()) ** 3).mean() / excess.std() ** 3)
    kurt = float(((excess - excess.mean()) ** 4).mean() / excess.std() ** 4)
    gamma = 0.5772156649
    v = sharpe_variance / 52
    sr0 = math.sqrt(v) * ((1 - gamma) * norm.ppf(1 - 1 / trials) + gamma * norm.ppf(1 - 1 / (trials * math.e)))
    z = (sr - sr0) * math.sqrt(t - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(norm.cdf(z))


def fama_macbeth(rows, regressors):
    """Weekly cross-sectional OLS of next week's relative return on standardised regressors;
    the time series of coefficients is averaged with Newey-West standard errors.
    rows: list of (week index, {ticker: {"y": ..., regressor: ...}})."""
    coefs = {k: [] for k in regressors}
    r2 = []
    for _, data in rows:
        names = [t for t, v in data.items() if all(np.isfinite(v[k]) for k in regressors)]
        if len(names) < 30:
            continue
        x = np.array([[data[t][k] for k in regressors] for t in names])
        sd = x.std(axis=0)
        x = (x - x.mean(axis=0)) / np.where(sd > 1e-12, sd, 1)
        y = np.array([data[t]["y"] for t in names])
        design = np.column_stack([np.ones(len(names)), x])
        b = np.linalg.lstsq(design, y, rcond=None)[0]
        for k, v in zip(regressors, b[1:]):
            coefs[k].append(v)
        fitted = design @ b
        r2.append(1 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2))
    return {k: hac(v) for k, v in coefs.items()} | {"r2": float(np.mean(r2)), "weeks": len(r2)}


def event_study(events, daily, window):
    """Mean cumulative market-adjusted return around events; day 0 is the execution close.
    events: list of (ticker, event date, sign); daily: {ticker: (dates, abnormal returns)}."""
    lo, hi = window
    paths = {1: [], -1: []}
    for ticker, date, sign in events:
        dates, abnormal = daily[ticker]
        k = bisect_left(dates, date)
        if k >= len(dates) or dates[k] != date or k + lo < 1 or k + hi >= len(dates):
            continue
        segment = abnormal[k + lo:k + hi + 1]
        if np.isnan(segment).any():
            continue
        paths[sign].append(np.cumsum(segment))   # cumulative from the start of the window
    out = {}
    for sign, p in paths.items():
        p = np.array(p)
        out[sign] = {"n": len(p), "mean": p.mean(axis=0).tolist() if len(p) else [],
                     "se": (p.std(axis=0, ddof=1) / math.sqrt(len(p))).tolist() if len(p) > 1 else []}
    out["days"] = list(range(lo, hi + 1))
    return out
