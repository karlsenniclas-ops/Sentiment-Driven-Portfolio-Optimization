"""Black-Litterman views on an equal-weight prior and cost-aware mean-variance optimisation."""
from bisect import bisect_left

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import PARAMS


def ledoit_wolf(x):
    """Ledoit-Wolf (2004) shrinkage of the sample covariance towards a scaled identity."""
    x = x - x.mean(axis=0)
    n_obs, n = x.shape
    sample = x.T @ x / n_obs
    target = np.trace(sample) / n * np.eye(n)
    d2 = np.sum((sample - target) ** 2)
    b2 = sum(np.sum((np.outer(r, r) - sample) ** 2) for r in x) / n_obs ** 2
    shrink = min(b2, d2) / d2 if d2 > 0 else 1.0
    return shrink * target + (1 - shrink) * sample


def skill_history(signal, returns, outcome_stamp):
    """Weekly rank IC of a signal against realised returns, stamped with the time the week's
    last outcome became known. signal/returns: {week index: {ticker: value}}."""
    history = []
    for i, values in signal.items():
        names = [t for t in values if t in returns.get(i, {})]
        if len(names) < 10:
            continue
        x = np.array([values[t] for t in names])
        if x.std() < 1e-12:
            continue
        ic = spearmanr(x, [returns[i][t] for t in names]).statistic
        if np.isfinite(ic):
            history.append((outcome_stamp[i], float(ic)))
    history.sort()
    return [a for a, _ in history], np.cumsum([0.0] + [ic for _, ic in history])


def skill(history, cutoff):
    """Mean IC over the last three years of observed weeks, floored at zero: a signal without
    demonstrated skill gets no views."""
    stamps, cumulative = history
    n = bisect_left(stamps, cutoff)
    first = max(0, n - PARAMS["skill_window_weeks"])
    if n - first < PARAMS["ic_min_weeks"]:
        return 0.0
    return max(0.0, (cumulative[n] - cumulative[first]) / (n - first))


def posterior_mean(sigma, prior_weights, views):
    """Black-Litterman with He-Litterman view uncertainty (Omega = tau * diag(P Sigma P')).
    views: {position: alpha}. Prior returns pi = delta * Sigma * w_prior."""
    delta = PARAMS["risk_aversion"]
    pi = delta * sigma @ prior_weights
    if not views:
        return pi
    s = np.array(sorted(views))
    alpha = np.array([views[k] for k in s])
    tau = PARAMS["bl_tau"]
    a = tau * sigma[np.ix_(s, s)] + tau * np.diag(np.diag(sigma)[s])
    return pi + tau * sigma[:, s] @ np.linalg.solve(a, alpha)


def optimise(mu, sigma, pre, cost, cap, risk_only=False):
    """max mu'w - delta/2 w'Sigma w - (c/H) |w - w_pre|_1  s.t. sum w = 1, 0 <= w <= cap."""
    n = len(pre)
    w = cp.Variable(n)
    risk = 0.5 * PARAMS["risk_aversion"] * cp.quad_form(w, cp.psd_wrap(sigma))
    trade = cost / PARAMS["holding_weeks"] * cp.norm1(w - pre)
    objective = risk + trade if risk_only else -mu @ w + risk + trade
    problem = cp.Problem(cp.Minimize(objective), [cp.sum(w) == 1, w >= 0, w <= cap])
    try:
        problem.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        problem.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, max_iter=200000)
    x = np.clip(np.asarray(w.value).ravel(), 0, cap)
    return x / x.sum()


def simulate(ctx, strategies, cost_bps=None, cap=None, keep_weights=False, start=None, end=None):
    """Weekly walk-forward portfolio simulation.
    strategies: {name: spec}; spec = {"kind": "views", "signal": {i: {t: score}}, "active": {i: {t: sign}} | None,
    "standardise": bool} or {"kind": "minvar"}. Equal weight and the index are always added."""
    cost = (PARAMS["base_cost_bps"] if cost_bps is None else cost_bps) / 1e4
    cap = PARAMS["position_cap"] if cap is None else cap
    start, end = start or PARAMS["evaluation_start"], end or PARAMS["evaluation_end"]
    skills = {name: skill_history(spec["signal"], ctx["returns"], ctx["outcome_stamp"])
              for name, spec in strategies.items() if spec["kind"] == "views"}
    names = [*strategies, "EW"]
    held = {name: {} for name in names}    # ticker -> weight after last week's drift
    rows, weights = [], []
    for i in ctx["weeks"]:
        week = ctx["decisions"][i]["week"]
        if not start <= week <= end:
            continue
        universe = ctx["investable"][i]
        sigma = ctx["covariance"][i]
        vol = np.sqrt(np.diag(sigma))
        returns = np.array([ctx["returns"][i][t] for t in universe])
        n = len(universe)
        position = {t: k for k, t in enumerate(universe)}
        equal = np.full(n, 1 / n)
        row = {"week": week, "risk_free": ctx["risk_free"][i], "Index": ctx["index_return"][i], "n_stocks": n}
        for name in names:
            pre = np.array([held[name].get(t, 0.0) for t in universe])
            leaving = sum(v for t, v in held[name].items() if t not in position)
            if name == "EW":
                target, strength, active = equal, 0.0, 0
            else:
                spec = strategies[name]
                if spec["kind"] == "minvar":
                    target, strength, active = optimise(np.zeros(n), sigma, pre, cost, cap, risk_only=True), 0.0, 0
                else:
                    strength = skill(skills[name], ctx["decisions"][i]["cutoff"])
                    signal = spec["signal"].get(i, {})
                    if spec.get("active") is not None:
                        chosen = [t for t, s in spec["active"].get(i, {}).items() if s != 0 and t in position]
                        scores = {t: float(np.clip(signal.get(t, 0.0), -3, 3)) for t in chosen}
                    else:
                        values = np.array([signal.get(t, 0.0) for t in universe])
                        z = (values - values.mean()) / values.std() if values.std() > 1e-12 else 0 * values
                        scores = dict(zip(universe, z))
                    views = {position[t]: strength * vol[position[t]] * s for t, s in scores.items()} if strength > 0 else {}
                    active = len(views)
                    mu = posterior_mean(sigma, equal, views)
                    target = optimise(mu, sigma, pre, cost, cap)
            traded = np.abs(target - pre).sum() + leaving
            gross = float(target @ returns)
            row.update({name: gross - cost * traded, f"gross_{name}": gross, f"cost_{name}": cost * traded,
                        f"turnover_{name}": traded / 2, f"skill_{name}": strength, f"views_{name}": active,
                        f"holdings_{name}": int((target > 0.001).sum())})
            if keep_weights:
                weights.extend({"week": week, "strategy": name, "ticker": t, "weight": float(w)}
                               for t, w in zip(universe, target))
            grown = target * (1 + returns)
            held[name] = dict(zip(universe, grown / grown.sum()))
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(weights)
