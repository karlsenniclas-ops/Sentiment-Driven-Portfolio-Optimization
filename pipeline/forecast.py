"""Price forecasting candidates, refitted every January on all data observable before the
first decision of the year (expanding window), and their weekly rank IC."""
import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from .config import FACTORS, PARAMS, PRICE_KEYS
from .data import rank_normalise


def zscore(values):
    """Cross-sectional z-score of a {ticker: value} mapping."""
    tickers = list(values)
    x = np.array([values[t] for t in tickers], dtype=float)
    sd = x.std()
    z = (x - x.mean()) / sd if sd > 1e-12 else np.zeros(len(x))
    return dict(zip(tickers, map(float, z)))


class Panel:
    """Weekly inputs shared by all models: ranked features, relative targets and availability."""

    def __init__(self, prices):
        self.prices, self.decisions = prices, prices.decisions
        self.ranked = [rank_normalise(f, PRICE_KEYS) if len(f) > 1 else {} for f in prices.features]
        self.relative = []
        for i, targets in enumerate(prices.targets):
            rows = {t: r["realised_return"] for t, r in targets.items()
                    if t in prices.features[i] and r["realised_return"] is not None and not r["zero_horizon"]}
            mean = np.mean(list(rows.values())) if rows else 0.0
            self.relative.append({t: v - mean for t, v in rows.items()})
        self.years = sorted({d["week"][:4] for d in self.decisions})

    def year_indices(self, year):
        return [i for i, d in enumerate(self.decisions) if d["week"][:4] == year]

    def training_weeks(self, cutoff):
        """Weeks whose every outcome was observable before the cutoff."""
        out = []
        for i, d in enumerate(self.decisions):
            if d["week"] < PARAMS["train_start"] or not self.relative[i]:
                continue
            stamps = [self.prices.targets[i][t]["outcome_available"] for t in self.relative[i]]
            if max(stamps) < cutoff:
                out.append(i)
        return out


# --------------------------------------------------------------------------- models
def factor_forecast(panel, i):
    raw = panel.prices.features[i]
    if len(raw) < 2:
        return {}
    scores = [zscore({t: sign * f[key] for t, f in raw.items()}) for key, sign in FACTORS.items()]
    return {t: float(np.mean([s[t] for s in scores])) for t in raw}


def _xy(panel, weeks):
    x, y = [], []
    for i in weeks:
        for t, target in panel.relative[i].items():
            x.append([panel.ranked[i][t][k] for k in PRICE_KEYS])
            y.append(target)
    return np.asarray(x), np.asarray(y)


def tree_model(name):
    if name == "Ridge":
        from sklearn.linear_model import RidgeCV
        return RidgeCV(alphas=PARAMS["ridge_alphas"])          # penalty by generalised cross-validation
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_jobs=2, **PARAMS["random_forest"])
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(early_stopping=False, **PARAMS["gbm"])


def _sequences(panel, weeks, length):
    """Sequences of the last `length` weekly ranked feature vectors per stock."""
    x, y, keys = [], [], []
    for i in weeks:
        for t in panel.ranked[i]:
            if i - length + 1 < 0 or any(t not in panel.ranked[j] for j in range(i - length + 1, i + 1)):
                continue
            x.append([[panel.ranked[j][t][k] for k in PRICE_KEYS] for j in range(i - length + 1, i + 1)])
            y.append(panel.relative[i].get(t, np.nan))
            keys.append((i, t))
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), keys


class LSTMForecaster:
    def __init__(self):
        import torch
        self.torch = torch
        cfg = PARAMS["lstm"]
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.set_num_threads(2)

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(len(PRICE_KEYS), cfg["hidden"], batch_first=True)
                self.drop = torch.nn.Dropout(cfg["dropout"])
                self.head = torch.nn.Linear(cfg["hidden"], 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(self.drop(out[:, -1])).squeeze(-1)

        self.net, self.cfg = Net(), cfg

    def fit(self, x, y, weeks_of_rows):
        torch, cfg = self.torch, self.cfg
        y = y * 100  # percent units for a well-scaled loss
        last = sorted(set(weeks_of_rows))[-52:]
        hold = np.isin(weeks_of_rows, last)
        xt, yt = torch.tensor(x[~hold]), torch.tensor(y[~hold])
        xv, yv = torch.tensor(x[hold]), torch.tensor(y[hold])
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"])
        best, best_state, wait = np.inf, None, 0
        generator = torch.Generator().manual_seed(cfg["seed"])
        for _ in range(cfg["epochs"]):
            self.net.train()
            order = torch.randperm(len(xt), generator=generator)
            for b in range(0, len(xt), cfg["batch"]):
                idx = order[b:b + cfg["batch"]]
                opt.zero_grad()
                loss = torch.mean((self.net(xt[idx]) - yt[idx]) ** 2)
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val = float(torch.mean((self.net(xv) - yv) ** 2))
            if val < best - 1e-6:
                best, wait = val, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= cfg["patience"]:
                    break
        self.net.load_state_dict(best_state)

    def predict(self, x):
        self.net.eval()
        with self.torch.no_grad():
            return self.net(self.torch.tensor(x)).numpy() / 100


def arima_forecasts(panel, years, out):
    """Per-stock ARMA on weekly returns between cutoffs; order chosen by AIC each January,
    then one-step forecasts with the fitted parameters through the year."""
    from statsmodels.tsa.arima.model import ARIMA
    tickers = list(panel.prices.firms)
    series = {}
    for t in tickers:
        values = [s[t]["value"] if s.get(t) else np.nan for s in panel.prices.signals]
        v = np.array(values, dtype=float)
        r = np.full(len(v), np.nan)
        r[1:] = v[1:] / v[:-1] - 1
        series[t] = r
    for year in years:
        idx = panel.year_indices(year)
        first, last = idx[0], idx[-1]
        for t in tickers:
            train = series[t][:first + 1]
            observed = np.where(~np.isnan(train))[0]
            if len(observed) < 104:
                continue
            start = max(observed[0], first + 1 - PARAMS["arima_window_weeks"])
            train = train[start:]
            best = None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for order in PARAMS["arima_orders"]:
                    try:
                        fit = ARIMA(train, order=order, trend="c").fit()
                    except Exception:
                        continue
                    if np.isfinite(fit.aic) and (best is None or fit.aic < best.aic):
                        best = fit
                if best is None:
                    continue
                full = series[t][start:min(last + 2, len(series[t]))]
                predicted = best.apply(full).predict()   # element j: forecast of full[j] given full[:j]
            offset = start
            for i in idx:
                j = i + 1 - offset
                if t in panel.prices.features[i] and 0 <= j < len(predicted):
                    out[i][t] = float(predicted[j])


def price_forecasts(panel, name, first_year, last_year):
    """Walk-forward forecasts for one candidate; returns {decision index: {ticker: forecast}}."""
    years = [y for y in panel.years if first_year <= y <= last_year]
    out = defaultdict(dict)
    if name == "Factor":
        for year in years:
            for i in panel.year_indices(year):
                out[i] = factor_forecast(panel, i)
        return out
    if name == "ARIMA":
        arima_forecasts(panel, years, out)
        return out
    for year in years:
        idx = panel.year_indices(year)
        weeks = panel.training_weeks(panel.decisions[idx[0]]["cutoff"])
        if name == "LSTM":
            length = PARAMS["lstm"]["sequence_weeks"]
            x, y, keys = _sequences(panel, weeks, length)
            ok = ~np.isnan(y)
            model = LSTMForecaster()
            model.fit(x[ok], y[ok], np.array([k[0] for k, o in zip(keys, ok) if o]))
            xp, _, keys_p = _sequences(panel, idx, length)
            for (i, t), f in zip(keys_p, model.predict(xp)):
                out[i][t] = float(f)
        else:
            x, y = _xy(panel, weeks)
            model = tree_model(name).fit(x, y)
            for i in idx:
                rows = panel.ranked[i]
                if rows:
                    pred = model.predict(np.array([[rows[t][k] for k in PRICE_KEYS] for t in rows]))
                    out[i].update(dict(zip(rows, map(float, pred))))
        print(f"    {name} {year}: trained on {len(weeks)} weeks", flush=True)
    return out


def weekly_ic(panel, forecast, indices, investable=None):
    """Spearman correlation between forecast and realised return across stocks, per week."""
    ics = []
    for i in indices:
        f = forecast.get(i, {})
        names = [t for t in panel.relative[i] if t in f and (investable is None or t in investable[i])]
        if len(names) < 20:
            continue
        x = np.array([f[t] for t in names])
        y = np.array([panel.relative[i][t] for t in names])
        ics.append((i, float(spearmanr(x, y).statistic) if x.std() > 1e-15 else 0.0))
    return ics


def calibrate(panel, forecast, years):
    """Convert a forecast score into an expected relative return: slope (through the origin) of
    realised relative returns on the score over the last three years, refitted every January and
    floored at zero."""
    out = {}
    for year in years:
        idx = panel.year_indices(year)
        weeks = [i for i in panel.training_weeks(panel.decisions[idx[0]]["cutoff"])
                 if forecast.get(i)][-PARAMS["skill_window_weeks"]:]
        x = np.array([forecast[i][t] for i in weeks for t in panel.relative[i] if t in forecast[i]])
        y = np.array([panel.relative[i][t] for i in weeks for t in panel.relative[i] if t in forecast[i]])
        slope = max(0.0, float(x @ y / (x @ x))) if len(weeks) >= 52 and x @ x > 0 else 0.0
        for i in idx:
            if forecast.get(i):
                out[i] = {t: slope * v for t, v in forecast[i].items()}
    return out


CORRECTION_KEYS = {"gauges": ("firm", "sector", "firm_agree", "sector_agree"),
                   "gauges_and_weekly_tone": ("firm", "sector", "firm_agree", "sector_agree",
                                              "week_firm", "week_sector", "week_firm_agree", "week_sector_agree")}


def correction_inputs(news, i, t, direction, variant):
    """Sentiment inputs and their agreement with the direction of the raw price forecast."""
    row = news[i][t]
    values = [row["gauge_0"], row["gauge_1"]]
    if variant == "gauges_and_weekly_tone":
        values += [row["abnormal_firm"], row["abnormal_sector"]]
    s = np.sign(direction)
    if variant == "gauges":
        return values + [s * v for v in values]
    return values[:2] + [s * v for v in values[:2]] + values[2:] + [s * v for v in values[2:]]


def sentiment_correction(panel, news, price, raw, years, variant):
    """Price + sentiment correction: a Ridge model of the price forecast's out-of-sample errors on sentiment, refitted
    every January on errors observed before the first decision of the year. Inputs are the firm and
    sector gauges and their agreement with the direction of the price forecast, so sentiment can
    confirm, weaken or overturn the price signal. Returns the corrected forecast and the coefficients."""
    from sklearn.linear_model import RidgeCV
    combined, coefficients = {}, {}
    for year in years:
        idx = panel.year_indices(year)
        weeks = [i for i in panel.training_weeks(panel.decisions[idx[0]]["cutoff"])
                 if price.get(i) and panel.decisions[i]["week"] >= PARAMS["correction_start"]]
        x, y = [], []
        for i in weeks:
            for t, target in panel.relative[i].items():
                if t in price[i]:
                    x.append(correction_inputs(news, i, t, raw[i][t], variant))
                    y.append(target - price[i][t])
        model = RidgeCV(alphas=PARAMS["ridge_alphas"]).fit(np.array(x), np.array(y)) if len(weeks) >= 26 else None
        if model is not None:
            coefficients[year] = {"coef": dict(zip(CORRECTION_KEYS[variant], model.coef_.tolist())), "alpha": float(model.alpha_),
                                  "weeks": len(weeks)}
        for i in idx:
            if not price.get(i):
                continue
            names = list(price[i])
            adj = (model.predict(np.array([correction_inputs(news, i, t, raw[i][t], variant) for t in names]))
                   if model is not None else np.zeros(len(names)))
            combined[i] = {t: price[i][t] + float(a) for t, a in zip(names, adj)}
    return combined, coefficients
