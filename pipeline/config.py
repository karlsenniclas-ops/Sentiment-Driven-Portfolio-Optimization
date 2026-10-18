"""All fixed settings. Settings marked 'validation' are chosen on 2012-2014 by
`python project.py validate-price` and `validate-sentiment`, written to results/selection.json and then frozen."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "results"
FIG, TABLE = OUT / "figures", OUT / "tables"

PARAMS = {
    # Calendar: decision every Thursday 22:00 UTC (after the U.S. close), trade at each
    # stock's Friday close, hold one week.
    "first_decision_week": "2005-01-03",
    "last_decision_week": "2025-12-22",
    "calendar_start": "2004-10-01",
    "calendar_end": "2026-02-28",
    "observation_through_date": "2025-12-31",
    "cutoff_weekday": 3,
    "cutoff_hour_utc": 22,
    "price_publication_lag_minutes": 15,
    "max_price_age_days": 14,
    # Periods
    "news_start": "2010-10-01",
    "train_start": "2006-01-02",      # price models: features need one year of history
    "validation_start": "2012-01-02",
    "validation_end": "2014-12-29",
    "evaluation_start": "2015-01-05",
    "evaluation_end": "2025-12-22",
    # News and FinBERT
    "finbert_model": "ProsusAI/finbert",
    "roundup_min_firms": 3,          # a headline naming three or more firms is a market roundup
    "shrinkage_headlines": 5,        # tone shrunk by n / (n + k)
    # Sentiment gauge
    "baseline_weeks": 52,            # abnormal tone against the firm's own previous year
    "baseline_min_news_weeks": 10,
    "gauge_half_life_weeks": 2,
    "entry_band": 1.5,               # gauge in standard-deviation units
    "exit_band": 0.5,
    "sector_weight_grid": [0.0, 0.5],    # validation: firm-only or firm + sector sentiment
    # Price forecasting candidates, refitted every January on all earlier data
    "price_models": ["Ridge", "Factor", "ARIMA", "RandomForest", "GBM", "LSTM"],
    "ridge_alphas": [0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6],
    "arima_window_weeks": 520,
    "arima_orders": [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 0), (0, 0, 2), (2, 0, 1), (1, 0, 2), (2, 0, 2)],
    "random_forest": {"n_estimators": 300, "min_samples_leaf": 200, "max_features": 0.5, "random_state": 310},
    "gbm": {"max_iter": 300, "learning_rate": 0.03, "max_leaf_nodes": 15, "min_samples_leaf": 200,
            "l2_regularization": 1.0, "random_state": 310},
    "lstm": {"sequence_weeks": 8, "hidden": 32, "epochs": 40, "patience": 5, "batch": 512, "lr": 1e-3,
             "dropout": 0.2, "seed": 310},
    # Price + sentiment: Ridge sentiment correction trained on the price model's out-of-sample errors
    "correction_start": "2011-01-03",
    "correction_variants": ["gauges", "gauges_and_weekly_tone"],   # validation chooses
    # Portfolio construction
    "risk_aversion": 2.5,
    "bl_tau": 0.05,
    "covariance_weeks": 52,
    "ic_min_weeks": 52,
    "skill_window_weeks": 156,       # skill and calibration use the last three years (signals are non-stationary)
    "holding_weeks": 4,
    "position_cap": 0.05,            # 5% per issuer satisfies the UCITS 5/10/40 rule
    "base_cost_bps": 10.0,
    "cost_grid_bps": [0.0, 5.0, 10.0, 20.0],
    # Evaluation
    "bootstrap_seed": 310,
    "bootstrap_resamples": 2000,
    "bootstrap_block_weeks": 13,
    "hac_lags": 4,
    "decay_horizons": 8,
    "event_window_days": [-10, 40],
    "subperiods": {"2015-2019": ["2015-01-05", "2019-12-30"], "2020-2025": ["2020-01-06", "2025-12-22"]},
}

# Price and trading-activity predictors, ranked across stocks every week.
PRICE_KEYS = ("momentum_12_1", "reversal_4w", "reversal_1w", "volatility_60d", "max_return_20d",
              "beta_52w", "high_52w", "dollar_volume", "illiquidity", "abnormal_volume")
FACTORS = {"momentum_12_1": 1, "reversal_4w": 1, "volatility_60d": -1}

# Strategies: name -> information used for views (None = no views).
STRATEGIES = ["Sentiment", "Combined", "Price", "MinVar", "EW", "Index"]
LABELS = {"Sentiment": "Sentiment-driven", "Combined": "Price + sentiment", "Price": "Price only",
          "MinVar": "Minimum variance", "EW": "Equal weight", "Index": "MSCI World (URTH)"}
