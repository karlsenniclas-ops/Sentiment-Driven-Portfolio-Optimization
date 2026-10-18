"""Download daily prices (adjusted and unadjusted close, volume) from Yahoo Finance."""
import sys, time
from pathlib import Path
import pandas as pd
import yfinance as yf
sys.path.insert(0, str(Path(__file__).parent))
import universe_spec as u

OUT = Path(__file__).resolve().parents[1] / "data" / "raw_yahoo"
OUT.mkdir(parents=True, exist_ok=True)
EXTRA = ["URTH", "EURUSD=X", "GBPUSD=X"]
tickers = [r[0] for r in u.US + u.EU] + EXTRA + sys.argv[1:]
for t in tickers:
    path = OUT / f"{t.replace('=', '_')}.csv"
    if path.exists():
        continue
    for attempt in range(4):
        try:
            d = yf.download(t, start="2004-11-01", end="2026-01-10", progress=False, auto_adjust=False,
                            actions=False, threads=False)
            break
        except Exception as e:
            print(t, "error", e); time.sleep(10)
    if d is None or d.empty:
        print(t, "EMPTY"); continue
    d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    d[["Adj Close", "Close", "Volume"]].to_csv(path)
    print(t, len(d), d.index[0].date(), d.index[-1].date(), flush=True)
    time.sleep(0.5)
