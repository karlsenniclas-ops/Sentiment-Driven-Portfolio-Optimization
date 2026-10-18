"""Build data/universe.csv and data/prices/*.csv from the raw downloads.

Price sources: Yahoo Finance (data/raw_yahoo). Bank of New York Mellon is taken under its
current Yahoo symbol (BNY).
Firms without a price history covering their listed life in the sample are excluded and
listed with the reason (all were acquired or merged 2016-2020, or taken private in 2025).
"""
import csv, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import universe_spec as u

ROOT = Path(__file__).resolve().parents[1]
RAW, PRICES, FX = ROOT / "data" / "raw_yahoo", ROOT / "data" / "prices", ROOT / "data" / "fx"
PRICES.mkdir(parents=True, exist_ok=True)
FX.mkdir(parents=True, exist_ok=True)
EXCHANGE = {".PA": "XPAR", ".DE": "XETR", ".AS": "XAMS", ".BR": "XBRU", ".MC": "XMAD", ".MI": "XMIL",
            ".HE": "XHEL", ".L": "XLON"}
COUNTRY = {".PA": "France", ".DE": "Germany", ".AS": "Netherlands", ".BR": "Belgium", ".MC": "Spain",
           ".MI": "Italy", ".HE": "Finland", ".L": "United Kingdom"}
ALTERNATIVE = {"BK": ("BNY", "USD", "XNYS")}
# Vivendi split into four listed companies in December 2024; the adjusted series does not
# carry the spin-offs, so the stock leaves the universe at its last price before the split.
TRUNCATE = {"VIV.PA": "2024-12-06"}
# Yahoo's adjusted Inditex series has two permanent one-day jumps (+42%, +46%) with no matching
# corporate event; earlier prices are rescaled so that those two days carry a zero return.
LEVEL_ERRORS = {"ITX.MC": ["2005-09-27", "2006-09-26"]}
EXCLUDED = {"APC": "acquired by Occidental (2019); no price history available",
            "DOW": "merged into DowDuPont (2017); no price history available",
            "EMC": "acquired by Dell (2016); no price history available",
            "FOXA": "acquired by Disney (2019); no price history available",
            "MON": "acquired by Bayer (2018); no price history available",
            "RTN": "merged with United Technologies (2020); no price history available",
            "TWX": "acquired by AT&T (2018); no price history available",
            "WBA": "taken private (2025); no price history available",
            "URW.PA": "Euronext history not available; the U.S. OTC line is stale (76% zero daily returns)"}

rows = []
for region, spec in (("United States", u.US), ("Euro area", u.EU)):
    for ticker, name, sector, query, aliases, ambiguous in spec:
        suffix = "." + ticker.split(".")[-1] if "." in ticker else ""
        currency = "GBp" if suffix == ".L" else ("EUR" if suffix else "USD")
        exchange, country, source = EXCHANGE.get(suffix, "XNYS"), COUNTRY.get(suffix, "United States"), ticker
        if ticker in ALTERNATIVE:
            source, currency, exchange = ALTERNATIVE[ticker]
        row = {"ticker": ticker, "name": name, "region": region, "country": country, "sector": sector,
               "currency": currency, "exchange": exchange, "price_symbol": source, "query": query,
               "aliases": aliases, "ambiguous": ambiguous, "included": 0, "note": EXCLUDED.get(ticker, "")}
        path = RAW / f"{source}.csv"
        if ticker not in EXCLUDED and path.exists():
            frame = pd.read_csv(path, index_col=0, parse_dates=True).dropna(subset=["Adj Close"])
            frame = frame[frame["Adj Close"] > 0]
            for day in LEVEL_ERRORS.get(ticker, []):
                position = frame.index.get_loc(pd.Timestamp(day))
                ratio = frame["Adj Close"].iloc[position] / frame["Adj Close"].iloc[position - 1]
                frame.iloc[:position, frame.columns.get_loc("Adj Close")] *= ratio
            if ticker in TRUNCATE:
                frame = frame[frame.index <= TRUNCATE[ticker]]
                row["note"] = f"series ends {TRUNCATE[ticker]} (break-up of the company)"
            frame[["Adj Close"]].to_csv(PRICES / f"{ticker}.csv", date_format="%Y-%m-%d")
            frame[["Close", "Volume"]].to_csv(PRICES / f"{ticker}_volume.csv", date_format="%Y-%m-%d")
            row.update(included=1, first_price=frame.index[0].date(), last_price=frame.index[-1].date())
        rows.append(row)
for pair in ("EURUSD", "GBPUSD"):
    frame = pd.read_csv(RAW / f"{pair}_X.csv", index_col=0, parse_dates=True)["Close"].dropna()
    frame.rename("Close").to_csv(FX / f"{pair}.csv", date_format="%Y-%m-%d")
frame = pd.read_csv(RAW / "URTH.csv", index_col=0, parse_dates=True).dropna(subset=["Adj Close"])
frame[["Adj Close"]].to_csv(PRICES / "URTH.csv", date_format="%Y-%m-%d")
fields = ["ticker", "name", "region", "country", "sector", "currency", "exchange", "price_symbol", "query",
          "aliases", "ambiguous", "included", "first_price", "last_price", "note"]
with (ROOT / "data" / "universe.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(sum(r["included"] for r in rows), "of", len(rows), "firms included")
