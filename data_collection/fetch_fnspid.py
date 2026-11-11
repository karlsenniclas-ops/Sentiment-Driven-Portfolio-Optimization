"""Fetch time-stamped headlines for the study's US firms from FNSPID (Dong, Fan and Peng, 2024;
https://huggingface.co/datasets/Zihan1004/FNSPID, CC BY 4.0) for the post-test check in
diagnostics/fnspid_check.py. Streams the merged news file (23 GB, about 10 minutes on a fast
connection), keeps headlines of the US universe from October 2010 on and writes data/fnspid/*.csv.gz.
Only the headline, its publication time, ticker, publisher and URL are kept.
Usage: python data_collection/fetch_fnspid.py"""
import csv
import gzip
import io
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/"
FILES = {"fnspid": "nasdaq_exteral_data.csv"}   # Nasdaq.com news merged with the external (mainly Benzinga) records
ALIAS = {"FB": "META", "UTX": "RTX", "GOOG": "GOOGL", "BRK.B": "BRK-B", "BRK/B": "BRK-B", "BRKB": "BRK-B"}


def main():
    csv.field_size_limit(10 ** 9)
    with (ROOT / "data" / "universe.csv").open(encoding="utf-8") as stream:
        keep = {r["ticker"] for r in csv.DictReader(stream) if r["region"] == "United States" and r["included"] == "1"}
    out_dir = ROOT / "data" / "fnspid"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, file in FILES.items():
        with urllib.request.urlopen(BASE + file) as response, \
                gzip.open(out_dir / f"{name}.csv.gz", "wt", encoding="utf-8", newline="") as target:
            reader = csv.reader(io.TextIOWrapper(response, encoding="utf-8", errors="ignore", newline=""))
            header = next(reader)
            ix = {k: header.index(k) for k in ("Date", "Article_title", "Stock_symbol", "Publisher", "Url")}
            writer = csv.writer(target)
            writer.writerow(["ticker", "timestamp", "title", "publisher", "url"])
            rows = kept = 0
            for row in reader:
                rows += 1
                if len(row) < len(header) - 1:
                    continue
                ticker = row[ix["Stock_symbol"]].strip().upper()
                ticker = ALIAS.get(ticker, ticker)
                if ticker in keep and row[ix["Date"]][:7] >= "2010-10":
                    writer.writerow([ticker, row[ix["Date"]], row[ix["Article_title"]], row[ix["Publisher"]], row[ix["Url"]]])
                    kept += 1
        print(f"{file}: {rows:,} rows, kept {kept:,}", file=sys.stderr)


if __name__ == "__main__":
    main()
