"""Collect company headlines from Google News RSS, one query per firm and calendar month
(Google returns at most 100 items per query).

Resumable: windows already stored are skipped. Run on several machines with --reverse to
split the work; files are merged afterwards (union by window). Usage:
    python data_collection/collect_news.py [--reverse] [--only TICKER ...]
"""
import argparse, datetime as dt, json, random, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import universe_spec as u

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "headlines"
OUT.mkdir(parents=True, exist_ok=True)
FIRST, LAST = dt.date(2010, 10, 1), dt.date(2025, 12, 31)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


def windows():
    month = FIRST
    while month <= LAST:
        nxt = (month.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        yield month, nxt - dt.timedelta(days=1)
        month = nxt


def fetch(query, start, end):
    q = f"{query} after:{start - dt.timedelta(days=1)} before:{end + dt.timedelta(days=1)}"
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=30) as response:
        root = ET.fromstring(response.read())
    items = []
    for item in root.iter("item"):
        source = item.find("source")
        items.append({"title": item.findtext("title"), "pub": item.findtext("pubDate"),
                      "src": source.text if source is not None else None})
    return q, items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--after", default="0000-00", help="only months from YYYY-MM")
    parser.add_argument("--before", default="9999-99", help="only months before YYYY-MM")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    args = parser.parse_args()
    excluded = {"APC", "DOW", "EMC", "FOXA", "MON", "RTN", "TWX", "WBA", "URW.PA"}  # no usable prices
    firms = [(r[0], r[3]) for r in u.US + u.EU if r[0] not in excluded]
    firms = [f for i, f in enumerate(firms) if i % args.of == args.shard]
    if args.only:
        firms = [f for f in firms if f[0] in args.only]
    if args.reverse:
        firms = firms[::-1]
    pause, done_requests = 30, 0
    for ticker, query in firms:
        path = OUT / f"{ticker}.jsonl"
        have = set()
        if path.exists():
            for line in path.open(encoding="utf-8"):
                try:
                    have.add(json.loads(line)["from"])
                except Exception:
                    pass
        todo = [w for w in windows() if str(w[0]) not in have and args.after <= str(w[0])[:7] < args.before]
        if not todo:
            continue
        print(f"{ticker}: {len(todo)} windows to fetch", flush=True)
        for start, end in todo:
            while True:
                try:
                    q, items = fetch(query, start, end)
                    pause = 30
                    break
                except Exception as error:
                    print(f"  {ticker} {start}: {error}; sleeping {pause}s", flush=True)
                    time.sleep(pause)
                    pause = min(pause * 2, 1200)
            record = {"ticker": ticker, "from": str(start), "to": str(end), "query": q, "n": len(items),
                      "retrieved": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "items": items}
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            done_requests += 1
            time.sleep(random.uniform(1.0, 2.0))
    print("finished", done_requests, flush=True)


if __name__ == "__main__":
    main()
