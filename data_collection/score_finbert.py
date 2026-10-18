"""Score accepted headlines with FinBERT (ProsusAI/finbert). Incremental: titles already in
data/finbert_scores.jsonl are skipped. The scores are also written compressed to data/finbert_scores.jsonl.gz,
the copy kept in the repository. Usage: python data_collection/score_finbert.py [--complete-only]"""
import argparse
import csv
import gzip
import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.config import DATA, PARAMS  # noqa: E402
from pipeline.news import load_headlines, score_key  # noqa: E402

WINDOWS = 183  # Oct 2010 - Dec 2025


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--complete-only", action="store_true", help="score only firms whose collection is complete")
    args = parser.parse_args()
    universe = [r for r in csv.DictReader((DATA / "universe.csv").open(encoding="utf-8")) if r["included"] == "1"]
    if args.complete_only:
        def complete(t):
            path = DATA / "headlines" / f"{t}.jsonl"
            return path.exists() and sum(1 for _ in path.open(encoding="utf-8")) >= WINDOWS
        universe = [r for r in universe if complete(r["ticker"])]
    titles = sorted({r["title"] for r in load_headlines(universe, {})})
    out = DATA / "finbert_scores.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["key"] for line in out.open(encoding="utf-8")}
    todo = sorted((t for t in titles if score_key(t) not in done), key=len)   # similar lengths batch faster
    print(f"{len(titles):,} titles, {len(todo):,} to score", flush=True)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(PARAMS["finbert_model"])
    model = AutoModelForSequenceClassification.from_pretrained(PARAMS["finbert_model"]).eval()
    labels = {v: k for k, v in model.config.id2label.items()}
    with out.open("a", encoding="utf-8") as stream, torch.inference_mode():
        for i in range(0, len(todo), 64):
            batch = todo[i:i + 64]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
            probs = torch.softmax(model(**encoded).logits, dim=-1).tolist()
            for title, p in zip(batch, probs):
                stream.write(json.dumps({"key": score_key(title), "positive": p[labels["positive"]],
                                         "negative": p[labels["negative"]], "neutral": p[labels["neutral"]]}) + "\n")
            if i % 6400 == 0:
                print(f"  {i + len(batch):,} / {len(todo):,}", flush=True)
    with out.open("rb") as source, gzip.GzipFile(out.with_suffix(".jsonl.gz"), "wb", 9, mtime=0) as target:
        shutil.copyfileobj(source, target)


if __name__ == "__main__":
    main()
