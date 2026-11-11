"""Post-test checks of the headline data, added after supervisor feedback on headline-only sentiment.
They use the frozen pipeline and change no frozen choice.
1. FinBERT on headlines: accuracy on SEntFiN 1.0 (Sinha et al., 2022), 10,753 financial news headlines
   annotated by people for the sentiment towards each named company; FinBERT was not trained on them.
   Headlines naming one company are used, scored exactly as data_collection/score_finbert.py does.
2. Repeated headlines: the share of accepted headlines that repeat an earlier one for the same firm within
   three days in other words (at least half, or at least 70%, of their words shared), and the sentiment-driven
   and price + sentiment portfolios rebuilt without them.
Writes results/news_quality.json, results/tables/finbert_headlines.tex and macros in extra_numbers.tex.
Usage: python diagnostics/news_quality.py   (downloads SEntFiN on first use; about 15 minutes)"""
import csv
import datetime as dt
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import forecast as fc  # noqa: E402
from pipeline.config import DATA, OUT, PARAMS, TABLE  # noqa: E402
from pipeline.evaluate import paired_bootstrap, scalar_metrics  # noqa: E402
from pipeline.portfolio import simulate  # noqa: E402
from pipeline.run import (build_context, cached, finish_frame, market_inputs, news_inputs, signals,  # noqa: E402
                          strategy_specs, test_indices)
from pipeline.sentiment import news_panel  # noqa: E402
from diagnostics.report_figures import write_macros  # noqa: E402
from project import unpack_scores  # noqa: E402

SENTFIN_URL = ("https://raw.githubusercontent.com/pyRis/SEntFiN/eba43610fe3cb57a3e5773cd6c037da1f8992ad5/"
               "SEntFiN.csv")
LABELS = ("positive", "neutral", "negative")
DUPLICATE_DAYS, OVERLAPS = 3, (0.5, 0.7)   # share of words two headlines must share to count as a repeat


# --------------------------------------------------------------------------- 1. FinBERT on headlines
def finbert_on_headlines():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = DATA / "sentfin" / "SEntFiN.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SENTFIN_URL, path)
    rows = []
    with path.open(encoding="utf-8") as stream:
        for r in csv.DictReader(stream):
            # Annotations look like {'Entity': 'label', ...}; some entity names contain unescaped quotes.
            labels = re.findall(r":\s*'(positive|negative|neutral)'", r["Decisions"])
            if len(labels) == 1:
                rows.append((r["Title"], labels[0]))
    torch.manual_seed(0)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(PARAMS["finbert_model"])
    model = AutoModelForSequenceClassification.from_pretrained(PARAMS["finbert_model"]).eval()
    ids = {v: k for k, v in model.config.id2label.items()}
    predicted, tone = [], []
    with torch.inference_mode():
        for i in range(0, len(rows), 64):
            batch = [t for t, _ in rows[i:i + 64]]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
            for p in torch.softmax(model(**encoded).logits, dim=-1).tolist():
                predicted.append(max(LABELS, key=lambda k: p[ids[k]]))
                tone.append(p[ids["positive"]] - p[ids["negative"]])
    truth = [label for _, label in rows]
    tone = np.array(tone)
    confusion = {a: {b: sum(1 for x, y in zip(truth, predicted) if x == a and y == b) for b in LABELS} for a in LABELS}
    recall = {a: confusion[a][a] / sum(confusion[a].values()) for a in LABELS}
    precision = {b: confusion[b][b] / max(1, sum(confusion[a][b] for a in LABELS)) for b in LABELS}
    f1 = {k: 2 * precision[k] * recall[k] / (precision[k] + recall[k]) for k in LABELS}
    polar = [(x, y) for x, y in zip(truth, predicted) if x != "neutral" and y != "neutral"]
    strong = np.abs(tone) > 0.5
    return {"headlines": len(rows), "accuracy": float(np.mean([x == y for x, y in zip(truth, predicted)])),
            "macro_f1": float(np.mean(list(f1.values()))), "recall": recall, "precision": precision,
            "confusion": confusion,
            "opposite_sign_share": float(np.mean([x != y for x, y in polar])),
            "strong_tone_correct": float(np.mean([("positive" if s > 0 else "negative") == x
                                                  for s, x, k in zip(tone, truth, strong) if k])),
            "strong_tone_share": float(strong.mean())}


# --------------------------------------------------------------------------- 2. repeated headlines
def words(title):
    return {w for w in re.findall(r"[a-z0-9]+", title.lower()) if len(w) > 1}


def drop_repeats(records, overlap):
    """Keep the first of headlines for the same firm within three days sharing at least half of their words."""
    by_firm = defaultdict(list)
    for r in records:
        by_firm[r["ticker"]].append(r)
    kept = []
    for rows in by_firm.values():
        window = []
        for r in sorted(rows, key=lambda x: x["date"]):
            day, w = dt.date.fromisoformat(r["date"]), words(r["title"])
            window = [(d, ww) for d, ww in window if (day - d).days <= DUPLICATE_DAYS]
            if any(w and len(w & ww) / len(w | ww) >= overlap for _, ww in window):
                continue
            window.append((day, w))
            kept.append(r)
    return kept


def repeated_headlines():
    frozen = json.loads((OUT / "frozen.json").read_text())
    universe, clock, decisions, prices = market_inputs()
    panel = fc.Panel(prices)
    records, news = news_inputs(universe, clock, decisions, {})
    price_fc = cached(f"forecast_{frozen['price_model']}",
                      lambda: fc.price_forecasts(panel, frozen["price_model"], "2008", "2025"))
    price_cal = fc.calibrate(panel, price_fc, [str(y) for y in range(2009, 2026)])
    args = (frozen["sector_weight"], frozen["direction"], frozen["correction_variant"])
    ctx = cached("context", lambda: build_context(prices, panel))
    def run(n):
        g, active, combined, _ = signals(panel, n, price_fc, price_cal, *args)
        specs = strategy_specs(g, active, combined, price_fc)
        frame, _ = simulate(ctx, {k: specs[k] for k in ("Sentiment", "Combined")})
        return finish_frame(frame)
    base = run(news)
    stored = json.loads((OUT / "analysis.json").read_text())["metrics"]
    rf = base.risk_free.to_numpy()
    out = {"headlines": len(records), "reproduction_gap": max(
        abs(scalar_metrics(base[n].to_numpy(), rf)[k] - stored[n][k]) for n in ("Sentiment", "Combined") for k in ("cagr", "sharpe"))}
    key = f"gauge_{frozen['sector_weight']:g}"
    idx = test_indices(decisions)
    for overlap in OVERLAPS:
        kept = drop_repeats(records, overlap)
        news_kept = news_panel(kept, universe, clock, decisions)
        frame = run(news_kept)
        pairs = np.array([(news[i][t][key], news_kept[i][t][key]) for i in idx for t in news[i]])
        res = {"repeats": len(records) - len(kept), "repeat_share": 1 - len(kept) / len(records),
               "gauge_correlation": float(np.corrcoef(pairs.T)[0, 1]), "strategies": {}}
        for n in ("Sentiment", "Combined"):
            m = scalar_metrics(frame[n].to_numpy(), rf)
            boot = paired_bootstrap(frame[n].to_numpy(), base[n].to_numpy(), rf)
            res["strategies"][n] = {**m, "sharpe_vs_base_ci": boot["sharpe"]["ci"]}
        out[f"{overlap:g}"] = res
        print(overlap, json.dumps(res), flush=True)
    return out


# --------------------------------------------------------------------------- outputs
def write_outputs(r):
    fb, rep = r["finbert"], r["repeats"]
    names = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{FinBERT label (\% of row)} & \\",
             r"Annotated label & Positive & Neutral & Negative & Headlines \\", r"\midrule"]
    for a in LABELS:
        total = sum(fb["confusion"][a].values())
        cells = " & ".join(f"{100 * fb['confusion'][a][b] / total:.0f}" for b in LABELS)
        lines.append(f"{names[a]} & {cells} & {total:,} \\\\")
    lines += [r"\midrule", f"\\multicolumn{{5}}{{l}}{{Accuracy {100 * fb['accuracy']:.0f}\\%, macro F1 "
                           f"{fb['macro_f1']:.2f}; opposite sign in {100 * fb['opposite_sign_share']:.1f}\\% "
                           r"of polar predictions}\\", r"\bottomrule", r"\end{tabular}"]
    (TABLE / "finbert_headlines.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pct = lambda x: f"{100 * x:.0f}\\%"   # noqa: E731
    main_rep, strict = rep[f"{OVERLAPS[0]:g}"], rep[f"{OVERLAPS[1]:g}"]
    s, c = main_rep["strategies"]["Sentiment"], main_rep["strategies"]["Combined"]
    s7, c7 = strict["strategies"]["Sentiment"], strict["strategies"]["Combined"]
    macros = {"FbHeadlines": f"{fb['headlines']:,}", "FbAccuracy": pct(fb["accuracy"]), "FbMacroF": f"{fb['macro_f1']:.2f}",
              "FbOpposite": f"{100 * fb['opposite_sign_share']:.1f}\\%", "FbRecallNeutral": pct(fb["recall"]["neutral"]),
              "FbRecallPositive": pct(fb["recall"]["positive"]), "FbRecallNegative": pct(fb["recall"]["negative"]),
              "FbStrongCorrect": pct(fb["strong_tone_correct"]),
              "RepShare": f"{100 * main_rep['repeat_share']:.1f}\\%", "RepCount": f"{main_rep['repeats']:,}",
              "RepGaugeCorr": f"{main_rep['gauge_correlation']:.2f}",
              "RepShareStrict": f"{100 * strict['repeat_share']:.1f}\\%",
              "RepSentSharpeStrict": f"{s7['sharpe']:.2f}", "RepSentCAGRStrict": f"{100 * s7['cagr']:.2f}\\%",
              "RepCombSharpeStrict": f"{c7['sharpe']:.2f}", "RepCombCAGRStrict": f"{100 * c7['cagr']:.2f}\\%",
              "RepSentCI": f"{s['sharpe_vs_base_ci'][0]:+.2f} to {s['sharpe_vs_base_ci'][1]:+.2f}",
              "RepSentSharpe": f"{s['sharpe']:.2f}", "RepSentCAGR": f"{100 * s['cagr']:.2f}\\%",
              "RepCombSharpe": f"{c['sharpe']:.2f}", "RepCombCAGR": f"{100 * c['cagr']:.2f}\\%"}
    write_macros(macros)
    print(json.dumps(r, indent=1, default=str))


def main():
    path = OUT / "news_quality.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    if "--repeats-only" not in sys.argv:
        result["finbert"] = finbert_on_headlines()
        print(json.dumps(result["finbert"], indent=1), flush=True)
    if "--finbert-only" not in sys.argv:
        result["repeats"] = repeated_headlines()
    path.write_text(json.dumps(result, indent=1, default=str))
    write_outputs(result)


if __name__ == "__main__":
    if "--tables-only" in sys.argv:
        write_outputs(json.loads((OUT / "news_quality.json").read_text()))
    else:
        unpack_scores()
        main()
