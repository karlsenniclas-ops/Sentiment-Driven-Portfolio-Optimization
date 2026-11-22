"""Manual check of FinBERT: a blind sample of accepted test-period headlines labelled by the author.
Added after examiner feedback asking for a manual reading of FinBERT's scores. Uses the frozen data
preparation and changes no frozen choice.
  python diagnostics/manual_check.py sample   draws the sample into data/manual_check/headlines_to_label.xlsx
  python diagnostics/manual_check.py score    compares the author's labels with FinBERT and writes
                                              results/manual_check.json, results/tables/manual_check.tex
                                              and macros in extra_numbers.tex
The sheet does not show FinBERT's label, so the author labels blind."""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.config import DATA, OUT, TABLE  # noqa: E402
from pipeline.data import load_universe  # noqa: E402
from pipeline.news import attach_scores, load_headlines  # noqa: E402
from project import unpack_scores  # noqa: E402
from diagnostics.report_figures import write_macros  # noqa: E402

LABELS = ("positive", "neutral", "negative")
PER_CLASS = {"positive": 35, "neutral": 30, "negative": 35}   # stratified by FinBERT's label
SEED = 2026
SHEET = DATA / "manual_check" / "headlines_to_label.xlsx"
INSTRUCTIONS = (
    "How to label",
    "1. For each headline, choose in 'Your label' whether it is Positive, Neutral or Negative news for the named "
    "company's shareholders.",
    "2. Positive: likely good for the company's share price or outlook (contract won, earnings beat, upgrade, "
    "approval, record high).",
    "3. Negative: likely bad (lawsuit, fine, recall, loss, downgrade, scandal, strike, job cuts).",
    "4. Neutral: factual, mixed, or not really about the company's prospects (events, product features, "
    "a passing mention in an unrelated story).",
    "5. Judge from the headline alone, as on a first reading. Do not look anything up or use what the share "
    "price did afterwards.",
    "6. Spend about 20-30 seconds per headline. If torn between two labels, pick the more likely one and put "
    "an x under 'Unsure'.",
    "7. Do not open results/manual_check_key.json before you are done: it holds FinBERT's labels.",
    "8. Label all rows, save the file under the same name and run: python diagnostics/manual_check.py score",
)


def finbert_label(r):
    neutral = 1 - r["positive"] - r["negative"]
    return max(zip((r["positive"], neutral, r["negative"]), LABELS))[1]


def test_headlines():
    universe = load_universe()
    names = {f["ticker"]: f["name"] for f in universe}
    records = attach_scores(load_headlines(universe, {}), {})
    records = [r for r in records if "2015-01-01" <= r["date"] <= "2025-12-31"]
    for r in records:
        r["company"], r["finbert"] = names[r["ticker"]], finbert_label(r)
    return records


def sample():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation
    rng = random.Random(SEED)
    pool = {k: [] for k in LABELS}
    seen = set()
    for r in sorted(test_headlines(), key=lambda r: (r["ticker"], r["date"], r["title"])):
        if r["title"].casefold() not in seen:
            seen.add(r["title"].casefold())
            pool[r["finbert"]].append(r)
    rows = [r for k in LABELS for r in rng.sample(pool[k], PER_CLASS[k])]
    rng.shuffle(rows)
    wb = Workbook()
    ws = wb.active
    ws.title = "Label"
    header = ["ID", "Date", "Company", "Headline", "Your label", "Unsure (x)", "Note"]
    ws.append(header)
    for i, r in enumerate(rows, 1):
        ws.append([i, r["date"], r["company"], r["title"], None, None, None])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for col, width in zip("ABCDEFG", (6, 12, 22, 90, 14, 11, 30)):
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2, min_col=4, max_col=4):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    rule = DataValidation(type="list", formula1='"Positive,Neutral,Negative"', allow_blank=True)
    ws.add_data_validation(rule)
    rule.add(f"E2:E{len(rows) + 1}")
    ws.freeze_panes = "A2"
    guide = wb.create_sheet("Instructions")
    for line in INSTRUCTIONS:
        guide.append([line])
    guide.column_dimensions["A"].width = 120
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    wb.save(SHEET)
    key = [{"id": i, "ticker": r["ticker"], "date": r["date"], "title": r["title"],
            "finbert": r["finbert"], "tone": round(r["tone"], 4)} for i, r in enumerate(rows, 1)]
    (OUT / "manual_check_key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} headlines written to {SHEET}")


def score():
    from openpyxl import load_workbook
    key = {k["id"]: k for k in json.loads((OUT / "manual_check_key.json").read_text(encoding="utf-8"))}
    ws = load_workbook(SHEET, read_only=True).active
    pairs, tones, unsure = [], [], 0
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        mine = (row[4] or "").strip().lower()
        if mine not in LABELS:
            raise ValueError(f"headline {row[0]} has no valid label: {row[4]!r}")
        unsure += bool(row[5])
        pairs.append((mine, key[int(row[0])]["finbert"]))
        tones.append((LABELS.index(mine), key[int(row[0])]["tone"]))
    n = len(pairs)
    agree = sum(a == b for a, b in pairs)
    polar = [(a, b) for a, b in pairs if a != "neutral" and b != "neutral"]
    flipped = sum(a != b for a, b in polar)
    matrix = {a: Counter(b for x, b in pairs if x == a) for a in LABELS}
    precision = {b: (sum(1 for x, y in pairs if y == b and x == b) / max(1, sum(1 for _, y in pairs if y == b)))
                 for b in LABELS}
    # rank correlation between the author's label (+1, 0, -1) and FinBERT's tone s
    rho = pd.Series([1 - i for i, _ in tones]).corr(pd.Series([t for _, t in tones]), method="spearman")
    # the sample is stratified by FinBERT's label; reweight to the label shares of all test headlines
    seen, share = set(), Counter()
    for r in test_headlines():
        if r["title"].casefold() not in seen:
            seen.add(r["title"].casefold())
            share[r["finbert"]] += 1
    total = sum(share.values())
    agree_pop = sum(share[b] / total * precision[b] for b in LABELS)
    out = {"n": n, "agreement": agree / n, "agreement_reweighted": agree_pop, "spearman": rho,
           "finbert_label_shares": {b: share[b] / total for b in LABELS},
           "unsure": unsure, "opposite_sign": flipped,
           "polar_pairs": len(polar), "precision_by_finbert_label": precision,
           "matrix_rows_mine_cols_finbert": {a: dict(matrix[a]) for a in LABELS}}
    (OUT / "manual_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Author's label & \multicolumn{3}{c}{FinBERT label} & $n$\\", r"\cmidrule(lr){2-4}",
             r" & Positive & Neutral & Negative & \\", r"\midrule"]
    for a in LABELS:
        cells = " & ".join(str(matrix[a].get(b, 0)) for b in LABELS)
        lines.append(f"{a.capitalize()} & {cells} & {sum(matrix[a].values())}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLE / "manual_check.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_macros({"ManualN": str(n), "ManualAgree": f"{100 * agree / n:.0f}",
                  "ManualOpposite": str(flipped), "ManualPolar": str(len(polar)), "ManualUnsure": str(unsure),
                  "ManualAgreePop": f"{100 * agree_pop:.0f}", "ManualRho": f"{rho:.2f}"})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if "sample" in sys.argv:
        unpack_scores()
        sample()
    elif "score" in sys.argv:
        unpack_scores()
        score()
    else:
        print(__doc__)
