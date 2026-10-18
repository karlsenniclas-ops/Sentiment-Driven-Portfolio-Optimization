"""Sentiment-Driven Portfolio Optimization (IND310 Driftsoptimalisering).

One command reproduces every result, table and figure in the report from the stored data:
    python project.py
It checks that the code and validation choices match the frozen record (results/frozen.json), runs the
frozen 2015-2025 test, builds the post-test diagnostics, compares the key results with the stored ones and,
if pdflatex is installed, compiles report/main.pdf. Runtime: about an hour on a laptop.

The research protocol, stage by stage (each validation stage uses 2012-2014 only):
    python project.py validate-price      choose the price forecasting model by validation IC
    python project.py validate-sentiment  choose the sentiment gauge (firm or firm + sector) and its sign
    python project.py freeze              hash code, settings and selections before the test
    python project.py test                the single walk-forward test, 2015-2025 (refuses a second run)
    python project.py diagnostics         post-test tables and figures for the report
    python project.py report              compile report/main.pdf
    python project.py hashes              SHA-256 of every input file

Data collection and FinBERT scoring are separate scripts in data_collection/.
"""
import datetime as dt
import gzip
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys

import numpy as np

from pipeline import forecast as fc
from pipeline.config import DATA, OUT, PARAMS, ROOT, STRATEGIES
from pipeline.evaluate import hac
from pipeline.run import market_inputs, news_inputs


def validation_indices(decisions):
    return [i for i, d in enumerate(decisions) if PARAMS["validation_start"] <= d["week"] <= PARAMS["validation_end"]]


def summary(ics):
    values = [v for _, v in ics]
    return {**hac(values), "positive_share": float(np.mean(np.array(values) > 0)), "weekly": ics}


# --------------------------------------------------------------------------- validation
def validate_price():
    universe, clock, decisions, prices = market_inputs()
    panel = fc.Panel(prices)
    idx = validation_indices(decisions)
    path = OUT / "selection.json"
    result = json.loads(path.read_text()).get("price_candidates", {}) if path.exists() else {}
    for name in PARAMS["price_models"]:
        if name in result:
            continue
        print(f"Validation: {name}", flush=True)
        forecasts = fc.price_forecasts(panel, name, "2012", "2014")
        result[name] = summary(fc.weekly_ic(panel, forecasts, idx))
        print(f"  mean IC {result[name]['mean']:+.4f} (t {result[name]['t']:.2f})", flush=True)
    chosen = max(result, key=lambda k: result[k]["mean"])
    save_selection({"price_candidates": result, "price_model": chosen})
    print(f"Selected price model: {chosen}")


def validate_sentiment():
    """Sentiment choices on 2012-2014: the gauge (firm only or firm + sector) and its sign for the
    sentiment-driven portfolio, and the inputs of the price + sentiment correction model."""
    universe, clock, decisions, prices = market_inputs()
    panel = fc.Panel(prices)
    audit = {}
    _, news = news_inputs(universe, clock, decisions, audit)
    idx = validation_indices(decisions)
    result = {}
    for w in PARAMS["sector_weight_grid"]:
        gauge = {i: {t: news[i][t][f"gauge_{w:g}"] for t in panel.prices.features[i]} for i in idx}
        result[f"{w:g}"] = summary(fc.weekly_ic(panel, gauge, idx))
        print(f"Gauge sector weight {w:g}: mean IC {result[f'{w:g}']['mean']:+.4f} (t {result[f'{w:g}']['t']:.2f})")
    best = max(result, key=lambda k: abs(result[k]["mean"]))
    direction = 1 if result[best]["mean"] >= 0 else -1
    selection = json.loads((OUT / "selection.json").read_text())
    model = selection["price_model"]
    price = fc.price_forecasts(panel, model, "2008", "2014")
    calibrated = fc.calibrate(panel, price, [str(y) for y in range(2009, 2015)])
    price_ic = dict(fc.weekly_ic(panel, price, idx))
    corrections = {}
    for variant in PARAMS["correction_variants"]:
        combined, coefficients = fc.sentiment_correction(panel, news, calibrated, price, ["2012", "2013", "2014"], variant)
        ics = fc.weekly_ic(panel, combined, idx)
        corrections[variant] = {**summary(ics), "gain": hac([v - price_ic[i] for i, v in ics if i in price_ic]),
                                "coefficients": coefficients}
        print(f"Correction {variant}: IC {corrections[variant]['mean']:+.4f}, gain over price "
              f"{corrections[variant]['gain']['mean']:+.4f} (t {corrections[variant]['gain']['t']:.2f})")
    variant = max(corrections, key=lambda k: corrections[k]["gain"]["mean"])
    save_selection({"gauge_candidates": result, "sector_weight": float(best), "direction": direction,
                    "correction_candidates": corrections, "correction_variant": variant, "validation_audit": audit})
    print(f"Selected sector weight {best}, direction {direction:+d} "
          f"({'continuation' if direction > 0 else 'reversal'}); correction inputs: {variant}")


def save_selection(update):
    path = OUT / "selection.json"
    OUT.mkdir(exist_ok=True)
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(update)
    path.write_text(json.dumps(current, indent=1, default=str))


def fingerprint():
    digest = hashlib.sha256()
    for path in sorted([*(ROOT / "pipeline").glob("*.py"), ROOT / "project.py", OUT / "selection.json"]):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def freeze():
    selection = json.loads((OUT / "selection.json").read_text())
    for key in ("price_model", "sector_weight", "direction", "correction_variant"):
        if key not in selection:
            raise SystemExit(f"selection.json lacks {key}; run the validation stages first")
    record = {"frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "fingerprint": fingerprint(), "price_model": selection["price_model"],
              "sector_weight": selection["sector_weight"], "direction": selection["direction"],
              "correction_variant": selection["correction_variant"]}
    (OUT / "frozen.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))


# --------------------------------------------------------------------------- test
def test(force=False):
    from pipeline.run import run_test
    frozen = json.loads((OUT / "frozen.json").read_text())
    if frozen["fingerprint"] != fingerprint():
        raise SystemExit("Code, settings or selections changed after freezing; the test is not run.")
    lock = OUT / "test_run.json"
    if lock.exists() and not force:
        raise SystemExit("The test has already been run once (results/test_run.json).")
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    run_test(frozen)
    manifest = {"started_utc": started, "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "frozen": frozen, "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in
                             ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "torch", "cvxpy",
                              "matplotlib", "exchange_calendars", "transformers")},
                "input_hashes": hashes()}
    lock.write_text(json.dumps(manifest, indent=1))


def hashes():
    out = {}
    for path in sorted(p for p in DATA.rglob("*") if p.is_file() and "raw_yahoo" not in p.parts
                       and path_is_input(p)):
        out[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def path_is_input(path):
    return path.name != "finbert_scores.jsonl"     # the compressed copy is the stored input


# --------------------------------------------------------------------------- one-command reproduction
def unpack_scores():
    """The repository stores the FinBERT scores compressed; the pipeline reads the plain file."""
    plain, packed = DATA / "finbert_scores.jsonl", DATA / "finbert_scores.jsonl.gz"
    if not plain.exists() and packed.exists():
        print("Unpacking data/finbert_scores.jsonl.gz", flush=True)
        with gzip.open(packed, "rb") as source, plain.open("wb") as target:
            shutil.copyfileobj(source, target)


def diagnostics():
    from diagnostics import decision_table, report_figures
    report_figures.robustness_main()
    report_figures.uncertainty_main()
    report_figures.performance_main()
    report_figures.universe_main()
    report_figures.allocation_mechanism()
    report_figures.decision_sides()
    report_figures.tone()
    decision_table.main()


def report():
    folder = ROOT / "report"
    if not (shutil.which("pdflatex") and shutil.which("bibtex")):
        print("pdflatex/bibtex not found: compile report/main.tex yourself (pdflatex, bibtex, pdflatex x2).")
        return
    for command in (["pdflatex"], ["bibtex"], ["pdflatex"], ["pdflatex"]):
        target = "main" if command[0] == "bibtex" else "main.tex"
        subprocess.run([*command, *(["-interaction=nonstopmode"] if command[0] == "pdflatex" else []), target],
                       cwd=folder, stdout=subprocess.DEVNULL, check=False)
    for suffix in (".aux", ".bbl", ".blg", ".log", ".out"):
        (folder / f"main{suffix}").unlink(missing_ok=True)
    if (folder / "main.pdf").exists():
        shutil.copyfile(folder / "main.pdf", ROOT / "Sentiment_Driven_Portfolio_Optimization.pdf")
        print("Report: report/main.pdf and Sentiment_Driven_Portfolio_Optimization.pdf")


def reproduce():
    """Run everything from the stored data and the frozen choices, and compare with the stored results."""
    from pipeline.run import run_test
    unpack_scores()
    frozen = json.loads((OUT / "frozen.json").read_text())
    if frozen["fingerprint"] != fingerprint():
        raise SystemExit("Code or validation choices differ from the frozen record (results/frozen.json).")
    stored = OUT / "analysis.json"
    before = json.loads(stored.read_text())["metrics"] if stored.exists() else None
    run_test(frozen)
    diagnostics()
    after = json.loads(stored.read_text())["metrics"]
    if before:
        gap = max(abs(after[n][k] - before[n][k]) for n in STRATEGIES for k in ("cagr", "sharpe", "max_drawdown"))
        verdict = "match" if gap < 1e-6 else "differ from"
        print(f"Results {verdict} the stored results (largest difference in CAGR, Sharpe or drawdown: {gap:.2e}).")
    report()
    print("Done: tables in results/tables, figures in results/figures, data in results/.")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "reproduce"
    if stage in ("validate-price", "validate-sentiment", "test", "diagnostics", "reproduce"):
        unpack_scores()
    actions = {"reproduce": reproduce, "validate-price": validate_price, "validate-sentiment": validate_sentiment,
               "freeze": freeze, "test": lambda: test("--force" in sys.argv), "diagnostics": diagnostics,
               "report": report, "hashes": lambda: print(json.dumps(hashes(), indent=1))}
    actions.get(stage, lambda: print(__doc__))()
