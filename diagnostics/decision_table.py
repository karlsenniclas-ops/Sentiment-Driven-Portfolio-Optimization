"""Post-test diagnostic: how often the price signal and the sentiment gauge were right about a stock's
next-week relative return, and what happened when sentiment changed the price-based decision.
Rebuilds the frozen signals from cached inputs (results/frozen.json); it does not re-run or change the test.
Adds the numbers of Table 7 in the report as macros to results/tables/extra_numbers.tex.
Usage: python diagnostics/decision_table.py"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import forecast as fc  # noqa: E402
from pipeline.config import OUT  # noqa: E402
from pipeline.run import cached, market_inputs, news_inputs, signals, test_indices  # noqa: E402
from diagnostics.report_figures import write_macros  # noqa: E402


def main():
    frozen = json.loads((OUT / "frozen.json").read_text())
    universe, clock, decisions, prices = market_inputs()
    panel = fc.Panel(prices)
    _, news = news_inputs(universe, clock, decisions, {})
    price_fc = cached(f"forecast_{frozen['price_model']}", lambda: None)
    price_cal = fc.calibrate(panel, price_fc, [str(y) for y in range(2009, 2026)])
    g, _, combined, _ = signals(panel, news, price_fc, price_cal, frozen["sector_weight"], frozen["direction"],
                                frozen["correction_variant"])
    rows = {k: [] for k in ("all", "agree", "neutral", "disagree", "changed")}
    weekly = {k: [] for k in rows}       # weekly means, averaged over weeks as in the other diagnostics
    for i in test_indices(decisions):
        week = {k: [] for k in rows}
        f, c, rel = price_fc.get(i, {}), combined.get(i, {}), panel.relative[i]
        names = [t for t in f if t in rel and t in g.get(i, {}) and t in c]
        if len(names) < 20:
            continue
        zp, zc = fc.zscore({t: f[t] for t in names}), fc.zscore({t: c[t] for t in names})
        for t in names:
            p, s, r = np.sign(zp[t]), np.sign(g[i][t]), np.sign(rel[t])
            rec = {"price_right": p == r, "sent_right": s == r if s != 0 else np.nan,
                   "comb_right": np.sign(zc[t]) == r, "ret": p * rel[t]}
            week["all"].append(rec)
            agreement = p * g[i][t]
            week["agree" if agreement >= 0.5 else "disagree" if agreement <= -0.5 else "neutral"].append(rec)
            if np.sign(zc[t]) != p:
                week["changed"].append(rec)
        for k, v in week.items():
            rows[k] += v
            if v:
                weekly[k].append({m: np.nanmean([x[m] for x in v]) if m == "sent_right" else np.mean([x[m] for x in v])
                                  for m in ("price_right", "sent_right", "comb_right", "ret")})
    total = len(rows["all"])
    macros = {}
    for key in rows:
        v, w = rows[key], weekly[key]
        share = len(v) / total * 100
        pr = np.mean([x["price_right"] for x in w]) * 100
        sr = np.nanmean([x["sent_right"] for x in w]) * 100
        cr = np.mean([x["comb_right"] for x in w]) * 100
        ret = np.mean([x["ret"] for x in w]) * 1e4
        tag = key.capitalize()
        macros.update({f"Dec{tag}Share": f"{share:.0f}\\%", f"Dec{tag}Price": f"{pr:.1f}\\%",
                       f"Dec{tag}Sent": f"{sr:.1f}\\%", f"Dec{tag}Comb": f"{cr:.1f}\\%", f"Dec{tag}Ret": f"{ret:+.1f}"})
    write_macros(macros)
    print(macros)


if __name__ == "__main__":
    main()
