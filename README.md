# Sentiment-Driven Portfolio Optimization: Do Headlines Pay After Costs?

IND310 Driftsoptimalisering, NMBU. A weekly, long-only portfolio of the S&P 100 and EURO STOXX 50 members of
31 Dec 2014 (141 firms with usable prices), where FinBERT news sentiment either confirms a price-based forecast
or drives the allocation on its own. Models are chosen on 2012-2014 and tested once on 2015-2025 with
10 basis points per trade. The report is `Sentiment_Driven_Portfolio_Optimization.pdf`.

## Run it
```
pip install -r requirements.txt
python project.py
```
This one command reproduces every result, table and figure in the report from the stored data. It checks the
code and validation choices against the frozen record, runs the 2015-2025 test, builds the post-test
diagnostics, compares the key results with the stored ones and, if `pdflatex` and `bibtex` are installed,
compiles the report. It takes about an hour on a laptop; no internet connection or GPU is needed.

## Folder structure
```
project.py            one-command run and the research protocol stages
pipeline/             the model (frozen): settings, data, news, sentiment, forecasts, optimiser, evaluation
diagnostics/          post-test tables and figures for the report (read stored results only)
data_collection/      scripts that collected the data (Yahoo prices, Google News headlines, FinBERT scores)
data/                 stored inputs: prices, FX, risk-free rate, universe, headlines, FinBERT scores (.gz)
results/              selection.json, frozen.json, test outputs, tables/ and figures/
report/               LaTeX source of the report (elsword.cls = Elsevier Word layout in LaTeX)
```

## Research protocol
```
python project.py validate-price       choose the price forecasting model on 2012-2014 (IC)
python project.py validate-sentiment   choose the sentiment gauge and its sign on 2012-2014
python project.py freeze               SHA-256 fingerprint of code, settings and choices
python project.py test                 the single test run, 2015-2025
python project.py diagnostics          post-test tables and figures
python project.py report               compile report/main.pdf
python project.py hashes               SHA-256 of every input file
```
`test` refuses to run if the code or choices changed after `freeze`, and refuses a second run.
The original freeze (11 Oct 2026) and test run are kept in `results/original_run/`. On 12 Oct 2026 the
repository was repackaged (one-command run, compressed FinBERT scores, renamed folders and comments) without
changing the model; the freeze was renewed and the full run reproduced the stored results.

Collecting the data again (`data_collection/`) is not needed to reproduce the results and gives slightly
different data, because Yahoo and Google News change over time.

## Results (2015-2025, 10 bp per trade)
| Strategy | CAGR | Sharpe | Max drawdown |
|---|---|---|---|
| Sentiment-driven | 11.90% | 0.64 | -31.6% |
| Price + sentiment | 12.20% | 0.65 | -30.4% |
| Price only | 10.28% | 0.54 | -34.4% |
| Minimum variance | 9.41% | 0.60 | -28.2% |
| Equal weight | 12.15% | 0.63 | -36.7% |
| MSCI World (URTH) | 11.18% | 0.60 | -32.7% |

No difference is statistically significant (paired block bootstrap); see the report and `results/analysis.json`.
