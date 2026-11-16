# Sentiment-Driven Portfolio Optimization: Do Headlines Pay After Costs?

IND310 Driftsoptimalisering course project, NMBU. Tests whether Google News headline sentiment, measured by
FinBERT, improves a weekly long-only portfolio of 141 U.S. and euro-area large caps after trading costs.
Models were chosen on 2012-2014 and tested once on 2015-2025. The report is
`Sentiment_Driven_Portfolio_Optimization.pdf`.

## Run it
```
pip install -r requirements.txt
python project.py
```
Reproduces the test, tables, figures and report from the stored data (about an hour on a laptop). The test
refuses to run if the code or choices changed after the validation freeze. The checks added after the first
draft run separately: `diagnostics/coverage.py`, `diagnostics/fnspid_check.py` and `diagnostics/news_quality.py`.

## Structure
```
project.py        one-command run and research protocol
pipeline/         the frozen model
diagnostics/      post-test tables, figures and checks
data_collection/  scripts that collected the data
data/             stored inputs
results/          test outputs, tables and figures
report/           LaTeX source of the report
```

## Results (2015-2025, 10 bp per trade)
| Strategy | CAGR | Sharpe | Max drawdown |
|---|---|---|---|
| Sentiment-driven | 11.90% | 0.64 | -31.6% |
| Price + sentiment | 12.20% | 0.65 | -30.4% |
| Price only | 10.28% | 0.54 | -34.4% |
| Minimum variance | 9.41% | 0.60 | -28.2% |
| Equal weight | 12.15% | 0.63 | -36.7% |
| MSCI World (URTH) | 11.18% | 0.60 | -32.7% |

No difference is statistically significant (paired block bootstrap).
