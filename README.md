# Stress testing a cap-weighted MSFT/AAPL portfolio (MF731 HW 9, sheet `09_stress_testing_pub`)

$1M portfolio, 43.83% MSFT / 56.17% AAPL, EWMA (lambda = theta = 0.97) on daily log returns
9/2/11..9/1/16, day-one shock x2 = mu2 - 5 sigma2, x1 = conditional mean - 5 conditional sd,
then 9 simulated days with EWMA updates, 50,000 paths, K = 10, alpha = 0.95.

- Unstressed: 1-day VaR 14,383.87, sqrt(10) VaR 45,485.80, capital 136,457.39 (= lecture slide 45).
- (a) average 10-day loss 70,838 (exact 70,802)  (b) VaR 151,693
- (c) 69.91% of paths exceed sqrt(10) VaR  (d) 8.89% exceed 3 sqrt(10) VaR

```bash
pip install -r requirements.txt
python src/stress.py      # ~2 s, writes generated/ and figures/losses.png
latexmk -pdf main.tex
```

Prices are Yahoo Finance closes (split-adjusted, not dividend-adjusted), downloaded
2026-10-23 and frozen in `data/`; adjusted closes are in `data/stocks_adjclose.csv` for the
robustness line in the report.
