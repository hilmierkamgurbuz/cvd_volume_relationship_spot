# CVD & Volume Relationship Analyzer (Spot Data)

This project analyzes the relationship between Cumulative Volume Delta (CVD) and volume in Binance **spot markets**. It performs multi-timeframe trend analysis, calculates momentum, slope, and detects divergence between price and volume flow.

---

##  Features

- Real-time OHLCV data fetching from Binance spot market
- Cumulative Volume Delta (CVD) calculation
- Volume normalization using Z-score
- Linear regression-based trend slope detection
- Momentum calculation using Rate of Change (ROC)
- Price vs. CVD divergence detection
- Multi-timeframe analysis: 15m, 1h, 4h
- Visualizations with Matplotlib
- Summary table for each symbol with trend scores

---

##  Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
