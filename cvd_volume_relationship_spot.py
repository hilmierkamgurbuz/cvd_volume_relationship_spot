#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
from sklearn.linear_model import LinearRegression

client = Client()  # API anahtarlarını buraya ekleyebilirsin

# Binance'ten veri çekme
def get_ohlcv(symbol, interval='1h', lookback=100):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=lookback)
    df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume',
                                       'close_time', 'quote_asset_volume', 'number_of_trades',
                                       'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume', 'taker_buy_base']]
    df = df.astype(float)
    return df

# CVD hesaplama
def calculate_cvd(df):
    df['delta_volume'] = 2 * df['taker_buy_base'] - df['volume']
    df['cvd'] = df['delta_volume'].cumsum()
    return df

# Linear regression ile slope hesaplama
def linear_slope(series, window=20):
    y = series[-window:].values.reshape(-1, 1)
    x = np.arange(window).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model.coef_[0][0]

# Hacim normalizasyonu (Z-score ile)
def normalize_volume(df):
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
    return df

# CVD momentum hesaplama (ROC)
def cvd_momentum(df, window=20):
    df['cvd_roc'] = df['cvd'].pct_change().rolling(window).mean()
    return df

# Fiyat ve CVD Divergence kontrolü
def check_divergence(df):
    price_diff = df['close'] - df['close'].shift(1)
    cvd_diff = df['cvd'] - df['cvd'].shift(1)
    if price_diff.iloc[-1] > 0 and cvd_diff.iloc[-1] < 0:
        return "Bearish Divergence"
    elif price_diff.iloc[-1] < 0 and cvd_diff.iloc[-1] > 0:
        return "Bullish Divergence"
    return "No Divergence"

# Grafikleri çizme
def plot_cvd_volume(df, symbol, timeframe):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    fig.suptitle(f"{symbol} - {timeframe} | CVD & Hacim", fontsize=14)

    ax1.plot(df.index, df['cvd'], color='blue', label='CVD', linewidth=1.8)
    ax1.set_ylabel('CVD', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(df.index, df['volume'], color='gray', linestyle='--', label='Hacim', linewidth=1.4)
    ax2.set_ylabel('Hacim', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    ax1.set_xlabel("Zaman")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.show()

# Sembolleri analiz et
symbols = ['ETHUSDT', 'BTCUSDT']
summary_rows = []

# Önce analiz yap ve summary_rows'u doldur
for sym in symbols:
    timeframes = ['15m', '1h', '4h']
    score = 0
    trend_info = []

    for tf in timeframes:
        df = get_ohlcv(sym, interval=tf, lookback=50)
        df = calculate_cvd(df)
        df = normalize_volume(df)
        df = cvd_momentum(df)

        cvd_slope = linear_slope(df['cvd'], window=20)
        vol_slope = linear_slope(df['volume'], window=20)
        divergence = check_divergence(df)

        cvd_trend = "Yükselen" if cvd_slope > 0 else "Düşen"
        vol_trend = "Yükselen" if vol_slope > 0 else "Düşen"

        if cvd_slope > 0: score += 1
        if vol_slope > 0: score += 1

        trend_info.append((tf, vol_trend, cvd_trend))

    if score >= 5:
        genel_trend = "GÜÇLÜ YÜKSELİŞ"
    elif score >= 3:
        genel_trend = "NÖTR / KARARSIZ"
    else:
        genel_trend = "DÜŞÜŞ"

    row = {'Sembol': sym}
    for tf, vol, cvd in trend_info:
        row[f'{tf} Hacim'] = vol
        row[f'{tf} CVD'] = cvd
    row['Trend Skoru'] = f"{score}/6"
    row['Genel Trend'] = genel_trend
    summary_rows.append(row)

#  Özet tabloyu analizden sonra hemen göster
summary_df = pd.DataFrame(summary_rows)
print("\n Özet Trend Tablosu:\n")
print(summary_df.to_string(index=False))

#  Sonrasında detaylı analizleri yazdır ve grafikleri göster
for sym in symbols:
    print(f"\n➔ {sym} Trend Analizi:")
    timeframes = ['15m', '1h', '4h']
    for tf in timeframes:
        df = get_ohlcv(sym, interval=tf, lookback=50)
        df = calculate_cvd(df)
        df = normalize_volume(df)
        df = cvd_momentum(df)

        cvd_slope = linear_slope(df['cvd'], window=20)
        vol_slope = linear_slope(df['volume'], window=20)
        divergence = check_divergence(df)

        cvd_trend = "Yükselen" if cvd_slope > 0 else "Düşen"
        vol_trend = "Yükselen" if vol_slope > 0 else "Düşen"

        print(f"➔ {tf}: CVD: {cvd_trend} ({cvd_slope:.2f}), Hacim: {vol_trend} ({vol_slope:.2f}), Divergence: {divergence}")
        plot_cvd_volume(df, sym, tf)


# In[ ]:




