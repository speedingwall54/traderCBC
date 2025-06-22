"""
Behavioral Modeling of Retail Trader Activity
Author: Your Name Here
Last Updated: YYYY-MM-DD

This script models retail trading behavior around market events (e.g. earnings announcements)
using public stock data and fits motion-based curves (like damped oscillations) to the extrapolated behavior.

Dependencies: yfinance, numpy, pandas, matplotlib, scikit-learn, tensorflow, scipy
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.optimize import curve_fit

# ------------------ CONFIG ------------------

# 👉 Replace 'TICKER_SYMBOL' with any stock ticker (e.g., "AAPL")
TICKER = "TICKER_SYMBOL"  # <- Replace this with your ticker

# 👉 Replace with desired date window (ideally earnings week +/- a few days)
START_DATE = "YYYY-MM-DD"  # <- Replace this with your start date
END_DATE = "YYYY-MM-DD"    # <- Replace this with your end date

FORECAST_DAYS = 30  # Number of days to extrapolate behavior

# ------------------ SETUP ------------------

# For reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

# ------------------ DATA RETRIEVAL ------------------

df = yf.download(TICKER, start=START_DATE, end=END_DATE, interval="1d", progress=False)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# ------------------ CBC SIGNAL CONSTRUCTION ------------------

df['volume_pressure'] = df['Volume'] / df['Volume'].rolling(3, min_periods=1).median()
df['price_reaction'] = df['Close'].diff().abs()
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df['volatility'] = df['log_return'].rolling(3, min_periods=1).std()

def robust_z(series):
    med = series.median()
    mad = np.median(np.abs(series - med))
    return (series - med) / (mad if mad else 1)

df['z_vol'] = robust_z(df['volume_pressure'])
df['z_react'] = robust_z(df['price_reaction'])
df['z_volatility'] = robust_z(df['volatility'])
df['cbc'] = df[['z_vol', 'z_react', 'z_volatility']].mean(axis=1).fillna(0)

cbc = df['cbc'].values.reshape(-1, 1)

# ------------------ LSTM EXTRAPOLATION ------------------

scaler = StandardScaler()
cbc_scaled = scaler.fit_transform(cbc)

WINDOW = 3
X, y = [], []
for i in range(len(cbc_scaled) - WINDOW):
    X.append(cbc_scaled[i:i+WINDOW])
    y.append(cbc_scaled[i+WINDOW])
X, y = np.array(X), np.array(y)

model = Sequential([
    LSTM(50, input_shape=(X.shape[1], X.shape[2])),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=200, batch_size=1, verbose=0)

# Extrapolation
input_seq = cbc_scaled[-WINDOW:]
forecast = []
for _ in range(FORECAST_DAYS):
    next_val = model.predict(input_seq.reshape(1, WINDOW, 1), verbose=0)
    forecast.append(next_val[0, 0])
    input_seq = np.append(input_seq[1:], next_val, axis=0)

cbc_extrapolated = np.concatenate([cbc_scaled.flatten(), forecast])
cbc_final = scaler.inverse_transform(cbc_extrapolated.reshape(-1, 1)).flatten()
t = np.arange(len(cbc_final))

# ------------------ CURVE FITTING ------------------

def harmonic(t, A, w, phi, c): return A * np.cos(w * t + phi) + c
def sinusoid(t, A, w, phi, c): return A * np.sin(w * t + phi) + c
def damped(t, A, lambd, w, phi, c): return A * np.exp(-lambd * t) * np.cos(w * t + phi) + c
def exponential(t, A, lambd, c): return A * np.exp(-lambd * t) + c
def logistic(t, L, k, t0, c): return L / (1 + np.exp(-k * (t - t0))) + c
def gaussian(t, A, mu, sigma, c): return A * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2)) + c

models = [
    ("Simple Harmonic", harmonic, [1, 0.2, 0, 0]),
    ("Sinusoidal", sinusoid, [1, 0.2, 0, 0]),
    ("Damped Harmonic", damped, [1, 0.1, 0.2, 0, 0]),
    ("Exponential Decay", exponential, [1, 0.1, 0]),
    ("Logistic Growth", logistic, [1, 0.1, len(t)//2, 0]),
    ("Gaussian Bump", gaussian, [1, np.mean(t), 5, 0])
]

results = {}
for name, func, guess in models:
    try:
        popt, pcov = curve_fit(func, t, cbc_final, p0=guess, maxfev=10000)
        yhat = func(t, *popt)
        mse = mean_squared_error(cbc_final, yhat)
        r2 = r2_score(cbc_final, yhat)
        ci = 1.96 * np.sqrt(np.diag(pcov))
        results[name] = {
            "curve": yhat,
            "mse": mse,
            "r2": r2,
            "params": np.round(popt, 3).tolist(),
            "ci": np.round(ci, 3).tolist()
        }
    except:
        continue

# ------------------ RESULTS & PLOT ------------------

best_model = min(results, key=lambda k: results[k]['mse'])

print("\nModel Fit Summary:")
for name, res in results.items():
    print(f"{name}: MSE={res['mse']:.3f}, R²={res['r2']:.3f}, Parameters={res['params']}, 95% CI={res['ci']}")

plt.figure(figsize=(14, 6))
plt.plot(t, cbc_final, 'k-', label="CBC (Observed + Forecast)", linewidth=2)
for name, res in results.items():
    style = '-' if name == best_model else '--'
    plt.plot(t, res['curve'], style, label=f"{name} Fit (MSE={res['mse']:.2f})")

plt.title(f"{TICKER} CBC Model Fits")
plt.xlabel("Days Since Start")
plt.ylabel("Composite Behavioral Curve")
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.show()
