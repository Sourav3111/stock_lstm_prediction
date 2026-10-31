"""
data.py
-------
Module for downloading, exploring, engineering features, and preparing
sliding-window time-series data for Stock Price Prediction using Deep Learning.

Key capabilities:
1. Download 5 years of daily OHLCV data using yfinance and save to CSV.
2. Exploratory Data Analysis (EDA): Missing value inspection, summary statistics.
3. Feature Engineering:
   - Simple Moving Averages: SMA_7, SMA_30, SMA_90
   - Relative Strength Index: RSI (14-day Wilder's smoothing)
   - Moving Average Convergence Divergence: MACD line, Signal line, MACD histogram
   - Volume & Price
4. Chronological 80/20 train/test split (strict no-shuffling to preserve temporal causality).
5. Data-leakage-free MinMaxScaler scaling (fitted strictly on training set only).
6. 60-day sliding-window sequence generation for recurrent models (LSTM / GRU / MLP).
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def download_stock_data(
    ticker: str = "AAPL",
    period: str = "5y",
    save_csv: bool = True,
    data_dir: str = "data",
    force_download: bool = False
) -> pd.DataFrame:
    """
    Downloads historical daily stock data from Yahoo Finance and optionally saves to CSV.
    If a local cached CSV exists and force_download is False, loads directly from disk.

    Parameters:
        ticker (str): Asset ticker symbol (e.g. 'AAPL', 'BTC-USD', 'MSFT', 'NVDA').
        period (str): Historical period duration (e.g. '5y' for 5 years).
        save_csv (bool): Whether to persist downloaded data as a CSV file.
        data_dir (str): Directory where CSV files are saved.
        force_download (bool): If True, re-downloads even if local CSV exists.

    Returns:
        pd.DataFrame: Cleaned daily OHLCV dataframe.
    """
    clean_ticker = ticker.replace("-", "_").replace("^", "")
    csv_path = os.path.join(data_dir, f"{clean_ticker}_{period}_daily.csv")

    if not force_download and os.path.exists(csv_path):
        print(f"[Data] Loading cached historical dataset from: {csv_path}")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        # Verify index name
        df.index.name = "Date"
        print(f"[Data] Loaded {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()}).")
        return df

    print(f"\n[Data] Fetching {period} of daily OHLCV data for '{ticker}' from Yahoo Finance...")
    
    # Download raw historical data via yfinance
    df = yf.download(ticker, period=period, interval="1d", progress=False)

    # Flatten multi-index column headers in newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or len(df) == 0:
        raise ValueError(f"No price data retrieved for symbol '{ticker}'. Please verify the ticker.")

    # Drop any records with missing Close price
    df = df.dropna(subset=["Close"]).copy()
    
    # Ensure chronological order (oldest to newest)
    df = df.sort_index()

    # Retain core OHLCV columns
    core_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[core_cols]

    # Save to CSV
    if save_csv:
        os.makedirs(data_dir, exist_ok=True)
        # Sanitize ticker name for filename (e.g., BTC-USD -> BTC_USD)
        clean_ticker = ticker.replace("-", "_").replace("^", "")
        csv_path = os.path.join(data_dir, f"{clean_ticker}_{period}_daily.csv")
        df.to_csv(csv_path)
        print(f"[Data] Saved raw historical dataset to: {csv_path}")

    print(f"[Data] Successfully retrieved {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()}).")
    return df


def display_data_summary(df: pd.DataFrame, ticker: str = "AAPL"):
    """
    Prints the first few rows, missing value check, and descriptive statistics.
    """
    print("\n" + "=" * 70)
    print(f"            EXPLORATORY DATA SUMMARY: {ticker} (First 5 Rows)")
    print("=" * 70)
    print(df.head())

    print("\n" + "-" * 70)
    print("MISSING VALUES CHECK:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        pct = (count / len(df)) * 100
        print(f"  • {col:<12}: {count} missing values ({pct:.2f}%)")

    print("\n" + "-" * 70)
    print("DESCRIPTIVE BASIC STATISTICS:")
    print(df.describe().T[["count", "mean", "std", "min", "50%", "max"]])
    print("=" * 70)


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Computes Wilder's Relative Strength Index (RSI).
    RSI = 100 - (100 / (1 + RS)), where RS = Average Gain / Average Loss over `window` days.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's Exponential Smoothing
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    Computes Moving Average Convergence Divergence (MACD).
    - MACD Line = 12-day EMA - 26-day EMA
    - Signal Line = 9-day EMA of MACD Line
    - Histogram = MACD Line - Signal Line
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches dataset with key technical indicators:
    1. SMA_7  (7-day moving average): short-term market momentum
    2. SMA_30 (30-day moving average): intermediate trend
    3. SMA_90 (90-day moving average): quarterly macroeconomic baseline
    4. RSI_14 (14-day Relative Strength Index): overbought/oversold momentum indicator
    5. MACD, MACD_Signal, MACD_Hist: trend-following momentum oscillator
    """
    print("[Data] Engineering technical indicators (SMA 7/30/90, RSI 14, MACD 12/26/9)...")
    data = df.copy()

    # Simple Moving Averages
    data["SMA_7"] = data["Close"].rolling(window=7).mean()
    data["SMA_30"] = data["Close"].rolling(window=30).mean()
    data["SMA_90"] = data["Close"].rolling(window=90).mean()

    # RSI
    data["RSI"] = calculate_rsi(data["Close"], window=14)

    # MACD
    macd, sig, hist = calculate_macd(data["Close"], fast=12, slow=26, signal=9)
    data["MACD"] = macd
    data["MACD_Signal"] = sig
    data["MACD_Hist"] = hist

    # Drop warm-up rows (the first 90 days where SMA_90 is NaN)
    n_before = len(data)
    data = data.dropna()
    print(f"[Data] Dropped {n_before - len(data)} warm-up rows. Final clean dataset has {len(data)} rows.")

    return data


def create_sliding_windows(data_x: np.ndarray, data_y: np.ndarray, window_size: int = 60):
    """
    Constructs rolling window sequences of length `window_size`.
    Input X[i] contains features from day i to day i+window_size-1.
    Target y[i] is the Close price on day i+window_size (next-day prediction).
    """
    X, y = [], []
    for i in range(len(data_x) - window_size):
        X.append(data_x[i : i + window_size])
        y.append(data_y[i + window_size, 0])
    return np.array(X), np.array(y)


def prepare_dataset(
    df: pd.DataFrame,
    feature_cols: list = None,
    window_size: int = 60,
    train_ratio: float = 0.8
):
    """
    Splits the dataset chronologically (80% train, 20% test) and fits MinMaxScaler
    strictly on the training partition to eliminate lookahead bias / data leakage.

    WHY WE NEVER SHUFFLE TIME SERIES DATA:
    --------------------------------------
    In standard tabular machine learning (e.g., predicting house prices), observations
    are assumed to be Independent and Identically Distributed (I.I.D.). Shuffling is used
    to ensure classes and distributions are balanced across train and test sets.

    However, financial time series data has strict temporal causality:
    1. Past events cause future events (the arrow of time cannot run in reverse).
    2. Autocorrelation: Today's price is strongly correlated with yesterday's price.
    3. If we shuffle time-series data:
       - Future prices (e.g. from 2024) would leak into the training set.
       - Past prices (e.g. from 2022) would end up in the test set.
       - The model would simply interpolate between known future and past points,
         producing unrealistically high test metrics (e.g., R² > 0.999).
       - When deployed live to trade on genuinely unseen future data, the model
         inevitably collapses because future information is unavailable.

    By enforcing chronological splitting (train on the past 80%, test on the latest 20%)
    and fitting scalers strictly on train data, we guarantee rigorous out-of-sample evaluation.
    """
    if feature_cols is None:
        feature_cols = ["Close", "Volume", "SMA_7", "SMA_30", "SMA_90", "RSI", "MACD", "MACD_Signal"]

    print(f"\n[Data] Selected {len(feature_cols)} features: {feature_cols}")

    # 1. Chronological Split (No Shuffling)
    n_samples = len(df)
    split_idx = int(n_samples * train_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"[Data] Chronological Split: {len(train_df)} train days ({train_ratio*100:.0f}%), "
          f"{len(test_df)} test days ({(1 - train_ratio)*100:.0f}%)")
    print(f"       Train period: {train_df.index[0].date()} to {train_df.index[-1].date()}")
    print(f"       Test period:  {test_df.index[0].date()} to {test_df.index[-1].date()}")

    # 2. Scalers fit ONLY on training set
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    # Fit and transform training features
    scaled_train_X = scaler_X.fit_transform(train_df[feature_cols].values)
    scaled_train_y = scaler_y.fit_transform(train_df[["Close"]].values)

    # Prepend the last `window_size` days of training data to the test set
    # so that the very first test day has a complete 60-day preceding window!
    full_test_df = pd.concat([train_df.iloc[-window_size:], test_df])
    scaled_test_X = scaler_X.transform(full_test_df[feature_cols].values)
    scaled_test_y = scaler_y.transform(full_test_df[["Close"]].values)

    # 3. Create Sliding-Window Sequences
    X_train, y_train = create_sliding_windows(scaled_train_X, scaled_train_y, window_size=window_size)
    X_test, y_test = create_sliding_windows(scaled_test_X, scaled_test_y, window_size=window_size)

    test_dates = test_df.index

    print(f"[Data] Sliding sequences created (Window Size = {window_size} days):")
    print(f"       X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"       X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "train_df": train_df,
        "test_df": test_df,
        "test_dates": test_dates,
        "feature_cols": feature_cols,
        "window_size": window_size,
    }


if __name__ == "__main__":
    # Test script directly
    df = download_stock_data(ticker="AAPL", period="5y")
    display_data_summary(df, ticker="AAPL")
    featured_df = add_technical_indicators(df)
    dataset = prepare_dataset(featured_df, window_size=60)
    print("\n[Data] Module sanity check completed successfully.")
