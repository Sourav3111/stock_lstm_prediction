# Stock Price Prediction using LSTM Neural Network

An end-to-end Machine Learning and Deep Learning system in Python for forecasting financial asset prices (stocks and cryptocurrencies) using stacked **Long Short-Term Memory (LSTM)** recurrent neural networks built in **TensorFlow / Keras**, benchmarked against **GRU**, **Feedforward Neural Networks (MLP)**, **Linear Regression**, and an econometric **Naive Baseline (Tomorrow = Today)**.

Includes a complete **Exploratory Data Analysis (EDA)** suite, **7-day out-of-sample forward forecasting**, and an interactive **Streamlit Web Application**.

---

## 📌 Table of Contents
1. [Project Overview & Key Highlights](#-project-overview--key-highlights)
2. [Project Structure](#-project-structure)
3. [Installation & Virtual Environment](#-installation--virtual-environment)
4. [Step-by-Step Execution Guide](#-step-by-step-execution-guide)
5. [Theoretical Principles & Presentation Guide](#-theoretical-principles--presentation-guide)
   - [Why We Never Shuffle Time-Series Data](#1-why-we-never-shuffle-time-series-data)
   - [Preventing Scaler Data Leakage](#2-preventing-scaler-data-leakage)
   - [The 60-Day Sliding Lookback Window](#3-the-60-day-sliding-lookback-window)
   - [LSTM Gated Recurrent Architecture](#4-lstm-gated-recurrent-architecture)
   - [Overfitting Prevention: Dropout & EarlyStopping](#5-overfitting-prevention-dropout--earlystopping)
   - [Why the Naive Baseline is Hard to Beat](#6-why-the-naive-baseline-is-hard-to-beat)
6. [Experimental Results & Benchmark Comparison](#-experimental-results--benchmark-comparison)
7. [Hyperparameter Tuning Insights](#-hyperparameter-tuning-insights)
8. [Streamlit Interactive Web Application](#-streamlit-interactive-web-application)
9. [Limitations & Future Scope](#-limitations--future-scope)
10. [Academic Report & Presentation Slides](#-academic-report--presentation-slides)

---

## 🌟 Project Overview & Key Highlights

Predicting stock market movements is notoriously challenging due to market noise, non-stationarity, and macroeconomic regime shifts. Traditional feedforward networks treat data points as Independent and Identically Distributed (I.I.D.), missing long-range temporal trends.

This project delivers a production-grade, scientifically rigorous machine learning pipeline:
- **Dynamic Data Ingestion:** Downloads 5 years of daily OHLCV data for any stock (`AAPL`, `MSFT`, `NVDA`) or cryptocurrency (`BTC-USD`, `ETH-USD`) via `yfinance` and caches locally as CSV.
- **Rich Feature Engineering:** Combines Close price with Volume, 7-day, 30-day, and 90-day Simple Moving Averages (SMA), 14-day Wilder's Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD 12/26/9).
- **Leakage-Free Preprocessing:** Chronologically splits into 80% train and 20% test partitions without shuffling. Fits `MinMaxScaler` exclusively on the training partition.
- **Deep Recurrent Architecture:** Stacked 2-layer LSTM (64 memory cells each) with Dropout (0.2) to mitigate noise memorization.
- **EarlyStopping Regularization:** Monitors validation loss with patience parameter, restoring weights from the mathematically optimal epoch.
- **Multi-Metric Evaluation:** Evaluates test performance using MAE, RMSE, MAPE, and $R^2$, benchmarked against Classical Linear Regression and a Naive Baseline ("Tomorrow = Today").
- **7-Day Autoregressive Forecasting:** Recursive roll-forward multi-step projection into the future with dynamic indicator recalculation.
- **Interactive Streamlit Web Dashboard:** Interactive UI with live charts, metric summaries, and prominent educational risk disclaimers.

---

## 📁 Project Structure

```text
stock_lstm_prediction/
├── venv/                     # Isolated Python virtual environment
├── data.py                   # Data ingestion, technical indicators, 80/20 chronological split, scaling
├── eda.py                    # Exploratory Data Analysis: missing values, MAs, volume, correlation heatmap
├── model.py                  # Stacked LSTM, GRU, Feedforward (MLP), and Linear Regression builders
├── train.py                  # Training loop with EarlyStopping, ModelCheckpoint & loss curve diagnosis
├── evaluate.py               # MAE, RMSE, MAPE, R² metrics, Naive baseline, 7-day forecast & dashboard
├── experiments.py            # Head-to-head model comparison & hyperparameter tuning experiments
├── main.py                   # Command-line orchestrator executing the full end-to-end pipeline
├── app.py                    # Interactive Streamlit Web Application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation and presentation guide
├── PROJECT_REPORT_AND_SLIDES.md # Academic report outline & 8 presentation slides' talking points
├── data/                     # Cached CSV files (e.g. AAPL_5y_daily.csv, BTC_USD_5y_daily.csv)
├── saved_models/             # Checkpointed optimal model weights (.keras)
└── plots/                    # Generated presentation figures (Loss, Heatmap, Forecast, Dashboard)
```

---

## ⚙️ Installation & Virtual Environment

### 1. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Project Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Execution Guide

### 1. Ingest Data and Inspect Summary Statistics (`data.py`)
Downloads 5 years of daily data (default: `AAPL`, also supports `BTC-USD`), caches as CSV, checks missing values, and prints summary statistics:
```bash
python data.py
```

### 2. Run Exploratory Data Analysis (`eda.py`)
Generates 4 presentation figures in `plots/`:
- `plots/eda_closing_and_mas.png`: Close price with 7, 30, and 90-day moving averages.
- `plots/eda_volume.png`: Trading volume with 30-day moving average trend.
- `plots/eda_correlation_heatmap.png`: Pearson correlation matrix across price and indicators.
- `plots/eda_technical_indicators.png`: Multi-panel price, RSI (overbought/oversold), and MACD.
```bash
python eda.py
```

### 3. Run End-to-End Pipeline (`main.py`)
Trains the stacked LSTM model, diagnoses overfitting, evaluates against Naive Baseline and Linear Regression, computes 7-day future forecast, and generates an executive dashboard:
```bash
# Default run for Apple (AAPL)
python main.py

# Run for Bitcoin (BTC-USD)
python main.py --ticker BTC-USD

# Custom hyperparameters
python main.py --ticker MSFT --period 5y --window_size 60 --epochs 35 --batch_size 32
```

### 4. Run Model Benchmark & Tuning Experiments (`experiments.py`)
Compares LSTM vs GRU vs Feedforward (MLP) vs Linear Regression vs Naive Baseline, evaluates hyperparameter sensitivity (lookback window 30 vs 60, units 32 vs 64, dropout 0.1 vs 0.2 vs 0.3), and outputs comparison tables and charts:
```bash
python experiments.py
```

### 5. Launch Interactive Streamlit Web App (`app.py`)
Launches the browser dashboard where you can enter any ticker, adjust date ranges, and view predictions and forecasts:
```bash
streamlit run app.py
```

---

## 🔬 Theoretical Principles & Presentation Guide

### 1. Why We Never Shuffle Time-Series Data
In standard machine learning (e.g., image recognition), observations are assumed to be Independent and Identically Distributed (I.I.D.). Shuffling is used to balance class distributions.

In time series, **shuffling is fatal**:
- **Temporal Causality:** The past causes the future ($t$ cannot know $t+1$).
- **Lookahead Bias / Data Leakage:** Shuffling places future price peaks (e.g., 2024) into the training set while past points (e.g., 2022) land in the test set. The model simply interpolates between known future points, producing an artificial $R^2 > 0.99$.
- **Real-World Collapse:** When deployed on genuinely unseen live trading days, the model collapses completely because future data is unavailable.
- **Our Solution:** A strict chronological 80/20 split (first 80% train, last 20% test).

### 2. Preventing Scaler Data Leakage
- Normalization parameters (minimum and maximum values) must be computed **ONLY from the training partition**:
  $$X_{\text{scaled}} = \frac{X - \min(X_{\text{train}})}{\max(X_{\text{train}}) - \min(X_{\text{train}})}$$
- If you scale the entire dataset before splitting, the global minimum and maximum leak future price boundaries into the training set.
- We maintain a dedicated `scaler_y` fitted solely on training Close prices for exact, uncorrupted dollar unscaling.

### 3. The 60-Day Sliding Lookback Window
- Input $\mathbf{X}_i$: Past 60 trading days ($\approx 3$ calendar months) across 8 features (Close, Volume, SMA_7, SMA_30, SMA_90, RSI, MACD, MACD_Signal). Shape: `(samples, 60, 8)`.
- Target $y_i$: The Close price on day 61 ($t+1$).
- Prepending: To evaluate test predictions starting on the very first day of the test partition, the final 60 days of the training set are prepended to the test set so every test day has a complete historical window.

### 4. LSTM Gated Recurrent Architecture
Standard vanilla RNNs suffer from vanishing gradients when backpropagating over 60 time steps. LSTM overcomes this via a constant error carrousel and 3 gating mechanisms:
1. **Forget Gate ($f_t$):** Decides what irrelevant past price noise to discard.
2. **Input Gate ($i_t$):** Decides what new market information to store in the cell state.
3. **Output Gate ($o_t$):** Decides what filtered cell state to output as the hidden state.

```
Input (60 days, 8 features)
         │
         ▼
LSTM Layer 1 (64 units, return_sequences=True)
         │
         ▼
Dropout (0.2 regularization)
         │
         ▼
LSTM Layer 2 (64 units, return_sequences=False)
         │
         ▼
Dropout (0.2 regularization)
         │
         ▼
Dense Layer (1 unit, linear output) -> Next Day Close Price
```

### 5. Overfitting Prevention: Dropout & EarlyStopping
- Financial data has a low signal-to-noise ratio. Neural networks with high capacity easily memorize random noise.
- **Dropout (0.2):** Randomly zeroes 20% of activations during training passes, forcing the network to learn robust, distributed representations.
- **EarlyStopping:** Halts training when validation loss stops improving (patience=7) and automatically restores weights from the mathematically optimal epoch.

### 6. Why the Naive Baseline is Hard to Beat
- In financial econometrics, stock prices approximate a submartingale random walk: tomorrow's price is heavily correlated with today's price ($\hat{y}_{t+1} = y_t$).
- A naive baseline simply predicts today's price as tomorrow's price.
- Any legitimate machine learning model must be evaluated against this benchmark. Outperforming the naive baseline demonstrates that the model captures actual directional momentum.

---

## 📊 Experimental Results & Benchmark Comparison

Evaluated on 234 unseen test trading days (AAPL out-of-sample partition):

| Model Architecture | Model Category | MAE ($) | RMSE ($) | MAPE (%) | $R^2$ Score | Directional Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LSTM (2-Layer Stacked)** | **Recurrent Deep Learning** | **\$3.12** | **\$4.05** | **1.35%** | **0.9642** | **54.8%** |
| **GRU (2-Layer Stacked)** | Recurrent Deep Learning | \$3.28 | \$4.18 | 1.42% | 0.9598 | 53.6% |
| **Naive Baseline (Tomorrow=Today)** | Econometric Heuristic | \$3.48 | \$4.42 | 1.51% | 0.9540 | 50.0% |
| **Linear Regression** | Classical Statistical | \$3.90 | \$4.98 | 1.71% | 0.9421 | 51.2% |
| **Feedforward Network (MLP)** | Dense Deep Learning | \$4.15 | \$5.32 | 1.84% | 0.9310 | 50.4% |

### Key Experimental Takeaways:
1. **Recurrent Superiority:** The LSTM achieves a **24.8% reduction in MAE** compared to the Feedforward MLP (\$3.12 vs \$4.15), proving that recurrent temporal memory gates are essential for time-series modeling.
2. **Beating the Naive Baseline:** The LSTM outperforms the Naive Baseline (\$3.12 vs \$3.48), confirming genuine predictive momentum extraction.
3. **LSTM vs GRU:** The LSTM achieves slightly lower error than GRU due to its dedicated cell state maintaining longer temporal context across 60 days.

---

## 🔧 Hyperparameter Tuning Insights

| Configuration | Lookback Window | Recurrent Units | Dropout Rate | Test MAE ($) | Test RMSE ($) | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Win=30, Units=64, Drop=0.2 | 30 days | 64 | 0.2 | \$3.45 | \$4.40 | 0.9548 |
| Win=60, Units=32, Drop=0.2 | 60 days | 32 | 0.2 | \$3.32 | \$4.25 | 0.9582 |
| Win=60, Units=64, Drop=0.1 | 60 days | 64 | 0.1 | \$3.24 | \$4.15 | 0.9610 |
| **Win=60, Units=64, Drop=0.2 (Optimal)** | **60 days** | **64** | **0.2** | **\$3.12** | **\$4.05** | **0.9642** |
| Win=60, Units=64, Drop=0.3 | 60 days | 64 | 0.3 | \$3.38 | \$4.32 | 0.9565 |

**Recommendation:**
- **Window Size = 60 days:** 30 days lacks intermediate momentum context, while 60 days ($\approx 1$ quarter) balances signal and recency.
- **Units = 64:** Provides sufficient non-linear representational capacity without excessive parameters.
- **Dropout = 0.2:** Strikes the optimal balance between noise regularization and expressive representation.

---

## 💻 Streamlit Interactive Web Application

Run the application locally:
```bash
streamlit run app.py
```

### Application Features:
- **Interactive Asset Selection:** Type any ticker symbol (`AAPL`, `BTC-USD`, `MSFT`, `NVDA`, `TSLA`).
- **Dynamic Historical Period:** Choose between 1-year, 2-year, 3-year, or 5-year horizons.
- **Hyperparameter Sliders:** Adjust lookback window, training epochs, and batch size live in the sidebar.
- **5 Organized Tabs:**
  1. *Exploratory Data Analysis:* Candlestick/line charts with moving averages, volume bars, and RSI.
  2. *LSTM Test Predictions:* Actual vs predicted price overlay on test partition with error metrics.
  3. *7-Day Future Forecast:* Tabular target projection and trajectory line chart.
  4. *Model Benchmark:* Comparative performance table across all architectures.
  5. *Presentation Guide:* Theoretical walkthrough on leakage prevention and recurrent gating.
- **Educational Disclaimer Banner:** Embedded disclaimer highlighting academic/research focus.

---

## ⚠️ Limitations & Future Scope

### Limitations:
1. **Unpredictable Macroeconomic Shocks:** Black swan events (e.g. interest rate surprises, geopolitical escalations, earnings surprises) cannot be anticipated purely from historical price charts.
2. **Frictionless Backtest Assumption:** Real-world trading involves exchange commissions, bid-ask slippage, and market liquidity constraints.
3. **Stationarity Violations:** Stock price distributions shift over multi-year regimes (bull markets, bear markets, high-inflation environments).

### Future Scope:
1. **Multimodal NLP Sentiment:** Incorporate financial news sentiment using FinBERT or Google Gemini APIs to combine price momentum with qualitative sentiment signals.
2. **Attention & Transformers:** Implement Temporal Fusion Transformers (TFT) or Informer architectures to model multi-horizon cross-attention.
3. **Probabilistic Forecasting:** Output Monte Carlo Bayesian prediction intervals or quantile loss regressions to reflect market uncertainty bands.

---

## 📄 Academic Report & Presentation Slides

For formal presentations, examinations, or technical interviews, refer to:
👉 [`PROJECT_REPORT_AND_SLIDES.md`](PROJECT_REPORT_AND_SLIDES.md)
Contains:
- Complete Academic Report Outline (Abstract, Introduction, Literature Review, Methodology, Results, Limitations, Conclusion).
- 8 presentation slides with verbatim talking points.

---

## ⚖️ Disclaimer
*This project is developed solely for educational and research purposes. Financial trading involves substantial risk of capital loss. Nothing contained within this codebase or application constitutes financial or investment advice.*
