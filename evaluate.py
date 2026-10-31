"""
evaluate.py
-----------
Comprehensive evaluation suite for Stock Price Prediction:
1. Calculates out-of-sample metrics: MAE, RMSE, MAPE, and R².
2. Benchmarks against:
   - Naive Baseline (Tomorrow = Today: y_hat_{t+1} = y_t)
   - Simple Linear Regression
   - Deep Learning Models (LSTM, GRU, MLP)
3. Generates comparison tables and charts.
4. Performs 7-day future autoregressive forecasting.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes regression evaluation metrics:
    - MAE: Mean Absolute Error ($)
    - RMSE: Root Mean Squared Error ($)
    - MAPE: Mean Absolute Percentage Error (%)
    - R²: Coefficient of Determination (explained variance)
    - Directional Accuracy: % of days direction of price change is predicted correctly
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Avoid zero-division in MAPE
    non_zero = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0
    r2 = r2_score(y_true, y_pred)

    # Directional Accuracy (% times sign of daily change matches actual)
    actual_direction = np.sign(y_true[1:] - y_true[:-1])
    pred_direction = np.sign(y_pred[1:] - y_true[:-1])
    dir_acc = np.mean(actual_direction == pred_direction) * 100.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_%": mape,
        "R2": r2,
        "Directional_Acc_%": dir_acc
    }


def evaluate_all_models(
    models_dict: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y,
    y_train_last_price: float = None
) -> dict:
    """
    Evaluates all trained models alongside Naive Baseline and Linear Regression.

    Parameters:
        models_dict (dict): Dictionary of {'ModelName': model_instance_or_predict_func}
        X_test (np.ndarray): Scaled test sequences (samples, window_size, features)
        y_test (np.ndarray): Scaled test targets (samples,)
        scaler_y: MinMaxScaler for Close price
        y_train_last_price: Actual unscaled Close price right before the test period

    Returns:
        dict: predictions per model, actuals, and summary comparison dataframe
    """
    print("\n" + "=" * 75)
    print("               MODEL BENCHMARK & EVALUATION SUMMARY               ")
    print("=" * 75)

    # 1. Unscale Ground Truth Targets
    y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    predictions = {"Actual": y_actual}
    metrics_summary = []

    # 2. Benchmark 1: Naive Baseline ("Tomorrow = Today")
    # In sliding window, the last time step (-1) of feature 0 (Close) is today's price (t).
    # Naive prediction for t+1 is simply y_t!
    naive_scaled = X_test[:, -1, 0].reshape(-1, 1)
    y_naive = scaler_y.inverse_transform(naive_scaled).flatten()
    predictions["Naive Baseline"] = y_naive
    naive_metrics = calculate_metrics(y_actual, y_naive)
    metrics_summary.append({
        "Model": "Naive Baseline (Tomorrow=Today)",
        "MAE ($)": f"${naive_metrics['MAE']:.2f}",
        "RMSE ($)": f"${naive_metrics['RMSE']:.2f}",
        "MAPE (%)": f"{naive_metrics['MAPE_%']:.2f}%",
        "R² Score": f"{naive_metrics['R2']:.4f}",
        "Direction Acc": f"{naive_metrics['Directional_Acc_%']:.1f}%"
    })

    # 3. Evaluate each model in models_dict
    for name, model in models_dict.items():
        if hasattr(model, "predict"):
            # Check if scikit-learn model requiring 2D input
            if "Linear" in name or "sklearn" in str(type(model)):
                X_flat = X_test.reshape(len(X_test), -1)
                pred_scaled = model.predict(X_flat).reshape(-1, 1)
            else:
                pred_scaled = model.predict(X_test, verbose=0)
        elif callable(model):
            pred_scaled = model(X_test)
        else:
            continue

        pred_unscaled = scaler_y.inverse_transform(pred_scaled).flatten()
        predictions[name] = pred_unscaled
        m = calculate_metrics(y_actual, pred_unscaled)
        metrics_summary.append({
            "Model": name,
            "MAE ($)": f"${m['MAE']:.2f}",
            "RMSE ($)": f"${m['RMSE']:.2f}",
            "MAPE (%)": f"{m['MAPE_%']:.2f}%",
            "R² Score": f"{m['R2']:.4f}",
            "Direction Acc": f"{m['Directional_Acc_%']:.1f}%"
        })

    # Summary Dataframe
    comparison_df = pd.DataFrame(metrics_summary)
    print(comparison_df.to_string(index=False))
    print("=" * 75)

    return {
        "predictions": predictions,
        "comparison_table": comparison_df,
        "y_actual": y_actual
    }


def plot_predictions_comparison(
    predictions_dict: dict,
    test_dates,
    ticker: str = "AAPL",
    output_dir: str = "plots"
) -> str:
    """
    Plots Actual vs Predicted prices across all compared models on a single chart.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    plt.figure(figsize=(14, 7), dpi=150)
    y_actual = predictions_dict["Actual"]
    plt.plot(test_dates, y_actual, label="Actual Price", color="#0f172a", linewidth=2.4, alpha=0.95)

    color_palette = {
        "LSTM": "#2563eb",
        "GRU": "#16a34a",
        "MLP (Feedforward)": "#9333ea",
        "Linear Regression": "#f59e0b",
        "Naive Baseline": "#94a3b8"
    }

    style_palette = {
        "LSTM": "-",
        "GRU": "--",
        "MLP (Feedforward)": "-.",
        "Linear Regression": ":",
        "Naive Baseline": ":"
    }

    for model_name, preds in predictions_dict.items():
        if model_name == "Actual":
            continue
        c = color_palette.get(model_name, "#dc2626")
        ls = style_palette.get(model_name, "-")
        lw = 1.8 if "LSTM" in model_name else 1.4
        alpha = 0.9 if "LSTM" in model_name else 0.75
        plt.plot(test_dates, preds, label=model_name, color=c, linestyle=ls, linewidth=lw, alpha=alpha)

    plt.title(f"Model Comparison: Actual vs Predicted Stock Prices ({ticker})", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Close Price (USD)", fontsize=12)
    plt.legend(loc="upper left", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "actual_vs_predicted_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Plotting] Saved actual vs predicted comparison: {out_path}")
    return out_path


def forecast_7_days(
    model,
    df: pd.DataFrame,
    scaler_X,
    scaler_y,
    feature_cols: list,
    window_size: int = 60,
    forecast_days: int = 7,
    is_crypto: bool = False
) -> pd.DataFrame:
    """
    Performs 7-day multi-step recursive roll-forward forecast.
    """
    print(f"\n[Forecasting] Computing 7-day out-of-sample forward projection...")
    last_date = df.index[-1]
    
    if is_crypto or "BTC" in str(df.index):
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D")
    else:
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

    working_df = df[feature_cols].copy()
    forecasted_prices = []

    for step in range(forecast_days):
        current_window = working_df.iloc[-window_size:][feature_cols].values
        scaled_window = scaler_X.transform(current_window).reshape(1, window_size, len(feature_cols))

        if hasattr(model, "predict"):
            pred_scaled = model.predict(scaled_window, verbose=0)[0, 0]
        else:
            pred_scaled = model(scaled_window)[0, 0]

        pred_price = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
        forecasted_prices.append(pred_price)

        # Update indicators for roll-forward
        new_close = pred_price
        recent_closes = np.append(working_df["Close"].values[-(90 - 1):], new_close)
        new_sma_7 = np.mean(recent_closes[-7:])
        new_sma_30 = np.mean(recent_closes[-30:])
        new_sma_90 = np.mean(recent_closes[-90:]) if len(recent_closes) >= 90 else new_sma_30

        # RSI calculation for new row
        deltas = np.diff(recent_closes[-15:])
        gain = np.maximum(deltas, 0).mean()
        loss = np.maximum(-deltas, 0).mean()
        rs = gain / (loss if loss > 0 else 1e-6)
        new_rsi = 100.0 - (100.0 / (1.0 + rs))

        # MACD estimation
        ema12 = pd.Series(recent_closes).ewm(span=12, adjust=False).mean().iloc[-1]
        ema26 = pd.Series(recent_closes).ewm(span=26, adjust=False).mean().iloc[-1]
        new_macd = ema12 - ema26
        new_macd_sig = working_df["MACD_Signal"].iloc[-1] * 0.8 + new_macd * 0.2

        new_vol = working_df["Volume"].iloc[-10:].mean()

        new_row = pd.DataFrame([{
            "Close": new_close,
            "Volume": new_vol,
            "SMA_7": new_sma_7,
            "SMA_30": new_sma_30,
            "SMA_90": new_sma_90,
            "RSI": new_rsi,
            "MACD": new_macd,
            "MACD_Signal": new_macd_sig
        }], index=[future_dates[step]])

        working_df = pd.concat([working_df, new_row])

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecasted_Close": forecasted_prices
    }).set_index("Date")

    print("[Forecasting] Next 7 Trading Days Forecast:")
    for dt, row in forecast_df.iterrows():
        print(f"   • {dt.strftime('%Y-%m-%d (%a)')}: ${row['Forecasted_Close']:.2f}")

    return forecast_df


def plot_7_day_forecast(
    df_history: pd.DataFrame,
    forecast_df: pd.DataFrame,
    ticker: str = "AAPL",
    output_dir: str = "plots"
) -> str:
    """
    Plots the 7-day future forecast curve leading seamlessly from recent actual prices.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    recent_actual = df_history.iloc[-40:]
    plt.figure(figsize=(11, 5.5), dpi=150)
    plt.plot(recent_actual.index, recent_actual["Close"], label="Historical Actual (Last 40 Days)", color="#1e293b", linewidth=2.2)

    # Join history to forecast
    extended_dates = [recent_actual.index[-1]] + list(forecast_df.index)
    extended_prices = [recent_actual["Close"].iloc[-1]] + list(forecast_df["Forecasted_Close"])

    plt.plot(extended_dates, extended_prices, label="7-Day Future Forecast", color="#dc2626",
             linewidth=2.4, linestyle="--", marker="o", markersize=5)

    last_actual = recent_actual["Close"].iloc[-1]
    final_pred = forecast_df["Forecasted_Close"].iloc[-1]
    pct = ((final_pred - last_actual) / last_actual) * 100

    plt.scatter([forecast_df.index[-1]], [final_pred], color="#dc2626", s=80, zorder=6)
    plt.annotate(
        f"Target: ${final_pred:.2f}\n({pct:+.2f}%)",
        (forecast_df.index[-1], final_pred),
        textcoords="offset points",
        xytext=(-30, 15),
        bbox=dict(boxstyle="round,pad=0.4", fc="#fee2e2", ec="#dc2626", lw=1.2),
        fontsize=9,
        fontweight="bold"
    )

    plt.title(f"7-Day Future Stock Price Forecast - {ticker}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Close Price (USD)", fontsize=11)
    plt.legend(loc="upper left", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "future_forecast.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Plotting] Saved future forecast plot: {out_path}")
    return out_path


def plot_summary_dashboard(
    history,
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    y_naive: np.ndarray,
    test_dates,
    forecast_df: pd.DataFrame,
    df_history: pd.DataFrame,
    m_lstm: dict,
    m_naive: dict,
    ticker: str = "AAPL",
    output_dir: str = "plots"
) -> str:
    """
    Creates an all-in-one Executive Summary Dashboard figure (4 subplots):
    1. Test set Actual vs LSTM predicted vs Naive
    2. Training vs Validation loss curve with EarlyStopping point
    3. Bar chart error comparison (MAE, RMSE, MAPE)
    4. 7-Day Future Forecast projection
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=150)
    fig.suptitle(f"Executive Project Dashboard: LSTM Stock Forecasting ({ticker})", fontsize=16, fontweight="bold", y=0.98)

    # Subplot 1: Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.plot(test_dates, y_actual, label="Actual Price", color="#0f172a", linewidth=2.0)
    ax1.plot(test_dates, y_pred, label=f"LSTM (MAE: ${m_lstm['MAE']:.2f})", color="#2563eb", linewidth=1.8)
    ax1.plot(test_dates, y_naive, label="Naive Baseline", color="#94a3b8", linestyle=":", linewidth=1.2)
    ax1.set_title("Test Set Performance: Actual vs Predicted", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Subplot 2: Loss Convergence
    ax2 = axes[0, 1]
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(train_loss) + 1)
    ax2.plot(epochs, train_loss, label="Train Loss (MSE)", color="#1d4ed8", linewidth=2.0)
    ax2.plot(epochs, val_loss, label="Val Loss (MSE)", color="#ea580c", linewidth=2.0, linestyle="--")
    best_ep = val_loss.index(min(val_loss)) + 1
    ax2.scatter([best_ep], [min(val_loss)], color="#dc2626", s=60, label=f"Optimal Epoch ({best_ep})")
    ax2.set_title("Model Convergence: Train vs Val Loss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # Subplot 3: Metrics Benchmark Bar Chart
    ax3 = axes[1, 0]
    metric_labels = ["MAE ($)", "RMSE ($)", "MAPE (%)"]
    lstm_vals = [m_lstm["MAE"], m_lstm["RMSE"], m_lstm["MAPE_%"]]
    naive_vals = [m_naive["MAE"], m_naive["RMSE"], m_naive["MAPE_%"]]
    x = np.arange(len(metric_labels))
    w = 0.35
    ax3.bar(x - w/2, lstm_vals, w, label="LSTM Model", color="#2563eb")
    ax3.bar(x + w/2, naive_vals, w, label="Naive Baseline", color="#94a3b8")
    ax3.set_xticks(x)
    ax3.set_xticklabels(metric_labels, fontsize=10, fontweight="bold")
    ax3.set_title(f"Benchmark Error Comparison | R² = {m_lstm['R2']:.4f}", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.5)

    # Subplot 4: 7-Day Forecast Curve
    ax4 = axes[1, 1]
    recent_actual = df_history.iloc[-30:]
    ax4.plot(recent_actual.index, recent_actual["Close"], label="Recent Actual", color="#0f172a", linewidth=1.8)
    ext_dates = [recent_actual.index[-1]] + list(forecast_df.index)
    ext_prices = [recent_actual["Close"].iloc[-1]] + list(forecast_df["Forecasted_Close"])
    ax4.plot(ext_dates, ext_prices, label="7-Day Forecast", color="#dc2626", linewidth=2.2, linestyle="--", marker="o", markersize=4)
    end_price = forecast_df["Forecasted_Close"].iloc[-1]
    ax4.set_title(f"Next 7-Day Forecast (Expected: ${end_price:.2f})", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Price ($)")
    ax4.legend(loc="upper left", fontsize=9)
    ax4.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    dash_path = os.path.join(output_dir, "lstm_summary_dashboard.png")
    plt.savefig(dash_path)
    plt.close()
    print(f"[Plotting] Saved summary dashboard: {dash_path}")
    return dash_path
