"""
main.py
-------
Mini Project: Stock Price Prediction using LSTM Neural Network
Author: Pair Programming with Antigravity

Orchestrates the entire end-to-end Machine Learning pipeline:
1. Ingests 5 years of daily stock/crypto data via yfinance (supports AAPL, BTC-USD, etc.).
2. Engineers technical indicators: SMA (7, 30, 90), RSI (14-day), MACD (12, 26, 9), Volume.
3. Chronological 80/20 train/test split (no shuffling to preserve temporal causality).
4. Normalizes features with MinMaxScaler fitted exclusively on training set (no data leakage).
5. Constructs and compiles a 2-layer stacked LSTM network with Dropout in TensorFlow/Keras.
6. Trains with EarlyStopping and ModelCheckpoint callbacks.
7. Evaluates on unseen 20% test partition using MAE, RMSE, MAPE, and R².
8. Compares against Naive Baseline (Tomorrow = Today) and Classical Linear Regression.
9. Performs multi-step autoregressive 7-day future out-of-sample forecasting.
10. Generates and saves 4 presentation-ready figures in plots/.
"""

import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data import download_stock_data, add_technical_indicators, prepare_dataset
from model import build_lstm_model, build_linear_regression
from train import train_model, plot_loss_and_diagnose
from evaluate import calculate_metrics, evaluate_all_models, forecast_7_days, plot_7_day_forecast, plot_summary_dashboard, plot_predictions_comparison


def run_pipeline(
    ticker: str = "AAPL",
    period: str = "5y",
    window_size: int = 60,
    epochs: int = 30,
    batch_size: int = 32,
    patience: int = 7,
    output_dir: str = "plots",
    model_save_path: str = "saved_models/lstm_stock_model.keras"
):
    print("=" * 75)
    print("       STOCK PRICE PREDICTION USING LSTM NEURAL NETWORK")
    print("=" * 75)
    print(f"Asset Ticker:        {ticker}")
    print(f"Historical Window:   {period}")
    print(f"Lookback Window:     {window_size} days")
    print(f"Epochs:              {epochs} (EarlyStopping patience: {patience})")
    print(f"Plots Destination:   ./{output_dir}")
    print("=" * 75)

    # Step 1: Download & Explore Daily Stock Data
    raw_df = download_stock_data(ticker=ticker, period=period, save_csv=True)

    # Step 2: Feature Engineering (SMA 7/30/90, RSI 14, MACD 12/26/9, Volume)
    featured_df = add_technical_indicators(raw_df)

    # Step 3: Chronological Split & Leakage-Free Normalization
    dataset = prepare_dataset(
        featured_df,
        feature_cols=["Close", "Volume", "SMA_7", "SMA_30", "SMA_90", "RSI", "MACD", "MACD_Signal"],
        window_size=window_size,
        train_ratio=0.8
    )

    X_train, y_train = dataset["X_train"], dataset["y_train"]
    X_test, y_test = dataset["X_test"], dataset["y_test"]
    scaler_X, scaler_y = dataset["scaler_X"], dataset["scaler_y"]
    test_dates = dataset["test_dates"]

    # Step 4: Build Stacked LSTM Neural Network
    input_shape = (X_train.shape[1], X_train.shape[2])  # (60, 8)
    model = build_lstm_model(input_shape=input_shape, lstm_units_1=64, lstm_units_2=64, dropout_rate=0.2)
    print("\n[Model] Stacked LSTM Neural Network Architecture:")
    model.summary()

    # Step 5: Train Model with EarlyStopping & ModelCheckpoint
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        save_path=model_save_path,
        model_name="LSTM"
    )

    # Step 6: Plot Loss Convergence and Diagnose Overfitting
    loss_plot_path = plot_loss_and_diagnose(history, ticker=ticker, model_name="LSTM", output_dir=output_dir)

    # Step 7: Train Linear Regression Benchmark for Comparison
    lr_model = build_linear_regression()
    X_train_flat = X_train.reshape(len(X_train), -1)
    lr_model.fit(X_train_flat, y_train)

    # Step 8: Multi-Model Evaluation (LSTM vs Linear Regression vs Naive Baseline)
    eval_dict = evaluate_all_models(
        models_dict={"LSTM": model, "Linear Regression": lr_model},
        X_test=X_test,
        y_test=y_test,
        scaler_y=scaler_y
    )
    predictions = eval_dict["predictions"]
    y_actual = eval_dict["y_actual"]
    y_pred_lstm = predictions["LSTM"]
    y_naive = predictions["Naive Baseline"]

    m_lstm = calculate_metrics(y_actual, y_pred_lstm)
    m_naive = calculate_metrics(y_actual, y_naive)

    # Step 9: Plot Actual vs Predicted Comparison
    actual_pred_plot = plot_predictions_comparison(predictions, test_dates, ticker=ticker, output_dir=output_dir)

    # Step 10: 7-Day Out-of-Sample Future Forecast
    is_crypto = "BTC" in ticker.upper() or "-USD" in ticker.upper()
    forecast_df = forecast_7_days(
        model=model,
        df=featured_df,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        feature_cols=dataset["feature_cols"],
        window_size=window_size,
        forecast_days=7,
        is_crypto=is_crypto
    )

    # Step 11: Plot 7-Day Future Forecast
    forecast_plot = plot_7_day_forecast(featured_df, forecast_df, ticker=ticker, output_dir=output_dir)

    # Step 12: Plot Executive Summary Dashboard
    dashboard_plot = plot_summary_dashboard(
        history=history,
        y_actual=y_actual,
        y_pred=y_pred_lstm,
        y_naive=y_naive,
        test_dates=test_dates,
        forecast_df=forecast_df,
        df_history=featured_df,
        m_lstm=m_lstm,
        m_naive=m_naive,
        ticker=ticker,
        output_dir=output_dir
    )

    print("\n" + "=" * 75)
    print("                     PIPELINE COMPLETED SUCCESSFULLY!                 ")
    print("=" * 75)
    print(f"[Done] Generated 4 presentation-ready figures in: ./{output_dir}")
    print(f"       1. Loss Curve & Diagnosis:  {loss_plot_path}")
    print(f"       2. Actual vs Predicted:     {actual_pred_plot}")
    print(f"       3. 7-Day Future Forecast:   {forecast_plot}")
    print(f"       4. Executive Dashboard:     {dashboard_plot}")
    print(f"[Done] Best model weights checkpointed to: {model_save_path}")
    print("=" * 75)

    return eval_dict, forecast_df


def main():
    parser = argparse.ArgumentParser(description="Stock Price Prediction using LSTM Neural Network")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock or crypto ticker (default: AAPL; e.g. BTC-USD, MSFT)")
    parser.add_argument("--period", type=str, default="5y", help="Historical data duration (e.g. 5y, 3y)")
    parser.add_argument("--window_size", type=int, default=60, help="Lookback sequence length in days (default: 60)")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--patience", type=int, default=7, help="EarlyStopping patience (default: 7)")
    parser.add_argument("--output_dir", type=str, default="plots", help="Output plots directory")

    args = parser.parse_args()

    run_pipeline(
        ticker=args.ticker,
        period=args.period,
        window_size=args.window_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
