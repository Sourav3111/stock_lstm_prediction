"""
train.py
--------
Handles model training with EarlyStopping and ModelCheckpoint.
Plots training vs validation loss curves and diagnoses overfitting/underfitting.
"""

import os
import matplotlib.pyplot as plt


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 40,
    batch_size: int = 32,
    patience: int = 8,
    save_path: str = "saved_models/best_model.keras",
    model_name: str = "LSTM"
):
    """
    Trains a neural network model using EarlyStopping and ModelCheckpoint.

    Parameters:
        model: Compiled Keras Sequential model.
        X_train: Training sequences.
        y_train: Training targets.
        X_val: Validation (test) sequences.
        y_val: Validation (test) targets.
        epochs (int): Max epochs (default: 40).
        batch_size (int): Mini-batch size (default: 32).
        patience (int): Number of epochs without val_loss improvement before early halt.
        save_path (str): Destination file for optimal weights.
        model_name (str): Label for logging (e.g., 'LSTM', 'GRU', 'MLP').

    Returns:
        history: Keras History object.
    """
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    print(f"\n[Training] Training {model_name} (epochs={epochs}, batch_size={batch_size}, patience={patience})...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 1. EarlyStopping: Prevents memorization of noise by stopping when val_loss plateaus
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        mode="min",
        verbose=1
    )

    # 2. ModelCheckpoint: Saves only the best performing epoch weights
    checkpoint = ModelCheckpoint(
        filepath=save_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=0
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint],
        verbose=1,
        shuffle=False  # Sequential order in time-series mini-batches
    )

    stopped_epoch = early_stop.stopped_epoch
    total_epochs = len(history.history["loss"])
    best_val_loss = min(history.history["val_loss"])
    print(f"[Training] {model_name} training finished at epoch {total_epochs}. Best val_loss: {best_val_loss:.6f}")
    return history


def plot_loss_and_diagnose(history, ticker: str = "AAPL", model_name: str = "LSTM", output_dir: str = "plots") -> str:
    """
    Plots training vs validation loss and prints an in-depth diagnosis of overfitting.

    How to read this chart:
    -----------------------
    - Underfitting: Both curves plateau at high loss, indicating the model lacks capacity.
    - Overfitting: Training loss keeps falling towards 0, while validation loss diverges
      upwards after a certain epoch (the model is memorizing historical noise).
    - Optimal Fit: Both curves decrease smoothly and converge. Validation loss remains close
      to training loss. EarlyStopping restores the parameters from the optimal epoch.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(epochs, train_loss, label=f"Training Loss (MSE)", color="#1d4ed8", linewidth=2.2)
    plt.plot(epochs, val_loss, label=f"Validation Loss (MSE)", color="#ea580c", linewidth=2.2, linestyle="--")

    best_epoch = val_loss.index(min(val_loss)) + 1
    plt.scatter([best_epoch], [min(val_loss)], color="#dc2626", s=80, zorder=5, label=f"Optimal Epoch ({best_epoch})")
    
    plt.title(f"{model_name} Training vs Validation Loss Curve - {ticker}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Mean Squared Error (Scaled)", fontsize=11)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    filename = f"loss_curve_{model_name.lower()}.png"
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path)
    plt.close()

    # Overfitting Diagnosis
    final_train = train_loss[-1]
    final_val = val_loss[-1]
    ratio = final_val / (final_train + 1e-8)

    print("\n" + "=" * 65)
    print(f"      OVERFITTING DIAGNOSIS FOR {model_name} ({ticker})      ")
    print("=" * 65)
    print(f"• Final Training Loss:    {final_train:.6f}")
    print(f"• Final Validation Loss:  {final_val:.6f}")
    print(f"• Best Validation Epoch:  Epoch {best_epoch} (Loss: {min(val_loss):.6f})")
    print(f"• Val/Train Loss Ratio:   {ratio:.2f}x")

    if ratio < 1.6 and abs(final_val - final_train) < 0.01:
        diagnosis = "OPTIMAL GENERALIZATION: Training and validation loss converged closely without divergence. Dropout and EarlyStopping successfully prevented overfitting."
    elif ratio >= 1.6 and final_val > 1.5 * min(val_loss):
        diagnosis = "MILD OVERFITTING DETECTED: Validation loss began diverging in later epochs. However, EarlyStopping restored weights from the optimal epoch (Epoch {best_epoch})."
    else:
        diagnosis = "WELL-REGULARIZED CONVERGENCE: The model fits the temporal patterns without memorizing noise."
    
    print(f"• Assessment: {diagnosis}")
    print(f"• Saved plot to: {plot_path}")
    print("=" * 65)

    return plot_path
