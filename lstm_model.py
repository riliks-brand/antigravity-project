"""
LSTM Model — Elite v3.0
========================
Time-series LSTM model with:
- RobustScaler (outlier-immune)
- Loss-pattern weighted training (Probability Modifier)
- Comparative accuracy reports
- Training curve visualization
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from config import Config
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/VPS
import matplotlib.pyplot as plt
import os
import logging

logger = logging.getLogger("LSTM")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler(Config.LOG_FILE, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("\033[96m%(asctime)s\033[0m [%(levelname)s] %(message)s"))
    logger.addHandler(_ch)


def prepare_sequential_data(df, sequence_length=Config.SEQUENCE_LENGTH):
    """
    Prepares sequential data for LSTM training.
    Applies RobustScaler and loss-pattern sample weighting.
    """
    logger.info("Preparing sequential data with lookback of %d candles...", sequence_length)

    import pandas as pd

    # Load trade history for sample weighting
    losses_df = None
    history_file = Config.TRADING_HISTORY_FILE
    if os.path.exists(history_file):
        try:
            losses_df = pd.read_csv(history_file)
            losses_df = losses_df[losses_df.get('pnl', pd.Series(dtype=float)) < 0]
            if not losses_df.empty:
                logger.info("[Weights] Loaded %d loss records from %s.", len(losses_df), history_file)
            else:
                losses_df = None
        except Exception as e:
            logger.warning("[Weights] Error loading history: %s", e)
            losses_df = None

    # Scale features using RobustScaler (immune to outliers)
    scaler = RobustScaler()
    feature_cols = [c for c in df.columns if c != 'Target']
    scaler.fit(df[feature_cols].values)

    # Drop rows where Target is NaN
    df_train = df.dropna(subset=['Target'])
    target = df_train['Target'].values
    features = df_train[feature_cols].values

    features_scaled = scaler.transform(features)

    X, y, sample_weights_list = [], [], []

    for i in range(len(features_scaled) - sequence_length):
        X.append(features_scaled[i:i + sequence_length])
        y.append(target[i + sequence_length])

        weight = 1.0
        if losses_df is not None and not losses_df.empty:
            last_idx = i + sequence_length - 1
            row = df_train.iloc[last_idx]
            rsi_val = row.get('RSI', 50)
            adx_val = row.get('ADX', 25)

            for _, loss_row in losses_df.iterrows():
                loss_rsi = loss_row.get('RSI', 50) if 'RSI' in losses_df.columns else 50
                loss_adx = loss_row.get('ADX', 25) if 'ADX' in losses_df.columns else 25

                if abs(loss_rsi - rsi_val) < 3.0 and abs(loss_adx - adx_val) < 5.0:
                    weight = 1.5  # Penalize patterns similar to past losses
                    break

        sample_weights_list.append(weight)

    X = np.array(X)
    y = np.array(y)
    weights_arr = np.array(sample_weights_list)

    # 80-20 split (chronological, no shuffle)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    train_weights = weights_arr[:split_index]

    logger.info("Training data: X=%s, Y=%s | Test data: X=%s, Y=%s",
                X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # Penalty impact report
    weighted_count = int(np.sum(train_weights > 1.0))
    if weighted_count > 0:
        logger.info("[Weights] Applied 1.5x penalty to %d/%d training samples.",
                    weighted_count, len(train_weights))

    return X_train, X_test, y_train, y_test, scaler, train_weights


def build_lstm_model(input_shape):
    """Builds the LSTM architecture with BatchNorm for stability."""
    logger.info("Building LSTM model (input shape: %s)...", input_shape)

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        BatchNormalization(),

        LSTM(64, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),

        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train_and_evaluate(X_train, X_test, y_train, y_test, sample_weights=None):
    """
    Trains the LSTM model with early stopping and learning rate reduction.
    Includes comparative accuracy analysis.
    """
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    # Comparative baseline (if weights are applied)
    baseline_accuracy = None
    has_weights = sample_weights is not None and np.any(sample_weights != 1.0)

    if has_weights:
        logger.info("[Baseline] Training unweighted model for comparison...")
        baseline_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        baseline_early = EarlyStopping(monitor='val_loss', patience=3,
                                        restore_best_weights=True, verbose=0)
        baseline_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=10,
            batch_size=64,
            callbacks=[baseline_early],
            verbose=0,
        )
        _, baseline_accuracy = baseline_model.evaluate(X_test, y_test, verbose=0)
        logger.info("[Baseline] Unweighted Accuracy: %.2f%%", baseline_accuracy * 100)
        del baseline_model

    # Main training
    logger.info("Starting main model training...")
    history = model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info("FINAL TEST ACCURACY: %.2f%% | LOSS: %.4f", accuracy * 100, loss)

    # Accuracy comparison report
    print(f"\n\033[95m{'='*55}\033[0m")
    print(f"\033[95m       🧠 INTELLIGENCE REPORT: ACCURACY\033[0m")
    print(f"\033[95m{'='*55}\033[0m")
    if has_weights and baseline_accuracy is not None:
        gain = (accuracy - baseline_accuracy) * 100
        emoji = "📈" if gain >= 0 else "📉"
        print(f"\033[95m  WITHOUT Loss Weights : {baseline_accuracy * 100:.2f}%\033[0m")
        print(f"\033[95m  WITH Loss Weights    : {accuracy * 100:.2f}%\033[0m")
        print(f"\033[95m  {emoji} Gain from Learning : {gain:+.2f}%\033[0m")
    else:
        print(f"\033[95m  Accuracy : {accuracy * 100:.2f}%\033[0m")
        print(f"\033[95m  (No loss history to compare)\033[0m")
    print(f"\033[95m{'='*55}\033[0m\n")

    # Save training curves
    try:
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train', color='#4CAF50', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation', color='#FF9800', linewidth=2)
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train', color='#4CAF50', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation', color='#FF9800', linewidth=2)
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(os.getcwd(), 'training_curves.png')
        plt.savefig(plot_path, dpi=100)
        plt.close()
        logger.info("Saved training curves to: %s", plot_path)
    except Exception as e:
        logger.warning("Failed to save training curves: %s", e)

    return model, history, accuracy
