"""
LSTM Model — Elite v3.2
========================
التغييرات عن v3.0:

المشكلة الأصلية:
  - Architecture كبير جداً (128→64 LSTM) مع 106 features → Overfitting مضمون
  - Train accuracy 75% / Val accuracy 50% = حافظ على data مش بيتعلم
  - Dropout منخفض جداً (0.3, 0.2, 0.1)
  - EarlyStopping بيقف في Epoch 6 دايماً = مش بيتعلم كافي

الحلول في v3.2:
  1. Feature Reduction قبل الـ LSTM (PCA-style selection لأهم 40 feature)
  2. Architecture أصغر وأعمق بـ Regularization أقوى
  3. Dropout أعلى (0.5, 0.4, 0.3)
  4. L2 Regularization على الـ LSTM layers
  5. EarlyStopping patience أعلى (10 بدل 5)
  6. Gradient Clipping لمنع exploding gradients
  7. حذف الـ Baseline comparison عشان يوفر وقت ومش مفيد
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from config import Config
import matplotlib
matplotlib.use('Agg')
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

# عدد الـ features بعد الـ selection
# v3.2: 40 features → v3.3: 25 features
# أقل features = أقل overfitting + الموديل بيركز على أهم signals بس
TOP_K_FEATURES = 25


# ─────────────────────────────────────────
# FEATURE SELECTION
# بيختار أهم K feature بدل ما يحط الـ 106 كلها
# ─────────────────────────────────────────

def select_top_features(X_flat, y, k=TOP_K_FEATURES):
    """
    يختار أهم K feature بناءً على F-score مع الـ target.
    X_flat: (n_samples, n_features) — آخر timestep بس للـ selection
    y: (n_samples,)
    Returns: selector object + selected feature indices
    """
    selector = SelectKBest(f_classif, k=min(k, X_flat.shape[1]))
    selector.fit(X_flat, y)
    selected_indices = selector.get_support(indices=True)
    logger.info("[FeatureSelect] Selected %d/%d features via F-score.",
                len(selected_indices), X_flat.shape[1])
    return selector, selected_indices


def apply_feature_selection(X_seq, selected_indices):
    """
    يطبق الـ feature selection على الـ sequence data.
    X_seq: (n_samples, seq_len, n_features)
    Returns: (n_samples, seq_len, k_features)
    """
    return X_seq[:, :, selected_indices]


# ─────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────

def prepare_sequential_data(df, sequence_length=Config.SEQUENCE_LENGTH):
    """
    Prepares sequential data for LSTM training.
    
    v3.3 FIX: Scaler now fits on TRAIN data only (was fitting on ALL data = data leakage).
    
    Pipeline order (leak-free):
    1. Drop rows without Target
    2. Determine train/test split point on RAW data
    3. Fit scaler on train portion ONLY
    4. Transform all data using train statistics
    5. Build sequences
    6. Split sequences at boundary matching the raw split
    7. Feature selection on train sequences only
    """
    logger.info("Preparing sequential data (lookback=%d, top_features=%d)...",
                sequence_length, TOP_K_FEATURES)

    import pandas as pd

    feature_cols = [c for c in df.columns if c != 'Target']

    # Step 1: Keep only rows with valid Target
    df_valid = df.dropna(subset=['Target'])
    target = df_valid['Target'].values
    features = df_valid[feature_cols].values

    # Step 2: Determine split point on RAW data (80% train / 20% test)
    split_raw = int(len(features) * 0.8)
    logger.info("[LeakFix] Raw data: %d rows | Train boundary: row %d | Test: row %d+",
                len(features), split_raw, split_raw)

    # Step 3: Fit scaler on TRAIN portion ONLY (fixes the leakage)
    scaler = RobustScaler()
    scaler.fit(features[:split_raw])

    # Step 4: Transform ALL data using train-derived statistics
    features_scaled = scaler.transform(features)

    # Step 5: Build sequences
    X, y = [], []
    for i in range(len(features_scaled) - sequence_length):
        X.append(features_scaled[i:i + sequence_length])
        y.append(target[i + sequence_length])

    X = np.array(X)   # (n, seq_len, n_features)
    y = np.array(y)

    # Step 6: Split sequences at boundary matching raw split
    # Sequence at index i has target = target[i + sequence_length]
    # First test target should be target[split_raw], so seq_split = split_raw - sequence_length
    seq_split = split_raw - sequence_length
    seq_split = max(1, min(seq_split, len(X) - 1))  # safety clamp

    X_train_raw, X_test_raw = X[:seq_split], X[seq_split:]
    y_train, y_test = y[:seq_split], y[seq_split:]

    logger.info("[LeakFix] Sequences: %d total | Train: %d | Test: %d",
                len(X), len(X_train_raw), len(X_test_raw))

    # Step 7: Feature selection — based on last timestep of training data only
    X_train_last = X_train_raw[:, -1, :]   # آخر candle بس للـ scoring
    selector, selected_indices = select_top_features(X_train_last, y_train)

    # Apply selection to both splits
    X_train = apply_feature_selection(X_train_raw, selected_indices)
    X_test  = apply_feature_selection(X_test_raw,  selected_indices)

    logger.info("After feature selection → X_train=%s, X_test=%s",
                X_train.shape, X_test.shape)

    # Sample weights (بسيطة — class balancing بس)
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weight_map = dict(zip(classes, weights))
    train_weights = np.array([weight_map[yi] for yi in y_train])

    logger.info("Training data: X=%s, Y=%s | Test data: X=%s, Y=%s",
                X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # نحتفظ بالـ selected_indices في الـ scaler object عشان نستخدمه وقت الـ inference
    scaler.selected_indices_ = selected_indices

    return X_train, X_test, y_train, y_test, scaler, train_weights


# ─────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────

def build_lstm_model(input_shape):
    """
    v3.2 Architecture: أصغر + Regularization أقوى

    القديم:  LSTM(128) → LSTM(64) → Dense(32) — كبير جداً، بيحفظ مش بيتعلم
    الجديد:  BiLSTM(48) → LSTM(32) → Dense(16) — أصغر، Dropout أقوى، L2 regularization
    
    Bidirectional LSTM:
    - بيقرأ الـ sequence من الاتجاهين (past → future و future → past)
    - بيلتقط patterns مش ممكن LSTM عادي يشوفها
    - مش بيستخدم future data — بس بيفهم السياق أحسن
    """
    logger.info("Building LSTM v3.2 model (input shape: %s)...", input_shape)

    model = Sequential([
        # Layer 1: Bidirectional LSTM — أصغر بكتير من الأصلي
        Bidirectional(
            LSTM(48,
                 return_sequences=True,
                 kernel_regularizer=l2(1e-4),
                 recurrent_regularizer=l2(1e-4)),
            input_shape=input_shape
        ),
        Dropout(0.5),           # أقوى بكتير من 0.3 الأصلي
        BatchNormalization(),

        # Layer 2: LSTM عادي — أصغر
        LSTM(32,
             return_sequences=False,
             kernel_regularizer=l2(1e-4),
             recurrent_regularizer=l2(1e-4)),
        Dropout(0.4),           # أقوى من 0.2 الأصلي
        BatchNormalization(),

        # Layer 3: Dense — أصغر بكتير
        Dense(16,
              activation='relu',
              kernel_regularizer=l2(1e-4)),
        Dropout(0.3),

        # Output
        Dense(1, activation='sigmoid'),
    ])

    # Gradient clipping — يمنع الـ exploding gradients
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0005,   # أبطأ شوية من 0.001 الأصلي
        clipnorm=1.0            # الجديد — يحد من حجم الـ gradients
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    # طباعة ملخص الـ architecture
    total_params = model.count_params()
    logger.info("[LSTM v3.2] Total parameters: {:,}".format(total_params))

    return model


# ─────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test, sample_weights=None):
    """
    Trains the LSTM model.
    v3.2 changes:
    - patience=10 (كان 5) — بيدي الموديل وقت أكتر يتعلم
    - ReduceLROnPlateau patience=5 (كان 3)
    - حذف الـ baseline comparison — كان بياخد وقت ومش بيفيد
    - batch_size=32 (كان 64) — gradients أدق
    """
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,            # كان 5 — بيدي فرصة أكتر
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001,        # الجديد — يتجاهل تحسينات صغيرة جداً
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,             # كان 3
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    logger.info("Starting LSTM v3.2 training (epochs=50, batch=32)...")
    history = model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        validation_data=(X_test, y_test),
        epochs=50,              # كان 30 — أكتر فرص للتعلم
        batch_size=32,          # كان 64 — أدق
        callbacks=callbacks,
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info("FINAL TEST ACCURACY: %.2f%% | LOSS: %.4f", accuracy * 100, loss)

    # Overfitting diagnosis
    final_train_acc = history.history['accuracy'][-1]
    gap = final_train_acc - accuracy
    overfit_status = "✅ Healthy" if gap < 0.08 else ("⚠️ Mild" if gap < 0.15 else "❌ Severe")

    print(f"\n\033[95m{'='*55}\033[0m")
    print(f"\033[95m        LSTM v3.2 TRAINING REPORT\033[0m")
    print(f"\033[95m{'='*55}\033[0m")
    print(f"\033[95m  Test Accuracy   : {accuracy * 100:.2f}%\033[0m")
    print(f"\033[95m  Train Accuracy  : {final_train_acc * 100:.2f}%\033[0m")
    print(f"\033[95m  Train-Test Gap  : {gap * 100:.2f}% — {overfit_status}\033[0m")
    print(f"\033[95m  Epochs Run      : {len(history.history['loss'])}\033[0m")
    print(f"\033[95m  Features Used   : {X_train.shape[2]} (selected from full set)\033[0m")
    print(f"\033[95m{'='*55}\033[0m\n")

    # Save training curves
    try:
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train', color='#4CAF50', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation', color='#FF9800', linewidth=2)
        plt.title('Model Accuracy (v3.2)')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy')
        plt.legend(); plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train', color='#4CAF50', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation', color='#FF9800', linewidth=2)
        plt.title('Model Loss (v3.2)')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.legend(); plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(os.getcwd(), 'training_curves.png')
        plt.savefig(plot_path, dpi=100)
        plt.close()
        logger.info("Saved training curves to: %s", plot_path)
    except Exception as e:
        logger.warning("Failed to save training curves: %s", e)

    return model, history, accuracy
