import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler
from config import Config
import matplotlib.pyplot as plt
import os

def prepare_sequential_data(df, sequence_length=Config.SEQUENCE_LENGTH):
    print(f"Preparing sequential data with lookback of {sequence_length} candles...")
    target = df['Target'].values
    features = df.drop(['Target'], axis=1).values
    
    # Scale features using RobustScaler (immune to outliers like long candle shadows)
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)
    
    X, y = [], []
    for i in range(len(features_scaled) - sequence_length):
        X.append(features_scaled[i:i+sequence_length])
        y.append(target[i+sequence_length])
        
    X = np.array(X)
    y = np.array(y)
    
    # 80-20 split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    print(f"Training data shape: X={X_train.shape}, Y={y_train.shape}")
    print(f"Testing data shape: X={X_test.shape}, Y={y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler

def build_lstm_model(input_shape):
    print("Building LSTM model architecture...")
    model = Sequential([
        # Layer 1
        LSTM(100, return_sequences=True, input_shape=input_shape),
        # Layer 2
        Dropout(0.2),
        # Layer 3
        LSTM(50, return_sequences=False),
        # Layer 4
        Dense(25, activation='relu'),
        # Output Layer
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # EarlyStopping to prevent memorizing the data
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )
    
    print("Starting Model Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30, # Moderate epochs, early stop will catch
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )
    
    print("Evaluating Model on Test Data...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"FINAL TEST ACCURACY: {accuracy*100:.2f}%")
    print(f"FINAL TEST LOSS: {loss:.4f}")
    
    # Plotting Curves
    plt.figure(figsize=(14, 5))
    
    # Accuracy Curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss Curve
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(os.getcwd(), 'training_curves.png')
    plt.savefig(plot_path)
    print(f"\nSaved Training Curves to: {plot_path}")
    
    return model, history, accuracy
