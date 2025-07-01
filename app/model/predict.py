# app/model/predict.py

import torch
import torch.nn as nn
import pandas as pd
import os
from app.model.utils import compute_rsi, load_crypto_data

# ---------------------------
# Define LSTM Model (same as in training)
# ---------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out  # raw logits (used with BCEWithLogitsLoss)

# ---------------------------
# Prepare Input Data
# ---------------------------
def prepare_input_data(df, sequence_length=50):
    # Select and normalize relevant features
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    df['ema'] = df['close'].ewm(span=14).mean()
    df['rsi'] = compute_rsi(df['close'])

    df.dropna(inplace=True)
    df = (df - df.mean()) / df.std()

    latest_seq = df[-sequence_length:].values
    input_tensor = torch.tensor(latest_seq, dtype=torch.float32).unsqueeze(0)  # shape: (1, 50, 7)
    return input_tensor

# ---------------------------
# Predict Next Move
# ---------------------------
def predict_next_move(df):
    model_path = 'model/trading_model.pth'

    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Trained model not found. Please run train_model.py first.")

    input_tensor = prepare_input_data(df)

    model = LSTMModel(input_size=7, hidden_size=128, num_layers=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor).item()

    prediction = torch.sigmoid(torch.tensor(output)).item()
    direction = "UP" if prediction >= 0.5 else "DOWN"

    return {
        "raw_output": prediction,
        "direction": direction
    }
