# app/model/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import numpy as np
from sklearn.metrics import accuracy_score
from app.model.utils import load_crypto_data

# ---------------------
# LSTM Model
# ---------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # No sigmoid here → we use BCEWithLogitsLoss
        return out

# ---------------------
# Data Preparation
# ---------------------
def prepare_sequences(df, sequence_length=50, threshold=0.5):
    """
    Create input sequences and binary targets (1 if price goes up significantly).
    """
    features = ['open', 'high', 'low', 'close', 'volume', 'ema', 'rsi']
    df = df[features].copy()
    df = (df - df.mean()) / df.std()

    data = df.values
    X, y = [], []

    for i in range(len(data) - sequence_length - 1):
        X.append(data[i:i + sequence_length])
        # Target is: 1 if price goes up more than threshold
        delta = data[i + sequence_length][3] - data[i + sequence_length - 1][3]
        y.append([1.0 if delta > threshold else 0.0])

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ---------------------
# Training Function
# ---------------------
def train_model():
    print("📊 Loading and preparing data...")
    df = load_crypto_data()
    X, y = prepare_sequences(df)

    # Split dataset
    total_size = len(X)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    dataset = TensorDataset(X, y)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    model = LSTMModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("🧠 Training model...")
    for epoch in range(30):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            output = model(batch_X)
            loss = criterion(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation accuracy
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_output = model(val_X)
                preds = (torch.sigmoid(val_output) > 0.5).float()
                all_preds.extend(preds.numpy())
                all_labels.extend(val_y.numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch [{epoch+1}/30] Loss: {total_loss:.4f} | Val Accuracy: {acc:.2%}")

    # Save model
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/trading_model.pth")
    print("✅ Model saved to model/trading_model.pth")

# ---------------------
# Entry Point
# ---------------------
if __name__ == "__main__":
    train_model()
