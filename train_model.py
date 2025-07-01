# train_model.py

"""
🚀 Standalone script to train the crypto trading model using historical data.

Usage:
    python train_model.py

This script:
- Loads historical crypto data from CSV
- Trains a binary LSTM classifier on price movement (UP/DOWN)
- Saves the trained model to model/trading_model.pth
"""

from app.model.train import train_model

def main():
    print("🔁 Initializing training process...")
    train_model()
    print("✅ Model training completed and saved!")

if __name__ == "__main__":
    main()
