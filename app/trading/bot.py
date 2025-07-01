# app/trading/bot.py

import torch
import pandas as pd
from app.trading.binance_api import get_binance_client
from app.model.utils import compute_rsi
from app.model.predict import prepare_input_data, LSTMModel

# Configuration
SYMBOL = "BTCUSDT"
QUANTITY = 0.001  # Adjust based on user's risk profile
MODEL_PATH = "model/trading_model.pth"

# -------------------------------------
# Load Trained PyTorch Model
# -------------------------------------
def load_model(path=MODEL_PATH):
    try:
        model = LSTMModel(input_size=7)  # OHLCV + EMA + RSI
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"❌ Model loading failed: {str(e)}")

# -------------------------------------
# Fetch & Process Real-Time Data
# -------------------------------------
def get_latest_data(symbol="BTCUSDT", interval="1m", limit=60):
    client = get_binance_client()
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)

    # Add technical indicators
    df['ema'] = df['close'].ewm(span=14).mean()
    df['rsi'] = compute_rsi(df['close'])
    df.dropna(inplace=True)

    return df

# -------------------------------------
# Execute Prediction & Trade
# -------------------------------------
def execute_trade():
    print("🤖 Running trading decision...")

    try:
        # Load model and data
        model = load_model()
        df = get_latest_data(symbol=SYMBOL)

        if len(df) < 50:
            return {"error": "⚠️ Not enough data to make prediction."}

        # Prepare input tensor
        input_tensor = prepare_input_data(df)

        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor).item()

        direction = "BUY" if prediction >= 0.5 else "SELL"
        current_price = df['close'].iloc[-1]
        print(f"📊 Prediction: {prediction:.4f} → {direction} at ${current_price:.2f}")

        # Execute order on Binance
        client = get_binance_client()
        if direction == "BUY":
            order = client.order_market_buy(symbol=SYMBOL, quantity=QUANTITY)
        else:
            order = client.order_market_sell(symbol=SYMBOL, quantity=QUANTITY)

        print("✅ Trade executed successfully.")
        return order

    except Exception as e:
        print("❌ Trade execution failed:", str(e))
        return {"error": str(e)}
