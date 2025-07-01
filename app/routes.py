# app/routes.py

from flask import Blueprint, jsonify, request
from app.model.predict import predict_next_move
from app.model.utils import load_crypto_data
from app.trading.bot import execute_trade

main = Blueprint('main', __name__)

# -----------------------
# Health Check Endpoint
# -----------------------
@main.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🚀 Binance Crypto Trading Bot is Running"}), 200


# -----------------------
# Prediction Endpoint
# -----------------------
@main.route("/predict", methods=["GET"])
def predict():
    try:
        symbol = request.args.get('symbol', 'BTC')
        df = load_crypto_data(symbol=symbol)
        prediction = predict_next_move(df)

        return jsonify({
            "symbol": symbol,
            "prediction": prediction
        }), 200

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# -----------------------
# Manual Trade Trigger
# -----------------------
@main.route("/trade", methods=["POST"])
def trade():
    try:
        result = execute_trade()
        return jsonify({"result": result}), 200

    except Exception as e:
        return jsonify({"error": f"Trade execution failed: {str(e)}"}), 500
