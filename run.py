# run.py

"""
🚀 Main entry point for the Binance Crypto Trading Bot.

This script:
1. Automatically starts the trading bot once at launch
2. Launches the Flask API server for:
   - GET  /              → Health check
   - POST /predict       → Make prediction
   - (optional) /trade   → Trigger a trade manually
"""

import os
from app import create_app
from app.trading.bot import execute_trade


def main():
    print("⚙️ Starting Binance trading bot...")

    # Optional: perform one trade at startup
    try:
        execute_trade()
    except Exception as e:
        print(f"❌ Trade failed at startup: {e}")

    print("🌐 Launching Flask API server...")

    # Start Flask app (in main thread to avoid signal error)
    app = create_app()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() in ["1", "true", "yes"]
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    main()
