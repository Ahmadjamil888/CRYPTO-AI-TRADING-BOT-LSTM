# app/trading/binance_api.py

import os
from dotenv import load_dotenv
from binance.client import Client

# Load environment variables from .env
load_dotenv()

def get_binance_client():
    """
    Initializes and returns a Binance client using API credentials from .env.
    
    Returns:
        binance.client.Client: Authenticated Binance API client.
    """
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        raise ValueError("Missing Binance API Key or Secret in .env")

    client = Client(api_key, api_secret)
    return client
