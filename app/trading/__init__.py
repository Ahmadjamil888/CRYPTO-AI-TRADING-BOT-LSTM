# app/trading/__init__.py

"""
Initialize the trading module.
This file can expose key trading functions to the rest of the app.
"""

from .bot import execute_trade
from .binance_api import get_binance_client
