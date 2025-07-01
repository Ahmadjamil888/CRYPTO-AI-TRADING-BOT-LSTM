
# Crypto AI Trading Bot using LSTM

This project is a deep learning-based cryptocurrency trading bot built using Long Short-Term Memory (LSTM) neural networks. It is designed to analyze historical crypto market data and predict the next move in price direction, enabling algorithmic trading strategies based on AI-powered forecasts.

## Features

- LSTM-based model for time-series prediction
- Real-time or batch prediction support
- Data preprocessing pipeline for crypto price history
- Integration-ready for trading execution (buy/sell) via APIs
- Modular and extensible codebase

## Technologies Used

- Python (80.1%)
- HTML (19.9%)
- TensorFlow / Keras or PyTorch (depending on version)
- NumPy, Pandas, Matplotlib
- Flask (for frontend display and interaction)
- Git & GitHub for version control

## File Structure

```
crypto-ai-trading-bot/
│
├── model/                 # LSTM model training and prediction
│   ├── train.py
│   └── predict.py
│
├── data/                  # Datasets and processing scripts
│   └── fetch_data.py
│
├── server/                # Backend API (Flask)
│   ├── app.py
│   └── routes.py
│
├── client/                # Frontend (HTML/CSS/JS or framework)
│   └── index.html
│
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md              # Project documentation
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Ahmadjamil888/CRYPTO-AI-TRADING-BOT-LSTM.git
cd CRYPTO-AI-TRADING-BOT-LSTM
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Set up your `.env` or config file for API keys and trading platform integration.

## Usage

To train or test the model:

```bash
python model/train.py        # Train the LSTM model
```
## Run the app

```
python run.py
```
Then open your browser and navigate to `http://localhost:5000`.
<img src="https://raw.githubusercontent.com/Ahmadjamil888/CRYPTO-AI-TRADING-BOT-LSTM/refs/heads/main/Screenshot%202025-07-01%20163115.png">
To run the web app:

Run the index.html file 
<img src="https://raw.githubusercontent.com/Ahmadjamil888/CRYPTO-AI-TRADING-BOT-LSTM/refs/heads/main/Screenshot%202025-07-01%20162959.png">


## Future Improvements

- Integration with Binance or Coinbase API for live trading
- Improved accuracy with hybrid models (LSTM + CNN)
- Dashboard with trading analytics and logs
- Stop-loss and profit-lock strategy implementation

## License

This project is open-source and available under the [MIT License](LICENSE).
