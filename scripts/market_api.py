import yfinance as yf
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 🔥 Allow frontend connection (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Market API running 🚀"}

TRENDING = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

def calculate_change(df, period):
    try:
        return round(((df["Close"].iloc[-1] - df["Close"].iloc[-period]) / df["Close"].iloc[-period]) * 100, 2)
    except:
        return 0

@app.get("/stocks")
def get_stocks(symbol: str = Query(None)):
    
    symbols = [symbol] if symbol else TRENDING
    result = []

    for sym in symbols:
        stock = yf.Ticker(sym)
        df = stock.history(period="7d", interval="1h")

        if df.empty:
            continue

        latest_price = round(df["Close"].iloc[-1], 2)

        data = {
            "symbol": sym,
            "price": latest_price,
            "change_1h": calculate_change(df, 1),
            "change_4h": calculate_change(df, 4),
            "change_8h": calculate_change(df, 8),
            "change_24h": calculate_change(df, 24),
            "change_1w": calculate_change(df, 24*5),
            "volume": int(df["Volume"].iloc[-1])
        }

        result.append(data)

    return result