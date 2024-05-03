import pandas as pd
import ta
import json

with open("kospi200_close.json", "r") as f:
    data = json.load(f)

for i, key in enumerate(data.keys()):
    stock_data = data[key]
    stock_data = pd.DataFrame(stock_data)
    stock_data["Date"] = pd.to_datetime(
        stock_data["Date"]
    )  # Convert "Date" column to datetime
    stock_data = stock_data.set_index("Date")

    stock_data["return"] = stock_data["return"] = (
        stock_data["Close"].shift(-1) / stock_data["Close"]
    ) - 1
    stock_data["return"] = stock_data["return"].shift(-1)

    stock_data["profit"] = 0
    stock_data.loc[stock_data["return"] > 0, "profit"] = 1

    macd_indicator = ta.trend.macd_diff(stock_data["Close"])

    # Example: Relative Strength Index (RSI)
    rsi_indicator = ta.momentum.rsi(stock_data["Close"])

    # Calculate 20-day moving average
    ma5 = stock_data["Close"].rolling(window=5).mean()
    ma20 = stock_data["Close"].rolling(window=20).mean()
    ma60 = stock_data["Close"].rolling(window=60).mean()

    # Calculate Bollinger Bands
    bb_upper = ta.volatility.bollinger_hband(stock_data["Close"])
    bb_lower = ta.volatility.bollinger_lband(stock_data["Close"])

    # Combine indicators with the original data
    stock_data["MACD"] = macd_indicator
    stock_data["RSI"] = rsi_indicator
    stock_data["MA5"] = ma5
    stock_data["MA20"] = ma20
    stock_data["MA60"] = ma60

    # Add Bollinger Bands to the DataFrame
    stock_data["BB_Upper"] = bb_upper
    stock_data["BB_Lower"] = bb_lower

    # print(stock_data.tail())
    stock_data.to_csv(f"csv/{i+1}_{key}_data.csv", mode="w", index=True)
    print(f"{i+1}번째 데이터 ({key}) 저장 완료")
