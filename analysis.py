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

    macd_indicator = ta.trend.macd_diff(stock_data["Close"])

    # Example: Relative Strength Index (RSI)
    rsi_indicator = ta.momentum.rsi(stock_data["Close"])

    # Calculate 20-day moving average
    ma5 = stock_data["Close"].rolling(window=5).mean()
    ma20 = stock_data["Close"].rolling(window=20).mean()
    ma60 = stock_data["Close"].rolling(window=60).mean()

    # Combine indicators with the original data
    stock_data["MACD"] = macd_indicator
    stock_data["RSI"] = rsi_indicator
    stock_data["MA5"] = ma5
    stock_data["MA20"] = ma20
    stock_data["MA60"] = ma60

    # print(stock_data.tail())
    stock_data.to_csv(f"csv/{i+1}_{key}_data.csv")
    print(f"{i+1}번째 데이터 ({key}) 저장 완료")


"""
# Visualize the indicators
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.figure(figsize=(10, 5))
plt.plot(stock_data.index, stock_data["Close"], label="Close Price")
plt.plot(stock_data.index, ma5, label="5-Day MA", color="orange")
plt.plot(stock_data.index, ma20, label="20-Day MA", color="yellow")
plt.plot(stock_data.index, ma60, label="60-Day MA", color="ivory")
plt.title("Stock Price with MACD, RSI, and Moving Average")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()

# Setting date interval to display only once a month
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

plt.twinx()
plt.plot(stock_data.index, stock_data["MACD"], color="r", label="MACD")
plt.plot(stock_data.index, stock_data["RSI"], color="g", label="RSI")
plt.ylabel("Indicator Value")
plt.legend(loc="upper left")
plt.show()
"""
