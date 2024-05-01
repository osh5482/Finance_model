import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

code = "005930"
stock_data = pd.read_csv(f"csv/1_{code}_data.csv")

plt.figure(figsize=(10, 5))
plt.plot(stock_data["Date"], stock_data["Close"], label="Close Price")
plt.plot(stock_data["Date"], stock_data["MA5"], label="5-Day MA", color="orange")
# plt.plot(stock_data["Date"], stock_data["MA20"], label="20-Day MA", color="yellow")
# plt.plot(stock_data["Date"], stock_data["MA60"], label="60-Day MA", color="ivory")
plt.plot(
    stock_data["Date"],
    stock_data["BB_Upper"],
    color="black",
    label="BB_Upper",
    alpha=0.5,
)
plt.plot(
    stock_data["Date"],
    stock_data["BB_Lower"],
    color="black",
    label="BB_Lower",
    alpha=0.5,
)

plt.title("Stock Price with MACD, RSI, and Moving Average")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()

# Setting date interval to display only once a month
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))

plt.twinx()
plt.plot(
    stock_data["Date"], np.log(stock_data["MACD"]), color="r", label="MACD", alpha=0.2
)
plt.plot(
    stock_data["Date"], np.log(stock_data["RSI"]), color="g", label="RSI", alpha=0.2
)

print(np.log(stock_data["MACD"]))
plt.ylabel("Indicator Value")
plt.legend(loc="upper left")
plt.show()
