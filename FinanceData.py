import json
import FinanceDataReader as fdr
import pandas as pd


kospi = fdr.StockListing("KOSPI")
kospi200 = kospi.truncate(after=199)
# print(kospi200)
stock_code = kospi200["Code"].to_list()
# print(stock_code)

kospi200_dict = {}
for i, code in enumerate(stock_code):
    data = {}
    df = fdr.DataReader(code, "2020-01-01", "2023-12-31")
    # print(df)

    df.index = df.index.strftime("%Y-%m-%d")
    date = df.index.to_list()
    open_ = df["Open"].to_list()
    close = df["Close"].to_list()
    low = df["Low"].to_list()
    high = df["High"].to_list()
    volume = df["Volume"].to_list()
    change = df["Change"].fillna(0).to_list()

    data["Date"] = date
    data["Open"] = open_
    data["High"] = high
    data["Low"] = low
    data["Close"] = close
    data["Volume"] = volume
    data["Change"] = change

    kospi200_dict[code] = data
    print(f"{i+1}번째 종목 ({code}) 데이터 저장 완료")

# print(kospi200_dict)

with open("kospi200_close.json", "w") as f:
    json.dump(kospi200_dict, f)
    print("json 파일 저장 완료")
