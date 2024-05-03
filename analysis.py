import pandas as pd
import ta
import json
import numpy as np

saved_data = 0
volume_zero_data = 0
small_data = 0

# json파일 로드 후 dict로 변환
with open("kospi200_2023-10-01_2024-04-30.json", "r") as f:
    data = json.load(f)

# dict의 values를 csv에 넣을 수 있도록 데이터프레임의 형태로 수정
for i, key in enumerate(data.keys()):
    # 특정 종목 데이터 로드
    stock_data = data[key]

    name = stock_data["Name"]
    del stock_data["Name"]

    stock_data = pd.DataFrame(stock_data)
    stock_data["Date"] = pd.to_datetime(
        stock_data["Date"]
    )  # Convert "Date" column to datetime
    stock_data = stock_data.set_index("Date")

    # 데이터 갯수가 200개 이하의 경우
    if len(stock_data) <= 60:
        print(f"{i+1}번째 좀목: {name} : 데이터의 수가 200개 이하입니다.")
        small_data += 1
        continue  # 다음 종목으로 넘어감

    # 거래량이 0인 데이터 체크 (3일 연속 0인 경우 확인)
    volume_zero_count = (stock_data["Volume"] == 0).astype(
        int
    )  # 거래량이 0이면 1로, 아니면 0으로 변환
    volume_zero_count_rolling = volume_zero_count.rolling(
        window=3
    ).sum()  # 3일간의 합 계산

    if volume_zero_count_rolling.max() >= 3:
        print(f"{i+1}번째 좀목: {name} : 3일 이상 거래량이 0입니다.")
        volume_zero_data += 1
        continue  # 다음 종목으로 넘어감

    # 어제 대비 주가 수익률
    stock_data["Return"] = (stock_data["Close"].shift(-1) / stock_data["Close"]) - 1
    stock_data["Return"] = stock_data["Return"].shift(1)

    macd_indicator = ta.trend.macd_diff(stock_data["Close"])
    rsi_indicator = ta.momentum.rsi(stock_data["Close"])

    ma5 = stock_data["Close"].rolling(window=5).mean()
    ma20 = stock_data["Close"].rolling(window=20).mean()
    ma60 = stock_data["Close"].rolling(window=60).mean()

    bb_upper = ta.volatility.bollinger_hband(stock_data["Close"])
    bb_lower = ta.volatility.bollinger_lband(stock_data["Close"])

    # Combine indicators with the original data
    stock_data["MACD"] = macd_indicator
    stock_data["RSI"] = rsi_indicator
    stock_data["MA5"] = ma5
    stock_data["MA20"] = ma20
    stock_data["MA60"] = ma60
    stock_data["BB_Upper"] = bb_upper
    stock_data["BB_Lower"] = bb_lower
    stock_data["UpDown"] = (stock_data["Return"] > 0).astype(int)

    # NaN값 제외를 위한 슬라이싱
    stock_data = stock_data[60:-2]

    # 정리된 데이터프레임 csv로 저징
    stock_data.to_csv(f"recent_data/{(saved_data+1):03}_{name}_{key}.csv")
    print(f"{saved_data+1}번째 좀목: {name} 저장 완료")
    saved_data += 1

print()
print(f"200개의 종목 중 {saved_data}개의 종목을 저장했습니다.")
print(f"데이터 부족으로 인해 누락시킨 종목 수: {small_data}개")
print(f"거래량 이슈로 인해 누락시킨 종목 수: {volume_zero_data}개")
