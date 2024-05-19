# 필요한 라이브러리를 가져옵니다.
import glob
import json
import time
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

tf.random.set_seed(42)


def create_models(file: str):
    stock_data = pd.read_csv(file)
    SnP_data = pd.read_csv("csv/000_SnP500_000000.csv")
    file = file[4:-4]
    idx, name, code = file.split("_")

    # stock_data["SnP"] = SnP_data["Close"]
    stock_data = pd.merge(
        stock_data, SnP_data[["Date", "Return"]], on="Date", how="left"
    )
    stock_data = stock_data.rename(
        columns={"Return_y": "Return_SnP", "Return_x": "Return"}
    )
    stock_data["Return_SnP"] = stock_data["Return_SnP"].fillna(method="ffill")

    cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Change",
        "Return",
        "MACD",
        "RSI",
        "MA5",
        "MA20",
        "MA60",
        "BB_Upper",
        "BB_Lower",
        "Return_SnP",
        "ans",
    ]

    # 새로운 데이터프레임 생성 및 변수형변환
    stock_data = stock_data[cols].astype(float)
    print(stock_data)

    # 데이터 정규화
    scaler = StandardScaler()
    scaler = scaler.fit(stock_data)
    stock_data_scaled = scaler.transform(stock_data)

    # LSTM에 필요한 데이터 형식으로 재구성
    pred_days = 1
    seq_len = 14
    input_dim = len(cols) - 1  # 새로운 input dimension

    trainX, trainY = [], []

    for i in range(seq_len, len(stock_data_scaled) + 1):
        trainX.append(stock_data_scaled[i - seq_len : i, :-1])
        trainY.append(stock_data_scaled[i - 1 : i, -1])

    trainX, trainY = np.array(trainX), np.array(trainY)
    print(trainX.shape, trainY.shape)

    # LSTM 모델 구축
    model = Sequential()
    model.add(Input(shape=(seq_len, input_dim)))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.01), loss="mse")

    history = model.fit(
        trainX,
        trainY,
        epochs=32,
        batch_size=512,
        validation_split=0.1,
        verbose=2,
    )
    # 훈련 후 모델 저장
    model.save(f"keras_models/{idx}_{name}_{code}_add_SnP.keras")

    print(f"{idx}_{name}_{code} 모델 저장 완료")


def main():
    start = time.perf_counter()
    file_faths = glob.glob("csv/000_KS200_000000.csv")
    for file in file_faths:
        create_models(file)
    end = time.perf_counter()
    t = end - start
    m = int(t // 60)
    s = t % 60
    avg_t = t / len(file_faths)
    avg_m = int(avg_t // 60)
    avg_s = avg_t % 60
    print(f"총 소요 시간: {m}분 {s:.02f}초")
    print(f"모델당 평균 학습시간: {avg_m}분 {avg_s:.02f}초")


if __name__ == "__main__":
    main()
