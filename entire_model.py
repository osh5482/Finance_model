# 필요한 라이브러리를 가져옵니다.
import glob
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import time

tf.random.set_seed(42)


def create_models(file_paths: list):

    cols = [
        "Open",
        "Low",
        "High",
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
        "ans",
    ]

    # LSTM에 필요한 데이터 형식으로 재구성
    pred_days = 1
    seq_len = 14
    input_dim = len(cols) - 1  # 새로운 input dimension

    # LSTM 모델 구축
    model = Sequential()
    model.add(Input(shape=(seq_len, input_dim)))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.01), loss="mse")
    model.summary()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",  # 모니터링 대상은 'val_loss'
        patience=8,  # 'val_loss'가 8번 연속으로 개선되지 않을 때 학습 중단
        verbose=2,  # 진행 상황 출력
    )

    for file in file_paths:
        stock_data = pd.read_csv(file)
        file = file[4:-4]
        idx, name, code = file.split("_")

        if idx == "000":
            continue

        # 새로운 데이터프레임 생성 및 변수형 변환
        stock_data = stock_data[cols].astype(float)

        # 데이터 정규화
        scaler = StandardScaler()
        scaler = scaler.fit(stock_data)
        stock_data_scaled = scaler.transform(stock_data)

        trainX, trainY = [], []

        for i in range(seq_len, len(stock_data_scaled) + 1):
            trainX.append(stock_data_scaled[i - seq_len : i, :-1])
            trainY.append(stock_data_scaled[i - 1 : i, -1])

        # print(len(trainX))
        # print(len(trainY))
        trainX, trainY = np.array(trainX), np.array(trainY)

        print(
            f"{idx}번째 종목 {name} 학습 데이터: X = {trainX.shape}, Y = {trainY.shape}"
        )

        history = model.fit(
            trainX,
            trainY,
            epochs=32,
            batch_size=256,
            validation_split=0.1,
            shuffle=False,
            verbose=1,
            # callbacks=[early_stopping],
        )

        print(f"{idx}번째 종목 {name} ({code}) 학습 완료")

        # 훈련 후 모델 저장
    model.save(f"keras_models/000_entire_close.keras")
    print(f"모델 저장 완료")


def main():
    start = time.perf_counter()
    file_faths = glob.glob("csv/*.csv")
    create_models(file_faths)
    end = time.perf_counter()
    sec = end - start
    m = int(sec // 60)
    sec = sec % 60
    print(f"학습 소요 시간: {m}분 {sec:.2f}초")


if __name__ == "__main__":
    main()
