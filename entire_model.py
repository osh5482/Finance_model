# 필요한 라이브러리를 가져옵니다.
import glob
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
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
    ]

    # LSTM에 필요한 데이터 형식으로 재구성
    pred_days = 1
    seq_len = 30
    input_dim = len(cols)  # 새로운 input dimension

    # LSTM 모델 구축
    model = Sequential()
    model.add(Input(shape=(seq_len, input_dim)))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(62, return_sequences=False))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.01), loss="mse")

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",  # 모니터링 대상은 'val_loss'
        patience=5,  # 'val_loss'가 5번 연속으로 개선되지 않을 때 학습 중단
        verbose=2,  # 진행 상황 출력
    )

    for file in file_paths:
        stock_data = pd.read_csv(file)
        file = file[4:-4]
        idx, name, code = file.split("_")

        # 새로운 데이터프레임 생성 및 변수형 변환
        stock_data = stock_data[cols].astype(float)

        # 데이터 정규화
        scaler = StandardScaler()
        scaler = scaler.fit(stock_data)
        train_data_scaled = scaler.transform(stock_data)

        trainX, trainY = [], []

        for i in range(seq_len, len(train_data_scaled) - pred_days + 1):
            trainX.append(train_data_scaled[i - seq_len : i, :])
            trainY.append(train_data_scaled[i + pred_days - 1 : i + pred_days, 3])

        trainX, trainY = np.array(trainX), np.array(trainY)
        # testX, testY = np.array(testX), np.array(testY)

        print(f"{name} 학습 데이터: X = {trainX.shape}, Y = {trainY.shape}")

        history = model.fit(
            trainX,
            trainY,
            epochs=32,
            batch_size=128,
            validation_split=0.1,
            shuffle=False,
            verbose=1,
            # callbacks=[early_stopping],
        )

        print(f"{idx}번째 종목 {name} ({code}) 학습 완료")

        # 훈련 후 모델 저장
    model.save(f"keras_models/000_entire_model_all.keras")
    print(f"모델 저장 완료")


def main():
    start = time.perf_counter()
    file_faths = glob.glob("csv/*.csv")
    create_models(file_faths)
    end = time.perf_counter()
    sec = round(end - start, 3)
    m = sec // 60
    sec = sec % 60
    print(f"학습 소요 시간: {m}분 {sec}초")


if __name__ == "__main__":
    main()
