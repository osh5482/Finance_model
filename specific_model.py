# 필요한 라이브러리를 가져옵니다.
import glob
import json
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

tf.random.set_seed(42)


def create_models(file: str):
    stock_data = pd.read_csv(file)
    file = file[4:-4]
    idx, name, code = file.split("_")

    cols = [
        # "Open",
        # "High",
        # "Low",
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

    # 새로운 데이터프레임 생성 및 변수형변환
    stock_data = stock_data[cols].astype(float)

    # 데이터 정규화
    scaler = StandardScaler()
    scaler = scaler.fit(stock_data)
    stock_data_scaled = scaler.transform(stock_data)

    # train 데이터와 test 데이터로 분할
    n_train = int(0.9 * stock_data_scaled.shape[0])
    train_data_scaled = stock_data_scaled[:n_train]
    test_data_scaled = stock_data_scaled[n_train:]

    # LSTM에 필요한 데이터 형식으로 재구성
    pred_days = 1
    seq_len = 30
    input_dim = len(cols)  # 새로운 input dimension

    trainX, trainY, testX, testY = [], [], [], []

    for i in range(seq_len, len(train_data_scaled) - pred_days + 1):
        trainX.append(train_data_scaled[i - seq_len : i, :])
        trainY.append(train_data_scaled[i + pred_days - 1 : i + pred_days, 0])

    for i in range(seq_len, len(test_data_scaled) - pred_days + 1):
        testX.append(test_data_scaled[i - seq_len : i, :])
        testY.append(test_data_scaled[i + pred_days - 1 : i + pred_days, 0])

    trainX, trainY = np.array(trainX), np.array(trainY)
    testX, testY = np.array(testX), np.array(testY)

    # LSTM 모델 구축
    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_len, input_dim), return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(1))  # 예측하고자 하는 값은 'Close' 가격이므로 Dense(1) 사용

    model.compile(optimizer=Adam(learning_rate=0.01), loss="mse")

    # 모델이 있으면 불러오고 예측, 없으면 새로운 학습 시작
    try:
        model = keras.models.load_model(f"keras_models/{name}.keras")
        print(f"{name}모델이 이미 존재합니다.")

    except:
        history = model.fit(
            trainX,
            trainY,
            epochs=32,
            batch_size=256,
            validation_split=0.1,
            verbose=1,
        )
        # 훈련 후 모델 저장
        model.save(f"keras_models/{name}.keras")
        loss_data = history.history
        with open(f"models_loss/{name}.json", "w") as f:
            json.dump(loss_data, f)

        print(f"{name}모델 및 loss 저장 완료")

        # plt.plot(history.history["loss"], label="Training loss")
        # plt.plot(history.history["val_loss"], label="Validation loss")
        # plt.legend()
        # plt.show()


def main():
    file_faths = glob.glob("csv/*.csv")
    for file in file_faths:
        create_models(file)


if __name__ == "__main__":
    main()
