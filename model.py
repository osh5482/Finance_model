import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def create_sequences(data, sequence_length):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(
            data[i : i + sequence_length, :-1]
        )  # Take all features except the target
        y.append(data[i + sequence_length, -1])  # The target value
    return np.array(x), np.array(y)


file_paths = glob.glob("./csv/*.csv")
num_features = 5  # Close, Volume, MA, MACD, RSI

model = Sequential(
    [
        LSTM(64, return_sequences=True, input_shape=(None, num_features), dropout=0.2),
        LSTM(128, return_sequences=True, dropout=0.2),
        LSTM(64, return_sequences=False),
        Dense(1),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

scaler = MinMaxScaler(feature_range=(0, 1))

for i, file in enumerate(file_paths):
    stock_data = pd.read_csv(file)
    # 데이터에 대한 정제작업 진행 부분이 필요함
    x_data = stock_data[["Close", "Volume", "MA5", "MACD", "RSI"]].fillna(0).values
    y_data = stock_data[["Close"]].shift(-1).fillna(0).values

    if len(y_data) < 100:
        continue

    # Concatenate x_data and y_data to apply scaling
    dataset_combined = np.hstack((x_data, y_data))
    dataset_scaled = scaler.fit_transform(dataset_combined)

    sequence_length = 30  # 과거 30일 데이터를 사용하여 내일의 종가를 예측합니다.
    x, y = create_sequences(dataset_scaled, sequence_length)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=32,
        batch_size=256,
        validation_data=(x_test, y_test),
        verbose=2,
    )
    print(f"{i+1}번째 종목 ({file}) 학습 완료")

model.save("Fianance_model.keras")
