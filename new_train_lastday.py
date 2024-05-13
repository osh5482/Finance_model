import csv
import datetime
import glob
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import matplotlib.pyplot as plt


model = keras.models.load_model("keras_models/000_KS200_000000.keras")
file = "csv/000_KS200_000000.csv"
stock_data = pd.read_csv(file)


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

history = model.fit(
    trainX,
    trainY,
    epochs=32,
    batch_size=256,
    validation_split=0.1,
    verbose=2,
)

# 훈련 후 모델 저장
model.save(f"keras_models/000_KS200_000000.keras")
