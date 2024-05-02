import datetime
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

pred_days = 1
seq_len = 30
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

scaler = StandardScaler()


def file_process(stock_data: pd.DataFrame):
    """테스트 데이터셋을 만들고 정규화합니다"""

    # 새로운 데이터프레임 생성 및 변수형 변환
    stock_data = stock_data[cols].astype(float)

    # 데이터 정규화
    scaler.fit(stock_data)
    test_data_scaled = scaler.transform(stock_data)

    testX = []

    for i in range(seq_len, len(test_data_scaled) - pred_days + 1):
        testX.append(test_data_scaled[i - seq_len : i, :])
    testX = np.array(testX)
    print(testX.shape)

    return testX


def run_model(testX: np.ndarray):
    """정규화된 데이터를 모델에 입력해 예측값을 출력합니다"""
    model = keras.models.load_model(f"keras_models/000_entire_model_all.keras")
    prediction = model.predict(testX)

    return prediction


def add_data(pred_data: np.ndarray):
    """예측값을 역정규화해 기존 데이터셋에 추가합니다"""
    # 원본 스케일로 역정규화
    print(pred_data.shape)

    dummy = np.zeros((len(pred_data), scaler.scale_.shape[0]))
    dummy[:, 3] = pred_data.reshape(-1)
    prediction_inversed = scaler.inverse_transform(dummy)
    prediction_inversed_df = pd.DataFrame(prediction_inversed, columns=cols)
    return prediction_inversed_df


def plot_df(dates, stock_data: pd.DataFrame, pred_data_inversed_df: pd.DataFrame):
    """실제 데이터와 예상 데이터를 plot"""
    plt.figure(figsize=(16, 9))
    plt.plot(
        dates,
        stock_data["Close"],
        color="green",
        marker=".",
        label="Original Close Price",
    )
    plt.plot(
        dates[seq_len:],
        pred_data_inversed_df["Close"],
        color="blue",
        marker=".",
        linestyle="--",
        label="Predicted Close Price",
    )
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.legend()
    plt.show()
    return


def main():
    file = "169_GS건설_006360"
    file_path = f"recent_data/{file}.csv"
    stock_data = pd.read_csv(file_path)
    idx, name, code = file.split("_")
    dates = pd.to_datetime(stock_data["Date"])

    testX = file_process(stock_data)
    prediction = run_model(testX)
    pred_data_inversed_df = add_data(prediction)

    plot_df(dates, stock_data, pred_data_inversed_df)

    return


if __name__ == "__main__":
    main()
