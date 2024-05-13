import csv
import datetime
import glob
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

pred_days = 1
seq_len = 14
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
    # stock_data = stock_data[-14:]
    # 데이터 정규화
    scaler.fit(stock_data)
    test_data_scaled = scaler.transform(stock_data)

    testX = []

    for i in range(seq_len, len(test_data_scaled) + 1):
        testX.append(test_data_scaled[i - seq_len : i, :])

    testX = np.array(testX)

    return testX


def run_model(testX: np.ndarray):
    """정규화된 데이터를 모델에 입력해 예측값을 출력합니다"""
    model = keras.models.load_model(f"keras_models/000_KS200_000000.keras")
    prediction = model.predict(testX)

    return prediction


def add_data(pred_data: np.ndarray):
    """예측값을 역정규화해 기존 데이터셋에 추가합니다"""
    # 원본 스케일로 역정규화

    dummy = np.zeros((len(pred_data), scaler.scale_.shape[0]))
    dummy[:, 3] = pred_data.reshape(-1)
    prediction_inversed = scaler.inverse_transform(dummy)
    prediction_inversed_df = pd.DataFrame(prediction_inversed, columns=cols)
    print(prediction_inversed_df["Close"])
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
        pred_data_inversed_df["Close"].shift(1),
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


def cal_direction(stock_data: pd.DataFrame, pred_data_inversed_df: pd.DataFrame):
    yesterday_close = stock_data["Close"]
    original_close = stock_data["Close"].shift(-1)
    predict_close = pred_data_inversed_df["Close"].shift(-1)

    # 계산할 값들을 담을 새로운 DataFrame 생성
    result_df = pd.DataFrame()

    # predict_close - yesterday_close & original_close - yesterday_close 계산
    result_df["pred_diff"] = predict_close - yesterday_close
    result_df["orig_diff"] = original_close - yesterday_close
    result_df["diff"] = abs(result_df["pred_diff"] - result_df["orig_diff"])
    result_df = result_df.dropna()

    # 두 차이의 곱이 양수면 1, 음수면 0을 반환하는 새 열 추가
    result_df["direction"] = (
        result_df["pred_diff"] * result_df["orig_diff"] > 0
    ).astype(int)

    diff = result_df["diff"].mean()
    corr_diff = result_df[result_df["direction"] == 1]["diff"].mean()
    incorr_diff = result_df[result_df["direction"] != 1]["diff"].mean()

    print(result_df)
    print(f"상승여부를 맞춘경우 예측값과 실체값의 오차: {corr_diff}")
    print(f"상승여부를 틀린경우 예측값과 실체값의 오차: {incorr_diff}")
    print(f"예측값과 실체값의 오차: {diff}")
    return result_df


def cal_correct_prob(file, result_df):

    # result_df = result_df.dropna()
    # print(len(result_df))
    # print(result_df["direction"].value_counts())

    # 1의 발생 횟수
    count_1 = result_df["direction"].value_counts()[1]
    # 전체 발생 횟수
    total_count = result_df["direction"].count()
    # 확률 계산
    probability_of_1 = count_1 / total_count
    # file = file[12:-4]
    print(result_df["direction"].value_counts())
    print(f"{file}의 변동방향을 맞출 확률:", probability_of_1)
    return probability_of_1


def main():
    file = "000_KS200_project"
    file_path = f"recent_data/{file}.csv"
    # paths = glob.glob("recent_data/*.csv")
    # csv_df = pd.DataFrame(columns=["code", "name", "prob"])

    stock_data = pd.read_csv(file_path)
    idx, name, code = file.split("_")
    # stock_data = stock_data[-15:]
    dates = pd.to_datetime(stock_data["Date"])

    testX = file_process(stock_data)
    prediction = run_model(testX)
    pred_data_inversed_df = add_data(prediction)

    # result_df = cal_direction(stock_data, pred_data_inversed_df)
    # prob = cal_correct_prob(file, result_df)
    plot_df(dates, stock_data, pred_data_inversed_df)

    return


if __name__ == "__main__":
    main()
