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
past = "000_KS200_111111"
scaler = StandardScaler()
past = pd.read_csv(f"csv/{past}.csv")
past = past[cols].astype(float)
scaler = scaler.fit(past)


def file_process(stock_data: pd.DataFrame):
    """테스트 데이터셋을 만들고 정규화합니다"""

    # 새로운 데이터프레임 생성 및 변수형 변환
    stock_data = stock_data[cols].astype(float)

    # 데이터 정규화
    # scaler.fit(stock_data)
    test_data_scaled = scaler.transform(stock_data)

    testX = []

    for i in range(seq_len, len(test_data_scaled) + 1):
        testX.append(test_data_scaled[i - seq_len : i, :])

    testX = np.array(testX)
    print(testX.shape)
    return testX


def run_model(testX: np.ndarray):
    """정규화된 데이터를 모델에 입력해 예측값을 출력합니다"""
    model = keras.models.load_model(f"keras_models/000_KS200_past.keras")
    prediction = model.predict(testX)

    return prediction


def add_data(pred_data: np.ndarray):
    """예측값을 역정규화해 기존 데이터셋에 추가합니다"""
    # 원본 스케일로 역정규화

    dummy = np.zeros((len(pred_data), scaler.scale_.shape[0]))
    dummy[:, 3] = pred_data.reshape(-1)
    prediction_inversed = scaler.inverse_transform(dummy)
    prediction_inversed_df = pd.DataFrame(prediction_inversed, columns=cols)

    return prediction_inversed_df


def plot_df(dates, stock_data: pd.DataFrame, pred_data_inversed_df: pd.DataFrame):
    """실제 데이터와 예상 데이터를 plot"""
    plt.figure(figsize=(16, 9))
    plt.plot(
        dates[:-1],
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

    diff = result_df["diff"]
    corr_diff = result_df[result_df["direction"] == 1]["diff"]
    incorr_diff = result_df[result_df["direction"] == 0]["diff"]
    result_df["diff_per"] = diff / original_close
    diff_per = result_df["diff_per"]

    print(f"상승여부를 맞춘경우 예측값과 실체값의 오차: {corr_diff.mean()}")
    print(f"상승여부를 틀린경우 예측값과 실체값의 오차: {incorr_diff.mean()}")
    print(f"예측값과 실체값의 오차: {diff.mean()}")
    print(f"주가대비 오차율: {diff_per.mean()}")
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
    file = "000_KS200_2010-2017"
    file_path = f"recent_data/{file}.csv"
    # paths = glob.glob("recent_data/*.csv")
    # csv_df = pd.DataFrame(columns=["code", "name", "prob"])

    stock_data = pd.read_csv(file_path)
    idx, name, code = file.split("_")

    print(stock_data)

    # stock_data = stock_data[10:]

    dates = pd.to_datetime(stock_data["Date"])

    next_date = dates.iloc[-1] + datetime.timedelta(days=1)
    dates = dates._append(pd.Series([next_date]))

    testX = file_process(stock_data)
    prediction = run_model(testX)
    pred_data_inversed_df = add_data(prediction)

    print(stock_data["Close"])
    print(pred_data_inversed_df["Close"])

    result_df = cal_direction(stock_data, pred_data_inversed_df)
    prob = cal_correct_prob(file, result_df)

    plot_df(dates, stock_data, pred_data_inversed_df)

    # recent_stock = glob.glob("recent_data/*.csv")
    # recent_stock = recent_stock[1:]
    # entire_prob = [["stock", "probability"]]
    # for stock in recent_stock:
    #     stock_df = pd.read_csv(stock)
    #     testX = file_process(stock_df)
    #     prediction = run_model(testX)
    #     pred_data_inversed_df = add_data(prediction)
    #     result_df = cal_direction(stock_df, pred_data_inversed_df)
    #     prob = cal_correct_prob(stock, result_df)
    #     stock_name = stock[12:-4]
    #     entire_prob.append([stock_name, prob])
    # print(entire_prob)
    # with open("entire_prob.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(entire_prob)

    return


if __name__ == "__main__":
    main()
