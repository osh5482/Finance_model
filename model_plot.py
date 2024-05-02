import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def check_models(file: str):
    stock_data = pd.read_csv(f"csv/{file}.csv")
    dates = pd.to_datetime(stock_data["Date"])
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

    # save original 'Close' prices for later
    original_close = stock_data["Close"].values

    # 새로운 데이터프레임 생성 및 변수형변환
    stock_data = stock_data[cols].astype(float)

    # 데이터 정규화
    scaler = StandardScaler()
    scaler = scaler.fit(stock_data)
    test_data_scaled = scaler.transform(stock_data)

    # 날짜 데이터 별도 저장 (미래의 플로팅을 위해)
    test_dates = dates  # [n_train:]

    # LSTM에 필요한 데이터 형식으로 재구성
    pred_days = 1  # 한번에 예측할 날짜
    seq_len = 60  # 예측에 쓸 일자

    testX = []

    for i in range(seq_len, len(test_data_scaled) - pred_days + 1):
        testX.append(test_data_scaled[i - seq_len : i, :])

    testX = np.array(testX)

    model = keras.models.load_model(f"keras_models/000_entire_model.keras")
    print("모델을 디스크에서 로드했습니다.")

    # 예측값 생성
    prediction = model.predict(testX)

    # 예측값을 넣을 배열 생성
    mean_values_pred = np.repeat(
        scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0
    )

    # 배열에 예측값 할당
    mean_values_pred[:, 0] = np.squeeze(prediction)

    # 배열을 역정규화
    y_pred = scaler.inverse_transform(mean_values_pred)[:, 0]

    # 예측값과 실측값의 오차의 평균
    predict_MSE = np.mean(
        abs(original_close[seq_len:] - y_pred)
        / original_close[len(original_close) - len(y_pred) :]
    )

    print(f"{file}모델의 주가대비 MSE: {predict_MSE}")

    # plotting
    plt.figure(figsize=(16, 9))

    # plot original 'Close' prices
    plt.plot(
        dates,
        original_close,
        color="green",
        marker=".",
        label="Original Close Price",
    )

    plt.plot(
        test_dates[seq_len:],
        y_pred,
        color="blue",
        marker=".",
        linestyle="--",
        label="Predicted Close Price",
    )
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(f"{file}")
    plt.legend()
    plt.show()
    # plt.show(block=False)
    # plt.pause(1)
    # plt.close()
    return {file: predict_MSE}


def main():
    # mse = {}
    # file_faths = glob.glob("csv/*.csv")
    # for file in file_faths:
    #     file = file[4:-4]
    #     print(file)
    #     result = check_models(file)
    #     mse[file] = result[file]

    # with open("mse_per.json", "w") as f:
    #     json.dump(mse, f)
    file = "001_삼성전자_005930"
    check_models(file)


if __name__ == "__main__":
    main()
