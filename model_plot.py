import glob
import json
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def check_models(file: str):
    stock_data = pd.read_csv(f"csv/{file}.csv")
    if len(stock_data["Close"]) < 100:
        return {file: None}
    dates = pd.to_datetime(stock_data["Date"])
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
    ]
    # save original 'Close' prices for later
    original_close = stock_data["Close"].values

    # 새로운 데이터프레임 생성 및 변수형변환
    stock_data = stock_data[cols].astype(float)

    # 데이터 정규화
    scaler = StandardScaler()
    scaler = scaler.fit(stock_data)
    stock_data_scaled = scaler.transform(stock_data)

    # train 데이터와 test 데이터로 분할
    n_train = int(0.9 * stock_data_scaled.shape[0])
    test_data_scaled = stock_data_scaled[n_train:]

    # 날짜 데이터 별도 저장 (미래의 플로팅을 위해)
    test_dates = dates[n_train:]

    # LSTM에 필요한 데이터 형식으로 재구성
    pred_days = 1
    seq_len = 30

    testX, testY = [], []

    for i in range(seq_len, len(test_data_scaled) - pred_days + 1):
        testX.append(test_data_scaled[i - seq_len : i, :])
        testY.append(test_data_scaled[i + pred_days - 1 : i + pred_days, 3])

    testX, testY = np.array(testX), np.array(testY)

    model = keras.models.load_model(f"keras_models/{file}.keras")
    print("모델을 디스크에서 로드했습니다.")

    # prediction
    prediction = model.predict(testX)

    # generate array filled with means for prediction
    mean_values_pred = np.repeat(
        scaler.mean_[np.newaxis, :], prediction.shape[0], axis=0
    )

    # substitute predictions into the first column
    mean_values_pred[:, 0] = np.squeeze(prediction)

    # inverse transform
    y_pred = scaler.inverse_transform(mean_values_pred)[:, 0]

    # generate array filled with means for testY
    mean_values_testY = np.repeat(scaler.mean_[np.newaxis, :], testY.shape[0], axis=0)

    # substitute testY into the first column
    mean_values_testY[:, 0] = np.squeeze(testY)

    # inverse transform
    testY_original = scaler.inverse_transform(mean_values_testY)[:, 0]

    predict_MSE = np.mean(
        np.sqrt((testY_original - y_pred) ** 2)
        / original_close[len(original_close) - len(y_pred) :]
    )

    print(f"{file}모델의 주가대비 MSE: {predict_MSE}")

    # plotting
    plt.figure(figsize=(14, 5))

    # plot original 'Close' prices
    plt.plot(dates, original_close, color="green", label="Original Close Price")

    # plot actual vs predicted
    # plt.plot(
    #     test_dates[seq_len:],
    #     testY_original,
    #     color="blue",
    #     label="Actual Close Price",
    # )
    plt.plot(
        test_dates[seq_len:],
        y_pred,
        color="blue",
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
    file = "001_\uc0bc\uc131\uc804\uc790(005930)"
    check_models(file)


if __name__ == "__main__":
    main()
