import FinanceDataReader as fdr
import ta

ks200 = fdr.DataReader("KS200", "2020-10-01")

print(ks200)

# 어제 대비 주가 수익률
ks200["Return"] = (ks200["Close"].shift(-1) / ks200["Close"]) - 1
ks200["Return"] = ks200["Return"].shift(1)

macd_indicator = ta.trend.macd_diff(ks200["Close"])
rsi_indicator = ta.momentum.rsi(ks200["Close"])

ma5 = ks200["Close"].rolling(window=5).mean()
ma20 = ks200["Close"].rolling(window=20).mean()
ma60 = ks200["Close"].rolling(window=60).mean()

bb_upper = ta.volatility.bollinger_hband(ks200["Close"])
bb_lower = ta.volatility.bollinger_lband(ks200["Close"])

# Combine indicators with the original data
ks200["MACD"] = macd_indicator
ks200["RSI"] = rsi_indicator
ks200["MA5"] = ma5
ks200["MA20"] = ma20
ks200["MA60"] = ma60
ks200["BB_Upper"] = bb_upper
ks200["BB_Lower"] = bb_lower
ks200["ans"] = ks200["Close"].shift(-1)

# NaN값 제외를 위한 슬라이싱
ks200 = ks200[60:-1]

# 정리된 데이터프레임 csv로 저징
ks200.to_csv(f"csv/000_KS200_000000.csv")
print(f"저장 완료")
