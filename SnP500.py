import FinanceDataReader as fdr

# # 학습용
# SnP500 = fdr.DataReader("S&P500", "2020-01-01")

# 프로젝트용
SnP500 = fdr.DataReader("S&P500", "2023-10-01")

SnP500["Return"] = (SnP500["Close"].shift(-1) / SnP500["Close"]) - 1
SnP500["Return"] = SnP500["Return"].shift(1)


# NaN값 제외를 위한 슬라이싱 (학습용은 -1붙이기)
# 프로젝트용 [60:]
SnP500 = SnP500[60:]
print(SnP500)

# # 학습용
# SnP500.to_csv(f"csv/000_SnP500_000000.csv")

# 프로젝트용
SnP500.to_csv(f"recent_data/000_SnP500_project.csv")


print(f"저장 완료")
