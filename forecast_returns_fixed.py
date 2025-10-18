"""
forecast_returns_fixed.py
Phiên bản vá: tương thích với mọi scikit-learn (fallback nếu 'squared' không được hỗ trợ)
Dự báo số lượng hàng hoàn theo Return_Date
"""

import os
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import math

# --- Cấu hình ---
INPUT_FILE = "ecommerce_returns_synthetic_data.csv"   # đổi theo file bạn có
OUTPUT_FORECAST_CSV = "forecast_for_tableau.csv"
FIG_FORECAST = "forecast_plot.png"
FIG_COMPONENTS = "components_plot.png"
FREQ = "D"  # 'D' ngày, 'W' tuần, 'M' tháng
DEBUG_ENV = False  # bật True nếu muốn in phiên bản sklearn / chữ ký hàm

# --- Hàm load & chuẩn hóa ---
def load_and_prepare(path=INPUT_FILE):
    df = pd.read_csv(path)
    # đảm bảo có cột Return_Date và Return_Status
    if "Return_Date" not in df.columns or "Return_Status" not in df.columns:
        raise ValueError("CSV phải có cột 'Return_Date' và 'Return_Status'.")
    df["Return_Date"] = pd.to_datetime(df["Return_Date"], errors="coerce")
    # chỉ chọn những record thực sự returned
    # chấp nhận nhiều cách viết: "returned", "Returned", "RETURNED"
    returns = df[df["Return_Status"].astype(str).str.lower() == "returned"].copy()
    if returns.empty:
        print("CẢNH BÁO: Không tìm thấy record có Return_Status == 'Returned'. Kết quả sẽ toàn 0.")
    # group theo ngày
    daily_returns = (
        returns.groupby("Return_Date")["Order_ID"]
        .count()
        .reset_index()
        .rename(columns={"Return_Date": "ds", "Order_ID": "y"})
    )
    if daily_returns.empty:
        # nếu rỗng, tạo khung ngày rỗng dựa trên ngày đặt hàng (để không crash)
        if "Order_Date" in df.columns:
            min_d = pd.to_datetime(df["Order_Date"], errors="coerce").min()
            max_d = pd.to_datetime(df["Order_Date"], errors="coerce").max()
        else:
            min_d = pd.Timestamp.today() - pd.Timedelta(days=365)
            max_d = pd.Timestamp.today()
        full = pd.DataFrame({"ds": pd.date_range(min_d, max_d, freq=FREQ)})
        full["y"] = 0
        return full
    # reindex full date range
    full = pd.DataFrame({"ds": pd.date_range(daily_returns["ds"].min(), daily_returns["ds"].max(), freq=FREQ)})
    daily_returns = full.merge(daily_returns, on="ds", how="left")
    daily_returns["y"] = daily_returns["y"].fillna(0)
    return daily_returns

# --- Main pipeline ---
def main():
    if DEBUG_ENV:
        import sklearn, inspect
        print("sklearn version:", sklearn.__version__)
        print("mean_squared_error signature:", inspect.signature(mean_squared_error))

    df = load_and_prepare(INPUT_FILE)
    print(f"Prepared data: {len(df)} rows  — from {df['ds'].min().date()} to {df['ds'].max().date()}")

    # train/test split
    test_days = int(max(7, 0.15 * len(df)))
    train_df = df.iloc[:-test_days].copy()
    test_df = df.iloc[-test_days:].copy()
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # fit Prophet
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(train_df)

    # forecast
    future_periods = max(30, test_days)
    future = m.make_future_dataframe(periods=future_periods, freq=FREQ)
    forecast = m.predict(future)

    # merge history into forecast for saving/plotting
    save_df = forecast[["ds","yhat","yhat_lower","yhat_upper"]].merge(df, on="ds", how="left")
    save_df = save_df.sort_values("ds").reset_index(drop=True)
    save_df["historic"] = save_df["y"].notna()
    save_df.to_csv(OUTPUT_FORECAST_CSV, index=False)
    print("Saved forecast CSV ->", OUTPUT_FORECAST_CSV)

    # Evaluate on test set: join forecast and true y
    eval_df = save_df.copy()
    # pick rows in the original test_ds range
    eval_test = eval_df[eval_df["ds"].isin(test_df["ds"])].copy()
    # drop rows missing y or yhat
    eval_test = eval_test.dropna(subset=["y","yhat"])
    if eval_test.empty:
        print("CẢNH BÁO: Không có điểm để đánh giá (eval_test rỗng sau dropna). Bỏ qua đánh giá.")
        mae = rmse = float("nan")
    else:
        # MAE
        mae = mean_absolute_error(eval_test["y"], eval_test["yhat"])
        # RMSE: thử gọi với squared=False, nếu không được thì fallback
        try:
            # một số môi trường không chấp nhận tham số squared
            rmse = mean_squared_error(eval_test["y"], eval_test["yhat"], squared=False)
        except TypeError:
            mse = mean_squared_error(eval_test["y"], eval_test["yhat"])
            rmse = math.sqrt(mse)
    print(f"Evaluation on holdout: MAE = {mae:.4f}, RMSE = {rmse:.4f}")

    # Plot forecast + actuals
    plt.figure(figsize=(14,6))
    plt.plot(save_df["ds"], save_df["yhat"], label="Forecast (yhat)")
    if save_df["y"].notna().any():
        plt.plot(save_df["ds"], save_df["y"], label="Actual (y)", linestyle='-', marker='o', markersize=3)
    plt.fill_between(save_df["ds"], save_df["yhat_lower"], save_df["yhat_upper"], alpha=0.2, label="Uncertainty")
    plt.axvline(x=train_df["ds"].max(), color="k", linestyle="--", alpha=0.6, label="Train/Test split")
    plt.xlabel("Date")
    plt.ylabel("Returns Count")
    plt.title("Dự báo số lượng hàng hoàn")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_FORECAST, dpi=150)
    print("Saved plot ->", FIG_FORECAST)

    # components
    fig2 = m.plot_components(forecast)
    fig2.set_size_inches(10,8)
    fig2.savefig(FIG_COMPONENTS, dpi=150)
    print("Saved components ->", FIG_COMPONENTS)

    # preview
    print("Preview (last 8 rows):")
    print(save_df.tail(8).to_string(index=False))
    
# === Thêm vào cuối file forecast_returns_fixed.py ===
def forecast_returns(input_file=INPUT_FILE, freq="D", periods=30):
    """
    Hàm dự báo số lượng hàng hoàn, trả về (dataframe, model)
    Dùng cho UI hoặc các script khác.
    """
    df = load_and_prepare(input_file)
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(df, on="ds", how="left")
    result["historic"] = result["y"].notna()
    return result, m

if __name__ == "__main__":
    main()
