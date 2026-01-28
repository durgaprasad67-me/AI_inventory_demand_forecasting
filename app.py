from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib
matplotlib.use("Agg")

from flask import Flask, request, jsonify, send_file, send_from_directory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__, static_folder="static")

PLOT_DIR = "static/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

INVENTORY_CSV = "static/inventory.csv"



@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    df = pd.read_csv(file, low_memory=False)
    df.columns = df.columns.str.strip()

    if "Date" not in df.columns or "Sales" not in df.columns:
        return jsonify({"error": "CSV must contain Date and Sales columns"}), 400

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    if "Store" in df.columns:
        df = df[df["Store"] == 1]
    if "Open" in df.columns:
        df = df[df["Open"] == 1]
    if "Promo" not in df.columns:
        df["Promo"] = 0
    if "StateHoliday" not in df.columns:
        df["StateHoliday"] = "none"

    df["StateHoliday"] = df["StateHoliday"].astype(str).replace("0", "none")
    df = pd.get_dummies(df, columns=["StateHoliday"])
    holiday_cols = [c for c in df.columns if c.startswith("StateHoliday_")]
    df[holiday_cols] = df[holiday_cols].astype(int)

    # prophet
    prophet_df = df[["Date", "Sales"] + holiday_cols].rename(
        columns={"Date": "ds", "Sales": "y"}
    )

    prophet = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    for col in holiday_cols:
        prophet.add_regressor(col)

    prophet.fit(prophet_df)

    future = prophet.make_future_dataframe(periods=90)
    for col in holiday_cols:
        future[col] = 0

    forecast = prophet.predict(future)

    fig = prophet.plot(forecast)
    plt.title("Prophet Forecast")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/prophet_forecast.png")
    plt.close()

    # xgboost
    df["day"] = df["Date"].dt.day
    df["week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["sales_lag_1"] = df["Sales"].shift(1)
    df["sales_lag_7"] = df["Sales"].shift(7)
    df["sales_lag_14"] = df["Sales"].shift(14)
    df["sales_lag_28"] = df["Sales"].shift(28)

    df["rolling_mean_7"] = df["Sales"].rolling(7).mean()
    df["rolling_mean_14"] = df["Sales"].rolling(14).mean()
    df["rolling_mean_28"] = df["Sales"].rolling(28).mean()

    features = [
        "day","week","month","year","day_of_week","is_weekend","Promo",
        "sales_lag_1","sales_lag_7","sales_lag_14","sales_lag_28",
        "rolling_mean_7","rolling_mean_14","rolling_mean_28"
    ]

    df_model = df[["Date"] + features + ["Sales"]].dropna()

    split = int(len(df_model) * 0.8)
    X = df_model[features]
    y = df_model["Sales"]

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    test_dates = df_model["Date"].iloc[split:]

    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

    # metrics
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    prophet_test = df.loc[y_test.index, ["Date"] + holiday_cols].rename(columns={"Date": "ds"})
    prophet_pred = prophet.predict(prophet_test)["yhat"].values

    p_rmse = float(np.sqrt(mean_squared_error(y_test, prophet_pred)))
    p_mae = float(mean_absolute_error(y_test, prophet_pred))

    # comparison plots
    plt.figure(figsize=(10,4))
    plt.plot(y_test.values, label="Actual")
    plt.plot(y_pred, label="XGBoost")
    plt.plot(prophet_pred, label="Prophet")
    plt.legend()
    plt.title("Prophet vs XGBoost")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/comparison.png")
    plt.close()

    # inventory
    safety_stock = 0.10 * y_pred
    inventory_required = y_pred + safety_stock

    inventory_df = pd.DataFrame({
        "Date": test_dates.values,
        "Forecasted_Demand": y_pred,
        "Safety_Stock": safety_stock,
        "Inventory_Required": inventory_required
    })

    inventory_df.to_csv(INVENTORY_CSV, index=False)

    plt.figure(figsize=(10,4))
    plt.plot(test_dates, inventory_df["Forecasted_Demand"], label="Forecasted Demand")
    plt.plot(test_dates, inventory_df["Inventory_Required"], label="Inventory Required")
    plt.legend()
    plt.title("Inventory Planning")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/inventory.png")
    plt.close()

    # json response
    return jsonify({
        "rmse": rmse,
        "mae": mae,
        "p_rmse": p_rmse,
        "p_mae": p_mae,
        "plots": {
            "comparison": "/static/plots/comparison.png",
            "inventory": "/static/plots/inventory.png",
            "prophet": "/static/plots/prophet_forecast.png"
        }
    })


@app.route("/download_inventory")
def download_inventory():
    return send_file(INVENTORY_CSV, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
