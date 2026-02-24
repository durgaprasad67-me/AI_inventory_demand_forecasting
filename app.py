from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import zscore
from io import BytesIO
import hashlib
import json
import secrets
import re
import jwt
from jwt.exceptions import InvalidTokenError
from dotenv import load_dotenv
import argparse
load_dotenv()


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

PLOT_DIR = "static/plots"
os.makedirs(PLOT_DIR, exist_ok=True)
INVENTORY_CSV = "static/inventory.csv"


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256").strip() or "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
ACCESS_COOKIE_NAME = "access_token"
JWT_COOKIE_SECURE = env_bool("JWT_COOKIE_SECURE", default=False)
JWT_COOKIE_SAMESITE = (os.getenv("JWT_COOKIE_SAMESITE") or "").strip().lower()
APP_USERNAME = os.getenv("APP_USERNAME", "admin").strip() or "admin"
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()
APP_HOST = os.getenv("APP_HOST", "0.0.0.0").strip() or "0.0.0.0"
APP_PORT = int(os.getenv("APP_PORT") or os.getenv("PORT") or "8000")
USERS_DB = os.getenv("USERS_DB", "users.json").strip() or "users.json"
IN_HF_SPACE = bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID"))
AUTH_BYPASS = env_bool("AUTH_BYPASS", default=False)

if not JWT_COOKIE_SAMESITE:
    JWT_COOKIE_SAMESITE = "none" if IN_HF_SPACE else "lax"
if JWT_COOKIE_SAMESITE not in {"lax", "strict", "none"}:
    JWT_COOKIE_SAMESITE = "none" if IN_HF_SPACE else "lax"
if JWT_COOKIE_SAMESITE == "none" and not JWT_COOKIE_SECURE:
    JWT_COOKIE_SECURE = True
if IN_HF_SPACE:
    # In Hugging Face Spaces (embedded/proxied), cookies need explicit cross-site settings.
    JWT_COOKIE_SAMESITE = "none"
    JWT_COOKIE_SECURE = True


if not JWT_SECRET or JWT_SECRET == "replace with a long random secret":
    if IN_HF_SPACE:
        JWT_SECRET = secrets.token_hex(32)
        print("Warning: JWT_SECRET is not set. Using an ephemeral secret for this Space session.")
    else:
        raise RuntimeError("Set a strong JWT_SECRET in .env before starting the app.")
if not APP_PASSWORD or APP_PASSWORD == "change me now":
    if IN_HF_SPACE:
        APP_PASSWORD = os.getenv("HF_DEMO_PASSWORD", "Demo@123!")
        print("Warning: APP_PASSWORD is not set. Using HF_DEMO_PASSWORD or default demo password.")
    else:
        raise RuntimeError("Set APP_PASSWORD in .env (not the placeholder value).")


def hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


def validate_username(username: str) -> str:
    username = username.strip()
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Login name must be at least 3 characters")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", username):
        raise HTTPException(
            status_code=400,
            detail="Login name should be like Username123 (start with a letter, letters/numbers only)",
        )
    return username


def suggest_usernames(username: str, users: dict, count: int = 3) -> list[str]:
    # Build a safe base from user input, then append random digits until unique.
    cleaned = re.sub(r"[^A-Za-z0-9]", "", (username or "").strip())
    if not cleaned:
        cleaned = "User"
    if not cleaned[0].isalpha():
        cleaned = f"User{cleaned}"
    base = cleaned[:12]

    existing_lower = {name.lower() for name in users.keys()}
    suggestions = []
    while len(suggestions) < count:
        candidate = f"{base}{secrets.randbelow(10000):04d}"
        candidate_lower = candidate.lower()
        if candidate_lower in existing_lower:
            continue
        if candidate_lower in {name.lower() for name in suggestions}:
            continue
        suggestions.append(candidate)
    return suggestions


def validate_password(password: str) -> None:
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if not re.search(r"\d", password):
        raise HTTPException(status_code=400, detail="Password must include at least one number")
    if not re.search(r"[^A-Za-z0-9]", password):
        raise HTTPException(
            status_code=400,
            detail="Password must include at least one special character",
        )


def load_users() -> dict:
    if not os.path.exists(USERS_DB):
        return {}
    try:
        with open(USERS_DB, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_users(users: dict) -> None:
    users_parent = os.path.dirname(USERS_DB)
    if users_parent:
        os.makedirs(users_parent, exist_ok=True)
    with open(USERS_DB, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def ensure_default_user() -> None:
    users = load_users()
    if users:
        return
    salt = secrets.token_hex(16)
    users[APP_USERNAME] = {
        "salt": salt,
        "password_hash": hash_password(APP_PASSWORD, salt),
    }
    save_users(users)


ensure_default_user()


def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def is_authenticated(request: Request) -> bool:
    if AUTH_BYPASS:
        return True
    token = request.cookies.get(ACCESS_COOKIE_NAME)
    if not token:
        return False
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return bool(payload.get("sub"))
    except InvalidTokenError:
        return False


def require_auth(request: Request) -> str:
    if AUTH_BYPASS:
        return "space-user"
    token = request.cookies.get(ACCESS_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    return username


@app.get("/login")
def login_page(request: Request):
    if is_authenticated(request):
        return RedirectResponse(url="/", status_code=303)
    if not os.path.exists("static/login.html"):
        raise HTTPException(status_code=404, detail="Login page not found")
    return FileResponse("static/login.html")


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    username = validate_username(username)

    users = load_users()
    record = users.get(username)
    if not record:
        raise HTTPException(status_code=404, detail="User does not exist. Please register first.")
    salt = record.get("salt")
    password_hash = record.get("password_hash")
    if not salt or not password_hash:
        raise HTTPException(status_code=401, detail="Invalid password")
    if hash_password(password, salt) != password_hash:
        raise HTTPException(status_code=401, detail="Invalid password")

    token = create_access_token(username)
    response = JSONResponse({"message": "Login successful"})
    response.set_cookie(
        key=ACCESS_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite=JWT_COOKIE_SAMESITE,
        secure=JWT_COOKIE_SECURE,
        max_age=JWT_EXPIRE_MINUTES * 60,
    )
    return response


@app.get("/register")
def register_page(request: Request):
    if is_authenticated(request):
        return RedirectResponse(url="/", status_code=303)
    if not os.path.exists("static/register.html"):
        raise HTTPException(status_code=404, detail="Register page not found")
    return FileResponse("static/register.html")


@app.post("/register")
def register(
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
):
    users = load_users()
    try:
        username = validate_username(username)
    except HTTPException as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "message": exc.detail,
                "suggested_usernames": suggest_usernames(username, users),
            },
        )
    validate_password(password)
    if password != confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    existing_lower = {name.lower() for name in users.keys()}
    if username.lower() in existing_lower:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "User already exists",
                "suggested_usernames": suggest_usernames(username, users),
            },
        )

    salt = secrets.token_hex(16)
    users[username] = {
        "salt": salt,
        "password_hash": hash_password(password, salt),
    }
    save_users(users)
    return JSONResponse({"message": "Registered successfully"})


@app.post("/logout")
def logout_post():
    response = JSONResponse({"message": "Logged out"})
    response.delete_cookie(ACCESS_COOKIE_NAME)
    return response


@app.get("/logout")
def logout_get():
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie(ACCESS_COOKIE_NAME)
    return response


@app.get("/")
def index(request: Request):
    if not is_authenticated(request):
        if not os.path.exists("static/login.html"):
            raise HTTPException(status_code=404, detail="Login page not found")
        return FileResponse("static/login.html")
    if not os.path.exists("static/index.html"):
        raise HTTPException(status_code=404, detail="Frontend file not found")
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    require_auth(request)
    contents = await file.read()
    try:
        df = pd.read_csv(BytesIO(contents), low_memory=False)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file")
    df.columns = df.columns.str.strip()

    if "Date" not in df.columns or "Sales" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain Date and Sales columns")

    try:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Date values in CSV")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Date", "Sales"]).copy()
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV has no valid Date/Sales rows")
    df = df.sort_values("Date")

    if "Store" in df.columns:
        df["Store"] = pd.to_numeric(df["Store"], errors="coerce")
        df = df[df["Store"] == 1]
    if "Open" in df.columns:
        df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
        df = df[df["Open"] == 1]
    if "Promo" not in df.columns:
        df["Promo"] = 0
    df["Promo"] = pd.to_numeric(df["Promo"], errors="coerce").fillna(0).astype(int)
    if "StateHoliday" not in df.columns:
        df["StateHoliday"] = "none"

    if df.empty:
        raise HTTPException(status_code=400, detail="No rows left after filtering")

    df["StateHoliday"] = df["StateHoliday"].astype(str).replace("0", "none")
    df = pd.get_dummies(df, columns=["StateHoliday"])
    holiday_cols = [c for c in df.columns if c.startswith("StateHoliday_")]
    df[holiday_cols] = df[holiday_cols].astype(int)

    # prophet~
    prophet_df = df[["Date", "Sales"] + holiday_cols].rename(
        columns={"Date": "ds", "Sales": "y"}
    )
    prophet_df = prophet_df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(prophet_df) < 2:
        raise HTTPException(
            status_code=400,
            detail="Not enough valid rows for Prophet training after cleaning.",
        )

    prophet = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    for col in holiday_cols:
        prophet.add_regressor(col)

    prophet.fit(prophet_df)

    future = prophet.make_future_dataframe(periods=90)
    for col in holiday_cols:
        future[col] = 0

    forecast = prophet.predict(future)

    plt.figure()
    prophet.plot(forecast)
    plt.title("Prophet Forecast")
    plt.savefig(f"{PLOT_DIR}/prophet_forecast.png")
    plt.close()

    # XGBOOST model
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
        "day", "week", "month", "year", "day_of_week", "is_weekend", "Promo",
        "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_28"
    ]

    df_model = df[features + ["Sales", "Date"]].dropna()
    if len(df_model) < 20:
        raise HTTPException(
            status_code=400,
            detail="Not enough usable rows after feature engineering. Provide more historical data.",
        )

    split = int(len(df_model) * 0.8)
    test_size = len(df_model) - split
    if split < 10 or test_size < 2:
        raise HTTPException(status_code=400, detail="Could not create train/test split from data")
    X = df_model[features]
    y = df_model["Sales"]

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    test_dates = df_model["Date"].iloc[split:]

    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_pred_non_negative = np.maximum(y_pred, 0)

    # metrics
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    prophet_test = df.loc[y_test.index, ["Date"] + holiday_cols].rename(columns={"Date": "ds"})
    prophet_pred = prophet.predict(prophet_test)["yhat"].values

    p_rmse = float(np.sqrt(mean_squared_error(y_test, prophet_pred)))
    p_mae = float(mean_absolute_error(y_test, prophet_pred))

    # AI model selection
    best_model = "XGBoost" if rmse < p_rmse else "Prophet"

    # adaptive stock
    y_test_mean = abs(float(y_test.mean()))
    error_ratio = rmse / y_test_mean if y_test_mean != 0 else 0.0
    if error_ratio > 0.25:
        safety_rate = 0.20
    elif error_ratio > 0.15:
        safety_rate = 0.15
    else:
        safety_rate = 0.10

    safety_stock = y_pred_non_negative * safety_rate
    inventory_required = y_pred_non_negative + safety_stock

    inventory_df = pd.DataFrame({
        "Date": test_dates.values,
        "Forecasted_Demand": y_pred_non_negative,
        "Safety_Stock": safety_stock,
        "Inventory_Required": inventory_required
    })
    inventory_df.to_csv(INVENTORY_CSV, index=False)

    # anomalies detection
    sales_non_null = df["Sales"].dropna()
    sales_std = float(sales_non_null.std(ddof=0)) if len(sales_non_null) else 0.0
    if sales_std == 0.0 or np.isnan(sales_std):
        anomalies = df.iloc[0:0][["Date", "Sales"]]
    else:
        z_scores = pd.Series(zscore(sales_non_null), index=sales_non_null.index)
        anomalies = df.loc[z_scores[np.abs(z_scores) > 3].index, ["Date", "Sales"]]

    # AI business layer: demand risk + reorder recommendation
    pred_mean = abs(float(np.mean(y_pred_non_negative))) if len(y_pred_non_negative) else 0.0
    pred_std = float(np.std(y_pred_non_negative)) if len(y_pred_non_negative) else 0.0
    forecast_volatility = (pred_std / pred_mean) if pred_mean != 0 else 0.0
    anomaly_ratio = (len(anomalies) / len(df)) if len(df) else 0.0

    error_component = min(error_ratio / 0.30, 1.0) * 40
    volatility_component = min(forecast_volatility / 0.50, 1.0) * 35
    anomaly_component = min(anomaly_ratio / 0.10, 1.0) * 25
    risk_score = int(round(error_component + volatility_component + anomaly_component))
    risk_score = max(0, min(100, risk_score))

    if risk_score >= 67:
        risk_level = "High"
        lead_time_days = 14
    elif risk_score >= 34:
        risk_level = "Medium"
        lead_time_days = 10
    else:
        risk_level = "Low"
        lead_time_days = 7

    avg_daily_demand = float(df_model["Sales"].tail(30).mean())
    avg_safety_stock = float(np.mean(safety_stock)) if len(safety_stock) else 0.0
    reorder_point = (avg_daily_demand * lead_time_days) + avg_safety_stock

    horizon_days = min(14, len(y_pred_non_negative))
    next_14_day_demand = float(np.sum(y_pred_non_negative[:horizon_days]))
    recommended_order_qty = next_14_day_demand + avg_safety_stock

    feature_importances = []
    for feat, imp in zip(features, xgb.feature_importances_):
        feature_importances.append({"feature": feat, "importance": float(imp)})
    top_drivers = sorted(feature_importances, key=lambda x: x["importance"], reverse=True)[:5]

    ai_summary = (
        f"Demand risk is {risk_level} ({risk_score}/100). "
        f"Reorder when stock falls below {reorder_point:.0f} units. "
        f"Suggested order quantity for the next 14 days is {recommended_order_qty:.0f} units."
    )

    anomaly_msg = (
        f"{len(anomalies)} demand anomalies detected"
        if len(anomalies) > 0 else
        "No demand anomalies detected"
    )

    # insights
    insights = [
        f"AI selected {best_model} as the best model based on lower error",
        f"Adaptive safety stock set to {int(safety_rate * 100)}%",
        anomaly_msg,
        ai_summary
    ]

    # comparison plots
    plt.figure(figsize=(10, 4))
    plt.plot(y_test.values, label="Actual")
    plt.plot(y_pred, label="XGBoost")
    plt.plot(prophet_pred, label="Prophet")
    plt.legend()
    plt.title("Model Comparison")
    plt.savefig(f"{PLOT_DIR}/comparison.png")
    plt.close()

    # plots inventory
    plt.figure(figsize=(10, 4))
    plt.plot(test_dates, inventory_required, label="Inventory Required")
    plt.plot(test_dates, y_pred_non_negative, label="Forecasted Demand")
    plt.legend()
    plt.title("AI Inventory Planning")
    plt.savefig(f"{PLOT_DIR}/inventory.png")
    plt.close()

    return {
        "rmse": rmse,
        "mae": mae,
        "p_rmse": p_rmse,
        "p_mae": p_mae,
        "best_model": best_model,
        "insights": insights,
        "ai": {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "lead_time_days": lead_time_days,
            "reorder_point": round(reorder_point, 2),
            "recommended_order_qty": round(recommended_order_qty, 2),
            "next_14_day_demand": round(next_14_day_demand, 2),
            "top_drivers": top_drivers,
            "summary": ai_summary
        },
        "plots": {
            "comparison": "/static/plots/comparison.png",
            "inventory": "/static/plots/inventory.png",
            "prophet": "/static/plots/prophet_forecast.png"
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/download_inventory")
def download_inventory(request: Request):
    require_auth(request)
    if not os.path.exists(INVENTORY_CSV):
        raise HTTPException(status_code=404, detail="Inventory file not found")
    return FileResponse(INVENTORY_CSV, media_type="text/csv", filename="inventory.csv")


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gen-jwt-secret",
        action="store_true",
        help="Generate a random 256-bit JWT secret (64 hex chars).",
    )
    parser.add_argument(
        "--gen-jwt-token",
        metavar="SUBJECT",
        help="Generate a JWT token for the given subject using JWT_SECRET.",
    )
    args = parser.parse_args()

    if args.gen_jwt_secret:
        print(secrets.token_hex(32))
    elif args.gen_jwt_token:
        print(create_access_token(args.gen_jwt_token))
    else:
        uvicorn.run(app, host=APP_HOST, port=APP_PORT)

