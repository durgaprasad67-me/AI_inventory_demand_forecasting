from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from starlette.middleware.base import BaseHTTPMiddleware
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
import urllib.request
import urllib.error
import jwt
from jwt.exceptions import InvalidTokenError
from dotenv import load_dotenv
load_dotenv()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
PLOT_DIR = os.path.join(STATIC_DIR, "plots")
LOGIN_HTML = os.path.join(STATIC_DIR, "login.html")
REGISTER_HTML = os.path.join(STATIC_DIR, "register.html")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html")
INVENTORY_CSV = os.path.join(STATIC_DIR, "inventory.csv")
USERS_DB = os.path.join(BASE_DIR, "users.json")


app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class BlockDirectStaticPagesMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path == "/static/index.html":
            return RedirectResponse(url="/app", status_code=303)
        if path == "/static/login.html":
            return RedirectResponse(url="/login", status_code=303)
        if path == "/static/register.html":
            return RedirectResponse(url="/register", status_code=303)
        return await call_next(request)


app.add_middleware(BlockDirectStaticPagesMiddleware)

os.makedirs(PLOT_DIR, exist_ok=True)


JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60
ACCESS_COOKIE_NAME = "access_token"
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
JWT_COOKIE_SECURE = os.getenv("JWT_COOKIE_SECURE", "").strip().lower() in {"1", "true", "yes", "on"}
if not os.getenv("JWT_COOKIE_SECURE"):
    JWT_COOKIE_SECURE = APP_ENV == "production"
JWT_COOKIE_SAMESITE = os.getenv("JWT_COOKIE_SAMESITE", "lax").strip().lower() or "lax"
APP_USERNAME = "admin"
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()
APP_PORT = os.getenv("APP_PORT", "8000").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1").strip()
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120").strip() or "120")
PASSWORD_HASH_ALGO = "pbkdf2_sha256"
PASSWORD_HASH_ITERATIONS = 310000


if not JWT_SECRET or JWT_SECRET == "replace with a long random secret":
    raise RuntimeError("Set a strong JWT_SECRET in .env before starting the app.")
if not APP_PASSWORD or APP_PASSWORD == "change me now":
    raise RuntimeError("Set APP_PASSWORD in .env (not the placeholder value).")


def legacy_hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


def hash_password(password: str, salt: str, iterations: int = PASSWORD_HASH_ITERATIONS) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        int(iterations),
    )
    return dk.hex()


def verify_password(password: str, record: dict) -> bool:
    salt = str(record.get("salt") or "")
    password_hash = str(record.get("password_hash") or "")
    if not salt or not password_hash:
        return False

    algo = str(record.get("password_algo") or "")
    if algo == PASSWORD_HASH_ALGO:
        iterations = int(record.get("password_iterations") or PASSWORD_HASH_ITERATIONS)
        return secrets.compare_digest(password_hash, hash_password(password, salt, iterations))

    # backward compatibility for older users created before PBKDF2 migration.
    return secrets.compare_digest(password_hash, legacy_hash_password(password, salt))


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
    # build a safe base from user input, then append random digits until unique.
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
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if not re.search(r"\d", password):
        raise HTTPException(status_code=400, detail="Password must include at least one number")
    if not re.search(r"[^A-Za-z0-9]", password):
        raise HTTPException(
            status_code=400,
            detail="Password must include at least one special character",
        )
    if not re.search(r"[A-Z]", password):
        raise HTTPException(status_code=400, detail="Password must include at least one uppercase letter")
    if not re.search(r"[a-z]", password):
        raise HTTPException(status_code=400, detail="Password must include at least one lowercase letter")


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
        "password_algo": PASSWORD_HASH_ALGO,
        "password_iterations": PASSWORD_HASH_ITERATIONS,
    }
    save_users(users)


ensure_default_user()


def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def is_authenticated(request: Request) -> bool:
    token = request.cookies.get(ACCESS_COOKIE_NAME)
    if not token:
        return False
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return bool(payload.get("sub"))
    except InvalidTokenError:
        return False


def has_stale_auth_cookie(request: Request) -> bool:
    token = request.cookies.get(ACCESS_COOKIE_NAME)
    return bool(token) and not is_authenticated(request)


def require_auth(request: Request) -> str:
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


def generate_ollama_insights(payload: dict) -> str:
    if not OLLAMA_MODEL:
        return "Local LLM insights are disabled. Set OLLAMA_MODEL in .env."

    prompt = (
        "You are an inventory planning analyst. "
        "Write concise business insights for store operations. "
        "Return 4 short bullet points and one final recommendation sentence.\n\n"
        "Forecast metrics:\n"
        f"{json.dumps(payload, indent=2)}"
    )
    body = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }

    try:
        req = urllib.request.Request(
            url=f"{OLLAMA_BASE_URL}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        text = str(data.get("response", "")).strip()
        return text or "Ollama returned an empty response."
    except urllib.error.HTTPError as exc:
        try:
            details = exc.read().decode("utf-8")
        except Exception:
            details = str(exc)
        return f"Local LLM insights unavailable (HTTP {exc.code}): {details}"
    except urllib.error.URLError:
        return (
            f"Local LLM insights unavailable: cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Start Ollama and run `ollama pull llama3.1` (or set OLLAMA_MODEL)."
        )
    except TimeoutError:
        return (
            "Local LLM insights unavailable: timed out. "
            "Try a smaller model (e.g., `phi3`) or increase OLLAMA_TIMEOUT_SECONDS in .env."
        )
    except Exception as exc:
        return f"Local LLM insights unavailable: {str(exc)}"


@app.get("/auth_status")
def auth_status(request: Request):
    token = request.cookies.get(ACCESS_COOKIE_NAME)
    if not token:
        return JSONResponse({
            "authenticated": False,
            "cookie_present": False,
            "username": None,
            "reason": "missing_cookie",
        })

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except InvalidTokenError as exc:
        return JSONResponse({
            "authenticated": False,
            "cookie_present": True,
            "username": None,
            "reason": str(exc) or "invalid_or_expired_token",
        })

    username = payload.get("sub")
    if not username:
        return JSONResponse({
            "authenticated": False,
            "cookie_present": True,
            "username": None,
            "reason": "missing_subject",
        })

    return JSONResponse({
        "authenticated": True,
        "cookie_present": True,
        "username": username,
        "reason": "ok",
    })


@app.get("/login")
def login_page(request: Request):
    if is_authenticated(request):
        return RedirectResponse(url="/app", status_code=303)
    if not os.path.exists(LOGIN_HTML):
        raise HTTPException(status_code=404, detail="Login page not found")
    response = FileResponse(LOGIN_HTML)
    if has_stale_auth_cookie(request):
        response.delete_cookie(ACCESS_COOKIE_NAME)
    return response


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    username = validate_username(username)

    users = load_users()
    username_lookup = {name.lower(): name for name in users.keys()}
    canonical_username = username_lookup.get(username.lower())
    record = users.get(canonical_username) if canonical_username else None
    if not record:
        raise HTTPException(status_code=404, detail="User does not exist. Please register first.")
    if not verify_password(password, record):
        raise HTTPException(status_code=401, detail="Invalid password")

    # Upgrade legacy password hashes on successful login.
    if record.get("password_algo") != PASSWORD_HASH_ALGO:
        salt = str(record.get("salt") or "")
        users[canonical_username] = {
            **record,
            "password_hash": hash_password(password, salt),
            "password_algo": PASSWORD_HASH_ALGO,
            "password_iterations": PASSWORD_HASH_ITERATIONS,
        }
        save_users(users)

    token = create_access_token(canonical_username or username)
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
        return RedirectResponse(url="/app", status_code=303)
    if not os.path.exists(REGISTER_HTML):
        raise HTTPException(status_code=404, detail="Register page not found")
    response = FileResponse(REGISTER_HTML)
    if has_stale_auth_cookie(request):
        response.delete_cookie(ACCESS_COOKIE_NAME)
    return response


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
        "password_algo": PASSWORD_HASH_ALGO,
        "password_iterations": PASSWORD_HASH_ITERATIONS,
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
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(ACCESS_COOKIE_NAME)
    return response


@app.get("/")
def root():
    return RedirectResponse(url="/login", status_code=303)


@app.get("/app")
def index(request: Request):
    if not is_authenticated(request):
        return RedirectResponse(url="/login", status_code=303)
    if not os.path.exists(INDEX_HTML):
        raise HTTPException(status_code=404, detail="Frontend file not found")
    return FileResponse(INDEX_HTML)


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
    plt.savefig(os.path.join(PLOT_DIR, "prophet_forecast.png"))
    plt.close()

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

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    prophet_test = df.loc[y_test.index, ["Date"] + holiday_cols].rename(columns={"Date": "ds"})
    prophet_pred = prophet.predict(prophet_test)["yhat"].values

    p_rmse = float(np.sqrt(mean_squared_error(y_test, prophet_pred)))
    p_mae = float(mean_absolute_error(y_test, prophet_pred))

    best_model = "XGBoost" if rmse < p_rmse else "Prophet"

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

    sales_non_null = df["Sales"].dropna()
    sales_std = float(sales_non_null.std(ddof=0)) if len(sales_non_null) else 0.0
    if sales_std == 0.0 or np.isnan(sales_std):
        anomalies = df.iloc[0:0][["Date", "Sales"]]
    else:
        z_scores = pd.Series(zscore(sales_non_null), index=sales_non_null.index)
        anomalies = df.loc[z_scores[np.abs(z_scores) > 3].index, ["Date", "Sales"]]

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

    feature_importances = [
        {"feature": feat, "importance": float(imp)}
        for feat, imp in zip(features, xgb.feature_importances_)
    ]
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

    insights = [
        f"AI selected {best_model} as the best model based on lower error",
        f"Adaptive safety stock set to {int(safety_rate * 100)}%",
        anomaly_msg,
        ai_summary
    ]

    llm_payload = {
        "best_model": best_model,
        "xgboost_rmse": round(rmse, 4),
        "xgboost_mae": round(mae, 4),
        "prophet_rmse": round(p_rmse, 4),
        "prophet_mae": round(p_mae, 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "lead_time_days": lead_time_days,
        "reorder_point": round(reorder_point, 2),
        "recommended_order_qty": round(recommended_order_qty, 2),
        "next_14_day_demand": round(next_14_day_demand, 2),
        "anomaly_count": int(len(anomalies)),
        "top_drivers": top_drivers,
    }
    llm_insights = generate_ollama_insights(llm_payload)

    plt.figure(figsize=(10, 4))
    plt.plot(y_test.values, label="Actual")
    plt.plot(y_pred, label="XGBoost")
    plt.plot(prophet_pred, label="Prophet")
    plt.legend()
    plt.title("Model Comparison")
    plt.savefig(os.path.join(PLOT_DIR, "comparison.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(test_dates, inventory_required, label="Inventory Required")
    plt.plot(test_dates, y_pred_non_negative, label="Forecasted Demand")
    plt.legend()
    plt.title("AI Inventory Planning")
    plt.savefig(os.path.join(PLOT_DIR, "inventory.png"))
    plt.close()

    return {
        "rmse": rmse,
        "mae": mae,
        "p_rmse": p_rmse,
        "p_mae": p_mae,
        "best_model": best_model,
        "insights": insights,
        "llm_insights": llm_insights,
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


@app.get("/download_inventory")
def download_inventory(request: Request):
    require_auth(request)
    if not os.path.exists(INVENTORY_CSV):
        raise HTTPException(status_code=404, detail="Inventory file not found")
    return FileResponse(INVENTORY_CSV, media_type="text/csv", filename="inventory.csv")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=int(APP_PORT or "8000"), reload=True)


