# AI Inventory Forecast

FastAPI app for inventory demand forecasting and planning using XGBoost + Prophet, with authentication and optional local LLM insights via Ollama.

## What this project does
- Auth-protected web app (`/login`, `/register`, `/app`)
- Upload sales CSV and train forecasting pipelines
- Compare XGBoost vs Prophet performance
- Generate inventory planning metrics and downloadable `inventory.csv`
- Save plots under `static/plots/`
- Optionally generate narrative insights with Ollama

## Project structure
- `app.py`: FastAPI backend and ML workflow
- `static/`: frontend pages and generated artifacts
- `requirements.txt`: Python dependencies
- `train.csv`, `test.csv`, `store.csv`: sample data files
- `.env` (local, ignored by git): secrets and runtime config

## Requirements
- Python 3.10+ (3.11 recommended)
- `pip`
- (Optional) Ollama running locally if you want `llm_insights`

## Quick start
1. Create and activate a virtual environment.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Create your `.env` from the example.

```bash
# Windows PowerShell
Copy-Item .env.example .env
# macOS/Linux
cp .env.example .env
```

4. Edit `.env` and set secure values for:
- `JWT_SECRET`
- `APP_PASSWORD`

5. Start the app.

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

6. Open `http://127.0.0.1:8000/login`
- Default username: `admin`
- Password: value of `APP_PASSWORD` in your `.env`

## Environment variables
Copy `.env.example` to `.env` and update values.

| Variable | Required | Default | Description |
|---|---|---|---|
| `JWT_SECRET` | Yes | none | Strong secret for JWT signing. App will not start without this. |
| `APP_PASSWORD` | Yes | none | Initial password for default `admin` user. App will not start without this. |
| `APP_PORT` | No | `8000` | Port used when running `python app.py`. |
| `APP_ENV` | No | `development` | App environment (`development`/`production`). |
| `JWT_COOKIE_SECURE` | No | auto | Set `true` for HTTPS in production. |
| `JWT_COOKIE_SAMESITE` | No | `lax` | Cookie SameSite mode (`lax`, `strict`, `none`). |
| `OLLAMA_BASE_URL` | No | `http://127.0.0.1:11434` | Ollama server URL. |
| `OLLAMA_MODEL` | No | `llama3.1` | Model used for local LLM insights. |
| `OLLAMA_TIMEOUT_SECONDS` | No | `120` | Timeout for Ollama API calls. |

## CSV input format
Minimum required columns:
- `Date` (parseable date)
- `Sales` (numeric)

Optional columns used if present:
- `Store` (rows filtered to `Store == 1`)
- `Open` (rows filtered to `Open == 1`)
- `Promo` (defaults to `0` when missing)
- `StateHoliday` (defaults to `none` when missing)

## API routes
- `GET /login`
- `POST /login`
- `GET /register`
- `POST /register`
- `GET /app`
- `POST /upload`
- `GET /download_inventory`
- `POST /logout`
- `GET /logout`

## Notes for contributors
- `.env`, `users.json`, `static/inventory.csv`, and `static/plots/*.png` are git-ignored local artifacts.
- On first run, if `users.json` is empty/missing, the app creates a default `admin` user.
- Keep secrets only in `.env` and never commit them.
