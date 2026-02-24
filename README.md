---
title: AI Inventory Forecast
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

## Deploy This FastAPI Project on Hugging Face Spaces

This Space runs a FastAPI app with Uvicorn.

### Default login (for Space demo mode)
- Username: `admin`
- Password: `Demo@123!`

Set Secrets in Space settings to override defaults:
- `JWT_SECRET` (required for production)
- `APP_PASSWORD` (or `HF_DEMO_PASSWORD`)
- Optional: `APP_USERNAME`, `JWT_EXPIRE_MINUTES`, `JWT_COOKIE_SECURE`

### Local run
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```
