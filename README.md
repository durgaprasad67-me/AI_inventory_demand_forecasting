AI Powered Inventory Demand Forecasting System

Overview :-
This project is an end-to-end AI powered inventory forecasting web application built using FastAPI, Prophet, and XGBoost.The system predicts future demand, calculates adaptive safety stock, generates a demand risk score, and provides intelligent reorder recommendations through a secure authenticated web interface.

Key Features
  JWT based user authentication (Login and Register)
  Hybrid forecasting engine using Prophet and XGBoost
  Automatic model selection based on lowest RMSE
  Adaptive safety stock calculation
  Demand risk scoring (0 to 100 scale)
  AI based reorder recommendations
  Anomaly detection using Z-score
  Feature importance analysis
  CSV upload and inventory export
  Automatic forecast and comparison plots
  Interactive dashboard interface

  
System Architecture :-
 Frontend (HTML + JavaScript)
  → FastAPI Backend (Authentication + API)
  → Forecasting Layer
 Prophet (Time Series Model)
 XGBoost (Structured ML Model)
  → Business Intelligence Layer
 Risk scoring
 Safety stock logic
 Reorder point calculation
 Demand volatility analysis
 → Visualization and CSV export

Forecasting Models :-
Prophet :

  Handles trend and seasonality
  Supports Weekly and yearly seasonality
  Supports holiday regressors
  XGBoost
  Uses lag features (1, 7, 14, 28 days)
  Uses rolling averages
  Uses date-based features
  Supports promo and weekend indicators
  Provides feature importance
  The system automatically selects the best model based on RMSE comparison.
  

Evaluation Metrics :-

  RMSE (Root Mean Squared Error)  
  MAE (Mean Absolute Error)

AI Business Logic :-

   Adaptive Safety Stock
   Safety stock adjusts dynamically based on forecast error ratio.
   Risk Score (0–100)
 Calculated using:
    Forecast error ratio
    Demand volatility
    Anomaly ratio
   
  Reorder Recommendation
  The system automatically calculates:
    Lead time days
    Reorder point
    Suggested order quantity
    Next 14 day demand forecast


Expected CSV Format

The uploaded CSV must contain:

Required columns:
  Date
  Sales

Optional columns:
  Store
  Promo
  StateHoliday
  Open

Technology Stack
Backend:
 FastAPI
 Uvicorn
 PyJWT
 Pandas
 NumPy
 Scikit-learn
 XGBoost
 Prophet
 CmdStanPy

Frontend:
  HTML
  CSS
  JavaScript

Visualization:
  Matplotlib

Authentication
  JWT based authentication
  HttpOnly cookies
  Secure login and register flow
  Protected dashboard routes

Installation

Clone the repository
  git clone https://github.com/yourusername/your-repo-name.git
  cd your-repo-name

Create virtual environment
  python -m venv venv
  venv\Scripts\activate (Windows)

Install dependencies
  pip install -r requirements.txt

If installing manually:
  pip install fastapi uvicorn python-multipart pandas numpy matplotlib scikit-learn xgboost prophet cmdstanpy PyJWT

  

Run the Application :
uvicorn main:app --reload

Open in browser:
http://127.0.0.1:8000

API documentation:
http://127.0.0.1:8000/docs

Output :
  The system provides:
  Model comparison results
  Demand risk analysis
  Reorder recommendations
  Feature importance table
  Inventory planning plots
  Downloadable inventory CSV


Future Improvements
  Replace SHA256 with bcrypt
  Add role-based authentication
  Deploy to cloud (AWS / Render)
  Add LightGBM comparison
  Implement time series cross validation
  Integrate PostgreSQL database
  Dockerize the application
  Add model tracking using MLflow

Project Summary :

   This project demonstrates a hybrid AI forecasting system combining Prophet and XGBoost with business intelligence logic for adaptive inventory planning, risk scoring, and automated reorder optimization using a full stack FastAPI web platform.
