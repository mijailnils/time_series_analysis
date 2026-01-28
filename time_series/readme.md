# Time Series Analysis - Learning Journey

Personal repository for learning time series analysis in Python, following LinkedIn Learning courses.

## 📚 Courses

- [Practical Python for Time Series Analysis](https://www.linkedin.com/learning/practical-python-for-time-series-analysis) - Jesus Lopez
- [Python for Time Series Forecasting](https://www.linkedin.com/learning/python-for-time-series-forecasting)

## 🚀 Quick Start (Windows)

### Prerequisites
- Python 3.10+
- Git Bash or Windows Terminal
- VSCode with Python extension

### Setup
```bash
# 1. Clone the repo
git clone https://github.com/mijailnils/time_series_analysis.git
cd time_series_analysis

# 2. Create virtual environment
python -m venv .venv

# 3. Activate (Git Bash)
source .venv/Scripts/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Open in VSCode
code .
```

## 📁 Project Structure
```
time_series_analysis/
├── data/                    # Datasets
│   ├── EIA/                # Energy Information Administration
│   ├── FRED/               # Federal Reserve Economic Data
│   ├── UCIrvine/           # UCI ML Repository
│   └── YFINANCE/           # Yahoo Finance
├── modules/                 # Utility functions
├── notebooks/               # Course notebooks
│   ├── 1_Foundation/       
│   ├── 2_Aggregation/      
│   └── 3_Regression/       
├── requirements.txt         
└── README.md
```

## 📈 Progress Tracker

### Course 1: Practical Python for Time Series Analysis

#### Module 1: Foundation
- [ ] Working with datetime indices
- [ ] Time series visualization
- [ ] Joining temporal datasets

#### Module 2: Aggregation
- [ ] GroupBy operations for time series
- [ ] Pivot tables with temporal data
- [ ] Resampling and frequency conversion

#### Module 3: Regression
- [ ] Simple linear regression with time series
- [ ] Categorical variables in temporal models
- [ ] Feature engineering for time-based predictions

### Course 2: Python for Time Series Forecasting

- [ ] ARIMA / SARIMA
- [ ] Exponential Smoothing (Holt-Winters)
- [ ] Prophet
- [ ] Model evaluation metrics (MAE, RMSE, MAPE)

## 📝 Key Concepts

- **Stationarity**: Mean and variance constant over time
- **Seasonality**: Repeating patterns at fixed intervals
- **Trend**: Long-term increase or decrease
- **Autocorrelation**: Correlation of series with lagged version of itself

## 🔗 Resources

- [Statsmodels Docs](https://www.statsmodels.org/stable/tsa.html)
- [Prophet Docs](https://facebook.github.io/prophet/)
- [Pandas Time Series Guide](https://pandas.pydata.org/docs/user_guide/timeseries.html)

---
**Author**: Mijail | **Started**: January 2026
```

3. Guardá (`Ctrl+S`)

4. Hacé lo mismo para `requirements.txt`:
```
pandas>=2.1.0
numpy>=1.26.0
pyarrow
fastparquet
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0
statsmodels>=0.14.0
scipy>=1.11.0
scikit-learn>=1.3.0
prophet>=1.1.5
ipykernel
nbformat
nbconvert
ruff