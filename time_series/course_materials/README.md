# Practical Python for Time Series Analysis

Welcome to the course repository for **Practical Python for Time Series Analysis** on LinkedIn Learning.

![lil-thumbnail-url]

## Course Overview

In this course, Python and data scientist Jesus Lopez focuses on practical time series analysis in Python, emphasizing explainability and discovering causal relationships. Discover methods for understanding why patterns emerge, what variables drive changes over time, and how to interpret results responsibly. All topics are taught through hands-on examples, mainly from the energy sector, so you can apply insights directly to real-world data. Build the understanding you need to transform raw time series data into meaningful explanations that inform strategic decisions.

### What You'll Learn

- **Time Series Fundamentals**: Understanding time-indexed data structures in pandas
- **Data Aggregation**: Mastering groupby, pivot, and resample operations for time series
- **Statistical Analysis**: Applying regression techniques to temporal data
- **Forecasting Models**: Implementing SARIMA, Exponential Smoothing, and Prophet
- **Practical Applications**: Working with real-world datasets from energy, finance, and economics

## Prerequisites

- Basic Python knowledge
- Familiarity with pandas and NumPy
- Understanding of basic statistics

## Setup Instructions

### Option 1: Using GitHub Codespaces (Recommended)

1. Click the "Code" button on this repository
2. Select "Open with Codespaces"
3. Click "New codespace"
4. Wait for the environment to build (this may take a few minutes)
5. The devcontainer will automatically install all dependencies using `uv` (ultra-fast Python package manager)

### Option 2: Using VS Code with Dev Containers

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. Install [VS Code](https://code.visualstudio.com/)
3. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
4. Clone this repository
5. Open the repository in VS Code
6. Click "Reopen in Container" when prompted

### Option 3: Local Installation with uv (Recommended)

1. Install [uv](https://github.com/astral-sh/uv):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone this repository
3. Create a virtual environment and install dependencies:

   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install pandas pyarrow fastparquet plotly nbformat nbconvert seaborn scikit-learn statsmodels ipykernel matplotlib numpy scipy prophet ruff jupyter notebook
   ```

4. Install Jupyter kernel:

   ```bash
   python -m ipykernel install --user --name time-series-analysis
   ```

### Option 4: Local Installation with pip

1. Ensure Python 3.10+ is installed
2. Clone this repository
3. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:

   ```bash
   pip install pandas pyarrow fastparquet plotly nbformat nbconvert seaborn scikit-learn statsmodels ipykernel matplotlib numpy scipy prophet ruff jupyter notebook
   ```

5. Install Jupyter kernel:

   ```bash
   python -m ipykernel install --user --name time-series-analysis
   ```

## Repository Structure

```
practical-python-for-time-series-analysis/
├── data/                  # Datasets organized by source
│   ├── EIA/              # Energy Information Administration data
│   ├── FRED/             # Federal Reserve Economic Data
│   ├── UCIrvine/         # UCI Machine Learning Repository data
│   └── YFINANCE/         # Yahoo Finance data
├── modules/              # Shared utility functions
│   └── utils.py          # Time series analysis helpers
├── notebooks/            # Course materials
│   ├── 1_Foundation/     # Basic time series concepts
│   ├── 2_Aggregation/    # Data transformation techniques
│   └── 3_Regression/     # Statistical modeling
└── pyproject.toml        # Project configuration and dependencies
```

## Course Modules

### 1. Foundation

- Working with datetime indices
- Time series visualization
- Joining temporal datasets

### 2. Aggregation

- GroupBy operations for time series
- Pivot tables with temporal data
- Resampling and frequency conversion

### 3. Regression

- Simple linear regression with time series
- Categorical variables in temporal models
- Feature engineering for time-based predictions

## Datasets

The course uses real-world datasets from various domains:

- **Energy**: California electricity demand and generation data
- **Economics**: Federal Reserve economic indicators (CPI, unemployment, etc.)
- **Finance**: Stock and ETF price data
- **Classic**: Airline passenger data for demonstration

## Dependencies

Key Python packages used in this course:

- **pandas**: Time series data manipulation
- **statsmodels**: Statistical modeling and forecasting
- **plotly**: Interactive visualizations
- **scikit-learn**: Machine learning utilities
- **seaborn**: Statistical plotting

## Getting Help

- Review the notebook examples in each module
- Check the `modules/utils.py` file for reusable functions
- Refer to the official documentation of the packages used

## License

This project is licensed under the terms included in the LinkedIn Learning course materials.

## Instructor

Jesus Lopez

Python and Data Scientist, Trainer, Consultant             

Check out my other courses on [LinkedIn Learning](https://www.linkedin.com/learning/instructors/jesus-lopez?u=104).


[0]: # (Replace these placeholder URLs with actual course URLs)

[lil-course-url]: https://www.linkedin.com/learning/practical-python-for-time-series-analysis
[lil-thumbnail-url]: https://media.licdn.com/dms/image/v2/D4E0DAQFVR6lftIVw2Q/learning-public-crop_675_1200/B4EZq4yo3wIwAY-/0/1764036874550?e=2147483647&v=beta&t=uzYxeJaNvPBgSHd8mUqr2V31v8_CEv1oAsskoBI2DPs


