import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import statsmodels.api as sm
from prophet import Prophet
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def configure_plotly_template(showlegend=False, width=1000, height=500):
    pio.templates.default = "plotly_dark"
    pd.options.plotting.backend = "plotly"
    pio.templates["plotly_dark"].layout.update(
        width=width, height=height, showlegend=showlegend, autosize=False
    )


def add_time_features(df):
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["weekend"] = df.index.dayofweek > 4
    return df


def collect_lr_results(fit_results: dict) -> pd.DataFrame:
    
    rows = []
    for name, model in fit_results.items():
        summary = model.summary2().tables
        coef = summary[1].loc["x1", "Coef."]
        pval = summary[1].loc["x1", "P>|t|"]
        stderr = summary[1].loc["x1", "Std.Err."]
        tval = summary[1].loc["x1", "t"]
        r2 = model.rsquared
        nobs = int(model.nobs)
        rows.append(
            {
                "Period": name,
                "Coef": coef,
                "StdErr": stderr,
                "t": tval,
                "p": pval,
                "R²": r2,
                "n": nobs,
            }
        )
    return pd.DataFrame(rows).set_index("Period")


def plot_residuals_histogram_with_normal_density(residuals):
    # Theoretical normal density
    x_vals = np.linspace(residuals.min(), residuals.max(), 100)
    y_vals = norm.pdf(x_vals, loc=0, scale=residuals.std())

    # Create figure
    plt.figure(figsize=(8, 6))

    # Add histogram
    plt.hist(
        residuals, bins=20, density=True, color="skyblue", alpha=0.75, label="Residuals"
    )

    # Add theoretical normal density
    plt.plot(x_vals, y_vals, color="red", linewidth=2, label="Normal Density")

    # Update layout
    plt.title("Histogram of Residuals with Normal Density Overlay")
    plt.xlabel("Residuals")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Example usage:
# plot_residuals_histogram_with_normal_density(df)


def plot_residuals_lag(residuals):
    plt.scatter(residuals[:-1], residuals[1:])
    plt.xlabel("Residual t-1")
    plt.ylabel("Residual t")
    plt.title("Residuals vs Lagged Residuals")
    plt.axhline(0, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.show()


def plot_residuals_vs_fitted(fitted, residuals):
    plt.figure(figsize=(8, 4))
    sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={"color": "red"})

    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")

    plt.tight_layout()
    plt.show()


def linear_regression_by_category(
    df: pd.DataFrame, category_column: str, target_column: str, feature_columns: list
):
    model_dict = {}
    for category in df[category_column].unique():
        df_filtered = df[df[category_column] == category].copy()
        df_filtered["intercept"] = 1

        X = df_filtered[["intercept", *feature_columns]]
        y = df_filtered[target_column]

        model = sm.OLS(y, X).fit()

        model_dict[category] = model

    return model_dict


def get_model_forecast(
    data_frame,
    column,
    order=(12, 1, 2),
    seasonal_order=None,
    horizon=48,
    column_name=None,
    forecast_exp=False,
    historical_predictions=True,
):
    df = data_frame.copy()
    series = df[column].dropna()

    if forecast_exp:
        series = np.log(series)

    p, d, q = order

    if seasonal_order is None:
        model = ARIMA(series, order=(p, d, q))
        model_name = f"ARIMA({p},{d},{q})"
    else:
        P, D, Q, m = seasonal_order
        model = SARIMAX(series, order=(p, d, q), seasonal_order=(P, D, Q, m))
        model_name = f"SARIMA({p},{d},{q})({P},{D},{Q},{m})"

    model_fit = model.fit()

    end = len(series) + horizon - 1
    if historical_predictions:
        forecast = model_fit.predict(start=series.index[0], end=end)
    else:
        forecast = model_fit.predict(start=len(series), end=end)

    if forecast_exp:
        forecast = np.exp(forecast)

    if column_name:
        model_name = column_name

    df_forecast = forecast.to_frame(name=model_name)
    df_combined = pd.concat([df, df_forecast], axis=1)

    return df_combined


def get_train_test_forecast(
    df_train,
    df_test,
    column,
    order=(12, 1, 2),
    seasonal_order=None,
    column_names=None,
    prediction_real=True,
    return_df=None,
):
    if return_df is None:
        return_df = ["Train", "Test"]
    train_series = df_train[column].dropna()
    test_series = df_test[column].dropna()
    p, d, q = order

    if seasonal_order is None:
        model = ARIMA(train_series, order=(p, d, q))
        model_name = f"ARIMA({p},{d},{q})"
    else:
        P, D, Q, m = seasonal_order
        model = SARIMAX(train_series, order=(p, d, q), seasonal_order=(P, D, Q, m))
        model_name = f"SARIMA({p},{d},{q})({P},{D},{Q},{m})"

    model_fit = model.fit()

    start, end = train_series.index[[0, -1]]
    forecast_train = model_fit.predict(start=start, end=end)

    start, end = test_series.index[[0, -1]]
    forecast_test = model_fit.predict(start=start, end=end)

    if column_names:
        model_name = column_names

    if prediction_real:
        forecast_train = np.exp(forecast_train)
        forecast_test = np.exp(forecast_test)

    df_train = pd.concat([df_train, forecast_train.to_frame(name=model_name)], axis=1)
    df_test = pd.concat([df_test, forecast_test.to_frame(name=model_name)], axis=1)

    dfs = {
        "Train": df_train,
        "Test": df_test,
    }

    if len(return_df) == 1:
        return dfs[return_df[0]]

    t = ()
    for r in return_df:
        t += (dfs[r],)

    return t


def preprocess_concat_diff(vtype, path):
    df = pd.read_parquet(path)
    df.columns = ["values"]
    df["vtype"] = vtype

    r = df["values"].diff().to_frame(name="values").assign(diff=True, vtype=vtype)
    df = pd.concat([df.assign(diff=False), r], axis=0)

    return df


def plot_residuals_histogram(residuals, bins=50):
    # Calculate histogram data
    counts, bin_edges = np.histogram(residuals, bins=bins, density=True)

    # Plot histogram of residuals using matplotlib
    plt.hist(
        bin_edges[:-1],
        bin_edges,
        weights=counts,
        alpha=0.6,
        color="g",
        label="Residuals",
    )

    # Plot the bell curve
    mu, std = residuals.mean(), residuals.std()
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = np.exp(-0.5 * ((x - mu) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    plt.plot(x, p, "k", linewidth=2, label="Bell Curve")

    # Add mean and standard deviation lines
    plt.axvline(mu, color="r", linestyle="dashed", linewidth=1, label=f"Mean: {mu:.2f}")
    plt.axvline(
        mu + std,
        color="b",
        linestyle="dashed",
        linewidth=1,
        label=f"Std Dev: {std:.2f}",
    )
    plt.axvline(mu - std, color="b", linestyle="dashed", linewidth=1)

    plt.title("Histogram of Residuals with Bell Curve")
    plt.xlabel("Residuals")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def get_model_forecast_exponential_smoothing(
    data_frame,
    column,
    horizon=48,
    column_name=None,
    class_config=None,
    fit_config=None,
):
    series = data_frame[column].dropna()

    # Default configurations
    class_config = class_config or {}
    fit_config = fit_config or {}

    # Create and fit the model
    model = ExponentialSmoothing(
        series,
        trend=class_config.get("trend"),
        seasonal=class_config.get("seasonal"),
        seasonal_periods=class_config.get("seasonal_periods"),
        damped_trend=class_config.get("damped_trend"),
    ).fit(
        smoothing_level=fit_config.get("smoothing_level"),
        smoothing_slope=fit_config.get("smoothing_slope"),
        smoothing_seasonal=fit_config.get("smoothing_seasonal"),
        damping_slope=fit_config.get("damping_slope"),
    )

    # Forecast and combine with original data
    forecast = model.forecast(steps=horizon)
    model_name = column_name or "ExponentialSmoothing"
    df_forecast = forecast.to_frame(name=model_name)

    return pd.concat([data_frame, df_forecast], axis=1)


def plot_prophet_forecast(series, df_prophet, title=None):
    df_prophet = df_prophet.set_index("ds")
    series = series.to_frame(name="historical")
    df = pd.concat([series, df_prophet], axis=1)
    # Rename columns

    # Ensure no negative values in 'FC Lower'
    df["yhat_lower"] = np.where(df["yhat_lower"] < 0, 0, df["yhat_lower"])

    fig = go.Figure()

    # Add lines
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["historical"],
            name="Historical",
            mode="lines+markers",
            line={"color": "#fa636e", "width": 1.5},
            marker={"color": "#fa636e", "size": 5, "symbol": "circle"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["yhat"].round(2),
            name="Forecast",
            line={"color": "#FFD700", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["yhat_upper"].round(2),
            name="Forecast Interval Upper Bound",
            line={"color": "rgba(212, 212, 217, 0.0)", "width": 0},
            showlegend=True,
            fill=None,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["yhat_lower"].round(2),
            name="Forecast Interval Lower Bound",
            line={"color": "rgba(212, 212, 217, 0.0)", "width": 0},
            fill="tonexty",
            fillcolor="rgba(212, 212, 217, 0.5)",
            showlegend=True,
        )
    )

    # Chart layout updates
    fig.update_layout(
        title=title,
        template="plotly_dark",  # Keep the dark background style
        hovermode="x unified",  # Update hover layout
        xaxis={"range": [df.index.min(), df_prophet.index.max()]},
    )

    fig


class TimeSeriesForecaster:
    def __init__(
        self,
        series=None,
        train=None,
        test=None,
        test_size=0.3,
        freq="ME",
        idx_offset=13,
    ):
        self.freq = freq
        self.idx_offset = idx_offset
        self.last_forecast_df = None  # Stores the last forecast DataFrame
        self.last_combined_df = (
            None  # Stores the last combined (historical + forecast) DataFrame
        )
        if series is not None:
            series = series.asfreq(self.freq)
            self.train, self.test = train_test_split(
                series, test_size=test_size, shuffle=False
            )
        elif train is not None and test is not None:
            self.train = train.asfreq(self.freq)
            self.test = test.asfreq(self.freq)
        else:
            raise ValueError("Provide either `series` or both `train` and `test`.")

    def sarima(self, model_params=None, forecast_exp=False, log_transform=False):
        """
        SARIMA forecast. If log_transform is True, log-transform the data before fitting and exponentiate the forecast.
        """
        model_params = model_params or {
            "order": (0, 1, 1),
            "seasonal_order": (0, 1, 1, 12),
        }
        train = np.log(self.train) if log_transform else self.train
        test = np.log(self.test) if log_transform else self.test
        model = SARIMAX(train, **model_params).fit()
        forecast_train = model.predict(start=train.index[0], end=train.index[-1])
        forecast_test = model.predict(start=test.index[0], end=test.index[-1])
        if log_transform or forecast_exp:
            forecast_train = np.exp(forecast_train)
            forecast_test = np.exp(forecast_test)
        return forecast_train, forecast_test

    def ets(self, model_params=None, log_transform=False):
        """
        Exponential Smoothing forecast. If log_transform is True, log-transform the data before fitting and exponentiate the forecast.
        """
        model_params = model_params or {
            "trend": "add",
            "seasonal": "mul",
            "seasonal_periods": 12,
        }
        train = np.log(self.train) if log_transform else self.train
        test = np.log(self.test) if log_transform else self.test
        model = ExponentialSmoothing(train, **model_params).fit()
        forecast_train = model.predict(start=train.index[0], end=train.index[-1])
        forecast_test = model.predict(start=test.index[0], end=test.index[-1])

        if log_transform:
            forecast_train = np.exp(forecast_train)
            forecast_test = np.exp(forecast_test)
        return forecast_train, forecast_test

    def prophet(self, model_params=None, log_transform=False):
        """
        Prophet forecast. If log_transform is True, log-transform the data before fitting and exponentiate the forecast.
        """
        model_params = model_params or {
            "yearly_seasonality": True,
            "weekly_seasonality": False,
            "daily_seasonality": False,
            "seasonality_mode": "multiplicative",
        }

        train = np.log(self.train) if log_transform else self.train
        test = np.log(self.test) if log_transform else self.test

        df_train = pd.DataFrame({"ds": train.index, "y": train.values})

        model = Prophet(**model_params)
        model.fit(df_train)

        test_size = len(test)
        future = model.make_future_dataframe(periods=test_size, freq=self.freq)
        forecast = model.predict(future).set_index("ds").asfreq(self.freq)

        forecast_train = forecast.iloc[:-test_size]["yhat"]
        forecast_test = forecast.iloc[-test_size:]["yhat"]

        if log_transform:
            forecast_train = np.exp(forecast_train)
            forecast_test = np.exp(forecast_test)
        return forecast_train, forecast_test

    def bulk_forecast(self, configs, metrics=None):
        """
        Run multiple models and return a DataFrame with forecasts and metrics.
        Stores the result in self.last_forecast_df and returns it.
        :param configs: dict of model configs
        :param metrics: dict of metric functions, e.g. {'rmse': root_mean_squared_error}
        :return: pd.DataFrame
        """
        results = []
        metrics = metrics or {}

        for model_name, config in configs.items():
            forecaster = getattr(self, model_name)
            f_train, f_test = forecaster(**config)
            forecast = {"train": f_train, "test": f_test}

            for split in ["train", "test"]:
                # Use idx_offset only for train split
                idx = self.idx_offset if split == "train" else 0
                data_real = getattr(self, split)[idx:]
                data_forecast = forecast[split][idx:]

                row = {
                    "model": model_name,
                    "split": split,
                    "values": data_forecast.values,
                    "datetime": data_forecast.index,
                }
                # Calculate all metrics
                for metric_name, metric_func in metrics.items():
                    row[metric_name] = metric_func(data_real, data_forecast)
                results.append(row)
        self.last_forecast_df = pd.DataFrame(results)
        return self.last_forecast_df

    def combine_with_historical(self, df_forecast=None, idx_offset=None):
        """
        Combine exploded forecast DataFrame with historical real data for both train and test splits.
        Stores the result in self.last_combined_df and returns it.
        :param df_forecast: DataFrame from bulk_forecast, exploded so each row is a datetime-value pair. If None, uses self.last_forecast_df.
        :param idx_offset: Optionally override self.idx_offset for train split
        :return: Combined DataFrame with columns: model, split, datetime, values
        """
        if df_forecast is None:
            df_forecast = getattr(self, "last_forecast_df", None)
            if df_forecast is None:
                raise ValueError(
                    "No forecast DataFrame provided or stored in the class."
                )
        idx_offset = self.idx_offset if idx_offset is None else idx_offset

        # Explode forecast DataFrame
        r = df_forecast.set_index(["model", "split"])[["datetime", "values"]]
        df_forecast_exploded = r.explode(["datetime", "values"])
        df_forecast_exploded = df_forecast_exploded.set_index(
            "datetime", append=True
        ).sort_index()

        # Prepare historical dataframes
        dfs = {"train": self.train, "test": self.test}

        for split in ["train", "test"]:
            idx = idx_offset if split == "train" else 0
            r = (
                dfs[split]
                .iloc[idx:]
                .to_frame(name="values")
                .reset_index()
                .rename(columns={dfs[split].index.name or "index": "datetime"})
                .assign(model="historical", split=split)
                .set_index(["model", "split", "datetime"])
            )
            df_forecast_exploded = pd.concat([df_forecast_exploded, r], axis=0)

        self.last_combined_df = df_forecast_exploded.reset_index()
        return self.last_combined_df


def get_model_forecast_prophet(
    df, column, column_name=None, mode="multiplicative", horizon=48, **kwargs
):
    r = df[[column]].dropna()

    df_prophet = pd.DataFrame(
        {
            "ds": r.index,
            "y": r[column].values,
        },
        index=r.index,
    )

    model = Prophet(seasonality_mode=mode, **kwargs)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=horizon, freq="ME")
    forecast = model.predict(future)

    if column_name is None:
        column_name = f"prophet_{mode}"

    df_forecast = forecast.set_index("ds")[["yhat"]].rename(
        columns={"yhat": column_name}
    )

    idx_forecast = df_forecast.index[-horizon:]
    df_forecast = df_forecast.loc[idx_forecast]

    return pd.concat([df, df_forecast], axis=1)


def plot_decomposition_comparison(series: pd.Series, period: int = 12) -> px.line:
    """
    Plot seasonal decomposition (additive and multiplicative) of a given time series using Plotly.

    Parameters:
        series (pd.Series): A time series with a DatetimeIndex.
        period (int): Seasonal period (e.g., 12 for monthly data with yearly seasonality).

    Returns:
        plotly.graph_objects.Figure: The decomposition visualization.
    """
    dfs = {}

    for model in ["additive", "multiplicative"]:
        result = sm.tsa.seasonal_decompose(series, model=model, period=period)
        r = (
            series.to_frame(name="values")
            .assign(trend=result.trend, seasonal=result.seasonal, residual=result.resid)
            .dropna()
        )
        r["model_result"] = (
            r.trend + r.seasonal + r.residual
            if model == "additive"
            else r.trend * r.seasonal * r.residual
        )
        dfs[model] = r

    df_combined = pd.concat(dfs, axis=1).melt(ignore_index=False).reset_index()
    df_combined.columns = ["month", "model", "component", "value"]

    fig = px.line(
        data_frame=df_combined,
        x="month",
        y="value",
        color="component",
        facet_col="model",
        facet_row="component",
        width=1500,
        height=1000,
        facet_col_spacing=0.1,
    )
    fig.update_yaxes(matches=None)
    for attr in dir(fig.layout):
        if attr.startswith("yaxis"):
            axis = getattr(fig.layout, attr)
            if axis:
                axis.showticklabels = True
    return fig

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def plot_explanatory_regression_grid(
    df: pd.DataFrame,
    target: str,
    explanatory_baseline: str,
    explanatory_transformed: str,
    categorical: str | None = None,
    title: str | None = None,
    period_colors: list | None = None,
    show_period_labels: bool = True,
    explanatory_colors: tuple[str, str] | None = None,  # high-contrast colors for baseline & transformed
):
    """
    2x2 Plotly layout:
      • Top (spans both cols): target on LEFT Y (dotted), baseline+transformed on RIGHT Y (solid, contrasting colors),
        with contiguous vrect bands for periods taken from `categorical` order.
        - If `df[categorical]` is CategoricalDtype, use its categories order.
        - Else, use order of first appearance over time.
      • Bottom: two regression panels with per-period OLS lines.

    Assumes df.index is datetime-like.
    """
    # Ensure chronological order and datetime index
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    # Default colors for the two explanatory series (distinct)
    if explanatory_colors is None:
        explanatory_colors = ("#1f77b4", "#ff7f0e")

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"secondary_y": True, "colspan": 2}, None], [{}, {}]],
        subplot_titles=[
            "Target (left Y) + Explanatory (right Y)",
            f"Regression: {target} vs {explanatory_baseline}",
            f"Regression: {target} vs {explanatory_transformed}",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # --- Top panel traces ---
    # Target -> dotted line on left Y
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df[target],
            mode="lines",
            name=target,
            line=dict(dash="dot", width=2),
        ),
        row=1, col=1, secondary_y=False,
    )

    # Explanatory baseline -> solid, high-contrast color on right Y
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df[explanatory_baseline],
            mode="lines",
            name=explanatory_baseline,
            line=dict(width=2, color=explanatory_colors[0]),
        ),
        row=1, col=1, secondary_y=True,
    )

    # Explanatory transformed -> solid, contrasting color on right Y
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df[explanatory_transformed],
            mode="lines",
            name=explanatory_transformed,
            line=dict(width=2, color=explanatory_colors[1]),
        ),
        row=1, col=1, secondary_y=True,
    )

    # --- Contiguous vrects for periods (top chart only) ---
    if categorical and categorical in df.columns:
        series = df[categorical]

        # 1) Decide the category order
        if pd.api.types.is_categorical_dtype(series):
            cats = list(series.cat.categories)
        else:
            # order by first appearance over time
            first_times = (
                df.groupby(categorical, sort=False)
                  .apply(lambda g: g.index.min())
                  .sort_values()
            )
            cats = list(first_times.index)

        # 2) Build (category, start_time) list following chosen order
        starts = []
        for cat in cats:
            sub = df[series == cat]
            if not sub.empty:
                starts.append((cat, sub.index.min()))
        # Filter out categories not present after any prior filtering
        starts = [t for t in starts if pd.notna(t[1])]
        # Already in the desired order; ensure sorted by time to compute edges
        starts_sorted_by_time = sorted(starts, key=lambda t: t[1])

        # 3) Default colors if none provided
        if period_colors is None:
            base_colors = ["#e5f5e0", "#fee6ce", "#deebf7", "#fdd0a2", "#c6dbef"]
            period_colors = [base_colors[i % len(base_colors)] for i in range(len(starts_sorted_by_time))]

        # 4) Add contiguous vrects: x1 = next start (last goes to max index)
        for i, (cat, x0) in enumerate(starts_sorted_by_time):
            x1 = starts_sorted_by_time[i + 1][1] if i < len(starts_sorted_by_time) - 1 else df.index.max()
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor=period_colors[i],
                opacity=0.3,
                line_width=0,
                row=1, col=1,  # only the top subplot
                annotation_text=str(cat) if show_period_labels else None,
                annotation_position="top left",
            )

    # --- Axis labels (top) ---
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text=target, row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text=f"{explanatory_baseline} / {explanatory_transformed}", row=1, col=1, secondary_y=True)

    # --- Bottom regressions with per-period fits ---
    color_arg = categorical if (categorical and categorical in df.columns) else None
    trend_scope = "trace" if color_arg else "overall"  # one OLS per period if colored

    # Left: baseline
    fig_base = px.scatter(
        df, x=explanatory_baseline, y=target,
        color=color_arg, trendline="ols", trendline_scope=trend_scope
    )
    for tr in fig_base.data:
        fig.add_trace(tr, row=2, col=1)

    # Right: transformed
    fig_trans = px.scatter(
        df, x=explanatory_transformed, y=target,
        color=color_arg, trendline="ols", trendline_scope=trend_scope
    )
    for tr in fig_trans.data:
        fig.add_trace(tr, row=2, col=2)

    # Axis labels (bottom)
    fig.update_xaxes(title_text=explanatory_baseline, row=2, col=1)
    fig.update_yaxes(title_text=target, row=2, col=1)
    fig.update_xaxes(title_text=explanatory_transformed, row=2, col=2)
    fig.update_yaxes(title_text=target, row=2, col=2)

    # Layout
    fig.update_layout(
        height=850, width=1150,
        title_text=title or f"Baseline vs Transformation — Target: {target}",
        legend_tracegroupgap=10, margin=dict(t=90),
    )
    
    # --- Legend: horizontal at top, no duplicates, no trendline items ---
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.07,          # slightly above the top subplot
            xanchor="center",
            x=0.5,
            title_text=""
        ),
        margin=dict(t=120)   # give space for the legend above the title
    )

    seen = set()
    for tr in fig.data:
        # Hide PX trendline legend entries (names like "Before trendline")
        if isinstance(tr.name, str) and tr.name.endswith(" trendline"):
            tr.showlegend = False
            continue

        # Normalize PX names like "period=Before" -> "Before"
        if isinstance(tr.name, str) and "=" in tr.name:
            tr.name = tr.name.split("=", 1)[-1]

        # Deduplicate any remaining repeated names
        if tr.name in seen:
            tr.showlegend = False
        else:
            seen.add(tr.name)

    return fig