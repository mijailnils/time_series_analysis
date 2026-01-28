# pandas_startup.py
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio

pd.set_option("display.min_rows", 4)
pd.set_option("display.max_rows", 7)

plt.rcParams["figure.autolayout"] = True

pio.templates.default = "plotly_dark"
pd.options.plotting.backend = "plotly"
pio.templates["plotly_dark"].layout.update(
    width=1000, height=500, showlegend=True, autosize=False, hovermode='x unified'
)
