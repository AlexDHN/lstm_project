#%%
import yfinance as yf
import pandas as pd

# .CSV generated from https://en.wikipedia.org/wiki/OMX_Nordic_40.
# Tickers were tweaked by hand in the .CSV to match Yahoo!Finance
stock_data = pd.read_csv("OMXN40.csv", sep=";")
stock_names = list(stock_data.Ticker)
stock_companies = list(stock_data.Company)

# Cleaning company names
stock_companies = [e.split(" (")[0] for e in stock_companies]

# Data query from yfinance
raw_query = yf.download(
    tickers=stock_names,
    period="30y",
    interval="1d",
    ignore_tz=True,
    group_by="ticker",
    threads=True,
)

# Only keeping the close prices
data = raw_query[[(stock, "Close") for stock in stock_names]]

# Renaming columns
data.columns = stock_companies

# Cleaning steps are done using suggestions from autoviz.
# Code snippet:
# > !pip install autoviz
# > from autoviz import data_cleaning_suggestions
# > data_cleaning_suggestions(data)

# First suggestion: dropping columns with insufficient history
data = data.drop(
    [
        "Sandvik",
        "Essity",
        "Chr. Hansen",
        "Ørsted",
        "DSV",
        "Atlas Copco",
        "GN Store Nord",
        "Telia",
        "Vestas Wind Systems",
        "KONE",
        "Neste",
        "Ambu",
    ],
    axis=1,
    errors="ignore",
)

# Second, remove outliers from one of the columns
data["Coloplast"] = data["Coloplast"][
    (data["Coloplast"] / data["Coloplast"].shift(1) - 1).abs() < 0.5
]

# Third suggestion : dropping rows because almost no stocks have data in the first rows
data = data.iloc[1000:-1]

# Next, imputation
data = data.ffill()

# Finally, convert to log-returns
data = data / data.shift(1) - 1

# Remove the first row after converting to log-returns
data = data.iloc[1:]

# Save the dataset
data.to_csv("dataset.csv")

# To read the dataset:
# > pd.read_csv("dataset.csv", index_col=0)
