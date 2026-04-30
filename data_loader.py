import pandas as pd
import numpy as np

# Loading csv
df = pd.read_csv("dane.csv", header = [0, 1], index_col = 0)

# Log returns
prices = df["Close"]
log_ret = (np.log(prices / prices.shift(1))).dropna()
print(log_ret)

# Mean return vector
mu = log_ret.mean() * 252
print(mu)

# Covariance matrix
sigma = log_ret.cov() * 252
print(sigma)