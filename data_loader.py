import os
import pandas as pd
import numpy as np

# Zapewnienie stabilnego wczytywania danych niezależnie od katalogu roboczego
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "dane.csv")

# Wczytanie danych z pliku CSV
df = pd.read_csv(csv_path, header=[0, 1], index_col=0)

# Dzienne logarytmiczne stopy zwrotu
prices = df["Close"]
log_ret = np.log(prices / prices.shift(1)).dropna()

# Annualizowany wektor oczekiwanych zwrotów (252 sesje giełdowe w roku)
mu = log_ret.mean() * 252

# Annualizowana macierz kowariancji
sigma = log_ret.cov() * 252

if __name__ == "__main__":
    print("=== DZIENNE STOPY ZWROTU ===")
    print(log_ret.head())
    print(f"\nWymiary danych: {log_ret.shape}")
    print("\n=== ANNUALIZOWANY WEKTOR ZWROTÓW (mu) ===")
    print(mu)
    print("\n=== ANNUALIZOWANA MACIERZ KOWARIANCJI (sigma) ===")
    print(sigma)