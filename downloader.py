import yfinance as yf

# Ticker lists
tick_list = ["PKN.WA", "PKO.WA", "PEO.WA", "KGH.WA", "PZU.WA", "LPP.WA", "ALE.WA", "SPL.WA",
              "CDR.WA", "DNP.WA", "MBK.WA", "ALR.WA", "TPE.WA", "PGE.WA", "BDX.WA"]

# Downloading data
df = yf.download(tick_list, period = "3y")

# Saving data to .csv
df.to_csv("dane.csv")
