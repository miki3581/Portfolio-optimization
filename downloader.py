import yfinance as yf

tick_list = ["PKN.WA", "PKO.WA", "PEO.WA", "KGH.WA", "PZU.WA", "LPP.WA", "ALE.WA", "SPL.WA",
              "CDR.WA", "DNP.WA", "MBK.WA", "PCO.WA", "ALR.WA", "TPE.WA", "PGE.WA", "BDX.WA"]

df = yf.download(tick_list, period = "3y")

df.to_csv("dane.csv")
