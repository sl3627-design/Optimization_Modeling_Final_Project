import json
import pandas as pd 

# 1. Load the dictionary back into memory
with open("stock_id_to_ticker.json", "r") as f:
    id_to_ticker_dict = json.load(f)


df = pd.read_csv("deterministic_asset_selection.csv")

selected_assets = df["stock_id"]

real_tickers = [id_to_ticker_dict[stock_id] for stock_id in selected_assets]


with open("asset_selection.txt", "w") as f:
    for i in range(len(selected_assets)//40):
        line = f"Asset Selection for {2019+i} : {real_tickers[40*i:40*(i+1)]}\n"
        f.write(line)