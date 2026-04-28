import json
import pandas as pd

# 1. Load the dictionary and reverse it (ticker -> stock_id)
with open("stock_id_to_ticker.json", "r") as f:
    id_to_ticker_dict = json.load(f)

ticker_to_id_dict = {v: k for k, v in id_to_ticker_dict.items()}  # Reversed

# df = pd.read_csv("deterministic_asset_selection.csv")

selected_tickers = ['TEL', 'HCA', 'MNST', 'YUM', 'VRSN', 'DVN', 'ROK', 'JNJ', 'GOOGL', 'AAPL', 'V', 'AMAT', 'APA', 'APP', 'ROL', 'EQT', 'NDSN', 'WMT', 'ORLY', 'ADM', 'ABBV', 'VRTX', 'WELL', 'LOW', 'IDXX', 'WEC', 'RSG', 'MPWR', 'NSC', 'ATO', 'XOM', 'VTR', 'FE', 'WM', 'WMB', 'DLTR', 'MA', 'MO', 'PM', 'ROST']


stock_ids = [ticker_to_id_dict[ticker] for ticker in selected_tickers]  # Encode

print(stock_ids)

# with open("asset_selection.txt", "w") as f:
#     for i in range(len(selected_tickers) // 40):
#         line = f"Asset Selection for {2019+i} : {stock_ids[40*i:40*(i+1)]}\n"
#         f.write(line)