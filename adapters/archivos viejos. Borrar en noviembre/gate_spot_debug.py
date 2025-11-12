import csv, pandas as pd
from pathlib import Path
from gate_spot_trades import fifo_spot_single_position

# --- carga CSV de Gate export ---
p = Path(r"Spot_TradeHistory_2025-11-06 14_37_05.csv")
df = pd.read_csv(p)

# adapta columnas a la estructura esperada por la función
trades = []
for _, r in df.iterrows():
    trades.append({
        "Time": r["Time"],
        "side": r["Trade type"],           # 'Buy'/'Sell'
        "price": float(r["Deal price"]),
        "amount": r["Deal amount"],        # '123 ALPACA'
        "total": r["Total"],               # '12.3 USDT'
        "fee": r["Fee"],                   # '0.12 ALPACA' o '0.01 USDT'
        "currency_pair": r["Market"],      # 'ALPACA/USDT'
    })

pos = fifo_spot_single_position(trades, exchange_name="gate")
print("POS:", pos[0] if pos else "N/A")
