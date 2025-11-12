# debug_save_whitebit.py
import os
from adapters.whitebit import (
    fetch_whitebit_open_positions,
    fetch_whitebit_funding_fees,
    save_whitebit_closed_positions,
)

# set WHITEBIT_API_KEY / WHITEBIT_API_SECRET antes de ejecutar

if __name__ == "__main__":
    print("— OPEN POS —")
    for r in fetch_whitebit_open_positions():
        print(r)

    print("\n— FUNDING (20) —")
    for r in fetch_whitebit_funding_fees(limit=20):
        print(r)

    print("\n— SAVE CLOSED (30 días) —")
    save_whitebit_closed_positions("portfolio.db", days=50, debug=True)
