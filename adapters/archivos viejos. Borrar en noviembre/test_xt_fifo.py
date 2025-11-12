# tests/test_xt_fifo.py
import os, sqlite3, time
from adapters.xt2 import _fifo_closed_from_trades_and_funding
import db_manager

def test_fifo_and_db(tmp_path):
    # trades: long 10 @ 100 -> sell 10 @ 110
    t0 = int(time.time()*1000) - 3600_000
    t1 = t0 + 10_000
    trades = [
        {"symbol":"ABC","side":"BUY","price":100.0,"qty":10.0,"fee":-0.50,"ts":t0},
        {"symbol":"ABC","side":"SELL","price":110.0,"qty":10.0,"fee":-0.50,"ts":t1},
    ]
    funding = [{"exchange":"xt","symbol":"ABC","income":0.5,"asset":"USDT","timestamp":t1-1000,"funding_rate":0.0,"type":"FUNDING_FEE"}]

    closed = _fifo_closed_from_trades_and_funding(trades, funding)
    assert len(closed) == 1
    row = closed[0]
    assert row["pnl"] == 100.0  # (110-100)*10
    assert abs(row["fee_total"] + 1.0) < 1e-9
    assert row["funding_total"] == 0.5
    assert row["realized_pnl"] == 99.5

    # persistencia en DB temporal
    test_db = tmp_path / "portfolio_xt_test.db"
    db_manager.DB_PATH = str(test_db)
    conn = sqlite3.connect(db_manager.DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS closed_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        exchange TEXT, symbol TEXT, side TEXT, size REAL,
        entry_price REAL, close_price REAL, open_time INTEGER, close_time INTEGER,
        pnl REAL, realized_pnl REAL, funding_total REAL, fee_total REAL,
        pnl_percent REAL, apr REAL, initial_margin REAL, notional REAL, leverage REAL, liquidation_price REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit(); conn.close()

    from db_manager import save_closed_position
    save_closed_position(row)

    conn = sqlite3.connect(db_manager.DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT exchange, symbol, pnl, realized_pnl, funding_total, fee_total FROM closed_positions")
    rec = cur.fetchone()
    conn.close()
    assert rec[0] == "xt" and rec[1] == "ABC"
    assert abs(rec[2] - 100.0) < 1e-9
    assert abs(rec[3] - 99.5) < 1e-9
    assert abs(rec[4] - 0.5) < 1e-9
    assert abs(rec[5] - (-1.0)) < 1e-9

