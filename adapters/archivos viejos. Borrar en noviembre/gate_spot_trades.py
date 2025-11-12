# gate_spot_trades.py — Reconstrucción de posiciones spot (Gate.io)
# - Estables USDC<->USDT: agrega en 1 posicion/5min con side='swapstable' (sin FIFO)
# - Otros pares: FIFO; inserta 1 posicion por ronda cerrada (inventario 0)
# - Inserta en closed_positions con UPSERT sobre (exchange,symbol,close_time,side)

import os
import time
import sqlite3
from typing import Dict, List, Any, Tuple
from collections import deque
from datetime import datetime


from gate2 import _request, _num
from universal_cache import init_universal_cache_db, get_cached_currency_pairs

STABLE_TOKENS = {"USDT", "USDC", "BUSD", "TUSD", "DAI"}
STABLE_PAIRS_CANON = {("USDC", "USDT"), ("USDT", "USDC")}

# ----------------------------- Helpers ---------------------------------

def _now_s() -> int:
    return int(time.time())

def _split_pair(cp: str) -> Tuple[str, str, str]:
    cp = (cp or "").upper().replace("/", "_")
    if "_" in cp:
        base, quote = cp.split("_", 1)
    else:
        base, quote = (cp[:-4], cp[-4:]) if len(cp) > 4 else (cp, "USDT")
    return base, quote, f"{base}_{quote}"

def _trade_ts(tr: Dict[str, Any]) -> int:
    if tr.get("create_time_ms"):
        return int(float(tr["create_time_ms"]) / 1000.0)
    if tr.get("create_time"):
        return int(float(tr["create_time"]))
    return _now_s()

def _is_stable_swap_pair(cp: str) -> bool:
    b, q, _ = _split_pair(cp)
    return (b, q) in STABLE_PAIRS_CANON

def _fee_usdt(tr: Dict[str, Any]) -> float:
    fee = float(tr.get("fee") or 0)
    if fee == 0:
        return 0.0
    fee_ccy = (tr.get("fee_currency") or "").upper()
    price = float(tr.get("price") or 0)
    base, quote, _ = _split_pair(tr.get("currency_pair", ""))
    if fee_ccy == "USDT":
        return fee
    if fee_ccy == base and price > 0:
        return fee * price
    if fee_ccy in STABLE_TOKENS:
        # 1:1 entre stables (si prefieres mapear USDC->USDT explícitamente, cambialo aquí)
        return fee
    return fee * price if price > 0 else 0.0

# ------------------------- Fetch de datos -------------------------------

def fetch_gate_spot_trades_by_pair(days_back: int = 29) -> Dict[str, List[Dict[str, Any]]]:
    """
    Devuelve dict { 'ALPACA_USDT': [trades...], ... } para los pares en el cache.
    Ventana capada a 29d para evitar el error de >30d.
    """
    print("DEBUG: Descargando trades spot de Gate.io por par...")
    init_universal_cache_db()
    pairs = get_cached_currency_pairs("gate") or ["BTC_USDT", "ETH_USDT"]
    days_back = max(1, min(int(days_back), 29))
    end_time = _now_s()
    start_time = end_time - days_back * 24 * 3600

    out: Dict[str, List[Dict[str, Any]]] = {}
    for cp in pairs:
        params = {"currency_pair": cp, "from": start_time, "to": end_time, "limit": 1000}
        try:
            trades = _request("GET", "/spot/trades", params=params) or []
            out[cp] = trades
            print(f"  OK {cp}: {len(trades)} trades")
            time.sleep(0.1)
        except Exception as e:
            msg = str(e)
            if "INVALID_PARAM_VALUE" in msg:
                # Reintento con 20 días por seguridad
                params["from"] = end_time - 20 * 24 * 3600
                try:
                    trades = _request("GET", "/spot/trades", params=params) or []
                    out[cp] = trades
                    print(f"  OK {cp}: {len(trades)} trades (20d)")
                except Exception as e2:
                    print(f"  ERROR {cp}: {e2}")
                    out[cp] = []
            else:
                print(f"  ERROR {cp}: {msg}")
                out[cp] = []
    return out

def fetch_gate_spot_balances(currency: str = None) -> List[Dict[str, Any]]:
    """
    Balances spot para saber si queda inventario (token aún en cuenta).
    """
    params = {}
    if currency:
        params["currency"] = currency
    try:
        balances = _request("GET", "/spot/accounts", params=params) or []
        return balances
    except Exception as e:
        print("ERROR fetch balances:", e)
        return []

# ----------------- Reconstrucción de posiciones ------------------------

def _group_stable_swaps(trades: List[Dict[str, Any]], window_sec: int = 300) -> List[Dict[str, Any]]:
    """
    Agrupa USDC<->USDT en 1 posicion por ventana de 5 min (sin FIFO).
    realized_pnl = (USDT_recibidos - USDC_vendidos) - fees
    """
    if not trades:
        return []
    trades = sorted(trades, key=_trade_ts)

    groups: List[List[Dict[str, Any]]] = []
    bucket = [trades[0]]
    for t in trades[1:]:
        if _trade_ts(t) - _trade_ts(bucket[-1]) <= window_sec:
            bucket.append(t)
        else:
            groups.append(bucket)
            bucket = [t]
    groups.append(bucket)

    positions = []
    for g in groups:
        base, quote, cp = _split_pair(g[0]["currency_pair"])
        symbol = f"{base}{quote}"
        qty_base = 0.0
        proceeds_usdt = 0.0
        fees_usdt = 0.0
        ts_last = _trade_ts(g[-1])

        for t in g:
            amount = float(t.get("amount") or 0)
            price = float(t.get("price") or 0)
            side = (t.get("side") or "").lower()
            fees_usdt += _fee_usdt(t)
            if side == "sell" and base in STABLE_TOKENS and quote == "USDT":
                qty_base += amount
                proceeds_usdt += amount * price
            elif side == "buy" and base == "USDT" and quote in STABLE_TOKENS:
                qty_base += amount
                proceeds_usdt -= amount * price
            else:
                # Mejor esfuerzo si llegase mezcla rara
                qty_base += amount if side == "sell" else amount
                proceeds_usdt += (amount * price if side == "sell" else -amount * price)

        pnl_price = proceeds_usdt - qty_base
        realized_pnl = pnl_price - fees_usdt

        positions.append({
            "exchange": "gate",
            "symbol": symbol,
            "side": "swapstable",
            "size": qty_base,
            "entry_price": 1.0,
            "close_price": (proceeds_usdt / qty_base) if qty_base > 0 else 1.0,
            "open_time": ts_last,
            "close_time": ts_last,
            "realized_pnl": realized_pnl,
            "fee_total": -abs(fees_usdt),
            "funding_total": 0.0,
            "notional": proceeds_usdt,
            "initial_margin": 0.0,
            "ignore_trade": 0,
            "raw": None
        })
    return positions

from typing import List, Dict, Any
from datetime import datetime

EPS = 1e-9

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def _parse_val_asset(v):
    """
    Acepta '123.45 USDT', '9.87 ALPACA' o numérico. Devuelve (valor_float, asset_str|None)
    """
    if v is None: 
        return 0.0, None
    s = str(v).strip()
    parts = s.split()
    if len(parts) == 2:
        return _safe_float(parts[0]), parts[1].upper()
    # numérico simple
    return _safe_float(s), None

def _fee_usdt_from_trade(t: Dict[str, Any]) -> float:
    """
    Convierte la fee a USDT.
    Soporta estructuras típicas de Gate:
      - {'fee': '0.0123 ALPACA', 'price': 0.0150}
      - {'fee': 0.0123, 'fee_currency': 'ALPACA', 'price': 0.0150}
      - {'fee': '0.10 USDT'} → ya en USDT
    """
    price = _safe_float(t.get("price") or t.get("Deal price"))
    fee_field = t.get("fee", t.get("Fee"))
    fee_cur = t.get("fee_currency")

    if fee_cur:  # forma API
        fee_val = _safe_float(fee_field)
        if fee_cur.upper() == "USDT":
            return fee_val
        else:
            # fee en base → convertir a USDT con el precio del trade
            return fee_val * price

    # forma CSV 'X ASSET'
    fee_val, fee_asset = _parse_val_asset(fee_field)
    if (fee_asset or "").upper() == "USDT" or fee_asset is None:
        return fee_val
    return fee_val * price

def _amount_from_trade(t: Dict[str, Any]) -> float:
    amt = t.get("amount", t.get("Deal amount"))
    if isinstance(amt, (int, float)):
        return float(amt)
    val, _ = _parse_val_asset(amt)
    return val

def _total_usdt_from_trade(t: Dict[str, Any]) -> float:
    tot = t.get("total", t.get("Total"))
    if isinstance(tot, (int, float)):
        return float(tot)
    val, _ = _parse_val_asset(tot)
    return val

def _side_from_trade(t: Dict[str, Any]) -> str:
    return str(t.get("side", t.get("Trade type", ""))).strip().lower()

def _ts_from_trade(t: Dict[str, Any]):
    # Time en CSV: 'YYYY-MM-DD HH:MM:SS'
    # API: ya timestamp/iso. Intentamos convertir de forma robusta.
    ts = t.get("ts") or t.get("Time") or t.get("time")
    if isinstance(ts, (int, float)):
        return datetime.utcfromtimestamp(float(ts))
    try:
        return datetime.fromisoformat(str(ts).replace("Z",""))
    except Exception:
        try:
            return datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

def fifo_spot_single_position(trades: List[Dict[str, Any]], exchange_name="gate") -> List[Dict[str, Any]]:
    """
    Si el flujo es BUY...BUY -> SELL...SELL (sin volver a BUY después del primer SELL),
    agrega TODO en una ÚNICA posición. Si no, puedes llamar al FIFO clásico.
    """
    if not trades:
        return []

    # Orden por tiempo ASC
    trades = sorted(trades, key=_ts_from_trade)

    # Detecta patrón buy-then-sell
    sides = [_side_from_trade(t) for t in trades]
    try:
        first_sell_idx = next(i for i, s in enumerate(sides) if s == "sell")
    except StopIteration:
        # Nunca vendiste → no hay cerrada
        return []

    buy_then_sell = (
        any(s == "buy" for s in sides[:first_sell_idx]) and
        all(s != "buy" for s in sides[first_sell_idx:])
    )
    if not buy_then_sell:
        # Aquí, si lo necesitas, re-usa tu FIFO clásico.
        # Para tu caso (ALPACA) sí es buy→sell puro, así que no entrarás aquí.
        return []

    # Agregación única
    total_buy_qty = total_buy_cost = 0.0
    total_sell_qty = total_sell_proceeds = 0.0
    total_fees_usdt = 0.0
    open_time = _ts_from_trade(trades[0])
    close_time = _ts_from_trade(trades[-1])

    for t in trades:
        side = _side_from_trade(t)
        px = _safe_float(t.get("price", t.get("Deal price")))
        qty = _amount_from_trade(t)
        total_fees_usdt += _fee_usdt_from_trade(t)
        if side == "buy":
            total_buy_qty += qty
            total_buy_cost += qty * px
        elif side == "sell":
            total_sell_qty += qty
            total_sell_proceeds += qty * px

    matched_qty = min(total_buy_qty, total_sell_qty)
    if matched_qty <= EPS:
        return []

    avg_buy = total_buy_cost / total_buy_qty if total_buy_qty > EPS else 0.0
    avg_sell = total_sell_proceeds / total_sell_qty if total_sell_qty > EPS else 0.0
    pnl_price = (avg_sell - avg_buy) * matched_qty
    realized_pnl = pnl_price - total_fees_usdt

    # nombre símbolo simple
    mkt = trades[0].get("currency_pair", trades[0].get("Market", "ALPACA/USDT"))
    base = mkt.split("/")[0] if "/" in mkt else mkt

    pos = {
        "exchange": exchange_name,
        "symbol": f"{base}-USDT",
        "side": "spotbuy",
        "size": matched_qty,
        "entry_price": avg_buy,
        "close_price": avg_sell,
        "open_time": open_time,
        "close_time": close_time,
        "realized_pnl": realized_pnl,
        "fee_total": -abs(total_fees_usdt),
        "funding_total": 0.0,
        "notional": total_sell_proceeds,
        "initial_margin": 0.0,
        "ignore_trade": 0,
        "raw": None
    }
    return [pos]


# ---------------------- INSERT/UPSERT en la DB -------------------------

def _insert_closed_positions(records: List[Dict[str, Any]], db_path: str = "portfolio.db") -> Tuple[int, int]:
    if not records:
        return 0, 0
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    saved = 0
    updated = 0
    
    for r in records:
        try:
            cur.execute("""
            INSERT INTO closed_positions
            (exchange, symbol, side, size, entry_price, close_price, open_time, close_time,
             realized_pnl, fee_total, funding_total, notional, initial_margin, ignore_trade)
            VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(exchange, symbol, close_time, side)
            DO UPDATE SET
                realized_pnl = excluded.realized_pnl,
                fee_total = excluded.fee_total,
                notional = excluded.notional
            """, (
                r["exchange"], r["symbol"], r["side"], r["size"], 
                r["entry_price"], r["close_price"], r["open_time"], r["close_time"],
                r["realized_pnl"], r["fee_total"], r["funding_total"], 
                r["notional"], r["initial_margin"], r["ignore_trade"]
            ))
            
            if cur.rowcount == 1:
                saved += 1
            else:
                updated += 1
                
        except sqlite3.Error as e:
            print(f"❌ Error SQLite insertando posición {r.get('symbol')}: {e}")
            continue
    
    conn.commit()
    conn.close()
    print(f"DEBUG: posiciones guardadas={saved} | actualizadas={updated}")
    return saved, updated

# --------------------- Orquestador público -----------------------------

def process_and_save_gate_spot_positions(all_trades_by_pair: Dict[str, List[Dict[str, Any]]],
                                         db_path: str = "portfolio.db") -> Tuple[int, int]:
    positions: List[Dict[str, Any]] = []
    
    print(f"DEBUG: Procesando {len(all_trades_by_pair)} pares")
    for cp, trades in all_trades_by_pair.items():
        print(f"DEBUG: Par {cp}: {len(trades)} trades")
        if not trades:
            continue
        if _is_stable_swap_pair(cp):
            stable_positions = _group_stable_swaps(trades, window_sec=300)
            print(f"DEBUG: {cp} - {len(stable_positions)} posiciones estables")
            positions.extend(stable_positions)
        else:
            fifo_positions = _fifo_positions_from_trades(trades)
            print(f"DEBUG: {cp} - {len(fifo_positions)} posiciones FIFO")
            positions.extend(fifo_positions)
    
    print(f"DEBUG: Total {len(positions)} posiciones a guardar")
    return _insert_closed_positions(positions, db_path=db_path)

def save_gate_spot_positions(db_path: str = "portfolio.db", days_back: int = 29) -> Tuple[int, int]:
    """
    Entry point para la app: reconstruye posiciones y las guarda en closed_positions.
    """
    if not os.path.exists(db_path):
        print("ERROR: database no encontrada:", db_path)
        return (0, 0)

    trades_by_pair = fetch_gate_spot_trades_by_pair(days_back=days_back)
    saved, updated = process_and_save_gate_spot_positions(trades_by_pair, db_path=db_path)
    print(f"Gate spot posiciones guardadas={saved} | actualizadas={updated}")
    return saved, updated

# ---------------------- Debug manual -----------------------------------

if __name__ == "__main__":
    # Debug simple ejecutable en Spyder: Run
    save_gate_spot_positions(db_path="portfolio.db", days_back=29)
