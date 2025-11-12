# gate_spot_trades.py — Reconstrucción de posiciones spot (Gate.io)
# - Usa SOLO los pares del universal cache.
# - Estables USDC<->USDT: agrega en 1 posicion/5min con side='swapstable' (sin FIFO).
# - Otros pares: fast-path BUY... -> SELL... (sin vuelta a BUY) = UNA posición; si no, FIFO por rondas (inventario 0).
# - UPSERT manual (UPDATE->INSERT) en closed_positions por (exchange,symbol,side,close_time).

import os
import time
import sqlite3
from typing import Dict, List, Any, Tuple
from collections import deque
from datetime import datetime

from adapters.gate2 import _request, _num
from universal_cache import init_universal_cache_db, get_cached_currency_pairs

STABLE_TOKENS = {"USDT", "USDC", "BUSD", "TUSD", "DAI"}
STABLE_PAIRS_CANON = {("USDC", "USDT"), ("USDT", "USDC")}
EPS = 1e-9

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
    # CSV fallbacks
    ts = tr.get("time") or tr.get("Time")
    if ts:
        try:
            return int(datetime.fromisoformat(str(ts).replace("Z","")).timestamp())
        except Exception:
            try:
                return int(datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S").timestamp())
            except Exception:
                pass
    return _now_s()

def _is_stable_swap_pair(cp: str) -> bool:
    b, q, _ = _split_pair(cp)
    return (b, q) in STABLE_PAIRS_CANON

def _fee_usdt(tr: Dict[str, Any]) -> float:
    """
    Convierte la fee a USDT:
      - API: fee (float) + fee_currency. Si fee en base, multiplica por price.
      - CSV: 'Fee' como '0.0123 ALPACA' o '0.10 USDT'.
    """
    price = float(tr.get("price") or tr.get("Deal price") or 0.0)
    fee = tr.get("fee")
    fee_ccy = (tr.get("fee_currency") or "").upper()

    def _parse_fee_str(s: str) -> Tuple[float, str]:
        parts = str(s).strip().split()
        if len(parts) == 2:
            try:
                return float(parts[0]), parts[1].upper()
            except Exception:
                return 0.0, "USDT"
        try:
            return float(s), "USDT"
        except Exception:
            return 0.0, "USDT"

    # API style
    if fee is not None:
        try:
            fee_val = float(fee)
            if fee_ccy == "USDT":
                return fee_val
            base, quote, _ = _split_pair(tr.get("currency_pair", ""))
            if fee_ccy == base and price > 0:
                return fee_val * price
            if fee_ccy in STABLE_TOKENS:
                return fee_val
            return fee_val * price if price > 0 else 0.0
        except Exception:
            pass

    # CSV style
    fee_csv = tr.get("Fee")
    if fee_csv is not None:
        fval, fccy = _parse_fee_str(fee_csv)
        if fccy == "USDT" or fccy in STABLE_TOKENS:
            return fval
        return fval * price if price > 0 else 0.0

    return 0.0

def _amount_from_trade(t: Dict[str, Any]) -> float:
    amt = t.get("amount", t.get("Deal amount"))
    if isinstance(amt, (int, float)):
        return float(amt)
    s = str(amt or "").strip().split()
    return float(s[0]) if s else 0.0

def _side_from_trade(t: Dict[str, Any]) -> str:
    return str(t.get("side", t.get("Trade type", ""))).strip().lower()

# ------------------------- Fetch de datos -------------------------------

def fetch_gate_spot_trades_by_pair(days_back: int = 29) -> Dict[str, List[Dict[str, Any]]]:
    """
    Devuelve dict { 'ALPACA_USDT': [trades...], ... } para los pares en el cache.
    NO hay fallback a BTC/ETH. Si el cache está vacío, no hace llamadas.
    Ventana capada a 29d para evitar el error de >30d.
    """
    print("DEBUG: Descargando trades spot de Gate.io por par...")
    init_universal_cache_db()
    pairs = get_cached_currency_pairs("gate")
    if not pairs:
        print("⚠️ No hay símbolos en caché para consultar. (universal_cache vacío)")
        return {}

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
        base, quote, _ = _split_pair(g[0]["currency_pair"])
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

def _single_or_fifo_positions(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Si el flujo es BUY...BUY -> SELL...SELL (sin volver a BUY tras el primer SELL), colapsa TODO en UNA posición.
    Si no, FIFO por rondas (cada inventario->0 es una posición).
    """
    if not trades:
        return []

    trades = sorted(trades, key=_trade_ts)
    base, quote, _ = _split_pair(trades[0]["currency_pair"])
    symbol = f"{base}{quote}"

    sides = [(_side_from_trade(t) or "").lower() for t in trades]
    first_sell_idx = next((i for i, s in enumerate(sides) if s == "sell"), None)
    buy_then_sell = (
        first_sell_idx is not None
        and any(s == "buy" for s in sides[:first_sell_idx])
        and all(s != "buy" for s in sides[first_sell_idx:])
    )

    if buy_then_sell:
        total_buy_qty = total_buy_cost = 0.0
        total_sell_qty = total_sell_proceeds = 0.0
        total_fees = 0.0
        open_time = _trade_ts(trades[0])
        close_time = _trade_ts(trades[-1])

        for t in trades:
            side = (_side_from_trade(t) or "").lower()
            qty = _amount_from_trade(t)
            px = float(t.get("price") or 0.0)
            total_fees += _fee_usdt(t)
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
        realized_pnl = pnl_price - total_fees

        print(f"DEBUG FIFO (fast-path): {symbol} -> 1 posición")
        return [{
            "exchange": "gate",
            "symbol": symbol,
            "side": "spotbuy",
            "size": matched_qty,
            "entry_price": avg_buy,
            "close_price": avg_sell,
            "open_time": open_time,
            "close_time": close_time,
            "realized_pnl": realized_pnl,
            "fee_total": -abs(total_fees),
            "funding_total": 0.0,
            "notional": total_sell_proceeds,
            "initial_margin": 0.0,
            "ignore_trade": 0,
            "raw": None
        }]

    # ---------- FIFO clásico por rondas ----------
    lots = deque()  # [qty, price]
    positions: List[Dict[str, Any]] = []
    round_data = {
        "qty_buy": 0.0, "cost": 0.0, "fees": 0.0,
        "qty_sell": 0.0, "proceeds": 0.0,
        "open_time": None, "close_time": None
    }

    for t in trades:
        side = (_side_from_trade(t) or "").lower()
        qty = _amount_from_trade(t)
        px = float(t.get("price") or 0.0)
        ts = _trade_ts(t)
        feeu = _fee_usdt(t)

        if side == "buy":
            lots.append([qty, px])
            round_data["qty_buy"] += qty
            round_data["cost"] += qty * px
            round_data["fees"] += feeu
            if round_data["open_time"] is None:
                round_data["open_time"] = ts

        elif side == "sell":
            remaining = qty
            round_data["close_time"] = ts
            while remaining > EPS and lots:
                q, p = lots[0]
                take = min(q, remaining)
                q -= take
                remaining -= take
                round_data["qty_sell"] += take
                round_data["proceeds"] += take * px
                if q <= EPS:
                    lots.popleft()
                else:
                    lots[0][0] = q
            round_data["fees"] += feeu

        # ¿Inventario 0? -> cerramos ronda
        inventory = sum(q for q, _ in lots)
        if inventory <= EPS and (round_data["qty_buy"] > 0 or round_data["qty_sell"] > 0):
            matched = min(round_data["qty_buy"], round_data["qty_sell"])
            if matched > EPS:
                avg_buy = round_data["cost"] / round_data["qty_buy"] if round_data["qty_buy"] > EPS else 0.0
                avg_sell = round_data["proceeds"] / round_data["qty_sell"] if round_data["qty_sell"] > EPS else 0.0
                pnl_price = (avg_sell - avg_buy) * matched
                realized_pnl = pnl_price - round_data["fees"]

                positions.append({
                    "exchange": "gate",
                    "symbol": symbol,
                    "side": "spotbuy",
                    "size": matched,
                    "entry_price": avg_buy,
                    "close_price": avg_sell,
                    "open_time": round_data["open_time"] or _trade_ts(trades[0]),
                    "close_time": round_data["close_time"] or _trade_ts(trades[-1]),
                    "realized_pnl": realized_pnl,
                    "fee_total": -abs(round_data["fees"]),
                    "funding_total": 0.0,
                    "notional": round_data["proceeds"],
                    "initial_margin": 0.0,
                    "ignore_trade": 0,
                    "raw": None
                })
            # reset ronda
            lots.clear()
            round_data = {
                "qty_buy": 0.0, "cost": 0.0, "fees": 0.0,
                "qty_sell": 0.0, "proceeds": 0.0,
                "open_time": None, "close_time": None
            }

    print(f"DEBUG FIFO: {symbol} - {len(trades)} trades → {len(positions)} posiciones")
    return positions

# ---------------------- INSERT/UPSERT en la DB -------------------------

def _insert_closed_positions(records: List[Dict[str, Any]], db_path: str = "portfolio.db") -> Tuple[int, int]:
    if not records:
        return 0, 0

    # fuerza enteros en times
    for r in records:
        if isinstance(r.get("open_time"), float):
            r["open_time"] = int(r["open_time"])
        if isinstance(r.get("close_time"), float):
            r["close_time"] = int(r["close_time"])

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    saved = 0
    updated = 0

    for r in records:
        # UPDATE por clave natural
        cur.execute("""
            UPDATE closed_positions
               SET size           = :size,
                   entry_price    = :entry_price,
                   close_price    = :close_price,
                   open_time      = :open_time,
                   realized_pnl   = :realized_pnl,
                   fee_total      = :fee_total,
                   funding_total  = :funding_total,
                   notional       = :notional,
                   initial_margin = :initial_margin
             WHERE exchange      = :exchange
               AND symbol        = :symbol
               AND side          = :side
               AND close_time    = :close_time
        """, r)
        if cur.rowcount and cur.rowcount > 0:
            updated += cur.rowcount
            continue

        # INSERT si no existía
        cur.execute("""
            INSERT INTO closed_positions
            (exchange, symbol, side, size, entry_price, close_price, open_time, close_time,
             realized_pnl, fee_total, funding_total, notional, initial_margin)
            VALUES
            (:exchange, :symbol, :side, :size, :entry_price, :close_price, :open_time, :close_time,
             :realized_pnl, :fee_total, :funding_total, :notional, :initial_margin)
        """, r)
        saved += 1

    conn.commit()
    conn.close()
    print(f"DEBUG: posiciones guardadas={saved} | actualizadas={updated}")
    return saved, updated

# --------------------- Orquestador público -----------------------------

def process_and_save_gate_spot_positions(all_trades_by_pair: Dict[str, List[Dict[str, Any]]],
                                         db_path: str = "portfolio.db") -> Tuple[int, int]:
    positions: List[Dict[str, Any]] = []
    for cp, trades in all_trades_by_pair.items():
        if not trades:
            continue
        if _is_stable_swap_pair(cp):
            positions.extend(_group_stable_swaps(trades, window_sec=300))
        else:
            positions.extend(_single_or_fifo_positions(trades))
    return _insert_closed_positions(positions, db_path=db_path)

def save_gate_spot_positions(db_path: str = "portfolio.db", days_back: int = 29) -> Tuple[int, int]:
    """
    Entry point para la app: reconstruye posiciones y las guarda en closed_positions.
    Usa SOLO los símbolos del universal cache.
    """
    if not os.path.exists(db_path):
        print("ERROR: database no encontrada:", db_path)
        return (0, 0)

    trades_by_pair = fetch_gate_spot_trades_by_pair(days_back=days_back)
    if not trades_by_pair:
        print("⚠️ Nada que procesar (sin símbolos o sin trades).")
        return (0, 0)
    saved, updated = process_and_save_gate_spot_positions(trades_by_pair, db_path=db_path)
    print(f"Gate spot posiciones guardadas={saved} | actualizadas={updated}")
    return saved, updated

# ---------------------- Debug manual -----------------------------------

if __name__ == "__main__":
    # Debug simple ejecutable en Spyder: Run
    save_gate_spot_positions(db_path="portfolio.db", days_back=29)

