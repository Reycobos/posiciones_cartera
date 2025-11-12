# adapters/xt.py
# XT.COM Perp adapter – open positions, funding, balances y closed positions por FIFO
from __future__ import annotations

import os, json, traceback, argparse
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

# SDK oficial (opcional)
try:
    from pyxt.perp import Perp
except Exception:
    Perp = None  # tolera que no esté instalado

# Helpers del proyecto (ajusta a tus módulos reales si fuera necesario)
try:
    from symbols import normalize_symbol
except Exception:
    # Fallback: normalizador básico por si no se puede importar el del proyecto
    import re
    def normalize_symbol(sym: str) -> str:
        if not sym:
            return ""
        s = sym.upper()
        s = re.sub(r'^PERP_', '', s)
        s = re.sub(r'(_|-)?(USDT|USDC|PERP)$', '', s)
        s = re.sub(r'[_-]+$', '', s)
        s = re.split(r'[_-]', s)[0]
        return s

try:
    from time import to_ms, to_s, utc_now_ms
except Exception:
    # Fallbacks mínimos si no existen
    import time as _pytime
    def utc_now_ms() -> int: return int(_pytime.time() * 1000)
    def to_ms(x: float | int) -> int: return int(float(x) * (1000 if float(x) < 1e12 else 1))
    def to_s(ms: int) -> int: return int(ms // 1000)

try:
    from money import to_float
except Exception:
    def to_float(x) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

# (Opcional) helpers de print del backend; si no existen, se usan no-ops
try:
    from portfoliov7 import (
        p_closed_sync_start, p_closed_sync_saved, p_closed_sync_done, p_closed_sync_none,
        p_open_summary, p_open_block,
        p_funding_fetching, p_funding_count,
        p_balance_equity,
    )
except Exception:
    def _noop(*a, **k): ...
    p_closed_sync_start = p_closed_sync_saved = p_closed_sync_done = p_closed_sync_none = _noop
    p_open_summary = p_open_block = _noop
    p_funding_fetching = p_funding_count = _noop
    p_balance_equity = _noop

# Guardado en SQLite (tu función ya recalcula pnl_percent, apr, sanity, etc.)
try:
    from db_manager import save_closed_position
except Exception:
    save_closed_position = None

EXCHANGE = "xt"
XT_PERP_HOST = os.getenv("XT_PERP_HOST", "https://fapi.xt.com")
XT_API_KEY = "96b1c999-b9df-42ed-9d4f-a4491aac7a1d"
XT_API_SECRET = "5f60a0a147d82db2117a45da129a6f4480488234"
# ----- ventana por defecto para TRADES/FIFO -----
DEFAULT_DAYS_TRADES = int(os.getenv("XT_TRADES_DAYS", "30"))

# ------------ util ------------

def _client() -> Optional[Perp]:
    if Perp is None:
        print("❌ pyxt no instalado. Ejecuta: pip install pyxt")
        return None
    try:
        return Perp(host=XT_PERP_HOST, access_key=XT_API_KEY, secret_key=XT_API_SECRET)
    except Exception as e:
        print(f"❌ XT client error: {e}")
        return None

def _call_many(obj, names: List[str], **kwargs):
    """Intenta varias firmas de método del SDK para tolerar cambios (sin nombre)."""
    for n in names:
        fn = getattr(obj, n, None)
        if callable(fn):
            try:
                r = fn(**kwargs) if kwargs else fn()
                if r is not None:
                    return r
            except Exception:
                continue
    return None

def _call_one_named(obj, names: List[str], **kwargs) -> Tuple[Optional[str], Any]:
    """Como _call_many pero devolviendo el nombre del método usado para debug RAW."""
    for n in names:
        fn = getattr(obj, n, None)
        if callable(fn):
            try:
                r = fn(**kwargs) if kwargs else fn()
                if r is not None:
                    return n, r
            except Exception as e:
                continue
    return None, None

def _side_from(s: str) -> str:
    s = (s or "").lower()
    if s in ("sell", "short", "close_short"):
        return "short"
    return "long"

def _ts_ms(x) -> int:
    try:
        return to_ms(int(x))
    except Exception:
        try:
            return to_ms(float(x))
        except Exception:
            return utc_now_ms()

def _ms_to_s(ms: int) -> int:
    return to_s(ms)

# ------------ API pública ------------

def fetch_xt_open_positions() -> List[Dict[str, Any]]:
    """
    Shape EXACTO para /api/positions:
    {
      "exchange": "xt","symbol": "<NORMALIZADO>","side": "long|short","size": float,
      "entry_price": float,"mark_price": float,"liquidation_price": float|0.0,
      "notional": float,"unrealized_pnl": float,"fee": float(negativo),
      "funding_fee": float,"realized_pnl": float(=fee+funding_fee)
    }
    """
    cli = _client()
    if not cli:
        return []

    raw = _call_many(cli, [
        "get_positions", "position_list", "get_account_positions", "positions"
    ]) or []

    out: List[Dict[str, Any]] = []
    for p in (raw or []):
        sym0 = p.get("symbol") or p.get("instId") or p.get("market") or ""
        base = normalize_symbol(sym0)
        qty  = to_float(p.get("quantity") or p.get("size") or p.get("pos") or 0)
        side = _side_from(p.get("side") or ("long" if qty >= 0 else "short"))
        size = abs(qty)
        entry = to_float(p.get("avgPrice") or p.get("entryPrice") or p.get("avg_entry_price") or 0)
        mark  = to_float(p.get("markPrice") or p.get("mark_price") or p.get("indexPrice") or entry)
        liq   = to_float(p.get("liquidationPrice") or p.get("liqPrice") or 0.0)
        unreal= to_float(p.get("unrealizedPnl") or p.get("uPnl") or p.get("unrealizedProfit") or (mark-entry)*size*(1 if side=="long" else -1))
        fees  = to_float(p.get("cumFee") or p.get("fee") or 0.0)
        if fees > 0:  # fee acumulado debe ser NEGATIVO si es costo
            fees = -abs(fees)
        fund  = to_float(p.get("realizedFunding") or p.get("funding") or 0.0)

        obj = {
            "exchange": EXCHANGE,
            "symbol": base,
            "side": side,
            "size": float(size),
            "entry_price": float(entry),
            "mark_price": float(mark),
            "liquidation_price": float(liq or 0.0),
            "notional": float(abs(size)*entry),
            "unrealized_pnl": float(unreal),
            "fee": float(fees),
            "funding_fee": float(fund),
            "realized_pnl": float(fees + fund),
        }
        out.append(obj)

    p_open_summary(EXCHANGE, len(out))
    for r in out:
        p_open_block(EXCHANGE, r["symbol"], r["size"], r["entry_price"], r["mark_price"],
                     r["unrealized_pnl"], r["funding_fee"], None, r["notional"], False)
    return out


def fetch_xt_funding_fees(limit: int = 50, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Shape EXACTO:
    {"exchange":"xt","symbol":"<NORMALIZADO>","income":float(+/-),"asset":"USDT|USDC|USD","timestamp":int(ms),"funding_rate":float|0.0,"type":"FUNDING_FEE"}
    """
    cli = _client()
    if not cli:
        return []
    p_funding_fetching(EXCHANGE)

    if end_ms is None:
        end_ms = utc_now_ms()
    if start_ms is None:
        start_ms = end_ms - 7*24*60*60*1000  # 7 días

    resp = _call_many(cli, ["get_funding_history", "funding_list", "get_funding_fee"],
                      startTime=start_ms, endTime=end_ms, limit=limit) or []

    out: List[Dict[str, Any]] = []
    for it in (resp or []):
        sym0 = it.get("symbol") or it.get("instId") or it.get("market") or ""
        base = normalize_symbol(sym0)
        ts   = _ts_ms(it.get("timestamp") or it.get("time") or it.get("created"))
        income = to_float(it.get("amount") or it.get("income") or it.get("delta") or 0.0)
        rate = to_float(it.get("fundingRate") or it.get("rate") or 0.0)
        asset = (it.get("asset") or it.get("ccy") or "USDT").upper()
        out.append({
            "exchange": EXCHANGE,
            "symbol": base,
            "income": float(income),  # + cobro / - pago
            "asset": "USDT" if asset not in ("USDT", "USDC", "USD") else asset,
            "timestamp": int(ts),
            "funding_rate": float(rate),
            "type": "FUNDING_FEE",
        })

    p_funding_count(EXCHANGE, len(out))
    return out


# --- helpers nuevos para XT balances (pegar en adapters/xt.py) ---

def _unwrap(obj):
    """Desempaqueta payloads típicos del SDK: (meta, data), {'data':...}, {'result':...}."""
    if obj is None:
        return None
    # Tupla tipo (meta, data) o (code, data)
    if isinstance(obj, tuple):
        # prioriza el último elemento que sea dict/list
        for x in reversed(obj):
            if isinstance(x, (dict, list)):
                return x
        return obj
    # {'data': ...} o {'result': ...}
    if isinstance(obj, dict):
        if isinstance(obj.get("data"), (dict, list)):
            return obj["data"]
        if isinstance(obj.get("result"), (dict, list)):
            return obj["result"]
        return obj
    return obj  # list u otro

def _sum_numbers(items, *keys):
    total = 0.0
    for it in items:
        if not isinstance(it, dict):
            continue
        for k in keys:
            v = it.get(k)
            if v is not None:
                try:
                    total += float(v)
                    break
                except Exception:
                    pass
    return total

def debug_dump_xt_balance_raw():
    cli = _client()
    if not cli:
        print("❌ pyxt no instalado.")
        return
    raw = _call_many(cli, [
        "get_account_capital", "get_balance_list", "balance_list", "get_account", "capital"
    ])
    try:
        print(json.dumps(raw, indent=2))
    except Exception:
        print(repr(raw))


def fetch_xt_all_balances(db_path: str = "portfolio.db") -> Dict[str, Any] | None:
    """
    Shape EXACTO /api/balances:
      {"exchange":"xt","equity":float,"balance":float,"unrealized_pnl":float,
       "initial_margin":0.0,"spot":0.0,"margin":0.0,"futures":float}
    """
    cli = _client()
    if not cli:
        return None

    # Probar varias firmas/nombres del SDK
    raw = _call_many(cli, [
        "get_account_capital",    # resumen de cuenta
        "get_balance_list",       # lista por moneda (nombres alternos)
        "balance_list",
        "get_account",
        "capital"
    ]) or {}

    data = _unwrap(raw)

    equity = balance = upnl = 0.0

    if isinstance(data, dict):
        # Resumen plano (o dentro de data/result ya desenvuelto)
        equity  = to_float(data.get("totalEquity") or data.get("equity") or data.get("nav") or 0.0)
        balance = to_float(data.get("totalWalletBalance") or data.get("walletBalance") or data.get("balance") or 0.0)
        upnl    = to_float(data.get("unrealizedPNL") or data.get("unrealizedPnl") or data.get("uPnl") or (equity - balance))
    elif isinstance(data, list):
        # Lista por moneda: sumar campos típicos
        # muchos perp devuelven walletBalance/balance + equity + unrealizedPnl por asset
        equity  = _sum_numbers(data, "equity", "marginBalance", "nav")
        balance = _sum_numbers(data, "walletBalance", "balance")
        upnl    = _sum_numbers(data, "unrealizedPNL", "unrealizedPnl", "uPnl")
        if upnl == 0.0 and equity and balance:
            upnl = equity - balance
    else:
        # Fallback agresivo por si viniera (meta, list/dict) que _unwrap no cazó
        if isinstance(raw, (tuple, list)):
            flat = []
            for x in raw:
                x = _unwrap(x)
                if isinstance(x, list):
                    flat.extend(x)
                elif isinstance(x, dict):
                    flat.append(x)
            if flat:
                equity  = _sum_numbers(flat, "equity", "marginBalance", "nav")
                balance = _sum_numbers(flat, "walletBalance", "balance")
                upnl    = _sum_numbers(flat, "unrealizedPNL", "unrealizedPnl", "uPnl")
                if upnl == 0.0 and equity and balance:
                    upnl = equity - balance

    obj = {
        "exchange": EXCHANGE,
        "equity": float(equity),
        "balance": float(balance),
        "unrealized_pnl": float(upnl),
        "initial_margin": 0.0,
        "spot": 0.0,
        "margin": 0.0,
        "futures": float(equity),
    }
    p_balance_equity(EXCHANGE, obj["equity"])
    return obj
    return obj


# --------- FIFO sobre fills + funding ----------

def _fetch_xt_user_trades(cli: Perp, start_ms: int, end_ms: int, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Devuelve fills del usuario (normalizados).
    Campos que usamos:
      symbol, side(BUY/SELL), price, qty, fee(negativo), ts(ms)
    """
    raws = _call_many(cli, ["get_user_trades", "user_trades", "my_trades", "get_fills"],
                      startTime=start_ms, endTime=end_ms, limit=limit) or []
    out: List[Dict[str, Any]] = []
    for t in raws:
        # Puede venir en varias claves
        sym0 = t.get("symbol") or t.get("instId") or t.get("market") or ""
        side0 = (t.get("side") or t.get("direction") or "").upper()
        qty0 = to_float(t.get("qty") or t.get("size") or t.get("vol") or t.get("sz") or 0.0)
        px0  = to_float(t.get("price") or t.get("fillPrice") or t.get("px") or 0.0)
        fee0 = to_float(t.get("fee") or t.get("commission") or 0.0)
        if fee0 > 0:
            fee0 = -abs(fee0)
        ts   = _ts_ms(t.get("timestamp") or t.get("time") or t.get("created") or t.get("fillTime") or 0)
        out.append({
            "symbol": normalize_symbol(sym0),
            "side": "BUY" if side0 in ("BUY", "OPEN_LONG", "CLOSE_SHORT") else "SELL",
            "price": float(px0),
            "qty": float(qty0),
            "fee": float(fee0),  # NEGATIVO
            "ts": int(ts),
        })
    return out


def _fifo_closed_from_trades_and_funding(trades: List[Dict[str, Any]],
                                         funding: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cierra posiciones por bloques donde el inventario neto vuelve a 0 (por símbolo),
    calculando PnL de precio FIFO real, fees y funding en [open, close].
    """
    fund_by_sym = defaultdict(list)
    for f in funding or []:
        fund_by_sym[f["symbol"]].append(f)
    for s in fund_by_sym:
        fund_by_sym[s].sort(key=lambda x: x["timestamp"])

    by_sym = defaultdict(list)
    for t in trades or []:
        by_sym[t["symbol"]].append(t)
    for s in by_sym:
        by_sym[s].sort(key=lambda x: x["ts"])

    closed: List[Dict[str, Any]] = []

    for sym, arr in by_sym.items():
        inv = 0.0
        lot_long: deque[Tuple[float, float]] = deque()   # (qty, price)
        lot_short: deque[Tuple[float, float]] = deque()  # (qty, price)
        block_trades: List[Dict[str, Any]] = []
        block_open_ts = None
        max_abs_inv = 0.0
        entry_acc = 0.0
        entry_qty = 0.0
        close_acc = 0.0
        close_qty = 0.0
        fee_sum = 0.0
        price_pnl = 0.0

        def _close_block():
            nonlocal block_trades, block_open_ts, max_abs_inv, entry_acc, entry_qty, close_acc, close_qty, fee_sum, price_pnl
            if not block_trades:
                return
            open_ts = block_open_ts or block_trades[0]["ts"]
            close_ts = block_trades[-1]["ts"]
            side = "long" if (block_trades[0]["side"] == "BUY") else "short"
            size = max_abs_inv
            entry_price = (entry_acc / entry_qty) if entry_qty > 0 else 0.0
            close_price = (close_acc / close_qty) if close_qty > 0 else entry_price

            f_total = 0.0
            for f in fund_by_sym.get(sym, []):
                if open_ts <= f["timestamp"] <= close_ts:
                    f_total += to_float(f.get("income", 0.0))

            closed.append({
                "exchange": EXCHANGE,
                "symbol": sym,
                "side": side,
                "size": float(size),
                "entry_price": float(entry_price),
                "close_price": float(close_price),
                "open_time": int(_ms_to_s(open_ts)),
                "close_time": int(_ms_to_s(close_ts)),
                "pnl": float(price_pnl),
                "realized_pnl": float(price_pnl + f_total + fee_sum),
                "funding_total": float(f_total),
                "fee_total": float(fee_sum),             # NEGATIVO
                "notional": float(abs(size) * entry_price),
                "initial_margin": None,
                "leverage": None,
                "liquidation_price": None,
            })

            block_trades.clear()
            block_open_ts = None
            max_abs_inv = 0.0
            entry_acc = entry_qty = close_acc = close_qty = 0.0
            fee_sum = 0.0
            price_pnl = 0.0

        for t in arr:
            if block_open_ts is None:
                block_open_ts = t["ts"]
            block_trades.append(t)
            fee_sum += to_float(t["fee"])

            if t["side"] == "BUY":
                q = t["qty"]; px = t["price"]; remain = q
                # cierra shorts
                while remain > 0 and lot_short:
                    sq, sp = lot_short[0]
                    matched = min(remain, sq)
                    price_pnl += (sp - px) * matched  # short: entry(sp) - close(px)
                    close_acc += px * matched
                    close_qty += matched
                    sq -= matched; remain -= matched
                    if sq <= 1e-12: lot_short.popleft()
                    else: lot_short[0] = (sq, sp)
                # abre long con resto
                if remain > 1e-12:
                    lot_long.append((remain, px))
                    entry_acc += px * remain
                    entry_qty += remain
                    inv += remain
                else:
                    inv += q
            else:  # SELL
                q = t["qty"]; px = t["price"]; remain = q
                # cierra longs
                while remain > 0 and lot_long:
                    lq, lp = lot_long[0]
                    matched = min(remain, lq)
                    price_pnl += (px - lp) * matched    # long: close(px) - entry(lp)
                    close_acc += px * matched
                    close_qty += matched
                    lq -= matched; remain -= matched
                    if lq <= 1e-12: lot_long.popleft()
                    else: lot_long[0] = (lq, lp)
                # abre short con resto
                if remain > 1e-12:
                    lot_short.append((remain, px))
                    entry_acc += px * remain
                    entry_qty += remain
                    inv -= remain
                else:
                    inv -= q

            max_abs_inv = max(max_abs_inv, abs(inv))

            if abs(inv) <= 1e-12:
                _close_block()

        # Si no vuelve a 0, queda abierto (no se guarda)

    return closed


def save_xt_closed_positions(db_path: str = "portfolio.db", days: int = 30, debug: bool = False) -> None:
    """
    Reconstruye posiciones cerradas de los últimos N días usando FIFO real (fills + funding) y guarda en SQLite.
    """
    cli = _client()
    if not cli:
        return
    p_closed_sync_start(EXCHANGE)

    end_ms = utc_now_ms()
    start_ms = end_ms - int(days * 24 * 60 * 60 * 1000)

    try:
        trades = _fetch_xt_user_trades(cli, start_ms, end_ms, limit=2000)
    except Exception as e:
        print(f"❌ XT trades error: {e}")
        trades = []

    try:
        funding = fetch_xt_funding_fees(limit=1000, start_ms=start_ms, end_ms=end_ms)
    except Exception:
        funding = []

    if debug:
        print(f"🔎 XT trades: {len(trades)} / funding: {len(funding)}")

    closed = _fifo_closed_from_trades_and_funding(trades, funding)
    saved = dup = 0

    for row in closed:
        try:
            if save_closed_position is None:
                raise RuntimeError("save_closed_position() no disponible")
            save_closed_position(row)
            saved += 1
            if debug:
                print(f"✅ XT guardada {row['symbol']} {row['side']} size={row['size']} realized={row['realized_pnl']:.6f}")
        except Exception as e:
            print(f"⚠️ fallo guardando {row.get('symbol')}: {e}")
            traceback.print_exc()

    if saved == 0:
        p_closed_sync_none(EXCHANGE)
    else:
        p_closed_sync_saved(EXCHANGE, saved, dup)
    p_closed_sync_done(EXCHANGE)


# ----------- DEBUGS EXTRA (RAW + TRAZA FIFO) -----------

def debug_raw_xt_trades(days: int = 30, symbol: Optional[str] = None, limit: int = 1000, max_print: int = 50):
    """Imprime el RAW de fills del SDK (sin normalizar)."""
    cli = _client()
    if not cli:
        return
    end_ms = utc_now_ms()
    start_ms = end_ms - int(days * 86400000)

    names = ["get_user_trades", "user_trades", "my_trades", "get_fills"]
    used, raw = _call_one_named(cli, names, startTime=start_ms, endTime=end_ms, limit=limit)
    print(f"🧾 RAW TRADES method={used} type={type(raw).__name__}")

    data = raw
    if isinstance(data, tuple):
        for part in reversed(data):
            if isinstance(part, (dict, list)):
                data = part
                break

    # Posible filtro por símbolo si es lista de dicts
    if symbol and isinstance(data, list) and data and isinstance(data[0], dict):
        key_opts = ("symbol", "instId", "market")
        def _match_sym(d):
            for k in key_opts:
                if k in d and normalize_symbol(str(d[k])) == normalize_symbol(symbol):
                    return True
            return False
        data = [d for d in data if _match_sym(d)]

    if isinstance(data, (list, tuple)):
        print(json.dumps(list(data)[:max_print], indent=2))
        if len(data) > max_print:
            print(f"... ({len(data)-max_print} más)")
    else:
        print(json.dumps(data, indent=2))


def debug_raw_xt_funding(days: int = 30, symbol: Optional[str] = None, limit: int = 200, max_print: int = 50):
    """Imprime el RAW de funding del SDK (sin normalizar)."""
    cli = _client()
    if not cli:
        return
    end_ms = utc_now_ms()
    start_ms = end_ms - int(days * 86400000)

    names = ["get_funding_history", "funding_list", "get_funding_fee"]
    used, raw = _call_one_named(cli, names, startTime=start_ms, endTime=end_ms, limit=limit)
    print(f"🧾 RAW FUNDING method={used} type={type(raw).__name__}")

    data = raw
    if isinstance(data, tuple):
        for part in reversed(data):
            if isinstance(part, (dict, list)):
                data = part
                break

    # Filtro por símbolo si posible
    if symbol and isinstance(data, list) and data and isinstance(data[0], dict):
        key_opts = ("symbol", "instId", "market")
        def _match_sym(d):
            for k in key_opts:
                if k in d and normalize_symbol(str(d[k])) == normalize_symbol(symbol):
                    return True
            return False
        data = [d for d in data if _match_sym(d)]

    if isinstance(data, (list, tuple)):
        print(json.dumps(list(data)[:max_print], indent=2))
        if len(data) > max_print:
            print(f"... ({len(data)-max_print} más)")
    else:
        print(json.dumps(data, indent=2))


def debug_preview_xt_closed(days: int = 30, symbol: Optional[str] = None):
    """Imprime las CLOSED reconstruidas (normalizadas) en JSON (lo que se guardaría)."""
    cli = _client()
    if not cli:
        return
    end_ms = utc_now_ms()
    start_ms = end_ms - int(days * 86400000)
    trades = _fetch_xt_user_trades(cli, start_ms, end_ms, limit=2000)
    funding = fetch_xt_funding_fees(limit=1000, start_ms=start_ms, end_ms=end_ms)
    if symbol:
        ns = normalize_symbol(symbol)
        trades = [t for t in trades if t["symbol"] == ns]
        funding = [f for f in funding if f["symbol"] == ns]
    closed = _fifo_closed_from_trades_and_funding(trades, funding)
    print(json.dumps(closed, indent=2))


def debug_trace_xt_fifo(days: int = 30, symbol: Optional[str] = None, max_matches: int = 200):
    """
    Traza paso a paso el algoritmo FIFO: inventario, emparejamientos, PnL parcial por match,
    sums de fees, entrada/salida promedio del bloque y funding dentro de la ventana.
    """
    cli = _client()
    if not cli:
        return

    end_ms = utc_now_ms()
    start_ms = end_ms - int(days * 86400000)
    trades = _fetch_xt_user_trades(cli, start_ms, end_ms, limit=2000)
    funding = fetch_xt_funding_fees(limit=1000, start_ms=start_ms, end_ms=end_ms)

    if symbol:
        ns = normalize_symbol(symbol)
        trades = [t for t in trades if t["symbol"] == ns]
        funding = [f for f in funding if f["symbol"] == ns]

    # agrupar por símbolo
    by_sym = defaultdict(list)
    for t in trades:
        by_sym[t["symbol"]].append(t)
    for s in by_sym:
        by_sym[s].sort(key=lambda x: x["ts"])

    fund_by_sym = defaultdict(list)
    for f in funding:
        fund_by_sym[f["symbol"]].append(f)
    for s in fund_by_sym:
        fund_by_sym[s].sort(key=lambda x: x["timestamp"])

    for sym, arr in by_sym.items():
        print(f"\n🔍 TRACE {sym}")
        inv = 0.0
        lot_long: deque[Tuple[float, float]] = deque()
        lot_short: deque[Tuple[float, float]] = deque()
        block_id = 0
        open_ts = None
        entry_acc = entry_qty = close_acc = close_qty = 0.0
        fee_sum = 0.0
        price_pnl = 0.0
        max_inv = 0.0
        match_count = 0

        def end_block(final_ts):
            nonlocal block_id, entry_acc, entry_qty, close_acc, close_qty, fee_sum, price_pnl, open_ts
            nonlocal match_count
            if open_ts is None:
                return
            block_id += 1
            entry_price = (entry_acc/entry_qty) if entry_qty > 0 else 0.0
            close_price = (close_acc/close_qty) if close_qty > 0 else entry_price
            f_total = sum(to_float(f.get("income", 0.0)) for f in fund_by_sym.get(sym, []) if open_ts <= f["timestamp"] <= final_ts)
            side = "long" if entry_qty and (arr[0]["side"] == "BUY") else "short"
            size = max_inv
            realized = price_pnl + f_total + fee_sum
            print(f"  └─ [BLOCK #{block_id}] side={side} size={size:.6f} open={open_ts} close={final_ts}")
            print(f"     entry_avg={entry_price:.6f} close_avg={close_price:.6f} price_pnl={price_pnl:.6f}")
            print(f"     fees(neg)={fee_sum:.6f} funding={f_total:.6f} realized={realized:.6f} matches={match_count}")
            # reset
            open_ts = None
            entry_acc = entry_qty = close_acc = close_qty = 0.0
            fee_sum = 0.0
            price_pnl = 0.0
            match_count = 0

        for t in arr:
            if open_ts is None:
                open_ts = t["ts"]
                max_inv = 0.0
            fee_sum += to_float(t["fee"])
            if t["side"] == "BUY":
                q = t["qty"]; px = t["price"]; remain = q
                while remain > 0 and lot_short:
                    sq, sp = lot_short[0]
                    matched = min(remain, sq)
                    pnl = (sp - px) * matched
                    price_pnl += pnl
                    close_acc += px * matched
                    close_qty += matched
                    match_count += 1
                    print(f"   • match SHORT {matched:.6f} @entry={sp:.6f} close={px:.6f} pnl={pnl:.6f}")
                    sq -= matched; remain -= matched
                    if sq <= 1e-12: lot_short.popleft()
                    else: lot_short[0] = (sq, sp)
                if remain > 1e-12:
                    lot_long.append((remain, px))
                    entry_acc += px * remain
                    entry_qty += remain
                    inv += remain
                else:
                    inv += q
            else:
                q = t["qty"]; px = t["price"]; remain = q
                while remain > 0 and lot_long:
                    lq, lp = lot_long[0]
                    matched = min(remain, lq)
                    pnl = (px - lp) * matched
                    price_pnl += pnl
                    close_acc += px * matched
                    close_qty += matched
                    match_count += 1
                    print(f"   • match LONG  {matched:.6f} @entry={lp:.6f} close={px:.6f} pnl={pnl:.6f}")
                    lq -= matched; remain -= matched
                    if lq <= 1e-12: lot_long.popleft()
                    else: lot_long[0] = (lq, lp)
                if remain > 1e-12:
                    lot_short.append((remain, px))
                    entry_acc += px * remain
                    entry_qty += remain
                    inv -= remain
                else:
                    inv -= q

            max_inv = max(max_inv, abs(inv))
            if abs(inv) <= 1e-12:
                end_block(t["ts"])

        # si quedó abierto, no cierra bloque (posición abierta); se informa
        if abs(inv) > 1e-12:
            print(f"  ⚠ Bloque INCOMPLETO: inventario final {inv:.6f} (posición abierta)")

# ---------------- CLI / AUTOEJECUCIÓN ----------------

def _autorun_default():
    """Ejecución por defecto si no se pasan argumentos: resumen rápido de datos y ejemplos."""
    cli = _client()
    if not cli:
        return
    print("🚀 XT DEBUG QUICK-RUN")
    end_ms = utc_now_ms()
    start_ms = end_ms - int(3 * 86400000)  # 3 días

    # sample RAW counts
    _, raw_tr = _call_one_named(cli, ["get_user_trades", "user_trades", "my_trades", "get_fills"],
                                startTime=start_ms, endTime=end_ms, limit=1000)
    _, raw_fu = _call_one_named(cli, ["get_funding_history", "funding_list", "get_funding_fee"],
                                startTime=start_ms, endTime=end_ms, limit=500)
    tr_n = len(raw_tr) if isinstance(raw_tr, (list, tuple)) else (len(raw_tr.get("data", [])) if isinstance(raw_tr, dict) else 0)
    fu_n = len(raw_fu) if isinstance(raw_fu, (list, tuple)) else (len(raw_fu.get("data", [])) if isinstance(raw_fu, dict) else 0)
    print(f" • RAW trades últimos 3 días: {tr_n}")
    print(f" • RAW funding últimos 3 días: {fu_n}")

    # preview closed normalizadas
    print("\n📦 CLOSED (reconstruidas) ejemplo:")
    debug_preview_xt_closed(days=30)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="XT adapter CLI / Debug")
    ap.add_argument("--opens", action="store_true", help="Muestra abiertas normalizadas")
    ap.add_argument("--funding", type=int, default=0, help="Muestra N registros de funding normalizado")
    ap.add_argument("--preview", action="store_true", help="Preview closed (normalizado, no guarda)")
    ap.add_argument("--trace-fifo", action="store_true", help="Traza paso a paso el algoritmo FIFO")
    ap.add_argument("--raw-trades", action="store_true", help="Imprime RAW de trades/fills del SDK")
    ap.add_argument("--raw-funding", action="store_true", help="Imprime RAW de funding del SDK")
    ap.add_argument("--days", type=int, default=30, help="Ventana en días para debug")
    ap.add_argument("--symbol", type=str, default=None, help="Filtrar por símbolo (opcional)")
    ap.add_argument("--save-closed", action="store_true", help="Reconstruye y guarda closed positions en DB")
    ap.add_argument("--balance", action="store_true", help="Muestra balance normalizado /api/balances")
    ap.add_argument("--balance-raw", action="store_true", help="Dump del payload crudo de XT para balances")
    args = ap.parse_args()

    ran = False
    if args.opens:
        print(json.dumps(fetch_xt_open_positions(), indent=2)); ran = True
    if args.funding:
        print(json.dumps(fetch_xt_funding_fees(limit=args.funding), indent=2)); ran = True
    if args.preview:
        debug_preview_xt_closed(days=args.days, symbol=args.symbol); ran = True
    if args.trace_fifo:
        debug_trace_xt_fifo(days=args.days, symbol=args.symbol); ran = True
    if args.raw_trades:
        debug_raw_xt_trades(days=args.days, symbol=args.symbol); ran = True
    if args.raw_funding:
        debug_raw_xt_funding(days=args.days, symbol=args.symbol); ran = True
    if args.save_closed:
        save_xt_closed_positions(days=args.days, debug=True); ran = True
    if args.balance_raw:
        debug_dump_xt_balance_raw()
    if args.balance:
        print(json.dumps(fetch_xt_all_balances() or {}, indent=2))    

    if not ran:
        _autorun_default()

# -------- export API --------
__all__ = [
    "fetch_xt_open_positions",
    "fetch_xt_funding_fees",
    "fetch_xt_all_balances",
    "save_xt_closed_positions",
    "debug_preview_xt_closed",
    "debug_dump_xt_opens",      # retrocompat si lo usabas
    "debug_dump_xt_funding",    # retrocompat si lo usabas
    "debug_raw_xt_trades",
    "debug_raw_xt_funding",
    "debug_trace_xt_fifo",
]

# (Retrocompat: si tenías estas funciones en otras versiones)
def debug_dump_xt_opens():
    print(json.dumps(fetch_xt_open_positions(), indent=2))

def debug_dump_xt_funding():
    print(json.dumps(fetch_xt_funding_fees(limit=100), indent=2))
