# adapters/xt.py
# XT.COM Perp adapter – open positions, funding, balances y closed positions por FIFO
from __future__ import annotations

import os, json, traceback
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

# SDK oficial (opcional)
try:
    from pyxt.perp import Perp
except Exception:
    Perp = None  # tolera que no esté instalado

# Helpers del proyecto (ajusta a tus módulos reales)
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
    """Intenta varias firmas de método del SDK para tolerar cambios."""
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


def fetch_xt_all_balances(db_path: str = "portfolio.db") -> Dict[str, Any] | None:
    """
    Shape EXACTO para /api/balances:
    {"exchange":"xt","equity":float,"balance":float,"unrealized_pnl":float,"initial_margin":0.0,"spot":0.0,"margin":0.0,"futures":float}
    """
    cli = _client()
    if not cli:
        return None

    # XT puede devolver tuple/list/dict y envolver el body
    raw = _call_many(cli, [
        "get_account_capital", "get_account", "capital",
        "balance_list", "get_balance_list"
    ])

    data = raw

    # --- unwrap robusto ---
    if isinstance(data, tuple):
        # suele venir (method, url, headers, params, body, data, code) o similar
        # buscamos la parte útil (dict/list) empezando desde el final
        for part in reversed(data):
            if isinstance(part, (dict, list)):
                data = part
                break

    if isinstance(data, dict) and ("data" in data or "result" in data):
        data = data.get("data") or data.get("result") or data

    equity = balance = upnl = 0.0
    if isinstance(data, dict):
        equity  = to_float(data.get("totalEquity") or data.get("equity") or data.get("nav") or 0.0)
        balance = to_float(data.get("totalWalletBalance") or data.get("balance") or 0.0)
        upnl    = to_float(data.get("unrealizedPNL") or data.get("unrealizedPnl") or (equity - balance))
    elif isinstance(data, list):
        # Lista por moneda/settle: sumamos equity y uPNL; balance ≈ equity - uPNL
        for it in data:
            if not isinstance(it, dict):
                continue
            e = to_float(
                it.get("equity") or it.get("balance") or
                it.get("cashBalance") or it.get("accountBalance") or 0.0
            )
            u = to_float(it.get("unrealizedPNL") or it.get("unrealizedPnl") or it.get("unrealizedProfit") or 0.0)
            if e == 0.0:
                e = to_float(it.get("available") or 0.0) + u
            equity += e
            upnl   += u
        balance = equity - upnl

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


# --------- FIFO sobre fills + funding ----------

def _fetch_xt_user_trades(cli: Perp, start_ms: int, end_ms: int, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Devuelve fills del usuario (tolerante a método).
    Campos que usamos:
      symbol, side(BUY/SELL), price, qty, fee(negativo), ts(ms)
    """
    raws = _call_many(cli, ["get_user_trades", "user_trades", "my_trades", "get_fills"],
                      startTime=start_ms, endTime=end_ms, limit=limit) or []
    out: List[Dict[str, Any]] = []
    for t in raws:
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


# -------- Debug helpers ----------

def debug_preview_xt_closed(days: int = 3, symbol: Optional[str] = None):
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

def debug_dump_xt_opens():
    print(json.dumps(fetch_xt_open_positions(), indent=2))

def debug_dump_xt_funding():
    print(json.dumps(fetch_xt_funding_fees(limit=100), indent=2))


__all__ = [
    "fetch_xt_open_positions",
    "fetch_xt_funding_fees",
    "fetch_xt_all_balances",
    "save_xt_closed_positions",
    "debug_preview_xt_closed",
    "debug_dump_xt_opens",
    "debug_dump_xt_funding",
]
