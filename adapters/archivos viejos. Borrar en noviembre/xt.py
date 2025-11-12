# adapters/xt.py
# XT.COM Perp adapter – open positions, funding, balances y closed positions por FIFO
# Requiere: pip install pyxt
from __future__ import annotations

from __future__ import annotations

import os, math, json, traceback
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]  # .../extended-web
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# SDK oficial
try:
    from pyxt.perp import Perp
except Exception:
    Perp = None  # manejamos gracefully si no está instalado

# Helpers / contratos ya usados en tu proyecto
from utils.symbols import normalize_symbol  # normalizador global del proyecto
from utils.time import to_ms, to_s, utc_now_ms
from utils.money import D, quant, normalize_fee, to_float
from db_manager import save_closed_position  # guarda en SQLite con tu nuevo esquema  (pnl, apr, etc)  :contentReference[oaicite:2]{index=2}

# (Opcional) helpers de print del backend; si no existen (import cycle), we noop
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
    """Intenta varias firmas de método del SDK para tolerar cambios de nombre."""
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

def _qty(sign_qty: float) -> Tuple[str, float]:
    """Convierte qty con signo en (side, abs_qty)."""
    return ("long", abs(sign_qty)) if sign_qty >= 0 else ("short", abs(sign_qty))

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

# ------------ API pública (expuesta en __all__) ------------
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

    # Tolerante a cambios de SDK:
    raw = _call_many(cli, [
        "get_positions", "position_list", "get_account_positions", "positions"
    ]) or []

    out: List[Dict[str, Any]] = []
    for p in (raw or []):
        # mapeo tolerante de campos
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
        if fees > 0:  # en tus contratos fee acumulado es NEGATIVO si es costo
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
    # prints por símbolo (opcional)
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
        # 7 días por defecto
        start_ms = end_ms - 7*24*60*60*1000

    # SDK tolerante
    # candidatos: get_funding_fee, get_funding_history, funding_list, funding_fees
    resp = _call_many(cli, ["get_funding_history", "funding_list", "get_funding_fee"], startTime=start_ms, endTime=end_ms, limit=limit) or []

    out: List[Dict[str, Any]] = []
    for it in (resp or []):
        sym0 = it.get("symbol") or it.get("instId") or it.get("market") or ""
        base = normalize_symbol(sym0)
        ts   = _ts_ms(it.get("timestamp") or it.get("time") or it.get("created"))
        income = to_float(it.get("amount") or it.get("income") or it.get("delta") or 0.0)
        # signo natural: + cobro / - pago
        rate = to_float(it.get("fundingRate") or it.get("rate") or 0.0)
        asset = (it.get("asset") or it.get("ccy") or "USDT").upper()
        out.append({
            "exchange": EXCHANGE,
            "symbol": base,
            "income": float(income),
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
    data = _call_many(cli, ["get_account_capital", "get_account", "capital"]) or {}
    # mapeo tolerante
    equity  = to_float(data.get("totalEquity") or data.get("equity") or data.get("nav") or 0.0)
    balance = to_float(data.get("totalWalletBalance") or data.get("balance") or 0.0)
    upnl    = to_float(data.get("unrealizedPNL") or data.get("unrealizedPnl") or (equity - balance))
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
    Campos esperados por este adapter (mapeados):
      symbol, side(BUY/SELL), price, qty, fee(negativo), timestamp(ms)
    """
    raws = _call_many(cli, ["get_user_trades", "user_trades", "my_trades", "get_fills"],
                      startTime=start_ms, endTime=end_ms, limit=limit) or []
    out = []
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
    calculando PnL de precio FIFO real, fees y funding sumado en la ventana [open, close].
    """
    # funding por símbolo ordenado por ts
    fund_by_sym = defaultdict(list)
    for f in funding or []:
        fund_by_sym[f["symbol"]].append(f)
    for s in fund_by_sym:
        fund_by_sym[s].sort(key=lambda x: x["timestamp"])

    # agrupar trades por símbolo y ordenar por ts
    by_sym = defaultdict(list)
    for t in trades or []:
        by_sym[t["symbol"]].append(t)
    for s in by_sym:
        by_sym[s].sort(key=lambda x: x["ts"])

    closed: List[Dict[str, Any]] = []

    for sym, arr in by_sym.items():
        inv = 0.0  # inventario (BUY +, SELL -)
        lot_long: deque[Tuple[float, float]] = deque()   # (qty, price)
        lot_short: deque[Tuple[float, float]] = deque()  # (qty, price)  para shorts consideramos "SELL" abre
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

            # funding total dentro de la ventana
            fund_list = fund_by_sym.get(sym, [])
            f_total = 0.0
            if fund_list:
                # binaria sencilla por rango
                for f in fund_list:
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

            # reset bloque
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
                # cierra shorts primero (si hay)
                q = t["qty"]
                px = t["price"]
                remain = q
                while remain > 0 and lot_short:
                    sq, sp = lot_short[0]
                    matched = min(remain, sq)
                    # para short: PnL = (entry - close)*qty, entry=sp, close=px
                    price_pnl += (sp - px) * matched
                    close_acc += px * matched
                    close_qty += matched
                    sq -= matched
                    remain -= matched
                    if sq <= 1e-12:
                        lot_short.popleft()
                    else:
                        lot_short[0] = (sq, sp)
                # lo que no cerró abre long
                if remain > 1e-12:
                    lot_long.append((remain, px))
                    entry_acc += px * remain
                    entry_qty += remain
                    inv += remain
                else:
                    inv += q  # cierra parcialmente (inv sube menos)
            else:  # SELL
                q = t["qty"]
                px = t["price"]
                remain = q
                # cierra longs primero
                while remain > 0 and lot_long:
                    lq, lp = lot_long[0]
                    matched = min(remain, lq)
                    # long: (close - entry)*qty = (px - lp)*matched
                    price_pnl += (px - lp) * matched
                    close_acc += px * matched
                    close_qty += matched
                    lq -= matched
                    remain -= matched
                    if lq <= 1e-12:
                        lot_long.popleft()
                    else:
                        lot_long[0] = (lq, lp)
                # lo que no cerró abre short
                if remain > 1e-12:
                    lot_short.append((remain, px))
                    entry_acc += px * remain
                    entry_qty += remain
                    inv -= remain
                else:
                    inv -= q  # cierra parcialmente

            max_abs_inv = max(max_abs_inv, abs(inv))

            # cuando el neto vuelve a 0, cerramos bloque
            if abs(inv) <= 1e-12:
                _close_block()

        # si por alguna razón quedó algo abierto (no volvió a 0), no se guarda como closed

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
            # save_closed_position ya recalcula métricas finales y hace sanity-checks. :contentReference[oaicite:3]{index=3}
            save_closed_position(row)
            saved += 1
            if debug:
                print(f"✅ XT guardada {row['symbol']} {row['side']} size={row['size']} realized={row['realized_pnl']:.6f}")
        except Exception as e:
            dup += 0  # si tuvieras lógica anti-dup, increméntala aquí
            print(f"⚠️ fallo guardando {row['symbol']}: {e}")
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
        trades = [t for t in trades if t["symbol"] == normalize_symbol(symbol)]
        funding = [f for f in funding if f["symbol"] == normalize_symbol(symbol)]
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

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="XT adapter CLI")
    ap.add_argument("--opens", action="store_true", help="Muestra abiertas normalizadas")
    ap.add_argument("--funding", type=int, default=0, help="Muestra N registros de funding")
    ap.add_argument("--preview", action="store_true", help="Preview closed (no guarda)")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--symbol", type=str, default=None)
    ap.add_argument("--save-closed", action="store_true", help="Reconstruye y guarda closed positions")
    args = ap.parse_args()

    if args.opens:
        debug_dump_xt_opens()
    if args.funding:
        print(json.dumps(fetch_xt_funding_fees(limit=args.funding), indent=2))
    if args.preview:
        debug_preview_xt_closed(days=args.days, symbol=args.symbol)
    if args["save_closed"] if isinstance(args, dict) else args.save_closed:
        save_xt_closed_positions(days=args.days, debug=True)
