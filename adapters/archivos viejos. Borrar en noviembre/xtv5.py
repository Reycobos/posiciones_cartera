# adapters/xt.py
# XT adapter (Futures + Spot) usando el SDK pyxt (perp.py/spot.py locales si no hay pip).
from __future__ import annotations

import os
import json
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

# ============ SDK (pip o archivos locales) ============
try:
    from pyxt.perp import Perp  # pip install pyxt
except Exception:
    from perp import Perp       # /mnt/data/perp.py

try:
    from pyxt.spot import Spot  # pip install pyxt
except Exception:
    from spot import Spot       # /mnt/data/spot.py

# ============ Helpers de impresión del backend (no-op si no existen) ============
def _noop(*a, **k): pass
try:
    from portfoliov7 import (
        p_balance_equity, p_balance_fetching, p_balance_done,
        p_funding_fetching, p_funding_count,
        p_open_fetching, p_open_count,
        p_closed_sync_start, p_closed_sync_saved, p_closed_sync_done, p_closed_sync_none,
    )
except Exception:
    p_balance_equity = p_balance_fetching = p_balance_done = _noop
    p_funding_fetching = p_funding_count = _noop
    p_open_fetching = p_open_count = _noop
    p_closed_sync_start = p_closed_sync_saved = p_closed_sync_done = p_closed_sync_none = _noop

# ============ Normalización de símbolos ============
try:
    from symbols import normalize_symbol
except Exception:
    import re
    def normalize_symbol(sym: str) -> str:
        if not sym: return ""
        s = sym.upper()
        s = re.sub(r'^PERP_', '', s)
        s = re.sub(r'(_|-)?(USDT|USDC|USD|PERP)$', '', s)
        s = re.sub(r'[_-]+$', '', s)
        s = re.split(r'[_/-]', s)[0]
        return s

# ============ DB manager ============
try:
    from db_manager import save_closed_position
except Exception:
    # Fallback que imprime lo que guardaríamos si el módulo no está.
    def save_closed_position(position: dict):
        print("⚠️ db_manager.save_closed_position no disponible; payload:")
        print(json.dumps(position, indent=2, ensure_ascii=False))

# ============ Config/ENV ============
EXCHANGE = "xt"
XT_API_KEY = "96b1c999-b9df-42ed-9d4f-a4491aac7a1d"
XT_API_SECRET = "5f60a0a147d82db2117a45da129a6f4480488234"
XT_FAPI_HOST = os.getenv("XT_FAPI_HOST", "https://fapi.xt.com")
XT_SAPI_HOST = os.getenv("XT_SAPI_HOST", "https://sapi.xt.com")

DEFAULT_DAYS_TRADES = int(os.getenv("XT_DAYS_TRADES", "14"))

# ============ Utils ============
def to_float(x) -> float:
    try: return float(x)
    except Exception: return 0.0

def utc_now_ms() -> int:
    return int(time.time() * 1000)

def _unwrap_result(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "result" in obj: return obj["result"]
        if "data" in obj: return obj["data"]
        if "items" in obj: return obj["items"]
        if "list" in obj: return obj["list"]
    return obj

def _fee_for_trade(order_type: Optional[str], price: float, qty: float) -> float:
    """
    Calcula la comisión por trade según orderType:
      MARKET -> 0.0588%
      LIMIT  -> 0.038%
    Devuelve SIEMPRE negativa (costo).
    """
    t = (order_type or "").upper()
    if "MARKET" in t:
        rate = 0.000588
    elif "LIMIT" in t:
        rate = 0.000380
    else:
        # si no llega el tipo, asumimos LIMIT por defecto (según tu instrucción)
        rate = 0.000380
    return -abs(rate * price * qty)
# =========================================================
#                      BALANCES (COMBINADO)
#   Futures:  /future/user/v1/balance/list   -> walletBalance
#   Spot:     /v4/balances
# =========================================================
_spot_cli: Optional[Spot] = None
_perp_cli: Optional[Perp] = None

def _get_spot() -> Spot:
    global _spot_cli
    if _spot_cli is None:
        _spot_cli = Spot(host=XT_SAPI_HOST, access_key=XT_API_KEY, secret_key=XT_API_SECRET)
    return _spot_cli

def _get_perp() -> Perp:
    global _perp_cli
    if _perp_cli is None:
        _perp_cli = Perp(host=XT_FAPI_HOST, access_key=XT_API_KEY, secret_key=XT_API_SECRET)
    return _perp_cli

def fetch_xt_all_balances(db_path: str = "portfolio.db") -> Dict[str, Any]:
    """
    Devuelve estructura EXACTA para /api/balances combinando:
    - spot: /v4/balances
    - futures: /future/user/v1/balance/list (get_account_capital)
    """
    p_balance_fetching(EXCHANGE)

    # -------- Spot --------
    spot_equity = 0.0
    spot_avail = 0.0
    try:
        cli_s = _get_spot()
        res_s = cli_s.balances(currencies=None)  # GET /v4/balances
        assets = (res_s or {}).get("assets") if isinstance(res_s, dict) else res_s
        if isinstance(assets, list):
            for a in assets:
                if not isinstance(a, dict): continue
                avail = to_float(a.get("availableAmount") or 0.0)
                total = to_float(a.get("totalAmount") or 0.0)
                # No tenemos conversión a USDT aquí; usamos total.
                spot_equity += total
                spot_avail += avail
    except Exception:
        # Spot puede no estar habilitado → seguimos con futuros
        pass

    # -------- Futuros --------
    cli_f = _get_perp()
    code, success, error = cli_f.get_account_capital()   # GET /future/user/v1/balance/list
    if error or code != 200 or success is None:
        raise RuntimeError(f"XT futures balance error: {error or code}")
    res_f = _unwrap_result(success)

    futures_equity = 0.0
    futures_unreal = 0.0
    if isinstance(res_f, list):
        for it in res_f:
            if isinstance(it, dict):
                futures_equity += to_float(it.get("walletBalance") or 0.0)
                futures_unreal += to_float(it.get("notProfit") or it.get("unrealizedProfit") or 0.0)
    elif isinstance(res_f, dict):
        arr = res_f.get("items") or res_f.get("list") or []
        for it in arr or []:
            if isinstance(it, dict):
                futures_equity += to_float(it.get("walletBalance") or 0.0)
                futures_unreal += to_float(it.get("notProfit") or it.get("unrealizedProfit") or 0.0)

    total_equity = spot_equity + futures_equity
    out = {
        "exchange": EXCHANGE,
        "equity": float(total_equity),
        "balance": float(total_equity),
        "unrealized_pnl": float(futures_unreal),
        "initial_margin": 0.0,
        "spot": float(spot_equity),
        "margin": 0.0,
        "futures": float(futures_equity),
    }
    p_balance_equity(EXCHANGE, out["equity"])
    p_balance_done(EXCHANGE)
    return out

# =========================================================
#                    FUNDING FEES
#   GET /future/user/v1/balance/funding-rate-list
# =========================================================
def fetch_xt_funding_fees(limit: int = 50,
                          start_ms: Optional[int] = None,
                          end_ms: Optional[int] = None,
                          symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    p_funding_fetching(EXCHANGE)
    cli = _get_perp()

    if end_ms is None:
        end_ms = utc_now_ms()
    if start_ms is None:
        start_ms = end_ms - 14 * 24 * 60 * 60 * 1000

    out: List[Dict[str, Any]] = []
    next_id: Optional[int] = None
    direction = "NEXT"
    path = "/future/user" + "/v1/balance/funding-rate-list"

    while len(out) < limit:
        page_size = min(100, max(1, limit - len(out)))
        params: Dict[str, Any] = {
            "limit": page_size,
            "direction": direction,
            "startTime": int(start_ms),
            "endTime": int(end_ms),
        }
        if symbol:
            params["symbol"] = symbol

        header = cli._create_sign(XT_API_KEY, XT_API_SECRET,
                                  path=path, bodymod="application/x-www-form-urlencoded",
                                  params=params)
        header["Content-Type"] = "application/x-www-form-urlencoded"
        url = cli.host + path
        code, success, error = cli._fetch(method="GET", url=url, headers=header, params=params, timeout=cli.timeout)
        if error or code != 200 or success is None:
            raise RuntimeError(f"XT funding error: {error or code}")

        res = _unwrap_result(success)
        items = []
        if isinstance(res, dict):
            items = res.get("items") or res.get("list") or []
        if not isinstance(items, list):
            items = []

        for it in items:
            sym_raw = str(it.get("symbol") or "")
            base = normalize_symbol(sym_raw)
            income = to_float(it.get("cast") or 0.0)
            asset = (it.get("coin") or "USDT").upper()
            ts = int(it.get("createdTime") or 0)
            ts = ts if ts > 10**12 else ts * 1000
            out.append({
                "exchange": EXCHANGE,
                "symbol": base,
                "income": float(income),
                "asset": "USDT" if asset not in ("USDT", "USDC", "USD") else asset,
                "timestamp": ts,
                "funding_rate": 0.0,
                "type": "FUNDING_FEE",
            })
            if len(out) >= limit:
                break

        if isinstance(res, dict) and res.get("hasNext") and items:
            next_id = items[-1].get("id")
            direction = "NEXT"
        else:
            break

    p_funding_count(EXCHANGE, len(out))
    return out

# =========================================================
#                    OPEN / ACTIVE POSITIONS
#   GET /future/user/v1/position/list  (o get_position(symbol))
# =========================================================
def fetch_xt_open_positions() -> List[Dict[str, Any]]:
    p_open_fetching(EXCHANGE)
    cli = _get_perp()

    # "todas": replicamos get_position() pero sin symbol
    path = "/future/user" + "/v1/position/list"
    params: Dict[str, Any] = {}
    header = cli._create_sign(XT_API_KEY, XT_API_SECRET, path=path,
                              bodymod="application/x-www-form-urlencoded", params=params)
    header["Content-Type"] = "application/x-www-form-urlencoded"
    url = cli.host + path
    code, success, error = cli._fetch(method="GET", url=url, headers=header, params=params, timeout=cli.timeout)
    if error or code != 200 or success is None:
        raise RuntimeError(f"XT positions error: {error or code}")

    arr = _unwrap_result(success)
    if not isinstance(arr, list):
        arr = (arr.get("items") or arr.get("list") or [])

    out: List[Dict[str, Any]] = []
    for p in arr or []:
        if not isinstance(p, dict): 
            continue
        sym_raw = str(p.get("symbol") or "")
        base = normalize_symbol(sym_raw)
        side0 = (p.get("positionSide") or "").upper()
        side = "long" if side0 in ("LONG", "BID", "BUY") else "short"

        size = abs(to_float(p.get("positionSize") or p.get("availableCloseSize") or 0.0))
        entry = to_float(p.get("entryPrice") or 0.0)
        mark  = to_float(p.get("calMarkPrice") or p.get("markPrice") or entry)
        liq   = to_float(p.get("breakPrice") or 0.0)  # blowout price
        upnl  = to_float(p.get("floatingPL") or p.get("unrealizedPnl") or 0.0)

        out.append({
            "exchange": EXCHANGE,
            "symbol": base,
            "side": side,
            "size": float(size),
            "entry_price": float(entry),
            "mark_price": float(mark),
            "liquidation_price": float(liq or 0.0),
            "notional": float(abs(size) * entry),
            "unrealized_pnl": float(upnl),
            "fee": 0.0,
            "funding_fee": 0.0,
            "realized_pnl": 0.0,
        })

    p_open_count(EXCHANGE, len(out))
    return out

# =========================================================
#                 CLOSED (Reconstrucción FIFO con órdenes)
#     No hay endpoint de fills → usamos /future/trade/v1/order/list-history
# =========================================================
def _orders_as_pseudo_fills(start_ms: int, end_ms: int,
                            symbol: Optional[str] = None,
                            limit: int = 2000) -> List[Dict[str, Any]]:
    """
    Convierte órdenes del histórico en “pseudo-fills”:
      price = avgPrice (si 0 → price)
      qty   = executedQty (>0)  <<--- tamaño correcto
      side  = orderSide (BUY/SELL)
      ts    = updatedTime (fallback createdTime)
      fee   = cálculo por orderType (MARKET/LIMIT)  <<--- aplicado por trade
    """
    cli = _get_perp()
    out: List[Dict[str, Any]] = []

    page_limit = min(1000, max(1, limit))
    direction = None
    last_id = None

    while len(out) < limit:
        code, success, error = cli.get_history_order(
            symbol=symbol, direction=direction, oid=last_id,
            limit=page_limit, start_time=int(start_ms), end_time=int(end_ms)
        )
        if error or code != 200 or success is None:
            raise RuntimeError(f"XT order history error: {error or code}")
        res = _unwrap_result(success)
        items = res if isinstance(res, list) else (res.get("items") or res.get("list") or [])
        if not items:
            break

        for it in items:
            if not isinstance(it, dict):
                continue

            # ---- tamaño de ejecución correcto ----
            qty = to_float(it.get("executedQty") or it.get("executedqty") or 0.0)
            if qty <= 0:
                continue

            prc = to_float(it.get("avgPrice") or it.get("avgprice") or it.get("price") or 0.0)
            side = (it.get("orderSide") or it.get("side") or "").upper()
            sym = it.get("symbol") or ""
            ts  = int(it.get("updatedTime") or it.get("createdTime") or 0)
            ts  = ts if ts > 10**12 else ts * 1000
            sd  = "BUY" if side in ("BUY", "LONG", "BID") else "SELL"

            # ---- fee por orderType (por trade) ----
            otype = it.get("orderType") or it.get("type") or ""
            fee   = _fee_for_trade(otype, prc, qty)

            out.append({
                "price": float(prc),
                "qty": abs(float(qty)),
                "side": sd,
                "fee": float(fee),              # NEGATIVA
                "symbol": sym,
                "timestamp": ts,
                "orderType": str(otype or "").upper(),
            })

        # paginado
        last = items[-1] if items else None
        if last and "id" in last:
            last_id = last["id"]
            direction = "NEXT"
        else:
            break

        if len(items) < page_limit:
            break

    out.sort(key=lambda x: x["timestamp"])
    return out[:limit]


def _fifo_blocks_from_fills(fills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Bloques cerrados por símbolo (net -> 0). Reglas:
      - side del bloque = primer trade (BUY=long, SELL=short)
      - entry_avg, close_avg ponderados por qty
      - size del bloque = suma de executedQty de entradas (no max net)
      - price_pnl FIFO real
      - fee_total = suma de fees de los trades del bloque (siempre NEGATIVA)
      - funding_total = 0.0 (se inyecta fuera si se desea)
    """
    by_sym: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in fills:
        s = normalize_symbol(f.get("symbol") or "")
        if not s:
            continue
        ff = dict(f)
        ff["symbol"] = s
        by_sym[s].append(ff)

    results: List[Dict[str, Any]] = []

    for sym, arr in by_sym.items():
        arr.sort(key=lambda x: x["timestamp"])
        i = 0
        while i < len(arr):
            first_side = arr[i]["side"]
            net = 0.0
            fees_sum = 0.0
            open_time = arr[i]["timestamp"]

            # colas FIFO de entradas según dirección
            q_long: deque[Tuple[float, float]] = deque()
            q_short: deque[Tuple[float, float]] = deque()

            # para medias y tamaño final
            entry_sum = close_sum = 0.0
            entry_qty = close_qty = 0.0

            j = i
            while j < len(arr):
                f = arr[j]
                qty = float(f["qty"])
                price = float(f["price"])
                side = f["side"]
                fees_sum += float(f.get("fee") or 0.0)

                if first_side == "BUY":
                    # long: entradas BUY, salidas SELL
                    if side == "BUY":
                        q_long.append((qty, price))
                        net += qty
                        entry_sum += price * qty
                        entry_qty += qty
                    else:
                        # cerrar long con SELL → emparejar contra q_long
                        qleft = qty
                        close_sum += price * qty
                        close_qty += qty
                        net -= qty
                        while qleft > 1e-12 and q_long:
                            lqty, lprice = q_long[0]
                            take = min(lqty, qleft)
                            # (close - entry) * qty
                            # lo acumulamos en price_pnl más tarde; aquí solo vaciamos entradas
                            lqty -= take
                            qleft -= take
                            if lqty <= 1e-12: q_long.popleft()
                            else: q_long[0] = (lqty, lprice)
                else:
                    # short: entradas SELL, salidas BUY
                    if side == "SELL":
                        q_short.append((qty, price))
                        net -= qty
                        entry_sum += price * qty
                        entry_qty += qty
                    else:
                        qleft = qty
                        close_sum += price * qty
                        close_qty += qty
                        net += qty
                        while qleft > 1e-12 and q_short:
                            sqty, sprice = q_short[0]
                            take = min(sqty, qleft)
                            sqty -= take
                            qleft -= take
                            if sqty <= 1e-12: q_short.popleft()
                            else: q_short[0] = (sqty, sprice)

                if abs(net) <= 1e-12 and entry_qty > 0 and close_qty > 0:
                    break
                j += 1

            if abs(net) > 1e-12 or entry_qty <= 0 or close_qty <= 0:
                # no cerró bloque desde i
                i = j + 1
                continue

            close_time = arr[j]["timestamp"]
            side_txt = "long" if first_side == "BUY" else "short"

            # ---- PnL FIFO real (reproduce el emparejamiento para sumar)
            price_pnl = 0.0
            ql: deque[Tuple[float, float]] = deque()
            qs: deque[Tuple[float, float]] = deque()
            for k in range(i, j + 1):
                f = arr[k]; qty = float(f["qty"]); price = float(f["price"]); side = f["side"]
                if side == "BUY":
                    # cierra short
                    while qty > 1e-12 and qs:
                        sqty, sprice = qs[0]
                        take = min(qty, sqty)
                        price_pnl += (sprice - price) * take
                        sqty -= take; qty -= take
                        if sqty <= 1e-12: qs.popleft()
                        else: qs[0] = (sqty, sprice)
                    if qty > 1e-12:
                        ql.append((qty, price))
                else:
                    # cierra long
                    while qty > 1e-12 and ql:
                        lqty, lprice = ql[0]
                        take = min(qty, lqty)
                        price_pnl += (price - lprice) * take
                        lqty -= take; qty -= take
                        if lqty <= 1e-12: ql.popleft()
                        else: ql[0] = (lqty, lprice)
                    if qty > 1e-12:
                        qs.append((qty, price))

            entry_avg = (entry_sum / entry_qty) if entry_qty > 0 else 0.0
            close_avg = (close_sum / close_qty) if close_qty > 0 else entry_avg

            # ---- tamaño del bloque = suma de entradas ejecutadas
            size_block = float(entry_qty)

            # ---- acumulados finales
            fee_total = -abs(fees_sum)         # SIEMPRE negativa
            funding_total = 0.0                 # se inyecta fuera si procede
            realized = price_pnl + funding_total + fee_total

            results.append({
                "exchange": EXCHANGE,
                "symbol": sym,
                "side": side_txt,
                "size": size_block,
                "entry_price": float(entry_avg),
                "close_price": float(close_avg),
                "open_time": int(open_time // 1000),
                "close_time": int(close_time // 1000),
                "pnl": float(price_pnl),
                "realized_pnl": float(realized),
                "funding_total": float(funding_total),
                "fee_total": float(fee_total),
                "notional": float(size_block * entry_avg),
                "initial_margin": None,
                "leverage": None,
                "liquidation_price": None,
            })

            i = j + 1

    return results


def _funding_sum_for(symbol_norm: str, start_ms: int, end_ms: int) -> float:
    """
    Suma funding fees (income) en [start_ms, end_ms] sólo para symbol_norm.
    """
    total = 0.0
    try:
        # XT acepta ventana + (opcional) symbol en formato exchange (btc_usdt).
        # Filtramos por símbolo normalizado para evitar “PERP_” y sufijos.
        lst = fetch_xt_funding_fees(limit=1000, start_ms=start_ms, end_ms=end_ms)
        for it in lst:
            if normalize_symbol(it.get("symbol")) == symbol_norm:
                total += to_float(it.get("income", 0.0))
    except Exception:
        pass
    return float(total)

def save_xt_closed_positions(db_path: str = "portfolio.db",
                             days: int = DEFAULT_DAYS_TRADES,
                             symbol: Optional[str] = None,
                             inject_funding: bool = True) -> Tuple[int, int]:
    """
    Reconstruye y guarda posiciones cerradas a partir de histórico de órdenes.
    Devuelve (guardadas, duplicadas).
    """
    p_closed_sync_start(EXCHANGE)

    end_ms = utc_now_ms()
    start_ms = end_ms - int(days * 86400000)

    # 1) pseudo-fills desde órdenes
    pseudo = _orders_as_pseudo_fills(start_ms, end_ms, symbol=symbol, limit=5000)
    if not pseudo:
        p_closed_sync_none(EXCHANGE)
        return (0, 0)

    # 2) bloques FIFO
    blocks = _fifo_blocks_from_fills(pseudo)
    if not blocks:
        p_closed_sync_none(EXCHANGE)
        return (0, 0)

    # 3) opcional: inyectar funding_total por bloque
    if inject_funding:
        for b in blocks:
            sym = b["symbol"]
            st = b["open_time"] * 1000
            en = b["close_time"] * 1000
            b["funding_total"] = _funding_sum_for(sym, st, en)
            b["realized_pnl"] = b["pnl"] - (-abs(b["fee_total"])) + b["funding_total"]

    # 4) persistencia
    saved = 0
    dup = 0
    for b in blocks:
        try:
            save_closed_position(b)
            saved += 1
        except Exception as e:
            # db_manager.save_closed_position ya imprime y deduplica; aquí sólo contamos
            if "UNIQUE" in str(e).upper() or "duplicate" in str(e).lower():
                dup += 1
            else:
                dup += 0  # contabilizamos como omitida sin clasificar
    if saved:
        p_closed_sync_saved(EXCHANGE, saved, dup)
        p_closed_sync_done(EXCHANGE)
    else:
        p_closed_sync_none(EXCHANGE)
    return (saved, dup)

# =========================================================
#                            DEBUGS
# =========================================================
def debug_raw_xt_balance_futures():
    cli = _get_perp()
    code, success, error = cli.get_account_capital()
    print("=== RAW FUTURES /balance/list ===")
    print(f"code: {code} error: {error}")
    print(json.dumps(success, indent=2, ensure_ascii=False))

def debug_raw_xt_positions():
    cli = _get_perp()
    path = "/future/user" + "/v1/position/list"
    params = {}
    header = cli._create_sign(XT_API_KEY, XT_API_SECRET, path=path,
                              bodymod="application/x-www-form-urlencoded", params=params)
    header["Content-Type"] = "application/x-www-form-urlencoded"
    url = cli.host + path
    code, success, error = cli._fetch(method="GET", url=url, headers=header, params=params, timeout=cli.timeout)
    print("=== RAW FUTURES /position/list ===")
    print(f"code: {code} error: {error}")
    print(json.dumps(success, indent=2, ensure_ascii=False))

def debug_dump_xt_opens():
    print("=== XT OPEN POSITIONS (normalized) ===")
    try:
        data = fetch_xt_open_positions()
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"total: {len(data)}")
    except Exception as e:
        print(f"❌ fetch_xt_open_positions error: {e}")

def debug_dump_xt_funding(limit: int = 30, days: int = 14, symbol: Optional[str] = None):
    print("=== XT FUNDING FEES (normalized) ===")
    end_ms = utc_now_ms()
    start_ms = end_ms - int(days * 86400000)
    try:
        data = fetch_xt_funding_fees(limit=limit, start_ms=start_ms, end_ms=end_ms, symbol=symbol)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        by = defaultdict(float)
        for it in data:
            by[it["symbol"]] += to_float(it["income"])
        print("\n--- summary ---")
        for s, v in by.items():
            print(f"{s}: {v:.8f}")
        print(f"total items: {len(data)}")
    except Exception as e:
        print(f"❌ fetch_xt_funding_fees error: {e}")

def debug_raw_xt_orders(days: int = DEFAULT_DAYS_TRADES, symbol: Optional[str] = None, limit: int = 100):
    """
    Dump RAW de /future/trade/v1/order/list-history (sirve para verificar qué campos trae tu cuenta).
    """
    end_ms = utc_now_ms()
    start_ms = end_ms - int(days * 86400000)
    cli = _get_perp()
    code, success, error = cli.get_history_order(
        symbol=symbol, direction=None, oid=None, limit=min(1000, limit),
        start_time=int(start_ms), end_time=int(end_ms)
    )
    print("=== RAW FUTURES /order/list-history ===")
    print(f"code: {code} error: {error}")
    print(json.dumps(success, indent=2, ensure_ascii=False))

def debug_preview_xt_closed(days: int = DEFAULT_DAYS_TRADES,
                            symbol: Optional[str] = None,
                            limit: int = 2000,
                            with_funding: bool = False) -> None:
    """
    Reconstrucción *preview* de posiciones cerradas usando pseudo-fills desde órdenes.
    No guarda en DB.
    """
    end_ms = utc_now_ms()
    start_ms = end_ms - int(days * 86400000)

    try:
        pseudo = _orders_as_pseudo_fills(start_ms, end_ms, symbol=symbol, limit=limit)
    except Exception as e:
        print("=== XT CLOSED PREVIEW (FIFO) ===")
        print(f"❌ _orders_as_pseudo_fills error: {e}")
        return

    print("=== XT CLOSED PREVIEW (FIFO) ===")
    if not pseudo:
        print("No se obtuvieron órdenes ejecutadas en la ventana dada.")
        return

    blocks = _fifo_blocks_from_fills(pseudo)
    if not blocks:
        print("No se detectaron bloques cerrados (neto=0).")
        return

    if with_funding:
        for b in blocks:
            sym = b["symbol"]
            st = b["open_time"] * 1000
            en = b["close_time"] * 1000
            b["funding_total"] = _funding_sum_for(sym, st, en)
            b["realized_pnl"] = b["pnl"] - (-abs(b["fee_total"])) + b["funding_total"]

    print(json.dumps(blocks, indent=2, ensure_ascii=False))
    print(f"\n--- summary ---\nclosed_blocks={len(blocks)}")

# =========================================================
#                            CLI
# =========================================================
if __name__ == "__main__":
    import argparse, traceback

    ap = argparse.ArgumentParser("XT adapter (pyxt) — Spot + Futures")
    ap.add_argument("--balance", action="store_true", help="Saldo combinado normalizado (/api/balances)")
    ap.add_argument("--positions", action="store_true", help="Posiciones abiertas normalizadas")
    ap.add_argument("--funding", type=int, default=0, help="N funding fees normalizados")
    ap.add_argument("--raw-balance", action="store_true", help="RAW FUTURES balance/list")
    ap.add_argument("--raw-positions", action="store_true", help="RAW FUTURES position/list")
    ap.add_argument("--raw-funding", action="store_true", help="RAW FUTURES funding-rate-list (14d)")
    ap.add_argument("--raw-orders", action="store_true", help="RAW FUTURES order/list-history (14d)")
    ap.add_argument("--closed-preview", action="store_true", help="Reconstrucción FIFO (preview, sin guardar)")
    ap.add_argument("--save-closed", action="store_true", help="Reconstruye y GUARDA cerradas en SQLite")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS_TRADES)
    ap.add_argument("--symbol", type=str, default=None, help="Símbolo formato exchange, p.ej. btc_usdt")

    args = ap.parse_args()

    try:
        if args.raw_balance:
            debug_raw_xt_balance_futures()
        if args.balance:
            print(json.dumps(fetch_xt_all_balances() or {}, indent=2, ensure_ascii=False))
        if args.raw_positions:
            debug_raw_xt_positions()
        if args.positions:
            debug_dump_xt_opens()
        if args.raw_funding:
            debug_dump_xt_funding(limit=50, days=args.days, symbol=args.symbol)
        if args.raw_orders:
            debug_raw_xt_orders(days=args.days, symbol=args.symbol, limit=200)
        if args.closed_preview:
            debug_preview_xt_closed(days=args.days, symbol=args.symbol, with_funding=True)
        if args.save_closed:
            saved, dup = save_xt_closed_positions(days=args.days, symbol=args.symbol, inject_funding=True)
            print(f"saved={saved} dup={dup}")
        # caso por defecto útil si no se pasa nada
        if not any([args.raw_balance, args.balance, args.raw_positions, args.positions,
                    args.raw_funding, args.raw_orders, args.closed_preview, args.save_closed]):
            # autotest rápido
            print("• Futures RAW balance:"); debug_raw_xt_balance_futures()
            print("• Futures RAW positions:"); debug_raw_xt_positions()
            print("• Normalizado /api/balances:"); print(json.dumps(fetch_xt_all_balances() or {}, indent=2))
    except Exception:
        print("❌ Uncaught error:\n" + traceback.format_exc())


