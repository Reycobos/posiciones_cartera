# adapters/xt.py
# XT adapter (Futures + Spot) usando el SDK pyxt (perp.py/spot.py locales si no hay pip).
from __future__ import annotations

import os
import json
import time
import argparse
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

DEFAULT_DAYS_TRADES = int(os.getenv("XT_DAYS_TRADES", "30"))  # 30 días por defecto

# ============ Utils ============
def to_float(x) -> float:
    try: return float(x)
    except Exception: return 0.0

def to_int(x) -> int:
    try: return int(x)
    except Exception: return 0

def utc_now_ms() -> int:
    return int(time.time() * 1000)

def _unwrap_result(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "result" in obj: return obj["result"]
        if "data" in obj: return obj["data"]
        if "items" in obj: return obj["items"]
        if "list" in obj: return obj["list"]
    return obj

def _fee_estimate(order_type: Optional[str], taker_maker: Optional[str], price: float, qty: float) -> float:
    """
    Si no viene fee, estima:
      MARKET (o TAKER) -> 0.0588%
      LIMIT  (o MAKER) -> 0.038%
    Devuelve SIEMPRE negativa (costo).
    """
    t = (order_type or "").upper()
    tm = (taker_maker or "").upper()
    if "MARKET" in t or tm == "TAKER":
        rate = 0.000588
    elif "LIMIT" in t or tm == "MAKER":
        rate = 0.000380
    else:
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
    try:
        cli_s = _get_spot()
        res_s = cli_s.balances(currencies=None)  # GET /v4/balances
        assets = (res_s or {}).get("assets") if isinstance(res_s, dict) else res_s
        if isinstance(assets, list):
            for a in assets:
                if not isinstance(a, dict): continue
                total = to_float(a.get("totalAmount") or 0.0)
                spot_equity += total
    except Exception:
        pass  # si no hay spot, continuamos con futuros

    # -------- Futuros --------
    cli_f = _get_perp()
    code, payload, error = cli_f.get_account_capital()   # GET /future/user/v1/balance/list
    if error or code != 200 or payload is None:
        raise RuntimeError(f"XT futures balance error: {error or code}")
    res_f = _unwrap_result(payload)

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
#                    OPEN / ACTIVE POSITIONS
#   GET /future/user/v1/position/list
# =========================================================
def fetch_xt_open_positions() -> List[Dict[str, Any]]:
    p_open_fetching(EXCHANGE)
    cli = _get_perp()
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
        arr = arr.get("items") or arr.get("list") or []

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
        start_ms = end_ms - 30 * 24 * 60 * 60 * 1000  # 30 días

    out: List[Dict[str, Any]] = []
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
            direction = "NEXT"
        else:
            break

    p_funding_count(EXCHANGE, len(out))
    return out

# =========================================================
#         TRANSACTIONS (TRADES) - /future/trade/v1/order/trade-list
# =========================================================
def fetch_xt_transactions(limit: int = 2000,
                          page: int = 1,
                          size: int = 100,
                          orderId: Optional[str] = None,
                          symbol: Optional[str] = None,
                          start_ms: Optional[int] = None,
                          end_ms: Optional[int] = None,
                          days: int = DEFAULT_DAYS_TRADES) -> List[Dict[str, Any]]:
    """
    Historial de trades (fills) de XT Futures.
    - Pagina / tamaño: GET /future/trade/v1/order/trade-list
    - Devuelve lista homogenizada con side, qty, price, fee, fee_coin, taker_or_maker, positionSide...
    """
    cli = _get_perp()
    if end_ms is None:
        end_ms = utc_now_ms()
    if start_ms is None:
        start_ms = end_ms - days * 24 * 60 * 60 * 1000

    out: List[Dict[str, Any]] = []
    current_page = page
    path = "/future/trade" + "/v1/order/trade-list"

    while len(out) < limit:
        page_size = min(size, max(1, limit - len(out)))
        params: Dict[str, Any] = {
            "page": current_page,
            "size": page_size,
            "startTime": int(start_ms),
            "endTime": int(end_ms)
        }
        if symbol:
            params["symbol"] = symbol
        if orderId:
            params["orderId"] = orderId

        header = cli._create_sign(XT_API_KEY, XT_API_SECRET,
                                  path=path, bodymod="application/x-www-form-urlencoded",
                                  params=params)
        header["Content-Type"] = "application/x-www-form-urlencoded"
        url = cli.host + path
        code, success, error = cli._fetch(method="GET", url=url, headers=header, params=params, timeout=cli.timeout)
        if error or code != 200 or success is None:
            # rate limit u otros → salimos
            break

        res = _unwrap_result(success)
        items = []
        if isinstance(res, dict):
            items = res.get("items") or res.get("list") or []
        if not isinstance(items, list):
            items = []

        for trade in items:
            sym_raw = str(trade.get("symbol") or "")
            base = normalize_symbol(sym_raw)
        # quantity viene en CONTRATOS → convertir a BASE usando contractSize
            cs = to_float(trade.get("contractSize") or 1.0)
            qty_contracts = to_float(trade.get("quantity") or trade.get("executedQty") or 0.0)
            qty = abs(qty_contracts) * (cs if cs > 0 else 1.0)  # <-- SIZE EN BASE
            
            price = to_float(trade.get("price") or 0.0)
            fee_raw = trade.get("fee")
            fee = -abs(to_float(fee_raw)) if fee_raw is not None else None  # costo en negativo o None
            
            fee_coin = (trade.get("feeCoin") or "USDT").upper()
            tm = (trade.get("takerMaker") or "").upper()  # "TAKER"/"MAKER"
            otype = (trade.get("orderType") or "").upper()
            
            # Si fee no viene, estimamos por tus reglas (sobre notional = price * qty_base)
            if fee is None:
                fee = _fee_estimate(otype, tm, price, qty)
    
            order_side = (trade.get("orderSide") or "").upper()  # "BUY"/"SELL"
            pos_side   = (trade.get("positionSide") or "").upper()  # "LONG"/"SHORT"
            ts = int(trade.get("timestamp") or 0)
            ts = ts if ts > 10**12 else ts * 1000

            out.append({
                "exchange": EXCHANGE,
                "symbol": base,
                "id": str(trade.get("execId") or ""),
                "order": str(trade.get("orderId") or ""),
                "side": "BUY" if order_side == "BUY" else "SELL",
                "positionSide": pos_side,
                "price": price,
                "qty": abs(qty),                     # SIZE EN BASE (contratos * contractSize)
                "qty_contracts": abs(qty_contracts), # referencia/debug
                "contract_size": cs,                 # referencia/debug
                "quoteQty": abs(qty) * price,
                "fee": float(fee),                   # negativa (real o estimada)
                "fee_coin": fee_coin,
                "taker_or_maker": tm.lower(),
                "timestamp": ts,
                "datetime": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ts/1000)),
                "type": "TRADE",
                "raw_data": trade
            })

        if len(items) < page_size:
            break
        current_page += 1
        time.sleep(0.05)

    return out

# =========================================================
#      CLOSED POSITIONS DESDE TRADES (FIFO + FUNDING)
# =========================================================
def _group_funding_by_symbol(funding_items: List[Dict[str, Any]]) -> Dict[str, List[Tuple[int, float]]]:
    """ funding_items normalizados: [{"symbol","income"(+-),"timestamp"(ms),...}] → {SYM: [(ts, income), ...]} """
    m: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for it in funding_items or []:
        s = normalize_symbol(it.get("symbol", ""))
        if not s:
            continue
        ts = int(it.get("timestamp") or 0)
        ts = ts if ts > 10**12 else ts * 1000
        inc = float(it.get("income") or 0.0)
        m[s].append((ts, inc))
    for s in m:
        m[s].sort(key=lambda t: t[0])
    return m

def _sum_funding_in_window(series: List[Tuple[int, float]], start_ms: int, end_ms: int) -> float:
    """Suma funding incomes en [start_ms, end_ms]."""
    if not series:
        return 0.0
    total = 0.0
    for ts, inc in series:
        if ts < start_ms:
            continue
        if ts > end_ms:
            break
        total += float(inc)
    return float(total)

def _tx_to_fill(tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convierte un trade → fill estándar para el motor FIFO de bloques.
    - side: 'BUY'/'SELL' (lado real de la orden)
    - inc: True si AUMENTA exposición absoluta (entrada), False si DISMINUYE (salida), calculado con positionSide.
      * LONG:  BUY aumenta, SELL reduce
      * SHORT: SELL aumenta, BUY reduce
    """
    try:
        price = float(tx.get("price"))
        qty   = abs(float(tx.get("qty") or tx.get("quantity") or 0.0))
        if qty <= 0 or price <= 0:
            return None
        side  = str(tx.get("side") or "BUY").upper()  # BUY/SELL real
        ps    = str(tx.get("positionSide") or "").upper()  # LONG/SHORT
        ts    = int(tx.get("timestamp") or 0)
        ts    = ts if ts > 10**12 else ts * 1000
        sym   = tx.get("symbol") or ""
        fee   = float(tx.get("fee") or 0.0)  # ya negativa en fetch

        # ¿Este trade aumenta exposición? (entrada)
        if ps == "LONG":
            inc = (side == "BUY")
        elif ps == "SHORT":
            inc = (side == "SELL")
        else:
            # Si no viene positionSide, deducimos por convención (más común: BUY incrementa long)
            inc = (side == "BUY")

        return {
            "price": price,
            "qty": qty,
            "side": side,        # BUY/SELL
            "inc": bool(inc),    # entrada/salida
            "fee": fee,          # negativa
            "symbol": sym,
            "timestamp": ts,
        }
    except Exception:
        return None

def _fifo_blocks_from_fills(fills: List[Dict[str, Any]],
                            funding_map: Optional[Dict[str, List[Tuple[int, float]]]] = None) -> List[Dict[str, Any]]:
    """
    Reconstruye BLOQUES CERRADOS por símbolo con FIFO real, agrupando por ventanas donde el net de 'inc/dec' vuelve a 0.
    Reglas:
      - side del bloque: según primera ENTRADA (inc=True): BUY→long, SELL→short
      - entry_avg: media ponderada de ENTRADAS (inc=True) del bloque
      - close_avg: media ponderada de SALIDAS (inc=False) del bloque
      - size del bloque = suma de qty de ENTRADAS
      - fee_total = suma de fees de todos los trades del bloque (NEGATIVA)
      - funding_total = suma de incomes en [open_time, close_time] por símbolo
      - pnl (precio) = FIFO real
      - realized_pnl = pnl + funding_total + fee_total
    """
    by_sym: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in fills:
        s = normalize_symbol(f.get("symbol") or "")
        if not s:
            continue
        g = dict(f)
        g["symbol"] = s
        by_sym[s].append(g)

    results: List[Dict[str, Any]] = []

    for sym, arr in by_sym.items():
        arr.sort(key=lambda x: x["timestamp"])
        n = len(arr)
        i = 0
        while i < n:
            # buscar primera ENTRADA (inc=True) para arrancar bloque
            while i < n and not arr[i]["inc"]:
                i += 1
            if i >= n:
                break

            first = arr[i]
            first_side = first["side"]  # BUY/SELL de la PRIMERA ENTRADA
            block_side_txt = "long" if first_side == "BUY" else "short"

            net = 0.0       # neto de exposición (entradas +, salidas -)
            fee_sum = 0.0
            open_time_ms = arr[i]["timestamp"]

            entry_sum = close_sum = 0.0
            entry_qty = close_qty = 0.0

            j = i
            while j < n:
                f = arr[j]
                qty = float(f["qty"])
                prc = float(f["price"])
                fee_sum += float(f.get("fee") or 0.0)

                if f["inc"]:
                    net += qty
                    entry_sum += prc * qty
                    entry_qty += qty
                else:
                    net -= qty
                    close_sum += prc * qty
                    close_qty += qty

                # bloque cerrado: net vuelve a 0 y hay entradas/salidas
                if abs(net) <= 1e-12 and entry_qty > 0 and close_qty > 0:
                    break

                j += 1

            # si no cerró el bloque, avanzar y volver a buscar
            if j >= n or abs(net) > 1e-12 or entry_qty <= 0 or close_qty <= 0:
                i = j + 1
                continue

            close_time_ms = arr[j]["timestamp"]

            # ---- calcular PnL FIFO real (match por BUY/SELL independientemente de inc)
            price_pnl = 0.0
            ql: deque[Tuple[float, float]] = deque()  # long lots (BUY entradas pendientes)
            qs: deque[Tuple[float, float]] = deque()  # short lots (SELL entradas pendientes)

            for k in range(i, j + 1):
                ff = arr[k]
                qty = float(ff["qty"])
                prc = float(ff["price"])
                sde = ff["side"]

                if sde == "BUY":
                    # cierra short si hay; si no, abre long
                    while qty > 1e-12 and qs:
                        sqty, sprice = qs[0]
                        take = min(qty, sqty)
                        price_pnl += (sprice - prc) * take  # short: sell@entry - buy@exit
                        sqty -= take; qty -= take
                        if sqty <= 1e-12: qs.popleft()
                        else: qs[0] = (sqty, sprice)
                    if qty > 1e-12:
                        ql.append((qty, prc))
                else:
                    # SELL cierra long si hay; si no, abre short
                    while qty > 1e-12 and ql:
                        lqty, lprice = ql[0]
                        take = min(qty, lqty)
                        price_pnl += (prc - lprice) * take  # long: sell@exit - buy@entry
                        lqty -= take; qty -= take
                        if lqty <= 1e-12: ql.popleft()
                        else: ql[0] = (lqty, lprice)
                    if qty > 1e-12:
                        qs.append((qty, prc))

            entry_avg = (entry_sum / entry_qty) if entry_qty > 0 else 0.0
            close_avg = (close_sum / close_qty) if close_qty > 0 else entry_avg
            size_block = float(entry_qty)

            # fees: SIEMPRE NEGATIVO (costo)
            fee_total = -abs(fee_sum)

            # funding en ventana
            funding_total = 0.0
            if funding_map and sym in funding_map:
                funding_total = _sum_funding_in_window(funding_map[sym], open_time_ms, close_time_ms)

            realized = price_pnl + funding_total + fee_total

            results.append({
                "exchange": EXCHANGE,
                "symbol": sym,
                "side": block_side_txt,
                "size": size_block,
                "entry_price": float(entry_avg),
                "close_price": float(close_avg),
                "open_time": int(open_time_ms // 1000),
                "close_time": int(close_time_ms // 1000),
                "pnl": float(price_pnl),                 # PnL puro de precio
                "realized_pnl": float(realized),         # precio + funding + fees
                "funding_total": float(funding_total),
                "fee_total": float(fee_total),           # negativa
                "notional": float(size_block * entry_avg),
                "initial_margin": None,
                "leverage": None,
                "liquidation_price": None,
            })

            i = j + 1

    return results

def reconstruct_xt_closed_from_transactions(days: int = DEFAULT_DAYS_TRADES,
                                            symbol: Optional[str] = None,
                                            limit: int = 5000,
                                            inject_funding: bool = True) -> List[Dict[str, Any]]:
    """
    Pipeline:
      1) fetch_xt_transactions(...): trades crudos
      2) map → fills (price, qty, side, inc, fee, symbol, ts)
      3) funding opcional de fetch_xt_funding_fees(...)
      4) _fifo_blocks_from_fills → lista de posiciones cerradas normalizadas
    """
    end_ms = utc_now_ms()
    start_ms = end_ms - days * 24 * 60 * 60 * 1000

    # 1) TRADES
    txs = fetch_xt_transactions(limit=limit, symbol=None if symbol in (None, "", "ALL") else symbol,
                                start_ms=start_ms, end_ms=end_ms, days=days)
    if not txs:
        print("❌ No hay transacciones en el rango.")
        return []

    # 2) Mapeo a fills
    fills: List[Dict[str, Any]] = []
    for tx in txs:
        f = _tx_to_fill(tx)
        if f:
            fills.append(f)

    if not fills:
        print("❌ No se generaron fills a partir de transacciones.")
        return []

    fills.sort(key=lambda x: x["timestamp"])

    # 3) Funding (opcional)
    funding_map = None
    if inject_funding:
        fund_items = fetch_xt_funding_fees(limit=2000, start_ms=start_ms, end_ms=end_ms, symbol=None)
        funding_map = _group_funding_by_symbol(fund_items)

    # 4) Reconstrucción FIFO
    blocks = _fifo_blocks_from_fills(fills, funding_map=funding_map)
    return blocks

# --- dedupe: evitar duplicados en closed_positions ---
def _exists_closed_in_db(row: dict, db_path: str = "portfolio.db") -> bool:
    """
    Considera duplicado si ya existe una fila con:
      exchange, symbol, close_time y (size, entry_price, close_price) ~=
    con tolerancia numérica pequeña para floats.
    """
    try:
        import sqlite3
        eps = 1e-6
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT 1
              FROM closed_positions
             WHERE exchange = ?
               AND symbol   = ?
               AND close_time = ?
               AND ABS(size - ?) < ?
               AND ABS(entry_price - ?) < ?
               AND ABS(close_price - ?) < ?
             LIMIT 1
            """,
            (
                row.get("exchange"),
                row.get("symbol"),
                int(row.get("close_time") or 0),
                float(row.get("size") or 0.0), eps,
                float(row.get("entry_price") or 0.0), eps,
                float(row.get("close_price") or 0.0), eps,
            ),
        )
        found = cur.fetchone() is not None
        conn.close()
        return found
    except Exception:
        # En caso de error en la consulta, no bloquear guardado.
        return False

def save_xt_closed_positions(db_path: str = "portfolio.db",
                             days: int = DEFAULT_DAYS_TRADES,
                             symbol: Optional[str] = None,
                             limit: int = 5000,
                             inject_funding: bool = True) -> int:
    """
    Reconstruye y guarda en SQLite usando db_manager.save_closed_position.
    Devuelve el número de bloques guardados.
    """
    p_closed_sync_start(EXCHANGE)
    blocks = reconstruct_xt_closed_from_transactions(days=days, symbol=symbol, limit=limit, inject_funding=inject_funding)
    if not blocks:
        p_closed_sync_none(EXCHANGE)
        return 0

    saved = 0
    dup = 0
    for b in blocks:
        # dedupe proactivo: evitamos insertar si ya existe una fila equivalente
        if _exists_closed_in_db(b, db_path):
            dup += 1
            continue
        try:
            save_closed_position(b)
            saved += 1
        except Exception as e:
            # fallback por si el DB manager llega a tener un índice único
            if "UNIQUE" in str(e).upper() or "duplicate" in str(e).lower():
                dup += 1
            else:
                # puedes loguear si quieres, pero no contamos como guardada
                pass

# ==========================
#        DEBUGS
# ==========================
def debug_dump_xt_transactions(days: int = DEFAULT_DAYS_TRADES, limit: int = 200, symbol: Optional[str] = None):
    end_ms = utc_now_ms()
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    txs = fetch_xt_transactions(limit=limit, symbol=symbol, start_ms=start_ms, end_ms=end_ms, days=days)

    if not txs:
        print("❌ No se encontraron transacciones.")
        return

    # Estadísticas
    stats = defaultdict(int)
    for tx in txs:
        stats[tx.get("symbol", "UNKNOWN")] += 1

    print(f"📊 ESTADÍSTICAS - {len(txs)} transacciones encontradas:")
    for s, c in stats.items():
        print(f"   {s}: {c} transacciones")

    print("\n📋 DETALLES DE TRANSACCIONES:")
    print("-" * 132)
    print(f"{'Fecha':<20} {'Symbol':<8} {'Side':<6} {'Cantidad':>10} {'Precio':>12} {'Total':>14} {'Fee':>12} {'T/M':>6} {'Order ID':>14}")
    print("-" * 132)

    txs_sorted = sorted(txs, key=lambda x: x.get("timestamp", 0), reverse=True)
    for tx in txs_sorted[:50]:
        dt = tx.get("datetime", "N/A")
        sym = tx.get("symbol", "N/A")
        side = tx.get("side", "N/A")
        qty = float(tx.get("qty", 0))
        price = float(tx.get("price", 0))
        total = qty * price
        fee = float(tx.get("fee", 0))
        tm = tx.get("taker_or_maker", "N/A")
        oid = str(tx.get("order", ""))[-12:]
        print(f"{dt:<20} {sym:<8} {side:<6} {qty:>10.6f} {price:>12.6f} {total:>14.6f} {fee:>12.6f} {tm:>6} {oid:>14}")

    print("\n🔍 EJEMPLO - Primera transacción (raw data):")
    print(json.dumps(txs_sorted[0].get("raw_data", {}), indent=2, ensure_ascii=False))

def debug_preview_xt_closed_from_transactions(days: int = DEFAULT_DAYS_TRADES,
                                              symbol: Optional[str] = None,
                                              limit: int = 5000,
                                              inject_funding: bool = True):
    blocks = reconstruct_xt_closed_from_transactions(days=days, symbol=symbol, limit=limit, inject_funding=inject_funding)
    if not blocks:
        print("❌ No se reconstruyeron posiciones cerradas.")
        return

    # resumen por símbolo
    agg = defaultdict(lambda: {"n": 0, "realized": 0.0, "fees": 0.0, "funding": 0.0})
    for b in blocks:
        s = b["symbol"]
        agg[s]["n"] += 1
        agg[s]["realized"] += float(b["realized_pnl"])
        agg[s]["fees"] += float(b["fee_total"])
        agg[s]["funding"] += float(b["funding_total"])

    print("=== XT CLOSED PREVIEW (FIFO desde trades) ===")
    for s, v in agg.items():
        print(f"{s}: blocks={v['n']} realized={v['realized']:.6f} fees={v['fees']:.6f} funding={v['funding']:.6f}")

    print("\n— Muestra (hasta 10 bloques):")
    for b in blocks[:10]:
        print(json.dumps(b, indent=2, ensure_ascii=False))

def debug_save_xt_closed(db_path: str = "portfolio.db",
                         days: int = DEFAULT_DAYS_TRADES,
                         symbol: Optional[str] = None,
                         limit: int = 5000,
                         inject_funding: bool = True):
    print(f"⏳ Guardando posiciones cerradas reconstruidas (days={days}, symbol={symbol or 'ALL'})…")
    n = save_xt_closed_positions(db_path=db_path, days=days, symbol=symbol, limit=limit, inject_funding=inject_funding)
    print(f"✅ Guardadas: {n}")

# ==========================
#         __all__
# ==========================
__all__ = [
    "fetch_xt_open_positions",
    "fetch_xt_funding_fees",
    "fetch_xt_all_balances",
    "save_xt_closed_positions",
    "debug_dump_xt_transactions",
    "debug_preview_xt_closed_from_transactions",
    "debug_save_xt_closed",
]

# ==========================
#           CLI
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XT adapter – balances, funding, trades → closed positions FIFO")
    parser.add_argument("--balances", action="store_true", help="Imprime balances normalizados")
    parser.add_argument("--opens", action="store_true", help="Imprime posiciones abiertas normalizadas")
    parser.add_argument("--funding", action="store_true", help="Imprime funding fees normalizados")
    parser.add_argument("--transactions", action="store_true", help="Debug: lista de transacciones (fills)")
    parser.add_argument("--closed-preview", action="store_true", help="Reconstruye y muestra posiciones cerradas (sin guardar)")
    parser.add_argument("--save-closed", action="store_true", help="Reconstruye y guarda posiciones cerradas en DB")
    parser.add_argument("--symbol", type=str, default=None, help="Símbolo (ej: BTCUSDT) o None para todos")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS_TRADES, help="Días hacia atrás para consultar")
    parser.add_argument("--limit", type=int, default=5000, help="Límite de items (funding/trades)")
    parser.add_argument("--db", type=str, default="portfolio.db", help="Ruta DB SQLite")

    args = parser.parse_args()

    if args.balances:
        print(json.dumps(fetch_xt_all_balances(db_path=args.db), indent=2))
    if args.opens:
        print(json.dumps(fetch_xt_open_positions(), indent=2))
    if args.funding:
        end_ms = utc_now_ms()
        start_ms = end_ms - args.days * 24 * 3600 * 1000
        ff = fetch_xt_funding_fees(limit=args.limit, start_ms=start_ms, end_ms=end_ms, symbol=args.symbol)
        print(json.dumps(ff, indent=2))
    if args.transactions:
        debug_dump_xt_transactions(days=args.days, limit=min(args.limit, 1000), symbol=args.symbol)
    if args.closed_preview:
        debug_preview_xt_closed_from_transactions(days=args.days, symbol=args.symbol, limit=args.limit, inject_funding=True)
    if args.save_closed:
        debug_save_xt_closed(db_path=args.db, days=args.days, symbol=args.symbol, limit=args.limit, inject_funding=True)

    # Si no se pasó ningún flag, muestra ayuda breve
    if not any([args.balances, args.opens, args.funding, args.transactions, args.closed_preview, args.save_closed]):
        print("Uso rápido:")
        print("  python adapters/xt.py --save-closed --days 30            # reconstruye y guarda en portfolio.db")
        print("  python adapters/xt.py --closed-preview --days 30         # solo vista previa (no guarda)")
        print("  python adapters/xt.py --transactions --days 30           # debug de trades")
        print("  python adapters/xt.py --funding --days 14                # funding fees normalizados")
        print("  python adapters/xt.py --balances                         # balances combinados (spot+futuros)")
