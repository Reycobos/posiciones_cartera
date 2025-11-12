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
#                    TRANSACTION DETAILS (TRADES) - MODIFICADO
#   GET /future/trade/v1/order/trade-list
# =========================================================
def fetch_xt_transactions(limit: int = 50,
                         page: int = 1,
                         size: int = 100,  # Aumentado para mayor eficiencia
                         orderId: Optional[int] = None,
                         symbol: Optional[str] = None,  # None = todos los símbolos
                         start_ms: Optional[int] = None,
                         end_ms: Optional[int] = None,
                         days: int = 30) -> List[Dict[str, Any]]:  # 30 días por defecto
    """
    Obtiene el historial de transacciones (trades) de XT Futures
    - Si symbol es None, busca en TODOS los símbolos
    - Por defecto busca en los últimos 30 días
    """
    try:
        # Inicializar cliente Perp
        cli = Perp(host=XT_FAPI_HOST, access_key=XT_API_KEY, secret_key=XT_API_SECRET)
        
        # Configurar tiempos - 30 días por defecto
        if end_ms is None:
            end_ms = utc_now_ms()
        if start_ms is None:
            start_ms = end_ms - days * 24 * 60 * 60 * 1000

        print(f"DEBUG - Buscando transacciones de los últimos {days} días")
        print(f"DEBUG - Rango temporal: {start_ms} -> {end_ms}")
        print(f"DEBUG - Símbolo: {'TODOS' if symbol is None else symbol}")

        out: List[Dict[str, Any]] = []
        current_page = page
        path = "/future/trade/v1/order/trade-list"
        total_fetched = 0

        while len(out) < limit and total_fetched < limit * 3:  # Límite de seguridad
            # Preparar parámetros
            page_size = min(size, limit - len(out))
            params: Dict[str, Any] = {
                "page": current_page,
                "size": page_size,
                "startTime": int(start_ms),
                "endTime": int(end_ms)
            }
            
            # Solo añadir symbol si se especifica (None = todos)
            if symbol:
                params["symbol"] = symbol
            if orderId:
                params["orderId"] = orderId

            print(f"DEBUG - Página {current_page}, tamaño {page_size}, símbolo: {'TODOS' if not symbol else symbol}")

            # Crear firma y headers
            header = cli._create_sign(XT_API_KEY, XT_API_SECRET,
                                    path=path, 
                                    bodymod="application/x-www-form-urlencoded",
                                    params=params)
            header["Content-Type"] = "application/x-www-form-urlencoded"
            
            url = cli.host + path
            
            # Realizar solicitud
            code, success, error = cli._fetch(method="GET", url=url, headers=header, params=params, timeout=cli.timeout)
            
            if error or code != 200 or success is None:
                print(f"ERROR - XT transactions error: {error or code}")
                if code == 429:  # Rate limit
                    time.sleep(1)
                    continue
                break

            # Procesar respuesta
            res = _unwrap_result(success)
            items = []
            if isinstance(res, dict):
                items = res.get("items") or res.get("list") or []
            if not isinstance(items, list):
                items = []

            print(f"DEBUG - Encontradas {len(items)} transacciones en página {current_page}")

            # Procesar cada transacción
            for trade in items:
                try:
                    sym_raw = str(trade.get("symbol") or "")
                    base = normalize_symbol(sym_raw)
                    
                    # Información del trade
                    exec_id = str(trade.get("execId") or "")
                    order_id = str(trade.get("orderId") or "")
                    price = to_float(trade.get("price") or 0.0)
                    quantity = to_float(trade.get("quantity") or 0.0)
                    fee = to_float(trade.get("fee") or 0.0)
                    fee_coin = (trade.get("feeCoin") or "USDT").upper()
                    taker_maker = trade.get("takerMaker", "").lower()
                    ts = int(trade.get("timestamp") or 0)
                    
                    # Ajustar timestamp si está en segundos
                    ts = ts if ts > 10**12 else ts * 1000
                    
                    # Determinar side basado en cantidad/fee (ajustar según lógica real)
                    # Esta es una aproximación - puede necesitar ajustes
                    total_cost = price * quantity
                    
                    # Para determinar compra/venta, normalmente se mira el contexto de la orden
                    # Como fallback, usamos una lógica simple
                    side = "BUY" if quantity > 0 else "SELL"
                    
                    transaction = {
                        "exchange": EXCHANGE,
                        "symbol": base,
                        "id": exec_id,
                        "order": order_id,
                        "side": side,
                        "price": price,
                        "qty": abs(quantity),  # Valor absoluto
                        "quoteQty": abs(total_cost),
                        "realized_pnl": 0.0,
                        "fee": abs(fee),
                        "fee_coin": fee_coin,
                        "timestamp": ts,
                        "datetime": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ts/1000)),
                        "type": "TRADE",
                        "taker_or_maker": taker_maker,
                        "raw_data": trade
                    }
                    
                    out.append(transaction)
                    total_fetched += 1
                    
                    if len(out) % 10 == 0:
                        print(f"DEBUG - Procesadas {len(out)} transacciones...")
                    
                    # Salir si alcanzamos el límite
                    if len(out) >= limit:
                        break
                        
                except Exception as e:
                    print(f"ERROR procesando trade {trade}: {e}")
                    continue

            # Verificar si hay más páginas
            if len(items) == 0 or len(items) < size or len(out) >= limit:
                print(f"DEBUG - No hay más páginas o se alcanzó el límite")
                break
                
            current_page += 1
            
            # Pequeña pausa para evitar rate limits
            time.sleep(0.1)

        print(f"DEBUG - Total transacciones obtenidas: {len(out)}")
        return out

    except Exception as e:
        print(f"ERROR in fetch_xt_transactions: {e}")
        import traceback
        traceback.print_exc()
        return []
# =========================================================
#      CLOSED POSITIONS DESDE TRADES (FIFO + FUNDING)
#   Usa fetch_xt_transactions() ya existente (XT Futures)
#   - size = suma de qty de ENTRADAS del bloque
#   - fee_total = suma de fees por trade (siempre NEGATIVA)
#   - pnl (price) = FIFO real
#   - realized_pnl = pnl + funding_total + fee_total
# =========================================================

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

def _tx_to_fill(tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convierte un trade de fetch_xt_transactions() a un fill estándar:
      { price, qty, side:'BUY'|'SELL', fee(neg), symbol, timestamp(ms) }
    - Usa 'side' del trade (BUY/SELL). Si faltara, BUY por defecto.
    - fee: si tx['fee'] existe, se asume COSTO positivo en USDT → guardamos NEGATIVO.
           si no existe, fee = 0 (o podrías aplicar una tabla si quieres fallback).
    """
    try:
        price = float(tx.get("price"))
        qty   = abs(float(tx.get("qty") or tx.get("quantity") or 0.0))
        if qty <= 0 or price <= 0:
            return None
        side  = str(tx.get("side") or tx.get("orderSide") or "BUY").upper()
        ts    = int(tx.get("timestamp") or 0)
        ts    = ts if ts > 10**12 else ts * 1000
        sym   = tx.get("symbol") or ""
        fee   = float(tx.get("fee") or 0.0)
        # La API devuelve fee como COSTO positivo → acumulamos NEGATIVO:
        fee   = -abs(fee)
        return {
            "price": price,
            "qty": qty,
            "side": "BUY" if side in ("BUY", "LONG", "BID") else "SELL",
            "fee": fee,
            "symbol": sym,
            "timestamp": ts,
        }
    except Exception:
        return None


def _group_funding_by_symbol(funding_items: List[Dict[str, Any]]) -> Dict[str, List[Tuple[int, float]]]:
    """
    funding_items normalizados: [{"symbol","income"(+-),"timestamp"(ms),...}]
    return: { SYM: [(ts, income), ...] ordenado por ts asc }
    """
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


def _fifo_blocks_from_fills(fills: List[Dict[str, Any]],
                            funding_map: Optional[Dict[str, List[Tuple[int, float]]]] = None) -> List[Dict[str, Any]]:
    """
    Reconstruye BLOQUES CERRADOS por símbolo (cuando net vuelve a 0).
    Reglas:
      - side del bloque = BUY → 'long', SELL → 'short'
      - entry_avg: media ponderada de entradas (en la dirección del bloque)
      - close_avg: media ponderada de salidas
      - size del bloque = suma de qty de ENTRADAS
      - fee_total = suma de fees de todos los trades del bloque (NEGATIVA)
      - funding_total = suma de incomes en la ventana [open_time, close_time] por símbolo (si funding_map)
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
        i = 0
        while i < len(arr):
            first = arr[i]
            first_side = first["side"]  # BUY/SELL
            block_side_txt = "long" if first_side == "BUY" else "short"
            net = 0.0
            fee_sum = 0.0
            open_time_ms = arr[i]["timestamp"]

            # medias y tamaño
            entry_sum = close_sum = 0.0
            entry_qty = close_qty = 0.0

            # colas para cálculo final FIFO (repetimos luego para sumar pnl)
            q_long: deque[Tuple[float, float]] = deque()   # (qty, price)
            q_short: deque[Tuple[float, float]] = deque()

            j = i
            while j < len(arr):
                f = arr[j]
                qty = float(f["qty"])
                prc = float(f["price"])
                side = f["side"]
                fee_sum += float(f.get("fee") or 0.0)

                if first_side == "BUY":
                    # bloque long: entradas BUY, salidas SELL
                    if side == "BUY":
                        net += qty
                        entry_sum += prc * qty
                        entry_qty += qty
                        q_long.append((qty, prc))
                    else:
                        net -= qty
                        close_sum += prc * qty
                        close_qty += qty
                        # no necesitamos emparejar aún; lo haremos para pnl más abajo
                else:
                    # bloque short: entradas SELL, salidas BUY
                    if side == "SELL":
                        net -= qty
                        entry_sum += prc * qty
                        entry_qty += qty
                        q_short.append((qty, prc))
                    else:
                        net += qty
                        close_sum += prc * qty
                        close_qty += qty

                if abs(net) <= 1e-12 and entry_qty > 0 and close_qty > 0:
                    break
                j += 1

            # si no cerró el bloque desde i, avanzar
            if abs(net) > 1e-12 or entry_qty <= 0 or close_qty <= 0:
                i = j + 1
                continue

            close_time_ms = arr[j]["timestamp"]

            # ---- calcular PnL FIFO real (reproducimos matching)
            price_pnl = 0.0
            ql: deque[Tuple[float, float]] = deque()
            qs: deque[Tuple[float, float]] = deque()
            for k in range(i, j + 1):
                ff = arr[k]
                qty = float(ff["qty"])
                prc = float(ff["price"])
                sde = ff["side"]
                if sde == "BUY":
                    # cierra short
                    while qty > 1e-12 and qs:
                        sqty, sprice = qs[0]
                        take = min(qty, sqty)
                        price_pnl += (sprice - prc) * take
                        sqty -= take; qty -= take
                        if sqty <= 1e-12: qs.popleft()
                        else: qs[0] = (sqty, sprice)
                    if qty > 1e-12:
                        ql.append((qty, prc))
                else:
                    # cierra long
                    while qty > 1e-12 and ql:
                        lqty, lprice = ql[0]
                        take = min(qty, lqty)
                        price_pnl += (prc - lprice) * take
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


def reconstruct_xt_closed_from_transactions(days: int = 30,
                                            symbol: Optional[str] = None,
                                            limit: int = 1000,
                                            inject_funding: bool = True) -> List[Dict[str, Any]]:
    """
    Pipeline:
      1) fetch_xt_transactions(...): trades crudos
      2) map → fills (price, qty, side, fee, symbol, ts)
      3) funding opcional de fetch_xt_funding_fees(...)
      4) _fifo_blocks_from_fills → lista de posiciones cerradas
    """
    end_ms = utc_now_ms()
    start_ms = end_ms - days * 24 * 60 * 60 * 1000

    # 1) TRADES
    txs = fetch_xt_transactions(limit=limit, symbol=symbol, start_ms=start_ms, end_ms=end_ms, days=days)
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
        fund_items = fetch_xt_funding_fees(limit=1000, start_ms=start_ms, end_ms=end_ms, symbol=None)
        funding_map = _group_funding_by_symbol(fund_items)

    # 4) Reconstrucción FIFO
    blocks = _fifo_blocks_from_fills(fills, funding_map=funding_map)

    return blocks


def save_xt_closed_positions(db_path: str = "portfolio.db",
                                               days: int = 30,
                                               symbol: Optional[str] = None,
                                               limit: int = 2000,
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
    for b in blocks:
        # realized_pnl ya viene como pnl + funding + fee
        # save_closed_position se encarga de recalibrar y completar si falta
        save_closed_position(b)
        saved += 1
        p_closed_sync_saved(EXCHANGE, 1)
    p_closed_sync_done(EXCHANGE, saved)
    return saved


# ==========================
#        DEBUGS
# ==========================
def debug_dump_xt_transactions(days: int = 30, limit: int = 50, symbol: Optional[str] = None):
    """
    Imprime tabla con las transacciones crudas (como tu ejemplo).
    """
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


def debug_preview_xt_closed_from_transactions(days: int = 30,
                                              symbol: Optional[str] = None,
                                              limit: int = 1000,
                                              inject_funding: bool = True):
    """
    Reconstruye y muestra los bloques cerrados (SIN guardar).
    """
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

    print("=== XT CLOSED PREVIEW (from transactions / FIFO) ===")
    for s, v in agg.items():
        print(f"{s}: blocks={v['n']} realized={v['realized']:.6f} fees={v['fees']:.6f} funding={v['funding']:.6f}")

    print("\n— Muestra (hasta 10):")
    for b in blocks[:10]:
        print(json.dumps(b, indent=2, ensure_ascii=False))



# Función de debug mejorada
def debug_transactions_all_symbols(days: int = 30, limit: int = 50, symbol: Optional[str] = None):
    """
    Función de debug para probar las transacciones en todos los símbolos o uno específico
    """
    print("=" * 80)
    print("DEBUG TRANSACCIONES XT - TODOS LOS SÍMBOLOS" if symbol is None else f"DEBUG TRANSACCIONES XT - {symbol}")
    print("=" * 80)
    
    end_ms = utc_now_ms()
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    
    print(f"Período: {days} días ({time.strftime('%Y-%m-%d', time.gmtime(start_ms/1000))} a {time.strftime('%Y-%m-%d', time.gmtime(end_ms/1000))})")
    print(f"Límite: {limit} transacciones")
    print(f"Símbolo: {'TODOS' if symbol is None else symbol}")
    
    transactions = fetch_xt_transactions(
        symbol=symbol,  # None = todos los símbolos
        start_ms=start_ms,
        end_ms=end_ms,
        limit=limit,
        days=days
    )
    
    if transactions:
        # Agrupar por símbolo para estadísticas
        symbol_stats = {}
        for tx in transactions:
            sym = tx.get('symbol', 'UNKNOWN')
            if sym not in symbol_stats:
                symbol_stats[sym] = 0
            symbol_stats[sym] += 1
        
        print(f"\n📊 ESTADÍSTICAS - {len(transactions)} transacciones encontradas:")
        for sym, count in symbol_stats.items():
            print(f"   {sym}: {count} transacciones")
        
        print(f"\n📋 DETALLES DE TRANSACCIONES:")
        print("-" * 140)
        print(f"{'Fecha':<20} {'Symbol':<10} {'Side':<6} {'Cantidad':<12} {'Precio':<12} {'Total':<15} {'Fee':<10} {'T/M':<6} {'Order ID':<15}")
        print("-" * 140)
        
        # Ordenar por timestamp (más reciente primero)
        transactions.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        for tx in transactions[:20]:  # Mostrar solo las 20 más recientes
            date_str = tx.get('datetime', 'N/A')
            symbol = tx.get('symbol', 'N/A')
            side = tx.get('side', 'N/A')
            qty = tx.get('qty', 0)
            price = tx.get('price', 0)
            total = tx.get('quoteQty', 0)
            fee = tx.get('fee', 0)
            taker_maker = tx.get('taker_or_maker', 'N/A')
            order_id = tx.get('order', 'N/A')[:12]  # Mostrar solo primeros 12 chars
            
            print(f"{date_str:<20} {symbol:<10} {side:<6} {qty:<12.6f} {price:<12.6f} {total:<15.6f} {fee:<10.6f} {taker_maker:<6} {order_id:<15}")
        
        if len(transactions) > 20:
            print(f"... y {len(transactions) - 20} transacciones más")
            
        # Mostrar datos raw de ejemplo para debug
        if transactions:
            print(f"\n🔍 EJEMPLO - Primera transacción (raw data):")
            first_tx = transactions[0].get('raw_data', {})
            print(json.dumps(first_tx, indent=2, ensure_ascii=False))
    else:
        print("❌ No se encontraron transacciones.")
        print("Posibles causas:")
        print("   - No hay transacciones en el período seleccionado")
        print("   - Problemas de autenticación")
        print("   - Rate limiting")
        print("   - Error de conexión")
    
    return transactions



# Ejemplos de uso
if __name__ == "__main__":
    print("🧪 TESTING XT TRANSACTIONS - VERSIÓN MEJORADA")
    
    # OPCIÓN 1: Buscar en TODOS los símbolos (últimos 30 días)
    print("\n1. 🎯 BUSCANDO EN TODOS LOS SÍMBOLOS (30 días):")
    all_transactions = debug_transactions_all_symbols(
        days=30,           # Últimos 30 días
        limit=50,          # Máximo 50 transacciones
        symbol=None        # TODOS los símbolos
    )
    
    # OPCIÓN 2: Buscar en un símbolo específico
    print("\n2. 🎯 BUSCANDO EN SÍMBOLO ESPECÍFICO (30 días):")
    specific_transactions = debug_transactions_all_symbols(
        days=30,           # Últimos 30 días  
        limit=30,          # Máximo 30 transacciones
        symbol="BTCUSDT"   # Símbolo específico
    )
    
    # OPCIÓN 3: Buscar en período más corto
    print("\n3. 🎯 BUSCANDO ÚLTIMOS 7 DÍAS:")
    week_transactions = debug_transactions_all_symbols(
        days=7,            # Últimos 7 días
        limit=20,          # Máximo 20 transacciones
        symbol=None        # TODOS los símbolos
    )

# Ejemplo de uso auto-ejecutable para testing
# =========================================================
#                    DEBUG AUTO-EJECUTABLE
# =========================================================
if __name__ == "__main__":
    print("🧪 INICIANDO DEBUG AUTO-EJECUTABLE DE TRANSACCIONES XT")
    print("=" * 80)
    
    # Configuración del debug
    DEBUG_SYMBOL = None  # None para todos los símbolos, o "BTCUSDT" para uno específico
    DEBUG_DAYS = 30      # Últimos 30 días por defecto
    DEBUG_LIMIT = 50     # Límite de transacciones a buscar
    
    print(f"🔧 CONFIGURACIÓN:")
    print(f"   📅 Período: últimos {DEBUG_DAYS} días")
    print(f"   🎯 Símbolo: {'TODOS' if DEBUG_SYMBOL is None else DEBUG_SYMBOL}")
    print(f"   📊 Límite: {DEBUG_LIMIT} transacciones")
    print("-" * 80)
    
    # Ejecutar la función de debug
    transactions = debug_transactions_all_symbols(
        days=DEBUG_DAYS,
        limit=DEBUG_LIMIT,
        symbol=DEBUG_SYMBOL
    )
    
    # Resumen final
    print("\n" + "=" * 80)
    print("🏁 RESUMEN DEL DEBUG")
    print("=" * 80)
    
    if transactions:
        # Estadísticas finales
        total_transactions = len(transactions)
        symbols = set(tx['symbol'] for tx in transactions)
        
        print(f"✅ DEBUG COMPLETADO EXITOSAMENTE")
        print(f"📈 Total de transacciones obtenidas: {total_transactions}")
        print(f"🎯 Símbolos encontrados: {len(symbols)}")
        print(f"📋 Lista de símbolos: {', '.join(sorted(symbols))}")
        
        # Mostrar las 5 transacciones más recientes
        print(f"\n🕒 ÚLTIMAS 5 TRANSACCIONES:")
        print("-" * 120)
        for tx in transactions[:5]:
            print(f"   {tx['datetime']} | {tx['symbol']:8} | {tx['side']:4} | {tx['qty']:10.6f} | {tx['price']:12.6f} | {tx['fee_coin']}: {tx['fee']:.6f}")
            
    else:
        print("❌ NO SE ENCONTRARON TRANSACCIONES")
        print("   Posibles causas:")
        print("   • No hay transacciones en el período seleccionado")
        print("   • Problemas de autenticación con la API")
        print("   • Error en la firma de la solicitud")
        print("   • Rate limiting de la API")
        
    print("=" * 80)
    print("🎯 Para modificar la configuración, edita las variables DEBUG_SYMBOL, DEBUG_DAYS y DEBUG_LIMIT")
    print("=" * 80)
    
    # Pausa final para ver resultados en Spyder
    input("\n⏎ Presiona Enter para finalizar...")