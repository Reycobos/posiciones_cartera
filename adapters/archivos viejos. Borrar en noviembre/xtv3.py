# adapters/xt.py
# XT adapter (Futures + Spot) usando el SDK pyxt (perp.py y spot.py locales si no hay pip).
from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple

# ============ SDK (pip o archivos locales) ============
try:
    from pyxt.perp import Perp  # pip install pyxt
except Exception:
    from perp import Perp       # /mnt/data/perp.py

try:
    from pyxt.spot import Spot
except Exception:
    from spot import Spot       # /mnt/data/spot.py

# ============ helpers del proyecto ============
try:
    from symbols import normalize_symbol
except Exception:
    import re
    def normalize_symbol(sym: str) -> str:
        if not sym: return ""
        s = sym.upper()
        s = re.sub(r'^PERP_', '', s)
        s = re.sub(r'(_|-)?(USDT|USDC|PERP)$', '', s)
        s = re.sub(r'[_-]+$', '', s)
        s = re.split(r'[_-]', s)[0]
        return s

def to_float(x) -> float:
    try: return float(x)
    except Exception: return 0.0

def utc_now_ms() -> int:
    return int(time.time() * 1000)

# logs no-op (si no existen en tu backend)
def _noop(*a, **k): pass
try:
    from portfoliov7 import (
        p_balance_equity, p_balance_fetching, p_balance_done,
        p_funding_fetching, p_funding_count,
        p_open_fetching, p_open_count
    )
except Exception:
    p_balance_equity = p_balance_fetching = p_balance_done = _noop
    p_funding_fetching = p_funding_count = _noop
    p_open_fetching = p_open_count = _noop

# ============ Config ============
EXCHANGE = "xt"
FAPI_HOST = os.getenv("XT_FAPI_HOST", "https://fapi.xt.com")
SAPI_HOST = os.getenv("XT_SAPI_HOST", "https://sapi.xt.com")
XT_API_KEY = "96b1c999-b9df-42ed-9d4f-a4491aac7a1d"
XT_API_SECRET = "5f60a0a147d82db2117a45da129a6f4480488234"
DEFAULT_DAYS_TRADES = int(os.getenv("XT_TRADES_DAYS", "30"))

# ============ Clientes ============
_perp: Optional[Perp] = None
_spot: Optional[Spot] = None

def _get_perp() -> Perp:
    global _perp
    if _perp is None:
        if not XT_API_KEY or not XT_API_SECRET:
            raise RuntimeError("Faltan XT_API_KEY / XT_API_SECRET.")
        _perp = Perp(host=FAPI_HOST, access_key=XT_API_KEY, secret_key=XT_API_SECRET, timeout=15)
    return _perp

def _get_spot() -> Spot:
    global _spot
    if _spot is None:
        if not XT_API_KEY or not XT_API_SECRET:
            raise RuntimeError("Faltan XT_API_KEY / XT_API_SECRET.")
        _spot = Spot(host=SAPI_HOST, access_key=XT_API_KEY, secret_key=XT_API_SECRET)
    return _spot

def _unwrap_result(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "result" in obj: return obj["result"]
        if "data" in obj: return obj["data"]
    return obj

# =========================================================
#                      BALANCES (COMBINADO)
#   Futures:  /future/user/v1/balance/list   -> walletBalance
#   Spot:     /v4/balances (filtrado USDT)   -> assets[].totalAmount
# =========================================================
def fetch_xt_all_balances(db_path: str = "portfolio.db") -> Dict[str, Any] | None:
    """
    Entrega la forma EXACTA /api/balances sumando Futuros + Spot(USDT):
    {
      "exchange":"xt","equity":float,"balance":float,"unrealized_pnl":0.0,
      "initial_margin":0.0,"spot":float(usdt_spot),"margin":0.0,"futures":float(futures_equity)
    }
    """
    p_balance_fetching(EXCHANGE)

    # -------- Futuros --------
    cli_f = _get_perp()
    code, success, error = cli_f.get_account_capital()   # GET /future/user/v1/balance/list
    if error or code != 200 or success is None:
        raise RuntimeError(f"XT futures balance error: {error or code}")
    res_f = _unwrap_result(success)
    futures_equity = 0.0
    if isinstance(res_f, list):
        for it in res_f:
            if isinstance(it, dict):
                futures_equity += to_float(it.get("walletBalance") or 0.0)
    elif isinstance(res_f, dict):
        arr = res_f.get("items") or res_f.get("list") or []
        if isinstance(arr, list):
            for it in arr:
                if isinstance(it, dict):
                    futures_equity += to_float(it.get("walletBalance") or 0.0)

    # -------- Spot (solo USDT) --------
    cli_s = _get_spot()
    # balances(currencies=['usdt']) -> { totalBtcAmount, assets: [ {currency, availableAmount, frozenAmount, totalAmount, convertBtcAmount} ] }
    spot_res = cli_s.balances(currencies=['usdt'])
    spot_total_usdt = 0.0
    try:
        assets = (spot_res or {}).get("assets") or []
        if isinstance(assets, list):
            for a in assets:
                if not isinstance(a, dict): continue
                curr = str(a.get("currency") or "").lower()
                if curr == "usdt":
                    spot_total_usdt += to_float(a.get("totalAmount") or a.get("availableAmount") or 0.0)
    except Exception:
        # fallback por si el SDK ya devuelve directamente la lista
        if isinstance(spot_res, list):
            for a in spot_res:
                if isinstance(a, dict) and str(a.get("currency","")).lower() == "usdt":
                    spot_total_usdt += to_float(a.get("totalAmount") or a.get("availableAmount") or 0.0)

    equity = futures_equity + spot_total_usdt
    obj = {
        "exchange": EXCHANGE,
        "equity": float(equity),
        "balance": float(equity),      # sin uPNL disponible, balance==equity
        "unrealized_pnl": 0.0,
        "initial_margin": 0.0,
        "spot": float(spot_total_usdt),
        "margin": 0.0,
        "futures": float(futures_equity),
    }
    p_balance_equity(EXCHANGE, obj["equity"])
    p_balance_done(EXCHANGE)
    return obj

# =========================================================
#                         FUNDING FEES
#   GET /future/user/v1/balance/funding-rate-list
#   (usando firma del SDK: _create_sign + _fetch)
# =========================================================
def fetch_xt_funding_fees(
    limit: int = 50,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    symbol: Optional[str] = None,
) -> List[Dict[str, Any]]:
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
        if next_id is not None:
            params["id"] = int(next_id)

        header = cli._create_sign(XT_API_KEY, XT_API_SECRET, path=path,
                                  bodymod="application/x-www-form-urlencoded", params=params)
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
#                   CLOSED POSITIONS (pendiente)
# =========================================================
def save_xt_closed_positions(db_path: str = "portfolio.db",
                             days: int = DEFAULT_DAYS_TRADES,
                             debug: bool = False) -> None:
    print("XT: sin endpoint de fills en la guía actual. Cuando compartas el de 'user fills', implemento FIFO y guardo en SQLite.")

# =========================================================
#                           DEBUGS
# =========================================================
def debug_raw_fapi_balances():
    cli = _get_perp()
    code, success, error = cli.get_account_capital()
    print("=== RAW FUTURES /balance/list ===")
    print("code:", code, "error:", error)
    print(json.dumps(success, indent=2, ensure_ascii=False))

def debug_raw_spot_balances(currencies: Optional[str] = "usdt"):
    cli = _get_spot()
    cur_list = [c.strip() for c in (currencies or "").split(",") if c.strip()] if currencies else None
    res = cli.balances(currencies=cur_list) if cur_list else cli.balances()
    print("=== RAW SPOT /v4/balances ===")
    print(json.dumps(res, indent=2, ensure_ascii=False))

def debug_raw_fapi_positions(symbol: Optional[str] = None):
    cli = _get_perp()
    if symbol:
        code, success, error = cli.get_position(symbol)
    else:
        path = "/future/user" + "/v1/position/list"
        params: Dict[str, Any] = {}
        header = cli._create_sign(XT_API_KEY, XT_API_SECRET, path=path,
                                  bodymod="application/x-www-form-urlencoded", params=params)
        header["Content-Type"] = "application/x-www-form-urlencoded"
        url = cli.host + path
        code, success, error = cli._fetch(method="GET", url=url, headers=header, params=params, timeout=cli.timeout)
    print("=== RAW FUTURES /position/list ===")
    print("code:", code, "error:", error)
    print(json.dumps(success, indent=2, ensure_ascii=False))

def debug_raw_fapi_funding(days: int = 14, symbol: Optional[str] = None, limit: int = 50):
    cli = _get_perp()
    end_ms = utc_now_ms()
    start_ms = end_ms - int(days * 86400000)
    params: Dict[str, Any] = {
        "limit": limit,
        "direction": "NEXT",
        "startTime": int(start_ms),
        "endTime": int(end_ms),
    }
    if symbol:
        params["symbol"] = symbol
    path = "/future/user" + "/v1/balance/funding-rate-list"
    header = cli._create_sign(XT_API_KEY, XT_API_SECRET, path=path,
                              bodymod="application/x-www-form-urlencoded", params=params)
    header["Content-Type"] = "application/x-www-form-urlencoded"
    url = cli.host + path
    code, success, error = cli._fetch(method="GET", url=url, headers=header, params=params, timeout=cli.timeout)
    print("=== RAW FUTURES /balance/funding-rate-list ===")
    print("code:", code, "error:", error)
    print(json.dumps(success, indent=2, ensure_ascii=False))



#=================================
#======Debug

# ======================= DEBUGS AVANZADOS =======================

from collections import deque, defaultdict

def debug_dump_xt_opens() -> None:
    """
    Imprime posiciones abiertas *normalizadas* y un pequeño resumen.
    """
    print("=== XT OPEN POSITIONS (normalized) ===")
    try:
        opens = fetch_xt_open_positions()
    except Exception as e:
        print(f"❌ fetch_xt_open_positions error: {e}")
        return
    print(json.dumps(opens, indent=2, ensure_ascii=False))
    # resumen rápido
    tot = len(opens)
    notional = sum(to_float(x.get("notional", 0.0)) for x in opens)
    upnl = sum(to_float(x.get("unrealized_pnl", 0.0)) for x in opens)
    by_sym = defaultdict(int)
    for x in opens:
        by_sym[x["symbol"]] += 1
    print(f"\n--- summary ---\ncount={tot}  notional={notional:.4f}  uPnL={upnl:.4f}")
    print("per symbol:", dict(by_sym))


def debug_dump_xt_funding(limit: int = 100, days: int = 14, symbol: Optional[str] = None) -> None:
    """
    Imprime funding fees *normalizados* y un resumen por símbolo.
    """
    print("=== XT FUNDING FEES (normalized) ===")
    end_ms = utc_now_ms()
    start_ms = end_ms - int(days * 86400000)
    try:
        funding = fetch_xt_funding_fees(limit=limit, start_ms=start_ms, end_ms=end_ms, symbol=symbol)
    except Exception as e:
        print(f"❌ fetch_xt_funding_fees error: {e}")
        return
    print(json.dumps(funding, indent=2, ensure_ascii=False))
    # resumen por símbolo
    by_sym_sum = defaultdict(float)
    for it in funding:
        by_sym_sum[it["symbol"]] += to_float(it.get("income", 0.0))
    print("\n--- summary ---")
    for s, v in by_sym_sum.items():
        print(f"{s}: {v:.8f}")
    print(f"total items: {len(funding)}")


# --------- soporte fills (intento de descubrimiento del endpoint) ---------
def _perp_sign_get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 20) -> Tuple[int, Any, Any]:
    """
    Hace GET firmado via Perp._create_sign + _fetch, misma firma ya validada.
    """
    cli = _get_perp()
    header = cli._create_sign(XT_API_KEY, XT_API_SECRET, path=path,
                              bodymod="application/x-www-form-urlencoded", params=(params or {}))
    header["Content-Type"] = "application/x-www-form-urlencoded"
    url = cli.host + path
    return cli._fetch(method="GET", url=url, headers=header, params=(params or {}), timeout=timeout)

def _unwrap_result_any(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "result" in obj: return obj["result"]
        if "data" in obj: return obj["data"]
        # a veces viene {"items":[...]}
        if "items" in obj: return obj["items"]
        if "list" in obj: return obj["list"]
    return obj

# Mapeo flexible de campos en fills (varía muchísimo según versión)
def _parse_fill(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict): 
        return None
    # precio
    price = item.get("price") or item.get("avgPrice") or item.get("dealPrice")
    # cantidad
    qty = item.get("size") or item.get("quantity") or item.get("qty") or item.get("volume") or item.get("dealVol")
    # lado
    side = (item.get("side") or item.get("direction") or "").upper()
    # comisión (negativa)
    fee = item.get("fee") or item.get("fees") or item.get("commission")
    # símbolo y tiempo
    sym = item.get("symbol") or item.get("instrument") or item.get("market")
    ts = item.get("timestamp") or item.get("time") or item.get("ts") or item.get("createdTime") or item.get("ctime")
    try:
        p = float(price)
        q = float(qty)
        f = float(fee) if fee is not None else 0.0
        t = int(ts)
        if t < 10**12:  # s → ms
            t *= 1000
    except Exception:
        return None
    sd = "BUY" if side in ("BUY", "LONG", "BID") else "SELL"
    return {"price": p, "qty": abs(q), "side": sd, "fee": float(f), "symbol": str(sym or ""), "ts": t}

def _fifo_blocks_from_fills(fills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reconstruye bloques cerrados (net -> 0) por símbolo usando FIFO real.
    Devuelve una lista de filas *normalizadas* de closed preview (sin funding ni margin).
    """
    by_sym: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in fills:
        base = normalize_symbol(f["symbol"])
        ff = dict(f)
        ff["symbol"] = base
        by_sym[base].append(ff)

    results: List[Dict[str, Any]] = []

    for sym, arr in by_sym.items():
        # ordenar por tiempo
        arr.sort(key=lambda x: x["ts"])
        i = 0
        while i < len(arr):
            # empezar bloque en i
            start_i = i
            first_side = arr[i]["side"]  # "BUY" abre long, "SELL" abre short
            net = 0.0
            max_abs_net = 0.0
            entries_q = 0.0
            exits_q = 0.0
            fees_sum = 0.0

            # cola de lotes abiertos para FIFO (qty, price)
            openq: deque = deque()
            price_pnl = 0.0

            j = i
            while j < len(arr):
                f = arr[j]
                qty = f["qty"]
                prc = f["price"]
                side = f["side"]
                fees_sum += float(f.get("fee", 0.0) or 0.0)

                if first_side == "BUY":
                    # long: entran BUY, salen SELL
                    if side == "BUY":
                        openq.append([qty, prc])
                        net += qty
                        entries_q += qty
                    else:  # SELL cierra
                        qleft = qty
                        exits_q += qty
                        net -= qty
                        # FIFO match
                        while qleft > 1e-12 and openq:
                            lot_qty, lot_price = openq[0]
                            take = min(lot_qty, qleft)
                            price_pnl += (prc - lot_price) * take
                            lot_qty -= take
                            qleft -= take
                            if lot_qty <= 1e-12:
                                openq.popleft()
                            else:
                                openq[0][0] = lot_qty
                else:
                    # short: entran SELL, salen BUY
                    if side == "SELL":
                        openq.append([qty, prc])
                        net -= qty
                        entries_q += qty
                    else:  # BUY cierra
                        qleft = qty
                        exits_q += qty
                        net += qty
                        while qleft > 1e-12 and openq:
                            lot_qty, lot_price = openq[0]
                            take = min(lot_qty, qleft)
                            # pnl de short = (entry - close) * qty
                            price_pnl += (lot_price - prc) * take
                            lot_qty -= take
                            qleft -= take
                            if lot_qty <= 1e-12:
                                openq.popleft()
                            else:
                                openq[0][0] = lot_qty

                max_abs_net = max(max_abs_net, abs(net))
                # si net vuelve a 0 → bloque cerrado
                if abs(net) <= 1e-12:
                    break
                j += 1

            # si no cerró, abortar bloque
            if abs(net) > 1e-12:
                # no bloque cerrado; salir del while
                i = j + 1
                continue

            # métricas del bloque [start_i..j]
            block = arr[start_i:j+1]
            open_time = block[0]["ts"]
            close_time = block[-1]["ts"]
            side_txt = "long" if first_side == "BUY" else "short"
            # medias ponderadas
            # entradas = trades con side del primer fill
            entry_sum = 0.0; entry_qty = 0.0
            close_sum = 0.0; close_qty = 0.0
            for f in block:
                if (first_side == "BUY" and f["side"] == "BUY") or (first_side != "BUY" and f["side"] == "SELL"):
                    entry_sum += f["price"] * f["qty"]; entry_qty += f["qty"]
                else:
                    close_sum += f["price"] * f["qty"]; close_qty += f["qty"]
            entry_avg = (entry_sum / entry_qty) if entry_qty > 0 else 0.0
            close_avg = (close_sum / close_qty) if close_qty > 0 else entry_avg

            fee_total = -abs(fees_sum)  # contrato: fees SIEMPRE negativas
            funding_total = 0.0         # preview: no traemos funding por bloque aquí
            realized = price_pnl - (-fee_total) + funding_total  # price - fees + funding

            results.append({
                "exchange": EXCHANGE,
                "symbol": sym,
                "side": side_txt,
                "size": float(max_abs_net),
                "entry_price": float(entry_avg),
                "close_price": float(close_avg),
                "open_time": int(open_time // 1000),
                "close_time": int(close_time // 1000),
                "pnl": float(price_pnl),
                "realized_pnl": float(realized),
                "funding_total": float(funding_total),
                "fee_total": float(fee_total),
                "notional": float(max_abs_net * entry_avg),
                "initial_margin": None,
                "leverage": None,
                "liquidation_price": None,
            })

            i = j + 1

    return results


def debug_preview_xt_closed(days: int = DEFAULT_DAYS_TRADES,
                            symbol: Optional[str] = None,
                            limit: int = 2000,
                            with_funding: bool = False) -> None:
    """
    Intenta localizar 'fills' y reconstruir *preview* de posiciones cerradas con FIFO real.
    No guarda en DB. Si no encuentra endpoint de fills, imprime RAW de los intentos.

    Params:
      days: ventana temporal para buscar fills
      symbol: si lo pasas, usa el formato del exchange ('btc_usdt'); se normaliza en la salida
      limit: máximo de fills a usar por preview
      with_funding: si True, intentará inyectar funding_total por bloque consultando el endpoint
                    funding-rate-list por símbolo (nota: puede ser costoso).
    """
    end_ms = utc_now_ms()
    start_ms = end_ms - int(days * 86400000)

    # candidatos habituales en XT (si alguno existe en tu versión, lo pillamos)
    CANDIDATES = [
        "/future/user/v1/trade/list",
        "/future/user/v1/order/trades",
        "/future/trade/v1/user-trades",
        "/future/user/v1/order/fills",
        "/future/user/v1/myTrades",
    ]

    found_items: List[Dict[str, Any]] = []
    tried = []
    print("=== XT CLOSED PREVIEW — endpoint discovery ===")
    for path in CANDIDATES:
        params = {
            "startTime": int(start_ms),
            "endTime": int(end_ms),
            "limit": min(1000, limit),
        }
        if symbol:
            params["symbol"] = symbol
        try:
            code, success, error = _perp_sign_get(path, params=params)
            tried.append({"path": path, "code": code, "error": error})
            if error or code != 200 or success is None:
                continue
            res = _unwrap_result_any(success)
            items = res if isinstance(res, list) else _unwrap_result_any(res)
            if isinstance(items, dict):
                # a veces viene {"items":[...]} o {"list":[...]}
                items = items.get("items") or items.get("list") or []
            if isinstance(items, list) and items:
                print(f"✅ fills endpoint encontrado: {path}  items={len(items)}")
                # dump corto
                print(json.dumps(items[:5], indent=2, ensure_ascii=False))
                # intentar parseo flexible
                parsed = []
                for it in items[:limit]:
                    p = _parse_fill(it)
                    if p: parsed.append(p)
                if parsed:
                    found_items = parsed
                    break
                else:
                    print("⚠️ El endpoint devolvió items pero no pude mapear campos (price/qty/side/fee/ts).")
        except Exception as e:
            tried.append({"path": path, "code": "EXC", "error": str(e)})
            continue

    if not found_items:
        print("\n❌ No se halló un endpoint de fills compatible en esta lista. Intentos:")
        print(json.dumps(tried, indent=2, ensure_ascii=False))
        print("Dime cuál es el endpoint de *user fills* correcto y lo ajusto al milímetro.")
        return

    # Reconstrucción FIFO preview
    print("\n=== Reconstrucción FIFO (preview, sin guardar) ===")
    blocks = _fifo_blocks_from_fills(found_items)
    print(json.dumps(blocks, indent=2, ensure_ascii=False))
    print(f"\n--- summary ---\nclosed_blocks={len(blocks)}")

    if with_funding and blocks:
        print("\n⚙️ with_funding=True → inyectar funding_total por bloque (puede tardar).")
        # Mínima integración: sumar funding por símbolo y ventana total (aprox),
        # o hacer por bloque (open..close) si quieres exactitud (más llamadas).
        # Aquí mostramos un ejemplo por ventana total para cada símbolo:
        by_sym = sorted(set(b["symbol"] for b in blocks))
        fund_map = {s: 0.0 for s in by_sym}
        for s in by_sym:
            try:
                f = fetch_xt_funding_fees(limit=500, start_ms=start_ms, end_ms=end_ms, symbol=f"{s.lower()}_usdt")
                fund_map[s] = sum(to_float(x.get("income", 0.0)) for x in f if normalize_symbol(x.get("symbol")) == s)
            except Exception as e:
                print(f"funding fetch error for {s}: {e}")
        print("\nFunding aprox por símbolo en ventana:")
        print(json.dumps(fund_map, indent=2, ensure_ascii=False))
# =========================================================
#                            CLI
# =========================================================
if __name__ == "__main__":
    import argparse, sys, traceback

    def _pf(*a):
        print(*a, flush=True)

    ap = argparse.ArgumentParser("XT adapter (pyxt) — Spot + Futures")
    ap.add_argument("--balance", action="store_true", help="Saldo combinado normalizado (/api/balances)")
    ap.add_argument("--positions", action="store_true", help="Posiciones abiertas normalizadas")
    ap.add_argument("--funding", type=int, default=0, help="N funding fees normalizados")
    ap.add_argument("--raw-balance", action="store_true", help="RAW FUTURES balance/list")
    ap.add_argument("--raw-spot", action="store_true", help="RAW SPOT balances")
    ap.add_argument("--raw-positions", action="store_true", help="RAW FUTURES position/list")
    ap.add_argument("--raw-funding", action="store_true", help="RAW FUTURES funding-rate-list")
    ap.add_argument("--symbol", type=str, default=None)
    ap.add_argument("--spot-currencies", type=str, default="usdt")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--check", action="store_true", help="Autotest rápido (por defecto si no hay flags)")
    ap.add_argument("--debug-opens", action="store_true", help="Dump de posiciones abiertas NORMALIZADAS")
    ap.add_argument("--debug-funding", type=int, default=0, help="Dump funding NORMALIZADO (N items)")
    ap.add_argument("--debug-closed", action="store_true", help="Preview FIFO de posiciones CERRADAS (sin guardar)")
    ap.add_argument("--with-funding", action="store_true", help="Para --debug-closed: sumar funding aprox por símbolo")
    args = ap.parse_args()

    # Banner + entorno detectado
    _pf("🚀 XT adapter runner")
    _pf(f"• FAPI_HOST={os.getenv('XT_FAPI_HOST', 'https://fapi.xt.com')}")
    _pf(f"• SAPI_HOST={os.getenv('XT_SAPI_HOST', 'https://sapi.xt.com')}")
    _pf(f"• API_KEY set: {'yes' if os.getenv('XT_API_KEY') else 'no'}")
    _pf(f"• Flags: {sys.argv[1:]}")

    ran = False
    try:
        if args.raw_balance:
            debug_raw_fapi_balances(); ran = True
        if args.raw_spot:
            debug_raw_spot_balances(args.spot_currencies); ran = True
        if args.raw_positions:
            debug_raw_fapi_positions(args.symbol); ran = True
        if args.raw_funding:
            debug_raw_fapi_funding(args.days, args.symbol, max(1, args.funding or 50)); ran = True
        if args.balance:
            _pf(json.dumps(fetch_xt_all_balances() or {}, indent=2)); ran = True
        if args.positions:
            _pf(json.dumps(fetch_xt_open_positions(), indent=2)); ran = True
        if args.funding:
            _pf(json.dumps(fetch_xt_funding_fees(limit=args.funding, symbol=args.symbol), indent=2)); ran = True
        if args.debug_opens:
            debug_dump_xt_opens(); ran = True

        if args.debug_funding:
            debug_dump_xt_funding(limit=max(1, args.debug_funding), days=args.days, symbol=args.symbol); ran = True
    
        if args.debug_closed:
            debug_preview_xt_closed(days=args.days, symbol=args.symbol, limit=2000, with_funding=args.with_funding); ran = True

        

        # Autorun si no hubo ninguna bandera o si se pidió --check
        if args.check or not ran:
            _pf("\n🧪 QUICK CHECK (autorun)")
            try:
                _pf("• Spot RAW:"); debug_raw_spot_balances(args.spot_currencies)
            except Exception as e:
                _pf(f"  spot error: {e}")
            try:
                _pf("• Futures RAW balance:"); debug_raw_fapi_balances()
            except Exception as e:
                _pf(f"  futures balance error: {e}")
            try:
                _pf("• Futures RAW positions:"); debug_raw_fapi_positions(args.symbol)
            except Exception as e:
                _pf(f"  futures positions error: {e}")
            try:
                _pf("• Normalizado /api/balances:")
                _pf(json.dumps(fetch_xt_all_balances() or {}, indent=2))
            except Exception as e:
                _pf(f"  normalize balance error: {e}")
    except Exception:
        _pf("❌ Uncaught error:\n" + traceback.format_exc())
