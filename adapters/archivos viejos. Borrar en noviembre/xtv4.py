# adapters/xt.py
# XT adapter (Futures + Spot) con firma estilo pyxt y reconstrucción FIFO de cerradas.
from __future__ import annotations

import os, json, time, argparse, sys, traceback
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

# ===================== SDK (pip o archivos locales) =====================
try:
    from pyxt.perp import Perp  # pip install pyxt
except Exception:
    from perp import Perp       # /mnt/data/perp.py (adjunto por ti)

try:
    from pyxt.spot import Spot
except Exception:
    try:
        from spot import Spot   # /mnt/data/spot.py (adjunto por ti)
    except Exception:
        Spot = None

# ===================== Helpers del proyecto =====================
try:
    from symbols import normalize_symbol  # tu normalizador oficial
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

# -------- imprimir estilo backend (no-ops si no existen) --------
def _noop(*a, **k): pass
try:
    from portfoliov7 import (
        p_balance_fetching, p_balance_done, p_balance_equity,
        p_funding_fetching, p_funding_count,
        p_open_fetching, p_open_count,
        p_closed_sync_start, p_closed_sync_saved, p_closed_sync_done, p_closed_sync_none,
    )
except Exception:
    p_balance_fetching = p_balance_done = p_balance_equity = _noop
    p_funding_fetching = p_funding_count = _noop
    p_open_fetching = p_open_count = _noop
    p_closed_sync_start = p_closed_sync_saved = p_closed_sync_done = p_closed_sync_none = _noop

# -------- DB manager --------
try:
    from db_manager import save_closed_position
except Exception:
    save_closed_position = None  # en debug puedes no guardar

# ===================== Config =====================
EXCHANGE = "xt"
FAPI_HOST = os.getenv("XT_FAPI_HOST", "https://fapi.xt.com")   # Futures
SAPI_HOST = os.getenv("XT_SAPI_HOST", "https://sapi.xt.com")   # Spot
XT_API_KEY = "96b1c999-b9df-42ed-9d4f-a4491aac7a1d"
XT_API_SECRET = "5f60a0a147d82db2117a45da129a6f4480488234"
DEFAULT_DAYS_TRADES = int(os.getenv("XT_TRADES_DAYS", "30"))   # ventana para fills/FIFO

# ===================== Clientes =====================
_perp: Optional[Perp] = None
_spot: Optional[Spot] = None

def _get_perp() -> Perp:
    global _perp
    if _perp is None:
        if not XT_API_KEY or not XT_API_SECRET:
            raise RuntimeError("Faltan XT_API_KEY / XT_API_SECRET.")
        _perp = Perp(host=FAPI_HOST, access_key=XT_API_KEY, secret_key=XT_API_SECRET, timeout=15)
    return _perp

def _get_spot() -> Optional[Spot]:
    global _spot
    if Spot is None:
        return None
    if _spot is None:
        if not XT_API_KEY or not XT_API_SECRET:
            raise RuntimeError("Faltan XT_API_KEY / XT_API_SECRET.")
        _spot = Spot(host=SAPI_HOST, access_key=XT_API_KEY, secret_key=XT_API_SECRET)
    return _spot

# ===================== Utils =====================
def _unwrap_result(x: Any) -> Any:
    """Desempaqueta formatos típicos: {'returnCode':0,...,'result':...} o {'data':...}."""
    if isinstance(x, dict):
        if "result" in x and x["result"] is not None:
            return x["result"]
        if "data" in x and x["data"] is not None:
            return x["data"]
        if "items" in x: return x["items"]
        if "list" in x: return x["list"]
    return x

def _perp_sign_get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 20) -> Tuple[int, Any, Any]:
    """
    GET firmado usando la firma interna del SDK (ya probada).
    Content-Type application/x-www-form-urlencoded, params en query.
    """
    cli = _get_perp()
    header = cli._create_sign(XT_API_KEY, XT_API_SECRET, path=path,
                              bodymod="application/x-www-form-urlencoded", params=(params or {}))
    header["Content-Type"] = "application/x-www-form-urlencoded"
    url = cli.host + path
    return cli._fetch(method="GET", url=url, headers=header, params=(params or {}), timeout=timeout)

def _ts_ms_guess(v: Any) -> int:
    try:
        t = int(v)
        return t if t > 10**12 else t * 1000
    except Exception:
        return utc_now_ms()

# ===================== API pública =====================

def fetch_xt_all_balances(db_path: str = "portfolio.db") -> Dict[str, Any]:
    """
    Estructura EXACTA /api/balances:
    {"exchange":"xt","equity":float,"balance":float,"unrealized_pnl":float,"initial_margin":0.0,
     "spot":float,"margin":0.0,"futures":float}
    """
    p_balance_fetching(EXCHANGE)
    futures_equity = 0.0
    try:
        cli = _get_perp()
        code, success, error = cli.get_account_capital()  # /future/user/v1/balance/list (wrapper)
        if error or code != 200 or not success:
            raise RuntimeError(error or f"code={code}")
        res = _unwrap_result(success)
        if isinstance(res, list):
            # suma por asset
            for a in res:
                futures_equity += to_float(a.get("walletBalance") or a.get("amount") or a.get("totalAmount") or 0.0)
        elif isinstance(res, dict):
            # resumen plano
            futures_equity = to_float(res.get("walletBalance") or res.get("amount") or res.get("totalAmount") or
                                      res.get("equity") or res.get("totalEquity") or 0.0)
    except Exception as e:
        print(f"❌ XT futures balance error: {e}")

    spot_usdt = 0.0
    try:
        sp = _get_spot()
        if sp:
            # /v4/balances (SAPI)
            # admite filtro 'currencies' (lista) -> devolvió bien con tu firma
            res = sp.balances()
            res2 = _unwrap_result(res)
            assets = []
            if isinstance(res2, dict) and "assets" in res2:
                assets = res2["assets"]
            elif isinstance(res2, list):
                assets = res2
            for a in assets or []:
                if not isinstance(a, dict): continue
                c = str(a.get("currency") or "").lower()
                if c == "usdt":
                    spot_usdt += to_float(a.get("totalAmount") or a.get("availableAmount") or 0.0)
    except Exception as e:
        print(f"❌ XT spot balance error: {e}")

    equity = futures_equity + spot_usdt
    obj = {
        "exchange": EXCHANGE,
        "equity": float(equity),
        "balance": float(equity),         # sin uPNL por asset, balance≈equity
        "unrealized_pnl": 0.0,
        "initial_margin": 0.0,
        "spot": float(spot_usdt),
        "margin": 0.0,
        "futures": float(futures_equity),
    }
    p_balance_equity(EXCHANGE, obj["equity"])
    p_balance_done(EXCHANGE)
    return obj

def fetch_xt_funding_fees(limit: int = 50,
                          start_ms: Optional[int] = None,
                          end_ms: Optional[int] = None,
                          symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Estructura EXACTA /api/funding:
    {"exchange":"xt","symbol":"<NORMALIZADO>","income":float(+/-),"asset":"USDT|USDC|USD",
     "timestamp":int(ms),"funding_rate":0.0,"type":"FUNDING_FEE"}
    """
    p_funding_fetching(EXCHANGE)
    cli = _get_perp()
    if end_ms is None: end_ms = utc_now_ms()
    if start_ms is None: start_ms = end_ms - 14*24*60*60*1000

    out: List[Dict[str, Any]] = []
    next_id: Optional[int] = None
    direction = "NEXT"
    path = "/future/user" + "/v1/balance/funding-rate-list"

    while len(out) < limit:
        page_size = min(100, max(1, limit - len(out)))
        params: Dict[str, Any] = {
            "limit": page_size, "direction": direction,
            "startTime": int(start_ms), "endTime": int(end_ms),
        }
        if symbol: params["symbol"] = symbol
        if next_id is not None: params["id"] = int(next_id)

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
            income = to_float(it.get("cast") or it.get("amount") or it.get("income") or 0.0)
            asset = (it.get("coin") or "USDT").upper()
            ts = _ts_ms_guess(it.get("createdTime") or it.get("time") or it.get("timestamp") or 0)
            out.append({
                "exchange": EXCHANGE,
                "symbol": base,
                "income": float(income),
                "asset": "USDT" if asset not in ("USDT", "USDC", "USD") else asset,
                "timestamp": int(ts),
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

def fetch_xt_open_positions() -> List[Dict[str, Any]]:
    """
    Estructura EXACTA /api/positions:
    {
      "exchange","symbol","side","size","entry_price","mark_price","liquidation_price",
      "notional","unrealized_pnl","fee","funding_fee","realized_pnl"
    }
    """
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
        if not isinstance(p, dict): continue
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

# ===================== Cerradas por FIFO (fills) =====================

# Candidatos típicos de *fills* en XT (varían por versión)
FILLS_CANDIDATES = [
    "/future/user/v1/trade/list",
    "/future/user/v1/order/trades",
    "/future/trade/v1/user-trades",
    "/future/user/v1/order/fills",
    "/future/user/v1/myTrades",
]

def _parse_fill(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normaliza un fill en {symbol, side(BUY/SELL), price, qty, fee(neg), ts(ms)}.
    Devuelve None si faltan campos críticos.
    """
    if not isinstance(item, dict): return None
    price = item.get("price") or item.get("avgPrice") or item.get("dealPrice") or item.get("fillPrice")
    qty   = item.get("size") or item.get("quantity") or item.get("qty") or item.get("volume") or item.get("dealVol") or item.get("filledQty")
    side0 = (item.get("side") or item.get("direction") or "").upper()
    fee   = item.get("fee") or item.get("fees") or item.get("commission")
    sym   = item.get("symbol") or item.get("instrument") or item.get("market") or item.get("instId")
    ts    = item.get("timestamp") or item.get("time") or item.get("ts") or item.get("createdTime") or item.get("ctime") or item.get("fillTime")
    try:
        p = float(price); q = abs(float(qty)); f = float(fee) if fee is not None else 0.0; t = _ts_ms_guess(ts)
    except Exception:
        return None
    if f > 0: f = -abs(f)  # siempre negativa en tu contrato
    side = "BUY" if side0 in ("BUY","OPEN_LONG","CLOSE_SHORT","BID") else "SELL"
    return {"symbol": normalize_symbol(str(sym or "")), "side": side, "price": p, "qty": q, "fee": f, "ts": int(t)}

# ======================= SDK-based fills discovery (new) =======================

def _call_perp_method_maybe(cli, name, kwargs_list):
    """
    Intenta llamar cli.<name>(**kwargs) con varias combinaciones de parámetros.
    Acepta retorno como (code, success, error) o dict/list directo.
    Devuelve: (ok: bool, items: list, raw: Any, meta: dict)
    """
    if not hasattr(cli, name):
        return False, [], None, {"method": name, "exists": False}

    fn = getattr(cli, name)
    tried = []
    for kw in kwargs_list:
        try:
            ret = fn(**kw)
        except TypeError:
            # reintento con 'symbol' solamente o sin args por compat.
            try:
                if "symbol" in kw:
                    ret = fn(symbol=kw["symbol"])
                else:
                    ret = fn()
            except Exception as e2:
                tried.append({"kwargs": kw, "error": f"{type(e2).__name__}: {e2}"})
                continue
        except Exception as e:
            tried.append({"kwargs": kw, "error": f"{type(e).__name__}: {e}"})
            continue

        code = None; success = None; error = None
        raw = ret
        if isinstance(ret, tuple) and len(ret) == 3:
            code, success, error = ret
            if error or (code is not None and code != 200) or success is None:
                tried.append({"kwargs": kw, "code": code, "error": error})
                continue
            raw = success

        data = _unwrap_result(raw)
        items = data if isinstance(data, list) else _unwrap_result(data)
        if isinstance(items, dict):
            items = items.get("items") or items.get("list") or []
        if isinstance(items, list) and items:
            return True, items, raw, {"method": name, "kwargs": kw, "code": code, "tried": tried}

        tried.append({"kwargs": kw, "code": code, "note": "empty"})
    return False, [], None, {"method": name, "exists": True, "tried": tried}


def _fetch_user_fills_via_sdk_methods(start_ms: int, end_ms: int, symbol: Optional[str], limit: int):
    """
    Sonda el SDK Perp buscando cualquier método de 'user trades' y adapta la salida a fills normalizados.
    """
    cli = _get_perp()
    candidates = [
        "get_user_trades", "user_trades", "get_trades", "get_trade_list",
        "get_fills", "get_fill_list", "get_deal_list", "get_match_results",
        "get_order_trades", "get_order_deals", "my_trades", "myTrades",
    ]
    # Variantes de parámetros típicos
    base = {"symbol": symbol, "startTime": int(start_ms), "endTime": int(end_ms), "limit": int(limit)}
    alt1 = {"symbol": symbol, "start_time": int(start_ms), "end_time": int(end_ms), "limit": int(limit)}
    alt2 = {"symbol": symbol, "start": int(start_ms), "end": int(end_ms), "limit": int(limit)}
    alt3 = {"startTime": int(start_ms), "endTime": int(end_ms), "limit": int(limit)}
    alt4 = {"start_time": int(start_ms), "end_time": int(end_ms), "limit": int(limit)}
    alt5 = {"symbol": symbol, "pageSize": int(min(1000, limit))}
    kwargs_list = [base, alt1, alt2, alt3, alt4, alt5, {}]

    logs = []
    for name in candidates:
        ok, items, raw, meta = _call_perp_method_maybe(cli, name, kwargs_list)
        logs.append(meta)
        if ok:
            parsed = []
            for it in items[:limit]:
                p = _parse_fill(it)
                if p: parsed.append(p)
            if parsed:
                print(f"✅ SDK fills via {name} with {meta.get('kwargs')}  parsed={len(parsed)}")
                return parsed, {"method": name, "kwargs": meta.get("kwargs"), "sample": items[:5], "tries": logs}

    print("❌ No SDK method returned fills with data. Tries:", json.dumps(logs, ensure_ascii=False, indent=2))
    return [], {"tries": logs}


# Sustituye tu _discover_user_fills por esta versión: primero SDK, luego REST-candidatos
def _discover_user_fills(start_ms: int, end_ms: int, symbol: Optional[str], limit: int) -> List[Dict[str, Any]]:
    parsed, meta = _fetch_user_fills_via_sdk_methods(start_ms, end_ms, symbol, limit)
    if parsed:
        return parsed

    # Fallback a REST-candidatos (en tu env devolvieron 404, se mantiene por compat)
    found: List[Dict[str, Any]] = []
    tried: List[Dict[str, Any]] = []
    for path in FILLS_CANDIDATES:
        params = {"startTime": int(start_ms), "endTime": int(end_ms), "limit": min(1000, limit)}
        if symbol: params["symbol"] = symbol
        try:
            code, success, error = _perp_sign_get(path, params=params)
            tried.append({"path": path, "code": code, "error": error})
            if error or code != 200 or success is None:
                continue
            data = _unwrap_result(success)
            items = data if isinstance(data, list) else _unwrap_result(data)
            if isinstance(items, dict):
                items = items.get("items") or items.get("list") or []
            if isinstance(items, list) and items:
                parsed = []
                for it in items[:limit]:
                    p = _parse_fill(it)
                    if p: parsed.append(p)
                if parsed:
                    print(f"✅ REST fills via {path} parsed={len(parsed)}")
                    return parsed
        except Exception as e:
            tried.append({"path": path, "code": "EXC", "error": str(e)})
            continue
    print("❌ No REST fills endpoint answered. Tries:", json.dumps(tried, indent=2, ensure_ascii=False))
    return []


# Debugs nuevos: listar métodos del SDK y ver fills vía SDK
def debug_list_perp_methods():
    cli = _get_perp()
    names = [n for n in dir(cli) if not n.startswith("_")]
    print("Perp() public methods:", len(names))
    for n in sorted(names):
        print("-", n)

def debug_raw_xt_fills_sdk(days: int = DEFAULT_DAYS_TRADES, symbol: Optional[str] = None, limit: int = 200):
    end_ms = utc_now_ms(); start_ms = end_ms - int(days * 86400000)
    parsed, meta = _fetch_user_fills_via_sdk_methods(start_ms, end_ms, symbol, limit)
    if parsed:
        print("=== SAMPLE RAW (first 5) from SDK ===")
        print(json.dumps(meta.get("sample"), indent=2, ensure_ascii=False))
        print("=== PARSED (first 10) ===")
        print(json.dumps(parsed[:10], indent=2, ensure_ascii=False))
        print(f"total parsed: {len(parsed)}")
    else:
        print("No parsed fills from SDK.")


def _fifo_blocks_from_fills(fills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Construye bloques [open..close] por símbolo real FIFO: cada vez que el neto vuelve a 0.
    Calcula:
      - side por primer fill del bloque
      - entry_avg, close_avg
      - size = máximo neto absoluto dentro del bloque
      - price_pnl FIFO real
      - fee_total (siempre negativo)
      - funding_total = 0.0 (se puede inyectar aparte)
    """
    by_sym: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in fills or []:
        if isinstance(f, dict) and f.get("symbol"):
            by_sym[f["symbol"]].append(f)
    for s in by_sym:
        by_sym[s].sort(key=lambda x: x["ts"])

    results: List[Dict[str, Any]] = []
    for sym, arr in by_sym.items():
        inv = 0.0
        lot_long: deque[Tuple[float, float]] = deque()
        lot_short: deque[Tuple[float, float]] = deque()
        i = 0
        while i < len(arr):
            net0 = inv
            j = i
            fees_sum = 0.0
            open_time = arr[i]["ts"]
            max_abs_net = abs(inv)
            first_side = arr[i]["side"]
            # recorrer hasta que neto vuelva a net0
            while j < len(arr):
                f = arr[j]; qty = f["qty"]; price = f["price"]; side = f["side"]
                fees_sum += f.get("fee", 0.0)
                if side == "BUY":
                    inv += qty
                    lot_long.append([qty, price])
                    # si había lotes cortos abiertos, se cierran contra BUY (reduce short)
                    while qty > 0 and lot_short:
                        sqty, sprice = lot_short[0]
                        take = min(qty, sqty)
                        qty -= take
                        sqty -= take
                        # PnL de precio para cerrar short: (entry(short)-close(buy)) * qty
                        # pero guardamos para calculo final de bloque, aquí solo vaciamos lotes
                        if sqty <= 0: lot_short.popleft()
                        else: lot_short[0][0] = sqty
                else:  # SELL
                    inv -= qty
                    lot_short.append([qty, price])
                    while qty > 0 and lot_long:
                        lqty, lprice = lot_long[0]
                        take = min(qty, lqty)
                        qty -= take
                        lqty -= take
                        if lqty <= 0: lot_long.popleft()
                        else: lot_long[0][0] = lqty

                max_abs_net = max(max_abs_net, abs(inv))
                j += 1
                if inv == net0:
                    break

            if inv != net0:
                # no hay bloque cerrado desde i -> fin
                break

            close_time = arr[j-1]["ts"] if j-1 >= i else arr[i]["ts"]

            # Calcular PnL de precio FIFO real
            # reconstruimos mini-proceso: apilamos lotes según 'primer side'
            block = arr[i:j]
            # reproducir cola FIFO para PnL
            price_pnl = 0.0
            q_long: deque[Tuple[float, float]] = deque()
            q_short: deque[Tuple[float, float]] = deque()
            for f in block:
                qty = f["qty"]; price = f["price"]; side = f["side"]
                if side == "BUY":
                    # cierra short si existe
                    while qty > 0 and q_short:
                        sqty, sprice = q_short[0]
                        take = min(qty, sqty)
                        qty -= take; sqty -= take
                        # short abierto en sprice, se cierra comprando a price:
                        price_pnl += (sprice - price) * take  # (entry - close) * size (short)
                        if sqty <= 0: q_short.popleft()
                        else: q_short[0] = (sqty, sprice)
                    if qty > 0:
                        q_long.append((qty, price))
                else:
                    while qty > 0 and q_long:
                        lqty, lprice = q_long[0]
                        take = min(qty, lqty)
                        qty -= take; lqty -= take
                        # long abierto en lprice, se cierra vendiendo a price:
                        price_pnl += (price - lprice) * take  # (close - entry) * size (long)
                        if lqty <= 0: q_long.popleft()
                        else: q_long[0] = (lqty, lprice)
                    if qty > 0:
                        q_short.append((qty, price))

            # entry_avg y close_avg (simple, ponderados por qty según dirección inicial del bloque)
            entry_sum = close_sum = 0.0
            entry_qty = close_qty = 0.0
            for f in block:
                if (first_side == "BUY" and f["side"] == "BUY") or (first_side != "BUY" and f["side"] == "SELL"):
                    entry_sum += f["price"] * f["qty"]; entry_qty += f["qty"]
                else:
                    close_sum += f["price"] * f["qty"]; close_qty += f["qty"]
            entry_avg = (entry_sum / entry_qty) if entry_qty > 0 else 0.0
            close_avg = (close_sum / close_qty) if close_qty > 0 else entry_avg

            side_txt = "long" if first_side == "BUY" else "short"
            fee_total = -abs(fees_sum)  # tu contrato exige fees siempre negativo
            funding_total = 0.0         # se puede inyectar por bloque con otra pasada si quieres
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

            i = j + 1  # siguiente bloque
    return results

def save_xt_closed_positions(db_path: str = "portfolio.db",
                             days: int = DEFAULT_DAYS_TRADES,
                             symbol: Optional[str] = None,
                             with_funding: bool = False,
                             limit_fills: int = 5000) -> None:
    """
    Reconstruye FIFO y guarda en SQLite usando save_closed_position().
    Respeta tus cálculos: pnl, pnl_percent, apr los remata save_closed_position.
    """
    if save_closed_position is None:
        print("⚠️ db_manager.save_closed_position no disponible; solo preview.")
        return

    p_closed_sync_start(EXCHANGE)
    end_ms = utc_now_ms()
    start_ms = end_ms - int(days * 86400000)

    fills = _discover_user_fills(start_ms, end_ms, symbol, limit_fills)
    if not fills:
        p_closed_sync_none(EXCHANGE)
        return

    funding = []
    if with_funding:
        # funding aproximado en la ventana (si quieres exactitud: por bloque, [open,close])
        try:
            f = fetch_xt_funding_fees(limit=1000, start_ms=start_ms, end_ms=end_ms,
                                      symbol=(symbol if symbol else None))
            # normalizado: symbol base en mayúsculas
            funding = [{"symbol": normalize_symbol(x.get("symbol","")), "timestamp": int(x.get("timestamp",0)),
                        "income": to_float(x.get("income",0.0))} for x in (f or [])]
        except Exception as e:
            print(f"⚠️ funding fetch error: {e}")

    blocks = _fifo_blocks_from_fills(fills)

    saved = 0
    for row in blocks:
        try:
            # save_closed_position: delega validaciones y métricas (apr, pnl_percent, leverage...)
            save_closed_position(db_path, {
                "exchange": row["exchange"],
                "symbol": row["symbol"],
                "side": row["side"],
                "size": row["size"],
                "entry_price": row["entry_price"],
                "close_price": row["close_price"],
                "open_time": row["open_time"],
                "close_time": row["close_time"],
                "realized_pnl": row["realized_pnl"],
                "funding_total": row["funding_total"],
                "fee_total": row["fee_total"],
                "pnl": row["pnl"],
                "notional": row["notional"],
                "leverage": row["leverage"],
                "initial_margin": row["initial_margin"],
                "liquidation_price": row["liquidation_price"],
            })
            saved += 1
            p_closed_sync_saved(EXCHANGE)
        except Exception as e:
            print(f"❌ save_closed_position error: {e}\nrow={json.dumps(row, ensure_ascii=False)}")

    if saved:
        p_closed_sync_done(EXCHANGE)
    else:
        p_closed_sync_none(EXCHANGE)

# ===================== DEBUGS (CLI) =====================

def debug_raw_fapi_balances():
    cli = _get_perp()
    code, success, error = cli.get_account_capital()
    print("=== RAW FUTURES /balance/list ===")
    print("code:", code, "error:", error)
    print(json.dumps(success, indent=2, ensure_ascii=False))

def debug_raw_spot_balances(currencies: Optional[str] = None):
    sp = _get_spot()
    if not sp:
        print("Spot SDK no disponible")
        return
    cur_list = [c.strip() for c in (currencies or "").split(",") if c.strip()] if currencies else None
    res = sp.balances(currencies=cur_list) if cur_list else sp.balances()
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
        "limit": limit, "direction": "NEXT",
        "startTime": int(start_ms), "endTime": int(end_ms),
    }
    if symbol: params["symbol"] = symbol
    path = "/future/user" + "/v1/balance/funding-rate-list"
    header = cli._create_sign(XT_API_KEY, XT_API_SECRET, path=path,
                              bodymod="application/x-www-form-urlencoded", params=params)
    header["Content-Type"] = "application/x-www-form-urlencoded"
    url = cli.host + path
    code, success, error = cli._fetch(method="GET", url=url, headers=header, params=params, timeout=cli.timeout)
    print("=== RAW FUTURES /balance/funding-rate-list ===")
    print("code:", code, "error:", error)
    print(json.dumps(success, indent=2, ensure_ascii=False))

def debug_dump_xt_opens() -> None:
    print("=== XT OPEN POSITIONS (normalized) ===")
    try:
        opens = fetch_xt_open_positions()
        print(json.dumps(opens, indent=2, ensure_ascii=False))
        tot = len(opens)
        notional = sum(to_float(x.get("notional", 0.0)) for x in opens)
        upnl = sum(to_float(x.get("unrealized_pnl", 0.0)) for x in opens)
        by_sym = defaultdict(int)
        for x in opens: by_sym[x["symbol"]] += 1
        print(f"\n--- summary ---\ncount={tot}  notional={notional:.4f}  uPnL={upnl:.4f}")
        print("per symbol:", dict(by_sym))
    except Exception as e:
        print(f"❌ fetch_xt_open_positions error: {e}")

def debug_dump_xt_funding(limit: int = 100, days: int = 14, symbol: Optional[str] = None) -> None:
    print("=== XT FUNDING FEES (normalized) ===")
    end_ms = utc_now_ms(); start_ms = end_ms - int(days * 86400000)
    try:
        funding = fetch_xt_funding_fees(limit=limit, start_ms=start_ms, end_ms=end_ms, symbol=symbol)
        print(json.dumps(funding, indent=2, ensure_ascii=False))
        by_sym_sum = defaultdict(float)
        for it in funding: by_sym_sum[it["symbol"]] += to_float(it.get("income", 0.0))
        print("\n--- summary ---")
        for s, v in by_sym_sum.items(): print(f"{s}: {v:.8f}")
        print(f"total items: {len(funding)}")
    except Exception as e:
        print(f"❌ fetch_xt_funding_fees error: {e}")

def debug_raw_xt_fills(days: int = DEFAULT_DAYS_TRADES, symbol: Optional[str] = None, limit: int = 500):
    print("=== RAW FILLS (endpoint discovery) ===")
    end_ms = utc_now_ms(); start_ms = end_ms - int(days * 86400000)
    tried = []
    for path in FILLS_CANDIDATES:
        params = { "startTime": int(start_ms), "endTime": int(end_ms), "limit": min(1000, limit) }
        if symbol: params["symbol"] = symbol
        try:
            code, success, error = _perp_sign_get(path, params=params)
            tried.append({"path": path, "code": code, "error": error})
            if error or code != 200 or success is None:
                continue
            data = _unwrap_result(success)
            items = data if isinstance(data, list) else _unwrap_result(data)
            if isinstance(items, dict):
                items = items.get("items") or items.get("list") or []
            print(f"\n✅ {path} → {len(items or [])} items")
            print(json.dumps((items or [])[:10], indent=2, ensure_ascii=False))
            return
        except Exception as e:
            tried.append({"path": path, "code": "EXC", "error": str(e)})
            continue
    print("❌ Ningún endpoint devolvió datos. Intentos:\n", json.dumps(tried, indent=2, ensure_ascii=False))

def debug_preview_xt_closed(days: int = DEFAULT_DAYS_TRADES,
                            symbol: Optional[str] = None,
                            limit: int = 5000,
                            with_funding: bool = False) -> None:
    print("=== XT CLOSED PREVIEW (FIFO) ===")
    end_ms = utc_now_ms(); start_ms = end_ms - int(days * 86400000)
    fills = _discover_user_fills(start_ms, end_ms, symbol, limit)
    if not fills:
        print("No hay fills → no se puede reconstruir.")
        return
    blocks = _fifo_blocks_from_fills(fills)
    print(json.dumps(blocks, indent=2, ensure_ascii=False))
    print(f"\n--- summary ---\nclosed_blocks={len(blocks)}")
    if with_funding and blocks:
        # funding por ventana (aprox, por símbolo)
        sym_set = sorted(set(b["symbol"] for b in blocks))
        end_ms = utc_now_ms(); start_ms = end_ms - int(days * 86400000)
        fund_map = {s: 0.0 for s in sym_set}
        for s in sym_set:
            try:
                raw = fetch_xt_funding_fees(limit=1000, start_ms=start_ms, end_ms=end_ms, symbol=f"{s.lower()}_usdt")
                fund_map[s] = sum(to_float(x.get("income", 0.0)) for x in raw if normalize_symbol(x.get("symbol")) == s)
            except Exception as e:
                print(f"funding fetch error for {s}: {e}")
        print("\nFunding aprox por símbolo en ventana:")
        print(json.dumps(fund_map, indent=2, ensure_ascii=False))

# ===================== __all__ =====================
__all__ = [
    "fetch_xt_open_positions",
    "fetch_xt_funding_fees",
    "fetch_xt_all_balances",
    "save_xt_closed_positions",
    # debugs útiles desde otros módulos
    "debug_preview_xt_closed",
    "debug_dump_xt_opens",
    "debug_dump_xt_funding",
    "debug_raw_xt_fills",
    "debug_raw_fapi_positions",
    "debug_raw_fapi_funding",
    "debug_raw_fapi_balances",
    "debug_raw_spot_balances",
]

# ===================== CLI =====================
if __name__ == "__main__":
    ap = argparse.ArgumentParser("XT adapter (pyxt) — Spot+Futures + FIFO cerradas")
    ap.add_argument("--balance", action="store_true", help="Saldo combinado normalizado (/api/balances)")
    ap.add_argument("--positions", action="store_true", help="Posiciones abiertas normalizadas")
    ap.add_argument("--funding", type=int, default=0, help="N funding fees normalizados (0=omit)")
    ap.add_argument("--raw-balance", action="store_true", help="RAW FUTURES balance/list")
    ap.add_argument("--raw-spot", action="store_true", help="RAW SPOT balances")
    ap.add_argument("--raw-positions", action="store_true", help="RAW FUTURES position/list")
    ap.add_argument("--raw-funding", action="store_true", help="RAW FUTURES funding-rate-list")
    ap.add_argument("--raw-fills", action="store_true", help="RAW discovery de fills (muestra 10)")
    ap.add_argument("--debug-opens", action="store_true", help="Dump de posiciones abiertas NORMALIZADAS")
    ap.add_argument("--debug-funding", type=int, default=0, help="Dump funding NORMALIZADO (N items)")
    ap.add_argument("--debug-closed", action="store_true", help="Preview FIFO de posiciones CERRADAS (sin guardar)")
    ap.add_argument("--save-closed", action="store_true", help="Reconstruye FIFO y guarda en SQLite")
    ap.add_argument("--db", type=str, default="portfolio.db")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS_TRADES)
    ap.add_argument("--symbol", type=str, default=None, help="Ej: btc_usdt (se normaliza en salida)")
    ap.add_argument("--list-perp-methods", action="store_true", help="Lista métodos públicos del SDK Perp")
    ap.add_argument("--raw-fills-sdk", action="store_true", help="Muestra fills vía SDK (si el SDK los expone)")

    args = ap.parse_args()

    print("🚀 XT adapter runner")
    print(f"• FAPI_HOST={FAPI_HOST}")
    print(f"• SAPI_HOST={SAPI_HOST}")
    print(f"• API_KEY set: {'yes' if os.getenv('XT_API_KEY') else 'no'}")
    print(f"• Flags: {sys.argv[1:]}")

    try:
        if args.raw_balance: debug_raw_fapi_balances()
        if args.raw_spot:    debug_raw_spot_balances()
        if args.raw_positions: debug_raw_fapi_positions()
        if args.raw_funding: debug_raw_fapi_funding(days=args.days, symbol=args.symbol, limit=50)
        if args.raw_fills:   debug_raw_xt_fills(days=args.days, symbol=args.symbol, limit=500)
        if args.list_perp_methods:
            debug_list_perp_methods()
        
        if args.raw_fills_sdk:
            debug_raw_xt_fills_sdk(days=args.days, symbol=args.symbol, limit=500)

        if args.balance:
            print(json.dumps(fetch_xt_all_balances() or {}, indent=2, ensure_ascii=False))

        if args.positions or args.debug_opens:
            debug_dump_xt_opens()

        if args.funding > 0 or args.debug_funding > 0:
            n = max(args.funding, args.debug_funding)
            debug_dump_xt_funding(limit=n, days=args.days, symbol=args.symbol)

        if args.debug_closed:
            debug_preview_xt_closed(days=args.days, symbol=args.symbol, limit=5000, with_funding=False)

        if args.save_closed:
            save_xt_closed_positions(db_path=args.db, days=args.days, symbol=args.symbol,
                                     with_funding=False, limit_fills=5000)

        if not any([
            args.raw_balance, args.raw_spot, args.raw_positions, args.raw_funding, args.raw_fills,
            args.balance, args.positions, args.funding, args.debug_opens, args.debug_funding,
            args.debug_closed, args.save_closed
        ]):
            # Autotest rápido: abre + funding 10 + preview cerradas
            debug_dump_xt_opens()
            debug_dump_xt_funding(limit=10, days=min(args.days, 14), symbol=args.symbol)
            debug_preview_xt_closed(days=min(args.days, 7), symbol=args.symbol, limit=1000, with_funding=False)

    except Exception:
        traceback.print_exc()
