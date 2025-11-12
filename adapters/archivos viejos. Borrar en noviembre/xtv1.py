# adapters/xt.py
# XT.COM Perp adapter (HTTP) — firma según doc X+Y, endpoints balance/positions/funding
from __future__ import annotations

import os, json, time as _time, hmac, hashlib, traceback, argparse
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
import requests

# --- Integración con tu proyecto ---
try:
    from symbols import normalize_symbol  # normalizador del proyecto
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

try:
    from time import utc_now_ms
except Exception:
    def utc_now_ms() -> int: return int(_time.time() * 1000)

try:
    from money import to_float
except Exception:
    def to_float(x) -> float:
        try: return float(x)
        except Exception: return 0.0

try:
    from db_manager import save_closed_position  # para futuras cerradas por FIFO
except Exception:
    save_closed_position = None

# --- Config ---
EXCHANGE = "xt"
XT_PERP_HOST = os.getenv("XT_PERP_HOST", "https://sapi.xt.com")
XT_API_KEY = "96b1c999-b9df-42ed-9d4f-a4491aac7a1d"
XT_API_SECRET = "5f60a0a147d82db2117a45da129a6f4480488234"

DEFAULT_DAYS_TRADES = int(os.getenv("XT_TRADES_DAYS", "30"))

# ------------ Firma EXACTA (según texto que pasaste) ------------
# X = "validate-appkey=<>&validate-timestamp=<>&[validate-recvwindow=<>]&[validate-algorithms=HmacSHA256]"
# Y = "#<path>[#<query>][#<body>]"   (query: claves ordenadas k=v unidas por &; body: JSON sin ordenar)
# sign = HMAC_SHA256(secret, X + Y)  -> header: validate-signature: <hex>
def _now_ms() -> int:
    return int(_time.time() * 1000)

def _canon_query(params: Optional[Dict[str, Any]]) -> str:
    if not params: return ""
    items = sorted((k, v) for k, v in params.items() if v is not None)
    return "&".join(f"{k}={v}" for k, v in items)

def _json_or_str(body: Any) -> str:
    if body is None or body == "":
        return ""
    if isinstance(body, str):
        return body
    # JSON string sin ordenar claves
    return json.dumps(body, separators=(',', ':'), ensure_ascii=False)

def _build_headers_signed(path: str,
                          method: str = "GET",
                          query: Optional[Dict[str, Any]] = None,
                          body: Any = None,
                          recv_window_ms: int = 5000) -> Dict[str, str]:
    ts = _now_ms()
    alg = "HmacSHA256"

    # ---- X ----
    X_parts = [
        f"validate-appkey={XT_API_KEY}",
        f"validate-timestamp={ts}",
    ]
    if recv_window_ms:
        X_parts.append(f"validate-recvwindow={recv_window_ms}")
    X_parts.append(f"validate-algorithms={alg}")  # opcional, lo incluimos

    X = "&".join(X_parts)

    # ---- Y ----
    q = _canon_query(query)
    body_str = _json_or_str(body) if method.upper() != "GET" else ""
    if q and body_str:
        Y = f"#{path}#{q}#{body_str}"
    elif q:
        Y = f"#{path}#{q}"
    elif body_str:
        Y = f"#{path}#{body_str}"
    else:
        Y = f"#{path}"

    sign_payload = f"{X}{Y}"
    sig = hmac.new((XT_API_SECRET or "").encode("utf-8"),
                   sign_payload.encode("utf-8"), hashlib.sha256).hexdigest()

    headers = {
        "validate-appkey": XT_API_KEY,
        "validate-timestamp": str(ts),
        "validate-algorithms": alg,
        "validate-recvwindow": str(recv_window_ms),
        "validate-signature": sig,
        # compat (algunos servidores aceptan el prefijo xt-*)
        "xt-validate-appkey": XT_API_KEY,
        "xt-validate-timestamp": str(ts),
        "xt-validate-algorithms": alg,
        "xt-validate-recvwindow": str(recv_window_ms),
        "xt-validate-signature": sig,
        "validate-signversion": "2",
        "xt-validate-signversion": "2",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    return headers

def _unwrap_xt(resp_json: Any) -> Any:
    if not isinstance(resp_json, dict):
        return resp_json
    # doc muestra {"result":[...], "returnCode":0}
    if "result" in resp_json:
        return resp_json["result"]
    if "data" in resp_json:
        return resp_json["data"]
    return resp_json

def _http_xt(method: str, path: str,
             params: Optional[Dict[str, Any]] = None,
             body: Any = None,
             timeout: int = 20) -> Any:
    base = XT_PERP_HOST.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    headers = _build_headers_signed(path=path, method=method, query=params, body=body)
    url = f"{base}{path}"
    if method.upper() == "GET":
        r = requests.get(url, headers=headers, params=params or {}, timeout=timeout)
    elif method.upper() == "POST":
        r = requests.post(url, headers=headers, params=params or {}, data=_json_or_str(body), timeout=timeout)
    else:
        raise ValueError("Unsupported HTTP method")
    r.raise_for_status()
    return r.json()

# ----------------- FUNDING FEES -----------------
def fetch_xt_funding_fees(limit: int = 50,
                          start_ms: Optional[int] = None,
                          end_ms: Optional[int] = None,
                          symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Normalización EXACTA para /api/funding:
    {
      "exchange":"xt","symbol":"<NORMALIZADO>","income":float(+/-),
      "asset":"USDT|USDC|USD","timestamp":int(ms),"funding_rate":0.0,"type":"FUNDING_FEE"
    }
    """
    if end_ms is None:
        end_ms = utc_now_ms()
    if start_ms is None:
        start_ms = end_ms - 14*24*60*60*1000  # 14 días por defecto

    path = "/future/user/v1/balance/funding-rate-list"
    out: List[Dict[str, Any]] = []
    next_id: Optional[int] = None
    direction = "NEXT"

    while len(out) < limit:
        page_size = min(100, max(1, limit - len(out)))
        params = {
            "limit": page_size,
            "direction": direction,
            "startTime": int(start_ms),
            "endTime": int(end_ms),
        }
        if symbol:
            params["symbol"] = symbol  # XT espera 'btc_usdt' etc. (igual normalizamos luego)
        if next_id is not None:
            params["id"] = int(next_id)

        data = _http_xt("GET", path, params=params)
        res = _unwrap_xt(data)
        # res esperado: {"hasNext":bool, "hasPrev":bool, "items":[...]}
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
            ts = ts if ts > 10**12 else ts * 1000  # ms
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

    return out

# ----------------- BALANCES -----------------
def fetch_xt_all_balances(db_path: str = "portfolio.db") -> Dict[str, Any] | None:
    """
    Forma EXACTA /api/balances:
    {"exchange":"xt","equity":float,"balance":float,"unrealized_pnl":float,
     "initial_margin":0.0,"spot":0.0,"margin":0.0,"futures":float}
    """
    # prefer compat -> más campos (amount/totalAmount/notProfit)
    path1 = "/future/user/v1/compat/balance/list"
    path2 = "/future/user/v1/balance/list"
    try:
        data = _http_xt("GET", path1)
    except Exception:
        data = _http_xt("GET", path2)

    res = _unwrap_xt(data)
    # res esperado: list[ dict{ walletBalance, notProfit, totalAmount/amount } ]
    equity = balance = upnl = 0.0
    if isinstance(res, list):
        for it in res:
            if not isinstance(it, dict): continue
            # equity: totalAmount (margin balance) o amount (net) si no viene total
            e = to_float(it.get("totalAmount") or it.get("amount") or 0.0)
            b = to_float(it.get("walletBalance") or 0.0)
            u = to_float(it.get("notProfit") or 0.0)
            equity += e if e > 0 else (b + u)
            balance += b
            upnl += u
    elif isinstance(res, dict):
        # algunos devuelven en {"list":[...]}
        arr = res.get("list") or res.get("items") or []
        if isinstance(arr, list):
            for it in arr:
                if not isinstance(it, dict): continue
                e = to_float(it.get("totalAmount") or it.get("amount") or 0.0)
                b = to_float(it.get("walletBalance") or 0.0)
                u = to_float(it.get("notProfit") or 0.0)
                equity += e if e > 0 else (b + u)
                balance += b
                upnl += u

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
    return obj

# ----------------- OPEN POSITIONS -----------------
def fetch_xt_open_positions() -> List[Dict[str, Any]]:
    """
    Forma EXACTA /api/positions (tu front):
    {
      "exchange":"xt","symbol":"<NORMALIZADO>","side":"long|short","size":float,
      "entry_price":float,"mark_price":float,"liquidation_price":float|0.0,
      "notional":float,"unrealized_pnl":float,"fee":float(negativo),"funding_fee":float,"realized_pnl":float
    }
    """
    # prefer /position (active). Fallback a /position/list si el primero no existe
    try:
        data = _http_xt("GET", "/future/user/v1/position")
    except Exception:
        data = _http_xt("GET", "/future/user/v1/position/list")

    arr = _unwrap_xt(data)
    if not isinstance(arr, list):
        return []

    out: List[Dict[str, Any]] = []
    for p in arr:
        if not isinstance(p, dict):
            continue
        sym_raw = str(p.get("symbol") or "")
        base = normalize_symbol(sym_raw)
        side0 = (p.get("positionSide") or "").upper()
        side = "long" if side0 in ("LONG", "BID", "BUY") else "short"
        size = abs(to_float(p.get("positionSize") or p.get("availableCloseSize") or 0.0))
        entry = to_float(p.get("entryPrice") or 0.0)
        mark  = to_float(p.get("calMarkPrice") or p.get("markPrice") or entry)
        liq   = to_float(p.get("breakPrice") or 0.0)  # doc: Blowout price
        upnl  = to_float(p.get("floatingPL") or 0.0)  # ya viene calculado por el exchange
        # Este endpoint no trae fee/funding acumulado por posición activa:
        fee_acc   = 0.0
        funding   = 0.0
        realized  = fee_acc + funding

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
            "fee": float(-abs(fee_acc)),  # contrato: fee acumulado NEGATIVO si es costo
            "funding_fee": float(funding),
            "realized_pnl": float(realized),
        })
    return out

# ----------------- CLOSED POSITIONS (stub) -----------------
def save_xt_closed_positions(db_path: str = "portfolio.db",
                             days: int = DEFAULT_DAYS_TRADES,
                             debug: bool = False) -> None:
    """
    XT (doc adjunta) NO expone endpoint de 'posiciones cerradas'. Para guardar closed_positions
    en tu SQLite hay que reconstruirlas con FIFO a partir de 'user trades/fills'.
    Cuando compartas el endpoint HTTP de 'fills' (p.ej. /future/user/v1/trade/list),
    implemento el cálculo FIFO y llamaré save_closed_position(row) por cada bloque.
    """
    msg = ("XT closed positions no disponible por HTTP en la doc pasada — "
           "necesito el endpoint de 'user trades/fills' para hacer FIFO real y guardar en DB.")
    print(f"ℹ️ {msg}")
    if debug:
        traceback.print_stack()

# ----------------- DEBUGS (RAW) -----------------
def debug_raw_xt_balances():
    try:
        print("🧾 RAW /compat/balance/list")
        print(json.dumps(_http_xt("GET", "/future/user/v1/compat/balance/list"), indent=2))
    except Exception as e:
        print(f"compat/balance/list error: {e}")
    try:
        print("🧾 RAW /balance/list")
        print(json.dumps(_http_xt("GET", "/future/user/v1/balance/list"), indent=2))
    except Exception as e:
        print(f"balance/list error: {e}")

def debug_raw_xt_positions():
    try:
        print("🧾 RAW /position")
        print(json.dumps(_http_xt("GET", "/future/user/v1/position"), indent=2))
    except Exception as e:
        print(f"position error: {e}")
    try:
        print("🧾 RAW /position/list")
        print(json.dumps(_http_xt("GET", "/future/user/v1/position/list"), indent=2))
    except Exception as e:
        print(f"position/list error: {e}")

def debug_raw_xt_funding(days: int = 14, symbol: Optional[str] = None, page: int = 1, page_limit: int = 100):
    end_ms = utc_now_ms()
    start_ms = end_ms - int(days * 86400000)
    params = {
        "limit": min(100, page_limit),
        "direction": "NEXT",
        "startTime": int(start_ms),
        "endTime": int(end_ms),
    }
    if symbol:
        params["symbol"] = symbol
    print("🧾 RAW /balance/funding-rate-list")
    print(json.dumps(_http_xt("GET", "/future/user/v1/balance/funding-rate-list", params=params), indent=2))

# ----------------- CLI -----------------
def _autorun():
    print("🚀 XT HTTP adapter quick check")
    try:
        b = fetch_xt_all_balances()
        print("balances(normalized) =", json.dumps(b, indent=2))
    except Exception as e:
        print("balances error:", e)
    try:
        f = fetch_xt_funding_fees(limit=20)
        print("funding(normalized) =", json.dumps(f[:5], indent=2), f"... total={len(f)}")
    except Exception as e:
        print("funding error:", e)
    try:
        p = fetch_xt_open_positions()
        print("positions(normalized) =", json.dumps(p, indent=2))
    except Exception as e:
        print("positions error:", e)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="XT HTTP Adapter (balances / positions / funding)")
    ap.add_argument("--balance", action="store_true", help="Ver balances normalizados")
    ap.add_argument("--positions", action="store_true", help="Ver posiciones abiertas normalizadas")
    ap.add_argument("--funding", type=int, default=0, help="Ver N registros funding normalizados")
    ap.add_argument("--raw-balances", action="store_true")
    ap.add_argument("--raw-positions", action="store_true")
    ap.add_argument("--raw-funding", action="store_true")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--symbol", type=str, default=None)
    ap.add_argument("--save-closed", action="store_true")
    args = ap.parse_args()

    ran = False
    if args.raw_balances:
        debug_raw_xt_balances(); ran = True
    if args.raw_positions:
        debug_raw_xt_positions(); ran = True
    if args.raw_funding:
        debug_raw_xt_funding(days=args.days, symbol=args.symbol); ran = True
    if args.balance:
        print(json.dumps(fetch_xt_all_balances() or {}, indent=2)); ran = True
    if args.positions:
        print(json.dumps(fetch_xt_open_positions(), indent=2)); ran = True
    if args.funding:
        print(json.dumps(fetch_xt_funding_fees(limit=args.funding, start_ms=None, end_ms=None, symbol=args.symbol), indent=2)); ran = True
    if args.save_closed:
        save_xt_closed_positions(debug=True); ran = True
    if not ran:
        _autorun()

__all__ = [
    "fetch_xt_open_positions",
    "fetch_xt_funding_fees",
    "fetch_xt_all_balances",
    "save_xt_closed_positions",
    "debug_raw_xt_balances",
    "debug_raw_xt_positions",
    "debug_raw_xt_funding",
]

