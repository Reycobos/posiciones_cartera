# adapters/xt.py
# XT Futures adapter (FAPI) usando el SDK oficial pyxt (perp.py).
from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List, Optional

# --- Intenta importar el SDK instalado; si no, usa los archivos locales que me pasaste ---
try:
    from pyxt.perp import Perp  # pip install pyxt
except Exception:
    # fallback a los ficheros locales en tu repo/proyecto
    from perp import Perp  # <-- tu /perp.py subido

# ---- utilidades del proyecto ----
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

# ---- helpers de logging (NO-OP si no existen en tu backend) ----
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

# ---- Configuración ----
EXCHANGE = "xt"
FAPI_HOST = os.getenv("XT_FAPI_HOST", "https://fapi.xt.com")
XT_API_KEY = "96b1c999-b9df-42ed-9d4f-a4491aac7a1d"
XT_API_SECRET = "5f60a0a147d82db2117a45da129a6f4480488234"
DEFAULT_DAYS_TRADES = int(os.getenv("XT_TRADES_DAYS", "30"))

# =====================================================================
#                     CLIENTE pyxt (perp) + wrappers
# =====================================================================

_client: Optional[Perp] = None

def _get_client() -> Perp:
    global _client
    if _client is None:
        if not XT_API_KEY or not XT_API_SECRET:
            raise RuntimeError("Faltan XT_API_KEY / XT_API_SECRET en el entorno.")
        _client = Perp(host=FAPI_HOST, access_key=XT_API_KEY, secret_key=XT_API_SECRET, timeout=15)
    return _client

def _unwrap_result(obj: Any) -> Any:
    """
    El SDK pyxt normalmente retorna (code, success, error) y success suele ser:
      {'returnCode':0, 'result': ...} ó {'result': ...}
    """
    if isinstance(obj, dict):
        if "result" in obj: return obj["result"]
        if "data" in obj: return obj["data"]
        return obj
    return obj

# =====================================================================
#                             BALANCES
# Endpoint: GET /future/user/v1/balance/list
# =====================================================================

def fetch_xt_all_balances(db_path: str = "portfolio.db") -> Dict[str, Any] | None:
    """
    Forma EXACTA /api/balances:
    {
      "exchange":"xt","equity":float,"balance":float,"unrealized_pnl":float,
      "initial_margin":0.0,"spot":0.0,"margin":0.0,"futures":float
    }
    """
    p_balance_fetching(EXCHANGE)
    cli = _get_client()

    # Usamos el método del SDK ya incluido en tu perp.py: get_account_capital()
    code, success, error = cli.get_account_capital()
    if error or code != 200 or success is None:
        raise RuntimeError(f"XT balances error: {error or code}")

    res = _unwrap_result(success)
    equity = balance = upnl = 0.0

    # Según tu doc: result es una lista de assets con walletBalance, etc.
    if isinstance(res, list):
        for it in res:
            if not isinstance(it, dict): 
                continue
            wb = to_float(it.get("walletBalance") or 0.0)
            # Este endpoint (v1/balance/list) no trae notProfit; dejamos uPNL = 0.0
            equity += wb
            balance += wb
    elif isinstance(res, dict):
        arr = res.get("items") or res.get("list") or []
        if isinstance(arr, list):
            for it in arr:
                if not isinstance(it, dict): continue
                wb = to_float(it.get("walletBalance") or 0.0)
                equity += wb
                balance += wb

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
    p_balance_done(EXCHANGE)
    return obj

# =====================================================================
#                           FUNDING FEES
# Endpoint: GET /future/user/v1/balance/funding-rate-list
# =====================================================================

def fetch_xt_funding_fees(
    limit: int = 50,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    symbol: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Shape EXACTO /api/funding:
    {
      "exchange":"xt","symbol":"<NORMALIZADO>","income":float,
      "asset":"USDT|USDC|USD","timestamp":int(ms),"funding_rate":0.0,"type":"FUNDING_FEE"
    }
    """
    p_funding_fetching(EXCHANGE)
    cli = _get_client()

    if end_ms is None:
        end_ms = utc_now_ms()
    if start_ms is None:
        start_ms = end_ms - 14 * 24 * 60 * 60 * 1000

    out: List[Dict[str, Any]] = []
    next_id: Optional[int] = None
    direction = "NEXT"

    # No existe método en tu perp.py para este endpoint; llamamos a _fetch con firma propia del SDK
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

        # Firmamos igual que el SDK: x-www-form-urlencoded cuando hay query/params
        bodymod = "application/x-www-form-urlencoded"
        header = cli._create_sign(XT_API_KEY, XT_API_SECRET, path=path, bodymod=bodymod, params=params)
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

# =====================================================================
#                         OPEN / ACTIVE POSITIONS
# Endpoint preferido: GET /future/user/v1/position/list
# =====================================================================

def fetch_xt_open_positions() -> List[Dict[str, Any]]:
    """
    Forma EXACTA (lo que consume /api/positions):
    {
      "exchange":"xt","symbol":"<NORMALIZADO>","side":"long|short","size":float,
      "entry_price":float,"mark_price":float,"liquidation_price":float|0.0,
      "notional":float,"unrealized_pnl":float,"fee":float(negativo),"funding_fee":float,"realized_pnl":float
    }
    """
    p_open_fetching(EXCHANGE)
    cli = _get_client()

    # El SDK trae Perp.get_position(symbol), pero queremos "todas":
    # llamamos igual que get_position pero sin symbol => lista completa
    path = "/future/user" + "/v1/position/list"
    params: Dict[str, Any] = {}  # si quieres filtrar: {"symbol":"btc_usdt"}
    bodymod = "application/x-www-form-urlencoded"
    header = cli._create_sign(XT_API_KEY, XT_API_SECRET, path=path, bodymod=bodymod, params=params)
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
            "fee": float(0.0),          # el endpoint no trae fees acumuladas de la posición activa
            "funding_fee": float(0.0),  # idem
            "realized_pnl": float(0.0),
        })

    p_open_count(EXCHANGE, len(out))
    return out

# =====================================================================
#                   CLOSED POSITIONS (no expuesto por HTTP)
# =====================================================================

def save_xt_closed_positions(db_path: str = "portfolio.db",
                             days: int = DEFAULT_DAYS_TRADES,
                             debug: bool = False) -> None:
    """
    Tu documentación y el SDK expuesto no incluyen un endpoint de 'posiciones cerradas'.
    Para guardar en SQLite (tabla closed_positions) necesito el historial de trades/fills
    (p.ej. /future/trade/v1/user-trades o similar). En cuanto lo tengas, implemento FIFO real.
    """
    msg = ("XT: sin endpoint de fills en la info/SDK actual; "
           "cuando facilites el de 'user fills', armo FIFO y guardo via save_closed_position().")
    print(msg)
    if debug:
        import traceback; traceback.print_stack()

# =====================================================================
#                                DEBUGS
# =====================================================================

def debug_raw_xt_fapi_balances():
    cli = _get_client()
    code, success, error = cli.get_account_capital()
    print("=== RAW /future/user/v1/balance/list ===")
    print("code:", code, "error:", error)
    print(json.dumps(success, indent=2))

def debug_raw_xt_fapi_positions(symbol: Optional[str] = None):
    cli = _get_client()
    if symbol:
        code, success, error = cli.get_position(symbol)
    else:
        # "todas" como en fetch_xt_open_positions
        path = "/future/user" + "/v1/position/list"
        params = {}
        header = cli._create_sign(XT_API_KEY, XT_API_SECRET, path=path,
                                  bodymod="application/x-www-form-urlencoded", params=params)
        header["Content-Type"] = "application/x-www-form-urlencoded"
        url = cli.host + path
        code, success, error = cli._fetch(method="GET", url=url, headers=header, params=params, timeout=cli.timeout)
    print("=== RAW /future/user/v1/position/list ===")
    print("code:", code, "error:", error)
    print(json.dumps(success, indent=2))

def debug_raw_xt_fapi_funding(days: int = 14, symbol: Optional[str] = None, limit: int = 50):
    cli = _get_client()
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
    print("=== RAW /future/user/v1/balance/funding-rate-list ===")
    print("code:", code, "error:", error)
    print(json.dumps(success, indent=2))

# =====================================================================
#                               CLI
# =====================================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("XT adapter (pyxt) — balances / positions / funding")
    ap.add_argument("--balance", action="store_true", help="Balances normalizados (/api/balances)")
    ap.add_argument("--positions", action="store_true", help="Posiciones abiertas normalizadas")
    ap.add_argument("--funding", type=int, default=0, help="N funding fees normalizados")
    ap.add_argument("--raw-balance", action="store_true", help="Dump RAW balance/list")
    ap.add_argument("--raw-positions", action="store_true", help="Dump RAW position/list")
    ap.add_argument("--raw-funding", action="store_true", help="Dump RAW funding-rate-list")
    ap.add_argument("--symbol", type=str, default=None)
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()

    if args.raw_balance:
        debug_raw_xt_fapi_balances()
    if args.raw_positions:
        debug_raw_xt_fapi_positions(args.symbol)
    if args.raw_funding:
        debug_raw_xt_fapi_funding(args.days, args.symbol, max(1, args.funding or 50))
    if args.balance:
        print(json.dumps(fetch_xt_all_balances() or {}, indent=2))
    if args.positions:
        print(json.dumps(fetch_xt_open_positions(), indent=2))
    if args.funding:
        print(json.dumps(fetch_xt_funding_fees(limit=args.funding, symbol=args.symbol), indent=2))
