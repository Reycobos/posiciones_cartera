# adapters/kucoin.py
from __future__ import annotations
import os, time, hmac, hashlib, base64, json, sqlite3
from urllib.parse import urlencode
from typing import Any, Dict, List, Optional

#====== Imports para prints
# from pp import (
#     p_closed_debug_header, p_closed_debug_count, p_closed_debug_norm_size,
#     p_closed_debug_prices, p_closed_debug_pnl, p_closed_debug_times, p_closed_debug_normalized,
#     p_open_summary, p_open_block,
#     p_funding_fetching, p_funding_count,
#     p_balance_equity
# )
#===========================


DB_PATH = "portfolio.db"
import requests

# =========================
# Config y credenciales
# =========================
KUCOIN_BASE_URL      = os.getenv("KUCOIN_BASE_URL", "https://api-futures.kucoin.com")  # Futuros
KUCOIN_SPOT_BASE     = os.getenv("KUCOIN_SPOT_BASE", "https://api.kucoin.com")         # Spot

KUCOIN_API_KEY       = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET    = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE= os.getenv("KUCOIN_API_PASSPHRASE")

# Helper DB (tu función existente)
try:
    from db_manager import save_closed_position
except Exception:
    def save_closed_position(_: Dict[str, Any]):
        raise RuntimeError("db_manager.save_closed_position no disponible")

# =========================
# Helpers generales
# =========================
def _has_creds() -> bool:
    return bool(KUCOIN_API_KEY and KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE)

def _now_ms() -> int:
    return int(time.time() * 1000)

def _normalize_symbol(sym: str) -> str:
    if not sym: return ""
    s = sym.upper().replace("-", "")
    # quita sufijos comunes
    for suf in ("USDT","USDC","USD","PERP","USDTM"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s

def _symbol_no_dash(sym: str) -> str:
    return (sym or "").upper().replace("-", "")

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

# =========================
# Firma y headers KuCoin v3
# =========================
def kucoin_server_timestamp_ms() -> str:
    """Timestamp del servidor (ms) como string; si falla, local."""
    try:
        r = requests.get(f"{KUCOIN_BASE_URL}/api/v1/timestamp", timeout=5)
        r.raise_for_status()
        return str(int(r.json().get("data")))
    except Exception:
        return str(_now_ms())

def _kucoin_sign(timestamp: str, method: str, endpoint: str, secret: str, body: str = "") -> str:
    str_to_sign = f"{timestamp}{method.upper()}{endpoint}{body}"
    signature = hmac.new(secret.encode("utf-8"),
                         str_to_sign.encode("utf-8"),
                         hashlib.sha256).digest()
    return base64.b64encode(signature).decode()

def _kucoin_headers(api_key: str, api_secret: str, passphrase: str,
                    method: str, endpoint_with_query: str, body: str = "") -> dict:
    """
    endpoint_with_query: ej "/api/v1/account-overview?currency=USDT"
    """
    if not (api_key and api_secret and passphrase):
        raise ValueError("Credenciales KuCoin faltantes (API_KEY/SECRET/PASSPHRASE).")

    timestamp = kucoin_server_timestamp_ms()
    sign = _kucoin_sign(timestamp, method, endpoint_with_query, api_secret, body)

    passphrase_signed = base64.b64encode(
        hmac.new(api_secret.encode("utf-8"),
                 passphrase.encode("utf-8"),
                 hashlib.sha256).digest()
    ).decode()

    return {
        "KC-API-TIMESTAMP": timestamp,
        "KC-API-SIGN": sign,
        "KC-API-KEY": api_key,
        "KC-API-PASSPHRASE": passphrase_signed,
        "KC-API-KEY-VERSION": "3",
        "Content-Type": "application/json"
    }

def _get(base: str, endpoint_with_query: str, method: str = "GET", body: Optional[dict] = None) -> dict:
    """
    Llamada GET/POST firmada a KuCoin.
    Para GET, body debe ser None.
    """
    body_str = "" if body is None else json.dumps(body, separators=(",", ":"))
    headers = _kucoin_headers(
        KUCOIN_API_KEY, KUCOIN_API_SECRET, KUCOIN_API_PASSPHRASE,
        method, endpoint_with_query, body_str
    )
    url = f"{base}{endpoint_with_query}"
    if method.upper() == "GET":
        r = requests.get(url, headers=headers, timeout=20)
    else:
        r = requests.post(url, headers=headers, data=body_str, timeout=20)
    r.raise_for_status()
    return r.json() if r.text else {}

# =========================
# Balances (Spot/Margin/Futures separados)
# =========================
def _fetch_futures_account_usdt() -> dict:
    """
    GET /api/v1/account-overview?currency=USDT (Futuros)
    """
    endpoint = "/api/v1/account-overview?currency=USDT"
    resp = _get(KUCOIN_BASE_URL, endpoint, "GET")
    data = resp.get("data", {}) if isinstance(resp, dict) else {}
    return {
        "equity": _safe_float(data.get("accountEquity", 0)),
        "available": _safe_float(data.get("availableBalance", 0)),
        "unrealized_pnl": _safe_float(data.get("unrealisedPNL", 0)),
        "margin_balance": _safe_float(data.get("marginBalance", 0)),
        "initial_margin": _safe_float(data.get("positionMargin", 0)),  # mejor proxy si existe
    }

def _fetch_spot_accounts() -> List[dict]:
    """
    GET /api/v1/accounts (Spot)
    """
    endpoint = "/api/v1/accounts"
    resp = _get(KUCOIN_SPOT_BASE, endpoint, "GET")
    arr = resp.get("data", []) if isinstance(resp, dict) else []
    out = []
    for acct in arr:
        bal = _safe_float(acct.get("balance", 0))
        avail = _safe_float(acct.get("available", 0))
        holds = _safe_float(acct.get("holds", 0))
        if bal > 0 or holds > 0:
            out.append({
                "currency": acct.get("currency"),
                "balance": bal,
                "available": avail,
                "holds": holds,
                "type": acct.get("type"),
            })
    return out

def fetch_kucoin_all_balances() -> Dict[str, Any]:
    """
    Devuelve el dict EXACTO que consume tu UI para un exchange:
    {
      "exchange": "kucoin",
      "equity": float,
      "balance": float,
      "unrealized_pnl": float,
      "initial_margin": float,
      "spot": float,
      "margin": float,
      "futures": float
    }
    """
    if not _has_creds():
        # Evita caer con .encode() cuando las creds están vacías
        print("⚠️ KuCoin: faltan credenciales. Devolviendo ceros para balances.")
        return {
            "exchange": "kucoin",
            "equity": 0.0,
            "balance": 0.0,
            "unrealized_pnl": 0.0,
            "initial_margin": 0.0,
            "spot": 0.0,
            "margin": 0.0,
            "futures": 0.0,
        }
    try:
        fut = _fetch_futures_account_usdt()
        spot_rows = _fetch_spot_accounts()
        # Suma spot en stablecoins (simplificación sin mark-to-usd)
        spot_usd = 0.0
        for s in spot_rows:
            cur = (s.get("currency") or "").upper()
            if cur in ("USDT", "USDC", "USD"):
                spot_usd += _safe_float(s.get("balance", 0))

        futures_equity = fut["equity"]
        futures_unreal = fut["unrealized_pnl"]
        initial_margin = fut.get("initial_margin", 0.0)

        equity = spot_usd + futures_equity
        balance = spot_usd + _safe_float(fut.get("available", 0))

        return {
            "exchange": "kucoin",
            "equity": equity,
            "balance": balance,
            "unrealized_pnl": futures_unreal,
            "initial_margin": initial_margin,
            "spot": spot_usd,
            "margin": 0.0,              # Si implementas margin spot, mapa aquí
            "futures": futures_equity,  # bucket de derivados
        }
    except Exception as e:
        print(f"❌ KuCoin balances error: {e}")
        return {
            "exchange": "kucoin",
            "equity": 0.0,
            "balance": 0.0,
            "unrealized_pnl": 0.0,
            "initial_margin": 0.0,
            "spot": 0.0,
            "margin": 0.0,
            "futures": 0.0,
        }

# =========================
# Open Positions
# =========================
def fetch_kucoin_open_positions() -> List[Dict[str, Any]]:
    """
    Shape estándar para OPEN:
    realized_pnl = fee + funding_fee (regla del front para abiertas)
    """
    if not _has_creds():
        return []
    try:
        endpoint = "/api/v1/positions"
        resp = _get(KUCOIN_BASE_URL, endpoint, "GET")
        rows = resp.get("data", []) if isinstance(resp, dict) else []
        out: List[Dict[str, Any]] = []
        for pos in rows:
            qty = _safe_float(pos.get("currentQty", 0))
            if qty == 0:
                continue
            side = "long" if qty > 0 else "short"
            entry = _safe_float(pos.get("avgEntryPrice", pos.get("avgEntryPrice", 0)))
            mark  = _safe_float(pos.get("markPrice", 0))
            unreal = _safe_float(pos.get("unrealisedPnl", pos.get("unrealisedPNL", 0)))

            # fees / funding acumulados (algunos campos varían por doc/modelo)
            fee = _safe_float(pos.get("cumulativeTradeFee", pos.get("dealComm", 0)))
            # normaliza fee a negativa si no lo está:
            if fee > 0: fee = -abs(fee)
            funding_fee = _safe_float(pos.get("cumulativeFundingFee", pos.get("fundingFee", 0)))
            realized_open = fee + funding_fee

            out.append({
                "exchange": "kucoin",
                "symbol": _normalize_symbol(pos.get("symbol", "")),
                "side": side,
                "size": abs(qty),
                "entry_price": entry,
                "mark_price": mark,
                "liquidation_price": _safe_float(pos.get("liquidationPrice", 0)),
                "notional": _safe_float(pos.get("positionValue", pos.get("posCost", 0))),
                "unrealized_pnl": unreal,   # SOLO precio
                "fee": fee,
                "funding_fee": funding_fee,
                "realized_pnl": realized_open,  # regla OPEN
            })
        return out
    except Exception as e:
        print(f"❌ KuCoin positions error: {e}")
        return []

# =========================
# Funding History
# =========================
def fetch_funding_kucoin(limit: int = 100, symbol: Optional[str] = None,
                         startAt: Optional[int] = None, endAt: Optional[int] = None) -> List[Dict[str, Any]]:
    if not _has_creds():
        return []
    try:
        endpoint = "/api/v1/funding-history"
        params = []
        if symbol:  params.append(f"symbol={symbol}")
        if startAt: params.append(f"startAt={int(startAt)}")
        if endAt:   params.append(f"endAt={int(endAt)}")
        if limit:   params.append(f"pageSize={min(int(limit), 200)}")
        query = ("?" + "&".join(params)) if params else ""
        full_endpoint = endpoint + query

        resp = _get(KUCOIN_BASE_URL, full_endpoint, "GET")
        rows = resp.get("data", []) if isinstance(resp, dict) else []
        out: List[Dict[str, Any]] = []
        for f in rows:
            sym = _normalize_symbol(f.get("symbol", ""))  # <- ahora sí incluimos el símbolo normalizado
            out.append({
                "exchange": "kucoin",               
                "symbol": _symbol_no_dash(f.get("symbol", "")),               
                "asset": f.get("settleCurrency", "USDT"),
                "timestamp": int(f.get("createdAt", 0)),
                "funding_rate": _safe_float(f.get("fundingRate", 0)),
                "type": "FUNDING_FEE",
            })
        return out
    except Exception as e:
        print(f"❌ KuCoin funding error: {e}")
        return []

# =========================
# Closed Positions (history) + persistencia

from typing import Optional, List, Dict

def fetch_closed_positions_kucoin(symbol: Optional[str] = None,
                                  start_time: Optional[int] = None,
                                  end_time: Optional[int] = None,
                                  limit: int = 100,
                                  page_id: int = 1,
                                  debug: bool = False) -> List[Dict]:
    """
    Salida lista para DB/UI:
      - realized_pnl     -> TOTAL neto (KuCoin 'pnl')  [la app lo pinta como 'Realized']
      - fee_total        -> coste SIEMPRE negativo
      - funding_total    -> funding con signo nativo (+ cobro / − pago)
      - fees/funding_fee -> alias para el front (mismo valor que *_total)
      - size             -> |price_pnl| / |close-entry| ; fallback con ROE/leverage si delta≈0
    """
    if not _has_creds():
        return []

    try:
        endpoint = "/api/v1/history-positions"
        params = []
        if symbol:                  params.append(f"symbol={symbol}")
        if start_time is not None:  params.append(f"startAt={int(start_time)}")  # antes ponías 'from'
        if end_time   is not None:  params.append(f"endAt={int(end_time)}")      # antes 'to'
        if limit is not None:       params.append(f"pageSize={min(int(limit), 200)}")
        if page_id is not None:     params.append(f"currentPage={int(page_id)}")
        query = "?" + "&".join(params) if params else ""
        full_endpoint = endpoint + query
        payload = _get(KUCOIN_BASE_URL, full_endpoint, "GET")
        items = (payload.get("data") or {}).get("items", [])
        out: List[Dict] = []
        if debug:
            meta = (payload.get("data") or {})
            print(f"KuCoin raw -> page={meta.get('currentPage')} size={meta.get('pageSize')} total={meta.get('totalNum')}")

        for pos in items:
            # --- tiempos / precios base ---
            open_ms   = int(_safe_float(pos.get("openTime", 0)))
            close_ms  = int(_safe_float(pos.get("closeTime", 0)))
            entry     = _safe_float(pos.get("openPrice", 0))
            close_p   = _safe_float(pos.get("closePrice", 0))
            diff      = close_p - entry

            # --- números de la API ---
            pnl_net      = _safe_float(pos.get("pnl", 0))         # TOTAL neto
            trade_fee    = _safe_float(pos.get("tradeFee", 0))    # suele venir positivo (coste)
            funding_raw  = _safe_float(pos.get("fundingFee", 0))  # + cobro / − pago
            lev          = _safe_float(pos.get("leverage", 0))
            roe          = _safe_float(pos.get("roe", 0))          # decimal; p.ej. -0.0152

            # --- normalizaciones / derivados ---
            fee_total   = -abs(trade_fee)                          # SIEMPRE negativo
            # PnL de precio (lo que tu app quiere mostrar como 'PnL'):
            #   price_pnl = pnl_net - fees - funding  (OJO: fees ya negativas)
            price_pnl   = pnl_net - fee_total - funding_raw

            # Size exacto desde PnL de precio
            size = 0.0
            if abs(diff) > 0:
                size = abs(price_pnl) / abs(diff)
            elif entry > 0 and lev > 0 and roe:
                # Fallback: ROE ≈ pnl_net / (notional/leverage)  => size ≈ |pnl_net| * lev / (|roe| * entry)
                try:
                    size = abs(pnl_net) * lev / (abs(roe) * entry)
                except Exception:
                    size = 0.0
            

            side = (pos.get("side", "") or "").lower() or "closed"
            #nuevo v2
            notional = entry * size if entry and size else 0.0
            
            # ===== Base de capital para % y APR =====
            # Preferimos margen explícito si KuCoin lo trae (nombres más comunes), si no: notional/leverage
            raw_margin = None
            for k in ("positionMargin", "margin", "initialMargin", "marginAmount"):
                v = pos.get(k)
                if v is not None:
                    try:
                        raw_margin = float(v)
                        break
                    except Exception:
                        pass
            
            if raw_margin is not None and raw_margin > 0:
                base_capital = raw_margin
            elif notional and lev and lev > 0:
                base_capital = notional / lev
            else:
                # último recurso: usar notional para no dejarlo en cero
                base_capital = notional
            pnl_percent = (pnl_net / base_capital) * 100.0 if base_capital else 0.0
            days = max((close_ms - open_ms) / 1000.0 / 86400.0, 1e-9)
            apr_val = pnl_percent * (365.0 / days) if days > 0 else 0.0

            out.append({
                "exchange": "kucoin",
                "symbol": _normalize_symbol(pos.get("symbol", "")),
                "side": side,
                "size": float(size),
                "entry_price": entry,
                "close_price": close_p,
                "open_time": int(open_ms // 1000),
                "close_time": int(close_ms // 1000),

                # === DB (lo que guardamos) ===
                "pnl": float(price_pnl),    #nuevo esquema     
                "pnl_percent": float(pnl_percent),     #nuevo esquema
                "apr": float(apr_val),                 # nuevo esquema
                "realized_pnl": float(pnl_net),        # TOTAL neto (la app lo pinta en 'Realized')
                "fee_total": float(fee_total),         # coste negativo
                "funding_total": float(funding_raw),   # +/−
                "notional": float(entry * size) if entry and size else 0.0,
                "leverage": float(lev),
                "liquidation_price": None,

                # === Alias para la ruta/UI (por si devuelves dicts tal cual) ===
                "fees": float(fee_total),
                "funding_fee": float(funding_raw),

                # === Útil para revisiones puntuales (puedes borrar si no lo usas) ===
                # "pnl_price": float(price_pnl),
            })

        if debug:
            print(f"✅ KuCoin closed positions: {len(out)}")
        return out

    except Exception as e:
        print(f"❌ KuCoin closed positions error: {e}")
        return []




# db_manager.py

def save_kucoin_closed_positions(position: dict):
    import sqlite3
    def _f(x, d=0.0):
        try: return float(x)
        except: return d

    def _price_pnl(side, entry, close, size):
        s = (side or "").lower()
        return (entry - close) * size if s == "short" else (close - entry) * size

    exchange = position.get("exchange")
    symbol   = position.get("symbol")
    side     = position.get("side")

    size     = _f(position.get("size"))
    entry    = _f(position.get("entry_price"))
    close    = _f(position.get("close_price"))
    open_s   = int(position.get("open_time") or 0)
    close_s  = int(position.get("close_time") or 0)

    fee_total     = -abs(_f(position.get("fee_total", 0.0)))       # fees siempre negativos
    funding_total = _f(position.get("funding_total", 0.0))
    notional      = _f(position.get("notional", size * entry))
    leverage      = _f(position.get("leverage"))
    liq_price     = _f(position.get("liquidation_price"))

    # margen inicial si viene del adapter
    initial_margin = position.get("initial_margin")
    initial_margin = _f(initial_margin) if initial_margin is not None else None

    # pnl de precio y realized
    pnl_price = _f(position.get("pnl", _price_pnl(side, entry, close, size)))
    realized  = _f(position.get("realized_pnl", pnl_price + funding_total + fee_total))

    # denominador ROI/APR: initial_margin > notional/leverage > notional
    if initial_margin and initial_margin > 0:
        base_capital = initial_margin
    elif notional and leverage and leverage > 0:
        base_capital = notional / leverage
    else:
        base_capital = notional

    pnl_percent = position.get("pnl_percent")
    if pnl_percent is None:
        pnl_percent = (realized / base_capital) * 100.0 if base_capital else 0.0

    # días con suelo anti-división-por-cero
    days = 0.0
    if open_s and close_s:
        days = max((close_s - open_s) / 86400.0, 1e-9)
    apr = position.get("apr")
    if apr is None:
        apr = pnl_percent * (365.0 / days) if days > 0 else 0.0

    # INSERT con columnas y placeholders 1:1. Nada de inventos.
    sql = (
        "INSERT INTO closed_positions ("
        "exchange, symbol, side, size, entry_price, close_price, "
        "open_time, close_time, pnl, realized_pnl, funding_total, fee_total, "
        "pnl_percent, apr, initial_margin, notional, leverage, liquidation_price"
        ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    )

    vals = (
        exchange, symbol, side, size, entry, close,
        open_s, close_s, pnl_price, realized, funding_total, fee_total,
        float(pnl_percent), float(apr), initial_margin, notional, leverage, liq_price
    )

    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute(sql, vals)
    conn.commit()
    conn.close()


# def debug_preview_kucoin_closed(days: int = 3, symbol: Optional[str] = None):
#     """
#     Dry-run: muestra lo que fetch_closed_positions_kucoin entregaría a save_closed_position.
#     """
#     import time
#     now = int(time.time() * 1000)
#     start = now - min(days, 7) * 24 * 60 * 60 * 1000  # límite de ventana de KuCoin

#     items = fetch_closed_positions_kucoin(symbol=symbol, start_time=start, end_time=now, limit=50, debug=False)
#     for i, it in enumerate(items, 1):
#         print(f"[{i}] {it['exchange']} {it['symbol']} {it['side']} "
#               f"size={it['size']:.4f} entry={it['entry_price']:.6f} close={it['close_price']:.6f}")
#         # Lo que realmente guardará la DB:
#         print(f"    realized_pnl(net)={it['realized_pnl']:.6f}  "
#               f"fee_total(neg)={it['fee_total']:.6f}  funding_total={it['funding_total']:.6f}")
#         # Para cruzar con el front: PnL (precio) = net - fees - funding
#         price_pnl = it['realized_pnl'] - it['fee_total'] - it['funding_total']
#         print(f"    [UI-check] price_pnl(expected)={price_pnl:.6f}  "
#               f"delta=close-entry={it['close_price']-it['entry_price']:.6f}  "
#               f"size≈{it['size']:.4f}")


