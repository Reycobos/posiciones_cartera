# adapters/bybitv1.py
# Integración Bybit v5 — Open Positions, Funding Fees y Closed Positions (persistencia)
# Requiere: BYBIT_API_KEY / BYBIT_API_SECRET en entorno

from __future__ import annotations
import os, hmac, hashlib, time, math
from typing import Any, Dict, List, Optional, Tuple
import requests

# Utils del proyecto
try:
    from symbols import normalize_symbol  # normalización EXACTA de símbolos
except Exception:
    # fallback mínimo por si se ejecuta aislado
    import re
    def normalize_symbol(sym: str) -> str:
        if not sym: return ""
        s = sym.upper()
        s = re.sub(r'^PERP_', '', s)
        s = re.sub(r'(_|-)?(USDT|USDC|USD|PERP)$', '', s)
        s = re.sub(r'[_-]+$', '', s)
        s = re.split(r'[_/-]', s)[0]
        return s

try:
    from money import to_float, normalize_fee  # fee siempre NEGATIVA
except Exception:
    def to_float(x): 
        try: return float(x)
        except: return 0.0
    def normalize_fee(x): 
        try: 
            v=float(x)
            return -abs(v)
        except: 
            return 0.0

try:
    from time import to_ms, to_s  # conversores ms/s reales del proyecto
except Exception:
    def to_ms(ts): t=int(float(ts or 0)); return t if t>=10**12 else t*1000
    def to_s(ts):  t=int(float(ts or 0)); return t//1000 if t>=10**12 else t

# === Toggle de prints con estilo similar al proyecto (no-op si falta) ===
def _noop(*a, **k): pass
p_open_summary      = _noop
p_open_block        = _noop
p_funding_fetching  = _noop
p_funding_count     = _noop
p_closed_debug_header = _noop
p_closed_debug_count  = _noop
p_closed_debug_norm_size = _noop
p_closed_debug_prices = _noop
p_closed_debug_pnl    = _noop
p_closed_debug_times  = _noop
p_closed_debug_normalized = _noop

# Intenta enlazar helpers si el adapter se ejecuta dentro del server
for _name in (
    "p_open_summary","p_open_block","p_funding_fetching","p_funding_count",
    "p_closed_debug_header","p_closed_debug_count","p_closed_debug_norm_size",
    "p_closed_debug_prices","p_closed_debug_pnl","p_closed_debug_times","p_closed_debug_normalized"
):
    try:
        import builtins
        fn = getattr(builtins, _name, None)
        if callable(fn):
            globals()[_name] = fn
    except Exception:
        pass

BYBIT_BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

def _now_ms() -> int: return int(time.time()*1000)

def _sign_v5(query: str, timestamp: str, recv_window: str="5000") -> str:
    """
    Firma v5: sign = HMAC_SHA256(secret, timestamp + api_key + recv_window + query_string)
    """
    payload = f"{timestamp}{BYBIT_API_KEY}{recv_window}{query}"
    return hmac.new(BYBIT_API_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()

def _request(method: str, path: str, params: Dict[str, Any] | None=None, timeout: int=30) -> dict:
    """
    Request v5 firmado para endpoints privados GET.
    """
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        raise RuntimeError("Faltan credenciales BYBIT_API_KEY/BYBIT_API_SECRET en entorno.")
    params = {k: v for k, v in (params or {}).items() if v not in (None, "", [])}
    # ordenar por clave alfabética
    qs = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
    ts = str(_now_ms())
    recv = "5000"
    sig = _sign_v5(qs, ts, recv)
    headers = {
        "X-BAPI-API-KEY": BYBIT_API_KEY,
        "X-BAPI-TIMESTAMP": ts,
        "X-BAPI-RECV-WINDOW": recv,
        "X-BAPI-SIGN": sig,
    }
    url = f"{BYBIT_BASE_URL}{path}"
    if method.upper() == "GET":
        if qs:
            url = f"{url}?{qs}"
        r = requests.get(url, headers=headers, timeout=timeout)
    else:
        # Por ahora solo usamos GET en este adapter
        r = requests.request(method.upper(), url, headers=headers, timeout=timeout)
    data = r.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error {data.get('retCode')} - {data.get('retMsg')}")
    return data.get("result", {}) or {}

# =============================
# A) OPEN POSITIONS (shape B)
# =============================
def fetch_bybit_open_positions(category: str="linear", settle_coin: Optional[str]="USDT", limit: int=200, max_pages: int=5) -> List[dict]:
    """
    Devuelve posiciones ABIERTAS normalizadas para /api/positions (shape EXACTO B).
    """
    out: List[dict] = []
    cursor = None
    page = 0
    total = 0
    while True:
        page += 1
        if page > max_pages: break
        params = {"category": category, "limit": limit}
        if settle_coin:
            params["settleCoin"] = settle_coin
        if cursor:
            params["cursor"] = cursor
        res = _request("GET", "/v5/position/list", params=params)
        items = (res.get("list") or [])
        total += len(items)
        if not items: break

        for p in items:
            # Bybit retorna 'size' siempre POSITIVO; 'side' indica dirección
            try:
                size = abs(to_float(p.get("size")))
            except Exception:
                size = 0.0
            if size <= 0:
                continue  # ignorar vacías

            side_raw = (p.get("side") or "").lower()
            side = "long" if side_raw == "buy" else "short"
            entry = to_float(p.get("avgPrice"))
            mark  = to_float(p.get("markPrice"))
            liq_raw = p.get("liqPrice")
            liq = to_float(liq_raw) if (liq_raw not in (None, "",)) else 0.0
            symbol_raw = p.get("symbol") or ""
            symbol = normalize_symbol(symbol_raw)

            # notional y PnL
            notional = size * (mark if mark > 0 else entry)
            if side == "long":
                unreal = (mark - entry) * size
            else:
                unreal = (entry - mark) * size

            # Fees/funding acumulados no vienen por símbolo en esta call; dejamos 0.0
            fee_total = 0.0
            funding_total = 0.0
            realized_pnl = fee_total + funding_total

            pos = {
                "exchange": "bybit",
                "symbol": symbol,
                "side": side,
                "size": float(size),
                "entry_price": float(entry),
                "mark_price": float(mark),
                "liquidation_price": float(liq),
                "notional": float(notional),
                "unrealized_pnl": float(unreal),
                "fee": float(fee_total),
                "funding_fee": float(funding_total),
                "realized_pnl": float(realized_pnl),
            }
            out.append(pos)

            # Debug estilo proyecto
            p_open_block(
                "bybit", symbol, size, entry, mark,
                unrealized=float(unreal),
                realized_funding=None,
                total_unsettled=None,
                notional=notional,
                extra_verification=False
            )

        cursor = res.get("nextPageCursor")
        if not cursor: break

    p_open_summary("bybit", len(out))
    return out

# ==================================
# B) FUNDING FEES persistentes (C)
# ==================================
def _txlog_windowed_fetch(category="linear", currency="USDT", since: Optional[int]=None, limit: int=50, max_pages: int=50) -> List[dict]:
    """
    Lee /v5/account/transaction-log en ventanas de 7 días (regla de Bybit).
    Filtra type=SETTLEMENT y funding != ''.
    """
    events: List[dict] = []

    def _pull(start_ms: int, end_ms: int) -> None:
        nonlocal events
        cursor = None
        pages = 0
        while True:
            pages += 1
            if pages > max_pages: break
            params = {
                "accountType": "UNIFIED",
                "category": category,
                "currency": currency,
                "type": "SETTLEMENT",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": limit,
            }
            if cursor: params["cursor"] = cursor
            res = _request("GET", "/v5/account/transaction-log", params=params)
            lst = res.get("list") or []
            for it in lst:
                if str(it.get("type","")).upper() != "SETTLEMENT":
                    continue
                if it.get("funding","") in (None, "",):
                    continue
                events.append(it)
            cursor = res.get("nextPageCursor")
            if not cursor: break

    p_funding_fetching("bybit")
    now = _now_ms()
    if since is None:
        # por defecto: últimas 24h (comportamiento del endpoint si no pasas fechas)
        # pero hacemos una ventana manual de 24h para uniformidad con el resto
        since = now - 24*3600*1000

    # dividir en ventanas de 7 días
    start = int(since)
    while start < now:
        end = min(start + 7*24*3600*1000 - 1, now)
        _pull(start, end)
        start = end + 1

    p_funding_count("bybit", len(events))
    return events

def fetch_bybit_funding_fees(limit: int=50, since: Optional[int]=None, category: str="linear", currency: str="USDT") -> List[dict]:
    """
    Normaliza a C):
    {
      "exchange": "bybit",
      "symbol": "<BASE>",
      "income": float (+ cobro / - pago),
      "asset": "USDT"|"USDC"|"USD",
      "timestamp": ms,
      "funding_rate": float | 0.0,
      "type": "FUNDING_FEE"
    }
    """
    raw = _txlog_windowed_fetch(category=category, currency=currency, since=since, limit=limit)
    out: List[dict] = []
    for it in raw:
        sym = normalize_symbol(it.get("symbol",""))
        # funding: positivo=GASTO (pago), negativo=REBATE (cobro)
        f_raw = it.get("funding", "")
        try: f_val = float(f_raw)
        except: f_val = 0.0
        income = -f_val  # invertimos signo a nuestro convenio (+ cobro / - pago)
        ts = to_ms(it.get("transactionTime"))
        fr = it.get("feeRate")
        try: fr_val = float(fr)
        except: fr_val = 0.0

        ev = {
            "exchange": "bybit",
            "symbol": sym,
            "income": float(income),
            "asset": (it.get("currency") or "USDT").upper(),
            "timestamp": int(ts),
            "funding_rate": float(fr_val),
            "type": "FUNDING_FEE",
            "external_id": str(it.get("id") or ""),
        }
        out.append(ev)
    return out

# ===================================
# C) BALANCES (shape D) — resumido
# ===================================
def fetch_bybit_all_balances(db_path: str="portfolio.db") -> dict:
    """
    Usa /v5/account/wallet-balance (accountType=UNIFIED).
    Devuelve la estructura EXACTA de /api/balances (D).
    """
    res = _request("GET", "/v5/account/wallet-balance", params={"accountType": "UNIFIED"})
    lst = res.get("list") or []
    acc = lst[0] if lst else {}
    equity = to_float(acc.get("totalEquity"))
    wallet = to_float(acc.get("totalWalletBalance"))
    unreal = equity - wallet
    obj = {
        "exchange": "bybit",
        "equity": float(equity),
        "balance": float(wallet),
        "unrealized_pnl": float(unreal),
        "initial_margin": 0.0,
        "spot": 0.0,
        "margin": 0.0,
        "futures": float(equity),
    }
    try:
        # si el server define helper, muéstralo
        from builtins import p_balance_equity as _pbe  # type: ignore
        _pbe("bybit", obj["equity"])
    except Exception:
        pass
    return obj

# ==================================================
# D) CLOSED POSITIONS (persistencia SQLite) (E)
# ==================================================
def save_bybit_closed_positions(db_path: str="portfolio.db", days: int=30, debug: bool=False) -> Tuple[int,int]:
    """
    Guarda filas en tabla closed_positions usando db_manager.save_closed_position.
    Intenta usar /v5/position/closed-pnl (v5). Si el campo falta, usa defaults conservadores.
    Retorna (guardadas, duplicadas_omitidas=0).
    """
    try:
        from db_manager import save_closed_position  # escribe en DB real del proyecto
    except Exception as e:
        raise RuntimeError(f"No se pudo importar db_manager.save_closed_position: {e}")

    since_ms = _now_ms() - int(days)*24*3600*1000
    cursor = None
    saved = 0
    dup = 0
    page = 0
    while True:
        page += 1
        if page > 40: break  # cap de seguridad
        params = {
            "category": "linear",
            "startTime": since_ms,
            "limit": 200,
        }
        if cursor: params["cursor"] = cursor
        # NOTA: este endpoint existe en v5; si tu cuenta no lo expone, ajusta aquí.
        res = _request("GET", "/v5/position/closed-pnl", params=params)
        items = res.get("list") or []
        if debug:
            p_closed_debug_count(len(items))
        if not items: break

        for it in items:
            # Mapeos robustos (nombres varían entre cuentas/tipos)
            sym = normalize_symbol(it.get("symbol",""))
            side_raw = (it.get("side") or "").lower()
            side = "long" if side_raw == "buy" else "short"

            qty = it.get("qty") or it.get("size") or it.get("cumExecQty") or 0
            size = abs(to_float(qty))

            entry = to_float(it.get("avgEntryPrice") or it.get("avgEntry") or it.get("entryPrice") or it.get("openPrice") or 0)
            close = to_float(it.get("avgExitPrice")  or it.get("exitPrice") or it.get("closePrice") or it.get("lastPrice") or 0)

            # Tiempos (s) para DB
            ot_ms = to_ms(it.get("createdTime") or it.get("createdAt") or it.get("openTime") or 0)
            ct_ms = to_ms(it.get("closedTime")  or it.get("updatedTime") or it.get("closeTime") or 0)
            open_s  = to_s(ot_ms)
            close_s = to_s(ct_ms)

            # PnLs
            realized = to_float(it.get("closedPnl") or it.get("realisedPnl") or it.get("realizedPnl") or 0)
            fee_total = normalize_fee(it.get("execFee") or it.get("fee") or 0)
            funding_total = to_float(it.get("funding") or 0)

            # PnL puro de precio (si no viene de origen, lo recalcula db_manager)
            pnl_price = None  # dejamos que db_manager lo recalcule si es necesario

            notional = abs(size) * entry if entry > 0 else 0.0
            lev = to_float(it.get("leverage") or 0)
            liq_price = to_float(it.get("liqPrice") or 0)

            # Debug bonito
            if debug:
                p_closed_debug_header(sym)
                p_closed_debug_norm_size(side, size)
                p_closed_debug_prices(entry, close)
                p_closed_debug_pnl(realized, fee_total, funding_total)
                p_closed_debug_times(ot_ms, ct_ms, open_s, close_s)

            row = {
                "exchange": "bybit",
                "symbol": sym,
                "side": side,
                "size": size,
                "entry_price": entry,
                "close_price": close,
                "open_time": open_s,
                "close_time": close_s,
                "pnl": pnl_price,
                "realized_pnl": realized,
                "funding_total": funding_total,
                "fee_total": fee_total,
                "notional": notional,
                "leverage": lev if lev>0 else None,
                "initial_margin": None,  # lo resuelve db_manager si falta
                "liquidation_price": liq_price if liq_price>0 else None,
            }
            try:
                save_closed_position(row)  # DB real del proyecto (recalcula % y APR)
                saved += 1
            except Exception as e:
                # Si db_manager ignora duplicados por PK/unique, cuenta como duplicado
                msg = str(e).lower()
                if ("unique" in msg or "duplicate" in msg):
                    dup += 1
                else:
                    raise

        cursor = res.get("nextPageCursor")
        if not cursor: break

    return saved, dup

# ==================================
# Debug helpers (CLI-friendly)
# ==================================
def debug_preview_bybit_closed(days: int=3, symbol: Optional[str]=None) -> None:
    try:
        count, _ = save_bybit_closed_positions(days=days, debug=True)
        print(f"🔍 Preview closed (days={days}) -> {count} filas (guardadas contra DB).")
    except Exception as e:
        print(f"❌ preview closed error: {e}")

def debug_dump_bybit_opens() -> None:
    try:
        opens = fetch_bybit_open_positions()
        print(f"📈 BYBIT opens = {len(opens)}")
        for o in opens:
            print(o)
    except Exception as e:
        print(f"❌ dump opens error: {e}")

def debug_dump_bybit_funding(days: int=7) -> None:
    try:
        since = _now_ms() - days*24*3600*1000
        evs = fetch_bybit_funding_fees(limit=50, since=since)
        print(f"💵 BYBIT funding (last {days}d): {len(evs)}")
        for e in evs[:10]:
            print(e)
        if len(evs) > 10: print("…")
    except Exception as e:
        print(f"❌ dump funding error: {e}")

__all__ = [
    "fetch_bybit_open_positions",
    "fetch_bybit_funding_fees",
    "fetch_bybit_all_balances",
    "save_bybit_closed_positions",
    "debug_preview_bybit_closed",
    "debug_dump_bybit_opens",
    "debug_dump_bybit_funding",
]
