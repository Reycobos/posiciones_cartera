from __future__ import annotations
import os, time, hmac, hashlib, json
from typing import Any, Dict, List, Optional, Tuple
import requests

# ==== Utils proyecto (fallbacks seguros si ejecutas este archivo suelto) ====
try:
    from utils.symbols import normalize_symbol  # normalización EXACTA global del proyecto
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

def _to_float(x) -> float:
    try: return float(x)
    except: return 0.0

def _neg_fee(x) -> float:
    # fees siempre negativas en tu contrato
    v = _to_float(x)
    return -abs(v)

def _now_ms() -> int: return int(time.time()*1000)
def _to_ms(ts) -> int:
    try:
        v = int(float(ts))
    except:
        return 0
    return v if v >= 10**12 else v*1000
def _to_s(ts) -> int:
    v = _to_ms(ts)
    return v//1000

# ================================
# Config y firma Bybit v5
# ================================
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BYBIT_BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
RECV_WINDOW = "5000"
TIMEOUT = 30

def _sign_v5(query: str, timestamp: str, recv_window: str=RECV_WINDOW) -> str:
    payload = f"{timestamp}{BYBIT_API_KEY}{recv_window}{query}"
    return hmac.new(BYBIT_API_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()

def _qs(params: Dict[str, Any]) -> str:
    cleaned = {}
    for k, v in (params or {}).items():
        if v in (None, "", []): continue
        if isinstance(v, bool): v = str(v).lower()
        cleaned[k] = v
    return "&".join([f"{k}={cleaned[k]}" for k in sorted(cleaned.keys())])

def _get(path: str, params: Dict[str, Any] | None=None, auth: bool=True) -> dict:
    params = params or {}
    url = f"{BYBIT_BASE_URL}{path}"
    headers = {}
    if auth:
        ts = str(_now_ms())
        qs = _qs(params)
        sig = _sign_v5(qs, ts, RECV_WINDOW)
        headers = {
            "X-BAPI-API-KEY": BYBIT_API_KEY,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": RECV_WINDOW,
            "X-BAPI-SIGN": sig,
        }
        if qs: url = f"{url}?{qs}"
    else:
        qs = _qs(params)
        if qs: url = f"{url}?{qs}"
    r = requests.get(url, headers=headers, timeout=TIMEOUT)
    data = r.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error {data.get('retCode')} - {data.get('retMsg')}")
    return data.get("result") or {}


def fetch_bybit_all_balances(db_path: str = "portfolio.db") -> dict:
    """
    /v5/account/wallet-balance (accountType=UNIFIED)
    Devuelve la estructura EXACTA para /api/balances (D):
    {
      "exchange": "bybit",
      "equity": float,
      "balance": float,
      "unrealized_pnl": float,
      "initial_margin": 0.0,
      "spot": 0.0,
      "margin": 0.0,
      "futures": float
    }
    """
    # Llama con firma v5 usando el helper _get de este módulo (v3)
    res = _get("/v5/account/wallet-balance", params={"accountType": "UNIFIED"}, auth=True)
    lst = res.get("list") or []
    acc = lst[0] if lst else {}

    equity = _to_float(acc.get("totalEquity"))
    wallet = _to_float(acc.get("totalWalletBalance"))
    unreal = equity - wallet  # PnL no realizado

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

    # Log opcional con el estilo global si el servidor define el helper
    try:
        from builtins import p_balance_equity as _pbe  # type: ignore
        _pbe("bybit", obj["equity"])
    except Exception:
        pass

    return obj



# ======================================================
# FUNDING LOGS (usado para OPEN y para CLOSED también)
# ======================================================
def _fetch_tx_logs(category="linear", currency="USDT", start_ms: int=0, end_ms: Optional[int]=None, limit: int=50, max_pages: int=50) -> List[dict]:
    """
    /v5/account/transaction-log (type=SETTLEMENT) en ventanas <=7 días.
    funding: positivo = gasto (pago), negativo = rebate (cobro)  -> nosotros invertimos el signo.
    """
    if end_ms is None:
        end_ms = _now_ms()
    events: List[dict] = []

    def _window_pull(s_ms: int, e_ms: int):
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
                "startTime": s_ms,
                "endTime": e_ms,
                "limit": limit,
            }
            if cursor: params["cursor"] = cursor
            res = _get("/v5/account/transaction-log", params=params, auth=True)
            lst = res.get("list") or []
            for it in lst:
                if str(it.get("type","")).upper() != "SETTLEMENT": continue
                if it.get("funding","") in (None, "",): continue
                events.append(it)
            cursor = res.get("nextPageCursor")
            if not cursor: break

    now = end_ms
    s = int(start_ms)
    while s < end_ms:
        e = min(s + 7*24*3600*1000 - 1, end_ms)
        _window_pull(s, e)
        s = e + 1
        if s <= 0: break

    return events

def _sum_funding_by_symbol(category="linear", currency="USDT", start_ms: int=0, end_ms: Optional[int]=None, symbols_filter: Optional[set]=None) -> Dict[str, float]:
    """
    Devuelve mapping symbol(BASE) -> income_sum (+cobro/-pago) entre start_ms..end_ms
    """
    lst = _fetch_tx_logs(category=category, currency=currency, start_ms=start_ms, end_ms=end_ms)
    acc: Dict[str, float] = {}
    for it in lst:
        sym = normalize_symbol(it.get("symbol",""))
        if symbols_filter and sym not in symbols_filter:
            continue
        try:
            # funding: positivo=gasto, negativo=reembolso  -> income = -funding
            inc = -float(it.get("funding","0"))
        except:
            inc = 0.0
        acc[sym] = acc.get(sym, 0.0) + inc
    return acc

# ==========================================
# OPEN POSITIONS (con realized / fee funding)
# ==========================================
def fetch_bybit_open_positions(category: str="linear", settle_coin: str="USDT", limit: int=200, max_pages: int=5, enrich_funding: bool=True) -> List[dict]:
    """
    Forma EXACTA B. realized_pnl = cumRealisedPnl (Bybit), funding_fee = suma txlog símbolo (desde min(createdTime)),
    fee = realized_pnl - funding_fee (→ fee negativa si hay costo).
    """
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        raise RuntimeError("BYBIT_API_KEY/BYBIT_API_SECRET no configuradas.")

    out: List[dict] = []
    cursor = None
    page = 0
    min_created = None
    symbols_set = set()

    while True:
        page += 1
        if page > max_pages: break
        params = {"category": category, "limit": limit}
        # Si no pasas symbol, con settleCoin te devuelve solo size>0
        if settle_coin: params["settleCoin"] = settle_coin
        if cursor: params["cursor"] = cursor

        res = _get("/v5/position/list", params=params, auth=True)
        items = res.get("list") or []
        if not items: break

        for p in items:
            size = abs(_to_float(p.get("size")))
            if size <= 0:  # por seguridad
                continue

            side = ("long" if (p.get("side","").lower()=="buy") else "short")
            entry = _to_float(p.get("avgPrice"))
            mark  = _to_float(p.get("markPrice"))
            liq_raw = p.get("liqPrice")
            liq = _to_float(liq_raw) if liq_raw not in ("", None) else 0.0
            sym = normalize_symbol(p.get("symbol",""))

            # notional y unreal según tu contrato
            notional = size * (mark if mark>0 else entry)
            unrealised = p.get("unrealisedPnl")
            if unrealised not in (None, "",):
                # Bybit ya lo trae -> en moneda de settle
                unreal = _to_float(unrealised)
            else:
                if side == "long":
                    unreal = (mark - entry) * size
                else:
                    unreal = (entry - mark) * size

            # realized acumulado (incluye funding+fees según tu observación)
            realized = _to_float(p.get("cumRealisedPnl") or p.get("curRealisedPnl") or 0.0)

            # Track para enriquecimiento con funding
            symbols_set.add(sym)
            ctime = _to_ms(p.get("createdTime"))
            if ctime:
                min_created = ctime if (min_created is None or ctime < min_created) else min_created

            pos = {
                "exchange": "bybit",
                "symbol": sym,
                "side": side,
                "size": float(size),
                "entry_price": float(entry),
                "mark_price": float(mark),
                "liquidation_price": float(liq),
                "notional": float(notional),
                "unrealized_pnl": float(unreal),
                "fee": 0.0,           # se rellena luego si enrich_funding
                "funding_fee": 0.0,   # se rellena luego si enrich_funding
                "realized_pnl": float(realized),
            }
            out.append(pos)

        cursor = res.get("nextPageCursor")
        if not cursor: break

    # Enriquecer con funding por símbolo y derivar fee = realized - funding
    if enrich_funding and out and symbols_set:
        start_ms = min_created or (_now_ms() - 7*24*3600*1000)
        funding_map = _sum_funding_by_symbol(category=category, currency=settle_coin or "USDT",
                                             start_ms=start_ms, end_ms=_now_ms(), symbols_filter=symbols_set)
        for o in out:
            f = float(funding_map.get(o["symbol"], 0.0))
            o["funding_fee"] = f
            o["fee"] = float(o["realized_pnl"] - f)  # NEGATIVA si hay costo → correcto con tu contrato

    return out

# =======================================
# CLOSED PNL (fetch normalizado + save DB)
# =======================================
def fetch_bybit_closed_pnl(category: str="linear", days: int=7, symbol: Optional[str]=None, limit: int=100, max_pages: int=40, currency: str="USDT") -> List[dict]:
    """
    Devuelve filas normalizadas (shape E) listas para guardar:
    - pnl (precio puro) = closedPnl
    - fee_total = -(openFee + closeFee)
    - funding_total = suma funding en txlog entre createdTime..updatedTime por símbolo
    - realized_pnl = pnl + fee_total + funding_total
    """
    if days <= 0: days = 7
    end_ms = _now_ms()
    start_ms = end_ms - days*24*3600*1000

    rows: List[dict] = []
    cursor = None
    page = 0
    while True:
        page += 1
        if page > max_pages: break
        params = {
            "category": category,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }
        if symbol:
            params["symbol"] = symbol
        if cursor:
            params["cursor"] = cursor

        res = _get("/v5/position/closed-pnl", params=params, auth=True)
        lst = res.get("list") or []
        if not lst and not res.get("nextPageCursor"):
            break

        # Pre-cálculo de funding por símbolo y ventana (lo haremos por cada item
        # para respetar exactamente su intervalo created..updated)
        for it in lst:
            sym = normalize_symbol(it.get("symbol",""))
            side = "long" if (str(it.get("side","")).lower()=="buy") else "short"

            qty = it.get("closedSize") or it.get("qty") or "0"
            size = abs(_to_float(qty))

            entry = _to_float(it.get("avgEntryPrice") or it.get("orderPrice") or 0.0)
            close = _to_float(it.get("avgExitPrice")  or 0.0)

            created_ms = _to_ms(it.get("createdTime"))
            updated_ms = _to_ms(it.get("updatedTime"))
            open_s  = _to_s(created_ms)
            close_s = _to_s(updated_ms)

            pnl_price = _to_float(it.get("closedPnl") or 0.0)
            open_fee  = abs(_to_float(it.get("openFee") or 0.0))
            close_fee = abs(_to_float(it.get("closeFee") or 0.0))
            fee_total = -(open_fee + close_fee)

            # funding entre created..updated (SETTLEMENT logs)
            funding_map = _sum_funding_by_symbol(category=category, currency=currency,
                                                 start_ms=created_ms, end_ms=updated_ms,
                                                 symbols_filter={sym})
            funding_total = float(funding_map.get(sym, 0.0))

            realized_pnl = pnl_price + fee_total + funding_total
            notional = size * entry if entry>0 else 0.0
            lev = _to_float(it.get("leverage") or 0.0)
            liq = None  # el endpoint no da liqPrice

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
                "realized_pnl": realized_pnl,
                "funding_total": funding_total,
                "fee_total": fee_total,
                "notional": notional,
                "leverage": lev if lev>0 else None,
                "initial_margin": None,
                "liquidation_price": liq,
            }
            rows.append(row)

        cursor = res.get("nextPageCursor")
        if not cursor: break

    return rows

def save_bybit_closed_positions(db_path: str="portfolio.db", category: str="linear", days: int=30, symbol: Optional[str]=None, currency: str="USDT") -> Tuple[int,int]:
    """
    Guarda en SQLite (tabla closed_positions) usando db_manager.save_closed_position(row).
    Retorna (guardadas, duplicadas_omitidas).
    """
    try:
        from db_manager import save_closed_position
    except Exception as e:
        raise RuntimeError(f"db_manager.save_closed_position no disponible: {e}")

    rows = fetch_bybit_closed_pnl(category=category, days=days, symbol=symbol, currency=currency)
    saved = 0
    dup = 0
    for row in rows:
        try:
            save_closed_position(row)
            saved += 1
        except Exception as e:
            # si hay UNIQUE constraint en tu DB, cuéntalo como duplicado
            msg = str(e).lower()
            if "unique" in msg or "duplicate" in msg:
                dup += 1
            else:
                raise
    return saved, dup

# ========== Debug rápido (ejecución directa) ==========
def _demo():
    print("== Bybit v2 demo ==")
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        print("❌ Define BYBIT_API_KEY/BYBIT_API_SECRET en variables de entorno.")
        return
    try:
        opens = fetch_bybit_open_positions(category="linear", settle_coin="USDT", enrich_funding=True)
        print(f"OPEN count: {len(opens)}")
        for o in opens:
            print(o)
    except Exception as e:
        print("❌ opens:", e)

    try:
        closed = fetch_bybit_closed_pnl(category="linear", days=7, symbol=None, currency="USDT")
        print(f"CLOSED rows (preview): {len(closed)}")
        for r in closed[:5]:
            print(r)
    except Exception as e:
        print("❌ closed fetch:", e)

if __name__ == "__main__":
    _demo()

__all__ = [
    "fetch_bybit_all_balances",
    "fetch_bybit_open_positions",
    "fetch_bybit_funding_fees",  # si ya tienes esta en otro archivo, puedes omitir exportarla aquí
    "fetch_bybit_closed_pnl",
    "save_bybit_closed_positions",
]