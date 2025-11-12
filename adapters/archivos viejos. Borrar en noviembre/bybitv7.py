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


# ======================================================
# UNIFIED TRANSACTION LOGS CACHE (para funding de OPEN y para funding fees)
# ======================================================
_TRANSACTION_LOGS_CACHE: Dict[tuple, List[dict]] = {}

def _fetch_transaction_logs_cached(
    category: str = "linear", 
    currency: str = "USDT", 
    start_ms: int = 0, 
    end_ms: Optional[int] = None,
    limit: int = 50, 
    max_pages: int = 50,
    force_refresh: bool = False
) -> List[dict]:
    """
    Cachea los transaction logs para reutilizar entre diferentes funciones.
    Clave de cache: (category, currency, start_ms, end_ms)
    """
    cache_key = (category, currency, start_ms, end_ms or _now_ms())
    
    if not force_refresh and cache_key in _TRANSACTION_LOGS_CACHE:
        return _TRANSACTION_LOGS_CACHE[cache_key]
    
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

    s = int(start_ms)
    while s < end_ms:
        e = min(s + 7*24*3600*1000 - 1, end_ms)
        _window_pull(s, e)
        s = e + 1
        if s <= 0: break

    _TRANSACTION_LOGS_CACHE[cache_key] = events
    return events

def _sum_funding_by_symbol(
    category: str = "linear", 
    currency: str = "USDT", 
    start_ms: int = 0, 
    end_ms: Optional[int] = None, 
    symbols_filter: Optional[set] = None
) -> Dict[str, float]:
    """
    Devuelve mapping symbol(BASE) -> funding_sum (positivo = pago, negativo = cobro)
    Usa la cache unificada de transaction logs.
    """
    lst = _fetch_transaction_logs_cached(
        category=category, 
        currency=currency, 
        start_ms=start_ms, 
        end_ms=end_ms
    )
    
    acc: Dict[str, float] = {}
    for it in lst:
        sym = normalize_symbol(it.get("symbol",""))
        if symbols_filter and sym not in symbols_filter:
            continue
        try:
            # funding: positivo = pago (gasto), negativo = cobro (ingreso)
            inc = float(it.get("funding","0"))
        except:
            inc = 0.0
        acc[sym] = acc.get(sym, 0.0) + inc
    return acc

# ==========================================
# OPEN POSITIONS (con realized / fee funding)
# ==========================================
def fetch_bybit_open_positions(
    category: str = "linear", 
    settle_coin: str = "USDT", 
    limit: int = 200, 
    max_pages: int = 5, 
    enrich_funding: bool = True
) -> List[dict]:
    """
    Usa la cache unificada de transaction logs para calcular funding.
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
        if settle_coin: params["settleCoin"] = settle_coin
        if cursor: params["cursor"] = cursor

        res = _get("/v5/position/list", params=params, auth=True)
        items = res.get("list") or []
        if not items: break

        for p in items:
            size = abs(_to_float(p.get("size")))
            if size <= 0:
                continue

            side = ("long" if (p.get("side","").lower()=="buy") else "short")
            entry = _to_float(p.get("avgPrice"))
            mark  = _to_float(p.get("markPrice"))
            liq_raw = p.get("liqPrice")
            liq = _to_float(liq_raw) if liq_raw not in ("", None) else 0.0
            sym = normalize_symbol(p.get("symbol",""))

            notional = size * (mark if mark>0 else entry)
            unrealised = p.get("unrealisedPnl")
            if unrealised not in (None, "",):
                unreal = _to_float(unrealised)
            else:
                if side == "long":
                    unreal = (mark - entry) * size
                else:
                    unreal = (entry - mark) * size

            realized = _to_float(p.get("cumRealisedPnl") or p.get("curRealisedPnl") or 0.0)

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
                "fee": 0.0,
                "funding_fee": 0.0,
                "realized_pnl": float(realized),
            }
            out.append(pos)

        cursor = res.get("nextPageCursor")
        if not cursor: break

    # Enriquecer con funding usando la cache unificada
    if enrich_funding and out and symbols_set:
        start_ms = min_created or (_now_ms() - 7*24*3600*1000)
        funding_map = _sum_funding_by_symbol(
            category=category, 
            currency=settle_coin or "USDT",
            start_ms=start_ms, 
            end_ms=_now_ms(), 
            symbols_filter=symbols_set
        )
        for o in out:
            f = float(funding_map.get(o["symbol"], 0.0))
            o["funding_fee"] = f  # positivo=pago, negativo=cobro
            o["fee"] = float(o["realized_pnl"] - f)

    return out

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

# =======================================
# FUNDING FEES individuales (para la app)
# =======================================
def fetch_bybit_funding_fees(
    category: str = "linear", 
    currency: str = "USDT", 
    since: Optional[int] = None, 
    limit: int = 50
) -> List[dict]:
    """
    Devuelve funding fees individuales en formato estándar.
    Reutiliza la cache unificada de transaction logs.
    
    Formato:
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
    start_ms = since if since is not None else (_now_ms() - 24*3600*1000)
    
    # Usa la cache unificada
    raw_logs = _fetch_transaction_logs_cached(
        category=category,
        currency=currency,
        start_ms=start_ms,
        end_ms=_now_ms(),
        limit=limit
    )
    
    out: List[dict] = []
    for it in raw_logs:
        sym = normalize_symbol(it.get("symbol",""))
        
        # funding: positivo = pago (gasto), negativo = cobro (ingreso)
        # Para la app: invertimos signo (+ cobro / - pago)
        f_raw = it.get("funding", "")
        try: 
            f_val = float(f_raw)
            income = -f_val  # invertimos para el formato estándar
        except: 
            income = 0.0
            
        ts = _to_ms(it.get("transactionTime"))
        fr = it.get("feeRate")
        try: 
            fr_val = float(fr)
        except: 
            fr_val = 0.0

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



# ======================================================
# TX LOGS (TRADE + SETTLEMENT) para FIFO de cerradas
# ======================================================
_TXLOG_CACHE_GENERIC: Dict[tuple, List[dict]] = {}

def _fetch_txlogs_windowed(
    category: str,
    currency: str,
    start_ms: int,
    end_ms: int,
    type_filter: str,             # "TRADE" | "SETTLEMENT"
    limit: int = 50,
    max_pages: int = 80,
) -> List[dict]:
    """
    Descarga transaction-log por ventanas de 7d con filtro de tipo.
    Usa una caché separada para no invalidar la de funding ya existente.
    """
    cache_key = (category, currency, start_ms, end_ms, type_filter, limit)
    if cache_key in _TXLOG_CACHE_GENERIC:
        return _TXLOG_CACHE_GENERIC[cache_key]

    out: List[dict] = []

    def _pull(s_ms: int, e_ms: int):
        cursor = None
        pages = 0
        while True:
            pages += 1
            if pages > max_pages:
                break
            params = {
                "accountType": "UNIFIED",
                "category": category,
                "currency": currency,
                "type": type_filter,
                "startTime": s_ms,
                "endTime": e_ms,
                "limit": limit,
            }
            if cursor:
                params["cursor"] = cursor
            res = _get("/v5/account/transaction-log", params=params, auth=True)
            lst = res.get("list") or []
            for it in lst:
                if str(it.get("type", "")).upper() != type_filter:
                    continue
                out.append(it)
            cursor = res.get("nextPageCursor")
            if not cursor:
                break

    s = int(start_ms)
    while s < end_ms:
        e = min(s + 7 * 24 * 3600 * 1000 - 1, end_ms)
        _pull(s, e)
        s = e + 1

    # ordena cronológicamente por seguridad
    out.sort(key=lambda x: _to_ms(x.get("transactionTime")))
    _TXLOG_CACHE_GENERIC[cache_key] = out
    return out


# ======================================================
# Reconstrucción FIFO con TX LOGS (TRADE + SETTLEMENT)
# ======================================================
def fetch_bybit_closed_positions_fifo(
    days: int = 30,
    category: str = "linear",
    currency: str = "USDT",
    symbol: Optional[str] = None,
    eps_qty: float = 1e-6,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Reconstruye posiciones cerradas usando sólo transaction-log:
      - TRADES (type=TRADE) para las operaciones (qty, price, fee)
      - FUNDING (type=SETTLEMENT) para funding en la ventana del bloque
    Bloque = tramo donde el neto por símbolo vuelve a ~0 (FIFO por agregación temporal).
    """
    now = _now_ms()
    start_ms = now - int(days) * 24 * 3600 * 1000

    # 1) TRADE logs (todas las ejecuciones de derivados)
    trades = _fetch_txlogs_windowed(
        category=category, currency=currency, start_ms=start_ms, end_ms=now, type_filter="TRADE", limit=50
    )

    if symbol:
        sym_base = normalize_symbol(symbol)
        trades = [t for t in trades if normalize_symbol(t.get("symbol", "")) == sym_base]

    # 2) SETTLEMENT logs (funding)
    settlements = _fetch_txlogs_windowed(
        category=category, currency=currency, start_ms=start_ms, end_ms=now, type_filter="SETTLEMENT", limit=50
    )
    # Index funding por símbolo BASE
    funding_by_base: Dict[str, List[dict]] = {}
    for it in settlements:
        base = normalize_symbol(it.get("symbol", ""))
        funding_by_base.setdefault(base, []).append(it)

    # 3) Agrupar trades por símbolo BASE
    trades_by_base: Dict[str, List[dict]] = {}
    for t in trades:
        base = normalize_symbol(t.get("symbol", ""))
        if not base:
            continue
        trades_by_base.setdefault(base, []).append(t)

    results: List[Dict[str, Any]] = []

    for base, items in trades_by_base.items():
        # Ordenados por tiempo (ya vienen ordenados, reforzamos)
        items.sort(key=lambda x: _to_ms(x.get("transactionTime")))

        # Normalizar
        norm: List[Dict[str, Any]] = []
        for it in items:
            try:
                side = (it.get("side") or "").upper()        # "Buy" | "Sell"
                qty = _to_float(it.get("qty") or 0)
                price = _to_float(it.get("tradePrice") or 0)
                fee_raw = _to_float(it.get("fee") or 0)      # +gasto / -rebate
                # Para FIFO: qty con signo por lado
                signed = qty if side == "BUY" else -qty
                ts = _to_ms(it.get("transactionTime"))
                norm.append({
                    "qty": qty,
                    "price": price,
                    "fee": abs(fee_raw),     # coste absoluto para "fee_total" siempre negativa
                    "signed": signed,
                    "ts": ts,
                })
            except Exception:
                continue

        if not norm:
            continue

        # 4) Construir bloques por neto
        net = 0.0
        block: List[Dict[str, Any]] = []

        def _close_block(bl: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not bl:
                return None
            buys  = [x for x in bl if x["signed"] > 0]
            sells = [x for x in bl if x["signed"] < 0]
            if not buys or not sells:
                return None

            buy_qty  = sum(x["qty"] for x in buys)
            sell_qty = sum(x["qty"] for x in sells)
            if buy_qty <= 0 or sell_qty <= 0:
                return None

            avg_buy  = sum(x["qty"] * x["price"] for x in buys) / buy_qty
            avg_sell = sum(x["qty"] * x["price"] for x in sells) / sell_qty

            is_short = bl[0]["signed"] < 0
            side = "short" if is_short else "long"
            entry_avg = avg_sell if is_short else avg_buy
            close_avg = avg_buy  if is_short else avg_sell
            size = min(buy_qty, sell_qty)

            # PnL de precio (puro)
            if is_short:
                pnl_price = (entry_avg - close_avg) * size
            else:
                pnl_price = (close_avg - entry_avg) * size

            # Fees → siempre negativas: fee_total = -(suma_abs_fees)
            fees_abs = sum(x["fee"] for x in bl)
            fee_total = -abs(fees_abs)

            # Funding en ventana [open_ts, close_ts]
            open_ts  = min(x["ts"] for x in bl)
            close_ts = max(x["ts"] for x in bl)
            f_sum_income = 0.0
            for f in funding_by_base.get(base, []):
                ts_f = _to_ms(f.get("transactionTime"))
                if open_ts <= ts_f <= close_ts:
                    # Por doc: funding > 0 => gasto (pago). Income = -funding.
                    f_val = _to_float(f.get("funding") or 0.0)
                    f_sum_income += (-f_val)  # +cobro / -pago

            realized_pnl = pnl_price + fee_total + f_sum_income

            row = {
                "exchange": "bybit",
                "symbol": base,
                "side": side,
                "size": float(size),
                "entry_price": float(entry_avg),
                "close_price": float(close_avg),
                "open_time": int(open_ts // 1000),
                "close_time": int(close_ts // 1000),
                "pnl": float(pnl_price),
                "realized_pnl": float(realized_pnl),
                "funding_total": float(f_sum_income),  # +cobro / -pago
                "fee_total": float(fee_total),         # siempre NEGATIVA
                "notional": float(entry_avg * size),
                "leverage": None,                      # Bybit no lo da en txlog
                "initial_margin": None,
                "liquidation_price": None,
            }
            return row

        for tr in norm:
            net += tr["signed"]
            block.append(tr)
            if abs(net) <= eps_qty:
                rec = _close_block(block)
                if rec:
                    results.append(rec)
                    if debug:
                        print(f"  ✅ [{base}] {rec['side']} size={rec['size']:.6f} "
                              f"entry={rec['entry_price']:.6f} close={rec['close_price']:.6f} "
                              f"pnl={rec['pnl']:.6f} fee={rec['fee_total']:.6f} funding={rec['funding_total']:.6f}")
                block, net = [], 0.0

        # Flush si quedó cerca de 0 (ej. cierre justo en el límite de ventana)
        if block and abs(net) <= eps_qty:
            rec = _close_block(block)
            if rec:
                results.append(rec)
                if debug:
                    print(f"  ✅ [FLUSH {base}] {rec['side']} size={rec['size']:.6f} "
                          f"entry={rec['entry_price']:.6f} close={rec['close_price']:.6f} "
                          f"pnl={rec['pnl']:.6f} fee={rec['fee_total']:.6f} funding={rec['funding_total']:.6f}")

    return results


def save_bybit_closed_positions(
    db_path: str = "portfolio.db",
    days: int = 30,
    category: str = "linear",
    currency: str = "USDT",
    symbol: Optional[str] = None,
    debug: bool = False,
) -> Tuple[int, int]:
    """
    Guarda en SQLite usando db_manager.save_closed_position(rows FIFO).
    """
    try:
        from db_manager import save_closed_position
    except Exception as e:
        raise RuntimeError(f"db_manager.save_closed_position no disponible: {e}")

    rows = fetch_bybit_closed_positions_fifo(
        days=days, category=category, currency=currency, symbol=symbol, debug=debug
    )
    saved = 0
    dup = 0
    for row in rows:
        try:
            save_closed_position(row)
            saved += 1
        except Exception as e:
            msg = str(e).lower()
            if "unique" in msg or "duplicate" in msg:
                dup += 1
            else:
                raise
    if debug:
        print(f"✅ Bybit FIFO cerradas guardadas: {saved} (omitidas {dup})")
    return saved, dup

# ========== Debug rápido (ejecución directa) ==========
def _demo():
    print("== Bybit v3 demo (UNIFIED TRANSACTION LOGS) ==")
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
        funding_fees = fetch_bybit_funding_fees(category="linear", since=_now_ms()-7*24*3600*1000)
        print(f"FUNDING FEES count: {len(funding_fees)}")
        for f in funding_fees[:5]:
            print(f)
    except Exception as e:
        print("❌ funding fees:", e)

    try:
        closed = fetch_bybit_closed_positions_fifo(category="linear", days=7, symbol=None, currency="USDT")
        print(f"CLOSED rows (preview): {len(closed)}")
        for r in closed[:5]:
            print(r)
    except Exception as e:
        print("❌ closed fetch:", e)

if __name__ == "__main__":
    _demo()

__all__ = [
    "fetch_bybit_open_positions",
    "fetch_bybit_funding_fees",  # ¡Ahora exportamos esta función!
    "fetch_bybit_closed_positions_fifo",
    "save_bybit_closed_positions",
]

