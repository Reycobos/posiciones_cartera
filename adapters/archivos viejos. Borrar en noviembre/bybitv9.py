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
    Devuelve mapping symbol(BASE) -> funding_sum
    
    🔧 CORRECCIÓN CRÍTICA DE SIGNOS:
    En Bybit UI: negativo = cobrado (a favor), positivo = pagado (en contra)
    En la API "funding": positivo = gasto, negativo = ingreso
    
    Para tu app (income style): positivo = cobro, negativo = pago
    Por tanto: income = funding (sin invertir, ya viene correcto desde API)
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
            # 🔧 CORRECCIÓN: API funding ya viene con el signo correcto
            # positivo en API = gasto (pagado), negativo = ingreso (cobrado)
            # Lo invertimos para income style: +cobro / -pago
            f_val = float(it.get("funding","0"))
            inc = -f_val  # Invertir: negativo API -> positivo income (cobrado)
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

            base = normalize_symbol(p.get("symbol",""))
            notional = entry * size

            # cumRealisedPnL: API reporta (funding + fees) acumulados
            cum_realized = _to_float(p.get("cumRealisedPnl") or 0.0)

            ts_created_raw = p.get("createdTime")
            ts_created = _to_ms(ts_created_raw) if ts_created_raw not in ("", None) else 0
            if ts_created > 0:
                if not min_created or ts_created < min_created:
                    min_created = ts_created
                symbols_set.add(base)

            # PnL no realizado (precio)
            if side == "long":
                unrealized_pnl = (mark - entry) * size
            else:
                unrealized_pnl = (entry - mark) * size

            obj = {
                "exchange": "bybit",
                "symbol": base,
                "side": side,
                "size": float(size),
                "entry_price": float(entry),
                "mark_price": float(mark),
                "liquidation_price": float(liq),
                "notional": float(notional),
                "unrealized_pnl": float(unrealized_pnl),
                "fee": 0.0,          # placeholder
                "funding_fee": 0.0,  # placeholder
                "realized_pnl": float(cum_realized),
            }
            out.append(obj)

        cursor = res.get("nextPageCursor")
        if not cursor: break

    # Enriquecer con funding/fees reales desde transaction logs
    if enrich_funding and symbols_set and min_created:
        funding_sums = _sum_funding_by_symbol(
            category=category,
            currency=settle_coin,
            start_ms=int(min_created),
            end_ms=_now_ms(),
            symbols_filter=symbols_set
        )
        for obj in out:
            sym = obj["symbol"]
            fund_val = funding_sums.get(sym, 0.0)
            
            # 🔧 fund_val ya viene como income style (+cobro / -pago)
            obj["funding_fee"] = fund_val
            
            # fee = realized_pnl - funding_fee
            obj["fee"] = obj["realized_pnl"] - fund_val

    return out


# ==========================================
# FUNDING FEES (historial para /api/funding)
# ==========================================
def fetch_bybit_funding_fees(
    category: str = "linear",
    currency: str = "USDT",
    limit: int = 50,
    max_pages: int = 50,
    since: Optional[int] = None,
) -> List[dict]:
    """
    Devuelve funding fees en formato estandarizado para /api/funding.
    
    🔧 CORRECCIÓN: invierte signo de API para income style (+cobro / -pago)
    """
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        raise RuntimeError("BYBIT_API_KEY/BYBIT_API_SECRET no configuradas.")

    start_ms = int(since) if since else (_now_ms() - 30*24*3600*1000)
    
    lst = _fetch_transaction_logs_cached(
        category=category,
        currency=currency,
        start_ms=start_ms,
        end_ms=_now_ms(),
        limit=limit,
        max_pages=max_pages
    )

    out: List[dict] = []
    for it in lst:
        base = normalize_symbol(it.get("symbol",""))
        ts = _to_ms(it.get("transactionTime"))
        
        # 🔧 CORRECCIÓN: invertir signo API -> income style
        f_val = _to_float(it.get("funding","0"))
        income = -f_val  # negativo API = positivo income (cobrado)
        
        funding_rate = _to_float(it.get("fundingRate","0"))
        
        out.append({
            "exchange": "bybit",
            "symbol": base,
            "income": float(income),  # +cobro / -pago
            "asset": currency,
            "timestamp": int(ts),
            "funding_rate": float(funding_rate),
            "type": "FUNDING_FEE",
        })

    return out


# ==========================================
# BALANCES
# ==========================================
def fetch_bybit_all_balances(
    db_path: str = "portfolio.db",
    account_type: str = "UNIFIED"
) -> dict:
    """
    Usa Bybit v5 Wallet Balance endpoint.
    """
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        raise RuntimeError("BYBIT_API_KEY/BYBIT_API_SECRET no configuradas.")

    params = {"accountType": account_type}
    res = _get("/v5/account/wallet-balance", params=params, auth=True)
    lst = res.get("list") or []
    if not lst:
        return {
            "exchange": "bybit",
            "equity": 0.0,
            "balance": 0.0,
            "unrealized_pnl": 0.0,
            "initial_margin": 0.0,
            "spot": 0.0,
            "margin": 0.0,
            "futures": 0.0,
        }

    acc = lst[0]
    total_equity = _to_float(acc.get("totalEquity") or 0.0)
    total_wallet_balance = _to_float(acc.get("totalWalletBalance") or 0.0)
    unrealized_pnl = _to_float(acc.get("totalPerpUPL") or 0.0)
    total_initial_margin = _to_float(acc.get("totalInitialMargin") or 0.0)

    return {
        "exchange": "bybit",
        "equity": float(total_equity),
        "balance": float(total_wallet_balance),
        "unrealized_pnl": float(unrealized_pnl),
        "initial_margin": float(total_initial_margin),
        "spot": 0.0,
        "margin": 0.0,
        "futures": float(total_equity),
    }


# ==========================================
# CLOSED POSITIONS (FIFO desde fills)
# ==========================================
def fetch_bybit_closed_positions_fifo(
    days: int = 30,
    category: str = "linear",
    currency: str = "USDT",
    symbol: Optional[str] = None,
    debug: bool = False,
) -> List[dict]:
    """
    Reconstruye posiciones cerradas desde fills (trades) con cálculo FIFO.
    
    🔧 CORRECCIÓN CRÍTICA:
    - funding en API: positivo = pagado, negativo = cobrado
    - Invertimos para income style: +cobro / -pago
    """
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        raise RuntimeError("BYBIT_API_KEY/BYBIT_API_SECRET no configuradas.")

    end_ms = _now_ms()
    start_ms = end_ms - (days * 24 * 3600 * 1000)

    # 1) Obtener fills (executions)
    trades_by_symbol: Dict[str, List[dict]] = {}
    cursor = None
    max_pages_trades = 100

    for _ in range(max_pages_trades):
        params = {
            "category": category,
            "limit": 100,
            "startTime": start_ms,
            "endTime": end_ms,
        }
        if symbol:
            params["symbol"] = symbol.upper() + "USDT"
        if cursor:
            params["cursor"] = cursor

        res = _get("/v5/execution/list", params=params, auth=True)
        items = res.get("list") or []
        if not items:
            break

        for t in items:
            raw_sym = t.get("symbol","")
            base = normalize_symbol(raw_sym)
            if base not in trades_by_symbol:
                trades_by_symbol[base] = []
            trades_by_symbol[base].append(t)

        cursor = res.get("nextPageCursor")
        if not cursor:
            break

    if not trades_by_symbol:
        if debug:
            print("No se encontraron trades en el rango de tiempo especificado.")
        return []

    # 2) Obtener funding fees
    funding_logs = _fetch_transaction_logs_cached(
        category=category,
        currency=currency,
        start_ms=start_ms,
        end_ms=end_ms,
        limit=50,
        max_pages=50
    )

    funding_by_base: Dict[str, List[dict]] = {}
    for f in funding_logs:
        base = normalize_symbol(f.get("symbol",""))
        if base not in funding_by_base:
            funding_by_base[base] = []
        funding_by_base[base].append(f)

    # 3) Reconstrucción FIFO por símbolo
    results: List[dict] = []
    eps_qty = 1e-6

    for base, items in trades_by_symbol.items():
        if not items:
            continue

        # Normalizar trades
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

            # 🔧 CORRECCIÓN: Funding en ventana [open_ts, close_ts]
            open_ts  = min(x["ts"] for x in bl)
            close_ts = max(x["ts"] for x in bl)
            f_sum_income = 0.0
            for f in funding_by_base.get(base, []):
                ts_f = _to_ms(f.get("transactionTime"))
                if open_ts <= ts_f <= close_ts:
                    # API: funding > 0 => pagado (gasto), < 0 => cobrado (ingreso)
                    # Invertimos para income style: +cobro / -pago
                    f_val = _to_float(f.get("funding") or 0.0)
                    f_sum_income += (-f_val)  # negativo API -> positivo income

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
    Guarda en SQLite usando verificación explícita de duplicados.
    """
    try:
        from db_manager import save_closed_position
    except Exception as e:
        raise RuntimeError(f"db_manager.save_closed_position no disponible: {e}")

    rows = fetch_bybit_closed_positions_fifo(
        days=days, category=category, currency=currency, symbol=symbol, debug=debug
    )
    
    # Conexión a la base de datos para verificar duplicados
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    saved = 0
    dup = 0
    for row in rows:
        try:
            # Verificar si ya existe esta posición (mismo patrón que BingX)
            cur.execute("""
                SELECT COUNT(*) FROM closed_positions 
                WHERE exchange = ? AND symbol = ? AND close_time = ? AND side = ?
            """, (row["exchange"], row["symbol"], row["close_time"], row["side"]))
            
            if cur.fetchone()[0] > 0:
                dup += 1
                if debug:
                    print(f"⏭️  Duplicado omitido: {row['exchange']} {row['symbol']} {row['side']} {row['close_time']}")
                continue
                
            # Si no existe, guardar
            save_closed_position(row)
            saved += 1
            if debug:
                print(f"✅ Guardada: {row['exchange']} {row['symbol']} {row['side']} {row['close_time']}")
                
        except Exception as e:
            msg = str(e).lower()
            if "unique" in msg or "duplicate" in msg:
                dup += 1
                if debug:
                    print(f"⏭️  Duplicado (por excepción): {row['exchange']} {row['symbol']}")
            else:
                print(f"❌ Error guardando posición {row.get('symbol')} (Bybit): {e}")
                # No re-lanzamos la excepción para continuar con las demás posiciones
    
    conn.close()
    
    if debug:
        print(f"✅ Bybit FIFO cerradas guardadas: {saved} (omitidas {dup})")
    return saved, dup

# ========== Debug rápido (ejecución directa) ==========
def _demo():
    print("== Bybit v9 FIXED demo (SIGNOS CORREGIDOS) ==")
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        print("❌ Define BYBIT_API_KEY/BYBIT_API_SECRET en variables de entorno.")
        return
    
    try:
        opens = fetch_bybit_open_positions(category="linear", settle_coin="USDT", enrich_funding=True)
        print(f"\n📊 OPEN POSITIONS count: {len(opens)}")
        for o in opens:
            print(f"  {o['symbol']:8} {o['side']:5} | funding_fee={o['funding_fee']:>10.4f} | fee={o['fee']:>10.4f}")
    except Exception as e:
        print("❌ opens:", e)

    try:
        funding_fees = fetch_bybit_funding_fees(category="linear", since=_now_ms()-7*24*3600*1000)
        print(f"\n💸 FUNDING FEES count: {len(funding_fees)}")
        for f in funding_fees[:10]:
            print(f"  {f['symbol']:8} | income={f['income']:>10.4f} | rate={f['funding_rate']:>8.6f}")
    except Exception as e:
        print("❌ funding fees:", e)

    try:
        closed = fetch_bybit_closed_positions_fifo(category="linear", days=7, symbol=None, currency="USDT", debug=True)
        print(f"\n📦 CLOSED POSITIONS (preview): {len(closed)}")
        for r in closed[:5]:
            print(f"  {r['symbol']:8} {r['side']:5} | pnl={r['pnl']:>10.2f} | funding={r['funding_total']:>10.2f} | realized={r['realized_pnl']:>10.2f}")
    except Exception as e:
        print("❌ closed fetch:", e)

if __name__ == "__main__":
    _demo()

__all__ = [
    "fetch_bybit_open_positions",
    "fetch_bybit_funding_fees",
    "fetch_bybit_closed_positions_fifo",
    "save_bybit_closed_positions",
]
