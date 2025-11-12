from __future__ import annotations
import os, time, hmac, hashlib, requests, sqlite3
from urllib.parse import urlencode
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import json
import time

# Cache de símbolos activos
SYMBOL_CACHE_FILE = "bingx_active_symbols_cache.json"
SYMBOL_CACHE_TTL = 7 * 24 * 60 * 60  # 7 días en segundos

def _update_symbol_cache(symbols: list):
    """Actualiza el caché con nuevos símbolos activos"""
    try:
        current_cache = _load_symbol_cache()
        current_time = time.time()
        
        # Añadir nuevos símbolos o actualizar timestamp de existentes
        for symbol in symbols:
            current_cache[symbol] = current_time
        
        # Eliminar símbolos expirados
        current_cache = {sym: ts for sym, ts in current_cache.items() 
                        if current_time - ts <= SYMBOL_CACHE_TTL}
        
        with open(SYMBOL_CACHE_FILE, 'w') as f:
            json.dump(current_cache, f)
            
    except Exception as e:
        print(f"⚠️ Error actualizando caché de símbolos: {e}")

def _load_symbol_cache() -> dict:
    """Carga el caché de símbolos desde archivo"""
    try:
        if os.path.exists(SYMBOL_CACHE_FILE):
            with open(SYMBOL_CACHE_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _get_cached_symbols() -> list:
    """Obtiene símbolos del caché que aún son válidos"""
    cache = _load_symbol_cache()
    current_time = time.time()
    valid_symbols = [sym for sym, ts in cache.items() 
                    if current_time - ts <= SYMBOL_CACHE_TTL]
    return valid_symbols

# --- utils comunes (compártelos desde utils.py en tu proyecto real) -------------
import re
def normalize_symbol(sym: str) -> str:
    if not sym: return ""
    s = sym.upper()
    s = re.sub(r'^PERP_', '', s)
    s = re.sub(r'(_|-)?(USDT|USDC|USD|-USDT|PERP)$', '', s)
    s = re.sub(r'[_-]+$', '', s)
    s = re.split(r'[_/-]', s)[0]
    return s
# -------------------------------------------------------------------------------

BINGX_BASE = "https://open-api.bingx.com"
BINGX_API_KEY = os.getenv("BINGX_API_KEY", "")
BINGX_SECRET_KEY = os.getenv("BINGX_SECRET_KEY", "")

# db helper (ajusta al tuyo)
try:
    from db_manager import save_closed_position  # tu helper existente
except Exception:
    def save_closed_position(_: Dict[str, Any]):
        raise RuntimeError("db_manager.save_closed_position no disponible")


#----------- fin de los modulos para importar funding usando websocket y listen key
def _sign_params(params: dict) -> dict:
    params["timestamp"] = int(time.time() * 1000)
    qs = urlencode(params)
    signature = hmac.new(BINGX_SECRET_KEY.encode("utf-8"), qs.encode("utf-8"), hashlib.sha256).hexdigest()
    params["signature"] = signature
    return params
# === DEBUG de autenticación ===
    # print("[DEBUG][BingX] Query string:", qs)
    # print("[DEBUG][BingX] Signature:", signature)
    # print("[DEBUG][BingX] API Key:", BINGX_API_KEY[:6] + "..." if BINGX_API_KEY else "MISSING")
    return params

def _get(path, params=None):
    headers = {"X-BX-APIKEY": BINGX_API_KEY}
    p = _sign_params(params or {})
    url = BINGX_BASE + path
    # print(f"[DEBUG][BingX] GET {url} params={p}")  # debug URL + params
    r = requests.get(url, params=p, headers=headers, timeout=15)
    # print("[DEBUG][BingX] Status:", r.status_code)
    # print("[DEBUG][BingX] Raw response:", r.text[:500])  # evita prints enormes
    r.raise_for_status()
    return r.json()
# Balances
# ======================
# Balances (corregido - función de módulo)
def fetch_bingx_all_balances() -> Dict[str, Any]:
    """
    GET /openApi/swap/v3/user/balance
    - equity: incluye PnL no realizado (NAV)
    - balance: saldo "a pelo" sin PnL
    - unrealized_pnl: PnL no realizado
    """
    try:
        payload = _get("/openApi/swap/v3/user/balance")
        rows = payload.get("data", []) if isinstance(payload, dict) else (payload or [])
        if isinstance(rows, dict):
            # por si algún entorno devuelve data como dict
            rows = rows.get("balances") or rows.get("assets") or []

        spot_usd = 0.0
        margin_usd = 0.0

        fut_balance = 0.0          # saldo de la cuenta de perps (sin PnL)
        fut_equity = 0.0           # NAV de perps (con PnL no realizado)
        fut_unreal = 0.0           # PnL no realizado
        fut_used_margin = 0.0      # margen usado (opcional)

        for b in rows:
            asset = str(b.get("asset") or "").upper()
            bal   = float(b.get("balance") or 0)
            eq    = float(b.get("equity") or 0)
            un    = float(b.get("unrealizedProfit") or 0)
            used  = float(b.get("usedMargin") or 0)

            if asset in ("USDT", "USDC"):
                fut_balance += bal
                # si equity viniera 0, calculamos fallback: balance + unrealized
                fut_equity += eq if eq != 0 else (bal + un)
                fut_unreal += un
                fut_used_margin += used

        return {
            "exchange": "bingx",
            # Equity TOTAL del exchange debe incluir el PnL no realizado
            "equity": spot_usd + margin_usd + fut_equity,
            # Balance “limpio” sin PnL (lo usas para mostrar el saldo a secas)
            "balance": spot_usd + margin_usd + fut_balance,
            "unrealized_pnl": fut_unreal,
            "initial_margin": fut_used_margin,
            "spot": spot_usd,
            "margin": margin_usd,
            # Dejamos 'futures' como saldo sin PnL; el PnL se muestra aparte
            "futures": fut_balance,
        }
    except Exception as e:
        print(f"❌ BingX balance error: {e}")
        return {
            "exchange": "bingx",
            "equity": 0.0,
            "balance": 0.0,
            "unrealized_pnl": 0.0,
            "initial_margin": 0.0,
            "spot": 0.0,
            "margin": 0.0,
            "futures": 0.0,
        }
# ======================
# Posiciones
# ======================

    

# ======================
# Closed Positions
# ======================  

# ---------- Helpers de símbolo/num ----------
def _bx_to_dash(sym: str) -> str:
    """BTCUSDT -> BTC-USDT (lo exige el endpoint positionHistory)."""
    if not sym:
        return sym
    s = sym.upper()
    if "-" in s:
        return s
    for q in ("USDT", "USDC"):
        if s.endswith(q):
            return s[:-len(q)] + "-" + q
    return s

def _bx_no_dash(sym: str) -> str:
    """BTC-USDT -> BTCUSDT (consistente con tu DB/UI)."""
    return (sym or "").upper().replace("-", "")
def _uniq(seq):
    seen=set(); out=[]
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _num(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# =========================
# Helpers de tiempo
# =========================
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

def _ms_to_str(ms: Optional[int | float | str]) -> str:
    try:
        if ms in (None, "", "N/A"): return "N/A"
        ms = int(float(ms))
        # BingX usa ms-epoch
        dt = datetime.fromtimestamp(ms/1000, tz=timezone.utc)
        # Muestra en local si prefieres: .astimezone()
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(ms)

def _f(x, d=0.0):
    try: return float(x)
    except: return d

def _i(x, d=0):
    try: return int(x)
    except: return d


# FUNDING: wrapper con tiempos legibles + compat de parámetros
# ============================================================

def fetch_funding_bingx(limit=100, start_time=None, end_time=None, symbol: str | None = None):
    """
    /openApi/swap/v2/user/income  (incomeType=FUNDING_FEE)
    Devuelve: exchange, symbol (SIN guion), income(float), asset, timestamp(ms), type
    Nota: el campo correcto es 'income' (string), NO 'amount'.
    """
    try:
        params = {
            "incomeType": "FUNDING_FEE",
            "limit": min(int(limit), 1000),
        }
        if start_time: params["startTime"] = int(start_time)
        if end_time:   params["endTime"]   = int(end_time)
        # si filtras por símbolo, la API exige guion (BTC-USDT)
        if symbol:
            sym = symbol.upper()
            if "-" not in sym:
                if sym.endswith("USDT"): sym = sym[:-4] + "-USDT"
                elif sym.endswith("USDC"): sym = sym[:-4] + "-USDC"
            params["symbol"] = sym

        data = _get("/openApi/swap/v2/user/income", params=params)
        recs = data.get("data", []) if isinstance(data, dict) else []
        out = []
        for r in recs:
            # 'income' viene como string; conviértelo con tu helper _f
            inc = _f(r.get("income"))  # <----- CORRECCIÓN AQUÍ
            # por compatibilidad con posibles variantes antiguas, fallback a 'amount'
            if inc == 0.0 and (r.get("amount") is not None):
                inc = _f(r.get("amount"))

            out.append({
                "exchange": "bingx",
                # normalizamos SIN guion para tu UI/back (tú decides)
                "symbol": (r.get("symbol") or "").replace("-", "").upper(),
                "income": inc,
                "asset": r.get("asset", "USDT"),
                "timestamp": _i(r.get("time")),
                "type": r.get("incomeType", "FUNDING_FEE"),
                "info": r.get("info"),
                "trade_id": r.get("tradeId"),
                "tran_id": r.get("tranId"),
            })
        return out
    except Exception as e:
        print(f"❌ BingX funding error: {e}")
        return []

def fetch_funding_bingx_readable(limit=100, start_time=None, end_time=None, symbol: str | None = None):
    rows = fetch_funding_bingx(limit=limit, start_time=start_time, end_time=end_time, symbol=symbol)
    for r in rows:
        r["timestamp_hr"] = _ms_to_str(r.get("timestamp"))
    return rows

# Compat retro: algunas partes antiguas llamaban con start_ms/end_ms
def _bingx_fetch_funding_ms(start_ms=None, end_ms=None, limit=1000) -> List[Dict[str, Any]]:
    return fetch_funding_bingx(limit=limit, start_time=start_ms, end_time=end_ms)


# ============================================================
# OPEN POSITIONS: pedir todos los campos que devuelve la API
# ============================================================

def fetch_bingx_open_positions() -> List[Dict[str, Any]]:
    """
    GET /openApi/swap/v2/user/positions
    Incluye campos extra: positionId, isolated, availableAmt, initialMargin, margin,
    positionValue, riskRate, maxMarginReduction, pnlRatio, updateTime, etc.
    """
    try:
        data = _get("/openApi/swap/v2/user/positions")
        rows = data.get("data", []) if isinstance(data, dict) else []
        out: List[Dict[str, Any]] = []
        for pos in rows:
            raw_symbol = (pos.get("symbol") or "").upper()
            symbol = raw_symbol.replace("-USDT", "").replace("USDT", "")
            qty = _f(pos.get("positionAmt"))
            side = (pos.get("positionSide") or "").lower()
            if side not in ("long", "short"):
                side = "long" if qty >= 0 else "short"

            entry = _f(pos.get("avgPrice"))
            mark = _f(pos.get("markPrice"))
            unreal = _f(pos.get("unrealizedProfit"))
            realized = _f(pos.get("realisedProfit"))
            funding_raw = pos.get("cumFundingFee") or pos.get("fundingFee") or pos.get("funding") or 0
            funding_fee = _f(funding_raw)

            # Extra API fields (los dejamos en top-level para que tu UI pueda usarlos)
            out.append({
                "exchange": "bingx",
                "symbol": symbol,
                "symbol_raw": raw_symbol,
                "position_id": pos.get("positionId"),
                "side": side,
                "size": abs(qty),
                "entry_price": entry,
                "mark_price": mark,
                "unrealized_pnl": unreal,
                "realized_pnl": realized,
                "funding_fee": funding_fee,
                "leverage": _f(pos.get("leverage")),
                "notional": _f(pos.get("positionValue")),  # mejor que positionInitialMargin
                "liquidation_price": _f(pos.get("liquidationPrice")),
                "isolated": bool(pos.get("isolated")) if pos.get("isolated") is not None else None,
                "available_amt": _f(pos.get("availableAmt")),
                "initial_margin": _f(pos.get("initialMargin")),
                "margin": _f(pos.get("margin")),
                "risk_rate": pos.get("riskRate"),
                "max_margin_reduction": pos.get("maxMarginReduction"),
                "pnl_ratio": pos.get("pnlRatio"),
                "update_time": _i(pos.get("updateTime")),
                "update_time_hr": _ms_to_str(pos.get("updateTime")),
            })

        # ACTUALIZAR CACHÉ CON SÍMBOLOS ABIERTOS
        try:
            if out:  # Si hay posiciones abiertas
                open_symbols = [pos['symbol_raw'] for pos in out]
                _update_symbol_cache(open_symbols)
        except Exception as e:
            print(f"[CACHE] Error actualizando caché: {e}")

        return out  # ← ¡IMPORTANTE: Devuelve las posiciones!
    except Exception as e:
        print(f"❌ BingX positions error: {e}")
        return []

# ============================================================
# CLOSED POSITIONS: pedir más campos + arreglar include_funding
# ============================================================


# ============getting Symbols from market data============
# === PUBLIC GET (sin firma) para endpoints públicos de BingX ===
def _get_public(path: str, params: Optional[dict] = None) -> dict:
    p = dict(params or {})
    # este endpoint exige 'timestamp' aunque no requiera firma
    p.setdefault("timestamp", int(time.time() * 1000))
    url = BINGX_BASE + path
    r = requests.get(url, params=p, timeout=15)
    r.raise_for_status()
    return r.json()

def fetch_bingx_perp_symbols(debug: bool = False) -> List[str]:
    """
    Devuelve la lista de símbolos USDT-M Perp desde /openApi/swap/v2/quote/contracts
    en formato con guion (p.ej. 'BTC-USDT'), que es el que exige positionHistory.
    """
    try:
        payload = _get_public("/openApi/swap/v2/quote/contracts")
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        # algunos payloads traen { "data": { "contracts": [...] } }
        if isinstance(rows, dict):
            rows = rows.get("contracts", [])
        symbols = []
        for it in rows:
            sym = (it.get("symbol") or it.get("contractSymbol") or "").upper()
            if not sym:
                continue
            # Normalizamos a 'XXX-USDT'
            if "-" not in sym:
                # si viniera como BTCUSDT => BTC-USDT
                if sym.endswith("USDT"):
                    sym = sym[:-4] + "-USDT"
                elif sym.endswith("USDC"):
                    sym = sym[:-4] + "-USDC"
            symbols.append(sym)
        # únicos y ordenados
        symbols = sorted(set(symbols))
        if debug:
            print(f"🔎 BingX contracts: {len(symbols)} símbolos (ej: {symbols[:10]})")
        return symbols
    except Exception as e:
        print(f"❌ fetch_bingx_perp_symbols error: {e}")
        return []
# ========================================



import time as _time
import json, os, sqlite3


def _collect_active_symbols_for_closed(days: int = 30, debug: bool = False, max_symbols: int | None = None) -> list[str]:
    """
    SOLUCIÓN MEJORADA: Usa caché de símbolos activos + fallback a contratos
    """
    # 1) Primero usar símbolos del caché (más eficiente)
    cached_symbols = _get_cached_symbols()
    
    if debug:
        print(f"📋 Símbolos en caché: {len(cached_symbols)}")
    
    # Si hay símbolos en caché, usarlos directamente
    if cached_symbols:
        symbols = cached_symbols
        if debug:
            print(f"🎯 Usando {len(symbols)} símbolos del caché")
    else:
        # 2) Fallback: todos los contratos disponibles
        if debug:
            print("ℹ️ Caché vacío, usando todos los contratos")
        symbols = fetch_bingx_perp_symbols(debug=debug)
    
    # 3) Limitar si es necesario (pero con límite más alto)
    if max_symbols is not None and len(symbols) > max_symbols:
        symbols = symbols[:max_symbols]
        if debug:
            print(f"🔒 Limitado a {max_symbols} símbolos")

    if debug:
        print(f"🧾 Símbolos finales para CLOSED: {len(symbols)}")
        print("   " + ", ".join(symbols[:20]) + (" …" if len(symbols) > 20 else ""))

    return symbols

def fetch_closed_positions_bingx(
    symbols=None,
    days: int = 30,
    include_funding: bool = True,
    page_size: int = 200,
    debug: bool = False,
    autodetect_when_empty: bool = True,
    autodetect_max_symbols: int | None = 100,   # evita barrer los 500+ por defecto
    throttle_ms: int = 200,                   # evita rate limit
) -> list[dict]:
    """
    GET /openApi/swap/v1/trade/positionHistory
    Si 'symbols' es None o [], y 'autodetect_when_empty' está True:
      - construye una lista de símbolos "activos" (open + funding) y si está vacía, fallback a contracts.
    Además imprime cada símbolo y cuántas filas se obtienen.
    """
    import time as _time, json

    # === NUEVO: autodetección ===
    if (not symbols) and autodetect_when_empty:
        symbols = _collect_active_symbols_for_closed(days=days, debug=debug, max_symbols=autodetect_max_symbols)
        if not symbols:
            print("⚠️ [BingX] No hay símbolos a consultar (autodetección vacía).")
            return []

    symbols = list(symbols or [])
    now_ms = int(_time.time() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000
    results: list[dict] = []

    # funding opcional (indexado por símbolo sin guion)
    funding_idx = {}
    if include_funding:
        try:
            f_all = fetch_funding_bingx(limit=1000, start_time=start_ms, end_time=now_ms)
            from collections import defaultdict as _dd
            tmp = _dd(list)
            for f in f_all:
                tmp[(f.get("symbol") or "").upper().replace("-", "")].append(f)
            funding_idx = dict(tmp)
            if debug:
                print(f"ℹ️ funding records: {sum(len(v) for v in funding_idx.values())}")
        except Exception as e:
            if debug: print(f"[funding preload] error: {e}")

    scanned = 0
    for sym in symbols:
        sym_dash = _bx_to_dash(sym)
        scanned += 1
        if debug:
            print(f"[BingX] positionHistory {sym_dash}  ({scanned}/{len(symbols)})   window={_ms_to_str(start_ms)}→{_ms_to_str(now_ms)}")

        page = 1
        while True:
            params = {
                "symbol": sym_dash,
                "startTs": int(start_ms),
                "endTs": int(now_ms),
                "pageId": page,   # si no devuelve, prueba pageIndex en tu entorno
                "pageSize": int(page_size),
                "recvWindow": 5000,
            }
            payload = _get("/openApi/swap/v1/trade/positionHistory", params)

            data_list = []
            if isinstance(payload, dict):
                if isinstance(payload.get("data"), dict):
                    data_list = payload["data"].get("positionHistory", [])
                elif isinstance(payload.get("data"), list):
                    data_list = payload["data"]
            count = len(data_list) if data_list else 0
            if debug:
                print(f"   • {sym_dash}: {count} filas (page={page})")
                if page == 1 and count:
                    # imprime la primera fila con fechas legibles
                    r0 = data_list[0]
                    print(f"     ⏱ first: open={_ms_to_str(r0.get('openTime'))}  close={_ms_to_str(r0.get('updateTime'))}")

            if not data_list:
                break

            # normalización (igual que tu versión que funcionaba)
            for row in data_list:
                try:
                    open_ms  = _i(row.get("openTime"))
                    close_ms = _i(row.get("updateTime"))
                    qty_close = _f(row.get("closePositionAmt") or row.get("positionAmt"))
                    entry_price = _f(row.get("avgPrice"))
                    close_price = _f(row.get("avgClosePrice"))
                    price_pnl   = _f(row.get("realisedProfit"))
                    net_profit  = row.get("netProfit")
                    net_profit  = _f(net_profit) if net_profit not in (None, "") else None
                    funding_total = _f(row.get("totalFunding"))
                    fee_total     = _f(row.get("positionCommission"))
                    lev           = _f(row.get("leverage"))
                    side          = (row.get("positionSide") or "").lower() or "closed"

                    realized_pnl = net_profit if net_profit is not None else (price_pnl + funding_total + fee_total)
                    pos_id = row.get("positionId")
                    isolated = row.get("isolated")
                    close_all = row.get("closeAllPositions")
                    pos_amt = _f(row.get("positionAmt"))

                    results.append({
                        "exchange": "bingx",
                        "symbol": _bx_no_dash(row.get("symbol") or sym_dash),
                        "symbol_raw": row.get("symbol") or sym_dash,
                        "position_id": pos_id,
                        "side": side,
                        "size": abs(qty_close),
                        "entry_price": entry_price,
                        "close_price": close_price,
                        "open_time": int(open_ms/1000) if open_ms else None,
                        "close_time": int(close_ms/1000) if close_ms else None,
                        "open_time_hr": _ms_to_str(open_ms),
                        "close_time_hr": _ms_to_str(close_ms),
                        "realized_pnl": realized_pnl,
                        "pnl": price_pnl,
                        "funding_total": funding_total,
                        "fee_total": fee_total,
                        "fee": fee_total,
                        "notional": entry_price * abs(qty_close),
                        "leverage": lev,
                        "isolated": bool(isolated) if isolated is not None else None,
                        "close_all_positions": bool(close_all) if close_all is not None else None,
                        "position_amt": pos_amt,
                    })
                except Exception as e:
                    if debug:
                        print(f"   [WARN] fila malformada {row}: {e}")
                    continue

            if len(data_list) < page_size:
                break
            page += 1

        # pequeño throttle para no chocar el rate-limit
        _time.sleep(throttle_ms / 1000.0)

    if debug:
        print(f"✅ BingX closed positions totales: {len(results)}")
    return results



# ============================================================
# SAVE WRAPPER público (firma que usa portfoliov6.6)
# ============================================================

def save_bingx_closed_positions(
    db_path="portfolio.db",
    symbols=None,
    days=30,
    include_funding=True,
    debug=False,
) -> None:
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return

    # Si no pasan símbolos, autodetecta (open + funding + fallback)
    if not symbols:
        symbols = _collect_active_symbols_for_closed(days=days, debug=debug, max_symbols=64)
        if not symbols:
            print("⚠️ No hay símbolos para consultar y guardar.")
            return

    positions = fetch_closed_positions_bingx(
        symbols=symbols,
        days=days,
        include_funding=include_funding,
        page_size=200,
        debug=debug,
        autodetect_when_empty=False,  # ya los pasamos
    )
    if not positions:
        print("⚠️ No se obtuvieron posiciones cerradas de BingX.")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    saved = skipped = 0
    for pos in positions:
        try:
            cur.execute("""SELECT COUNT(*) FROM closed_positions
                           WHERE exchange=? AND symbol=? AND close_time=?""",
                        (pos["exchange"], pos["symbol"], pos["close_time"]))
            if cur.fetchone()[0]:
                skipped += 1
                continue
            save_closed_position(pos)
            saved += 1
        except Exception as e:
            print(f"⚠️ Error guardando {pos.get('symbol')} (BingX): {e}")
    conn.close()
    print(f"✅ BingX guardadas: {saved} | omitidas (duplicadas): {skipped}")


def debug_bingx_scan_closed(
    days: int = 30,
    max_symbols: Optional[int] = None,
    symbol_filter: Optional[str] = None,   # regex o prefijo (usa ^BTC para prefijo)
    only_active: bool = False,             # cruza con funding para reducir
    sleep_ms: int = 250,                   # throttling entre requests
    debug_payload: bool = False            # imprime RAW del primer símbolo con datos
) -> List[Dict[str, Any]]:
    """
    Escanea /positionHistory por todos (o un subconjunto) de símbolos USDT-M Perp.
    - Imprime la lista de símbolos detectados
    - Por símbolo: nº de filas y primera/última con fechas legibles
    - Devuelve lista concatenada de todas las filas normalizadas (tu formato)
    """
    now_ms = int(_time.time() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000

    # 1) base: todos los símbolos de contracts
    symbols = fetch_bingx_perp_symbols(debug=True)

    # 2) filtro activo (funding income) opcional
    if only_active:
        print("🧮 Filtrando por símbolos con actividad de funding (incomeType=FUNDING_FEE)...")
        active = fetch_funding_bingx(limit=1000, start_time=start_ms, end_time=now_ms)
        act_set = { (r.get("symbol") or "").upper().replace("-", "") for r in active }
        # convertir a formato con guion
        act_dash = set()
        for s in act_set:
            if s.endswith("USDT"):
                act_dash.add(s[:-4] + "-USDT")
            elif s.endswith("USDC"):
                act_dash.add(s[:-4] + "-USDC")
        before = len(symbols)
        symbols = [s for s in symbols if s in act_dash]
        print(f"   Activos: {len(symbols)} (de {before})")

    # 3) filtro por regex/prefijo
    if symbol_filter:
        rx = re.compile(symbol_filter, re.IGNORECASE)
        before = len(symbols)
        symbols = [s for s in symbols if rx.search(s)]
        print(f"🔎 Filtro '{symbol_filter}': {len(symbols)} (de {before})")

    # 4) limitar cantidad
    if max_symbols is not None:
        symbols = symbols[:max_symbols]

    if not symbols:
        print("⚠️ No hay símbolos que escanear.")
        return []

    print("🧾 Símbolos a escanear:")
    print("   " + ", ".join(symbols[:30]) + (" …" if len(symbols) > 30 else ""))

    all_rows: List[Dict[str, Any]] = []
    first_raw_dumped = False
    scanned = 0

    for sym_dash in symbols:
        scanned += 1
        # request
        params = {
            "symbol": sym_dash,
            "startTs": int(start_ms),
            "endTs": int(now_ms),
            "pageIndex": 1,
            "pageSize": 200,
            "recvWindow": 5000,
        }
        payload = _get("/openApi/swap/v1/trade/positionHistory", params)

        # extraer lista
        data_list = []
        if isinstance(payload, dict):
            if isinstance(payload.get("data"), dict):
                data_list = payload["data"].get("positionHistory", [])
            elif isinstance(payload.get("data"), list):
                data_list = payload["data"]
        count = len(data_list) if data_list else 0

        print(f"• {scanned:>3}/{len(symbols)} {sym_dash}: {count} filas")

        if count > 0:
            # dump RAW sólo del primer símbolo que tenga datos
            if debug_payload and not first_raw_dumped:
                import json
                print("   ── RAW payload (primer símbolo con filas) ──")
                print(json.dumps(payload, indent=2, ensure_ascii=False))
                first_raw_dumped = True

            # primeras/últimas con fechas legibles
            first = data_list[0]
            last  = data_list[-1]
            f_ot, f_ct = first.get("openTime"), first.get("updateTime")
            l_ot, l_ct = last.get("openTime"),  last.get("updateTime")
            print(f"   ⏱  first: open={_ms_to_str(f_ot)}  close={_ms_to_str(f_ct)}")
            print(f"   ⏱   last: open={_ms_to_str(l_ot)}  close={_ms_to_str(l_ct)}")

            # normaliza como en fetch_closed_positions_bingx
            for row in data_list:
                try:
                    open_ms  = _i(row.get("openTime"))
                    close_ms = _i(row.get("updateTime"))
                    qty_close = _f(row.get("closePositionAmt") or row.get("positionAmt"))
                    entry_price = _f(row.get("avgPrice"))
                    close_price = _f(row.get("avgClosePrice"))
                    price_pnl   = _f(row.get("realisedProfit"))
                    net_profit  = row.get("netProfit")
                    net_profit  = _f(net_profit) if net_profit not in (None, "") else None
                    funding_total = _f(row.get("totalFunding"))
                    fee_total     = _f(row.get("positionCommission"))
                    lev           = _f(row.get("leverage"))
                    side          = (row.get("positionSide") or "").lower() or "closed"

                    realized_pnl = net_profit if net_profit is not None else (price_pnl + funding_total + fee_total)
                    pos_id = row.get("positionId")
                    isolated = row.get("isolated")
                    close_all = row.get("closeAllPositions")
                    pos_amt = _f(row.get("positionAmt"))

                    all_rows.append({
                        "exchange": "bingx",
                        "symbol": (row.get("symbol") or "").replace("-", ""),
                        "symbol_raw": row.get("symbol"),
                        "position_id": pos_id,
                        "side": side,
                        "size": abs(qty_close),
                        "entry_price": entry_price,
                        "close_price": close_price,
                        "open_time": int(open_ms/1000) if open_ms else None,
                        "close_time": int(close_ms/1000) if close_ms else None,
                        "open_time_hr": _ms_to_str(open_ms),
                        "close_time_hr": _ms_to_str(close_ms),
                        "realized_pnl": realized_pnl,
                        "pnl": price_pnl,
                        "funding_fee": funding_total,
                        "fee": fee_total,
                        "notional": entry_price * abs(qty_close),
                        "leverage": lev,
                        "isolated": bool(isolated) if isolated is not None else None,
                        "close_all_positions": bool(close_all) if close_all is not None else None,
                        "position_amt": pos_amt,
                    })
                except Exception as e:
                    print(f"   [WARN] fila malformada en {sym_dash}: {e}")
                    continue

        # throttle básico
        _time.sleep(sleep_ms / 1000.0)

    print(f"✅ Escaneo completado. Total filas recogidas: {len(all_rows)}")
    return all_rows

def force_cache_update():
    """Fuerza la actualización del caché con posiciones abiertas actuales"""
    print("🔄 Actualizando caché de símbolos...")
    positions = fetch_bingx_open_positions()
    if positions:
        symbols = [pos['symbol_raw'] for pos in positions]
        _update_symbol_cache(symbols)
        print(f"✅ Caché actualizado con {len(symbols)} símbolos: {symbols}")
    else:
        print("ℹ️ No hay posiciones abiertas para actualizar caché")
        
if __name__ == "__main__":
    # ... código existente ...
    
    # Actualizar caché primero
    force_cache_update()
    
    # Ver qué hay en caché
    cached = _get_cached_symbols()
    print(f"📋 Símbolos en caché: {len(cached)}")
    if cached:
        print(f"   Ejemplos: {cached[:10]}")
#============================================================
# DEBUG ejecutable (no usa start_ms erróneo y muestra fechas)
# ============================================================

if __name__ == "__main__":
    import json

    print("=== DEBUG BINGX - POSITIONS ABIERTAS (RAW + FECHAS) ===")
    try:
        opens = fetch_bingx_open_positions()
        print(json.dumps(opens, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"[DEBUG ERROR] open positions: {e}")

    test_symbols = ["KAITO-USDT","MYX-USDT","GIGGLE-USDT", "AT-USDT", "ENSO-USDT"]
    for s in test_symbols:
        print(f"\n=== DEBUG BINGX - POSITION_HISTORY {s} (últimos 30 días) ===")
        try:
            res = fetch_closed_positions_bingx(symbols=[s], days=30, include_funding=True, page_size=200, debug=True)
            print(f"Resultado filas: {len(res)}")
            print(json.dumps(res, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"[DEBUG ERROR] closed positions {s}: {e}")

    print("\n=== DEBUG BINGX - FUNDING (últimos 100) RAW ===")
    try:
        fund_raw = fetch_funding_bingx(limit=1)
        print(json.dumps(fund_raw[:50], indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"[DEBUG ERROR] funding raw: {e}")

    print("\n=== DEBUG BINGX - FUNDING (con fechas legibles) ===")
    try:
        fund_nice = fetch_funding_bingx_readable(limit=2)
        print(json.dumps(fund_nice, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"[DEBUG ERROR] funding readable: {e}")
        
if __name__ == "__main__":
    # Escanear TODO el mercado (ojo con rate limit). Consejo: empieza con filtros.
    rows = debug_bingx_scan_closed(
        days=30,
        max_symbols=20,            # prueba con 20 primero
        symbol_filter=None,        # por ejemplo: r"^(BTC|ETH|KAITO)-USDT$"
        only_active=False,         # True para cruzar con funding e ir solo a activos
        sleep_ms=250,
        debug_payload=True         # muestra RAW del primer símbolo que devuelva filas
    )
    print(f"Filas totales normalizadas: {len(rows)}")    
    
if __name__ == "__main__":
    rows = fetch_closed_positions_bingx(
        symbols=None,      # ← forzamos autodetección
        days=30,
        include_funding=True,
        debug=True,        # ← imprime símbolos y conteos
        autodetect_when_empty=True,
        autodetect_max_symbols=200,  # prueba con 64 primero
        throttle_ms=200,
    )
    print(f"TOTAL filas: {len(rows)}")
    
# # ============================================================
# # DEBUG ESPECÍFICO PARA POSITION HISTORY (SPYDER) (DEEPSEEK)
# # ============================================================

# def debug_position_history_raw(symbols: list = None, days: int = 30):
#     """
#     Debug especial para ver la respuesta RAW del endpoint positionHistory
#     """
#     import json
#     from datetime import datetime
    
#     if symbols is None:
#         symbols = ["KAITO-USDT", "MYX-USDT", "GIGGLE-USDT", "AT-USDT",  "ENSO-USDT"]  # Tus símbolos de prueba
    
#     now_ms = int(time.time() * 1000)
#     start_ms = now_ms - days * 24 * 60 * 60 * 1000
    
#     print(f"🔍 DEBUG POSITION HISTORY RAW")
#     print(f"⏰ Rango: {_ms_to_str(start_ms)} → {_ms_to_str(now_ms)}")
#     print(f"📋 Símbolos: {symbols}")
#     print("=" * 80)
    
#     for symbol in symbols:
#         print(f"\n🎯 Consultando: {symbol}")
        
#         # Primera página
#         params = {
#             "symbol": symbol,
#             "startTs": int(start_ms),
#             "endTs": int(now_ms),
#             "pageId": 1,
#             "pageSize": 200,
#             "recvWindow": 5000,
#         }
        
#         try:
#             payload = _get("/openApi/swap/v1/trade/positionHistory", params)
            
#             print(f"✅ Status: ÉXITO")
#             print(f"📊 Estructura respuesta: {type(payload)}")
            
#             if isinstance(payload, dict):
#                 print(f"📦 Keys en payload: {list(payload.keys())}")
                
#                 # Ver estructura de data
#                 data = payload.get("data", {})
#                 print(f"📁 Tipo de 'data': {type(data)}")
                
#                 if isinstance(data, dict):
#                     print(f"🔑 Keys en 'data': {list(data.keys())}")
#                     position_history = data.get("positionHistory", [])
#                 elif isinstance(data, list):
#                     position_history = data
#                     print(f"📊 'data' es lista directa con {len(position_history)} elementos")
#                 else:
#                     position_history = []
#                     print(f"❓ 'data' es de tipo: {type(data)}")
                
#                 print(f"📈 Elementos en positionHistory: {len(position_history)}")
                
#                 # Mostrar primeros 3 elementos en crudo
#                 if position_history:
#                     print(f"\n📋 Primeros 3 elementos RAW:")
#                     for i, pos in enumerate(position_history[:3]):
#                         print(f"  [{i}] {json.dumps(pos, indent=2, ensure_ascii=False)}")
#                 else:
#                     print("❌ No hay datos en positionHistory")
                    
#                 # Mostrar códigos de error si existen
#                 if payload.get("code"):
#                     print(f"🚨 Código error: {payload.get('code')}")
#                 if payload.get("msg"):
#                     print(f"📢 Mensaje: {payload.get('msg')}")
                    
#             else:
#                 print(f"❓ Payload no es dict: {type(payload)}")
#                 print(f"📄 Contenido: {payload}")
                
#         except Exception as e:
#             print(f"❌ ERROR: {e}")
#             import traceback
#             traceback.print_exc()
        
#         print("-" * 60)
#         time.sleep(0.5)  # Throttle

# def debug_funding_for_symbols(symbols: list = None, limit: int = 50):
#     """
#     Debug para ver funding de símbolos específicos
#     """
#     if symbols is None:
#         symbols = ["KAITO", "MYX", "GIGGLE"]
    
#     print(f"\n💰 DEBUG FUNDING PARA SÍMBOLOS")
#     print("=" * 60)
    
#     for symbol in symbols:
#         print(f"\n🔍 Símbolo: {symbol}")
#         funding = fetch_funding_bingx(limit=limit, symbol=f"{symbol}USDT")
#         print(f"📊 Registros funding: {len(funding)}")
        
#         if funding:
#             for f in funding[:5]:  # Primeros 5
#                 print(f"  • {f['symbol']}: {f['income']} {f['asset']} @ {_ms_to_str(f['timestamp'])}")
#         else:
#             print("  ❌ Sin registros de funding")

# if __name__ == "__main__":
#     # ... (tu código existente)
    
#     # NUEVO DEBUG PARA SPYDER
#     print("\n" + "="*80)
#     print("🔍 DEBUG ESPECÍFICO POSITION HISTORY - SPYDER")
#     print("="*80)
    
#     # Cambia estos símbolos por los que no te están cargando
#     problematic_symbols = ["MYX-USDT", "GIGGLE-USDT", "AT-USDT", "ENSO-USDT"]  # Tus símbolos problemáticos
    
#     debug_position_history_raw(symbols=problematic_symbols, days=60)
#     debug_funding_for_symbols(symbols=["MYX", "GIGGLE"])