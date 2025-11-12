from __future__ import annotations
import os, time, hmac, hashlib, requests, sqlite3, json, re
from urllib.parse import urlencode
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv

# ============================================================
# CONFIGURACIÓN Y CONSTANTES
# ============================================================

load_dotenv()
BINGX_BASE = "https://open-api.bingx.com"
BINGX_API_KEY = os.getenv("BINGX_API_KEY", "")
BINGX_SECRET_KEY = os.getenv("BINGX_SECRET_KEY", "")

# Cache de símbolos activos
SYMBOL_CACHE_FILE = "bingx_active_symbols_cache.json"
SYMBOL_CACHE_TTL = 7 * 24 * 60 * 60  # 7 días

# ============================================================
# HELPERS BÁSICOS
# ============================================================

def normalize_symbol(sym: str) -> str:
    """Normaliza símbolos quitando USDT/USDC y otros sufijos"""
    if not sym: return ""
    s = sym.upper()
    
    # Quitar prefijos y sufijos
    s = re.sub(r'^PERP_', '', s)
    s = re.sub(r'[-_]?USDT$', '', s)
    s = re.sub(r'[-_]?USDC$', '', s)
    s = re.sub(r'[-_]?USD$', '', s)
    s = re.sub(r'[-_]?PERP$', '', s)
    s = re.sub(r'[-_/]+$', '', s)
    
    # Tomar primera parte si hay separadores
    if re.search(r'[-_/]', s):
        s = re.split(r'[-_/]', s)[0]
    
    return s

def _bx_to_dash(sym: str) -> str:
    """Convierte BTCUSDT -> BTC-USDT para la API"""
    if not sym: return sym
    s = sym.upper()
    if "-" in s: return s
    for q in ("USDT", "USDC"):
        if s.endswith(q):
            return s[:-len(q)] + "-" + q
    return s

def _bx_no_dash(sym: str) -> str:
    """Convierte BTC-USDT -> BTCUSDT"""
    return (sym or "").upper().replace("-", "")

def _ms_to_str(ms: Optional[int | float | str]) -> str:
    """Convierte timestamp ms a string legible"""
    try:
        if ms in (None, "", "N/A"): return "N/A"
        ms = int(float(ms))
        dt = datetime.fromtimestamp(ms/1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(ms)

def _f(x, d=0.0):
    try: return float(x)
    except: return d

def _i(x, d=0):
    try: return int(x)
    except: return d

def _uniq(seq):
    seen=set(); out=[]
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

# ============================================================
# CACHÉ DE SÍMBOLOS ACTIVOS
# ============================================================

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
    return [sym for sym, ts in cache.items() if current_time - ts <= SYMBOL_CACHE_TTL]

def _update_symbol_cache(symbols: list):
    """Actualiza el caché con nuevos símbolos activos"""
    try:
        current_cache = _load_symbol_cache()
        current_time = time.time()
        
        for symbol in symbols:
            current_cache[symbol] = current_time
        
        # Eliminar expirados
        current_cache = {sym: ts for sym, ts in current_cache.items() 
                        if current_time - ts <= SYMBOL_CACHE_TTL}
        
        with open(SYMBOL_CACHE_FILE, 'w') as f:
            json.dump(current_cache, f)
    except Exception as e:
        print(f"⚠️ Error actualizando caché: {e}")

def debug_cache_status():
    """Muestra el estado actual del caché"""
    cached = _get_cached_symbols()
    print(f"🔍 ESTADO DEL CACHÉ: {len(cached)} símbolos")
    if cached:
        print(f"   Símbolos: {cached[:10]}{'...' if len(cached) > 10 else ''}")
    else:
        print("   ❌ Caché vacío")

def force_cache_update():
    """Fuerza la actualización del caché con posiciones abiertas actuales"""
    print("🔄 Actualizando caché de símbolos...")
    positions = fetch_bingx_open_positions()
    if positions:
        symbols = [pos['symbol_raw'] for pos in positions]
        _update_symbol_cache(symbols)
        print(f"✅ Caché actualizado con {len(symbols)} símbolos")
    else:
        print("ℹ️ No hay posiciones abiertas para actualizar caché")

# ============================================================
# AUTENTICACIÓN Y REQUESTS
# ============================================================

def _sign_params(params: dict) -> dict:
    params["timestamp"] = int(time.time() * 1000)
    qs = urlencode(params)
    signature = hmac.new(BINGX_SECRET_KEY.encode("utf-8"), qs.encode("utf-8"), hashlib.sha256).hexdigest()
    params["signature"] = signature
    return params

def _get(path, params=None):
    headers = {"X-BX-APIKEY": BINGX_API_KEY}
    p = _sign_params(params or {})
    url = BINGX_BASE + path
    r = requests.get(url, params=p, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()

def _get_public(path: str, params: Optional[dict] = None) -> dict:
    p = dict(params or {})
    p.setdefault("timestamp", int(time.time() * 1000))
    url = BINGX_BASE + path
    r = requests.get(url, params=p, timeout=15)
    r.raise_for_status()
    return r.json()

# ============================================================
# ENDPOINTS DE MERCADO
# ============================================================

def fetch_bingx_perp_symbols(debug: bool = False) -> List[str]:
    """Obtiene lista de símbolos de contratos perpetuos"""
    try:
        payload = _get_public("/openApi/swap/v2/quote/contracts")
        rows = payload.get("data", [])
        if isinstance(rows, dict):
            rows = rows.get("contracts", [])
        
        symbols = []
        for it in rows:
            sym = (it.get("symbol") or it.get("contractSymbol") or "").upper()
            if not sym: continue
            
            if "-" not in sym:
                if sym.endswith("USDT"): sym = sym[:-4] + "-USDT"
                elif sym.endswith("USDC"): sym = sym[:-4] + "-USDC"
            symbols.append(sym)
        
        symbols = sorted(set(symbols))
        if debug:
            print(f"🔎 BingX contracts: {len(symbols)} símbolos")
        return symbols
    except Exception as e:
        print(f"❌ Error fetching symbols: {e}")
        return []

# ============================================================
# BALANCES
# ============================================================

def fetch_bingx_all_balances() -> Dict[str, Any]:
    """Obtiene balances de la cuenta"""
    try:
        payload = _get("/openApi/swap/v3/user/balance")
        rows = payload.get("data", []) if isinstance(payload, dict) else (payload or [])
        if isinstance(rows, dict):
            rows = rows.get("balances") or rows.get("assets") or []

        fut_balance = fut_equity = fut_unreal = fut_used_margin = 0.0

        for b in rows:
            asset = str(b.get("asset") or "").upper()
            if asset in ("USDT", "USDC"):
                bal = _f(b.get("balance"))
                eq = _f(b.get("equity"))
                un = _f(b.get("unrealizedProfit"))
                used = _f(b.get("usedMargin"))
                
                fut_balance += bal
                fut_equity += eq if eq != 0 else (bal + un)
                fut_unreal += un
                fut_used_margin += used

        return {
            "exchange": "bingx",
            "equity": fut_equity,
            "balance": fut_balance,
            "unrealized_pnl": fut_unreal,
            "initial_margin": fut_used_margin,
            "spot": 0.0,
            "margin": 0.0,
            "futures": fut_balance,
        }
    except Exception as e:
        print(f"❌ BingX balance error: {e}")
        return {
            "exchange": "bingx", "equity": 0.0, "balance": 0.0, 
            "unrealized_pnl": 0.0, "initial_margin": 0.0,
            "spot": 0.0, "margin": 0.0, "futures": 0.0
        }

# ============================================================
# POSICIONES ABIERTAS
# ============================================================

def fetch_bingx_open_positions() -> List[Dict[str, Any]]:
    """Obtiene posiciones abiertas y actualiza el caché"""
    try:
        data = _get("/openApi/swap/v2/user/positions")
        rows = data.get("data", []) if isinstance(data, dict) else []
        out = []
        
        for pos in rows:
            raw_symbol = (pos.get("symbol") or "").upper()
            symbol = normalize_symbol(raw_symbol)
            qty = _f(pos.get("positionAmt"))
            side = (pos.get("positionSide") or "").lower()
            if side not in ("long", "short"):
                side = "long" if qty >= 0 else "short"

            out.append({
                "exchange": "bingx",
                "symbol": symbol,
                "symbol_raw": raw_symbol,
                "position_id": pos.get("positionId"),
                "side": side,
                "size": abs(qty),
                "entry_price": _f(pos.get("avgPrice")),
                "mark_price": _f(pos.get("markPrice")),
                "unrealized_pnl": _f(pos.get("unrealizedProfit")),
                "realized_pnl": _f(pos.get("realisedProfit")),
                "funding_fee": _f(pos.get("cumFundingFee") or pos.get("fundingFee") or 0),
                "leverage": _f(pos.get("leverage")),
                "notional": _f(pos.get("positionValue")),
                "liquidation_price": _f(pos.get("liquidationPrice")),
                "isolated": bool(pos.get("isolated")) if pos.get("isolated") is not None else None,
                "update_time": _i(pos.get("updateTime")),
                "update_time_hr": _ms_to_str(pos.get("updateTime")),
            })

        # Actualizar caché con símbolos activos
        if out:
            open_symbols = [pos['symbol_raw'] for pos in out]
            _update_symbol_cache(open_symbols)

        return out
    except Exception as e:
        print(f"❌ BingX positions error: {e}")
        return []

# ============================================================
# POSICIONES CERRADAS
# ============================================================

def _collect_active_symbols_for_closed(debug: bool = False) -> list[str]:
    """Obtiene símbolos del caché para consultar posiciones cerradas"""
    cached_symbols = _get_cached_symbols()
    
    if debug:
        print(f"📋 Usando {len(cached_symbols)} símbolos del caché")
        if cached_symbols:
            print(f"   Ejemplos: {cached_symbols[:10]}{'...' if len(cached_symbols) > 10 else ''}")

    return cached_symbols

def fetch_closed_positions_bingx(
    symbols=None,
    days: int = 30,
    include_funding: bool = True,
    debug: bool = False,
) -> list[dict]:
    """Obtiene posiciones cerradas usando símbolos del caché"""
    
    # Usar caché si no se especifican símbolos
    if not symbols:
        symbols = _collect_active_symbols_for_closed(debug=debug)
        if not symbols:
            if debug:
                print("⚠️ No hay símbolos en caché para consultar")
            return []

    symbols = list(symbols)
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000
    results = []

    if debug:
        print(f"🔍 Consultando {len(symbols)} símbolos ({_ms_to_str(start_ms)} → {_ms_to_str(now_ms)})")

    for sym in symbols:
        sym_dash = _bx_to_dash(sym)
        if debug:
            print(f"   • {sym_dash}")

        page = 1
        while True:
            params = {
                "symbol": sym_dash,
                "startTs": start_ms,
                "endTs": now_ms,
                "pageId": page,
                "pageSize": 200,
                "recvWindow": 5000,
            }
            
            try:
                payload = _get("/openApi/swap/v1/trade/positionHistory", params)
                
                # Extraer datos
                data_list = []
                if isinstance(payload, dict):
                    if isinstance(payload.get("data"), dict):
                        data_list = payload["data"].get("positionHistory", [])
                    elif isinstance(payload.get("data"), list):
                        data_list = payload["data"]
                
                if not data_list:
                    break

                # Procesar cada posición cerrada
                for row in data_list:
                    try:
                        open_ms = _i(row.get("openTime"))
                        close_ms = _i(row.get("updateTime"))
                        qty_close = _f(row.get("closePositionAmt") or row.get("positionAmt"))
                        entry_price = _f(row.get("avgPrice"))
                        close_price = _f(row.get("avgClosePrice"))
                        price_pnl = _f(row.get("realisedProfit"))
                        funding_total = _f(row.get("totalFunding"))
                        fee_total = _f(row.get("positionCommission"))
                        
                        raw_symbol = _bx_no_dash(row.get("symbol") or sym_dash)
                        
                        results.append({
                            "exchange": "bingx",
                            "symbol": normalize_symbol(raw_symbol),
                            "symbol_raw": row.get("symbol") or sym_dash,
                            "position_id": row.get("positionId"),
                            "side": (row.get("positionSide") or "").lower() or "closed",
                            "size": abs(qty_close),
                            "entry_price": entry_price,
                            "close_price": close_price,
                            "open_time": int(open_ms/1000) if open_ms else None,
                            "close_time": int(close_ms/1000) if close_ms else None,
                            "open_time_hr": _ms_to_str(open_ms),
                            "close_time_hr": _ms_to_str(close_ms),
                            "realized_pnl": price_pnl + funding_total + fee_total,
                            "pnl": price_pnl,
                            "funding_total": funding_total,
                            "fee_total": fee_total,
                            "fee": fee_total,
                            "notional": entry_price * abs(qty_close),
                            "leverage": _f(row.get("leverage")),
                        })
                    except Exception as e:
                        if debug:
                            print(f"      [WARN] Error procesando fila: {e}")
                        continue

                # Paginación
                if len(data_list) < 200:
                    break
                page += 1
                
            except Exception as e:
                if debug:
                    print(f"      [ERROR] {e}")
                break

        time.sleep(0.2)  # Throttle

    if debug:
        print(f"✅ Posiciones cerradas encontradas: {len(results)}")
    
    return results

# ============================================================
# FUNDING
# ============================================================

def fetch_funding_bingx(limit=100, start_time=None, end_time=None, symbol: str | None = None):
    """Obtiene historial de funding"""
    try:
        params = {"incomeType": "FUNDING_FEE", "limit": min(int(limit), 1000)}
        if start_time: params["startTime"] = int(start_time)
        if end_time: params["endTime"] = int(end_time)
        if symbol:
            sym = symbol.upper()
            if "-" not in sym:
                if sym.endswith("USDT"): sym = sym[:-4] + "-USDT"
                elif sym.endswith("USDC"): sym = sym[:-4] + "-USDC"
            params["symbol"] = sym

        data = _get("/openApi/swap/v2/user/income", params=params)
        recs = data.get("data", []) if isinstance(data, dict) else []
        
        return [{
            "exchange": "bingx",
            "symbol": (r.get("symbol") or "").replace("-", "").upper(),
            "income": _f(r.get("income")),
            "asset": r.get("asset", "USDT"),
            "timestamp": _i(r.get("time")),
            "type": r.get("incomeType", "FUNDING_FEE"),
        } for r in recs]
    except Exception as e:
        print(f"❌ BingX funding error: {e}")
        return []

# ============================================================
# GUARDADO EN BASE DE DATOS
# ============================================================

# db helper
try:
    from db_manager import save_closed_position
except Exception:
    def save_closed_position(_: Dict[str, Any]):
        raise RuntimeError("db_manager.save_closed_position no disponible")

def save_bingx_closed_positions(
    db_path="portfolio.db",
    symbols=None,
    days=30,
    include_funding=True,
    debug=False,
) -> None:
    """Guarda posiciones cerradas en la base de datos"""
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return

    positions = fetch_closed_positions_bingx(
        symbols=symbols,
        days=days,
        include_funding=include_funding,
        debug=debug,
    )
    
    if not positions:
        print("⚠️ No se obtuvieron posiciones cerradas de BingX.")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    saved = skipped = 0
    
    for pos in positions:
        try:
            # Verificar si ya existe
            cur.execute("""SELECT COUNT(*) FROM closed_positions
                           WHERE exchange=? AND symbol=? AND close_time=?""",
                        (pos["exchange"], pos["symbol"], pos["close_time"]))
            if cur.fetchone()[0]:
                skipped += 1
                continue
            
            save_closed_position(pos)
            saved += 1
        except Exception as e:
            print(f"⚠️ Error guardando {pos.get('symbol')}: {e}")
    
    conn.close()
    print(f"✅ BingX: {saved} guardadas, {skipped} omitidas (duplicadas)")

# ============================================================
# DEBUG Y EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    print("=== BINGX ADAPTER DEBUG ===")
    
    # 1. Estado del caché
    debug_cache_status()
    force_cache_update()
    debug_cache_status()
    
    # 2. Posiciones abiertas
    print("\n=== POSICIONES ABIERTAS ===")
    opens = fetch_bingx_open_positions()
    print(f"Encontradas: {len(opens)}")
    for pos in opens:
        print(f"  {pos['symbol']} {pos['side']} {pos['size']} @ {pos['entry_price']}")
    
    # 3. Posiciones cerradas
    print("\n=== POSICIONES CERRADAS ===")
    closed = fetch_closed_positions_bingx(debug=True)
    print(f"Encontradas: {len(closed)}")
    
    # 4. Guardar en BD
    print("\n=== GUARDADO EN BD ===")
    save_bingx_closed_positions(debug=True)
    
    print("\n=== DEBUG COMPLETADO ===")
