# adapters/paradex.py
# Paradex adapter — v1.0
# Requisitos:
# - __all__ con 4 funciones públicas (opens, funding, balances, closed->DB)
# - Normalización de símbolo EXACTA (rule A)
# - Shapes EXACTOS para abiertas (B), funding (C) y balances (D)
# - Persistencia de cerradas usando db_manager.save_closed_position(...) (E)
# - Respeta toggles/prints del proyecto (usa helpers si existen)
# - CLI de debug: --dry-run (default), --save-closed, --funding N, --opens

from __future__ import annotations
import os, re, time, json, math, argparse, random
from typing import Any, Dict, List, Optional, Iterable
import requests
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timedelta
from collections import defaultdict, deque
import sqlite3
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass
# Importar db_manager
try:
    from db_manager import save_closed_position, init_db
except ImportError as e:
    print(f"❌ Error importando db_manager: {e}")
    raise

# =======================
# =======================
# Configuración del adapter
# =======================
BASE_URL = "https://api.prod.paradex.trade/v1"
EXCHANGE = "paradex"
# JWT debe refrescarse externamente (expira ~5m)
PARADEX_JWT = "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0eXAiOiJhdCtKV1QiLCJhdXRob3JpemVkX2FjY291bnRzIjpbIjB4MTAxM2FjY2E2Y2NjMzA3NGQ1ODZhZDBhYjZmODdiYzJkYTEzZWMwYWQ2ODkxODg4OGI2OWU2YTE0ZmVjMDUxIiwiMHgyNDJiZTBkNWFkNjU3MmNkMWI1ZTQwZTc3MThiYTI2YzRhNWRmYmYzNmQ2YmFlZDE0YmMzZTEzMWQ4Y2VjOWMiLCIweDQyNzI1ZjBlZGFlODZhNjFlNTgwMmE2NzhjZmVmNWJmNWRjMjQ4NWViZDI0ZGMyZDBhYWU2MmE3ZThlMjBkMSIsIjB4NTBmNjMzMTdlNDRmNWZjMDRjOTgxMTFmMWE1Y2RhMTkwZjExNjFmMmVmOTJkZTNkNzA3YmUxZWE2OTVhNGMyIiwiMHhlMGQwMGRiNDExYjJmNTViNjFjZTEyNDUwYTVhYWMxOGQ5Zjc3NDgyMDFhNDVjODM4YTQ5NTg0MTFiZTkyYyJdLCJ0b2tlbl91c2FnZSI6InJlYWRvbmx5IiwicHVia2V5IjoiMHgyN2E4YTExNjBmYjQwZGJlZGNmODNhNTdlZTAwOWZkM2JmZjMxNWQwM2E1MmQ4YzkyMTk3NmZiYTNlOTJkYiIsImlzcyI6IlBhcmFkZXggcHJvZCIsInN1YiI6IjB4NWM0MTVmYWJkMjI5YTFlNTA4MWZlYzg5OWQ1NDgyYzlkOWM2Mzk0NWFmODVlZWZjZWJmMDY2ZTU0MTlkMTgwIiwiZXhwIjoxNzkyMjExNTIzLCJuYmYiOjE3NjA2NzU1MjMsImlhdCI6MTc2MDY3NTUyMywianRpIjoiZDc5NmQxMTktYjU0Zi00MWI0LWI2Y2UtMjRkOTMyN2I1MjVmIn0.AuUNR-Oss0Hv4ZGAsPoflDUNRYM3is2EblCvHxcbm4hbrKsems7VD30lkcBkZ9SCpJAaUBoPzaZRkubhZcoVdQ"




# =======================
# Helpers de consola / toggles (fallback si no existen)
# =======================
def _import_console_helpers():
    mods = ("portfoliov6.7", "portfoliov6.8", "portfoliov6.9")
    for m in mods:
        try:
            mod = __import__(m, fromlist=["*"])
            return mod
        except Exception:
            continue
    return None

_console = _import_console_helpers()

if _console:
    PRINT_CLOSED_SYNC   = getattr(_console, "PRINT_CLOSED_SYNC", True)
    PRINT_CLOSED_DEBUG  = getattr(_console, "PRINT_CLOSED_DEBUG", False)
    PRINT_OPEN_POSITIONS= getattr(_console, "PRINT_OPEN_POSITIONS", False)
    PRINT_FUNDING       = getattr(_console, "PRINT_FUNDING", False)
    PRINT_BALANCES      = getattr(_console, "PRINT_BALANCES", False)

    p_closed_sync_start = getattr(_console, "p_closed_sync_start", lambda x: print(f"⏳ Sync {x} closed..."))
    p_closed_sync_saved = getattr(_console, "p_closed_sync_saved", lambda x,s,k: print(f"✅ {x} saved={s} skipped={k}"))
    p_closed_sync_done  = getattr(_console, "p_closed_sync_done",  lambda x: print(f"✅ Done {x} closed."))
    p_closed_sync_none  = getattr(_console, "p_closed_sync_none",  lambda x: print(f"⚠️ No closed {x}"))

    p_closed_debug_header = getattr(_console, "p_closed_debug_header", lambda s: print(f"🔎 {s}"))
    p_closed_debug_count  = getattr(_console, "p_closed_debug_count",  lambda n: print(f"📦 count={n}"))
    p_closed_debug_norm_size = getattr(_console, "p_closed_debug_norm_size", lambda side,size: print(f"   🎯 {side} size={size}"))
    p_closed_debug_prices = getattr(_console, "p_closed_debug_prices", lambda e,c: print(f"   💰 entry={e} close={c}"))
    p_closed_debug_pnl    = getattr(_console, "p_closed_debug_pnl",    lambda p,fee,f: print(f"   📊 price_pnl={p} fee={fee} funding={f}"))
    p_closed_debug_times  = getattr(_console, "p_closed_debug_times",  lambda oraw,craw,o,c: print(f"   ⏰ open_raw={oraw} close_raw={craw} | open_s={o} close_s={c}"))
    p_closed_debug_normalized = getattr(_console, "p_closed_debug_normalized", lambda sym,p: print(f"   ✅ {sym} realized={p}"))

    p_open_summary = getattr(_console, "p_open_summary", lambda ex,c: print(f"📈 {ex}: {c} open"))
    p_open_block   = getattr(_console, "p_open_block",   lambda *a, **k: None)

    p_funding_fetching = getattr(_console, "p_funding_fetching", lambda x: print(f"🔍 funding {x}..."))
    p_funding_count    = getattr(_console, "p_funding_count",    lambda x,n: print(f"📦 funding count={n} ({x})"))

    p_balance_equity   = getattr(_console, "p_balance_equity",   lambda x,e: print(f"💼 {x} equity={e:.2f}"))
else:
    PRINT_CLOSED_SYNC   = True
    PRINT_CLOSED_DEBUG  = False
    PRINT_OPEN_POSITIONS= False
    PRINT_FUNDING       = False
    PRINT_BALANCES      = True

    def p_closed_sync_start(x): print(f"⏳ Sync {x} closed...")
    def p_closed_sync_saved(x,s,k): print(f"✅ {x} saved={s} skipped={k}")
    def p_closed_sync_done(x): print(f"✅ Done {x} closed.")
    def p_closed_sync_none(x): print(f"⚠️ No closed {x}")
    def p_closed_debug_header(s): print(f"🔎 {s}")
    def p_closed_debug_count(n): print(f"📦 count={n}")
    def p_closed_debug_norm_size(side,size): print(f"   🎯 {side} size={size}")
    def p_closed_debug_prices(e,c): print(f"   💰 entry={e} close={c}")
    def p_closed_debug_pnl(p,fee,f): print(f"   📊 price_pnl={p} fee={fee} funding={f}")
    def p_closed_debug_times(oraw,craw,o,c): print(f"   ⏰ open_raw={oraw} close_raw={craw} | open_s={o} close_s={c}")
    def p_closed_debug_normalized(sym,p): print(f"   ✅ {sym} realized={p}")
    def p_open_summary(ex,c): print(f"📈 {ex}: {c} open")
    def p_open_block(*a, **k): pass
    def p_funding_fetching(x): print(f"🔍 funding {x}...")
    def p_funding_count(x,n): print(f"📦 funding count={n} ({x})")
    def p_balance_equity(x,e): print(f"💼 {x} equity={e:.2f}")

# =======================
# DB helper (obligatorio)
# =======================
try:
    from db_manager import save_closed_position
except Exception as e:
    raise RuntimeError("No se pudo importar db_manager.save_closed_position") from e

# =======================
# Utilidades de red / transformación
# =======================
def _headers() -> Dict[str, str]:
    tok = os.getenv("PARADEX_JWT", PARADEX_JWT)
    if not tok:
        raise RuntimeError("Falta PARADEX_JWT (JWT válido de Paradex).")
    return {"Accept": "application/json", "Authorization": f"Bearer {tok}"}

def _get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 20, retries: int = 3) -> Any:
    url = f"{BASE_URL}{path}"
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=_headers(), timeout=timeout)
            if r.status_code >= 400:
                last = RuntimeError(f"HTTP {r.status_code} {r.text}")
            else:
                try:
                    return r.json()
                except Exception:
                    return r.text
        except Exception as e:
            last = e
        # backoff simple
        time.sleep(0.25 * (i + 1) + random.random() * 0.1)
    if last:
        raise last
    return None

def _num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _ms_or_s_to_ms(ts: Optional[int|float|str]) -> Optional[int]:
    if ts is None:
        return None
    try:
        v = int(float(ts))
        return v * 1000 if v < 10_000_000_000 else v
    except Exception:
        return None

def _to_s(ts: Optional[int|float|str]) -> Optional[int]:
    ms = _ms_or_s_to_ms(ts)
    return None if ms is None else int(ms // 1000)

# ================
# A) Normalización
# ================
def normalize_symbol(sym: str) -> str:
    """Normaliza símbolos de Paradex (ej: 'BTC-USD-PERP' -> 'BTCUSD')"""
    if not sym: 
        return ""
    
    # Lista de sufijos a eliminar (en orden de prioridad)
    suffixes = [
        "-USD-PERP", "_USD_PERP", 
        "-PERP", "_PERP", 
        "-USD", "_USD",
        "-USDC", "_USDC"  # Por si acaso
    ]
    
    # Eliminar sufijos
    normalized = sym
    for suffix in suffixes:
        normalized = normalized.replace(suffix, "")
    
    # Eliminar cualquier guion o subrayado restante
    normalized = normalized.replace("-", "").replace("_", "")
    
    # # Debug opcional
    # if sym != normalized:
    #     print(f"🔤 [DEBUG] Normalizado: '{sym}' -> '{normalized}'")
    
    return normalized.upper()
def _side_to_str(s: str) -> str:
    return "short" if (s or "").upper() == "SHORT" else "long"



# ==============================
# B) Open positions (shape EXACTO)
# ==============================
def fetch_paradex_open_positions() -> List[Dict[str, Any]]:
    """
    GET /positions
    Mapea status=OPEN a:
    {
      "exchange": "paradex",
      "symbol": "<NORMALIZADO>",
      "side": "long"|"short",
      "size": float (positivo),
      "entry_price": float,
      "mark_price": float,
      "liquidation_price": float|0.0,
      "notional": float,
      "unrealized_pnl": float,   // SOLO precio
      "fee": float,              // acumulado; NEGATIVO si coste
      "funding_fee": float,      // + cobro / - pago
      "realized_pnl": float      // fee + funding_fee (abiertas)
    }
    """
    try:
        data = _get("/positions") or {}
        rows = data.get("results") if isinstance(data, dict) else data
        rows = rows or []
        opens = [r for r in rows if str(r.get("status","")).upper() == "OPEN"]
        p_open_summary(EXCHANGE, len(opens))

        out: List[Dict[str, Any]] = []
        for r in opens:
            market = r.get("market","")
            sym = normalize_symbol(market)
            side = _side_to_str(r.get("side",""))
            size = abs(_num(r.get("size"), 0.0))
            entry = _num(r.get("average_entry_price"))
            liq  = _num(r.get("liquidation_price"), 0.0)
            mark = _num(r.get("mark_price")) or entry

            # PnL total que da Paradex (precio + funding pendientes)
            unrealized_total = _num(r.get("unrealized_pnl"), 0.0)
            funding_unsettled = _num(r.get("unrealized_funding_pnl"), 0.0)
            price_unreal = unrealized_total - funding_unsettled

            notional = abs(size * (mark if mark > 0 else entry))
            fee_accum = 0.0  # Paradex no expone fees en /positions

            # ======= NUEVO: sumar funding "realizado en el ciclo" si está PARCIAL =======
            extra_funding_realized = 0.0
            try:
                cyc = _current_cycle_info(market, days_back=14)
                if cyc and cyc.get("partial") and cyc.get("start_ms"):
                    extra_funding_realized = _sum_funding_payments_since(
                        market=market,
                        start_ms=int(cyc["start_ms"]),
                        page_size=500,
                        max_pages=10,
                    )
            except Exception as e:
                # fallar suave: si algo va mal, no sumamos extra
                print(f"⚠️ Paradex funding extra (partial) fallo en {market}: {e}")

            funding_for_position = funding_unsettled + extra_funding_realized
            realized_for_row = fee_accum + funding_for_position

            out.append({
                "exchange": EXCHANGE,
                "symbol": sym,
                "side": side,
                "size": size,
                "entry_price": entry,
                "mark_price": mark,
                "liquidation_price": liq,
                "notional": notional,
                "unrealized_pnl": price_unreal,
                "fee": fee_accum,
                "funding_fee": funding_for_position,   # ← ahora incluye 'unsettled' + 'realized del ciclo parcial'
                "realized_pnl": realized_for_row
            })

            if PRINT_OPEN_POSITIONS:
                p_open_block(EXCHANGE, sym, size, entry, mark, price_unreal,
                             realized_funding=funding_for_position, total_unsettled=funding_unsettled,
                             notional=notional, extra_verification=True)
        return out
    except Exception as e:
        print(f"❌ Error fetching Paradex open positions: {e}")
        return []
#=============
# Calculo funding por dia en funding fees
#================    

    
# === Helpers para funding sintético (proyección 8h desde el día siguiente) ===

def _start_of_day_utc_ms(ts_ms: int) -> int:
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    day0 = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
    return int(day0.timestamp() * 1000)

def _next_day_00utc_ms(ts_ms: int) -> int:
    return _start_of_day_utc_ms(ts_ms) + 24 * 3600 * 1000

def _slots_8h_from_next_day(last_paid_ts_ms: int, now_ms: int) -> list[int]:
    """Devuelve timestamps (ms, UTC) en 00:00, 08:00, 16:00 desde el día siguiente a last_paid hasta now."""
    slots = []
    day_ms = _next_day_00utc_ms(last_paid_ts_ms)  # día siguiente 00:00 UTC
    eight_h = 8 * 3600 * 1000
    while day_ms <= now_ms:
        for k in range(3):  # 00:00, 08:00, 16:00
            t = day_ms + k * eight_h
            if t <= now_ms:
                slots.append(t)
        day_ms += 24 * 3600 * 1000
    return slots

def _last_real_funding_ts_by_market(results_rows: list[dict]) -> dict[str,int]:
    """Busca el último timestamp real por market en las filas reales ya obtenidas (por símbolo normalizado)."""
    last = {}
    for r in results_rows:
        if r.get("type") != "FUNDING_FEE":  # solo reales
            continue
        sym = r.get("symbol")
        ts = int(r.get("timestamp") or 0)
        if not sym or not ts:
            continue
        if ts > last.get(sym, 0):
            last[sym] = ts
    return last

def _open_unsettled_by_market() -> dict[str, float]:
    """
    Lee /positions y devuelve, por market (normalizado), el funding no asentado (unrealized_funding_pnl)
    solo si la posición está abierta (status OPEN) y size != 0. Si no hay campo, retorna 0.0.
    """
    out = {}
    data = _get("/positions") or {}
    rows = data.get("results") if isinstance(data, dict) else data
    for r in rows or []:
        if str(r.get("status","")).upper() != "OPEN":
            continue
        mkt = r.get("market","")
        sym = normalize_symbol(mkt)
        if not sym:
            continue
        size = _num(r.get("size"), 0.0)
        if abs(size) <= 1e-12:
            continue
        unsettled = _num(r.get("unrealized_funding_pnl"), 0.0)
        # Si no hay nada pendiente, no proyectamos
        if abs(unsettled) <= 1e-9:
            continue
        out[sym] = unsettled
    return out

def _generate_estimated_rows(sym: str, unsettled: float, last_paid_ts_ms: int) -> list[dict]:
    """Divide 'unsettled' a partes iguales entre los slots 8h desde el día siguiente a last_paid."""
    now_ms = int(time.time() * 1000)
    slots = _slots_8h_from_next_day(last_paid_ts_ms, now_ms)
    if not slots:
        return []
    per = unsettled / len(slots)
    asset = "USD"
    rows = []
    for t in slots:
        rows.append({
            "exchange": EXCHANGE,
            "symbol": sym,
            "income": per,
            "asset": asset,
            "timestamp": int(t),
            "funding_rate": 0.0,
            "type": "FUNDING_ESTIMATE",   # <- para distinguir en el front
            "estimated": True
        })
    return rows

# =========================================
# C) Funding history (shape EXACTO requerido)
# =========================================
def fetch_paradex_funding_fees(limit: int = 50,
                               markets: Optional[Iterable[str]] = None,
                               start_at_ms: Optional[int] = None,
                               end_at_ms: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    GET /funding/payments?market=...&cursor=...
    Retorna lista normalizada:
    { exchange, symbol, income, asset, timestamp, funding_rate(0.0), type:"FUNDING_FEE" }
    """
    p_funding_fetching(EXCHANGE)

    try:
        # Asset por defecto
        asset = "USD"
        
        # Si no especificas mercados, obtener todos los disponibles
        if markets is None:
            try:
                pos_data = _get("/positions") or {}
                pos_rows = pos_data.get("results") if isinstance(pos_data, dict) else pos_data
                markets = sorted({r.get("market","") for r in (pos_rows or []) if r.get("market")})
            except Exception:
                markets = []

        results: List[Dict[str, Any]] = []
        
        for market in markets or []:
            cursor = None
            pulled = 0
            
            while pulled < limit:
                params: Dict[str, Any] = {"market": market, "limit": min(100, limit - pulled)}
                if start_at_ms: 
                    params["start_at"] = int(start_at_ms)
                if end_at_ms:   
                    params["end_at"] = int(end_at_ms)
                if cursor:      
                    params["cursor"] = cursor

                page = _get("/funding/payments", params=params) or {}
                items = page.get("results", [])
                
                if not items:
                    break
                    
                for it in items:
                    ts_ms = _ms_or_s_to_ms(it.get("created_at")) or 0
                    results.append({
                        "exchange": EXCHANGE,
                        "symbol": normalize_symbol(market),
                        "income": _num(it.get("payment"), 0.0),
                        "asset": asset,
                        "timestamp": ts_ms,
                        "funding_rate": 0.0,
                        "type": "FUNDING_FEE",
                    })
                    pulled += 1
                    if pulled >= limit:
                        break
                        
                if pulled >= limit:
                    break
                    
                cursor = page.get("next") or None
                if not cursor:
                    break

        if PRINT_FUNDING:
            p_funding_count(EXCHANGE, len(results))
            # === Inyectar proyección 8h desde el día siguiente al último funding real ===
        try:
            # 1) último funding real por símbolo ya obtenido
            last_real_by_sym = _last_real_funding_ts_by_market(results)
       
            # 2) funding no asentado por símbolo en posiciones abiertas
            unsettled_by_sym = _open_unsettled_by_market()
       
            # 3) para cada símbolo con pendiente, generar slots desde el día siguiente
            for sym, unsettled in unsettled_by_sym.items():
                # si no hay registro real previo, intenta pedir el último 1-row rápido
                last_ts = last_real_by_sym.get(sym)
                if not last_ts:
                    # reconstruir market string desde sym no siempre es 1:1; preferimos pedir /funding/payments con cualquier market que matchee
                    # intento: tomar el primer market de /positions que produce este sym
                    try:
                        pos_data = _get("/positions") or {}
                        pos_rows = pos_data.get("results") if isinstance(pos_data, dict) else pos_data
                        market_guess = None
                        for pr in pos_rows or []:
                            if normalize_symbol(pr.get("market","")) == sym:
                                market_guess = pr.get("market")
                                break
                        if market_guess:
                            page = _get("/funding/payments", params={"market": market_guess, "limit": 1}) or {}
                            it = (page.get("results") or [None])[0]
                            if it:
                                last_ts = _ms_or_s_to_ms(it.get("created_at")) or None
                    except Exception:
                        pass
       
                if not last_ts:
                    # Sin referencia real: por seguridad, no proyectamos
                    continue
       
                est_rows = _generate_estimated_rows(sym, unsettled, int(last_ts))
                results.extend(est_rows)
        except Exception as e:
            print(f"⚠️ Proyección funding (estimado) falló: {e}")
        return results
        
    except Exception as e:
        print(f"❌ Error fetching Paradex funding fees: {e}")
        return []

# ======================================
# D) Balances — dict único (shape EXACTO)
# ======================================
def fetch_paradex_all_balances(db_path: str = "portfolio.db") -> Dict[str, Any]:
    """
    GET /account → mapea a:
    {
      "exchange": "paradex",
      "equity": float,
      "balance": float,
      "unrealized_pnl": float,
      "initial_margin": float,
      "spot": 0.0,
      "margin": 0.0,
      "futures": float
    }
    """
    try:
        acc = _get("/account") or {}
        
        # Valores principales
        equity = _num(acc.get("account_value", 0.0))
        balance = _num(acc.get("free_collateral", 0.0))
        initial_margin = _num(acc.get("initial_margin_requirement", 0.0))
        
        # Para Paradex, el unrealized PnL no está disponible directamente en /account
        # Podemos calcularlo sumando el PnL no realizado de las posiciones abiertas
        unrealized_pnl = 0.0
        try:
            open_positions = fetch_paradex_open_positions()
            unrealized_pnl = sum(pos.get("unrealized_pnl", 0.0) for pos in open_positions)
        except:
            pass

        obj = {
            "exchange": EXCHANGE,
            "equity": equity,
            "balance": balance,
            "unrealized_pnl": unrealized_pnl,
            "initial_margin": initial_margin,
            "spot": 0.0,      # Paradex es solo derivados
            "margin": 0.0,    # No hay margin trading separado
            "futures": equity, # Todo el equity está en futuros
        }
        
        if PRINT_BALANCES:
            p_balance_equity(EXCHANGE, equity)
        return obj
        
    except Exception as e:
        print(f"❌ Error fetching Paradex balances: {e}")
        return {
            "exchange": EXCHANGE,
            "equity": 0.0,
            "balance": 0.0,
            "unrealized_pnl": 0.0,
            "initial_margin": 0.0,
            "spot": 0.0,
            "margin": 0.0,
            "futures": 0.0,
        }
#=======================
# E) Closed positions
#=======================
    
def _get_all_fills(days_back: int = 60) -> List[Dict]:
    """Obtiene TODOS los fills de los últimos N días con paginación completa"""
    # Convertir days_back a integer si viene como string
    try:
        days_back = int(days_back)
    except (ValueError, TypeError):
        days_back = 60  # Valor por defecto    
    
    all_fills = []
    cursor = None
    url = f"{BASE_URL}/fills"
    
    # Calcular timestamp de inicio (60 días atrás en milisegundos)
    start_time = int((time.time() - (days_back * 24 * 60 * 60)) * 1000)
    
    params = {
        "start_at": start_time,
        "page_size": 1000  # Máximo permitido por la API
    }
    
    page_count = 0
    total_fills = 0
    
    print(f"📅 [DEBUG] Buscando fills desde: {datetime.fromtimestamp(start_time/1000)}")
    
    while True:
        try:
            current_params = params.copy()
            if cursor:
                current_params["cursor"] = cursor
            
            print(f"🔍 [DEBUG] Solicitando página {page_count + 1}...")
            
            response = requests.get(
                url, 
                params=current_params, 
                headers=_headers(), 
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                fills = data.get("results", [])
                
                if fills:
                    all_fills.extend(fills)
                    total_fills += len(fills)
                    print(f"✅ [DEBUG] Página {page_count + 1}: {len(fills)} fills")
                    
                    # Verificar paginación
                    next_cursor = data.get("next")
                    if next_cursor:
                        cursor = next_cursor
                        page_count += 1
                        time.sleep(0.2)  # Rate limiting
                    else:
                        print(f"🎯 [DEBUG] No más páginas. Total: {total_fills} fills")
                        break
                else:
                    print("ℹ️  [DEBUG] No hay más fills")
                    break
            else:
                print(f"❌ [DEBUG] Error HTTP {response.status_code}: {response.text}")
                break
                
        except Exception as e:
            print(f"❌ [DEBUG] Error en página {page_count + 1}: {e}")
            break
    
    print(f"📊 [DEBUG] Total de fills obtenidos: {len(all_fills)}")
    return all_fills

# =======================
# Helpers para ciclo abierto y funding realizado (Paradex)
# =======================
def _get_fills_market(market: str, start_at_ms: int | None = None,
                      end_at_ms: int | None = None, page_size: int = 1000,
                      max_pages: int = 8) -> list[dict]:
    """
    Trae fills de un mercado concreto con paginación.
    Si el API no soportara ?market, filtramos client-side igualmente.
    """
    acc, cursor = [], None
    for _ in range(max_pages):
        params = {"page_size": page_size}
        if start_at_ms is not None: params["start_at"] = int(start_at_ms)
        if end_at_ms   is not None: params["end_at"]   = int(end_at_ms)
        if cursor: params["cursor"] = cursor
        # Intento con filtro market:
        params_with_market = dict(params, market=market)
        page = _get("/fills", params=params_with_market) or {}
        items = page.get("results", [])
        # Fallback: filtra client-side por si el server ignora 'market'
        if market:
            items = [it for it in items if (it.get("market") == market)]
        if not items:
            break
        acc.extend(items)
        cursor = page.get("next")
        if not cursor:
            break
    return acc

def _current_cycle_info(market: str, days_back: int = 30) -> dict | None:
    """
    Determina el 'ciclo abierto' actual para un mercado:
    - start_ms: timestamp (ms) del primer fill del ciclo actual (último tramo donde net != 0)
    - partial: True si en el ciclo hay BUY y SELL y el neto final != 0 (parcialmente cerrada)
    - net, buy_sum, sell_sum: info auxiliar
    """
    start_cut = int(time.time() * 1000) - int(days_back) * 24 * 3600 * 1000
    fills = _get_fills_market(market, start_at_ms=start_cut, page_size=1000, max_pages=10)
    if not fills:
        return None
    fills.sort(key=lambda x: int(x.get("created_at", 0)))

    net = 0.0
    last_zero_idx = -1
    for i, f in enumerate(fills):
        sz = float(f.get("size", 0))
        signed = sz if (f.get("side") == "BUY") else -sz
        net += signed
        if abs(net) < 1e-9:
            last_zero_idx = i

    start_idx = last_zero_idx + 1
    if start_idx >= len(fills):
        # No hay tramo abierto visible en ventana
        return None

    seg = fills[start_idx:]
    buy_sum = sum(float(x.get("size", 0)) for x in seg if x.get("side") == "BUY")
    sell_sum = sum(float(x.get("size", 0)) for x in seg if x.get("side") == "SELL")
    partial = (buy_sum > 0 and sell_sum > 0 and abs(buy_sum - sell_sum) > 1e-9)
    start_ms = int(seg[0].get("created_at", 0)) if seg else None
    return {"start_ms": start_ms, "partial": partial, "net": buy_sum - sell_sum,
            "buy_sum": buy_sum, "sell_sum": sell_sum}

def _sum_funding_payments_since(market: str, start_ms: int | None,
                                page_size: int = 500, max_pages: int = 10) -> float:
    """
    Suma funding payments (raw) desde start_ms (incluido) para un mercado.
    """
    total = 0.0
    cursor = None
    for _ in range(max_pages):
        params = {"market": market, "page_size": page_size}
        if start_ms is not None:
            params["start_at"] = int(start_ms)
        if cursor:
            params["cursor"] = cursor
        page = _get("/funding/payments", params=params) or {}
        items = page.get("results", [])
        if not items:
            break
        for it in items:
            ts = _ms_or_s_to_ms(it.get("created_at")) or 0
            if start_ms is not None and ts < start_ms:
                continue
            total += _num(it.get("payment"), 0.0)
        cursor = page.get("next")
        if not cursor:
            break
    return total

def _parse_timestamp(ts: Optional[int]) -> int:
    """Convierte timestamp de Paradex (ms) a segundos para la DB"""
    if not ts:
        return int(time.time())
    return ts // 1000

def _fifo_realized_pnl(block_trades):
    """
    block_trades: lista de dicts con campos:
      - 'side' (BUY o SELL)
      - 'size' (qty absoluta, >0)
      - 'price'
    Devuelve PnL por precio (sin fees/funding) usando FIFO real:
     - Si hay lot largo abierto (qty>0) y llega un SELL, cierra contra ese lot.
     - Si hay lot corto abierto (qty<0) y llega un BUY, cierra contra ese lot.
    """
    lots = deque()  # cada lot: {'qty': signed_qty, 'price': entry_price}
    realized = 0.0
    eps = 1e-12

    for t in block_trades:
        side = t["side"]
        qty = float(t["size"])
        p = float(t["price"])
        if side == "BUY":
            q_signed = qty
        else:
            q_signed = -qty

        if abs(q_signed) < eps:
            continue

        if q_signed > 0:  # BUY: primero cierra shorts existentes
            remaining = q_signed
            while remaining > eps and lots and lots[0]["qty"] < 0:
                open_lot = lots[0]
                match_qty = min(remaining, -open_lot["qty"])
                # short: PnL = (entry - exit) * qty
                realized += (open_lot["price"] - p) * match_qty
                open_lot["qty"] += match_qty  # menos negativo
                remaining -= match_qty
                if abs(open_lot["qty"]) < eps:
                    lots.popleft()
            # lo que sobre abre long
            if remaining > eps:
                lots.append({"qty": remaining, "price": p})

        else:  # SELL (q_signed < 0): primero cierra longs existentes
            remaining = -q_signed
            while remaining > eps and lots and lots[0]["qty"] > 0:
                open_lot = lots[0]
                match_qty = min(remaining, open_lot["qty"])
                # long: PnL = (exit - entry) * qty
                realized += (p - open_lot["price"]) * match_qty
                open_lot["qty"] -= match_qty
                remaining -= match_qty
                if open_lot["qty"] < eps:
                    lots.popleft()
            # lo que sobre abre short
            if remaining > eps:
                lots.append({"qty": -remaining, "price": p})

    # Al cerrar el bloque (net=0) no deberían quedar lots
    # Si queda algún residuo muy pequeño, lo ignoramos (tolerancia numérica).
    return realized

def _identify_positions(fills: List[Dict]) -> List[List[Dict]]:
    """
    Identifica posiciones individuales usando un enfoque FIFO por mercado
    """
    positions = []
    
    # Ordenar todos los fills por timestamp
    fills.sort(key=lambda x: x.get("created_at", 0))
    
    # Agrupar por mercado
    market_groups = defaultdict(list)
    for fill in fills:
        market = fill.get("market")
        if market:
            market_groups[market].append(fill)
    
    print(f"🔍 [DEBUG] Mercados encontrados: {list(market_groups.keys())}")
    
    # Para cada mercado, usar enfoque FIFO para identificar posiciones
    for market, market_fills in market_groups.items():
        print(f"📊 [DEBUG] Procesando mercado {market}: {len(market_fills)} fills")
        
        net = 0.0
        max_net_abs = 0.0
        current_block = []
        open_time = None
        
        for fill in market_fills:
            if open_time is None and abs(float(fill.get("size", 0))) > 1e-9:
                open_time = fill.get("created_at")
            
            fill_size = float(fill.get("size", 0))
            fill_side = fill.get("side", "")
            
            # Calcular signed quantity
            signed_qty = fill_size if fill_side == "BUY" else -fill_size
            net += signed_qty
            current_block.append(fill)
            
            current_net_abs = abs(net)
            if current_net_abs > max_net_abs:
                max_net_abs = current_net_abs
            
            # Si el net es cero (o muy cercano), tenemos una posición cerrada
            if abs(net) < 1e-9 and len(current_block) > 1 and max_net_abs > 1e-9:
                close_time = fill.get("created_at")
                positions.append(current_block.copy())
                
                print(f"📍 [DEBUG] Posición identificada en {market}: {len(current_block)} fills")
                
                # Reset para siguiente posición
                current_block = []
                net = 0.0
                max_net_abs = 0.0
                open_time = None
    
    print(f"🎯 [DEBUG] Total de posiciones identificadas: {len(positions)}")
    return positions

def _calculate_position_metrics(position_fills: List[Dict]) -> Dict[str, Any]:
    """Calcula métricas detalladas para una posición específica usando FIFO"""
    if not position_fills:
        return {}
    
    # Ordenar fills por timestamp
    position_fills.sort(key=lambda x: x.get("created_at", 0))
    
    first_fill = position_fills[0]
    last_fill = position_fills[-1]
    market = first_fill.get("market", "")
    symbol_normalized = normalize_symbol(market)
    
    # Calcular PnL usando FIFO
    price_pnl = _fifo_realized_pnl(position_fills)
    
    # Calcular total de compras y ventas
    total_buy_size = sum(float(f.get("size", 0)) for f in position_fills if f.get("side") == "BUY")
    total_sell_size = sum(float(f.get("size", 0)) for f in position_fills if f.get("side") == "SELL")
    
    # Determinar side de la posición basado en el primer trade significativo
    first_trade = None
    for fill in position_fills:
        if abs(float(fill.get("size", 0))) > 1e-9:
            first_trade = fill
            break
    
    if first_trade:
        side = "long" if first_trade.get("side") == "BUY" else "short"
    else:
        side = "long"  # Por defecto
    
    # Calcular precios promedios ponderados
    if side == "long":
        entry_trades = [f for f in position_fills if f.get("side") == "BUY"]
        if entry_trades:
            entry_avg = sum(float(f.get("size", 0)) * float(f.get("price", 0)) for f in entry_trades) / total_buy_size
        else:
            entry_avg = float(first_fill.get("price", 0))
        
        exit_trades = [f for f in position_fills if f.get("side") == "SELL"]
        if exit_trades:
            close_avg = sum(float(f.get("size", 0)) * float(f.get("price", 0)) for f in exit_trades) / total_sell_size
        else:
            close_avg = float(last_fill.get("price", 0))
    else:  # short
        entry_trades = [f for f in position_fills if f.get("side") == "SELL"]
        if entry_trades:
            entry_avg = sum(float(f.get("size", 0)) * float(f.get("price", 0)) for f in entry_trades) / total_sell_size
        else:
            entry_avg = float(first_fill.get("price", 0))
        
        exit_trades = [f for f in position_fills if f.get("side") == "BUY"]
        if exit_trades:
            close_avg = sum(float(f.get("size", 0)) * float(f.get("price", 0)) for f in exit_trades) / total_buy_size
        else:
            close_avg = float(last_fill.get("price", 0))
    
    # Calcular tamaño de la posición (máximo net absoluto durante el ciclo)
    net = 0.0
    max_net_abs = 0.0
    for fill in position_fills:
        fill_size = float(fill.get("size", 0))
        fill_side = fill.get("side", "")
        signed_qty = fill_size if fill_side == "BUY" else -fill_size
        net += signed_qty
        current_net_abs = abs(net)
        if current_net_abs > max_net_abs:
            max_net_abs = current_net_abs
    
    size = max_net_abs
    
    # Calcular fees y funding
    total_fees = sum(float(f.get("fee", 0)) for f in position_fills)
    total_funding = sum(float(f.get("realized_funding", 0)) for f in position_fills)
    
    # PnL neto (precio + funding - fees)
    realized_pnl = price_pnl + total_funding - total_fees
    
    # Calcular notional
    notional = size * entry_avg
    
    return {
        "exchange": EXCHANGE,
        "symbol": symbol_normalized,
        "side": side,
        "size": size,
        "entry_price": entry_avg,
        "close_price": close_avg,
        "open_time": _parse_timestamp(first_fill.get("created_at")),
        "close_time": _parse_timestamp(last_fill.get("created_at")),
        "pnl": price_pnl,  # PnL solo por precio
        "realized_pnl": realized_pnl,  # PnL neto (precio + funding - fees)
        "funding_total": total_funding,
        "fee_total": -abs(total_fees),  # Fees siempre negativos
        "notional": notional,
        "fills_count": len(position_fills),
        "total_buy_size": total_buy_size,
        "total_sell_size": total_sell_size
    }

# =======================
# Función principal mejorada
# =======================
def reconstruct_paradex_positions(days_back: int = 60, debug: bool = True) -> None:
    """
    Reconstruye TODAS las posiciones cerradas de Paradex usando FIFO
    """
    print("🚀 [DEBUG] Iniciando reconstrucción COMPLETA de posiciones Paradex...")
    print(f"📅 [DEBUG] Período: últimos {days_back} días")
    
    try:
        # Obtener TODOS los fills históricos
        print("📥 [DEBUG] Obteniendo TODOS los fills (esto puede tomar tiempo)...")
        all_fills = _get_all_fills(days_back)
        
        if not all_fills:
            print("ℹ️  [DEBUG] No se encontraron fills para procesar")
            return
        
        # Identificar posiciones individuales usando FIFO
        print("🔍 [DEBUG] Identificando posiciones individuales...")
        positions = _identify_positions(all_fills)
        
        if not positions:
            print("ℹ️  [DEBUG] No se identificaron posiciones completas")
            return
        
        # Procesar cada posición
        saved_positions = 0
        print(f"🔄 [DEBUG] Procesando {len(positions)} posiciones identificadas...")
        
        for i, position_fills in enumerate(positions, 1):
            market = position_fills[0].get("market", "Unknown")
            print(f"\n📊 [DEBUG] Procesando posición {i}/{len(positions)}: {market}")
            
            # Calcular métricas detalladas usando FIFO
            position_data = _calculate_position_metrics(position_fills)
            
            # Solo guardar posiciones con tamaño significativo (> 0.001)
            if position_data and position_data.get("size", 0) > 0.001:
                print(f"✅ [DEBUG] Posición {i} reconstruída:")
                print(f"   Mercado: {position_data['symbol']}")
                print(f"   Side: {position_data['side']}")
                print(f"   Size: {position_data['size']:.6f}")
                print(f"   Entry: ${position_data['entry_price']:.6f}")
                print(f"   Close: ${position_data['close_price']:.6f}")
                print(f"   PnL (precio): ${position_data['pnl']:.6f}")
                print(f"   Realized PnL: ${position_data['realized_pnl']:.6f}")
                print(f"   Funding: ${position_data['funding_total']:.6f}")
                print(f"   Fees: ${position_data['fee_total']:.6f}")
                print(f"   Fills: {position_data['fills_count']}")
                print(f"   Open: {datetime.fromtimestamp(position_data['open_time'])}")
                print(f"   Close: {datetime.fromtimestamp(position_data['close_time'])}")
                print(f"   Duración: {timedelta(seconds=position_data['close_time'] - position_data['open_time'])}")
                
                # Guardar en base de datos
                try:
                    # Agregar leverage por defecto para Paradex
                    position_data["leverage"] = 3.0
                    position_data["liquidation_price"] = 0.0
                    position_data["initial_margin"] = position_data["notional"] / 3.0
                    position_data["_lock_size"] = True  # Evitar recálculo en db_manager
                    
                    save_closed_position(position_data)
                    saved_positions += 1
                    print(f"💾 [DEBUG] Posición {i} guardada en DB")
                    
                except Exception as e:
                    print(f"❌ [DEBUG] Error guardando posición {i}: {e}")
                    import traceback
                    traceback.print_exc()
                
            else:
                print(f"⚠️  [DEBUG] Posición {i} descartada (size {position_data.get('size', 0) if position_data else 0} demasiado pequeño)")
            
            print("-" * 60)
        
        print(f"\n🎉 [DEBUG] RECONSTRUCCIÓN COMPLETADA!")
        print(f"📊 Resumen:")
        print(f"   • Fills procesados: {len(all_fills)}")
        print(f"   • Posiciones identificadas: {len(positions)}")
        print(f"   • Posiciones guardadas: {saved_positions}")
        print(f"   • Período cubierto: {days_back} días")
        
    except Exception as e:
        print(f"❌ [DEBUG] Error en reconstrucción: {e}")
        import traceback
        traceback.print_exc()

# =======================
# Función adicional para verificar datos existentes
# =======================
def _position_exists_in_db(position_data: Dict) -> bool:
    """Verifica si una posición ya existe en la base de datos"""
    try:
        conn = sqlite3.connect("portfolio.db")
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM closed_positions 
            WHERE exchange = ? AND symbol = ? AND side = ? 
            AND open_time = ? AND close_time = ?
            AND ABS(size - ?) < 0.001
        """, (
            position_data['exchange'],
            position_data['symbol'], 
            position_data['side'],
            position_data['open_time'],
            position_data['close_time'],
            position_data['size']
        ))
        
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
        
    except Exception as e:
        print(f"❌ Error verificando posición en DB: {e}")
        return False

def save_paradex_closed_positions(days_back: int = 60) -> int:
    """
    Función principal del adapter para guardar posiciones cerradas de Paradex.
    Sigue el mismo patrón que otros exchanges y se integra con portfoliov6.9.
    
    Args:
        days_back: Número de días hacia atrás para buscar posiciones (puede ser string o int)
        
    Returns:
        Número de posiciones guardadas
    """
    # Convertir days_back a integer si viene como string
    try:
        days_back = int(days_back)
    except (ValueError, TypeError):
        days_back = 60  # Valor por defecto si la conversión falla
    
    p_closed_sync_start(EXCHANGE)
    
    try:
        # Obtener TODOS los fills históricos
        all_fills = _get_all_fills(days_back)
        
        if not all_fills:
            p_closed_sync_none(EXCHANGE)
            return 0
        
        # Identificar posiciones individuales usando FIFO
        positions = _identify_positions(all_fills)
        
        if not positions:
            p_closed_sync_none(EXCHANGE)
            return 0
        
        saved_count = 0
        skipped_count = 0
        
        if PRINT_CLOSED_DEBUG:
            p_closed_debug_header(f"Paradex - {len(positions)} posiciones identificadas")
            p_closed_debug_count(len(positions))
        
        # Procesar cada posición
        for i, position_fills in enumerate(positions, 1):
            market = position_fills[0].get("market", "Unknown")
            
            # Calcular métricas detalladas usando FIFO
            position_data = _calculate_position_metrics(position_fills)
            
            # Solo guardar posiciones con tamaño significativo (> 0.001)
            if position_data and position_data.get("size", 0) > 0.001:
                # VERIFICAR DUPLICADO antes de procesar (igual que en Extended)
                if _position_exists_in_db(position_data):
                    print(f"⚠️ [DEBUG] Posición ya existe en DB, omitiendo: {position_data['symbol']}")
                    skipped_count += 1
                    continue
                    
                if PRINT_CLOSED_DEBUG:
                    symbol = position_data["symbol"]
                    side = position_data["side"]
                    size = position_data["size"]
                    entry = position_data["entry_price"]
                    close = position_data["close_price"]
                    pnl = position_data["pnl"]
                    funding = position_data["funding_total"]
                    fee = position_data["fee_total"]
                    
                    p_closed_debug_norm_size(side, size)
                    p_closed_debug_prices(entry, close)
                    p_closed_debug_pnl(pnl, fee, funding)
                    p_closed_debug_times(
                        position_fills[0].get("created_at"),
                        position_fills[-1].get("created_at"),
                        position_data["open_time"],
                        position_data["close_time"]
                    )
                    p_closed_debug_normalized(symbol, position_data["realized_pnl"])
                
                # Preparar datos para la base de datos
                try:
                    # Agregar campos requeridos por db_manager
                    position_data["leverage"] = 3.0  # Leverage por defecto para Paradex
                    position_data["liquidation_price"] = 0.0
                    position_data["initial_margin"] = position_data["notional"] / 3.0
                    position_data["_lock_size"] = True  # Evitar recálculo en db_manager
                    
                    # Guardar en base de datos
                    save_closed_position(position_data)
                    saved_count += 1
                    
                except Exception as e:
                    print(f"❌ Error guardando posición {market}: {e}")
                    skipped_count += 1
            else:
                skipped_count += 1
        
        # Mostrar resumen
        p_closed_sync_saved(EXCHANGE, saved_count, skipped_count)
        
        if saved_count > 0:
            p_closed_sync_done(EXCHANGE)
        
        return saved_count
        
    except Exception as e:
        print(f"❌ Error en save_paradex_closed_positions: {e}")
        import traceback
        traceback.print_exc()
        return 0

    # =======================
# # =======================
# # DEBUG STANDALONE: ver RAW de funding desde Paradex
# # (Pégalo al final de paradexv8s.py)
# # =======================
# import os, json, time
# try:
#     import requests
# except Exception:
#     raise RuntimeError("Falta el paquete 'requests' (pip install requests)")

# # Usa BASE_URL/_headers del archivo si ya existen; si no, define defaults
# if 'BASE_URL' not in globals():
#     BASE_URL = "https://api.prod.paradex.trade/v1"

# def _headers_debug():
#     # Si ya hay _headers() definido en el archivo, úsalo
#     if '_headers' in globals() and callable(globals()['_headers']):
#         return globals()['_headers']()
#     # Fallback: construye headers con JWT desde env o constante local PARADEX_JWT
#     jwt = os.getenv("PARADEX_JWT") or globals().get("PARADEX_JWT")
#     if not jwt:
#         raise RuntimeError(
#             "Falta el token. Define PARADEX_JWT en el entorno o como constante en el archivo."
#         )
#     return {
#         "Accept": "application/json",
#         "Authorization": f"Bearer {jwt}",
#     }

# def _get_raw_paradex(path, params=None):
#     url = f"{BASE_URL}{path}"
#     h = _headers_debug()
#     r = requests.get(url, headers=h, params=params or {}, timeout=30)
#     print(f"\nGET {r.request.url}\n-> {r.status_code}")
#     try:
#         data = r.json()
#     except Exception:
#         print("⚠️ Respuesta no-JSON (primeros 2k chars):")
#         print((r.text or "")[:2000])
#         raise
#     return data

# def debug_print_funding_raw(market, page_size=100, start_at_ms=None, end_at_ms=None,
#                             pages=1, cursor=None, save_path=None, sleep_between=0.2):
#     """
#     Imprime el RAW de /v1/funding/payments con paginación.
#     - market: p.ej. 'BTC-USD-PERP' o 'WCT-USD-PERP'
#     - page_size: 1..5000
#     - start_at_ms / end_at_ms: epoch en ms (opcional)
#     - pages: nº de páginas a seguir usando 'next'
#     - cursor: si ya tienes un 'next' previo
#     - save_path: si lo pasas, guarda todo en JSONL (una línea por registro)
#     """
#     assert market, "market requerido, p.ej. BTC-USD-PERP"
#     acc = []
#     curr = cursor
#     for i in range(max(1, int(pages))):
#         params = {"market": market, "page_size": int(page_size)}
#         if start_at_ms is not None: params["start_at"] = int(start_at_ms)
#         if end_at_ms   is not None: params["end_at"]   = int(end_at_ms)
#         if curr: params["cursor"] = curr

#         raw = _get_raw_paradex("/funding/payments", params)
#         # Imprime RAW tal cual
#         print("\n🧾 RAW PAGE", i + 1)
#         print(json.dumps(raw, indent=2, ensure_ascii=False))

#         # Acumula resultados para guardar si hace falta
#         results = raw.get("results") or []
#         acc.extend(results)

#         curr = raw.get("next")
#         if not curr:
#             print("✅ No hay más páginas (next=null).")
#             break
#         time.sleep(sleep_between)

#     if save_path:
#         with open(save_path, "w", encoding="utf-8") as f:
#             for row in acc:
#                 f.write(json.dumps(row, ensure_ascii=False) + "\n")
#         print(f"💾 Guardado JSONL con {len(acc)} filas en: {save_path}")

# # =======================
# # FORMATEO DE FECHAS (legibles) + DEBUG HUMANO PARA FUNDING
# # =======================
# from datetime import datetime, timezone
# try:
#     from zoneinfo import ZoneInfo  # Python 3.9+
# except Exception:
#     ZoneInfo = None

# # Cambia aquí tu zona horaria preferida
# DEFAULT_TZ_NAME = "Europe/Zurich"

# def _get_tz(tz_name: str | None = None):
#     """Devuelve tzinfo para imprimir en hora local deseada (fallback: local del sistema o UTC)."""
#     tz_name = tz_name or DEFAULT_TZ_NAME
#     if ZoneInfo:
#         try:
#             return ZoneInfo(tz_name)
#         except Exception:
#             pass
#     # Fallback: tz local del sistema
#     try:
#         return datetime.now().astimezone().tzinfo
#     except Exception:
#         return timezone.utc

# def fmt_datetime_ms(ms: int | float | str, tz_name: str | None = None) -> str:
#     """Convierte epoch ms a 'YYYY-MM-DD HH:MM:SS TZ'."""
#     try:
#         ms = int(float(ms))
#     except Exception:
#         return ""
#     tz = _get_tz(tz_name)
#     dt = datetime.fromtimestamp(ms / 1000, tz)
#     return dt.strftime("%Y-%m-%d %H:%M:%S %Z")

# def humanize_delta_ms(ms: int | float | str, now_ms: int | None = None) -> str:
#     """Devuelve 'hace 3 h 12 m', 'hace 2 d', etc. (si es futuro: 'en ...')."""
#     try:
#         t_ms = int(float(ms))
#     except Exception:
#         return ""
#     if now_ms is None:
#         now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
#     diff = now_ms - t_ms
#     future = diff < 0
#     diff = abs(diff)

#     # Unidades
#     s = diff // 1000
#     m = s // 60
#     h = m // 60
#     d = h // 24

#     def seg(val, unit):
#         return f"{val} {unit}"

#     out = ""
#     if d >= 1:
#         out = seg(d, "d")
#         h = h % 24
#         if h: out += f" {h} h"
#     elif h >= 1:
#         out = seg(h, "h")
#         m = m % 60
#         if m: out += f" {m} m"
#     elif m >= 1:
#         out = seg(m, "m")
#     else:
#         out = seg(s, "s")

#     return f"en {out}" if future else f"hace {out}"

# def debug_print_funding_human(
#     market: str,
#     page_size: int = 100,
#     start_at_ms: int | None = None,
#     end_at_ms: int | None = None,
#     pages: int = 1,
#     cursor: str | None = None,
#     tz_name: str | None = DEFAULT_TZ_NAME,
# ):
#     """
#     Igual que el RAW, pero imprime cada fila de funding con fechas legibles.
#     Requiere las helpers _get_raw_paradex / BASE_URL / _headers ya definidas en el archivo.
#     """
#     assert market, "market requerido, p.ej. BTC-USD-PERP"
#     curr = cursor
#     total = 0
#     for i in range(max(1, int(pages))):
#         params = {"market": market, "page_size": int(page_size)}
#         if start_at_ms is not None: params["start_at"] = int(start_at_ms)
#         if end_at_ms   is not None: params["end_at"]   = int(end_at_ms)
#         if curr: params["cursor"] = curr

#         raw = _get_raw_paradex("/funding/payments", params)
#         results = raw.get("results") or []
#         print(f"\n📄 PÁGINA {i+1} | filas={len(results)}  next={(raw.get('next') and 'sí') or 'no'}")

#         for r in results:
#             created = r.get("created_at")
#             created_loc = fmt_datetime_ms(created, tz_name)
#             created_ago = humanize_delta_ms(created)
#             market_r = r.get("market")
#             pay = r.get("payment")
#             idx = r.get("index")
#             fid = r.get("fill_id")
#             rid = r.get("id")
#             print(f" • {market_r:<16} | created={created_loc} ({created_ago}) | payment={pay} | index={idx} | fill_id={fid} | id={rid}")
#         total += len(results)

#         curr = raw.get("next")
#         if not curr:
#             break
#     print(f"\n✅ Total filas impresas (human): {total}")

# # ====== Activadores directos al ejecutar el archivo ======
# if __name__ == "__main__":
#     # Activa uno o ambos según necesites
#     RUN_FUNDING_HUMAN = True

#     # Parámetros por defecto
#     MARKET      = "OM-USD-PERP"   # ej.: "BTC-USD-PERP"
#     PAGE_SIZE   = 500              # 1..5000
#     PAGES       = 2
#     START_AT_MS = None             # ej.: 1729123200000
#     END_AT_MS   = None             # ej.: 1729209600000

#     if RUN_FUNDING_HUMAN:
#         debug_print_funding_human(
#             market=MARKET,
#             page_size=PAGE_SIZE,
#             start_at_ms=START_AT_MS,
#             end_at_ms=END_AT_MS,
#             pages=PAGES,
#             cursor=None,
#             tz_name=DEFAULT_TZ_NAME,
#         )






# # Test de la función normalize_symbol (puedes borrarlo después)
# def test_normalize_symbol():
#     test_cases = [
#         "BTC-USD-PERP",
#         "ETH-USD-PERP", 
#         "WCT-USD-PERP",
#         "ASTER-USD-PERP",
#         "BTC-PERP",
#         "ETH-USD",
#         "SOL_USD_PERP"
#     ]
    
#     print("🧪 Probando normalize_symbol:")
#     for case in test_cases:
#         result = normalize_symbol(case)
#         print(f"   '{case}' -> '{result}'")

# # Llamar la función de prueba al final del archivo
# if __name__ == "__main__":
#     test_normalize_symbol()
#     # ... el resto de tu código main ...
# =======================
# Ejecución directa para debug
# # =======================
# if __name__ == "__main__":
#     print("=" * 70)
#     print("🔧 MODO DEBUG - Reconstructor COMPLETO de Posiciones Paradex")
#     print("=" * 70)
    

#     # Inicializar base de datos
#     try:
#         init_db()
#         print("✅ [DEBUG] Base de datos inicializada")
#     except Exception as e:
#         print(f"❌ [DEBUG] Error inicializando DB: {e}")
    
#     # Reconstruir TODAS las posiciones (últimos 60 días)
#     print("\n🔄 Iniciando reconstrucción completa...")
#     reconstruct_paradex_positions(days_back=60, debug=True)