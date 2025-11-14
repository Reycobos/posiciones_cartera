from flask import Flask, render_template, jsonify, request
import pandas as pd
import requests
import time
import hashlib
from dotenv import load_dotenv
import hmac
import os
from urllib.parse import urlencode
import json
import base64
import nacl.signing
import datetime
import math
from datetime import datetime, timezone
import json, urllib
from base64 import urlsafe_b64encode
from base58 import b58decode, b58encode
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from requests import Request, Session
from collections import defaultdict
from datetime import datetime, timedelta
from collections import defaultdict
import sqlite3
from db_manager import init_db, save_closed_position, init_funding_db, upsert_funding_events, last_funding_ts, load_funding
# En portfoliov7.8.py
from adapters.gate_spot_trades import save_gate_spot_positions
from adapters.bitget_spot_trades import save_bitget_spot_positions


from universal_cache import (
    init_universal_cache_db, 
    update_cache_from_positions,
    get_cache_stats, 
    add_manual_pair,
    update_sync_timestamp,
    get_last_sync_timestamp,
    detect_closed_positions,
    get_cached_symbols
)


    
#==== codigo para sincronizar con universal_cache    
# TOGGLE para activar sync inteligente


SMART_SYNC_ENABLED = True
SMART_SYNC_LOOKBACK_HOURS = 48  # Ventana de seguridad: siempre mira 48h atrás mínimo
SMART_SYNC_MAX_DAYS = 60        # Límite superior para evitar llamadas masivas

def smart_sync_closed_positions(exchange_name: str, 
                                force_full_sync: bool = False,
                                debug: bool = False) -> int:
    """
    Sincroniza posiciones cerradas de forma inteligente:
    - Si detecta posición cerrada → sync desde last_sync con ventana de seguridad
    - Si force_full_sync=True → sync completo (ignora caché)
    - Si es primera vez → sync de SMART_SYNC_MAX_DAYS días
    
    Retorna: número de posiciones guardadas
    """
    
    if not should_sync(exchange_name):
        if debug:
            print(f"⏭️  {exchange_name} deshabilitado en CLOSED_EXCHANGES")
        return 0
    
    # 1. Obtener posiciones abiertas actuales
    try:
        fetch_positions = POSITIONS_FUNCTIONS.get(exchange_name)
        if not fetch_positions:
            print(f"⚠️  No hay función de posiciones para {exchange_name}")
            return 0
            
        current_positions = fetch_positions()
        
    except Exception as e:
        print(f"⚠️  Error obteniendo posiciones abiertas de {exchange_name}: {e}")
        current_positions = []
    
    # 2. Detectar cambios (posiciones cerradas)
    disappeared_symbols = detect_closed_positions(exchange_name, current_positions)
    # ✅ DEBUG
    print(f"🔍 {exchange_name}:")
    print(f"   📦 Posiciones actuales: {[p.get('symbol') for p in current_positions]}")
    print(f"   🗄️  Cache tiene: {get_cached_symbols(exchange_name)}")  # Nueva función helper
    print(f"   🎯 Detectadas cerradas: {disappeared_symbols}")
    
    # 3. Calcular ventana temporal
    last_sync_ms = get_last_sync_timestamp(exchange_name)
    now_ms = int(time.time() * 1000)
    
    if force_full_sync or last_sync_ms is None:
        # Sync completo
        days_to_sync = UNIVERSAL_CACHE_TTL_DAYS
        reason = "primera vez" if last_sync_ms is None else "forzado"
        if debug:
            print(f"🔄 {exchange_name}: Sync completo ({reason}) - {days_to_sync} días")
    
    elif disappeared_symbols:
        # Hay posiciones cerradas detectadas → sync desde last_sync con buffer
        lookback_ms = FUNDING_GRACE_HOURS * 3600 * 1000  # Reutilizamos el toggle de funding
        since_ms = max(0, last_sync_ms - lookback_ms)
        days_to_sync = min(
            int((now_ms - since_ms) / (24 * 3600 * 1000)) + 1,
            UNIVERSAL_CACHE_TTL_DAYS
        )
        
        if debug:
            print(f"🎯 {exchange_name}: Detectadas {len(disappeared_symbols)} posiciones cerradas")
            print(f"   Símbolos: {', '.join(list(disappeared_symbols)[:5])}")
            print(f"   Sync desde: {_fmt_ms(since_ms)} ({days_to_sync} días)")
    
    else:
        # No hay cambios → skip
        if debug:
            print(f"✅ {exchange_name}: Sin cambios detectados, skip sync")
        return 0
    
    # 4. Ejecutar sync con la función del adapter
    # ⚠️ CORRECCIÓN: usar el diccionario global SYNC_FUNCTIONS (sin 's' extra)
    sync_fn = SYNC_FUNCTIONS.get(exchange_name)  # ← Variable local con nombre diferente
    if not sync_fn:
        print(f"⚠️  No hay función de sync para {exchange_name}")
        return 0
    
    try:
        if debug:
            print(f"⏳ Sincronizando {exchange_name} ({days_to_sync} días)...")
        
        # Llamar con el parámetro days calculado
        result = sync_fn(days=days_to_sync)  # ← Usar sync_fn (local), no sync_functions (global)
        saved = result if isinstance(result, int) else (result[0] if isinstance(result, tuple) else 0)
        
        # 5. Actualizar timestamps y caché
        update_sync_timestamp(exchange_name)
        update_cache_from_positions(exchange_name, current_positions)
        
        if debug:
            print(f"✅ {exchange_name}: {saved} posiciones guardadas")
        
        return saved
        
    except Exception as e:
        print(f"❌ Error en sync de {exchange_name}: {e}")
        import traceback
        traceback.print_exc()
        return 0
    
#==== fin del codigo para sincronizar con universal_cache    
# Configuración
SPOT_TRADE_ALL = True
SPOT_TRADE_EXCHANGES = {"gate": True}


DB_PATH = "portfolio.db"
# =========================
# 🎛️ TOGGLES FUNDING
# =========================
SYNC_FUNDING_ON_START = True     # Sincroniza funding al arrancar el servidor
SYNC_FUNDING_ON_EMPTY = True     # Si /api/funding no encuentra datos, fuerza una sync
FUNDING_DEFAULT_DAYS = None     # Días por defecto que devuelve /api/funding
# =========================
FUNDING_GRACE_HOURS = 36          # margen de seguridad desde la última ejecución
FUNDING_GATE_ACTIVITY = True     # solo sincronizar exchanges “activos”
FUNDING_ACTIVE_WINDOW_DAYS = 7   # si hubo cerradas en estos días, consideramos activo
FUNDING_CACHE_TTL_SEC = 300      # cache de open positions (5 min)

UNIVERSAL_CACHE_TTL_DAYS = 7  # TOGGLE CONFIGURABLE

# =====================================================
# 🎛️ CONFIGURACIÓN DE EXCHANGES A SINCRONIZAR
# =====================================================
CLOSED_ALL = True
CLOSED_EXCHANGES = {
    "backpack": False,
    "aden": False, 
    "bingx": False,
    "aster": False,
    "binance": False,
    "extended": False,
    "kucoin": False,
    "gate": False,
    "mexc": False,
    "bitget": False,
    "okx": False,
    "paradex": False,
    "hyperliquid": False,
    "whitebit": False,
    "xt": False,
    "bybit": False,
    "lbank": False
}
# Cambia a True solo los exchanges que quieres sincronizar
# Función helper para verificar si un exchange debe sincronizarse
def should_sync(exchange_name):
    if CLOSED_ALL:
        return True
    return CLOSED_EXCHANGES.get(exchange_name, False)


BALANCE_ALL = True  # Si pones True, pedirá balances de TODOS los exchanges
BALANCE_EXCHANGES = {
    "backpack": False,
    "aden": False,
    "bingx": False,
    "aster": False,
    "binance": False,
    "extended": False,
    "kucoin": False,
    "gate": False,
    "mexc": False,
    "bitget": True,
    "okx": False,
    "paradex": False,
    "hyperliquid": True,
    "whitebit": False,
    "xt": False,
    "bybit": True,
    "lbank": True
}

# =========================
# 🎛️ TOGGLES DE PRINT
# =========================
PRINT_CLOSED_SYNC = True        # 1) Sincronización de posiciones cerradas (inicio/fin por exchange)
PRINT_CLOSED_DEBUG = True       # 2) Debug "bonito" de normalización de posiciones cerradas
PRINT_OPEN_POSITIONS = False     # 3) Posiciones abiertas (resumen por exchange + bloques por símbolo)
PRINT_FUNDING = True            # 4) Funding: solo # de registros por exchange
PRINT_BALANCES = False           # 5) Balances: solo equity total por exchange


# Configuración
SPOT_TRADE_ALL = True
SPOT_TRADE_EXCHANGES = {
    "gate": True,
    "lbank": True,
    "bitget": True,
    
}
#============== Helper para formatear tiempos a horas legibles
from datetime import datetime, timezone

def _fmt_ms(ms) -> str:
    """Convierte ms/seg a 'YYYY-MM-DD HH:MM:SS UTC'."""
    try:
        ms = int(ms or 0)
        if ms and ms < 1_000_000_000_000:  # venía en segundos
            ms *= 1000
        return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(ms)
    
#============== Helper para formatear tiempos a horas legibles

#======= Nuevo sistema de funding , a partit de version 7.3
# ===== Estado de sincronización funding (por exchange) =====
def _init_funding_sync_state(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS funding_sync_state (
        exchange TEXT PRIMARY KEY,
        last_run_ms INTEGER,
        last_ingested_ms INTEGER
    )
    """)
    conn.commit()
    conn.close()

def _get_sync_state(exchange: str, db_path=DB_PATH):
    conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT last_run_ms, last_ingested_ms FROM funding_sync_state WHERE exchange=?", (exchange,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else {"last_run_ms": None, "last_ingested_ms": None}

def _set_sync_state(exchange: str, last_run_ms: int, last_ingested_ms: int | None, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO funding_sync_state(exchange, last_run_ms, last_ingested_ms)
    VALUES(?,?,?)
    ON CONFLICT(exchange) DO UPDATE SET
        last_run_ms=excluded.last_run_ms,
        last_ingested_ms=COALESCE(excluded.last_ingested_ms, funding_sync_state.last_ingested_ms)
    """, (exchange, last_run_ms, last_ingested_ms))
    conn.commit(); conn.close()
    
    

# ===== Cache de exchanges con posiciones abiertas (para gate de funding) =====
_OPEN_POS_CACHE = {"ts": 0, "exchanges": set()}

def _update_open_pos_cache(ex_list: set):
    _OPEN_POS_CACHE["ts"] = int(time.time())
    _OPEN_POS_CACHE["exchanges"] = set(e.lower() for e in ex_list)

def _get_active_exchanges_from_cache():
    now = int(time.time())
    if now - _OPEN_POS_CACHE["ts"] <= FUNDING_CACHE_TTL_SEC:
        return set(_OPEN_POS_CACHE["exchanges"])
    return set()


def _exchanges_with_recent_closed(days=FUNDING_ACTIVE_WINDOW_DAYS, db_path=DB_PATH):
    try:
        cutoff = int(time.time()) - days*24*3600
        conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT LOWER(exchange) AS ex
            FROM closed_positions
            WHERE COALESCE(close_time, open_time, 0) >= ?
        """, (cutoff,))
        rows = [r["ex"] for r in cur.fetchall()]
        conn.close()
        return set(rows)
    except Exception:
        return set()

def _determine_exchanges_to_sync():
    # punto de partida: todos los pullers
    all_ex = set(FUNDING_PULLERS.keys())

    if not FUNDING_GATE_ACTIVITY:
        return list(all_ex)

    active_open = _get_active_exchanges_from_cache()
    active_closed = _exchanges_with_recent_closed()
    # siempre incluye los marcados “True” en CLOSED_EXCHANGES por si quieres forzar
    forced = {ex for ex, on in CLOSED_EXCHANGES.items() if on}

    # union de activity + forced; si queda vacío, como fallback usa “binance” si existe
    result = (active_open | active_closed | forced) & all_ex
    if not result and "binance" in all_ex:
        result = {"binance"}
    return sorted(result)

def _call_funding_with_since(fn, since_ms: int):
    # intenta pasar since=; si no acepta, llama “a secas” y filtra por timestamp
    try:
        return fn(since=since_ms) or []
    except TypeError:
        data = fn() or []
        res = []
        for e in data:
            t = e.get("timestamp") or e.get("time") or e.get("created_at")
            if _safe_ts(t) >= since_ms:
                res.append(e)
        return res

#======= Fin del Nuevo sistema de funding , a partit de version 7.3
# =========================
# =========================
# 🧩 HELPERS DE FORMATO
# =========================
def _ex_disp(name: str) -> str:
    """Nombre consistente para consola."""
    mapping = {
        "binance": "Binance",
        "backpack": "Backpack",
        "aden": "Aden",
        "bingx": "BingX",
        "aster": "Aster",
        "extended": "Extended",
        "kucoin": "KuCoin",
        "gate": "Gate.io",
        "mexc": "MEXC",
        "bitget": "Bitget",
        "okx": "OKX",
        "paradex":"Paradex",
        "hyperliquid": "Hyperliquid",
        "whitebit": "Whitebit",
        "xt": "XT",
        "lbank": "Lbank",
    }
    return mapping.get((name or "").lower(), name)

# ========== 1) CERRADAS: SINCRONIZACIÓN ==========
def p_closed_sync_start(exchange: str):
    if PRINT_CLOSED_SYNC:
        print(f"⏳ Sincronizando fills cerrados de {_ex_disp(exchange)}...")

def p_closed_sync_saved(exchange: str, saved: int, dup: int):
    if PRINT_CLOSED_SYNC:
        # Mensaje uniforme para todos
        print(f"✅ Guardadas {saved} posiciones cerradas de {_ex_disp(exchange)} (omitidas {dup} duplicadas).")

def p_closed_sync_done(exchange: str):
    if PRINT_CLOSED_SYNC:
        print(f"✅ Posiciones cerradas de {_ex_disp(exchange)} actualizadas correctamente.")

def p_closed_sync_none(exchange: str):
    if PRINT_CLOSED_SYNC:
        # Para los casos "no hay resultados"
        print(f"⚠️ No se obtuvieron posiciones cerradas de {_ex_disp(exchange)}.")

# ========== 2) CERRADAS: DEBUG BONITO ==========
def p_closed_debug_header(symbol: str):
    if PRINT_CLOSED_DEBUG:
        print(f"🔎 {symbol.upper()}")

def p_closed_debug_count(n: int):
    if PRINT_CLOSED_DEBUG:
        print(f"📦 DEBUG: Se recibieron {n} registros de posiciones cerradas")

def p_closed_debug_norm_size(side: str, size: float):
    if PRINT_CLOSED_DEBUG:
        print(f"   📏 Size: {size:.4f} | 🎯 Side: {side}")

def p_closed_debug_prices(entry: float, close: float):
    if PRINT_CLOSED_DEBUG:
        print(f"   💰 Entry: {entry} | Close: {close}")

def p_closed_debug_pnl(pnl: float, fee: float, funding: float):
    if PRINT_CLOSED_DEBUG:
        print(f"   📊 PnL: {pnl} | Fee: {fee} | Funding: {funding}")

def p_closed_debug_times(open_raw, close_raw, open_sec, close_sec):
    if PRINT_CLOSED_DEBUG:
        print(f"   ⏰ Open raw: {open_raw} | Close raw: {close_raw}")
        print(f"   ⏰ Open sec: {open_sec} | Close sec: {close_sec}")

def p_closed_debug_normalized(symbol: str, pnl: float):
    if PRINT_CLOSED_DEBUG:
        print(f"   ✅ Normalizada: {symbol.upper()} - PnL: {pnl}")

# ========== 3) ABIERTAS ==========
def p_open_summary(exchange: str, count: int):
    if PRINT_OPEN_POSITIONS:
        print(f"📈 {_ex_disp(exchange)}: {count} posiciones abiertas")

def p_open_block(exchange: str, symbol: str, qty: float, entry: float, mark: float,
                 unrealized: float, realized_funding: float | None, total_unsettled: float | None,
                 notional: float | None, extra_verification: bool = False):
    if not PRINT_OPEN_POSITIONS:
        return
    print(f"   🔎 {symbol.upper()}")
    print(f"      📦 Quantity: {qty}")
    print(f"      💰 Entry: {entry} | Mark: {mark}")
    print(f"      📉 Unrealized PnL: {unrealized}")
    if realized_funding is not None:
        print(f"      💵 Realized Funding: {realized_funding}")
    if total_unsettled is not None:
        print(f"      🧮 Total (API Unsettled): {total_unsettled}")
    if extra_verification and realized_funding is not None:
        # muestra línea estilo “Verificación: x + y = z” si procede
        z = (unrealized or 0) + (realized_funding or 0)
        print(f"      ✅ Verificación: {unrealized} + {realized_funding} = {z}")
    if notional is not None:
        print(f"      🏦 Notional: {notional}")

# ========== 4) FUNDING ==========
def p_funding_fetching(exchange: str):
    if PRINT_FUNDING:
        print(f"🔍 DEBUG: Obteniendo FUNDING FEES (USDT) de {_ex_disp(exchange)}...")

def p_funding_count(exchange: str, n: int):
    if PRINT_FUNDING:
        print(f"📦 DEBUG: Se recibieron {n} registros de funding")

# ========== 5) BALANCES ==========
def p_balance_equity(exchange: str, equity: float):
    if PRINT_BALANCES:
        print(f"💼 {_ex_disp(exchange)} equity total: {equity:.2f}")
        
        


# =========== Imports adapters=======


from adapters.backpack import (
    fetch_positions_backpack,
    fetch_funding_backpack,
    fetch_account_backpack,
    save_backpack_closed_positions,
)

from adapters.aden import (
    fetch_positions_aden,
    fetch_funding_aden,
    fetch_account_aden,
    save_aden_closed_positions,
    _send_request,
    )

from adapters.extended import (
    fetch_account_extended,
    fetch_open_extended_positions, 
    fetch_funding_extended,
    save_extended_closed_positions
)

from adapters.gate import (
    fetch_gate_open_positions,
    fetch_gate_funding_fees,
    fetch_gate_all_balances,
    save_gate_closed_positions
)

from adapters.aster import (
    fetch_aster_open_positions,
    pull_funding_aster,
    fetch_account_aster,
    save_aster_closed_positions,
    )

from adapters.binance import (
    fetch_account_binance,
    fetch_positions_binance_enriched, 
    pull_funding_binance,
    save_binance_closed_positions
)

from adapters.bingx import (
    fetch_bingx_all_balances,
    fetch_bingx_open_positions,
    fetch_funding_bingx,
    save_bingx_closed_positions,
    debug_cache_status,
    force_cache_update,
)

from adapters.kucoin import (
    fetch_kucoin_all_balances,
    fetch_kucoin_open_positions,
    fetch_funding_kucoin,
    save_kucoin_closed_positions,
)
from adapters.mexc import (
    fetch_mexc_all_balances,
    fetch_mexc_open_positions,
    fetch_mexc_funding_fees,
    save_mexc_closed_positions,
)

from adapters.bitget import (
    fetch_bitget_all_balances,
    fetch_bitget_open_positions, 
    fetch_bitget_funding_fees,
    save_bitget_closed_positions
)

from adapters.okx import (
    fetch_okx_open_positions,
    fetch_okx_funding_fees,
    fetch_okx_all_balances,
    save_okx_closed_positions,
)

from adapters.paradexv9s import (
      fetch_paradex_open_positions,
      fetch_paradex_funding_fees,
      fetch_paradex_all_balances,
      save_paradex_closed_positions,
)

from adapters.hyperliquidv5 import (
    fetch_hyperliquid_open_positions,
    fetch_hyperliquid_funding_fees,
    fetch_hyperliquid_all_balances,
    save_hyperliquid_closed_positions
)

from adapters.whitebit import (
    fetch_whitebit_open_positions,
    fetch_whitebit_funding_fees,
    fetch_whitebit_all_balances,
    save_whitebit_closed_positions,
)

from adapters.xt1 import (
    fetch_xt_open_positions,
    fetch_xt_funding_fees,
    fetch_xt_all_balances,
    save_xt_closed_positions,
)

from adapters.bybit import (
    fetch_bybit_open_positions,
    fetch_bybit_funding_fees,
    fetch_bybit_all_balances,
    save_bybit_closed_positions,
)
from adapters.lbank import (
    fetch_lbank_all_balances,
    save_lbank_closed_positions
)


__all__ = [
    "fetch_bingx_all_balances", "fetch_bingx_open_positions", "fetch_funding_bingx", "save_bingx_closed_positions",
]

#==== funcion para spot trades
def should_sync_spot(exchange_name):
    if SPOT_TRADE_ALL:
        return True
    return SPOT_TRADE_EXCHANGES.get(exchange_name, False)

# listen_key = get_listen_key()
# from bingx_ws_listener import start_bingx_ws_listener
# latest_funding_bingx = start_bingx_ws_listener(listen_key)


app = Flask(__name__)

from api_manual_import import bp_manual_import
app.register_blueprint(bp_manual_import)

# Verificar que la carpeta templates existe
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
if not os.path.exists(template_dir):
    print(f"⚠️ Creando carpeta ss: {template_dir}")
    os.makedirs(template_dir)

TEMPLATE_FILE = "indexv2.html" 
DB_PATH = "portfolio.db"




BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
BYBIT_BASE_URL = "https://api.bybit.com"





def should_fetch_balance(exchange_name: str) -> bool:
    """
    Devuelve True si se deben pedir balances para 'exchange_name' según
    la configuración de BALANCE_ALL y BALANCE_EXCHANGES.
    """
    return bool(BALANCE_ALL or BALANCE_EXCHANGES.get(exchange_name, False))    
def main_balances():
    print("💰 Iniciando consulta de balances...")

    # --- Ejemplo de funciones. Ajusta nombres/args a tu API real ---
    balance_functions = {
        "backpack": lambda: fetch_account_backpack(db_path="portfolio.db"),
        "aden":     lambda: fetch_account_aden(db_path="portfolio.db", debug=False),
        "aster":    lambda: fetch_account_aster()(db_path="portfolio.db"),
        "binance":  lambda: fetch_account_binance(db_path="portfolio.db", spot=True, futures=True),
        "extended": lambda: fetch_account_extended(db_path="portfolio.db"),
        "kucoin": lambda: fetch_kucoin_all_balances(),
        "gate":     lambda: fetch_gate_all_balances(settles=("usdt",)),
        "bingx": lambda: fetch_bingx_all_balances(),
        "mexc": fetch_mexc_all_balances,
        "bitget": lambda: fetch_bitget_funding_fees(limit=2000, chunk=100, max_pages=50),
        "okx":     lambda: fetch_okx_all_balances(db_path="portfolio.db"),
        "paradex": lambda: fetch_paradex_all_balances(db_path="portfolio.db", days=60, debug=PRINT_CLOSED_DEBUG),
        "hyperliquid": lambda: fetch_hyperliquid_all_balances("portfolio.db", days=60, debug=False),
        "whitebit": lambda: fetch_whitebit_all_balances(),
        "xt": lambda: fetch_xt_all_balances(db_path="portfolio.db"),
        "bybit": lambda: fetch_bybit_all_balances(db_path="portfolio.db"),
        "lbank": lambda: fetch_lbank_all_balances(db_path="portfolio.db"),
        
    }
    
    # ---------------------------------------------------------------

    for exchange_name, fetch_fn in balance_functions.items():
        if should_fetch_balance(exchange_name):
            print(f"⏳ Pidiendo balances de {exchange_name.capitalize()}...")
            try:
                balances = fetch_fn()
                # Si tu flujo necesita persistir, hazlo aquí:
                # save_balances("portfolio.db", exchange=exchange_name, data=balances)
                print(f"✅ Balances de {exchange_name.capitalize()} obtenidos correctamente.")
            except Exception as e:
                print(f"❌ Error al pedir balances de {exchange_name.capitalize()}: {e}")
        else:
            print(f"⏭️  Saltando {exchange_name.capitalize()}")

    print("🧩 Consulta de balances completada.")
#===============Codigo para insertar funding en base de datos

def _ms_now(): return int(time.time()*1000)

def _safe_ts(x):
    x = int(x or 0)
    return x*1000 if x and x < 1_000_000_000_000 else x

# Mapea cada adapter a una función que devuelva "eventos recientes".
# Si el adapter no acepta timestamps, usa un limit alto y luego filtramos por ts.
FUNDING_PULLERS = {
    "backpack":       lambda: fetch_funding_backpack(limit=500),
    "aster":          lambda **kw: pull_funding_aster(**kw),
    "binance":        lambda **kw: pull_funding_binance(**kw),  # tiene income history
    "aden":           lambda: fetch_funding_aden(limit=500),
    "extended":       lambda: fetch_funding_extended(limit=500, debug=False),
    "kucoin":         lambda: fetch_funding_kucoin(limit=500),
    "mexc":           lambda: fetch_mexc_funding_fees(limit=1000),
    "gate":           lambda: fetch_gate_funding_fees(limit=1000),
    "okx":            lambda: fetch_okx_funding_fees(limit=1000),
    "bitget":         lambda: fetch_bitget_funding_fees(limit=2000, chunk=100, max_pages=200),
    "bingx":          lambda: fetch_funding_bingx(limit=1000, start_time=_ms_now()-14*24*3600*1000, end_time=_ms_now()),
    "paradex":        lambda: fetch_paradex_funding_fees(limit=1000),
    "hyperliquid":    lambda: fetch_hyperliquid_funding_fees(limit=1000),
    "whitebit":       lambda: fetch_whitebit_funding_fees(limit=1000),
    "xt":             lambda: fetch_xt_funding_fees(limit=1000),
    "bybit":          lambda **kw: fetch_bybit_funding_fees(limit=50, **kw),
}

def _normalize_ms(ev):
    ev = dict(ev)
    ts = ev.get("timestamp") or ev.get("time") or ev.get("created_at")
    ev["timestamp"] = _safe_ts(ts)
    return ev

def _std_event(exchange: str, ev: dict) -> dict:
    """
    Normaliza un evento de funding a nuestro formato esperado por la DB.
    - Asegura 'exchange'
    - Normaliza 'symbol' (quita USDT/USDC/USD/PERP)
    - Asegura 'asset', 'type'
    - Convierte 'income' a float
    - Genera 'external_id' si falta (para evitar colisiones/NULLs)
    """
    out = dict(ev or {})
    out["exchange"] = (out.get("exchange") or exchange).lower()

    # símbolo base (reusa tu helper)
    out["symbol"] = _base_symbol(
        out.get("symbol") or out.get("market") or out.get("pair") or ""
    )

    out["asset"] = out.get("asset") or "USDT"
    out["type"] = out.get("type") or out.get("incomeType") or "FUNDING_FEE"

    try:
        out["income"] = float(out.get("income"))
    except Exception:
        out["income"] = float(out.get("amount") or 0)

    # timestamp ya viene en ms desde _normalize_ms
    out["timestamp"] = int(out.get("timestamp") or 0)

    # external_id robusto
    eid = out.get("external_id") or out.get("tranId") or out.get("txId") or out.get("id")
    if eid:
        out["external_id"] = str(eid)
    else:
        # fallback determinista para evitar duplicados/NULLs
        out["external_id"] = f"{out['exchange']}|{out['symbol']}|{out['timestamp']}|{out['income']}"
    return out

def sync_all_funding(exchanges: list | None = None,
                     force_days: int | None = None,
                     verbose: bool = True) -> dict:
    """
    Sincroniza funding:
      - Si force_days es int: desactiva gate y sincroniza TODOS los exchanges
        desde now - force_days días (no usa estado previo).
      - Si force_days es None: modo incremental + gate por actividad, con margen FUNDING_GRACE_HOURS.
    """
    now_ms = _ms_now()
    inserted_by_ex = {}

    # 1) Decide exchanges a procesar
    if exchanges is not None and len(exchanges) > 0:
        target_ex = [e.lower() for e in exchanges]
    else:
        if isinstance(FUNDING_DEFAULT_DAYS, int) and FUNDING_DEFAULT_DAYS > 0:
            # Si hay valor en el toggle, úsalo como force_days
            force_days = FUNDING_DEFAULT_DAYS
        if isinstance(force_days, int) and force_days > 0:
            # Modo forzado: todos los exchanges definidos en pullers
            target_ex = sorted(FUNDING_PULLERS.keys())
        else:
            # Modo normal: gate por actividad
            target_ex = _determine_exchanges_to_sync()

    # 2) Lógica de "desde cuándo" pedir
    grace_ms = FUNDING_GRACE_HOURS * 3600 * 1000
    if isinstance(force_days, int) and force_days > 0:
        since_ms_global = now_ms - force_days * 24 * 3600 * 1000
        mode = f"FORCED {force_days}d"
    else:
        since_ms_global = None
        mode = "INCREMENTAL"

    if verbose:
        print(f"🔧 Funding sync mode: {mode}; exchanges={target_ex}")

    for ex in target_ex:
        try:
            if ex not in FUNDING_PULLERS:
                inserted_by_ex[ex] = 0
                if verbose:
                    print(f"   · {ex}: no hay puller definido")
                continue

            # 2a) calcula since_ms por exchange
            if since_ms_global is not None:
                since_ms = since_ms_global
            else:
                state = _get_sync_state(ex)
                since_ms = 0
                if state["last_ingested_ms"]:
                    since_ms = max(since_ms, int(state["last_ingested_ms"]))
                if state["last_run_ms"]:
                    since_ms = max(since_ms, int(state["last_run_ms"]))
                since_ms = max(0, since_ms - grace_ms)

            # 3) Llama al adapter con soporte de `since` si puede; si no, filtramos
            raw = _call_funding_with_since(FUNDING_PULLERS[ex], since_ms) or []
            raw_ms = [_normalize_ms(r) for r in raw]
            norm = [_std_event(ex, r) for r in raw_ms]
            recent = [e for e in norm if int(e.get("timestamp") or 0) >= since_ms]

            # 4) Inserta
            inserted = upsert_funding_events(recent)
            inserted_by_ex[ex] = inserted

            # 5) Actualiza estado
            max_ingested = max([e["timestamp"] for e in recent], default=None)
            # Siempre dejamos constancia de last_run; last_ingested solo si insertamos algo
            _set_sync_state(ex, last_run_ms=now_ms, last_ingested_ms=max_ingested)

            if verbose:
                since_hr = _fmt_ms(since_ms)
                print(f"🔁 Funding {ex}: recibidos={len(raw)} norm={len(norm)} nuevos={inserted} (since={since_hr})")

        except Exception as e:
            print(f"❌ Funding sync error {ex}: {e}")
            inserted_by_ex[ex] = 0
            # Aún así marca last_run para no bloquear futuros ciclos
            _set_sync_state(ex, last_run_ms=now_ms, last_ingested_ms=None)

    return inserted_by_ex

@app.route('/api/funding')
def api_funding():
    """Lee funding desde SQLite; si ?refresh=1, sincroniza antes.
       Si la tabla está vacía y SYNC_FUNDING_ON_EMPTY=True, sincroniza una vez."""
    try:
        # 1) Forzar sync si se pide (?refresh=1) y, opcionalmente, con días forzados
        force_days_q = request.args.get("days", default=None, type=int)
        if request.args.get("refresh") == "1":
            sync_all_funding(force_days=force_days_q, verbose=False)  # usa None si no hay número

        # 2) Parse robusto de 'days' (puede venir None/''/'none'/'null' o un número)
        raw_days = request.args.get("days", default=None)
        days = None
        if raw_days is not None and str(raw_days).strip() != "" and str(raw_days).lower() not in ("none", "null", "false"):
            try:
                days = int(raw_days)
            except Exception:
                days = None
        # fallback al toggle si es int; si el toggle es None, mantenemos None
        if days is None and isinstance(FUNDING_DEFAULT_DAYS, int):
            days = FUNDING_DEFAULT_DAYS

        exchange = request.args.get("exchange")
        symbol   = request.args.get("symbol")
        include_estimates = request.args.get("estimates", "1") != "0"

        data = load_funding(days=days, exchange=exchange, symbol=symbol,
                            include_estimates=include_estimates, limit=10000)

        # Si la tabla está vacía y está activado el auto-sync en vacío, dispara una vez
        if not data and SYNC_FUNDING_ON_EMPTY:
            sync_all_funding(force_days=force_days_q, verbose=False)
            data = load_funding(days=days, exchange=exchange, symbol=symbol,
                                include_estimates=include_estimates, limit=10000)

        return jsonify({"funding": data})
    except Exception as e:
        print(f"❌ /api/funding error: {e}")
        return jsonify({"funding": []})
# Routers

import re

def _base_symbol(sym: str) -> str:
    """
    2ZUSDT -> 2Z
    2ZUSDC -> 2Z
    KAITO-PERP -> KAITO
    KAITOUSDC-PERP -> KAITO
    """
    s = (sym or "").upper()
    # quitar quotes al final (con - o / opcional)
    s = re.sub(r'[-_/]?(USDT|USDC)$', '', s)
    # quitar sufijo PERP al final
    s = re.sub(r'[-_/]?PERP$', '', s)
    return s



@app.route("/api/closed_positions")
def api_closed_positions():
    """
    Devuelve grupos de posiciones cerradas:
      1) Delta-neutral (short futures + spotbuy)  <- PRIMERO (marca emparejados)
      2) Clustering normal de futuros             <- SEGUNDO (excluye marcados)
      3) Swaps de stablecoins
      4) Spots sueltos (spotbuy/spotsell) no emparejados
    """
    try:
        WINDOW_SEC = 15 * 60       # 15 minutos
        SIZE_EPS_REL = 0.001       # 0.1%

        conn = sqlite3.connect("portfolio.db")
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
             SELECT exchange, symbol, side, size, entry_price, close_price, pnl,
                    realized_pnl, funding_total AS funding_fee,
                    fee_total AS fees, pnl_percent, apr, initial_margin, notional, 
                    open_time, close_time
             FROM closed_positions
             ORDER BY open_time ASC
        """)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()

        for r in rows:
            r["side"] = (r.get("side") or "").lower()

        # Clasificación
        futures_trades = [r for r in rows if r["side"] in ("long", "short")]
        spot_trades    = [r for r in rows if r["side"] in ("spotbuy", "spotsell", "swapstable")]

        # Index por base
        def _base_symbol(sym: str) -> str:
            s = (sym or "").upper()
            # ejemplos: ALPACAUSDT -> ALPACA ; BTCUSDT -> BTC
            if s.endswith("USDT"):
                return s[:-4]
            if s.endswith("USDC"):
                return s[:-4]
            if s.endswith("USD"):
                return s[:-3]
            return s

        futures_by_base = {}
        for r in futures_trades:
            base = _base_symbol(r["symbol"])
            r["_base"] = base
            futures_by_base.setdefault(base, []).append(r)

        spot_by_base = {}
        for r in spot_trades:
            base = _base_symbol(r["symbol"])
            r["_base"] = base
            spot_by_base.setdefault(base, []).append(r)

        # ==========================================================
        # PASO 1: EMPAREJAMIENTO DELTA NEUTRAL (PRIMERO)
        # ==========================================================
        delta_neutral_groups = []
        paired_futures_ids = set()

        for base, futures in futures_by_base.items():
            short_futures = [f for f in futures if f["side"] == "short"]
            matching_spot_buys = [s for s in spot_by_base.get(base, []) if s["side"] == "spotbuy"]

            for short in short_futures:
                short_id = id(short)
                if short_id in paired_futures_ids:
                    continue

                short_time = int(short.get("open_time") or 0)
                short_size = float(short.get("size") or 0.0)

                # Encuentra el mejor spotbuy: cerca en el tiempo y tamaño similar
                best_spot = None
                min_time_diff = float("inf")

                for spot in matching_spot_buys:
                    spot_time = int(spot.get("open_time") or 0)
                    spot_size = float(spot.get("size") or 0.0)
                    time_diff = abs(short_time - spot_time)

                    if short_size <= 0:
                        continue

                    size_rel = abs(spot_size - short_size) / max(short_size, 1e-12)
                    if time_diff <= 3600 and size_rel <= 0.10:  # 1h y 10% tolerancia
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            best_spot = spot

                if best_spot:
                    # 🔥 Marca ANTES del clustering normal
                    paired_futures_ids.add(short_id)
                    short["_paired_delta_neutral"] = True
                    best_spot["_paired_delta_neutral"] = True

                    legs = [short, best_spot]
                    size_total = short_size
                    notional_total = (float(short.get("notional") or 0.0) +
                                      float(best_spot.get("notional") or 0.0))
                    fees_total = (float(short.get("fees") or 0.0) +
                                  float(best_spot.get("fees") or 0.0))
                    funding_total = float(short.get("funding_fee") or 0.0)
                    realized_total = (float(short.get("realized_pnl") or 0.0) +
                                      float(best_spot.get("realized_pnl") or 0.0))
                    pnl_total = realized_total

                    open_time = min(short_time, int(best_spot.get("open_time") or 0))
                    close_time = max(int(short.get("close_time") or 0),
                                     int(best_spot.get("close_time") or 0))

                    delta_group = {
                        "symbol": base,
                        "positions": legs,
                        "size_total": size_total,
                        "notional_total": notional_total,
                        "pnl_total": pnl_total,
                        "fees_total": fees_total,
                        "funding_total": funding_total,
                        "realized_total": realized_total,
                        "entry_avg": float(best_spot.get("entry_price") or 0.0),
                        "close_avg": float(short.get("close_price") or 0.0),
                        "open_date": datetime.fromtimestamp(open_time).strftime("%Y-%m-%d %H:%M") if open_time else "-",
                        "close_date": datetime.fromtimestamp(close_time).strftime("%Y-%m-%d %H:%M") if close_time else "-",
                        "type": "delta_neutral",
                    }
                    delta_neutral_groups.append(delta_group)
                    matching_spot_buys.remove(best_spot)
                    

        # ==========================================================
        # PASO 2: CLUSTERING NORMAL DE FUTUROS (EXCLUYE EMPAREJADOS)
        # ==========================================================
        normal_groups = []

        for base, items in futures_by_base.items():
            # Filtra los ya emparejados
            items = [p for p in items if not p.get("_paired_delta_neutral")]

            items.sort(key=lambda x: (int(x.get("open_time") or 0), int(x.get("close_time") or 0)))
            clusters = []

            for p in items:
                ot = int(p.get("open_time") or 0)
                ct = int(p.get("close_time") or 0)
                size = float(p.get("size") or 0.0)

                best_idx = -1
                best_score = None
                for i, c in enumerate(clusters):
                    time_diff = min(abs(ot - c["open_ref"]), abs(ct - c["close_ref"]))
                    fits_time = time_diff <= WINDOW_SEC
                    ref = max(1e-12, c["size_ref"])
                    fits_size = abs(size - c["size_ref"]) / ref <= SIZE_EPS_REL
                    if fits_time or fits_size:
                        score = (time_diff, abs(size - c["size_ref"]))
                        if best_score is None or score < best_score:
                            best_score = score
                            best_idx = i

                if best_idx == -1:
                    clusters.append({
                        "legs": [p],
                        "open_ref": ot or ct,
                        "close_ref": ct or ot,
                        "size_ref": size
                    })
                else:
                    c = clusters[best_idx]
                    c["legs"].append(p)
                    c["open_ref"] = min(c["open_ref"], ot or c["open_ref"])
                    c["close_ref"] = max(c["close_ref"], ct or c["close_ref"])
                    c["size_ref"] = (c["size_ref"] + size) / 2.0 if size > 0 else c["size_ref"]

            for c in clusters:
                legs = c["legs"]
                size_total = sum(float(x.get("size") or 0.0) for x in legs)
                notional_total = sum(float(x.get("notional") or 0.0) for x in legs)
                fees_total = sum(float(x.get("fees") or 0.0) for x in legs)
                funding_total = sum(float(x.get("funding_fee") or 0.0) for x in legs)
                realized_total = sum(float(x.get("realized_pnl") or 0.0) for x in legs)
                pnl_fifo_total = sum(float(x.get("pnl") or 0.0) for x in legs)

                # fallback simple si no hay pnl FIFO
                pnl_simple_total = 0.0
                for x in legs:
                    size_val = float(x.get("size") or 0.0)
                    entry_val = float(x.get("entry_price") or 0.0)
                    close_val = float(x.get("close_price") or 0.0)
                    side_val = (x.get("side") or "").lower()
                    pnl_simple_total += (entry_val - close_val) * size_val if side_val == "short" \
                                        else (close_val - entry_val) * size_val

                pnl_total = pnl_fifo_total if abs(pnl_fifo_total) != 0 else pnl_simple_total

                if size_total > 0:
                    entry_weighted = sum(float(x.get("entry_price") or 0.0) * float(x.get("size") or 0.0) for x in legs) / size_total
                    close_weighted = sum(float(x.get("close_price") or 0.0) * float(x.get("size") or 0.0) for x in legs) / size_total
                else:
                    entry_weighted = 0.0
                    close_weighted = 0.0

                open_time = min(int(x.get("open_time") or 0) for x in legs if x.get("open_time") is not None) if legs else None
                close_time = max(int(x.get("close_time") or 0) for x in legs if x.get("close_time") is not None) if legs else None

                normal_groups.append({
                    "symbol": base,
                    "positions": legs,
                    "size_total": size_total,
                    "notional_total": notional_total,
                    "pnl_total": pnl_total,
                    "pnl_fifo_total": pnl_fifo_total,
                    "pnl_simple_total": pnl_simple_total,
                    "pnl_price_sum": pnl_fifo_total,
                    "pnl_price_avg": (sum(float(x.get("pnl_percent") or 0.0) for x in legs) / max(len(legs), 1)),
                    "apr_avg": (sum(float(x.get("apr") or 0.0) for x in legs) / max(len(legs), 1)),
                    "fees_total": fees_total,
                    "funding_total": funding_total,
                    "realized_total": realized_total,
                    "entry_avg": entry_weighted,
                    "close_avg": close_weighted,
                    "open_date": datetime.fromtimestamp(open_time).strftime("%Y-%m-%d %H:%M") if open_time else "-",
                    "close_date": datetime.fromtimestamp(close_time).strftime("%Y-%m-%d %H:%M") if close_time else "-",
                    "type": "futures"
                })

        # ==========================================================
        # PASO 3: SWAPS DE STABLECOINS
        # ==========================================================
        stable_swap_groups = []
        for swap in [r for r in spot_trades if r["side"] == "swapstable"]:
            stable_group = {
                "symbol": _base_symbol(swap["symbol"]),
                "positions": [swap],
                "size_total": float(swap.get("size") or 0.0),
                "notional_total": float(swap.get("notional") or 0.0),
                "pnl_total": float(swap.get("realized_pnl") or 0.0),
                "fees_total": float(swap.get("fees") or 0.0),
                "funding_total": 0.0,
                "realized_total": float(swap.get("realized_pnl") or 0.0),
                "entry_avg": float(swap.get("entry_price") or 0.0),
                "close_avg": float(swap.get("close_price") or 0.0),
                "open_date": datetime.fromtimestamp(int(swap.get("open_time") or 0)).strftime("%Y-%m-%d %H:%M") if swap.get("open_time") else "-",
                "close_date": datetime.fromtimestamp(int(swap.get("close_time") or 0)).strftime("%Y-%m-%d %H:%M") if swap.get("close_time") else "-",
                "type": "stable_swap"
            }
            stable_swap_groups.append(stable_group)

        # ==========================================================
        # PASO 4: SPOT-ONLY (spotbuy/spotsell no emparejados)
        # ==========================================================
        spot_only_groups = []
        for r in spot_trades:
            if r["side"] not in ("spotbuy", "spotsell"):
                continue
            if r.get("_paired_delta_neutral"):
                continue  # ya utilizado en delta-neutral

            base = _base_symbol(r["symbol"])
            fifo = float(r.get("pnl") or 0.0)
            realized = float(r.get("realized_pnl") or 0.0)
            fees = float(r.get("fees") or 0.0)
            funding = float(r.get("funding_fee") or 0.0)
            pnl_total = fifo if abs(fifo) > 0 else (realized - fees - funding)

            grp = {
                "symbol": base,
                "positions": [r],
                "size_total": float(r.get("size") or 0.0),
                "notional_total": float(r.get("notional") or 0.0),
                "pnl_total": pnl_total,
                "fees_total": fees,
                "funding_total": funding,
                "realized_total": realized,
                "entry_avg": float(r.get("entry_price") or 0.0),
                "close_avg": float(r.get("close_price") or 0.0),
                "open_date": datetime.fromtimestamp(int(r.get("open_time") or 0)).strftime("%Y-%m-%d %H:%M") if r.get("open_time") else "-",
                "close_date": datetime.fromtimestamp(int(r.get("close_time") or 0)).strftime("%Y-%m-%d %H:%M") if r.get("close_time") else "-",
                "type": "spot"
            }
            spot_only_groups.append(grp)

        # Combinar y ordenar
        all_groups = normal_groups + delta_neutral_groups + stable_swap_groups + spot_only_groups
        all_groups.sort(key=lambda g: g["close_date"], reverse=True)

        print(f"📊 Grupos: futures={len(normal_groups)}, delta_neutral={len(delta_neutral_groups)}, stable_swaps={len(stable_swap_groups)}, spot_only={len(spot_only_groups)}")
        return jsonify({"closed_positions": all_groups})

    except Exception as e:
        print(f"❌ Error leyendo/agrupando closed_positions: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"closed_positions": []})

@app.route("/")
def index():
    return render_template(TEMPLATE_FILE)





@app.route('/api/balances')
def get_balances():
    balances = []

    # Aden
    aden_data = _send_request("GET", "/v1/positions")
    aden_account = fetch_account_aden(aden_data)
    aden_positions = fetch_positions_aden(aden_data)
    if aden_account:
        aden_account["positions"] = aden_positions
        # Asegurar claves estándar
        aden_account.setdefault("spot", 0)
        aden_account.setdefault("margin", 0)
        aden_account.setdefault("futures", aden_account.get("equity", 0))
        balances.append(aden_account)

    # Binance - combina spot + futures
    binance_data = fetch_account_binance()
    if binance_data:
        # En fetch_account_binance ya separa spot y futures internamente
        # Asumiendo que balance incluye spot y futures_wallet_balance es solo futuros
        spot_balance = binance_data.get("balance", 0) - binance_data.get("initial_margin", 0)
        binance_data.update({
            "spot": max(spot_balance, 0),  # Asegurar no negativo
            "margin": 0,
            "futures": binance_data.get("initial_margin", 0)
        })
        balances.append(binance_data)

    # Aster
    aster_data = fetch_account_aster()
    if aster_data:
        aster_data.setdefault("spot", 0)
        aster_data.setdefault("margin", 0)
        aster_data.setdefault("futures", aster_data.get("equity", 0))
        balances.append(aster_data)

    # Extended
    extended_data = fetch_account_extended()
    if extended_data:
        extended_data.setdefault("spot", 0)
        extended_data.setdefault("margin", 0)
        extended_data.setdefault("futures", extended_data.get("equity", 0))
        balances.append(extended_data)

    # BingX
    bingx_data = fetch_bingx_all_balances()
    if bingx_data:
        bingx_data.setdefault("spot", 0)
        bingx_data.setdefault("margin", 0)
        bingx_data.setdefault("futures", bingx_data.get("equity", 0))
        balances.append(bingx_data)

    # Bybit
    bybit_data = fetch_bybit_all_balances()
    if bybit_data:
        bybit_data.setdefault("spot", 0)
        bybit_data.setdefault("margin", bybit_data.get("margin_balance", 0))
        bybit_data.setdefault("futures", 0)
        balances.append(bybit_data)

    # Backpack
    backpack_data = fetch_account_backpack()
    if backpack_data:
        backpack_data.setdefault("spot", backpack_data.get("balance", 0))
        backpack_data.setdefault("margin", 0)
        backpack_data.setdefault("futures", 0)
        balances.append(backpack_data)

    # KuCoin
    try:
        ku = fetch_kucoin_all_balances() or {}
        if ku:
            # El adapter ya entrega la forma final (floats), no hay nested dicts
            ku_obj = {
                "exchange": "kucoin",
                "equity": float(ku.get("equity", 0.0)),
                "balance": float(ku.get("balance", 0.0)),
                "unrealized_pnl": float(ku.get("unrealized_pnl", 0.0)),
                "spot": float(ku.get("spot", 0.0)),
                "margin": float(ku.get("margin", 0.0)),
                "futures": float(ku.get("futures", 0.0)),
            }
            balances.append(ku_obj)
    except Exception as e:
        print(f"❌ KuCoin balances error in route: {e}")


        
     #Gate.io   
    gate_data = fetch_gate_all_balances(settles=("usdt",))
    if gate_data:
        gate_spot = sum(bal['available'] + bal['locked'] for bal in gate_data.get('spot', []))
        gate_futures = sum(fut['balance'] for fut in gate_data.get('futures', []))
        
        gate_data.update({
            "exchange": "gate",
            "equity": gate_spot + gate_futures,
            "balance": gate_spot + gate_futures,
            "unrealized_pnl": sum(fut['unrealized_pnl'] for fut in gate_data.get('futures', [])),
            "spot": gate_spot,
            "margin": 0,
            "futures": gate_futures,
        })
        balances.append(gate_data)
        print(f"✅ Gate.io balances: spot={gate_spot:.2f}, futures={gate_futures:.2f}")
        
        
    # MEXC
    try:
        mexc_data = fetch_mexc_all_balances()  # ← SIN settles, sin settle
        if mexc_data: balances.append(mexc_data)
        print("✅ MEXC balances OK")
    except Exception as e:
        print(f"❌ MEXC balances error: {e}")
        
    # Bitget
    try:
        bitget_data = fetch_bitget_all_balances()
        if bitget_data: 
            balances.append(bitget_data)
        print("✅ Bitget balances OK")
    except Exception as e:
        print(f"❌ Bitget balances error: {e}")
        
    # OKX
    try:
        okx_data = fetch_okx_all_balances()
        if okx_data: 
            balances.append(okx_data)
        print("✅ OKX balances OK")
    except Exception as e:
        print(f"❌ OKX balances error: {e}")
        
    # Paradex
    try:
        paradex_data = fetch_paradex_all_balances()
        if paradex_data: 
            balances.append(paradex_data)
        print("✅ Paradex balances OK")
    except Exception as e:
        print(f"❌ Paradex balances error: {e}")
        
    
        
      # Hyperliquid
    try:
        hyper_data = fetch_hyperliquid_all_balances()
        if hyper_data:
            balances.append(hyper_data)
        print("✅ Hyperliquid balances OK")
    except Exception as e:
        print(f"❌ Hyperliquid balances error: {e}")  
     
        # Whitebit
    try:
        whitebit_data = fetch_whitebit_all_balances()
        if whitebit_data:
            balances.append(whitebit_data)
        print("✅ whitebit balances OK")
    except Exception as e:
        print(f"❌ whitebit balances error: {e}")    
        
        # XT
    try:
        xt_bal = fetch_xt_all_balances()
        if xt_bal:
            balances.append(xt_bal)
        print("✅ XT balances OK")
    except Exception as e:
        print(f"❌ XT balances error: {e}")
        
        

    # Totales
    total_equity = sum(b.get("equity", 0) for b in balances)
    total_balance = sum(b.get("balance", 0) for b in balances)
    total_unreal = sum(b.get("unrealized_pnl", 0) for b in balances)

    return jsonify({
        "totals": {
            "equity": total_equity,
            "balance": total_balance,
            "unrealized_pnl": total_unreal,
        },
        "exchanges": balances
    })


@app.route('/api/positions')
def get_positions():
    print("📡 Solicitando datos de posiciones.")
    all_positions = []

    # Backpack
    try:
        all_positions.extend(fetch_positions_backpack())
    except Exception as e:
        print(f"❌ Backpack positions error: {e}")

    # Aster
    try:
        all_positions.extend(fetch_aster_open_positions())
    except Exception as e:
        print(f"❌ Aster positions error: {e}")

    # Binance
    try:
        all_positions.extend(fetch_positions_binance_enriched())
    except Exception as e:
        print(f"❌ Binance positions error: {e}")

    # BingX
    try:
        all_positions.extend(fetch_bingx_open_positions())
    except Exception as e:
        print(f"❌ BingX positions error: {e}")

    # Aden / Orderly
    try:
        aden_data = _send_request("GET", "/v1/positions")
        all_positions.extend(fetch_positions_aden(aden_data))
    except Exception as e:
        print(f"❌ Aden positions error: {e}")

    # KuCoin
    try:
        all_positions.extend(fetch_kucoin_open_positions())
    except Exception as e:
        print(f"❌ KuCoin positions error: {e}")
    # Extended
    try:
        all_positions.extend(fetch_open_extended_positions())
    except Exception as e:
        print(f"❌ KuCoin positions error: {e}")

    # Gate.io
    try:
        all_positions.extend(fetch_gate_open_positions(settle="usdt"))
    except Exception as e:
        print(f"❌ Gate.io positions error: {e}")

    # MEXC  ← sin settle, gracias
    try:
        mexc_positions = fetch_mexc_open_positions()
        all_positions.extend(mexc_positions)
        print(f"✅ MEXC posiciones: {len(mexc_positions)}")
    except Exception as e:
        print(f"❌ MEXC positions error: {e}")
    # Bitget   
    try:
        bitget_positions = fetch_bitget_open_positions()
        all_positions.extend(bitget_positions)
        print(f"✅ Bitget posiciones: {len(bitget_positions)}")
    except Exception as e:
        print(f"❌ Bitget positions error: {e}")
        
    # OKX
    try:
        okx_positions = fetch_okx_open_positions()
        all_positions.extend(okx_positions)
        print(f"✅ OKX posiciones: {len(okx_positions)}")
    except Exception as e:
        print(f"❌ OKX positions error: {e}")
     
    # Paradex
    try:
        paradex_positions = fetch_paradex_open_positions()
        all_positions.extend(paradex_positions)
        print(f"✅ Paradex posiciones: {len(paradex_positions)}")
    except Exception as e:
        print(f"❌ Paradex positions error: {e}")
        
    # Hyperliquid
    try:
        hyper_positions = fetch_hyperliquid_open_positions()
        all_positions.extend(hyper_positions)
        print(f"✅ Hyperliquid posiciones: {len(hyper_positions)}")
    except Exception as e:
        print(f"❌ Hyperliquid positions error: {e}")
     #Whitebit   
    try:
        wb_pos = fetch_whitebit_open_positions()
        all_positions.extend(wb_pos)
        print(f"✅ WhiteBIT posiciones: {len(wb_pos)}")
    except Exception as e:
        print(f"❌ WhiteBIT positions error: {e}")      
     
        # XT
    try:
        xt_pos = fetch_xt_open_positions()
        all_positions.extend(xt_pos)
        print(f"✅ XT posiciones: {len(xt_pos)}")
    except Exception as e:
        print(f"❌ XT positions error: {e}")
        
        # Bybit
    try:
        bybit_pos = fetch_bybit_open_positions()
        all_positions.extend(bybit_pos)
        print(f"✅ Bybit posiciones: {len(bybit_pos)}")
    except Exception as e:
        print(f"❌ Bybit positions error: {e}")
        
        
    # Actualiza cache de exchanges activos con posiciones abiertas
    try:
        active = set(p.get("exchange","").lower() for p in all_positions if p.get("exchange"))
        _update_open_pos_cache(active)
    except Exception:
        pass
    

    print(f"📊 Total posiciones: {len(all_positions)}")
    
    
    return jsonify({"positions": all_positions})




# =====================================================
# 🚀 BLOQUE FINAL LIMPIO — EJECUCIÓN PRINCIPAL
# =====================================================

SYNC_FUNCTIONS = {
    "backpack": lambda days=60: save_backpack_closed_positions("portfolio.db"),
    "aden": lambda days=60: save_aden_closed_positions("portfolio.db", debug=False),
    "bingx": lambda days=30: save_bingx_closed_positions("portfolio.db", symbols=None, days=days, include_funding=True, debug=True),
    "aster": lambda days=50: save_aster_closed_positions("portfolio.db", days=days, debug=False),
    "binance": lambda days=55: save_binance_closed_positions("portfolio.db", days=days, debug=False),
    "extended": lambda days=60: save_extended_closed_positions("portfolio.db", debug=False),
    "kucoin": lambda days=60: save_kucoin_closed_positions("portfolio.db", debug=False),
    "gate": lambda days=60: save_gate_closed_positions("portfolio.db"),
    "mexc": lambda days=10: save_mexc_closed_positions("portfolio.db", days=days, debug=PRINT_CLOSED_DEBUG),
    "bitget": lambda days=60: save_bitget_closed_positions("portfolio.db", days=days, debug=False),
    "okx": lambda days=60: save_okx_closed_positions("portfolio.db", days=days, debug=False),
    "paradex": lambda days=60: save_paradex_closed_positions("portfolio.db"),
    "hyperliquid": lambda days=60: save_hyperliquid_closed_positions("portfolio.db", days=days, debug=False),
    "whitebit": lambda days=50: save_whitebit_closed_positions("portfolio.db", days=days, debug=False),
    "xt": lambda days=60: save_xt_closed_positions("portfolio.db", days=days),
    "bybit": lambda days=60: save_bybit_closed_positions("portfolio.db", days=days, debug=False),
    "lbank": lambda days=60: save_lbank_closed_positions("portfolio.db", days=days),
}

# Diccionario de funciones para obtener posiciones abiertas
POSITIONS_FUNCTIONS = {
    "backpack": lambda: fetch_positions_backpack(),
    "aster": lambda: fetch_aster_open_positions(),
    "binance": lambda: fetch_positions_binance_enriched(),
    "bingx": lambda: fetch_bingx_open_positions(),
    "extended": lambda: fetch_open_extended_positions(),
    "aden": lambda: fetch_positions_aden(_send_request("GET", "/v1/positions")),
    "kucoin": lambda: fetch_kucoin_open_positions(),
    "gate": lambda: fetch_gate_open_positions(settle="usdt"),
    "mexc": lambda: fetch_mexc_open_positions(),
    "bitget": lambda: fetch_bitget_open_positions(),
    "okx": lambda: fetch_okx_open_positions(),
    "paradex": lambda: fetch_paradex_open_positions(),
    "hyperliquid": lambda: fetch_hyperliquid_open_positions(),
    "whitebit": lambda: fetch_whitebit_open_positions(),
    "xt": lambda: fetch_xt_open_positions(),
    "bybit": lambda: fetch_bybit_open_positions(),
}

# borrar despues
def main():
    print("🚀 Iniciando actualización de portfolio.")
    # Inicializar cache universal
    init_universal_cache_db()

    # ✅ Actualizar cache para TODOS los exchanges que tengan posiciones abiertas
    print("🔄 Actualizando cache universal para todos los exchanges...")
    
    for ex_name, fetch_positions_func in POSITIONS_FUNCTIONS.items():
        if should_sync(ex_name):
            try:
                print(f"   📦 {ex_name.capitalize()}: obteniendo posiciones...")
                positions = fetch_positions_func()
                update_cache_from_positions(ex_name, positions)
                print(f"   ✅ {ex_name.capitalize()}: {len(positions)} posiciones en cache")
            except Exception as e:
                print(f"   ⚠️ {ex_name.capitalize()}: error - {e}")
    
    # Mostrar estadísticas del cache
    stats = get_cache_stats()
    print(f"📊 Cache universal: {stats['total_entries']} símbolos (TTL: {UNIVERSAL_CACHE_TTL_DAYS} días)")
    # =====================================================
    # 🧠 SMART SYNC - Alternativa 3
    #   1) Forzar una vez Bitget (full sync corto)
    #   2) Luego bucle normal con force_full_sync=False
    # =====================================================
    if SMART_SYNC_ENABLED:
        print("🧠 Modo Smart Sync activado")

        # (1) Fuerza un full sync “de arranque” SOLO para Bitget
        try:
            saved_bitget = smart_sync_closed_positions(
                "bitget",
                force_full_sync=True,
                debug=PRINT_CLOSED_SYNC
            )
            print(f"⚡ Bitget full sync inicial: {saved_bitget} posiciones guardadas")
        except Exception as e:
            print(f"❌ Error en full sync inicial de Bitget: {e}")

        # (2) Bucle normal para el resto de exchanges
        total_saved = 0
        for exchange_name in SYNC_FUNCTIONS.keys():
            try:
                saved = smart_sync_closed_positions(
                    exchange_name,
                    force_full_sync=False,  # ahora modo normal
                    debug=PRINT_CLOSED_SYNC
                )
                total_saved += saved
            except Exception as e:
                print(f"❌ Error en sync de {exchange_name}: {e}")

        print(f"✅ Smart Sync completado: {total_saved} posiciones totales guardadas")

    else:
        # ===========================
        # 📋 Modo Legacy (como lo tenías)
        # ===========================
        print("📋 Modo Legacy Sync")

        # Función especial para BingX que necesita preparación (igual que tu código)
        def sync_bingx():
            try:
                debug_cache_status()
                force_cache_update()
                debug_cache_status()
            except Exception:
                pass

            save_bingx_closed_positions(
                db_path="portfolio.db",
                symbols=None,
                days=30,
                include_funding=True,
                debug=True
            )

        # Ejecutar sincronización para cada exchange configurado
        for exchange_name, sync_function in SYNC_FUNCTIONS.items():
            if should_sync(exchange_name):
                print(f"⏳ Sincronizando fills cerrados de {exchange_name.capitalize()}.")
                try:
                    if exchange_name == "bingx":
                        sync_bingx()
                    else:
                        sync_function()
                    print(f"✅ Posiciones cerradas de {exchange_name.capitalize()} actualizadas correctamente.")
                except Exception as e:
                    print(f"❌ Error en {exchange_name}: {e}")
            else:
                print(f"⏭️  Saltando {exchange_name.capitalize()}")

    # =====================================================
    # 📦 SPOT TRADES SYNC (opcional; déjalo igual que lo tenías)
    # =====================================================
    spot_sync_functions = {
        "gate":   lambda: save_gate_spot_positions("portfolio.db", days_back=40),
        # si ya tienes importado save_bitget_spot_positions, puedes activarlo:
        # "bitget": lambda: save_bitget_spot_positions("portfolio.db", days_back=40),
    }

    for exchange_name, sync_function in spot_sync_functions.items():
        if should_sync_spot(exchange_name):
            print(f"⏳ Sincronizando SPOT TRADES de {exchange_name.capitalize()}.")
            try:
                sync_function()
                print(f"✅ Trades de spot de {exchange_name.capitalize()} actualizados correctamente.")
            except Exception as e:
                print(f"❌ Error en spot trades de {exchange_name}: {e}")

    print("🧩 Sincronización completada.")


# def main():
#     print("🚀 Iniciando actualización de portfolio...")
#     # Inicializar cache universal
#     init_universal_cache_db()
    
#     # Función para actualizar cache de un exchange
#     def update_exchange_cache(exchange_name, fetch_positions_func):
#         try:
#             print(f"🔄 Actualizando cache para {exchange_name}...")
#             positions = fetch_positions_func()
#             update_cache_from_positions(exchange_name, positions)
#         except Exception as e:
#             print(f"⚠️ Error actualizando cache de {exchange_name}: {e}")
    
#     # Actualizar cache de exchanges con posiciones abiertas
#     # ✅ Actualizar cache para TODOS los exchanges que tengan posiciones abiertas
#     # ⚡ Haz un full sync de Bitget una vez (ej.: 7 días) para inicializar su histórico
#     smart_sync_closed_positions("bitget", force_full_sync=True, debug=PRINT_CLOSED_SYNC)

#     for ex_name, fetch_positions_func in POSITIONS_FUNCTIONS.items():
#         if should_sync(ex_name):
#             update_exchange_cache(ex_name, fetch_positions_func)
    
    
#     # Mostrar estadísticas del cache
#     stats = get_cache_stats()
#     print(f"📊 Cache universal: {stats['total_entries']} símbolos (TTL: {UNIVERSAL_CACHE_TTL_DAYS} días)")
    
#     # =====================================================
#     # 🧠 SMART SYNC - Sincronización inteligente
#     # =====================================================
    
#     if SMART_SYNC_ENABLED:
#         print("🧠 Modo Smart Sync activado")
        
#         total_saved = 0
#         for exchange_name in SYNC_FUNCTIONS.keys():
#             saved = smart_sync_closed_positions(
#                 exchange_name, 
#                 force_full_sync=False,  # Cambiar a True para forzar sync completo
#                 debug=PRINT_CLOSED_SYNC
#             )
#             total_saved += saved
        
#         print(f"✅ Smart Sync completado: {total_saved} posiciones totales guardadas")
    
#     else:
#         # Modo legacy (sync tradicional)
#         print("📋 Modo Legacy Sync")
        
#         # Función especial para BingX que necesita preparación
#         def sync_bingx():
#             try:
#                 debug_cache_status()
#                 force_cache_update()
#                 debug_cache_status()
#             except Exception:
#                 pass
        
#             save_bingx_closed_positions(
#                 db_path="portfolio.db",
#                 symbols=None,
#                 days=30,
#                 include_funding=True,
#                 debug=True
#             )
        
#         # Ejecutar sincronización para cada exchange configurado
#         for exchange_name, sync_function in SYNC_FUNCTIONS.items():
#             if should_sync(exchange_name):
#                 print(f"⏳ Sincronizando fills cerrados de {exchange_name.capitalize()}...")
#                 try:
#                     # Para BingX usa la función especial, para el resto usa la del dict
#                     if exchange_name == "bingx":
#                         sync_bingx()
#                     else:
#                         sync_function()
#                     print(f"✅ Posiciones cerradas de {exchange_name.capitalize()} actualizadas correctamente.")
#                 except Exception as e:
#                     print(f"❌ Error en {exchange_name}: {e}")
#             else:
#                 print(f"⏭️  Saltando {exchange_name.capitalize()}")
    
#     # =====================================================
#     # 📦 SPOT TRADES SYNC
#     # =====================================================
    
#     spot_sync_functions = {
#         "gate": lambda: save_gate_spot_positions("portfolio.db", days_back=40),
#         "bitget": lambda: save_bitget_spot_positions("portfolio.db", days_back=40),
#     }
    
#     for exchange_name, sync_function in spot_sync_functions.items():
#         if should_sync_spot(exchange_name):
#             print(f"⏳ Sincronizando SPOT TRADES de {exchange_name.capitalize()}...")
#             try:
#                 sync_function()
#                 print(f"✅ Trades de spot de {exchange_name.capitalize()} actualizados correctamente.")
#             except Exception as e:
#                 print(f"❌ Error en spot trades de {exchange_name}: {e}")
 
#     print("🧩 Sincronización completada.")


#====================================================
# codigo para boton de guardar automaticamente

def save_closed_positions_generic(exchange: str, days: int = 30, db_path: str = "portfolio.db", debug: bool = False):
    """
    Dispara el guardado de posiciones cerradas para un exchange concreto usando la
    ventana 'days' y devuelve métricas estándar.
    - Usa SYNC_FUNCTIONS[exchange](days=days)
    - Actualiza el timestamp de sync y refresca la caché de posiciones si es posible.
    """
    exchange = (exchange or "").lower()
    if exchange not in SYNC_FUNCTIONS:
        raise ValueError(f"No hay función de sync definida para '{exchange}'")

    # 1) Ejecutar la función del adapter
    res = SYNC_FUNCTIONS[exchange](days=days)
    if isinstance(res, tuple):
        # admite (saved, updated) o similar
        saved = int(res[0] or 0)
        updated = int(res[1] or 0) if len(res) > 1 else 0
        skipped = 0
    elif isinstance(res, int):
        saved = int(res)
        updated = 0
        skipped = 0
    elif isinstance(res, dict):
        saved  = int(res.get("inserted") or res.get("saved") or 0)
        updated = int(res.get("updated") or 0)
        skipped = int(res.get("skipped") or 0)
    else:
        saved = updated = skipped = 0

    # 2) Actualizar timestamps y caché universal
    try:
        update_sync_timestamp(exchange)
    except Exception:
        pass

    try:
        fetch_positions = POSITIONS_FUNCTIONS.get(exchange)
        if fetch_positions:
            current_positions = fetch_positions()
            update_cache_from_positions(exchange, current_positions)
    except Exception:
        pass

    return {"inserted": saved, "updated": updated, "skipped": skipped, "days": days, "exchange": exchange}
from flask import Blueprint, request, jsonify

def _resolve_direct_closed_saver(exchange: str):
    """
    Busca dinámicamente una función del adapter que GUARDE posiciones cerradas
    sin depender del cache. Intenta varios nombres típicos.
    """
    exchange = (exchange or "").lower()
    module_name_candidates = [exchange]  # p.ej. "bitget.py" -> import bitget
    # Si usas prefijos tipo "adapters.bitget", añade aquí:
    module_name_candidates += [f"adapters.{exchange}"]

    func_name_candidates = [
        f"save_{exchange}_closed_positions",
        f"save_closed_positions_{exchange}",
        f"{exchange}_save_closed_positions",
        "save_closed_positions",  # por si el adapter exporta genérico
    ]

    last_err = None
    for modname in module_name_candidates:
        try:
            mod = __import__(modname, fromlist=["*"])
        except Exception as e:
            last_err = e
            continue
        for fn in func_name_candidates:
            saver = getattr(mod, fn, None)
            if callable(saver):
                return saver
    # Si no se encontró nada, caemos a SYNC_FUNCTIONS para no romper
    if exchange in SYNC_FUNCTIONS:
        return SYNC_FUNCTIONS[exchange]
    if last_err:
        raise last_err
    raise RuntimeError(f"No encuentro función de guardado para '{exchange}'")

def force_load_closed_positions(exchange: str, days: int = 30, db_path: str = "portfolio.db", **kw):
    """
    Fuerza la descarga+guardado de posiciones cerradas ignorando cache de símbolos.
    Llama directamente a la función del adapter.
    """
    saver = _resolve_direct_closed_saver(exchange)
    # La mayoría de savers aceptan (db_path, days, debug). Pasamos kwargs defensivos.
    res = None
    try:
        res = saver(db_path=db_path, days=days, **kw)
    except TypeError:
        # Adapters antiguos: admitir (days) o (start_ms, end_ms)
        from time import time as _t
        end_ms = int(_t() * 1000)
        start_ms = end_ms - int(days) * 24 * 60 * 60 * 1000
        try:
            res = saver(start_ms=start_ms, end_ms=end_ms, db_path=db_path, **kw)
        except TypeError:
            # último intento: solo days
            res = saver(days=days)

    # Normaliza métricas
    if isinstance(res, tuple):
        saved = int(res[0] or 0)
        updated = int(res[1] or 0) if len(res) > 1 else 0
        skipped = 0
    elif isinstance(res, int):
        saved, updated, skipped = int(res), 0, 0
    elif isinstance(res, dict):
        saved  = int(res.get("inserted") or res.get("saved") or 0)
        updated = int(res.get("updated") or 0)
        skipped = int(res.get("skipped") or 0)
    else:
        saved = updated = skipped = 0

    return {"inserted": saved, "updated": updated, "skipped": skipped}

api = Blueprint("api", __name__)

@app.post("/api/closed/load")
def api_closed_load():
    """
    Fuerza la carga de posiciones cerradas (sin cache) para un exchange y ventana de días.
    Body: { "exchange": "bitget", "days": 30 }
    """
    try:
        payload = request.get_json(force=True) or {}
        exchange = (payload.get("exchange") or "").strip().lower()
        days = int(payload.get("days") or 30)
        days = max(1, min(60, days))

        if not exchange:
            return jsonify({"error": "Falta 'exchange'"}), 400

        result = force_load_closed_positions(exchange=exchange, days=days, db_path=DB_PATH, debug=True)
        # (Opcional) refrescar cache de abiertas para la UI, pero NO es requisito para el guardado:
        try:
            fetch_positions = POSITIONS_FUNCTIONS.get(exchange)
            if fetch_positions:
                current_positions = fetch_positions()
                update_cache_from_positions(exchange, current_positions)
        except Exception:
            pass

        return jsonify({"exchange": exchange, "days": days, **result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# fin de la funcion para guardar automaticamente
#=====================================================
@app.post("/api/closed/sync")
def api_closed_sync():
    """
    Dispara la carga 'normal' (smart sync con caché) de posiciones cerradas.
    Body (opcional):
      { "exchange": "bitget" }  -> solo ese exchange
      { "force_full_sync": true } -> fuerza full sync (ignora caché)
    Si no se especifica 'exchange', itera por TODOS los exchanges de SYNC_FUNCTIONS.
    """
    try:
        payload = request.get_json(silent=True) or {}
        exchange = (payload.get("exchange") or "").strip().lower()
        force_full = bool(payload.get("force_full_sync") or False)

        targets = [exchange] if exchange else list(SYNC_FUNCTIONS.keys())

        results = {}
        total_saved = 0

        # Ejecuta smart sync por exchange
        for ex in targets:
            try:
                saved = smart_sync_closed_positions(
                    ex,
                    force_full_sync=force_full,
                    debug=PRINT_CLOSED_SYNC
                )
                results[ex] = {"inserted": int(saved)}
                total_saved += int(saved)

                # Refrescar caché de abiertas para la UI
                try:
                    fetch_positions = POSITIONS_FUNCTIONS.get(ex)
                    if fetch_positions:
                        current_positions = fetch_positions()
                        update_cache_from_positions(ex, current_positions)
                except Exception:
                    pass

            except Exception as e:
                results[ex] = {"error": str(e)}

        # Respuesta homogénea
        return jsonify({
            "mode": "smart_sync",
            "force_full_sync": force_full,
            "targets": targets,
            "total_inserted": total_saved,
            "by_exchange": results
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

 #=====================================================
# 🏁 EJECUCIÓN PRINCIPAL (Init DB + Main + Flask)
# =====================================================
from flask import request

def _parse_ts_seconds(x):
    """
    Acepta: int (s o ms) o ISO8601; devuelve epoch segundos.
    """
    try:
        n = int(float(x))
        return n if n >= 1_000_000_000 else n  # si te enviaran s, mantenlo en s
    except Exception:
        pass
    try:
        # ISO → ms → s
        ms = datetime.fromisoformat(str(x).replace("Z","")).timestamp() * 1000.0
        return int(ms // 1000)
    except Exception:
        return 0

@app.route("/api/manual/closed", methods=["POST"])
def api_manual_closed():
    """
    Inserta una posición cerrada manual.
    Campos mínimos: exchange, symbol, side, size, entry_price, close_price, close_time.
    Opcionales: open_time, funding_total, fee_total, realized_pnl, initial_margin, notional, leverage, liquidation_price.
    """
    try:
        data = request.get_json(force=True) or {}
        # Normalizar y sanear
        payload = {
            "exchange": (data.get("exchange") or "").strip().lower(),
            "symbol": (data.get("symbol") or "").strip().upper(),
            "side": (data.get("side") or "").strip().lower(),
            "size": float(data.get("size") or 0),
            "entry_price": float(data.get("entry_price") or 0),
            "close_price": float(data.get("close_price") or 0),
            "open_time": int(_parse_ts_seconds(data.get("open_time") or 0)),
            "close_time": int(_parse_ts_seconds(data.get("close_time") or 0)),
            "funding_total": float(data.get("funding_total") or 0),
            "fee_total": float(data.get("fee_total") or 0),
            "realized_pnl": data.get("realized_pnl"),     # puede ser None → se compone
            "initial_margin": data.get("initial_margin"),
            "notional": data.get("notional"),
            "leverage": data.get("leverage"),
            "liquidation_price": data.get("liquidation_price"),
        }

        # Validación mínima
        req_fields = ["exchange","symbol","side","size","entry_price","close_price","close_time"]
        for f in req_fields:
            if not payload.get(f) and payload.get(f) != 0:
                return jsonify({"error": f"Campo requerido: {f}"}), 400

        # Llama a la función estándar de guardado (se encarga de APR/ROI/etc.)
        from db_manager import save_closed_position
        save_closed_position(payload)

        # Recupera el id insertado por combinación (exchange, symbol, close_time) más reciente
        import sqlite3
        conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
        cur.execute("""
            SELECT id FROM closed_positions
            WHERE exchange=? AND symbol=? AND close_time=?
            ORDER BY id DESC LIMIT 1
        """, (payload["exchange"], payload["symbol"], payload["close_time"]))
        row = cur.fetchone()
        conn.close()
        new_id = row[0] if row else None

        return jsonify({"ok": True, "id": new_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🧱 Inicializando base de datos...")
    init_db()
    init_funding_db()
    _init_funding_sync_state()
    
    from db_manager import migrate_spot_support
    migrate_spot_support()
    
    # ✅ NUEVO: Inicializar tabla de sync timestamps
    from universal_cache import init_sync_timestamps_table
    init_sync_timestamps_table()
    # ✅ AGREGAR AQUÍ:
    from universal_cache import migrate_universal_cache
    migrate_universal_cache()  # ← Migra la columna last_seen

    if SYNC_FUNDING_ON_START:
        force = None
        try:
            if isinstance(FUNDING_DEFAULT_DAYS, int) and FUNDING_DEFAULT_DAYS > 0:
                force = int(FUNDING_DEFAULT_DAYS)
        except Exception:
            force = None

        print("🔄 Sincronizando funding al arranque..."
              + (f" (forzado {force} días)" if force else " (incremental)"))

        sync_all_funding(force_days=force, verbose=True)

    print("✅ Base de datos lista. Ejecutando sincronización inicial...")
    main()

    print("🌐 Lanzando servidor Flask...")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
