# db_manager.py  — versión con migración y nuevas columnas
import sqlite3
from collections import defaultdict
import statistics
import math
import time as _t
import pandas as pd

DB_PATH = "portfolio.db"

NEW_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS closed_positions_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange TEXT,
    symbol TEXT,
    side TEXT,
    size REAL,
    entry_price REAL,
    close_price REAL,
    open_time INTEGER,
    close_time INTEGER,
    pnl REAL,                 -- precio puro
    realized_pnl REAL,        -- neto (incluye fees + funding)
    funding_total REAL,
    fee_total REAL,
    pnl_percent REAL,         -- realized_pnl / notional * 100
    apr REAL,   
    initial_margin REAL,               -- anualizado con (close-open)
    notional REAL,
    leverage REAL,
    liquidation_price REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

TARGET_COL_ORDER = [
    "id","exchange","symbol","side","size","entry_price","close_price",
    "open_time","close_time","pnl","realized_pnl","funding_total","fee_total",
    "pnl_percent","apr", "initial_margin", "notional","leverage","liquidation_price","created_at"
]

def _table_cols(conn, table):
    cur = conn.execute(f"PRAGMA table_info({table})")
    
def _rebuild_if_needed(conn):
    cur = conn.cursor()
    cols = _table_cols(conn, "closed_positions")

    # Si no existe, crea directamente con el nuevo esquema
    if not cols:
        cur.executescript(NEW_SCHEMA_SQL.replace("_new", ""))
        conn.commit()
        return

    # ¿Faltan columnas nuevas?
    wanted = {"pnl", "pnl_percent", "apr", "initial_margin"}
    if wanted.issubset(set(cols)):
        # Nada que migrar
        return

    # 1) Crear tabla nueva con esquema correcto
    cur.executescript(NEW_SCHEMA_SQL)

    # 2) Preparar lista de columnas destino (EXCLUYENDO id autoincrement)
    dest_cols = [
        "exchange","symbol","side","size","entry_price","close_price",
        "open_time","close_time","pnl","realized_pnl","funding_total","fee_total",
        "pnl_percent","apr","initial_margin","notional","leverage","liquidation_price","created_at"
    ]  # 19 columnas

    existing = set(cols)

    # 3) Construir SELECT con el mismo número de expresiones que dest_cols
    select_exprs = []
    for c in dest_cols:
        if c in existing:
            # La columna existe en la vieja: la copiamos tal cual
            select_exprs.append(c)
        elif c == "created_at":
            # Si la vieja no tenía created_at, lo rellenamos con ahora
            select_exprs.append("CURRENT_TIMESTAMP AS created_at")
        else:
            # Columna nueva que no existía: NULL
            select_exprs.append(f"NULL AS {c}")

    # 4) Ejecutar INSERT con columnas y SELECT emparejados 1:1
    insert_cols_sql = ", ".join(dest_cols)
    select_cols_sql = ", ".join(select_exprs)

    cur.execute(f"""
        INSERT INTO closed_positions_new ({insert_cols_sql})
        SELECT {select_cols_sql}
        FROM closed_positions
    """)

    # 5) Sustituir tablas
    cur.execute("DROP TABLE closed_positions")
    cur.execute("ALTER TABLE closed_positions_new RENAME TO closed_positions")

    conn.commit()    
    


#========codigos de chatGPT, borrar si no funciona
def _has_col(conn, table, col):
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())

def _add_col_if_missing(conn, table, col, sql_type):
    if not _has_col(conn, table, col):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {sql_type}")
        
def _has_col(conn, table, col):
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())

def _add_col_if_missing(conn, table, col, sql_type):
    if not _has_col(conn, table, col):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {sql_type}")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS closed_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        exchange TEXT, symbol TEXT, side TEXT, size REAL,
        entry_price REAL, close_price REAL,
        open_time INTEGER, close_time INTEGER,
        pnl REAL, realized_pnl REAL, funding_total REAL, fee_total REAL,
        pnl_percent REAL, apr REAL,
        initial_margin REAL,                 -- 👈 aquí
        notional REAL, leverage REAL, liquidation_price REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    try:
        cur.execute("ALTER TABLE closed_positions ADD COLUMN initial_margin REAL")
    except sqlite3.OperationalError:
        # La columna ya existe, ignorar el error
        pass
    conn.commit()
    # por si la tabla ya existía sin initial_margin en instalaciones viejas
    _add_col_if_missing(conn, "closed_positions", "initial_margin", "REAL")
    conn.commit()
    conn.close()

#======== FIN DE LOS codigos de chatGPT, borrar si no funciona

        


# ========= Helpers de cálculo =========

#=============CODIGO 1 MODIFICACION DEL CODIGO 2 POR CHATGPT

def _safe(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _price_pnl(side, entry, close, size):
    if side and side.lower() == "short":
        return (entry - close) * size
    return (close - entry) * size

def _pnl_percent(realized_pnl, notional):
    if notional and abs(notional) > 0:
        return 100.0 * (realized_pnl / notional)
    return 0.0

def _apr(realized_pnl, notional, open_s, close_s):
    try:
        days = max((_safe(close_s) - _safe(open_s)) / 86400.0, 1e-9)
        if days <= 0 or notional == 0:
            return 0.0
        return 100.0 * (realized_pnl / notional) * (365.0 / days)
    except Exception:
        return 0.0
    
def save_closed_position(position: dict):
    import sqlite3
    import traceback
    
    print(f"🔍 [DEBUG save_closed_position] Iniciando guardado para: {position.get('exchange')} {position.get('symbol')}")
    
    def _f(x, d=0.0):
        try: 
            return float(x)
        except: 
            return d

    def _positive(x):
        return x is not None and x > 0

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
    liq_price     = _f(position.get("liquidation_price"))
    
    # 1) PnL de precio: usa el que mande la API, si no, recalcúlalo
    pnl_price_api = position.get("pnl")
    pnl_price = _f(pnl_price_api) if pnl_price_api is not None else _price_pnl(side, entry, close, _f(size, 0.0))
    
        # 2) Reconstrucción de size si viene escalada o 0 y hay PnL de precio

    
    size_locked = bool(position.get("_lock_size"))
    diff = abs(close - entry)
    if not size_locked:
        if (size <= 0 or (diff > 0 and abs(pnl_price) > 0 and abs(size - abs(pnl_price)/diff) / max(1.0, abs(size)) > 0.05)):
            if diff > 0 and abs(pnl_price) > 0:
                size = abs(pnl_price) / diff
    
    # 3) Notional SIEMPRE a precio de entrada
    entry_notional = abs(size) * entry
    notional_api   = _f(position.get("notional", 0.0))
    notional       = entry_notional if entry_notional > 0 else notional_api
    
    # 4) Realized neto: si no viene, compónlo (precio + funding + fees)
    realized_api = position.get("realized_pnl")
    realized = _f(realized_api) if realized_api is not None else (pnl_price + funding_total + fee_total)
    
    # 5) Resolver leverage (si hay). Si no, usar default por exchange.
    #    Puedes mover este mapping a config/toggles.py si prefieres.
    DEFAULT_LEVERAGE = {"gate": 5}
    lev_raw = position.get("leverage")
    leverage = _f(lev_raw) if lev_raw is not None else 0.0
    if not _positive(leverage):
        # Si la API dio initial_margin válido, deduce leverage = notional / margin
        im_raw = position.get("initial_margin")
        im_val = _f(im_raw) if im_raw is not None else 0.0
        if _positive(im_val) and (notional > 0 or entry_notional > 0):
            leverage = (notional if notional > 0 else entry_notional) / im_val
        elif exchange in DEFAULT_LEVERAGE and _positive(DEFAULT_LEVERAGE[exchange]):
            leverage = DEFAULT_LEVERAGE[exchange]  # ← Gate: 5 por defecto
        else:
            leverage = 0.0
    
    # 6) Margen inicial
    initial_margin = position.get("initial_margin")
    initial_margin = _f(initial_margin) if initial_margin is not None else 0.0
    if not _positive(initial_margin):
        if _positive(leverage):
            initial_margin = (notional if notional > 0 else entry_notional) / leverage
        else:
            # Último recurso: usa entry_notional (ROI más conservador)
            initial_margin = entry_notional
    
    # 7) Métricas: SIEMPRE desde realized_pnl
    base_capital = initial_margin if _positive(initial_margin) else (notional if _positive(notional) else 0.0)
    pnl_percent = (realized / base_capital) * 100.0 if _positive(base_capital) else 0.0
    
    days = max((close_s - open_s) / 86400.0, 1e-9) if (open_s and close_s) else 0.0
    apr = pnl_percent * (365.0 / days) if days > 0 else 0.0


    # INSERT con columnas y placeholders 1:1
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

    # DEBUG DETALLADO
    print(f"📋 [DEBUG] Datos a insertar:")
    print(f"  exchange: {exchange}")
    print(f"  symbol: {symbol}")
    print(f"  side: {side}")
    print(f"  size: {size}")
    print(f"  entry_price: {entry}")
    print(f"  close_price: {close}")
    print(f"  open_time: {open_s} ({pd.to_datetime(open_s, unit='s') if open_s else 'N/A'})")
    print(f"  close_time: {close_s} ({pd.to_datetime(close_s, unit='s') if close_s else 'N/A'})")
    print(f"  pnl: {pnl_price}")
    print(f"  realized_pnl: {realized}")
    print(f"  funding_total: {funding_total}")
    print(f"  fee_total: {fee_total}")
    print(f"  pnl_percent: {pnl_percent}")
    print(f"  apr: {apr}")
    print(f"  initial_margin (resuelto): {initial_margin}")
    print(f"  notional: {notional}")
    print(f"  leverage: {leverage}")
    print(f"  liquidation_price: {liq_price}")

    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    
    try:
        print(f"🚀 [DEBUG] Ejecutando SQL...")
        cur.execute(sql, vals)
        conn.commit()
        print(f"✅ [DEBUG] POSICIÓN GUARDADA EXITOSAMENTE: {exchange} {symbol}")
        
        # Verificación
        cur.execute(
            "SELECT COUNT(*) FROM closed_positions WHERE exchange = ? AND symbol = ? AND close_time = ?", 
            (exchange, symbol, close_s)
        )
        count = cur.fetchone()[0]
        print(f"🔍 [DEBUG] Verificación en DB: {count} registros encontrados para esta posición")
        
    except Exception as e:
        print(f"❌ [DEBUG] ERROR al guardar posición {exchange} {symbol}: {e}")
        print(f"🔍 [DEBUG] SQL: {sql}")
        print(f"🔍 [DEBUG] Valores: {vals}")
        traceback.print_exc()
        conn.rollback()
    finally:
        conn.close()
        
#=============FIN DEL CODIGO 1. MODIFICACION DEL CODIGO 2 POR CHATGPT


# #=============CODIGO 2. fuciona

# def _safe(x, default=0.0):
#     try:
#         return float(x)
#     except Exception:
#         return default

# def _price_pnl(side, entry, close, size):
#     if side and side.lower() == "short":
#         return (entry - close) * size
#     return (close - entry) * size

# def _pnl_percent(realized_pnl, notional):
#     if notional and abs(notional) > 0:
#         return 100.0 * (realized_pnl / notional)
#     return 0.0

# def _apr(realized_pnl, notional, open_s, close_s):
#     try:
#         days = max((_safe(close_s) - _safe(open_s)) / 86400.0, 1e-9)
#         if days <= 0 or notional == 0:
#             return 0.0
#         return 100.0 * (realized_pnl / notional) * (365.0 / days)
#     except Exception:
#         return 0.0
    
# def save_closed_position(position: dict):
#     import sqlite3
#     import traceback
    
#     print(f"🔍 [DEBUG save_closed_position] Iniciando guardado para: {position.get('exchange')} {position.get('symbol')}")
    
#     def _f(x, d=0.0):
#         try: 
#             return float(x)
#         except: 
#             return d

#     def _positive(x):
#         return x is not None and x > 0

#     def _price_pnl(side, entry, close, size):
#         s = (side or "").lower()
#         return (entry - close) * size if s == "short" else (close - entry) * size

#     exchange = position.get("exchange")
#     symbol   = position.get("symbol")
#     side     = position.get("side")

#     size     = _f(position.get("size"))
#     entry    = _f(position.get("entry_price"))
#     close    = _f(position.get("close_price"))
#     open_s   = int(position.get("open_time") or 0)
#     close_s  = int(position.get("close_time") or 0)

#     fee_total     = -abs(_f(position.get("fee_total", 0.0)))       # fees siempre negativos
#     funding_total = _f(position.get("funding_total", 0.0))
#     liq_price     = _f(position.get("liquidation_price"))
    
#     # 1) PnL de precio: usa el que mande la API, si no, recalcúlalo
#     pnl_price_api = position.get("pnl")
#     pnl_price = _f(pnl_price_api) if pnl_price_api is not None else _price_pnl(side, entry, close, _f(size, 0.0))
    
#     # 2) Reconstrucción de size si viene escalada o 0 y hay PnL de precio
#     diff = abs(close - entry)
#     if (size <= 0 or (diff > 0 and abs(pnl_price) > 0 and abs(size - abs(pnl_price)/diff) / max(1.0, abs(size)) > 0.05)):
#         if diff > 0 and abs(pnl_price) > 0:
#             size = abs(pnl_price) / diff
    
#     # 3) Notional SIEMPRE a precio de entrada
#     entry_notional = abs(size) * entry
#     notional_api   = _f(position.get("notional", 0.0))
#     notional       = entry_notional if entry_notional > 0 else notional_api
    
#     # 4) Realized neto: si no viene, compónlo (precio + funding + fees)
#     realized_api = position.get("realized_pnl")
#     realized = _f(realized_api) if realized_api is not None else (pnl_price + funding_total + fee_total)
    
#     # 5) Resolver leverage (si hay). Si no, usar default por exchange.
#     #    Puedes mover este mapping a config/toggles.py si prefieres.
#     DEFAULT_LEVERAGE = {"gate": 5}
#     lev_raw = position.get("leverage")
#     leverage = _f(lev_raw) if lev_raw is not None else 0.0
#     if not _positive(leverage):
#         # Si la API dio initial_margin válido, deduce leverage = notional / margin
#         im_raw = position.get("initial_margin")
#         im_val = _f(im_raw) if im_raw is not None else 0.0
#         if _positive(im_val) and (notional > 0 or entry_notional > 0):
#             leverage = (notional if notional > 0 else entry_notional) / im_val
#         elif exchange in DEFAULT_LEVERAGE and _positive(DEFAULT_LEVERAGE[exchange]):
#             leverage = DEFAULT_LEVERAGE[exchange]  # ← Gate: 5 por defecto
#         else:
#             leverage = 0.0
    
#     # 6) Margen inicial
#     initial_margin = position.get("initial_margin")
#     initial_margin = _f(initial_margin) if initial_margin is not None else 0.0
#     if not _positive(initial_margin):
#         if _positive(leverage):
#             initial_margin = (notional if notional > 0 else entry_notional) / leverage
#         else:
#             # Último recurso: usa entry_notional (ROI más conservador)
#             initial_margin = entry_notional
    
#     # 7) Métricas: SIEMPRE desde realized_pnl
#     base_capital = initial_margin if _positive(initial_margin) else (notional if _positive(notional) else 0.0)
#     pnl_percent = (realized / base_capital) * 100.0 if _positive(base_capital) else 0.0
    
#     days = max((close_s - open_s) / 86400.0, 1e-9) if (open_s and close_s) else 0.0
#     apr = pnl_percent * (365.0 / days) if days > 0 else 0.0


#     # INSERT con columnas y placeholders 1:1
#     sql = (
#         "INSERT INTO closed_positions ("
#         "exchange, symbol, side, size, entry_price, close_price, "
#         "open_time, close_time, pnl, realized_pnl, funding_total, fee_total, "
#         "pnl_percent, apr, initial_margin, notional, leverage, liquidation_price"
#         ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
#     )

#     vals = (
#         exchange, symbol, side, size, entry, close,
#         open_s, close_s, pnl_price, realized, funding_total, fee_total,
#         float(pnl_percent), float(apr), initial_margin, notional, leverage, liq_price
#     )

#     # DEBUG DETALLADO
#     print(f"📋 [DEBUG] Datos a insertar:")
#     print(f"  exchange: {exchange}")
#     print(f"  symbol: {symbol}")
#     print(f"  side: {side}")
#     print(f"  size: {size}")
#     print(f"  entry_price: {entry}")
#     print(f"  close_price: {close}")
#     print(f"  open_time: {open_s} ({pd.to_datetime(open_s, unit='s') if open_s else 'N/A'})")
#     print(f"  close_time: {close_s} ({pd.to_datetime(close_s, unit='s') if close_s else 'N/A'})")
#     print(f"  pnl: {pnl_price}")
#     print(f"  realized_pnl: {realized}")
#     print(f"  funding_total: {funding_total}")
#     print(f"  fee_total: {fee_total}")
#     print(f"  pnl_percent: {pnl_percent}")
#     print(f"  apr: {apr}")
#     print(f"  initial_margin (resuelto): {initial_margin}")
#     print(f"  notional: {notional}")
#     print(f"  leverage: {leverage}")
#     print(f"  liquidation_price: {liq_price}")

#     conn = sqlite3.connect(DB_PATH)
#     cur  = conn.cursor()
    
#     try:
#         print(f"🚀 [DEBUG] Ejecutando SQL...")
#         cur.execute(sql, vals)
#         conn.commit()
#         print(f"✅ [DEBUG] POSICIÓN GUARDADA EXITOSAMENTE: {exchange} {symbol}")
        
#         # Verificación
#         cur.execute(
#             "SELECT COUNT(*) FROM closed_positions WHERE exchange = ? AND symbol = ? AND close_time = ?", 
#             (exchange, symbol, close_s)
#         )
#         count = cur.fetchone()[0]
#         print(f"🔍 [DEBUG] Verificación en DB: {count} registros encontrados para esta posición")
        
#     except Exception as e:
#         print(f"❌ [DEBUG] ERROR al guardar posición {exchange} {symbol}: {e}")
#         print(f"🔍 [DEBUG] SQL: {sql}")
#         print(f"🔍 [DEBUG] Valores: {vals}")
#         traceback.print_exc()
#         conn.rollback()
#     finally:
#         conn.close()
        
# #=============FIN DEL CODIGO 2. 

