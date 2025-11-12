
from flask import Flask, render_template, jsonify
import pandas as pd
import requests
import time
import hashlib
from dotenv import load_dotenv
import hmac
import os
from urllib.parse import urlencode
import json
import datetime
import math
from datetime import datetime, timezone
import json, urllib
from base64 import urlsafe_b64encode
from base58 import b58decode, b58encode
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from requests import Request, Session
import sqlite3
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

# ==== UTILIDADES DE TIEMPO (añade esto cerca de tus imports) ====
from zoneinfo import ZoneInfo

def _ms_to_str(ms: int | float | str | None, tz: str = "Europe/Zurich") -> str:
    """Convierte milisegundos epoch a 'dd-mm-YYYY HH:MM:SS TZ'. Maneja None/str."""
    try:
        if ms is None or ms == "" or ms == "N/A":
            return "N/A"
        ms = int(float(ms))
        dt = datetime.fromtimestamp(ms / 1000, tz=ZoneInfo("UTC")).astimezone(ZoneInfo(tz))
        return dt.strftime("%d-%m-%Y %H:%M:%S %Z")
    except Exception:
        return str(ms)

def _safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

# Aden (Orderly)

ORDERLY_SECRET = "GhxcFHy4s1b9EpguzyTUTdGAdEtnGXGNFEhe1gSc1WBN"  # clave privada en base58
ORDERLY_ACCOUNT_ID = os.getenv("ORDERLY_ACCOUNT_ID")
ORDERLY_BASE_URL = "https://api.orderly.org"

# generar private key desde secret base58
_private_key = Ed25519PrivateKey.from_private_bytes(b58decode(ORDERLY_SECRET))
_session = Session()

if not ORDERLY_SECRET:
    raise ValueError("❌ FALTA la variable ORDERLY_SECRET en el archivo .env")

try:
    _private_key = Ed25519PrivateKey.from_private_bytes(b58decode(ORDERLY_SECRET))
except Exception as e:
    raise ValueError(f"❌ Error al decodificar ORDERLY_SECRET: {e}")

_session = Session()

# derivar public key base58 (esta es la que va en el header orderly-key)
ORDERLY_PUBLIC_KEY_B58 = "ed25519:" + b58encode(
    _private_key.public_key().public_bytes_raw()
).decode("utf-8")



def _sign_request(req: Request) -> Request:
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    json_str = ""
    if req.json:
        json_str = json.dumps(req.json, separators=(',', ':'))

    url = urllib.parse.urlparse(req.url)
    message = str(ts) + req.method + url.path + json_str
    if url.query:
        message += "?" + url.query

    signature = urlsafe_b64encode(_private_key.sign(message.encode())).decode("utf-8")

    headers = {
        "orderly-timestamp": str(ts),
        "orderly-account-id": ORDERLY_ACCOUNT_ID,
        "orderly-key": ORDERLY_PUBLIC_KEY_B58,
        "orderly-signature": signature,
    }

    req.headers.update(headers)
    return req

def _send_request(method: str, path: str, params=None):
    """Versión simplificada sin debug"""
    url = f"{ORDERLY_BASE_URL}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    
    req = Request(method, url)
    signed = _sign_request(req).prepare()
    res = _session.send(signed, timeout=15)
    res.raise_for_status()
    return res.json()

# =====================
# Funciones con prints estandarizados estilo
# =====================
def fetch_account_aden(data=None):
    """Obtener cuenta de Aden usando estadísticas diarias"""
    try:
        print("🔍 DEBUG: Obteniendo CUENTA de Aden...")
        
        # Obtener fecha actual y de ayer para el rango
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Llamar al endpoint de estadísticas diarias
        stats_data = _send_request("GET", "/v1/client/statistics/daily", {
            "start_date": yesterday,
            "end_date": today,
            "size": 1
        })
        
        rows = stats_data.get("data", {}).get("rows", [])
        if not rows:
            print("⚠️ No se encontraron datos de estadísticas diarias")
            return None
        
        # Tomar el registro más reciente
        latest_stat = rows[0]
        account_value = float(latest_stat.get("account_value", 0))
        
        print(f"💼 Aden account_value: {account_value:.2f}")
        
        # Para otros campos necesarios, podemos usar valores estimados o dejar 0
        # ya que el endpoint principal no los proporciona directamente
        return {
            "exchange": "aden",
            "equity": account_value,
            "balance": account_value,  # Mismo que equity por ahora
            "unrealized_pnl": 0,      # No disponible en este endpoint
            "initial_margin": 0,       # No disponible en este endpoint  
            "total_collateral": account_value,
        }
        
    except Exception as e:
        print(f"❌ Aden account error: {e}")
        return None





def fetch_funding_aden(limit=100):
    """
    Funding history de Aden - VERSIÓN DEFINITIVA
    """
    try:
        print("🔍 DEBUG: Obteniendo FUNDING FEES (USDC) de Aden...")
        data = _send_request("GET", "/v1/funding_fee/history", {"size": min(int(limit), 500)})
        rows = data.get("data", {}).get("rows", [])
        if not rows:
            print("📦 DEBUG: Se recibieron 0 registros de funding")
            return []

        funding = []
        for f in rows:
            try:
                sym_raw = f.get("symbol", "")
                clean_sym = sym_raw.replace("perp_", "").replace("_usdc", "").upper()
                
                # INVERTIR SIGNO: Si la API reporta negativo para ganancias, lo hacemos positivo
                amt = -float(f.get("funding_fee", 0.0))
                fr = float(f.get("funding_rate", 0.0)) if f.get("funding_rate") is not None else None
                ts = f.get("created_time") or f.get("timestamp") or ""

                funding.append({
                    "exchange": "aden",
                    "symbol": clean_sym,
                    "income": amt,  # Positivo = ganancia
                    "asset": "USDC", 
                    "timestamp": ts,
                    "funding_rate": fr,
                    "type": "FUNDING_FEE"
                })
            except Exception as e:
                print(f"[WARNING] Error processing Aden funding row: {e}")
                continue

        print(f"📦 DEBUG: Se recibieron {len(funding)} registros de funding")
        return funding

    except Exception as e:
        print(f"[ERROR] Failed to fetch Aden funding: {e}")
        return []
# ================ Posiciones abiertas ==========================

def get_last_closed_info_for_symbol(symbol: str, limit: int = 200):
    """
    Devuelve (last_close_ts, last_position_id) para 'symbol' desde /v1/position_history.
    Si no hay cerradas: (None, None).
    """
    try:
        data = _send_request("GET", "/v1/position_history", {"symbol": symbol, "limit": str(limit)})
        rows = (data.get("data", {}) or {}).get("rows", []) or []
        if not rows:
            return None, None
        # Nos quedamos con la fila de mayor close_timestamp
        best = max((r for r in rows if r.get("close_timestamp") is not None),
                   key=lambda r: int(r.get("close_timestamp")), default=None)
        if not best:
            return None, None
        return int(best["close_timestamp"]), best.get("position_id")
    except Exception as e:
        print(f"⚠️ get_last_closed_info_for_symbol error: {e}")
        return None, None

def _reconstruct_position_funding_since(symbol: str, start_ms: int | None) -> float:
    """
    Suma funding_fee (con signo normalizado: positivo si recibes) para 'symbol'
    desde 'start_ms' (inclusive) hasta ahora, consultando /v1/funding_fee/history
    con filtros. Si start_ms es None, no pone filtro temporal.
    """
    realized_funding = 0.0
    try:
        params = {"symbol": symbol, "size": "500", "page": "1"}
        if start_ms is not None:
            params["start_t"] = str(int(start_ms))
        fd = _send_request("GET", "/v1/funding_fee/history", params)
        rows = fd.get("data", {}).get("rows", []) or []
        for r in rows:
            # La API reporta fees con signo "exchange": si "Pay" suele ser negativo para ti.
            # En tu pipeline ya estabas normalizando como '-funding_fee' => mantenemos eso:
            realized_funding += -float(r.get("funding_fee", 0) or 0)
        return realized_funding
    except Exception as e:
        print(f"⚠️ funding filtered error, fallback sin filtro: {e}")

    # Fallback sin filtro + filtrado local
    try:
        fd_all = _send_request("GET", "/v1/funding_fee/history", {"size": "500"})
        rows_all = fd_all.get("data", {}).get("rows", []) or []
        for r in rows_all:
            if r.get("symbol") == symbol:
                c_ms = r.get("created_time")
                if start_ms is None or (c_ms and int(c_ms) >= int(start_ms)):
                    realized_funding += -float(r.get("funding_fee", 0) or 0)
        return realized_funding
    except Exception as e:
        print(f"❌ funding fallback error: {e}")
        return 0.0

def fetch_positions_aden(data=None):
    """
    Obtiene posiciones abiertas y reconstruye funding
    CORTANDO por close_timestamp de la última cerrada del MISMO símbolo.
    Asigna un position_id SINTÉTICO = (last_closed_id or 0) + 1.
    """
    try:
        if data is None:
            data = _send_request("GET", "/v1/positions")

        server_ts = data.get("timestamp")
        print(f"🕒 Server timestamp: {server_ts} -> {_ms_to_str(server_ts)}")

        positions_data = data.get("data", {}).get("rows", []) or []
        print(f"📦 Aden: {len(positions_data)} posiciones abiertas")

        formatted_positions = []

        for i, pos in enumerate(positions_data):
            raw_symbol = pos.get("symbol", "")                  # p.ej. PERP_KAITO_USDC
            clean_symbol = raw_symbol.lower().replace("perp_", "").replace("_usdc", "").upper()

            # Paso 1: encontrar último cierre del mismo símbolo
            last_close_ts, last_closed_id = get_last_closed_info_for_symbol(raw_symbol)
            synthetic_id = (last_closed_id or 0) + 1

            # (opcional) sanity: updated_time de la abierta siempre debe ser >= last_close_ts
            upd_ms = pos.get("updated_time")
            if last_close_ts and upd_ms and int(upd_ms) < int(last_close_ts):
                print(f"⚠️ updated_time({upd_ms}) < last_close_ts({last_close_ts}) en {raw_symbol}")

            # Paso 2: funding desde el último cierre (si existe); si no, desde 'timestamp' de positions
            # Nota: tu caso KAITO demuestra que el 'timestamp' puede ser antiguo; usamos close_ts si existe.
            start_ms = last_close_ts if last_close_ts is not None else pos.get("timestamp")
            realized_funding = _reconstruct_position_funding_since(raw_symbol, start_ms)

            entry_price = float(pos.get("average_open_price", 0) or 0)
            mark_price  = float(pos.get("mark_price", 0) or 0)
            quantity    = float(pos.get("position_qty", 0) or 0)
            side        = "long" if quantity > 0 else "short"
            unrealized  = (mark_price - entry_price) * abs(quantity)
            notional    = float(pos.get("cost_position", 0) or 0)
            lev         = float(pos.get("leverage", 0) or 0)
            liq         = float(pos.get("est_liq_price", 0) or 0)

            print(f"   🧾 Pos {i}: {clean_symbol} | synthetic_position_id={synthetic_id}")
            print(f"      last_close_ts={last_close_ts} ({_ms_to_str(last_close_ts)})  | start_ms={start_ms} ({_ms_to_str(start_ms)})")
            print(f"      ✅ funding (reconstruido, desde corte): {realized_funding:.8f} USDC")

            formatted_positions.append({
                "symbol": clean_symbol,
                "exchange_symbol": raw_symbol,          # útil por si necesitas llamar endpoints
                "synthetic_position_id": synthetic_id,  # ID para tu UI
                "size": abs(quantity),
                "quantity": abs(quantity),
                "side": side,
                "unrealized_pnl": unrealized,
                "realized_pnl": realized_funding,
                "funding_fee": realized_funding,
                "entry_price": entry_price,
                "mark_price": mark_price,
                "leverage": lev,
                "liquidation_price": liq,
                "notional": notional,
                "exchange": "aden",
                "open_timestamp_ms_positions_field": pos.get("timestamp"),  # por transparencia
                "funding_start_ms": start_ms,          # el que realmente usamos
            })

        return formatted_positions

    except Exception as e:
        print(f"❌ Aden positions error: {e}")
        import traceback; traceback.print_exc()
        return []



# =============== POSICIONES CERRADAS (Aden/Orderly) ===============

def _normalize_aden_closed_position(pos_data: dict) -> Optional[Dict[str, Any]]:
    """
    Normaliza posición cerrada de Aden/Orderly:
    - 'pnl'  = PnL de precio (solo precio)
    - 'realized_pnl' = NETO (precio + fees + funding)
    - 'fees' negativas si son coste
    - 'funding_fee' conserva el signo de la API (negativo=pagado, positivo=cobrado)
    """
    try:
        raw_symbol = pos_data.get("symbol", "")
        clean_symbol = (
            raw_symbol.lower()
            .replace("perp_", "")
            .replace("_usdc", "")
            .upper()
        )

        # side
        side = (pos_data.get("side") or "").lower()
        if not side:
            size_raw = float(pos_data.get("position_qty", 0) or 0)
            side = "long" if size_raw >= 0 else "short"

        # tamaños y precios
        size = abs(float(pos_data.get("closed_position_qty") or pos_data.get("position_qty") or 0))
        entry_price = float(pos_data.get("avg_open_price") or pos_data.get("average_open_price") or 0)
        close_price = float(pos_data.get("avg_close_price") or 0)
        if size < 0.000001:
            return None

        # --- PnL de precio ---
        # La API pone 'realized_pnl' como PnL de precio. Usamos eso si viene; si no, lo calculamos.
        api_price_pnl = float(pos_data.get("realized_pnl") or 0)
        pnl_price_calc = (close_price - entry_price) * size * (1 if side == "long" else -1)
        pnl_price_only = api_price_pnl if api_price_pnl != 0 else pnl_price_calc

        # --- Fees & funding ---
        fee_raw = float(pos_data.get("trading_fee") or 0)
        # Normalizamos fees como coste negativo (si viniera positivo, lo pasamos a negativo)
        fee_total = fee_raw if fee_raw < 0 else -abs(fee_raw)

        # INVERTIR SIGNO del funding
        funding_total = -float(pos_data.get("accumulated_funding_fee") or 0)

        # --- Realized neto ---
        realized_pnl_net = pnl_price_only + fee_total + funding_total

        # timestamps
        open_time_raw = pos_data.get("open_timestamp") or pos_data.get("created_time") or 0
        close_time_raw = pos_data.get("close_timestamp") or pos_data.get("updated_time") or 0
        open_time = int(float(open_time_raw) / 1000) if open_time_raw else 0
        close_time = int(float(close_time_raw) / 1000) if close_time_raw else 0

        # leverage y notional
        leverage = float(pos_data.get("leverage") or 0)
        notional = float(pos_data.get("cost_position") or 0) or (entry_price * size)
        initial_margin = notional / leverage if leverage else 0

        closed_position = {
            "exchange": "aden",
            "symbol": clean_symbol,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "close_price": close_price,
            "notional": notional,

            # claves homogéneas con el resto de exchanges / frontend
            "pnl": pnl_price_only,                 # PnL de precio
            "realized_pnl": realized_pnl_net,      # NETO = precio + fees + funding
            "fees": fee_total,                     # coste negativo
            "fee_total": fee_total,                # alias
            "funding_fee": funding_total,          # signo de la API (negativo pagado)
            "funding_total": funding_total,        # alias

            "open_time": open_time,
            "close_time": close_time,
            "initial_margin": initial_margin,
            "leverage": leverage,
            "liquidation_price": float(pos_data.get("est_liq_price") or 0),
        }

        print(
            f"      ✅ Aden normalizada: {clean_symbol} | "
            f"price_pnl={pnl_price_only:.4f} | fees={fee_total:.4f} | funding={funding_total:.4f} | "
            f"realized(net)={realized_pnl_net:.4f}"
        )
        return closed_position

    except Exception as e:
        print(f"❌ ERROR normalizando posición cerrada de Aden: {e}")
        return None


def fetch_closed_positions_aden(debug=False):
    """
    Obtener posiciones cerradas de Aden / Orderly.
    Endpoint: GET /v1/position_history
    """
    try:
        print("🔍 DEBUG: Obteniendo POSICIONES CERRADAS (ADEN)...")
        print("   🌐 Llamando endpoint: /v1/position_history")
        data = _send_request("GET", "/v1/position_history", {"limit": 100})
        rows = data.get("data", {}).get("rows", [])
        if not rows:
            print("⚠️ No se encontraron posiciones en el historial de Aden.")
            return []

        print(f"📦 DEBUG: Se recibieron {len(rows)} registros del historial de posiciones")

        results = []
        for r in rows:
            # En Aden, las posiciones cerradas pueden no tener un status específico
            # o pueden tener status diferente. Procesamos todas y filtramos por lógica
            closed_pos = _normalize_aden_closed_position(r)
            if closed_pos:
                results.append(closed_pos)
                print(f"      ✅ Normalizada: {closed_pos['symbol']} - PnL: {closed_pos['realized_pnl']}")
            else:
                if debug:
                    symbol = r.get("symbol", "unknown")
                    print(f"      ❌ No se pudo normalizar posición: {symbol}")

        print(f"✅ DEBUG: {len(results)} posiciones cerradas normalizadas")
        return results

    except Exception as e:
        print(f"❌ Error al obtener posiciones cerradas de Aden: {e}")
        return []




def save_aden_closed_positions(db_path="portfolio.db", debug=False):
    """
    Guarda posiciones cerradas de Aden en SQLite.
    Versión mejorada similar a save_gate_closed_positions
    """
    import os, sqlite3
    from db_manager import save_closed_position

    print("💾 Guardando posiciones cerradas de Aden en portfolio.db")

    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return 0

    closed_positions = fetch_closed_positions_aden(debug=debug)
    if not closed_positions:
        print("⚠️ No se obtuvieron posiciones cerradas de Aden.")
        return 0

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    saved, skipped = 0, 0

    for pos in closed_positions:
        try:
            # Deduplicación por (exchange, symbol, close_time) - igual que Gate
            cur.execute("""
                SELECT COUNT(*) FROM closed_positions
                WHERE exchange = ? AND symbol = ? AND close_time = ?
            """, (pos["exchange"], pos["symbol"], pos["close_time"]))
            
            if cur.fetchone()[0]:
                skipped += 1
                continue

            # Mapear a los nombres que espera save_closed_position
            position_data = {
                "exchange": pos["exchange"],
                "symbol": pos["symbol"],
                "side": pos["side"],
                "size": pos["size"],
                "entry_price": pos["entry_price"],
                "close_price": pos["close_price"],
                "open_time": pos["open_time"],
                "close_time": pos["close_time"],
                "pnl": pos["pnl"],                              # precio
                "realized_pnl": pos["realized_pnl"],            # neto
                "fee_total": pos.get("fee_total", pos.get("fees", 0)),
                "funding_total": pos.get("funding_fee", 0),
                "notional": pos.get("notional", 0),
                "leverage": pos.get("leverage"),
                "liquidation_price": pos.get("liquidation_price"),
                "initial_margin": pos.get("initial_margin"),
            }

            save_closed_position(position_data)
            saved += 1

        except Exception as e:
            print(f"⚠️ Error guardando posición {pos.get('symbol')} (Aden): {e}")

    conn.close()
    print(f"✅ Aden guardadas: {saved} | omitidas (duplicadas): {skipped}")
    return saved


# ============================================================
# DEBUG: CLOSED POSITIONS EN FILAS HORIZONTALES (clave=valor)
# Endpoint: /v1/position_history  (params: symbol?, limit)
# ============================================================

from zoneinfo import ZoneInfo
from datetime import datetime
import json

def _ms_to_str(ms: int | float | str | None, tz: str = "Europe/Zurich") -> str:
    try:
        if ms in (None, "", "N/A"):
            return "N/A"
        ms = int(float(ms))
        dt = datetime.fromtimestamp(ms/1000, tz=ZoneInfo("UTC")).astimezone(ZoneInfo(tz))
        return dt.strftime("%d-%m-%Y %H:%M:%S %Z")
    except Exception:
        return str(ms)

def _fmt(x, n=8):
    try:
        f = float(x)
        return f"{f:.{n}f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)


def debug_closed_positions_aden_rows(symbol: str | None = None, limit: int = 200):
    """
    Muestra /v1/position_history como FILAS HORIZONTALES (una línea por posición):
      position_id=... | status=... | type=... | symbol=... | side=... | ...
      open=OPEN_MS (OPEN_FECHA) | close=CLOSE_MS (CLOSE_FECHA) | last_upd=LAST_MS (LAST_FECHA)
    """
    try:
        params = {"limit": str(limit)}
        if symbol:
            params["symbol"] = symbol

        data = _send_request("GET", "/v1/position_history", params)

        server_ts = data.get("timestamp")
        print("\n" + "="*160)
        print("🔎 DEBUG CLOSED POSITIONS (FILAS HORIZONTALES) /v1/position_history")
        print("="*160)
        if server_ts is not None:
            print(f"🕒 server.timestamp={server_ts} ({_ms_to_str(server_ts)})")

        rows = (data.get("data", {}) or {}).get("rows", []) or []
        print(f"📦 rows={len(rows)}")
        if not rows:
            print("(sin filas)")
            print("="*160)
            return

        # Cabecera única (opcional). Puedes comentarla si no la quieres.
        header_keys = [
            "position_id","status","type","symbol","side",
            "avg_open_price","avg_close_price",
            "max_position_qty","closed_position_qty",
            "trading_fee","accumulated_funding_fee",
            "insurance_fund_fee","liquidator_fee",
            "realized_pnl","leverage","open","close","last_upd"
        ]
        print(" | ".join(header_keys))
        print("-"*160)

        for r in rows:
            open_ms  = r.get("open_timestamp")
            close_ms = r.get("close_timestamp")
            upd_ms   = r.get("last_update_timestamp")

            # Construimos la fila horizontal
            parts = [
                f"position_id={r.get('position_id')}",
                f"status={r.get('status')}",
                f"type={r.get('type')}",
                f"symbol={r.get('symbol')}",
                f"side={r.get('side')}",
                f"avg_open_price={_fmt(r.get('avg_open_price'))}",
                f"avg_close_price={_fmt(r.get('avg_close_price'))}",
                f"max_position_qty={_fmt(r.get('max_position_qty'))}",
                f"closed_position_qty={_fmt(r.get('closed_position_qty'))}",
                f"trading_fee={_fmt(r.get('trading_fee'))}",
                f"accumulated_funding_fee={_fmt(r.get('accumulated_funding_fee'))}",
                f"insurance_fund_fee={_fmt(r.get('insurance_fund_fee'))}",
                f"liquidator_fee={_fmt(r.get('liquidator_fee'))}",
                f"realized_pnl={_fmt(r.get('realized_pnl'))}",
                f"leverage={_fmt(r.get('leverage'), n=4)}",
                f"open={open_ms} ({_ms_to_str(open_ms)})",
                f"close={close_ms} ({_ms_to_str(close_ms)})",
                f"last_upd={upd_ms} ({_ms_to_str(upd_ms)})",
            ]

            print(" | ".join(parts))

        print("="*160)

    except Exception as e:
        print(f"❌ debug_closed_positions_aden_rows error: {e}")
        import traceback; traceback.print_exc()



def debug_aden_funding():
    """Debug específico para funding de Aden"""
    print("\n" + "="*50)
    print("🔍 DEBUG ADEN - FUNDING FEES")
    print("="*50)
    
    try:
        # 1. Obtener funding history completo
        print("📋 Obteniendo historial completo de funding...")
        funding_data = _send_request("GET", "/v1/funding_fee/history", {"size": 200})
        funding_rows = funding_data.get("data", {}).get("rows", [])
        
        print(f"📦 Total registros de funding: {len(funding_rows)}")
        
        # 2. Agrupar por símbolo y mostrar sumarios
        funding_by_symbol = {}
        for i, fee in enumerate(funding_rows):
            symbol = fee.get("symbol", "unknown")
            funding_fee = float(fee.get("funding_fee", 0))
            timestamp = fee.get("created_time", "")
            
            if symbol not in funding_by_symbol:
                funding_by_symbol[symbol] = []
            
            funding_by_symbol[symbol].append({
                "amount": funding_fee,
                "timestamp": timestamp
            })
            
            # Mostrar primeros 5 registros
            if i < 5:
                print(f"   [{i}] {symbol}: {funding_fee} USDC - {timestamp}")
        
        # 3. Mostrar sumario por símbolo
        print("\n📊 SUMA DE FUNDING POR SÍMBOLO:")
        for symbol, fees in funding_by_symbol.items():
            total = sum(f["amount"] for f in fees)
            clean_symbol = symbol.replace("perp_", "").replace("_usdc", "").upper()
            print(f"   {clean_symbol}: {total:.6f} USDC ({len(fees)} registros)")
            
            # Mostrar distribución temporal
            if len(fees) > 1:
                first = min(f["timestamp"] for f in fees)
                last = max(f["timestamp"] for f in fees)
                print(f"        Período: {first} -> {last}")
        
        # 4. Obtener posiciones actuales
        print("\n📋 Obteniendo posiciones actuales...")
        positions_data = _send_request("GET", "/v1/positions")
        current_positions = positions_data.get("data", {}).get("rows", [])
        
        print(f"📦 Posiciones abiertas actualmente: {len(current_positions)}")
        
        for pos in current_positions:
            symbol = pos.get("symbol", "")
            clean_symbol = symbol.replace("perp_", "").replace("_usdc", "").upper()
            created_time = pos.get("created_time", "")
            print(f"   {clean_symbol}: creada en {created_time}")
            
            # Calcular funding solo para esta posición
            if symbol in funding_by_symbol:
                position_funding = 0.0
                for fee in funding_by_symbol[symbol]:
                    # Solo funding después de crear la posición
                    if fee["timestamp"] >= created_time:
                        position_funding += fee["amount"]
                
                print(f"        Funding para esta posición: {position_funding:.6f} USDC")
        
        return funding_by_symbol, current_positions
        
    except Exception as e:
        print(f"❌ Error en debug: {e}")
        import traceback
        traceback.print_exc()
        return {}, []


    
#==== DEBUG COMPLETO (nuevo) ====
def debug_aden_all():
    """
    Debug integral:
      - funding_fee/history: imprime RAW + fechas legibles (server y filas)
      - positions: imprime RAW + fechas legibles (server y filas)
      - para cada posición abierta: funding reconstruido desde 'timestamp'
    """
    print("\n" + "="*80)
    print("🔎 DEBUG ADEN - TODO (Funding + Positions)")
    print("="*80)

    # ----- Funding Fee History -----
    try:
        print("\n[1] FUNDING FEE HISTORY")
        ff = _send_request("GET", "/v1/funding_fee/history", {"size": "120", "page": "1"})
        server_ts = ff.get("timestamp")
        print(f"   🕒 server.timestamp: {server_ts} -> {_ms_to_str(server_ts)}")
        rows = ff.get("data", {}).get("rows", []) or []
        print(f"   📦 rows: {len(rows)}")
        for idx, r in enumerate(rows):
            c_ms = r.get("created_time")
            u_ms = r.get("updated_time")
            print(f"   ── Row[{idx}] RAW: {json.dumps(r, ensure_ascii=False)}")
            print(f"      created_time: {c_ms} -> {_ms_to_str(c_ms)}")
            print(f"      updated_time: {u_ms} -> {_ms_to_str(u_ms)}")
            if idx >= 4:
                # No inundar el log
                break
    except Exception as e:
        print(f"   ❌ funding_fee/history error: {e}")

    # ----- Positions -----
    try:
        print("\n[2] POSITIONS (abiertas)")
        pd = _send_request("GET", "/v1/positions")
        server_ts = pd.get("timestamp")
        print(f"   🕒 server.timestamp: {server_ts} -> {_ms_to_str(server_ts)}")
        prows = pd.get("data", {}).get("rows", []) or []
        print(f"   📦 rows: {len(prows)}")

        for i, pos in enumerate(prows):
            print(f"   ── Pos[{i}] RAW: {json.dumps(pos, ensure_ascii=False)}")
            open_ms = pos.get("timestamp")
            upd_ms  = pos.get("updated_time")
            print(f"      open.timestamp: {open_ms} -> {_ms_to_str(open_ms)}")
            print(f"      updated_time:   {upd_ms}  -> {_ms_to_str(upd_ms)}")

        # Además: reconstrucción de funding por cada posición (usando la función corregida)
        print("\n[3] FUNDING POR POSICIÓN (reconstruido)")
        formatted = fetch_positions_aden(data=pd)  # reutiliza la respuesta ya obtenida
        for i, fp in enumerate(formatted):
            print(f"   ◽ {i} {fp['symbol']} | side={fp['side']} | size={fp['size']}")
            print(f"      funding_fee: {fp['funding_fee']:.8f} | entry={fp['entry_price']:.6f} | mark={fp['mark_price']:.6f}")

    except Exception as e:
        print(f"   ❌ positions debug error: {e}")

    print("\n" + "="*80)
    print("✅ FIN DEBUG")
    print("="*80)

# ==== MAIN de debug (puedes reemplazar el main actual si quieres) ====
if __name__ == "__main__":
    print("🚀 INICIANDO DEBUG COMPLETO DE ADEN (ALL)")
    debug_aden_all()    
    debug_closed_positions_aden_rows()

