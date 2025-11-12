
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


def fetch_positions_aden(data=None):
    """Obtener posiciones de Aden - VERSIÓN CORREGIDA"""
    try:
        if data is None:
            data = _send_request("GET", "/v1/positions")
        positions_data = data.get("data", {}).get("rows", [])
        
        print(f"📦 Aden: {len(positions_data)} posiciones abiertas")
        
        # DEBUG: Mostrar información completa de las posiciones
        for i, pos in enumerate(positions_data):
            symbol = pos.get("symbol", "unknown")
            quantity = float(pos.get("position_qty", 0))
            created_time = pos.get("created_time", "N/A")
            position_id = pos.get("position_id", "N/A")
            print(f"   🧾 Posición {i}: {symbol}, size={quantity}, created_time={created_time}, position_id={position_id}")
        
        # Obtener funding history reciente
        try:
            funding_data = _send_request("GET", "/v1/funding_fee/history", {"size": 100})
            funding_rows = funding_data.get("data", {}).get("rows", [])
            print(f"💰 Aden: {len(funding_rows)} registros de funding obtenidos")
        except Exception as e:
            print(f"⚠️ No se pudo obtener funding history: {e}")
            funding_rows = []
        
        formatted_positions = []
        for pos in positions_data:
            raw_symbol = pos.get("symbol", "")
            clean_symbol = (
                raw_symbol.lower()
                .replace("perp_", "")
                .replace("_usdc", "")
                .upper()
            )
            
            # Obtener timestamp de creación de la posición - MANEJO SEGURO
            created_time_str = pos.get("created_time") or pos.get("open_timestamp") or "0"
            try:
                created_time_int = int(created_time_str) if created_time_str and created_time_str != "N/A" else 0
            except:
                created_time_int = 0
            
            print(f"   ⏰ {clean_symbol}: created_time={created_time_str} -> {created_time_int}")
            
            # Calcular funding para esta posición
            realized_funding = 0.0
            for fee in funding_rows:
                fee_symbol = fee.get("symbol", "")
                fee_time_str = fee.get("created_time", "0")
                
                try:
                    fee_time_int = int(fee_time_str) if fee_time_str and fee_time_str != "N/A" else 0
                except:
                    fee_time_int = 0
                
                # Solo sumar si es el mismo símbolo y ocurrió después de abrir la posición
                if fee_symbol == raw_symbol and fee_time_int >= created_time_int:
                    funding_amount = float(fee.get("funding_fee", 0))
                    # INVERTIR SIGNO: Aden reporta ganancias como negativas
                    realized_funding += -funding_amount
            
            entry_price = float(pos.get("average_open_price", 0))
            mark_price = float(pos.get("mark_price", 0))
            quantity = float(pos.get("position_qty", 0))
            unrealized_pnl = (mark_price - entry_price) * abs(quantity)
            notional = float(pos.get("cost_position", 0))
            lev = float(pos.get("leverage", 1))
            liq = float(pos.get("est_liq_price", 0))
            side = "long" if quantity > 0 else "short"

            formatted_pos = {
                "symbol": clean_symbol,
                "size": abs(quantity),
                "quantity": abs(quantity),
                "side": side,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": realized_funding,
                "funding_fee": realized_funding,
                "entry_price": entry_price,
                "mark_price": mark_price,
                "leverage": lev,
                "liquidation_price": liq,
                "notional": notional,
                "exchange": "aden"
            }
            formatted_positions.append(formatted_pos)
            
            print(f"   ✅ {clean_symbol}: size={abs(quantity):.4f}, entry={entry_price:.4f}, funding={realized_funding:.6f}")

        return formatted_positions
        
    except Exception as e:
        print(f"❌ Aden positions error: {e}")
        import traceback
        traceback.print_exc()
        return []
# =============== POSICIONES CERRADAS (Aden/Orderly) ===============

def _normalize_aden_closed_position(pos_data: dict) -> Optional[Dict[str, Any]]:
    """
    Normaliza posición cerrada de Aden - VERSIÓN DEFINITIVA
    """
    try:
        raw_symbol = pos_data.get("symbol", "")
        clean_symbol = (
            raw_symbol.lower()
            .replace("perp_", "")
            .replace("_usdc", "")
            .upper()
        )
        
        # Determinar side
        side = (pos_data.get("side") or "").lower()
        if not side:
            size_raw = float(pos_data.get("position_qty", 0))
            side = "long" if size_raw >= 0 else "short"
        
        # Obtener cantidades y precios
        size = abs(float(pos_data.get("closed_position_qty") or pos_data.get("position_qty") or 0))
        entry_price = float(pos_data.get("avg_open_price") or pos_data.get("average_open_price") or 0)
        close_price = float(pos_data.get("avg_close_price") or 0)
        
        if size < 0.000001:
            return None
        
        # Obtener PnL y fees
        realized_pnl = float(pos_data.get("realized_pnl") or 0)
        fee_total = float(pos_data.get("trading_fee") or 0)
        
        # INVERTIR SIGNO del funding
        funding_total = -float(pos_data.get("accumulated_funding_fee") or 0)
        
        # Calcular PnL solo de precio (sin fees/funding)
        pnl_price_only = (close_price - entry_price) * size * (1 if side == "long" else -1)
        
        # Timestamps
        open_time_raw = pos_data.get("open_timestamp") or pos_data.get("created_time") or 0
        close_time_raw = pos_data.get("close_timestamp") or pos_data.get("updated_time") or 0
        
        open_time = int(float(open_time_raw) / 1000) if open_time_raw else 0
        close_time = int(float(close_time_raw) / 1000) if close_time_raw else 0
        
        # Leverage y notional
        leverage = float(pos_data.get("leverage") or 0)
        notional = float(pos_data.get("cost_position") or 0) or (entry_price * size)
        initial_margin = notional / leverage if leverage else 0
        
        # ✅ ESQUEMA DEFINITIVO
        closed_position = {
            "exchange": "aden",
            "symbol": clean_symbol,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "close_price": close_price,
            "notional": notional,
            "fees": fee_total,
            "funding_fee": funding_total,  # Positivo = ganancia
            "realized_pnl": realized_pnl,
            "pnl": pnl_price_only,
            "open_time": open_time,
            "close_time": close_time,
            "initial_margin": initial_margin,
            "leverage": leverage,
            "liquidation_price": float(pos_data.get("est_liq_price") or 0)
        }
        
        print(f"      ✅ Aden normalizada: {clean_symbol} - Size: {size}, Realized PnL: {realized_pnl}, Funding: {funding_total}")
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
                "realized_pnl": pos["realized_pnl"],
                "funding_total": pos.get("funding_fee", 0),
                "pnl": pos.get("pnl"),
                "fee_total": pos.get("fees", 0),
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

def debug_aden_closed_positions():
    """Debug específico para posiciones cerradas de Aden"""
    print("\n" + "="*50)
    print("🔍 DEBUG ADEN - POSICIONES CERRADAS")
    print("="*50)
    
    try:
        # Obtener historial de posiciones
        print("📋 Obteniendo historial de posiciones...")
        data = _send_request("GET", "/v1/position_history", {"limit": 50})
        rows = data.get("data", {}).get("rows", [])
        
        print(f"📦 Total registros en historial: {len(rows)}")
        
        closed_positions = []
        for i, pos in enumerate(rows):
            status = pos.get("status", "")
            symbol = pos.get("symbol", "")
            clean_symbol = symbol.replace("perp_", "").replace("_usdc", "").upper()
            
            if status == "closed":
                realized_pnl = float(pos.get("realized_pnl", 0))
                funding_fee = float(pos.get("accumulated_funding_fee", 0))
                
                print(f"\n🔒 POSICIÓN CERRADA #{i}:")
                print(f"   Símbolo: {clean_symbol}")
                print(f"   Status: {status}")
                print(f"   Realized PnL: {realized_pnl}")
                print(f"   Funding Fee: {funding_fee}")
                print(f"   RAW DATA: {pos}")
                
                closed_positions.append(pos)
        
        print(f"\n📊 Total posiciones cerradas: {len(closed_positions)}")
        return closed_positions
        
    except Exception as e:
        print(f"❌ Error en debug: {e}")
        import traceback
        traceback.print_exc()
        return []

# Ejecutar debug
if __name__ == "__main__":
    print("🚀 INICIANDO DEBUG COMPLETO DE ADEN")
    
    # Debug funding
    funding_data, current_positions = debug_aden_funding()
    
    # Debug posiciones cerradas  
    closed_positions = debug_aden_closed_positions()
    
    print("\n" + "="*50)
    print("🎯 RESUMEN FINAL")
    print("="*50)
    print(f"📊 Funding records: {sum(len(v) for v in funding_data.values())}")
    print(f"📊 Posiciones abiertas: {len(current_positions)}")
    print(f"📊 Posiciones cerradas: {len(closed_positions)}")
    

