from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from requests import Request, Session
from collections import defaultdict
from datetime import datetime, timedelta
from collections import defaultdict
import sqlite3
from db_manager import init_db, save_closed_position
import os
import time
import hmac
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
import argparse
import sys
from difflib import SequenceMatcher

import requests

from dotenv import load_dotenv
load_dotenv()

#====== Imports para prints
# from pp import (
#     p_closed_debug_header, p_closed_debug_count, p_closed_debug_norm_size,
#     p_closed_debug_prices, p_closed_debug_pnl, p_closed_debug_times, p_closed_debug_normalized,
#     p_open_summary, p_open_block,
#     p_funding_fetching, p_funding_count,
#     p_balance_equity
# )
#===========================


GATE_HOST = "https://api.gateio.ws"
GATE_PREFIX = "/api/v4"
GATE_API_KEY = os.getenv("GATE_API_KEY")
GATE_API_SECRET = os.getenv("GATE_API_SECRET")

class GateV4Error(Exception):
    pass

def _require_keys():
    if not GATE_API_KEY or not GATE_API_SECRET:
        raise GateV4Error("Faltan GATE_API_KEY o GATE_API_SECRET en el entorno.")

def _sha512_hex(s: str) -> str:
    return hashlib.sha512(s.encode("utf-8")).hexdigest()

def _sign_v4(method: str, path: str, query: str, body: str, ts: int) -> str:
    """
    SIGN = HexEncode(HMAC_SHA512(secret, f"{METHOD}\\n{prefix+path}\\n{query}\\n{sha512(body)}\\n{Timestamp}"))
    """
    prehash = "\n".join([method.upper(), f"{GATE_PREFIX}{path}", query, _sha512_hex(body), str(ts)])
    return hmac.new(GATE_API_SECRET.encode("utf-8"), prehash.encode("utf-8"), hashlib.sha512).hexdigest()

def _headers(method: str, path: str, params: Optional[Dict[str, Any]], body_obj: Optional[Any]) -> Dict[str, str]:
    _require_keys()
    ts = int(time.time())
    query = "" if not params else urlencode(params, doseq=True)
    body = "" if body_obj is None else (body_obj if isinstance(body_obj, str) else json.dumps(body_obj, separators=(",", ":"), ensure_ascii=False))
    sign = _sign_v4(method, path, query, body, ts)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Timestamp": str(ts),
        "KEY": GATE_API_KEY,
        "SIGN": sign,
    }
    return headers

def _request(method: str, path: str, params: Optional[Dict[str, Any]] = None, body_obj: Optional[Any] = None, timeout: int = 15) -> Any:
    url = f"{GATE_HOST}{GATE_PREFIX}{path}"
    headers = _headers(method, path, params, body_obj)
    try:
        resp = requests.request(method.upper(), url, params=params, data=(None if body_obj is None else json.dumps(body_obj, separators=(",", ":"), ensure_ascii=False)), headers=headers, timeout=timeout)
    except requests.RequestException as e:
        raise GateV4Error(f"Error de red: {e}") from e
    if resp.status_code >= 400:
        raise GateV4Error(f"HTTP {resp.status_code}: {resp.text}")
    try:
        return resp.json()
    except ValueError:
        return resp.text

# ---------- Helpers de normalización ----------

def _contract_to_symbol(contract: str) -> str:
    # "BTC_USDT" -> "BTCUSDT"
    return contract.replace("_", "")

def _side_from_size(size: Any) -> str:
    try:
        f = float(size)
    except Exception:
        return "long"
    return "long" if f >= 0 else "short"

def _num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

# =============== BALANCES ===============

def fetch_gate_spot_balances() -> List[Dict[str, Any]]:
    """
    GET /spot/accounts
    Devuelve [{currency, available, locked}]
    """
    print("🔍 DEBUG: Obteniendo balances SPOT de Gate.io...")
    data = _request("GET", "/spot/accounts")
    out = []
    total_spot = 0.0
    
    for row in data or []:
        available = _num(row.get("available"))
        locked = _num(row.get("locked"))
        total = available + locked
        total_spot += total
        
        balance_info = {
            "exchange": "gate",
            "account": "spot",
            "currency": row.get("currency"),
            "available": available,
            "locked": locked,
            "raw": row,
        }
        out.append(balance_info)
        
        # Debug individual para cada moneda con balance significativo
        if total > 0.01:  # Solo mostrar balances mayores a 0.01
            print(f"   💰 SPOT {row.get('currency')}: Available={available:.6f}, Locked={locked:.6f}, Total={total:.6f}")
    
    print(f"📊 DEBUG: Total SPOT balance: {total_spot:.6f} USDT equivalente")
    print(f"📦 DEBUG: Se encontraron {len(out)} monedas en spot")
    return out

def fetch_gate_margin_balances() -> List[Dict[str, Any]]:
    """
    GET /margin/accounts
    Devuelve por par: base/quote con available/locked/borrowed/interest + riesgo.
    """
    print("🔍 DEBUG: Obteniendo balances MARGIN de Gate.io...")
    data = _request("GET", "/margin/accounts")
    out: List[Dict[str, Any]] = []
    total_margin_net = 0.0
    
    for row in data or []:
        cp = row.get("currency_pair")
        risk = row.get("risk")
        print(f"   📈 DEBUG MARGIN: Procesando par {cp}, riesgo: {risk}")
        
        for leg_key in ("base", "quote"):
            leg = (row.get(leg_key) or {})
            available = _num(leg.get("available"))
            locked = _num(leg.get("locked"))
            borrowed = _num(leg.get("borrowed"))
            interest = _num(leg.get("interest"))
            net_value = available + locked - borrowed - interest
            
            margin_info = {
                "exchange": "gate",
                "account": "margin",
                "currency_pair": cp,
                "leg": leg_key,  # base|quote
                "currency": leg.get("currency"),
                "available": available,
                "locked": locked,
                "borrowed": borrowed,
                "interest": interest,
                "risk": _num(risk, None),
                "raw": row,
            }
            out.append(margin_info)
            
            # Debug para posiciones margin significativas
            if abs(net_value) > 0.01 or borrowed > 0.01:
                print(f"      📊 {leg_key.upper()} {leg.get('currency')}: Net={net_value:.6f}, Available={available:.6f}, Borrowed={borrowed:.6f}")
            
            total_margin_net += net_value
    
    print(f"📊 DEBUG: Total neto MARGIN: {total_margin_net:.6f}")
    print(f"📦 DEBUG: Se procesaron {len(out)} legs de margin")
    return out

def fetch_gate_futures_balances(settle: str = "usdt") -> Dict[str, Any]:
    """
    GET /futures/{settle}/accounts  (settle: usdt | btc)
    Devuelve el objeto de la cuenta de futuros (incluye balance, available, unrealized_pnl, etc.)
    """
    print(f"🔍 DEBUG: Obteniendo balances FUTURES ({settle.upper()}) de Gate.io...")
    settle = (settle or "usdt").lower()
    
    try:
        data = _request("GET", f"/futures/{settle}/accounts")
    except GateV4Error as e:
        if "USER_NOT_FOUND" in str(e) or "please transfer funds first" in str(e):
            print(f"⚠️  DEBUG: No hay cuenta FUTURES {settle.upper()} - {e}")
            # Retornar un objeto vacío pero con la estructura esperada
            return {
                "exchange": "gate",
                "account": f"futures_{settle}",
                "currency": settle.upper(),
                "available": 0.0,
                "balance": 0.0,
                "unrealized_pnl": 0.0,
                "funding_balance": 0.0,
                "position_margin": 0.0,
                "order_margin": 0.0,
                "raw": {"error": str(e)},
            }
        else:
            raise  # Relanzar otros errores
    
    # --- Lectura robusta de campos (según docs Gate) ---
    available = _num(data.get("available"))
    # Gate usa 'total', NO 'balance'
    total_api = data.get("total")
    # Margen en posiciones: usar position_margin y, en cross, caer a cross_imr
    position_margin = _num(data.get("position_margin") or data.get("cross_imr"))
    order_margin = _num(data.get("order_margin"))

    # Si 'total' no viene, lo recomponemos según la fórmula oficial
    if total_api is None or total_api == "":
        total = available + position_margin + order_margin
    else:
        total = _num(total_api)

    # PnL no realizado (Gate mezcla ortografías)
    unrealized_pnl = _num(data.get("unrealized_pnl") or data.get("unrealised_pnl"))

    # Campo opcional añadido en versiones recientes
    funding_balance = _num(data.get("funding_balance"))

    # (Opcional) 'equity' = total + unrealized_pnl
    equity = total + unrealized_pnl

    
    # print(f"💰 DEBUG FUTURES {settle.upper()}:")
    # print(f"   Balance total: {total:.6f}")
    # print(f"   Disponible: {available:.6f}")
    # print(f"   PnL No Realizado: {unrealized_pnl:.6f}")
    # print(f"   Balance Funding: {funding_balance:.6f}")
    # print(f"   Margin en Posiciones: {position_margin:.6f}")
    # print(f"   Margin en Órdenes: {order_margin:.6f}")
    
    result = {
        "exchange": "gate",
        "account": f"futures_{settle}",
        "currency": settle.upper(),
        "available": available,
        "balance": total,
        "unrealized_pnl": unrealized_pnl,
        "funding_balance": funding_balance,
        "position_margin": position_margin,
        "order_margin": order_margin,
        "equity": equity,
        "raw": data,
    }
    return result

# =============== POSICIONES ABIERTAS (Futures) ===============

def fetch_gate_open_positions(settle: str = "usdt") -> List[Dict[str, Any]]:
    """
    GET /futures/{settle}/positions?holding=true
    Devuelve lista de posiciones abiertas con esquema exacto requerido
    """
    print(f"🔍 DEBUG: Obteniendo POSICIONES ABIERTAS ({settle.upper()}) de Gate.io...")
    settle = (settle or "usdt").lower()
    params = {"holding": "true"}
    
    data = _request("GET", f"/futures/{settle}/positions", params=params)
    print(f"📦 DEBUG: Se recibieron {len(data or [])} posiciones abiertas")
        # 🧾 RAW dump (tal cual API) – limitado a 3 elementos para no inundar la consola
    try:
        for i, row in enumerate((data or [])[:3]):
            try:
                print(f"   🧾 RAW[{i}]: {json.dumps(row, ensure_ascii=False)}")
            except Exception:
                print(f"   🧾 RAW[{i}]: {row}")
    except Exception as e:
        print(f"   ⚠️ DEBUG RAW dump error: {e}")
        
        # ====fin del debug, borrar despues

    out: List[Dict[str, Any]] = []
    
    for p in data or []:
        contract = p.get("contract") or p.get("name") or ""
        symbol = _contract_to_symbol(contract).upper()  # ✅ MAYÚSCULAS
        size = _num(p.get("size"))
        side = p.get("side") or _side_from_size(size)
        entry = _num(p.get("entry_price"))
        mark = _num(p.get("mark_price"))
        # Preferimos el campo oficial de Gate 'liq_price'; si falta, caemos a 'liquidation_price'
        liq_raw = p.get("liq_price")
        if liq_raw in (None, "", "0", 0):
            liq_raw = p.get("liquidation_price")
        
        # Siempre devolver número para que el front pueda hacer .toFixed(4)
        liquidation_price = _num(liq_raw, 0.0)


        # Notional: si Gate no lo da, estimamos con |size|*mark
        notional = abs(_num(p.get("value"), 0.0)) if p.get("value") is not None else abs(size * mark)
        unreal = _num(p.get("unrealised_pnl") or p.get("unrealized_pnl"))
        realized = _num(p.get("realised_pnl") or p.get("realized_pnl"))
        pnl_fee = _num(p.get("pnl_fee"))      # comisiones acumuladas
        pnl_fund = _num(p.get("pnl_fund"))    # funding neto acumulado

        # ✅ ESQUEMA EXACTO REQUERIDO
        position = {
            "exchange": "gate",
            "symbol": symbol,                    # ✅ MAYÚSCULAS
            "side": side.lower(),                # ✅ "long" | "short"
            "size": abs(size),                   # ✅ float positivo
            "entry_price": entry,                # ✅ float
            "mark_price": mark,                  # ✅ float
            "liquidation_price": liquidation_price,            # ✅ float | None
            "notional": notional,                # ✅ float
            "unrealized_pnl": unreal,            # ✅ float
            "fee": pnl_fee,                      # ✅ float (negativo si coste)
            "funding_fee": pnl_fund,             # ✅ float neto funding
            "realized_pnl": realized             # ✅ float
        }
        print(f"🧪 GATE {symbol} side={side} entry={entry} mark={mark} liq={liquidation_price}")

        out.append(position)
            
        
    print(f"✅ DEBUG: {len(out)} posiciones abiertas normalizadas")
    return out
# =============== FUNDING (cuenta) ===============

def fetch_gate_funding_fees(settle: str = "usdt", start_time: Optional[int] = None, end_time: Optional[int] = None, limit: int = 200) -> List[Dict[str, Any]]:
    """
    GET /futures/{settle}/account_book?type=fund
    Devuelve funding cobrado/pagado por contrato en tu cuenta.
    """

    settle = (settle or "usdt").lower()
    params: Dict[str, Any] = {"type": "fund", "limit": max(1, min(int(limit), 500))}
    if start_time:
        params["from"] = int(start_time)
        print(f"   ⏰ Desde: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    if end_time:
        params["to"] = int(end_time)
        print(f"   ⏰ Hasta: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")

    rows = _request("GET", f"/futures/{settle}/account_book", params=params) or []

    
    asset = settle.upper()
    out: List[Dict[str, Any]] = []
    total_funding = 0.0
    funding_by_symbol = defaultdict(float)
    
    for r in rows:
        ts = r.get("time")
        if isinstance(ts, (int, float, str)):
            try:
                ts_ms = int(float(ts) * 1000.0)
            except Exception:
                ts_ms = None
        else:
            ts_ms = None
            
        contract = r.get("contract") or ""
        symbol = _contract_to_symbol(contract)
        income = _num(r.get("change"))
        
        funding_info = {
            "exchange": "gate",
            "symbol": symbol,
            "income": income,  # + cobro, − pago
            "asset": asset,
            "timestamp": ts_ms,
            "type": "FUNDING_FEE",
            "raw": r,
        }
        out.append(funding_info)
        
        total_funding += income
        funding_by_symbol[symbol] += income
        
        # Debug para cada fee de funding
        if abs(income) > 0.0001:  # Solo mostrar funding significativo
            time_str = datetime.fromtimestamp(ts_ms/1000).strftime('%Y-%m-%d %H:%M:%S') if ts_ms else "N/A"
            # print(f"   💸 FUNDING {symbol}: {income:.6f} {asset} at {time_str}")

    # print(f"💰 DEBUG RESUMEN FUNDING:")
    # print(f"   Total funding neto: {total_funding:.6f} {asset}")
    for symbol, amount in funding_by_symbol.items():
        if abs(amount) > 0.0001:
            print(f"   {symbol}: {amount:.6f} {asset}")
    
    return out




# =============== POSICIONES CERRADAS (Futures) ===============



def fetch_gate_closed_positions(settle: str = "usdt", days_back: int = 30) -> List[Dict[str, Any]]:
    """
    Obtiene posiciones cerradas con esquema exacto requerido
    """
    print(f"🔍 DEBUG: Obteniendo POSICIONES CERRADAS ({settle.upper()})...")
    
    settle = (settle or "usdt").lower()
    end_time = int(time.time())
    start_time = end_time - (days_back * 24 * 60 * 60)
    
    #print(f"   ⏰ Rango de tiempo: {datetime.fromtimestamp(start_time)} a {datetime.fromtimestamp(end_time)}")
    
    # Usar endpoint de position_close que da mejor información
    params = {
        "limit": 1000,
        "from": start_time,
        "to": end_time,
    }
    
    try:
        print(f"   🌐 Llamando endpoint: /futures/{settle}/position_close")
        position_history = _request("GET", f"/futures/{settle}/position_close", params=params)
        #print(f"📦 DEBUG: Respuesta cruda de position_close: {position_history}")
        
        if not position_history:
            print("   ⚠️  El endpoint position_close devolvió una lista vacía o None")
            return []
        
        print(f"📦 DEBUG: Se recibieron {len(position_history)} registros de posiciones cerradas")
        
        closed_positions = []
        
        for i, pos in enumerate(position_history):
            # print(f"   🔍 Procesando posición {i+1}: {pos}")
            closed_pos = _normalize_gate_closed_position(pos, settle)
            if closed_pos:
                closed_positions.append(closed_pos)
                print(f"      ✅ Normalizada: {closed_pos['symbol']} - PnL: {closed_pos['realized_pnl']}")
            else:
                print(f"      ❌ No se pudo normalizar posición {i+1}")
        
        print(f"✅ DEBUG: {len(closed_positions)} posiciones cerradas normalizadas")
        return closed_positions
        
    except Exception as e:
        print(f"❌ ERROR obteniendo posiciones cerradas: {e}")
        import traceback
        traceback.print_exc()
        return []
    
#=========== funcion para cerrar sin duplicar

def save_gate_closed_positions(db_path: str = "portfolio.db", days_back: int = 30) -> int:
    """
    Guarda posiciones cerradas de Gate.io en SQLite evitando duplicados por (exchange, symbol, close_time).
    """
    print("💾 Guardando posiciones cerradas de Gate.io en portfolio.db")

    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return 0

    closed_positions = fetch_gate_closed_positions(days_back=days_back)
    if not closed_positions:
        print("⚠️ No se obtuvieron posiciones cerradas de Gate.io.")
        return 0

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    saved = 0
    skipped = 0

    for pos in closed_positions:
        try:
            # Deduplicación por (exchange, symbol, close_time) — igual que en BingX/KuCoin
            cur.execute("""
                SELECT COUNT(*) FROM closed_positions
                WHERE exchange = ? AND symbol = ? AND close_time = ?
            """, (pos["exchange"], pos["symbol"], pos["close_time"]))
            if cur.fetchone()[0]:
                skipped += 1
                continue

            # Mapear a los nombres que espera save_closed_position / DB
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
                "funding_total": pos.get("funding_fee", 0),  # funding_fee → funding_total
                "pnl": pos.get("pnl"),  # PnL solo de precio
                "fee_total": pos.get("fees", 0),             # fees → fee_total
                "notional": pos.get("notional", 0),
                "leverage": pos.get("leverage"),
                "liquidation_price": pos.get("liquidation_price"),
            }

            save_closed_position(position_data)
            saved += 1

        except Exception as e:
            print(f"⚠️ Error guardando posición {pos.get('symbol')}: {e}")

    conn.close()
    print(f"✅ Gate guardadas: {saved} | omitidas (duplicadas): {skipped}")
    return saved

def _normalize_gate_closed_position(pos_data: Dict, settle: str) -> Optional[Dict[str, Any]]:
    """
    Normaliza posición cerrada al esquema exacto requerido
    """
    try:
        # print(f"      🛠️  Normalizando posición: {pos_data}")
        
        contract = pos_data.get("contract", "")
        symbol = _contract_to_symbol(contract).upper()  # ✅ MAYÚSCULAS
        size = _num(pos_data.get("accum_size") or pos_data.get("max_size", 0))
        print(f"         📏 Size (from accum_size/max_size): {size}")
        pnl_price_only  = _num(pos_data.get("pnl_pnl", 0))
        

        
        # ✅ CORREGIDO: Determinar side basado en el signo de size o del campo side
        side_from_data = pos_data.get("side", "").lower()
        if side_from_data in ["long", "short"]:
            side = side_from_data
        else:
            side = "long" if size >= 0 else "short"
        
        print(f"         🎯 Side: {side} (from data: {side_from_data})")
        
        # ✅ CORREGIDO: Usar los precios correctos - long_price para long, short_price para short
        if side == "long":
            entry_price = _num(pos_data.get("long_price", 0))
            close_price = _num(pos_data.get("short_price", 0))
        else:  # short
            entry_price = _num(pos_data.get("short_price", 0))
            close_price = _num(pos_data.get("long_price", 0))
        
        print(f"         💰 Entry: {entry_price}, Close: {close_price}")
        
        # ✅ CORREGIDO: Usar accum_size o max_size en lugar de size
        size = _num(pos_data.get("accum_size") or pos_data.get("max_size", 0))
        print(f"         📏 Size (from accum_size/max_size): {size}")
        
        if abs(size) < 0.000001:  # Posición muy pequeña, ignorar
            print(f"         ⚠️  Size demasiado pequeño, ignorando")
            return None
        diff = abs(close_price - entry_price)
        if diff > 0 and abs(pnl_price_only) > 0:
            size_from_pnl = abs(pnl_price_only) / diff
            if size <= 0 or abs(size_from_pnl - size) / max(1.0, abs(size)) > 0.05:
                print(f"         🔧 Size fix: {size} → {size_from_pnl:.0f} (from price_pnl)")
                size = round(size_from_pnl)
        
        # ✅ CORREGIDO: Usar pnl_pnl para realized_pnl, pnl_fee para fee, pnl_fund para funding
        realized_pnl = _num(pos_data.get("pnl", 0))

        fee = _num(pos_data.get("pnl_fee", 0))
        funding_fee = _num(pos_data.get("pnl_fund", 0))
        # Comprobación: pnl_price_only + fee + funding_fee ≈ realized_pnl
        if abs((pnl_price_only + fee + funding_fee) - realized_pnl) > 1e-6:
            print(f"⚠️ Descuadre PnL en {symbol}: "
                  f"price={pnl_price_only}, fee={fee}, funding={funding_fee}, "
                  f"realized={realized_pnl}")
            
        # ✅ CORREGIDO: Usar accum_size o max_size en lugar de size

        if abs(size) < 0.000001:  # Posición muy pequeña, ignorar
            print(f"         ⚠️  Size demasiado pequeño, ignorando")
            return None
        diff = abs(close_price - entry_price)
        if diff > 0 and abs(pnl_price_only) > 0:
            size_from_pnl = abs(pnl_price_only) / diff
            if size <= 0 or abs(size_from_pnl - size) / max(1.0, abs(size)) > 0.05:
                print(f"         🔧 Size fix: {size} → {size_from_pnl:.0f} (from price_pnl)")
                size = round(size_from_pnl)
        
        print(f"         📊 PnL: {realized_pnl}, Fee: {fee}, Funding: {funding_fee}")
        # DEBUG: Mostrar los valores crudos para diagnóstico
        print(f"         🔍 VALORES PnL CRUDOS:")
        print(f"           pnl (realized_pnl): {pos_data.get('pnl')}")
        print(f"           pnl_pnl (price_pnl): {pos_data.get('pnl_pnl')}")
        print(f"           pnl_fee (fee): {pos_data.get('pnl_fee')}")
        print(f"           pnl_fund (funding): {pos_data.get('pnl_fund')}")
               
        # Notional estimado
        notional_entry = abs(size) * entry_price
        notional = abs(size) * close_price if close_price > 0 else abs(size) * entry_price
        
        # Timestamps - convertir a SEGUNDOS (no ms)
        close_time = pos_data.get("time")
        open_time = pos_data.get("first_open_time")
        
        print(f"         ⏰ Open time raw: {open_time}, Close time raw: {close_time}")
        
        if close_time:
            try:
                close_time_seconds = int(float(close_time))  # ✅ SEGUNDOS
            except:
                close_time_seconds = None
        else:
            close_time_seconds = None
            
        if open_time:
            try:
                open_time_seconds = int(float(open_time))  # ✅ SEGUNDOS
            except:
                open_time_seconds = None
        else:
            open_time_seconds = None
            
        print(f"         ⏰ Open time sec: {open_time_seconds}, Close time sec: {close_time_seconds}")
        
        # Leverage y liquidation price
        leverage = _num(pos_data.get("leverage"), None)
        liquidation_price = None  # Gate.io no proporciona liquidation_price en position_close
        initial_margin = (notional_entry / leverage) if leverage else None
        
        
        # ✅ ESQUEMA EXACTO REQUERIDO PARA POSICIONES CERRADAS
        closed_position = {
            "exchange": "gate",
            "symbol": symbol,                    # ✅ MAYÚSCULAS
            "side": side,                       # ✅ "long" | "short"
            "size": abs(size),                  # ✅ float
            "entry_price": entry_price,         # ✅ float
            "close_price": close_price,         # ✅ float
            "notional": notional_entry,               # ✅ float
            "fees": fee,                        # ✅ float (negativo si coste)
            "funding_fee": funding_fee,         # ✅ float neto funding
            "realized_pnl": realized_pnl,       # ✅ float
            "pnl": pnl_price_only,                        # ✅ opcional
            "open_time": open_time_seconds,     # ✅ SEGUNDOS epoch
            "close_time": close_time_seconds,   # ✅ SEGUNDOS epoch
            "initial_margin": initial_margin,
            "leverage": leverage,               # ✅ opcional
            "liquidation_price": liquidation_price  # ✅ opcional
        }
        
        print(f"         ✅ Posición normalizada: {closed_position['symbol']} - Size: {abs(size)}")
        return closed_position
        
    except Exception as e:
        print(f"❌ ERROR normalizando posición cerrada: {e}")
        import traceback
        traceback.print_exc()
        return None

# funcion original que funciona salvo el pnl percent.

# def _normalize_gate_closed_position(pos_data: Dict, settle: str) -> Optional[Dict[str, Any]]:
#     """
#     Normaliza posición cerrada al esquema exacto requerido
#     """
#     try:
#         # print(f"      🛠️  Normalizando posición: {pos_data}")
        
#         contract = pos_data.get("contract", "")
#         symbol = _contract_to_symbol(contract).upper()  # ✅ MAYÚSCULAS
        
#         # ✅ CORREGIDO: Usar accum_size o max_size en lugar de size
#         size = _num(pos_data.get("accum_size") or pos_data.get("max_size", 0))
#         print(f"         📏 Size (from accum_size/max_size): {size}")
        
#         if abs(size) < 0.000001:  # Posición muy pequeña, ignorar
#             print(f"         ⚠️  Size demasiado pequeño, ignorando")
#             return None
        
#         # ✅ CORREGIDO: Determinar side basado en el signo de size o del campo side
#         side_from_data = pos_data.get("side", "").lower()
#         if side_from_data in ["long", "short"]:
#             side = side_from_data
#         else:
#             side = "long" if size >= 0 else "short"
        
#         print(f"         🎯 Side: {side} (from data: {side_from_data})")
        
#         # ✅ CORREGIDO: Usar los precios correctos - long_price para long, short_price para short
#         if side == "long":
#             entry_price = _num(pos_data.get("long_price", 0))
#             close_price = _num(pos_data.get("short_price", 0))
#         else:  # short
#             entry_price = _num(pos_data.get("short_price", 0))
#             close_price = _num(pos_data.get("long_price", 0))
        
#         print(f"         💰 Entry: {entry_price}, Close: {close_price}")
        
#         # ✅ CORREGIDO: Usar pnl_pnl para realized_pnl, pnl_fee para fee, pnl_fund para funding
#         realized_pnl = _num(pos_data.get("pnl", 0))
#         pnl_price_only  = _num(pos_data.get("pnl_pnl", 0))
#         fee = _num(pos_data.get("pnl_fee", 0))
#         funding_fee = _num(pos_data.get("pnl_fund", 0))
#         # Comprobación: pnl_price_only + fee + funding_fee ≈ realized_pnl
#         if abs((pnl_price_only + fee + funding_fee) - realized_pnl) > 1e-6:
#             print(f"⚠️ Descuadre PnL en {symbol}: "
#                   f"price={pnl_price_only}, fee={fee}, funding={funding_fee}, "
#                   f"realized={realized_pnl}")
        
#         print(f"         📊 PnL: {realized_pnl}, Fee: {fee}, Funding: {funding_fee}")
#         # DEBUG: Mostrar los valores crudos para diagnóstico
#         print(f"         🔍 VALORES PnL CRUDOS:")
#         print(f"           pnl (realized_pnl): {pos_data.get('pnl')}")
#         print(f"           pnl_pnl (price_pnl): {pos_data.get('pnl_pnl')}")
#         print(f"           pnl_fee (fee): {pos_data.get('pnl_fee')}")
#         print(f"           pnl_fund (funding): {pos_data.get('pnl_fund')}")
               
#         # Notional estimado
#         notional = abs(size) * close_price if close_price > 0 else abs(size) * entry_price
        
#         # Timestamps - convertir a SEGUNDOS (no ms)
#         close_time = pos_data.get("time")
#         open_time = pos_data.get("first_open_time")
        
#         print(f"         ⏰ Open time raw: {open_time}, Close time raw: {close_time}")
        
#         if close_time:
#             try:
#                 close_time_seconds = int(float(close_time))  # ✅ SEGUNDOS
#             except:
#                 close_time_seconds = None
#         else:
#             close_time_seconds = None
            
#         if open_time:
#             try:
#                 open_time_seconds = int(float(open_time))  # ✅ SEGUNDOS
#             except:
#                 open_time_seconds = None
#         else:
#             open_time_seconds = None
            
#         print(f"         ⏰ Open time sec: {open_time_seconds}, Close time sec: {close_time_seconds}")
        
#         # Leverage y liquidation price
#         leverage = _num(pos_data.get("leverage"), None)
#         liquidation_price = None  # Gate.io no proporciona liquidation_price en position_close
        
        
#         # ✅ ESQUEMA EXACTO REQUERIDO PARA POSICIONES CERRADAS
#         closed_position = {
#             "exchange": "gate",
#             "symbol": symbol,                    # ✅ MAYÚSCULAS
#             "side": side,                       # ✅ "long" | "short"
#             "size": abs(size),                  # ✅ float
#             "entry_price": entry_price,         # ✅ float
#             "close_price": close_price,         # ✅ float
#             "notional": notional,               # ✅ float
#             "fees": fee,                        # ✅ float (negativo si coste)
#             "funding_fee": funding_fee,         # ✅ float neto funding
#             "realized_pnl": realized_pnl,       # ✅ float
#             "pnl": pnl_price_only,                        # ✅ opcional
#             "open_time": open_time_seconds,     # ✅ SEGUNDOS epoch
#             "close_time": close_time_seconds,   # ✅ SEGUNDOS epoch
#             "leverage": leverage,               # ✅ opcional
#             "liquidation_price": liquidation_price  # ✅ opcional
#         }
        
#         print(f"         ✅ Posición normalizada: {closed_position['symbol']} - Size: {abs(size)}")
#         return closed_position
        
#     except Exception as e:
#         print(f"❌ ERROR normalizando posición cerrada: {e}")
#         import traceback
#         traceback.print_exc()
#         return None
    
    

# =============== Ejemplos de orquestación opcional ===============

def fetch_gate_all_balances(settles: Tuple[str, ...] = ("usdt", "btc")) -> Dict[str, Any]:
    """
    Devuelve un dict con balances de spot, margin y futures.
    """
    print("🚀 DEBUG: Ejecutando fetch_gate_all_balances()")
    print("=" * 60)
    
    # Filtrar settles que realmente existen
    valid_futures = []
    for s in settles:
        try:
            balance = fetch_gate_futures_balances(s)
            valid_futures.append(balance)
        except Exception as e:
            print(f"⚠️  Saltando FUTURES {s.upper()}: {e}")
            continue
    
    result = {
        "spot": fetch_gate_spot_balances(),
        "margin": fetch_gate_margin_balances(),
        "futures": valid_futures,  # Solo las que funcionaron
    }
    
    print("=" * 60)
    print("✅ DEBUG: fetch_gate_all_balances() completado")
    return result

def fetch_gate_all_positions_and_funding(settle: str = "usdt") -> Dict[str, Any]:
    """
    Obtiene posiciones abiertas y funding fees
    """
    print("🚀 DEBUG: Ejecutando fetch_gate_all_positions_and_funding()")
    print("=" * 60)
    
    result = {
        "open_positions": fetch_gate_open_positions(settle=settle),
        "funding_fees": fetch_gate_funding_fees(settle=settle, limit=200),
    }
    
    print("=" * 60)
    print("✅ DEBUG: fetch_gate_all_positions_and_funding() completado")
    return result

# # Función de prueba para ejecutar todos los debugs
# def debug_all_gate_data():
#     """
#     Función para probar y ver todos los datos importantes de Gate.io
#     """
#     print("🔬 INICIANDO DEBUG COMPLETO DE GATE.IO")
#     print("=" * 80)
    
#     try:
#         # 1. Balances - usar solo USDT para futures
#         print("\n1️⃣  BALANCES:")
#         print("-" * 40)
#         all_balances = fetch_gate_all_balances(settles=("usdt",))  # Solo USDT
        
#         # 2. Posiciones y Funding - solo USDT
#         print("\n2️⃣  POSICIONES Y FUNDING:")
#         print("-" * 40)
#         positions_funding = fetch_gate_all_positions_and_funding(settle="usdt")  # Solo USDT
        
#         # 3. Resumen final
#         print("\n🎯 RESUMEN FINAL:")
#         print("-" * 40)
        
#         # Calcular totales
#         total_spot = sum(bal['available'] + bal['locked'] for bal in all_balances['spot'] if bal['available'] + bal['locked'] > 0.01)
#         total_futures = sum(fut['balance'] for fut in all_balances['futures'])
#         total_positions = len(positions_funding['open_positions'])
#         total_funding_records = len(positions_funding['funding_fees'])
        
#         print(f"💰 Total Spot significativo: {total_spot:.6f} USDT")
#         print(f"📊 Total Futures balance: {total_futures:.6f} USDT")
#         print(f"📈 Posiciones abiertas: {total_positions}")
#         print(f"💸 Registros de funding: {total_funding_records}")
        
#     except Exception as e:
#         print(f"❌ ERROR durante el debug: {e}")
#         import traceback
#         traceback.print_exc()
    
#     print("=" * 80)
#     print("🏁 DEBUG COMPLETADO")

# # Ejecutar debug si se llama directamente
# if __name__ == "__main__":
#     debug_all_gate_data()