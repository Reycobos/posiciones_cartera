from flask import Flask, render_template, jsonify
import pandas as pd
import requests
import time
import hashlib
import statistics
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
from db_manager import init_db, save_closed_position


# Backpack
BACKPACK_API_KEY = os.getenv("BACKPACK_API_KEY")
BACKPACK_API_SECRET = os.getenv("BACKPACK_API_SECRET") # Debe ser la clave privada ED25519 en base64
BACKPACK_BASE_URL = "https://api.backpack.exchange"

# Headers para Backpack
UA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}



# -------------- Backpack signer (Ed25519) --------------
def _bp_sign_message(instruction: str, params: dict | None, ts_ms: int, window_ms: int = 5000) -> str:
    """
    Construye el string a firmar y devuelve la firma en Base64.
    """
    # 1) params ordenados -> querystring
    query = ""
    if params:
        from urllib.parse import urlencode
        sorted_items = sorted(params.items())
        query = urlencode(sorted_items, doseq=True)

    # 2) instrucción + timestamp & window
    if query:
        to_sign = f"instruction={instruction}&{query}&timestamp={ts_ms}&window={window_ms}"
    else:
        to_sign = f"instruction={instruction}&timestamp={ts_ms}&window={window_ms}"

    # 3) firmar con Ed25519
    try:
        seed32 = base64.b64decode(BACKPACK_API_SECRET)
    except Exception as e:
        raise RuntimeError(f"Invalid BACKPACK_API_SECRET format (expected Base64): {e}")
    if len(seed32) != 32:
        raise RuntimeError(f"BACKPACK_API_SECRET must decode to 32 bytes, got {len(seed32)}")

    signing_key = nacl.signing.SigningKey(seed32)
    sig_bytes = signing_key.sign(to_sign.encode("utf-8")).signature
    sig_b64 = base64.b64encode(sig_bytes).decode("ascii")
    return sig_b64



def backpack_signed_request(method: str, path: str, instruction: str, params: dict | None = None, body: dict | None = None):
    """
    Llama a un endpoint privado con firma Backpack.
    """
    if not BACKPACK_API_KEY or not BACKPACK_API_SECRET:
        raise RuntimeError("Missing BACKPACK_API_KEY / BACKPACK_API_SECRET")

    ts_ms = int(time.time() * 1000)
    window_ms = 5000

    # Los params que se firman son:
    sign_params = params if method.upper() == "GET" else (body or {})

    signature_b64 = _bp_sign_message(instruction, sign_params, ts_ms, window_ms)
    headers = {
        "X-API-KEY": BACKPACK_API_KEY,
        "X-SIGNATURE": signature_b64,
        "X-TIMESTAMP": str(ts_ms),
        "X-WINDOW": str(window_ms),
        "Content-Type": "application/json; charset=utf-8",
        **UA_HEADERS
    }

    url = f"{BACKPACK_BASE_URL}{path}"
    if method.upper() == "GET":
        r = requests.get(url, headers=headers, params=params or {}, timeout=30)
    elif method.upper() == "POST":
        r = requests.post(url, headers=headers, json=(body or {}), timeout=30)
    else:
        raise ValueError(f"Unsupported method: {method}")

    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f"[backpack] HTTP error {e.response.status_code} for {path}: {e.response.text}")
        raise
    return r.json()

def _normalize_symbol(sym: str) -> str:
    """Normaliza símbolos de Backpack"""
    if not isinstance(sym, str):
        return sym
    parts = sym.split("_")
    if len(parts) >= 2:
        return parts[0] + parts[1]
    return sym

# -------------- Backpackconfig--------------
def fetch_account_backpack():
    """
    Backpack account equity via Capital Collateral.
    GET /api/v1/capital/collateral (Instruction: collateralQuery)
    """
    try:
        data = backpack_signed_request(
            "GET", "/api/v1/capital/collateral", instruction="collateralQuery", params=None
        )
        acct = data if isinstance(data, dict) else (data.get("data") or {})
        if not acct:
            return None

        def f(k):
            try:
                return float(acct.get(k, 0))
            except Exception:
                return 0.0

        return {
            "exchange": "backpack",
            "equity": f("netEquity"),
            "balance": f("netEquity"),  # Usar equity como balance
            "unrealized_pnl": f("pnlUnrealized"),
            "initial_margin": f("imf")  # Initial Margin Fraction
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch Backpack account: {e}")
        return None



def fetch_positions_backpack():
    """
    GET /api/v1/position (Instruction: positionQuery)
    Devuelve posiciones abiertas de Backpack.
    """
    try:
        data = backpack_signed_request(
            "GET", "/api/v1/position", instruction="positionQuery", params=None
        )
        items = data if isinstance(data, list) else (data.get("data") or [])
        if not items:
            return []

        positions = []
        for item in items:
            try:
                net_quantity = float(item.get("netQuantity", 0))
                
                # Filtrar posiciones cerradas (amount = 0)
                if net_quantity == 0:
                    continue
                    
                entry_price = float(item.get("entryPrice", 0))
                mark_price = float(item.get("markPrice", 0))
                notional = float(item.get("netExposureNotional", 0))
                unrealized_pnl = float(item.get("pnlUnrealized", 0))
                cumulative_funding = float(item.get("cumulativeFundingPayment", 0))
                
                # ✅ CALCULAR UNREALIZED PNL MANUALMENTE PARA VERIFICAR
                # Unrealized PnL = (Mark Price - Entry Price) * Quantity
                calculated_unrealized = (mark_price - entry_price) * net_quantity
                
                positions.append({
                    "exchange": "backpack",
                    "symbol": _normalize_symbol(item.get("symbol", "")),
                    "side": "LONG" if net_quantity >= 0 else "SHORT",
                    "size": abs(net_quantity),
                    "quantity": abs(net_quantity),
                    "entry_price": entry_price,
                    "mark_price": mark_price,
                    "unrealized_pnl": calculated_unrealized,  # ← USAR EL CALCULADO
                    "funding_fee": cumulative_funding,
                    "realized_pnl": cumulative_funding,
                    "notional": notional,
                    "liquidation_price": float(item.get("estLiquidationPrice", 0))
                })
                
                print(f"[DEBUG] Backpack position: {item.get('symbol')}")
                print(f"  Quantity: {net_quantity}")
                print(f"  Entry: {entry_price}, Mark: {mark_price}")
                print(f"  API Unrealized: {unrealized_pnl}")
                print(f"  CALCULATED Unrealized: {calculated_unrealized}")
                print(f"  Funding: {cumulative_funding}")
                print(f"  Notional: {notional}")
                      
            except Exception as e:
                print(f"[WARNING] Error processing Backpack position: {e}")
                continue

        print(f"[DEBUG] Backpack positions: {len(positions)} found")
        return positions
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch Backpack positions: {e}")
        return []
    
    
def fetch_funding_backpack(limit=1000):
    """
    GET /wapi/v1/history/funding (Instruction: fundingHistoryQueryAll)
    Trae los funding payments del usuario de Backpack.
    """
    try:
        params = {"limit": min(int(limit), 1000), "sortDirection": "Desc"}
        
        data = backpack_signed_request(
            "GET", "/wapi/v1/history/funding", instruction="fundingHistoryQueryAll", params=params
        )
        items = data if isinstance(data, list) else (data.get("data") or [])
        if not items:
            return []

        funding_payments = []
        for item in items:
            try:
                # ✅ PROBAR DIFERENTES CAMPOS POSIBLES
                amount = float(item.get("quantity", 0))

                # convertir timestamp ISO → epoch ms
                ts = item.get("intervalEndTimestamp", "") or item.get("timestamp", "")
                try:
                    # Manejar diferentes formatos de timestamp
                    if "T" in ts:
                        ts_ms = int(datetime.fromisoformat(ts.replace("Z", "")).timestamp() * 1000)
                    else:
                        ts_ms = int(ts) if ts else None
                except Exception:
                    ts_ms = None

                funding_payments.append({
                    "exchange": "backpack",
                    "symbol": _normalize_symbol(item.get("symbol", "")),
                    "income": amount,
                    "asset": "USDC",
                    "timestamp": ts_ms,
                    "funding_rate": float(item.get("fundingRate", 0)),
                    "type": "FUNDING_FEE"
                })
                
                # ✅ DEBUG: Mostrar cada registro de funding
                #print(f"[DEBUG] Backpack funding: {item.get('symbol')} = {amount}")
                
            except Exception as e:
                print(f"[WARNING] Error processing Backpack funding: {e}")
                print(f"[WARNING] Problematic item: {item}")
                continue

        #print(f"[DEBUG] Backpack funding: {len(funding_payments)} payments found, total: {sum(f['income'] for f in funding_payments):.4f}")
        return funding_payments
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch Backpack funding: {e}")
        return []

    


def _format_time(ts: str) -> str:
    """
    Convierte '2025-09-19T06:57:33.557' en '2025-09-19 06:57'
    """
    if not ts:
        return "-"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", ""))  # por si viene con Z
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts


#================ codigo de trades_processing    
    
def build_positions_from_trades(trades):
    """
    C posiciones cerradas a partir de trades.
    Cada trade debe tener: symbol, side, qty, price, commission, time.
    """
    positions = []
    grouped = defaultdict(list)

    for t in trades:
        grouped[t["symbol"]].append(t)

    for sym, sym_trades in grouped.items():
        sym_trades.sort(key=lambda x: x["time"])
        qty_net = 0.0
        entry_prices = []
        open_time = None
        fees_total = 0.0

        for t in sym_trades:
            side = 1 if t["side"].lower() == "buy" else -1
            qty = float(t["qty"]) * side
            price = float(t["price"])
            commission = float(t.get("commission", 0))
            fees_total += commission

            if qty_net == 0:
                open_time = t["time"]
                entry_prices = [price]

            qty_net += qty

            if abs(qty_net) < 1e-9:
                close_time = t["time"]
                close_price = price
                entry_price = statistics.mean(entry_prices)
                realized_pnl = sum(
                    (float(tr["price"]) - entry_price) * float(tr["qty"])
                    * (1 if tr["side"].lower() == "sell" else -1)
                    for tr in sym_trades if open_time <= tr["time"] <= close_time
                )

                positions.append({
                    "symbol": sym,
                    "side": "long" if qty > 0 else "short",
                    "size": abs(qty),
                    "entry_price": entry_price,
                    "close_price": close_price,
                    "open_time": open_time,
                    "close_time": close_time,
                    "realized_pnl": realized_pnl,
                    "fee_total": -fees_total
                })

                qty_net = 0.0
                fees_total = 0.0
                entry_prices = []
            else:
                entry_prices.append(price)

    return positions


def attach_funding_to_positions(positions, funding):
    """
    funding: lista con symbol, income, timestamp
    """
    for pos in positions:
        relevant = [
            f for f in funding
            if f["symbol"] == pos["symbol"]
            and pos["open_time"] <= f["timestamp"] <= pos["close_time"]
        ]
        pos["funding_total"] = sum(f["income"] for f in relevant)
    return positions


def process_closed_positions(exchange, trades, funding):
    """
    Calcula, ajusta funding y guarda en SQLite.
    """
    positions = build_positions_from_trades(trades)
    positions = attach_funding_to_positions(positions, funding)

    for pos in positions:
        pos["exchange"] = exchange
        save_closed_position(pos)

    print(f"✅ {exchange}: {len(positions)} posiciones cerradas guardadas.")
    
#/////// BackpackConfig//////// 

def _parse_ts_to_ms(ts):
    """Acepta ISO, epoch s/ms y devuelve epoch ms (o None)."""
    if not ts:
        return None
    try:
        if isinstance(ts, (int, float)):
            v = int(ts)
            return v if v > 10_000_000_000 else v * 1000  # ms si es grande
        ts_str = str(ts).replace("Z", "")
        dt = datetime.fromisoformat(ts_str)
        return int(dt.timestamp() * 1000)
    except Exception:
        try:
            v = int(ts)
            return v if v > 10_000_000_000 else v * 1000
        except Exception:
            return None

def _format_time(ts: str) -> str:
    """
    Convierte '2025-09-19T06:57:33.557' en '2025-09-19 06:57'
    """
    if not ts:
        return "-"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", ""))  # por si viene con Z
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts
    
def fetch_closed_positions_backpack(limit=1000, days=60, debug=False):
    """
    Reconstruye posiciones cerradas de Backpack (solo PERP/IPERP).
    - Size = Net máximo absoluto durante el ciclo completo
    - PnL calculado correctamente para posiciones escalonadas
    - Incluye funding payments reales
    """
    try:
        now_ms = int(time.time() * 1000)
        from_ms = now_ms - days * 24 * 60 * 60 * 1000

        path = "/wapi/v1/history/fills"
        instruction = "fillHistoryQueryAll"

        # OBTENER FUNDING PAYMENTS REALES
        if debug:
            print("🔍 Obteniendo funding payments de Backpack...")
        funding_payments = fetch_funding_backpack(limit=1000)
        
        # Crear índice de funding por símbolo y timestamp
        funding_by_symbol_time = {}
        for fp in funding_payments:
            symbol = fp["symbol"]
            timestamp = fp.get("timestamp")
            if symbol and timestamp:
                if symbol not in funding_by_symbol_time:
                    funding_by_symbol_time[symbol] = {}
                # Redondear timestamp a la hora más cercana (los funding suelen ser cada 1-8 horas)
                hour_key = timestamp // (3600 * 1000) * (3600 * 1000)
                funding_by_symbol_time[symbol][hour_key] = funding_by_symbol_time[symbol].get(hour_key, 0) + fp["income"]

        if debug:
            print(f"💰 Funding payments procesados: {sum(len(v) for v in funding_by_symbol_time.values())} registros")

        def _try_fetch(market_type: str | None):
            params = {
                "limit": min(int(limit), 1000),
                "sortDirection": "Asc",
                "from": from_ms,
                "to": now_ms,
            }
            if market_type:
                params["marketType"] = market_type
            return backpack_signed_request("GET", path, instruction, params=params)

        items = []
        # 1) Intento PERP
        try:
            data = _try_fetch("PERP")
            items += (data if isinstance(data, list) else (data.get("data") or []))
        except Exception as e1:
            if debug:
                print("[Backpack] PERP fetch failed:", e1)

        # 2) Intento IPERP
        try:
            data = _try_fetch("IPERP")
            items += (data if isinstance(data, list) else (data.get("data") or []))
        except Exception as e2:
            if debug:
                print("[Backpack] IPERP fetch failed:", e2)

        # 3) Fallback sin marketType
        if not items:
            if debug:
                print("[Backpack] Fallback sin marketType")
            data = _try_fetch(None)
            items = data if isinstance(data, list) else (data.get("data") or [])
            items = [it for it in items if "PERP" in (it.get("symbol") or "").upper()]

        if not items:
            if debug:
                print("[Backpack] No fills PERP/IPERP.")
            return []

        # --- Normalización de fills
        fills = []
        for f in items:
            try:
                sym = _normalize_symbol(f.get("symbol", ""))
                side = (f.get("side") or "").lower()
                qty = float(f.get("quantity", 0))
                price = float(f.get("price", 0))
                fee = float(f.get("fee") or f.get("feeAmount") or 0.0)
                ts = _parse_ts_to_ms(f.get("timestamp"))
                if ts is None:
                    continue
                signed = qty if side in ("bid", "buy") else -qty
                fills.append({
                    "symbol": sym, "side": side, "qty": qty, "price": price,
                    "fee": fee, "signed": signed, "ts": ts
                })
            except Exception as e:
                if debug:
                    print("[WARN] bad fill:", f, e)
                continue

        if not fills:
            if debug:
                print("[Backpack] No normalized fills.")
            return []

        fills.sort(key=lambda x: x["ts"])

        # --- RECONSTRUCCIÓN CON CÁLCULO CORRECTO DE PnL
        grouped = defaultdict(list)
        for f in fills:
            grouped[f["symbol"]].append(f)

        results = []
        
        # --- FIFO universal (long y short) sobre el bloque de trades ya neteado a 0
        from collections import deque
        def _fifo_realized_pnl(block_trades):
            """
            block_trades: lista de dicts con campos:
              - 'signed' (qty con signo: +buy, -sell)
              - 'qty'    (qty absoluta, >0)
              - 'price'
            Devuelve PnL por precio (sin fees/funding) usando FIFO real:
             - Si hay lot largo abierto (qty>0) y llega un SELL, cierra contra ese lot.
             - Si hay lot corto abierto (qty<0) y llega un BUY, cierra contra ese lot.
            """
            lots = deque()  # cada lot: {'qty': signed_qty, 'price': entry_price}
            realized = 0.0
            eps = 1e-12
        
            for t in block_trades:
                q_signed = t["signed"]
                p = t["price"]
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

        for sym, fs in grouped.items():
            net = 0.0
            max_net_abs = 0.0
            block = []
            open_ms = None

            for f in fs:
                if open_ms is None and abs(f["signed"]) > 1e-9:
                    open_ms = f["ts"]

                net += f["signed"]
                block.append(f)
                
                current_net_abs = abs(net)
                if current_net_abs > max_net_abs:
                    max_net_abs = current_net_abs

                if abs(net) < 1e-9 and len(block) > 1 and max_net_abs > 1e-9:
                    close_ms = f["ts"]

                    total_buy = sum(x["qty"] for x in block if x["signed"] > 0)
                    total_sell = sum(x["qty"] for x in block if x["signed"] < 0)
                    
                    # Determinar side basado en el primer trade significativo
                    first_trade = None
                    for trade in block:
                        if abs(trade["signed"]) > 1e-9:
                            first_trade = trade
                            break

                    if first_trade:
                        side = "long" if first_trade["signed"] > 0 else "short"
                    else:
                        side = "long"  # Por defecto

                    # Mantener el cálculo de precios según el side determinado
                    if side == "long":
                        entry_trades = [x for x in block if x["signed"] > 0]
                        if entry_trades:
                            entry_avg = sum(x["qty"] * x["price"] for x in entry_trades) / total_buy
                        else:
                            entry_avg = 0.0
                        close_trades = [x for x in block if x["signed"] < 0]
                        if close_trades:
                            close_avg = sum(abs(x["signed"]) * x["price"] for x in close_trades) / total_sell
                        else:
                            close_avg = 0.0
                    else:  # short
                        entry_trades = [x for x in block if x["signed"] < 0]
                        if entry_trades:
                            entry_avg = sum(abs(x["signed"]) * x["price"] for x in entry_trades) / total_sell
                        else:
                            entry_avg = 0.0
                        close_trades = [x for x in block if x["signed"] > 0]
                        if close_trades:
                            close_avg = sum(x["qty"] * x["price"] for x in close_trades) / total_buy
                        else:
                            close_avg = 0.0

                    # Size = Net máximo absoluto durante el ciclo
                    size = max_net_abs

                    fees = sum(x["fee"] for x in block)

                    # 🔧 CALCULAR PnL CORRECTAMENTE USANDO MÉTODO FIFO
                    # --- PnL por precio con FIFO real (long + short)
                    price_pnl = _fifo_realized_pnl(block)

                    # Fees del bloque
                    fees = sum(x["fee"] for x in block)

                    # PnL neto de fees (sin funding)
                    realized_pnl = price_pnl - fees

                    # Funding real en la ventana
                    funding_fee = 0.0
                    if sym in funding_by_symbol_time:
                        for hour_key, funding_amount in funding_by_symbol_time[sym].items():
                            if open_ms <= hour_key <= close_ms:
                                funding_fee += funding_amount
                                if debug:
                                    print(f"       Funding: +{funding_amount:.6f} at {datetime.fromtimestamp(hour_key/1000)}")

                    realized_pnl_with_funding = realized_pnl + funding_fee


                    results.append({
                        "exchange": "backpack",
                        "symbol": sym,
                        "side": side,
                        "size": size,
                        "entry_price": entry_avg,
                        "close_price": close_avg,
                        "notional": entry_avg * size,
                        "price_pnl": price_pnl,
                        "fees": fees,
                        "funding_fee": funding_fee,  # ✅ Funding real
                        "realized_pnl": realized_pnl_with_funding,  # ✅ PnL incluyendo funding
                        "open_date": datetime.fromtimestamp(open_ms / 1000).strftime("%Y-%m-%d %H:%M"),
                        "close_date": datetime.fromtimestamp(close_ms / 1000).strftime("%Y-%m-%d %H:%M"),
                    })

                    if debug:
                        print(f"[BP] {sym} {side.upper()} size={size:.4f}")
                        print(f"     entry={entry_avg:.6f} close={close_avg:.6f}")
                        print(f"     fees={fees:.4f} funding={funding_fee:.4f} pnl={realized_pnl_with_funding:.4f}")
                        print(f"     net_max={max_net_abs:.2f}, trades={len(block)}")
                        print(f"     PnL breakdown: price={realized_pnl:.4f} + funding={funding_fee:.4f}")

                    # Reset
                    block = []
                    net = 0.0
                    max_net_abs = 0.0
                    open_ms = None

        if debug:
            print(f"✅ Backpack closed positions: {len(results)}")
            # Resumen de funding por posición
            total_funding = sum(pos["funding_fee"] for pos in results)
            print(f"💰 Total funding en posiciones cerradas: {total_funding:.6f}")

        return results

    except Exception as e:
        print(f"❌ Error al reconstruir closed positions Backpack: {e}")
        import traceback
        traceback.print_exc()
        return []


def save_backpack_closed_positions(db_path="portfolio.db"):
    """
    Obtiene las posiciones cerradas ya reconstruidas desde Backpack (v7),
    imprime un debug detallado y guarda en la DB (tabla closed_positions).
    """
    import os
    import sqlite3
    from datetime import datetime

    # --- helpers --------------------------
    def _as_sec(x):
        """Convierte timestamps a segundos. Acepta ms (int/float grande), segundos o ISO string."""
        if x is None:
            return None
        if isinstance(x, (int, float)):
            # heurística: si es > 10^11 asumimos milisegundos
            return int(round(x / 1000)) if x > 1e11 else int(x)
        if isinstance(x, str):
            # intenta ISO (con o sin 'Z')
            try:
                return int(datetime.fromisoformat(x.replace("Z", "")).timestamp())
            except Exception:
                return None
        return None

    def _simple_price_pnl(side, entry, close, size):
        """PnL por precio con promedio simple (no FIFO) para comparar rápidamente."""
        if side and side.lower() == "short":
            return (entry - close) * size
        return (close - entry) * size

    # --------------------------------------

    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return

    closed_positions = fetch_closed_positions_backpack(limit=1000, days=60, debug=False)
    if not closed_positions:
        print("⚠️ No closed positions returned from Backpack.")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    saved_count = 0
    skipped = 0

    try:
        for pos in closed_positions:
            try:
                # Campos base con tolerancia a nombres
                exchange = pos.get("exchange", "backpack")
                symbol   = pos.get("symbol")
                side     = pos.get("side")
                size     = float(pos.get("size", 0.0))
                entry    = float(pos.get("entry_price", 0.0))
                close    = float(pos.get("close_price", 0.0))

                # Tiempos: v7 ya trae open_time/close_time (ms). Si no, usa open_date/close_date (ISO)
                open_ts  = _as_sec(pos.get("open_time", pos.get("open_date")))
                close_ts = _as_sec(pos.get("close_time", pos.get("close_date")))

                # Fees/funding con tolerancia
                fees_in      = pos.get("fees", pos.get("fee_total", 0.0)) or 0.0
                funding_in   = pos.get("funding_fee", pos.get("funding_total", 0.0)) or 0.0
                fees_abs     = float(fees_in)
                funding_tot  = float(funding_in)

                # En DB usamos fees NEGATIVOS (según tu esquema/logs)
                fee_total_for_db = -abs(fees_abs)

                # PnL: si v7 ya trae FIFO de precio (price_pnl) lo usamos; si no, simple
                price_pnl_fifo = pos.get("price_pnl")
                if price_pnl_fifo is None:
                    price_pnl_fifo = _simple_price_pnl(side, entry, close, size)
                price_pnl_fifo = float(price_pnl_fifo)

                # realized: preferimos el que venga; si no, precio - fees + funding
                realized_in = pos.get("realized_pnl")
                if realized_in is None:
                    realized_in = price_pnl_fifo + funding_tot + fee_total_for_db  # fee_total_for_db ya es negativo
                realized_in = float(realized_in)

                # notional: si no viene, estimamos
                notional = pos.get("notional")
                if notional is None:
                    notional = abs(size * entry)
                notional = float(notional)

                # Evitar duplicados exactos: misma exchange, símbolo y close_time (segundos)
                if close_ts is None:
                    # Si no hay close_ts, no podemos deduplicar: guardamos con aviso
                    print(f"⚠️ [{symbol}] close_time ausente; se guardará sin chequeo de duplicados.")
                else:
                    cur.execute("""
                        SELECT COUNT(*) FROM closed_positions
                        WHERE exchange = ? AND symbol = ? AND close_time = ?
                    """, (exchange, symbol, close_ts))
                    exists = cur.fetchone()[0]
                    if exists:
                        skipped += 1
                        continue

                # Debug comparativo FIFO vs simple
                simple_pnl = _simple_price_pnl(side, entry, close, size)
                print(f"🔎 [DEBUG closed_backpack_save] {symbol} {side} size={size:.4f}")
                print(f"    entry={entry:.6f} close={close:.6f}")
                print(f"    price_pnl(FIFO)={price_pnl_fifo:.4f}  price_pnl(simple)={simple_pnl:.4f}  Δ={price_pnl_fifo - simple_pnl:.4f}")
                print(f"    fees={fees_abs:.4f}  funding={funding_tot:.4f}  realized(FIFO)={realized_in:.4f}")
                print(f"    open={open_ts} close={close_ts}")

                # Payload final que se mandará a db_manager.save_closed_position
                record = {
                    "exchange": exchange,
                    "symbol": symbol,
                    "side": side,
                    "size": size,
                    "entry_price": entry,
                    "close_price": close,
                    "open_time": open_ts,
                    "close_time": close_ts,
                    # Usamos FIFO de precio como 'pnl' y realized FIFO como 'realized_pnl'
                    "pnl": price_pnl_fifo,
                    "realized_pnl": realized_in,
                    "funding_total": funding_tot,
                    "fee_total": fee_total_for_db,
                    "notional": notional,
                    "leverage": None,
                    "liquidation_price": None,
                    # No permitimos que el DB reconstruya size/PNL a partir de medias:
                    "_lock_size": True,
                }
                print("🔎 [DEBUG closed_backpack_save] payload:", record)

                # Guardar
                save_closed_position(record)
                saved_count += 1

            except Exception as e:
                print(f"⚠️ Error saving Backpack position {pos.get('symbol')}: {e}")
                continue

    finally:
        conn.close()

    print(f"✅ Guardadas {saved_count} posiciones cerradas de Backpack (omitidas {skipped} duplicadas).")

    
# =============== FUNCIONES DE DEBUG PARA SPYDER ===============
def debug_backpack_position_reconstruction(symbol_filter=None, days=60, debug=True):
    """
    Función de debug para analizar en detalle la reconstrucción de posiciones cerradas de Backpack.
    Ejecutar solo en Spyder para diagnóstico.
    """
    print("🔍 INICIANDO DEBUG DETALLADO DE BACKPACK POSITIONS")
    print("=" * 80)
    
    try:
        # Obtener funding payments
        print("\n📊 OBTENIENDO FUNDING PAYMENTS...")
        funding_payments = fetch_funding_backpack(limit=1000)
        print(f"📦 Total funding payments obtenidos: {len(funding_payments)}")
        
        # Crear índice de funding por símbolo
        funding_by_symbol = defaultdict(list)
        for fp in funding_payments:
            symbol = fp["symbol"]
            funding_by_symbol[symbol].append(fp)
        
        # Mostrar resumen de funding por símbolo
        for symbol, payments in funding_by_symbol.items():
            total_funding = sum(p["income"] for p in payments)
            print(f"   {symbol}: {len(payments)} payments, total: {total_funding:.6f}")
        
        # Obtener fills (trades)
        print("\n📊 OBTENIENDO TRADES (FILLS)...")
        now_ms = int(time.time() * 1000)
        from_ms = now_ms - days * 24 * 60 * 60 * 1000
        
        path = "/wapi/v1/history/fills"
        instruction = "fillHistoryQueryAll"
        
        params = {
            "limit": 1000,
            "sortDirection": "Desc",
            "from": from_ms,
            "to": now_ms,
        }
        
        data = backpack_signed_request("GET", path, instruction, params=params)
        items = data if isinstance(data, list) else (data.get("data") or [])
        items = [it for it in items if "PERP" in (it.get("symbol") or "").upper()]
        
        print(f"📦 Total trades obtenidos: {len(items)}")
        
        # Normalizar fills
        fills = []
        for f in items:
            try:
                sym = _normalize_symbol(f.get("symbol", ""))
                # Filtrar por símbolo si se especifica
                if symbol_filter and sym != symbol_filter:
                    continue
                    
                side = (f.get("side") or "").lower()
                qty = float(f.get("quantity", 0))
                price = float(f.get("price", 0))
                fee = float(f.get("fee") or f.get("feeAmount") or 0.0)
                ts = _parse_ts_to_ms(f.get("timestamp"))
                if ts is None:
                    continue
                signed = qty if side in ("bid", "buy") else -qty
                
                fill_data = {
                    "symbol": sym, "side": side, "qty": qty, "price": price,
                    "fee": fee, "signed": signed, "ts": ts,
                    "datetime": datetime.fromtimestamp(ts/1000).strftime("%Y-%m-%d %H:%M:%S")
                }
                fills.append(fill_data)
                
            except Exception as e:
                print(f"⚠️ Error procesando trade: {e}")
                continue
        
        if not fills:
            print("❌ No se encontraron trades para analizar")
            return
        
        # Agrupar por símbolo
        fills_by_symbol = defaultdict(list)
        for f in fills:
            fills_by_symbol[f["symbol"]].append(f)
        
        print(f"\n📊 TRADES POR SÍMBOLO:")
        for symbol, symbol_fills in fills_by_symbol.items():
            print(f"   {symbol}: {len(symbol_fills)} trades")
            
        ### parche para ver el nuevo codigo 
        
        # --- Helper: FIFO real con trazado (long & short)
        from collections import deque
        def _fifo_real_with_trace(block_trades):
            """
            Devuelve:
                realized_pnl: PnL por precio (sin fees/funding)
                lines: lista de strings para imprimir el detalle por trade
            """
            lots = deque()   # cada lot: {'qty': signed_qty, 'price': entry_price}
            realized = 0.0
            lines = []
            EPS = 1e-12
        
            def _net_and_avg():
                net = sum(l['qty'] for l in lots)
                if abs(net) < EPS:
                    return 0.0, None
                if net > 0:
                    tot = sum(max(0.0, l['qty']) * l['price'] for l in lots)
                    return net, tot / net
                else:
                    tot = sum((-min(0.0, l['qty'])) * l['price'] for l in lots)
                    return net, tot / (-net)
        
            for j, t in enumerate(block_trades, start=1):
                side = 'BUY' if t['signed'] > 0 else 'SELL'
                qty  = abs(t['signed'])
                px   = t['price']
                pnl_change = 0.0
                matches = []
        
                if t['signed'] > 0:  # BUY → cierra shorts primero
                    remaining = qty
                    while remaining > EPS and lots and lots[0]['qty'] < 0:
                        open_lot = lots[0]
                        match_qty = min(remaining, -open_lot['qty'])
                        pnl = (open_lot['price'] - px) * match_qty  # short: entry - exit
                        pnl_change += pnl
                        realized += pnl
                        open_lot['qty'] += match_qty  # menos negativo
                        remaining -= match_qty
                        matches.append(f"match short {match_qty:.4f} @{open_lot['price']:.4f} -> pnl {pnl:.4f}")
                        if open_lot['qty'] > -EPS:
                            lots.popleft()
                    if remaining > EPS:
                        lots.append({'qty': remaining, 'price': px})
                        matches.append(f"abre long {remaining:.4f}")
        
                else:  # SELL → cierra longs primero
                    remaining = qty
                    while remaining > EPS and lots and lots[0]['qty'] > 0:
                        open_lot = lots[0]
                        match_qty = min(remaining, open_lot['qty'])
                        pnl = (px - open_lot['price']) * match_qty   # long: exit - entry
                        pnl_change += pnl
                        realized += pnl
                        open_lot['qty'] -= match_qty
                        remaining -= match_qty
                        matches.append(f"match long {match_qty:.4f} @{open_lot['price']:.4f} -> pnl {pnl:.4f}")
                        if open_lot['qty'] < EPS:
                            lots.popleft()
                    if remaining > EPS:
                        lots.append({'qty': -remaining, 'price': px})
                        matches.append(f"abre short {remaining:.4f}")
        
                net, avg = _net_and_avg()
                if abs(net) < EPS:
                    pos_str = "FLAT"
                else:
                    pos_str = f"{'LONG' if net>0 else 'SHORT'} {abs(net):.4f} @ {avg:.4f}"
        
                lines.append(
                    f"      {j:2d}. {side:4s} {qty:8.4f} @ {px:8.4f} | "
                    f"{'; '.join(matches) if matches else '—'} | "
                    f"PnLΔ: {pnl_change:8.4f} | Pos: {pos_str}"
                )
        
            return realized, lines
        
        ###fin del  parche para ver el nuevo codigo 
        
        # Analizar cada símbolo
        for symbol, symbol_fills in fills_by_symbol.items():
            print(f"\n{'='*60}")
            print(f"🔍 ANALIZANDO SÍMBOLO: {symbol}")
            print(f"{'='*60}")
            
            # Ordenar trades por timestamp
            symbol_fills.sort(key=lambda x: x["ts"])
            
            # Mostrar todos los trades del símbolo
            print(f"\n📋 TODOS LOS TRADES DE {symbol}:")
            print(f"{'-'*40}")
            for i, trade in enumerate(symbol_fills):
                print(f"{i+1:2d}. {trade['datetime']} | {trade['side'].upper():4s} | "
                      f"Qty: {trade['qty']:8.4f} | Price: {trade['price']:8.4f} | "
                      f"Fee: {trade['fee']:6.4f} | Signed: {trade['signed']:8.4f}")
            
            # Reconstruir posiciones
            print(f"\n🔄 RECONSTRUYENDO POSICIONES CERRADAS:")
            print(f"{'-'*40}")
            
            net = 0.0
            max_net_abs = 0.0
            current_block = []
            open_time = None
            
            for i, trade in enumerate(symbol_fills):
                if open_time is None and abs(trade["signed"]) > 1e-9:
                    open_time = trade["ts"]
                
                net_before = net
                net += trade["signed"]
                current_block.append(trade)
                
                current_net_abs = abs(net)
                if current_net_abs > max_net_abs:
                    max_net_abs = current_net_abs
                
                print(f"\nTrade {i+1}: {trade['side'].upper()} {trade['qty']:.4f} @ {trade['price']:.4f}")
                print(f"  Net antes: {net_before:8.4f} | Net después: {net:8.4f} | Máximo histórico: {max_net_abs:8.4f}")
                
                # Verificar si se cerró una posición
                if abs(net) < 1e-9 and len(current_block) > 1 and max_net_abs > 1e-9:
                    close_time = trade["ts"]
                    
                    print(f"\n🎯 POSICIÓN CERRADA DETECTADA!")
                    print(f"   Período: {datetime.fromtimestamp(open_time/1000)} a {datetime.fromtimestamp(close_time/1000)}")
                    print(f"   Trades en bloque: {len(current_block)}")
                    print(f"   Size máximo: {max_net_abs:.4f}")
                    
                    # Calcular métricas de la posición
                    total_buy = sum(t["qty"] for t in current_block if t["signed"] > 0)
                    total_sell = sum(t["qty"] for t in current_block if t["signed"] < 0)
                    total_fees = sum(t["fee"] for t in current_block)
                    
                    # Determinar side
                    first_trade = None
                    for t in current_block:
                        if abs(t["signed"]) > 1e-9:
                            first_trade = t
                            break
                    
                    side = "long" if first_trade and first_trade["signed"] > 0 else "short"
                    
                    print(f"   Side: {side}")
                    print(f"   Total compras: {total_buy:.4f}")
                    print(f"   Total ventas: {total_sell:.4f}")
                    print(f"   Total fees: {total_fees:.6f}")
                    
                    # Calcular PnL con método FIFO
                    realized_pnl = 0.0
                    position_qty = 0.0
                    position_cost = 0.0
                    
                    print(f"\n   📊 CÁLCULO FIFO DETALLADO:")
                    for j, t in enumerate(current_block):
                        if t["signed"] > 0:  # BUY
                            old_qty = position_qty
                            old_cost = position_cost
                            
                            if position_qty == 0:
                                position_cost = t["price"]
                            else:
                                position_cost = (position_cost * position_qty + t["qty"] * t["price"]) / (position_qty + t["qty"])
                            position_qty += t["qty"]
                            
                            print(f"      {j+1}. BUY  {t['qty']:7.4f} @ {t['price']:8.4f} | "
                                  f"Posición: {position_qty:7.4f} @ {position_cost:8.4f}")
                                  
                        else:  # SELL
                            if position_qty > 0:
                                pnl_trade = (t["price"] - position_cost) * abs(t["signed"])
                                realized_pnl += pnl_trade
                                position_qty -= abs(t["signed"])
                                
                                print(f"      {j+1}. SELL {abs(t['signed']):7.4f} @ {t['price']:8.4f} | "
                                      f"PnL: {pnl_trade:7.4f} | Posición restante: {position_qty:7.4f}")
                    
                    realized_pnl_after_fees = realized_pnl - total_fees
                    
                    print(f"\n   💰 RESULTADOS PnL:")
                    print(f"      PnL por precio: {realized_pnl:9.4f}")
                    print(f"      Total fees:     {total_fees:9.4f}")
                    print(f"      PnL neto:       {realized_pnl_after_fees:9.4f}")
                    
                    # Buscar funding payments para esta posición
                    position_funding = 0.0
                    funding_details = []
                    if symbol in funding_by_symbol:
                        for fp in funding_by_symbol[symbol]:
                            fp_time = fp.get("timestamp")
                            if fp_time and open_time <= fp_time <= close_time:
                                position_funding += fp["income"]
                                funding_details.append({
                                    "time": datetime.fromtimestamp(fp_time/1000).strftime("%Y-%m-%d %H:%M"),
                                    "amount": fp["income"],
                                    "rate": fp.get("funding_rate", 0)
                                })
                    
                    print(f"\n   💸 FUNDING PAYMENTS:")
                    if funding_details:
                        for fd in funding_details:
                            print(f"      {fd['time']}: {fd['amount']:9.6f} (rate: {fd['rate']:.6f})")
                        print(f"      TOTAL FUNDING: {position_funding:9.6f}")
                    else:
                        print(f"      No se encontraron funding payments en el período")
                    
                    # PnL final incluyendo funding
                    final_pnl = realized_pnl_after_fees + position_funding
                    
                    print(f"\n   🎯 PnL FINAL:")
                    print(f"      PnL precio:    {realized_pnl_after_fees:9.4f}")
                    print(f"      Funding:       {position_funding:9.4f}")
                    print(f"      PnL TOTAL:     {final_pnl:9.4f}")
                    
                    print(f"\n   ✅ POSICIÓN RECONSTRUIDA:")
                    print(f"      {symbol} {side.upper()} | Size: {max_net_abs:.4f} | "
                          f"PnL: {final_pnl:.4f} | Fees: {total_fees:.4f} | Funding: {position_funding:.4f}")
                    
                    # Reset para siguiente posición
                    current_block = []
                    net = 0.0
                    max_net_abs = 0.0
                    open_time = None
                    print(f"\n{'─'*60}")
            
            # Mostrar posición abierta si queda alguna
            if abs(net) > 1e-9:
                print(f"\n⚠️  POSICIÓN ABIERTA DETECTADA:")
                print(f"   Net actual: {net:.4f}")
                print(f"   Trades en bloque actual: {len(current_block)}")
                print(f"   Size máximo: {max_net_abs:.4f}")
        
        print(f"\n✅ DEBUG COMPLETADO")
        
    except Exception as e:
        print(f"❌ ERROR en debug: {e}")
        import traceback
        traceback.print_exc()

def debug_backpack_symbol(symbol, days=60):
    """
    Debug específico para un símbolo
    """
    print(f"🔍 DEBUG ESPECÍFICO PARA {symbol}")
    debug_backpack_position_reconstruction(symbol_filter=symbol, days=days, debug=True)

# Bloque para ejecutar debug directamente desde Spyder
if __name__ == "__main__":
    # Ejemplo de uso - descomenta la línea que necesites:
    
    # Debug de un símbolo específico
    debug_backpack_symbol("KAITOUSDC", days=20)
    
    # Debug de todas las posiciones
    # debug_backpack_position_reconstruction(days=60)
    
    # Sincronización normal con la base de datos
    # save_backpack_closed_positions(debug=True)