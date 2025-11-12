# adapters/bitget.py
import os
import time
import hmac
import hashlib
import base64
import requests
from typing import List, Dict, Any, Optional
import re
from urllib.parse import urlencode
import json
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

# Configuración
BITGET_API_KEY = os.getenv("BITGET_API_KEY")
BITGET_API_SECRET = os.getenv("BITGET_API_SECRET")
BITGET_API_PASSPHRASE = os.getenv("BITGET_API_PASSPHRASE")
BITGET_BASE_URL = "https://api.bitget.com"

__all__ = [
    "fetch_bitget_all_balances",
    "fetch_bitget_open_positions", 
    "fetch_bitget_funding_fees",
    "save_bitget_closed_positions"
]

def normalize_symbol(sym: str) -> str:
    """Normaliza símbolos de Bitget según especificación"""
    if not sym: return ""
    s = sym.upper()
    s = re.sub(r'^PERP_', '', s)
    s = re.sub(r'(_|-)?(USDT|USDC|PERP)$', '', s)
    s = re.sub(r'[_-]+$', '', s)
    s = re.split(r'[_-]', s)[0]
    return s

def _bitget_sign(timestamp: str, method: str, request_path: str, 
                query_string: str = "", body: str = "") -> str:
    """Genera firma HMAC para Bitget"""
    message = timestamp + method.upper() + request_path
    if query_string:
        message += "?" + query_string
    if body:
        message += body
    
    mac = hmac.new(
        BITGET_API_SECRET.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    )
    return base64.b64encode(mac.digest()).decode()

def _bitget_request(method: str, path: str, params: Dict = None, 
                   body: Dict = None, version: str = "v2") -> Dict:
    """Realiza request autenticado a Bitget API asegurando orden idéntico al firmado."""
    if not all([BITGET_API_KEY, BITGET_API_SECRET, BITGET_API_PASSPHRASE]):
        raise ValueError("Missing Bitget API credentials")

    timestamp = str(int(time.time() * 1000))
    # ⚠️ construimos y reutilizamos EXACTAMENTE el mismo query string para firmar y para la URL
    query_string = urlencode(params or {}, doseq=True)
    body_str = json.dumps(body) if body else ""

    # la firma incluye ?query_string si existe
    signature = _bitget_sign(timestamp, method, path, query_string, body_str)

    headers = {
        "ACCESS-KEY": BITGET_API_KEY,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": BITGET_API_PASSPHRASE,
        "Content-Type": "application/json",
        "locale": "en-US"
    }

    # construimos URL final con el MISMO query string
    url = f"{BITGET_BASE_URL}{path}" + (f"?{query_string}" if query_string else "")

    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=30)
        else:
            response = requests.post(url, data=body_str if body_str else None, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Bitget API error ({path}): {e}")
        return {}

def fetch_bitget_all_balances() -> Dict[str, Any]:
    """
    Obtiene balances de Bitget mapeando Spot/Margin/Futures
    Devuelve estructura EXACTA para frontend Balances
    """
    try:
        # 1. Obtener funding assets (spot)
        spot_data = _bitget_request("GET", "/api/v2/account/funding-assets")
        spot_balance = 0.0
        
        if spot_data.get("code") == "00000":
            for asset in spot_data.get("data", []):
                if asset.get("usdtValue"):
                    spot_balance += float(asset["usdtValue"])
        
        # 2. Obtener cuenta de futuros USDT-M
        futures_balance = 0.0
        unrealized_pnl = 0.0
        initial_margin = 0.0
        
        futures_account = _bitget_request("GET", "/api/v2/mix/account/accounts", {
            "productType": "USDT-FUTURES"
        })
        
        if futures_account.get("code") == "00000":
            for account in futures_account.get("data", []):
                if account.get("usdtEquity"):
                    futures_balance += float(account["usdtEquity"])
                if account.get("crossedUnrealizedPL"):
                    unrealized_pnl += float(account["crossedUnrealizedPL"])
                # Calcular initial margin como equity - available
                if account.get("usdtEquity") and account.get("available"):
                    initial_margin += float(account["usdtEquity"]) - float(account.get("available", 0))
        
        # 3. Calcular equity total
        total_equity = spot_balance + futures_balance
        
        return {
            "exchange": "bitget",
            "equity": total_equity,
            "balance": spot_balance + futures_balance,  # saldo utilizable total
            "unrealized_pnl": unrealized_pnl,
            "initial_margin": initial_margin,
            "spot": spot_balance,
            "margin": 0.0,  # Bitget no tiene margin trading separado en esta API
            "futures": futures_balance
        }
        
    except Exception as e:
        print(f"❌ Bitget balances error: {e}")
        return {
            "exchange": "bitget",
            "equity": 0.0,
            "balance": 0.0,
            "unrealized_pnl": 0.0,
            "initial_margin": 0.0,
            "spot": 0.0,
            "margin": 0.0,
            "futures": 0.0
        }

def fetch_bitget_open_positions() -> List[Dict[str, Any]]:
    """
    Obtiene posiciones abiertas de Bitget Futures
    Shape EXACTO para frontend Open Positions
    """
    positions = []
    
    try:
        # Obtener todas las posiciones USDT-FUTURES
        data = _bitget_request("GET", "/api/v2/mix/position/all-position", {
            "productType": "USDT-FUTURES"
        })
        
        if data.get("code") != "00000":
            return positions
        
        for pos in data.get("data", []):
            try:
                symbol = normalize_symbol(pos.get("symbol", ""))
                hold_side = pos.get("holdSide", "").lower()
                size = float(pos.get("total", "0"))
                
                if size == 0:
                    continue
                
                # Calcular PnL no realizado SOLO por precio
                entry_price = float(pos.get("openPriceAvg", "0"))
                mark_price = float(pos.get("markPrice", "0"))
                
                if hold_side == "long":
                    unrealized_pnl = (mark_price - entry_price) * size
                else:  # short
                    unrealized_pnl = (entry_price - mark_price) * size
                
                # Fees y funding de la posición abierta
                fee_total = -abs(float(pos.get("deductedFee", "0")))  # negativo
                funding_fee = float(pos.get("totalFee", "0"))  # funding acumulado
                realized_pnl = fee_total + funding_fee  # solo fees + funding para abiertas
                
                positions.append({
                    "exchange": "bitget",
                    "symbol": symbol,
                    "side": hold_side,
                    "size": size,
                    "entry_price": entry_price,
                    "mark_price": mark_price,
                    "liquidation_price": float(pos.get("liquidationPrice", "0")),
                    "notional": float(pos.get("marginSize", "0")) * float(pos.get("leverage", "1")),
                    "unrealized_pnl": unrealized_pnl,
                    "fee": fee_total,
                    "funding_fee": funding_fee,
                    "realized_pnl": realized_pnl
                })
                
            except Exception as e:
                print(f"❌ Error procesando posición Bitget {pos.get('symbol')}: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Bitget open positions error: {e}")
    
    return positions

def fetch_bitget_funding_fees(limit: int = 2000,
                              since: int | None = None,
                              days: int | None = None,
                              chunk: int = 100,
                              max_pages: int = 200,
                              debug: bool = False) -> list[dict]:
    """
    User funding via Account Bill:
      GET /api/v2/mix/account/bill  (Rate limit 10 req/s UID)
    - productType=USDT-FUTURES
    - Filtra businessType = contract_settle_fee
    - Paginación hacia atrás con idLessThan (endId)
    - 'since' en ms o 'days' hacia atrás (se parte en ventanas <=30 días si se usa timebox)
    """
    out: list[dict] = []
    now_ms = int(time.time() * 1000)
    if since is None:
        days = 30 if days is None else int(days)
        since = now_ms - days*24*3600*1000
    since = int(since)

    id_less = None
    pages = 0
    while pages < max_pages and len(out) < limit:
        params = {
            "productType": "USDT-FUTURES",   # ¡mayúsculas!
            "limit": str(min(100, max(1, int(chunk))))
        }
        if id_less:
            params["idLessThan"] = id_less
        # (opcional) timebox por 30 días (si prefieres por tiempo en vez de idLessThan)
        # params["startTime"] = str(since)
        # params["endTime"]   = str(now_ms)

        data = _bitget_request("GET", "/api/v2/mix/account/bill", params=params)
        if data.get("code") != "00000":
            if debug: print("❌ bill error:", data)
            break

        payload = data.get("data", {}) or {}
        bills = payload.get("bills", []) or []
        end_id = payload.get("endId")

        if debug:
            cnt = len(bills)
            rng = [int(b.get("cTime", 0) or 0) for b in bills]
            print(f"PAGE {pages+1}: count={cnt} endId={end_id} "
                  f"range=({min(rng) if rng else None}..{max(rng) if rng else None})")

        for b in bills:
            try:
                if b.get("businessType") != "contract_settle_fee":
                    continue  # funding fees solamente
                ts = int(b.get("cTime", "0") or 0)
                if ts and ts < since:
                    continue
                amt = float(b.get("amount", "0") or 0)  # firmado por Bitget, puede ser +/- 
                sym_raw = b.get("symbol") or ""
                sym = normalize_symbol(sym_raw) if 'normalize_symbol' in globals() else sym_raw
                out.append({
                    "exchange": "bitget",
                    "symbol": sym or "GENERAL",
                    "symbol_raw": sym_raw,
                    "income": amt,                     # negativo=pago, positivo=cobro
                    "asset": b.get("coin", "USDT"),
                    "timestamp": ts,
                    "funding_rate": None,
                    "type": "FUNDING_FEE",
                    "external_id": str(b.get("billId") or f"bitget|{ts}|{amt:.8f}")
                })
                if len(out) >= limit:
                    break
            except Exception as e:
                if debug: print("  · skip bill:", e)

        if not bills or not end_id:
            break
        id_less = end_id
        pages += 1
        # pequeño respiro (RL 10/s)
        time.sleep(0.05)

    out.sort(key=lambda x: x["timestamp"] or 0)
    if debug:
        if out:
            print(f"TOTAL items={len(out)} range=({out[0]['timestamp']}..{out[-1]['timestamp']})")
        else:
            print("TOTAL items=0")
    return out

def save_bitget_closed_positions(db_path: str = "portfolio.db", days: int = 30, debug: bool = False) -> None:
    """
    Guarda posiciones cerradas de Bitget en SQLite con verificación de duplicados
    """
    try:
        from db_manager import save_closed_position
        import sqlite3
        
        # Verificar credenciales
        if not all([BITGET_API_KEY, BITGET_API_SECRET, BITGET_API_PASSPHRASE]):
            print("⚠️ Bitget: faltan credenciales. No se guardan cerradas.")
            return
        
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        # Obtener posiciones históricas
        data = _bitget_request("GET", "/api/v2/mix/position/history-position", {
            "productType": "usdt-futures",
            "startTime": str(start_time),
            "endTime": str(end_time),
            "limit": "100"
        })
        
        if data.get("code") != "00000":
            if debug:
                print(f"❌ Bitget API error: {data.get('msg', 'Unknown error')}")
            return
        
        positions = data.get("data", {}).get("list", [])
        if not positions:
            if debug:
                print("⚠️ No se encontraron posiciones cerradas en Bitget.")
            return

        # Verificar que la base de datos existe
        if not os.path.exists(db_path):
            print(f"❌ Database not found: {db_path}")
            return

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        saved = 0
        skipped = 0

        for pos in positions:
            try:
                symbol = normalize_symbol(pos.get("symbol", ""))
                hold_side = pos.get("holdSide", "").lower()
                size = float(pos.get("openTotalPos", "0"))
                
                if size == 0:
                    continue
                
                # Calcular realized_pnl TOTAL NETO
                net_profit = float(pos.get("netProfit", "0"))
                total_funding = float(pos.get("totalFunding", "0"))
                open_fee = float(pos.get("openFee", "0"))
                close_fee = float(pos.get("closeFee", "0"))
                fee_total = -abs(open_fee + close_fee)
                close_time = int(pos.get("uTime", "0")) // 1000
                
                # VERIFICAR DUPLICADOS (igual que KuCoin)
                cur.execute("""
                    SELECT COUNT(*) FROM closed_positions
                    WHERE exchange = ? AND symbol = ? AND close_time = ?
                """, ("bitget", symbol, close_time))
                
                if cur.fetchone()[0]:
                    skipped += 1
                    continue
                
                closed_position = {
                    "exchange": "bitget",
                    "symbol": symbol,
                    "side": hold_side,
                    "size": size,
                    "entry_price": float(pos.get("openAvgPrice", "0")),
                    "close_price": float(pos.get("closeAvgPrice", "0")),
                    "open_time": int(pos.get("cTime", "0")) // 1000,
                    "close_time": close_time,
                    "realized_pnl": net_profit,
                    "funding_total": total_funding,
                    "fee_total": fee_total,
                    "notional": float(pos.get("openAvgPrice", "0")) * size,
                    "leverage": 1.0,
                    "liquidation_price": None
                }
                
                # Guardar en DB
                save_closed_position(closed_position)
                saved += 1
                
                if debug:
                    print(f"✅ Bitget closed: {symbol} {hold_side} size={size} "
                          f"realized={net_profit:.4f}")
                          
            except Exception as e:
                print(f"❌ Error guardando posición cerrada Bitget: {e}")
                continue
        
        conn.close()
        # PRINT FINAL (igual que KuCoin)
        print(f"✅ Bitget guardadas: {saved} | omitidas (duplicadas): {skipped}")
            
    except Exception as e:
        print(f"❌ Bitget closed positions error: {e}")

if __name__ == "__main__":
    import argparse
    from datetime import datetime, timezone

    def _iso(ms: int) -> str:
        try:
            return datetime.fromtimestamp(int(ms)/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            return str(ms)

    ap = argparse.ArgumentParser()
    ap.add_argument("--funding-debug", action="store_true", help="Debug interest-history paginado")
    ap.add_argument("--days", type=int, default=60, help="Ventana hacia atrás (días) si no se pasa 'since'")
    ap.add_argument("--since", type=int, default=None, help="Epoch ms de corte inferior")
    ap.add_argument("--limit", type=int, default=1000, help="Máximo de items totales a traer")
    ap.add_argument("--chunk", type=int, default=100, help="Items por request (máx 100)")
    ap.add_argument("--max-pages", type=int, default=30, help="Máximo de páginas")
    args = ap.parse_args()

    if args.funding_debug:
        print("🔎 Bitget interest-history DEBUG")
        rows = fetch_bitget_funding_fees(limit=args.limit,
                                         since=args.since,
                                         days=args.days,
                                         chunk=args.chunk,
                                         max_pages=args.max_pages,
                                         debug=True)
        print(f"\nResumen: items={len(rows)}")
        if rows:
            print("  earliest:", _iso(rows[0]["timestamp"]), "latest:", _iso(rows[-1]["timestamp"]))
            print("  sample:", rows[:3])
    else:
        # Smoke rápido de balances/abiertas/funding corto
        print("== balances ==")
        print(fetch_bitget_all_balances())
        print("\n== open positions ==")
        print(fetch_bitget_open_positions())
        print("\n== funding (debug corto) ==")
        print(fetch_bitget_funding_fees(limit=10, days=3, chunk=50, max_pages=3, debug=True))



# def debug_preview_bitget_closed(days: int = 3, symbol: Optional[str] = None) -> None:
#     """Debug: previsualiza lo que se guardaría para posiciones cerradas"""
#     print(f"🔍 Debug Bitget Closed Positions (últimos {days} días)")
    
#     end_time = int(time.time() * 1000)
#     start_time = end_time - (days * 24 * 60 * 60 * 1000)
    
#     data = _bitget_request("GET", "/api/v2/mix/position/history-position", {
#         "productType": "USDT-FUTURES",
#         "startTime": start_time,
#         "endTime": end_time,
#         "limit": 20
#     })
    
#     if data.get("code") != "00000":
#         print("❌ No se pudieron obtener datos")
#         return
    
#     for pos in data.get("data", {}).get("list", []):
#         sym = normalize_symbol(pos.get("symbol", ""))
#         if symbol and sym != symbol:
#             continue
            
#         net_profit = float(pos.get("netProfit", "0"))
#         total_funding = float(pos.get("totalFunding", "0"))
#         open_fee = float(pos.get("openFee", "0"))
#         close_fee = float(pos.get("closeFee", "0"))
#         fee_total = -abs(open_fee + close_fee)
#         price_pnl = net_profit - total_funding - fee_total
        
#         print(f"\n📊 {sym} {pos.get('holdSide')}:")
#         print(f"   Size: {pos.get('openTotalPos')}")
#         print(f"   Entry: {pos.get('openAvgPrice')} | Close: {pos.get('closeAvgPrice')}")
#         print(f"   Realized PnL (neto): {net_profit:.6f}")
#         print(f"   Price PnL: {price_pnl:.6f}")
#         print(f"   Funding: {total_funding:.6f}")
#         print(f"   Fees: {fee_total:.6f}")
#         print(f"   Open: {pos.get('cTime')} | Close: {pos.get('uTime')}")

# if __name__ == "__main__":
#     # Smoke tests
#     import sys
    
#     if "--dry-run" in sys.argv:
#         print("🧪 Bitget Adapter Smoke Tests")
        
#         # Test normalización
#         test_symbols = ["BTCUSDT", "BTC-USDT", "BTC_USDT", "PERP_BTCUSDT", "ETHUSDC-PERP"]
#         print("\n🔧 Normalization tests:")
#         for sym in test_symbols:
#             print(f"   {sym} -> {normalize_symbol(sym)}")
        
#         # Test balances
#         print("\n💰 Balance test:")
#         balances = fetch_bitget_all_balances()
#         if balances:
#             print(f"   Equity: {balances['equity']:.2f}")
#             print(f"   Spot: {balances['spot']:.2f}")
#             print(f"   Futures: {balances['futures']:.2f}")
#             print(f"   Unrealized PnL: {balances['unrealized_pnl']:.2f}")
        
#         # Test open positions
#         print("\n📈 Open positions test:")
#         positions = fetch_bitget_open_positions()
#         for pos in positions[:3]:  # Mostrar primeras 3
#             print(f"   {pos['symbol']} {pos['side']} size={pos['size']} unrealized={pos['unrealized_pnl']:.2f}")
        
#         # Test funding
#         print("\n💸 Funding test:")
#         funding = fetch_bitget_funding_fees(limit=5)
#         for f in funding:
#             print(f"   {f['asset']}: {f['income']:.6f} rate={f['funding_rate']:.6f}")
        
#         # Test closed preview
#         print("\n📊 Closed positions preview:")
#         debug_preview_bitget_closed(days=1)
        
#         print("\n✅ Smoke tests completed")