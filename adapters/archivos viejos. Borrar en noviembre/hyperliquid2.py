# adapters/hyperliquidv2.py
import os
import time
import sqlite3
import requests
from collections import defaultdict, deque
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]  # .../extended-web
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db_manager import save_closed_position
from utils.symbols import normalize_symbol
from utils.money import D, usd, quant, normalize_fee, to_float
from utils.time import utc_now_ms, to_ms, to_s

# Configuración
HYPERLIQUID_API_URL = "https://api.hyperliquid.xyz/info"
HYPERLIQUID_ACCOUNT = "0x981690Ec51Bb332Ec6eED511C27Df325104cb461"

class HyperliquidFIFO:
    def __init__(self):
        self.fills = []
        self.funding_payments = []
        
    def fetch_user_fills(self, start_time: int, end_time: int):
        try:
            payload = {
                "type": "userFillsByTime",
                "user": HYPERLIQUID_ACCOUNT,
                "startTime": start_time,
                "endTime": end_time,
                "aggregateByTime": False
            }
            r = requests.post(HYPERLIQUID_API_URL, json=payload, timeout=30)
            data = r.json()
            if isinstance(data, list):
                # Perps únicamente (descarta spot @idx)
                return [f for f in data if not str(f.get("coin", "")).startswith("@")]
            return []
        except Exception as e:
            print(f"❌ Error fetching Hyperliquid fills: {e}")
            return []
    
    # =========================
    # 2) FUNDING: usa userFunding y agrupa por símbolo
    # =========================
    def fetch_user_funding(self, start_time: int, end_time: int):
        try:
            payload = {
                "type": "userFunding",
                "user": HYPERLIQUID_ACCOUNT,
                "startTime": start_time,
                "endTime": end_time
            }
            r = requests.post(HYPERLIQUID_API_URL, json=payload, timeout=30)
            data = r.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            print(f"❌ Error fetching Hyperliquid userFunding: {e}")
            return []
    
    def fetch_all_funding(self, symbols, start_time: int, end_time: int):
        """Devuelve {symbol: [ {time, usdc, fundingRate, szi} ]} solo de esos símbolos."""
        all_ev = self.fetch_user_funding(start_time, end_time)
        by_symbol = {s: [] for s in symbols}
        for ev in all_ev:
            delta = ev.get("delta", {})
            coin = delta.get("coin")
            if coin in by_symbol and delta.get("type") == "funding":
                by_symbol[coin].append({
                    "time": ev.get("time", 0),
                    "usdc": float(delta.get("usdc", 0)),           # **pago real**
                    "fundingRate": float(delta.get("fundingRate", 0)),
                    "szi": float(delta.get("szi", 0)),
                })
        return by_symbol
    
    # Mantén una compat de nombre si llamabas a 'fetch_funding_history'
    def fetch_funding_history(self, coin: str, start_time: int, end_time: int):
        """Compat: devuelve eventos de funding (userFunding) para un coin concreto."""
        m = self.fetch_all_funding([coin], start_time, end_time)
        return m.get(coin, [])
    
    # =========================
    # 3) BLOQUES FIFO: detecta dirección con 'dir' y cierra en net=0
    # =========================
    from decimal import Decimal
    from collections import defaultdict, deque
    
    def _signed_from_dir(self, dir_str: str, sz: Decimal) -> Decimal:
        # Reglas consistentes con Hyperliquid:
        #  Open Long (+),  Close Long (-),  Open Short (-),  Close Short (+)
        if "Open Long" in dir_str:   return sz
        if "Close Long" in dir_str:  return -sz
        if "Open Short" in dir_str:  return -sz
        if "Close Short" in dir_str: return sz
        # Fallback por si alguna vez viniera sólo Buy/Sell:
        if "Buy" in dir_str:  return sz
        if "Sell" in dir_str: return -sz
        return sz  # último recurso
    
    def calculate_fifo_blocks(self, fills):
        fills_by_symbol = defaultdict(list)
        for f in fills:
            coin = f.get("coin", "")
            if coin:
                fills_by_symbol[coin].append(f)
        for coin in fills_by_symbol:
            fills_by_symbol[coin].sort(key=lambda x: x.get("time", 0))
    
        blocks = []
        for coin, sf in fills_by_symbol.items():
            net = Decimal("0")
            open_t = None
            side = None
            cur_fills = []
            hist = []
    
            for f in sf:
                sz = Decimal(str(f.get("sz", "0")))
                dir_str = str(f.get("dir", ""))
                signed = self._signed_from_dir(dir_str, sz)
    
                # side del bloque se fija en el primer movimiento desde 0
                if side is None and signed != 0:
                    side = "long" if signed > 0 else "short"
                if open_t is None:
                    open_t = f.get("time", 0)
    
                net += signed
                cur_fills.append(f)
                hist.append(float(net))
    
                # Cierra bloque cuando vuelve exactamente a 0 (tolerancia pequeña)
                if abs(net) <= Decimal("0.0000001"):
                    if len(cur_fills) >= 2:
                        blocks.append({
                            "symbol": coin,
                            "fills": cur_fills.copy(),
                            "open_time": open_t,
                            "close_time": cur_fills[-1].get("time", 0),
                            "side": side,
                            "net_quantity_history": hist.copy(),
                        })
                    # reset
                    net = Decimal("0")
                    open_t = None
                    side = None
                    cur_fills.clear()
                    hist.clear()
    
            # Si queda algo sin cerrar, no lo contamos como cerrado
        return blocks
        
    # =========================
    # 4) PNL FIFO correcto + precios de entrada/cierre a partir de los MATCHES
    # =========================
    def calculate_fifo_pnl(self, block):
        if not block["fills"]:
            return {}
    
        is_long = (block["side"] == "long")
        fifo_q = deque()   # cada lote: {"size": Decimal, "price": Decimal}
        fees = Decimal("0")
    
        entry_notional_used = Decimal("0")
        exit_notional_used  = Decimal("0")
        matched_qty         = Decimal("0")
    
        entry_fills = []
        exit_fills  = []
    
        for f in block["fills"]:
            sz  = Decimal(str(f.get("sz", "0")))
            px  = Decimal(str(f.get("px", "0")))
            fee = Decimal(str(f.get("fee", "0")))
            dir_str = str(f.get("dir", ""))
    
            open_long   = "Open Long"   in dir_str
            close_long  = "Close Long"  in dir_str
            open_short  = "Open Short"  in dir_str
            close_short = "Close Short" in dir_str
    
            if is_long:
                if open_long:
                    entry_fills.append(f)
                    fifo_q.append({"size": sz, "price": px})
                elif close_long:
                    exit_fills.append(f)
                    remaining = sz
                    while remaining > 0 and fifo_q:
                        lot = fifo_q[0]
                        take = lot["size"] if lot["size"] <= remaining else remaining
                        # match
                        entry_notional_used += lot["price"] * take
                        exit_notional_used  += px * take
                        matched_qty         += take
                        # reduce
                        lot["size"] -= take
                        remaining   -= take
                        if lot["size"] <= 0:
                            fifo_q.popleft()
            else:
                # short: entrada = Open Short (venta), salida = Close Short (compra)
                if open_short:
                    entry_fills.append(f)
                    fifo_q.append({"size": sz, "price": px})
                elif close_short:
                    exit_fills.append(f)
                    remaining = sz
                    while remaining > 0 and fifo_q:
                        lot = fifo_q[0]
                        take = lot["size"] if lot["size"] <= remaining else remaining
                        # Para short el PnL es (entry_px - exit_px) * qty
                        entry_notional_used += lot["price"] * take
                        exit_notional_used  += px * take
                        matched_qty         += take
                        lot["size"] -= take
                        remaining   -= take
                        if lot["size"] <= 0:
                            fifo_q.popleft()
    
            fees += fee
    
        # PnL en USD
        if is_long:
            fifo_pnl = (exit_notional_used - entry_notional_used)
        else:
            fifo_pnl = (entry_notional_used - exit_notional_used)
    
        # Precios VWAP pero SOLO de lo emparejado
        if matched_qty > 0:
            entry_avg = (entry_notional_used / matched_qty)
            close_avg = (exit_notional_used  / matched_qty)
        else:
            entry_avg = Decimal("0")
            close_avg = Decimal("0")
    
        # tamaño máximo vivo del bloque
        max_size = Decimal(str(max([abs(q) for q in block["net_quantity_history"]] or [0])))
    
        return {
            "symbol": block["symbol"],
            "side": block["side"],
            "size": float(max_size),
            "entry_price": float(entry_avg),
            "close_price": float(close_avg),
            "open_time": block["open_time"],
            "close_time": block["close_time"],
            "pnl": float(fifo_pnl),
            "fee_total": float(fees),
            "entry_fills": entry_fills,
            "exit_fills": exit_fills,
            "fifo_queue_remaining": sum(l["size"] for l in fifo_q),
        }
        
    # =========================
    # 5) FUNDING por bloque: suma delta.usdc dentro del rango
    # =========================
    from decimal import Decimal
    
    def calculate_funding_for_block(self, block, funding_data):
        coin = block["symbol"]
        if coin not in funding_data:
            return 0.0
        ot = block["open_time"]
        ct = block["close_time"]
        total = Decimal("0")
        for ev in funding_data[coin]:
            t = ev.get("time", 0)
            if ot <= t <= ct:
                total += Decimal(str(ev.get("usdc", 0)))  # ya viene con signo
        return float(total)
        
    # =========================
    # 6) RECONSTRUCCIÓN PRINCIPAL: usa los cambios anteriores
    # =========================
    def reconstruct_closed_positions(self, days: int = 60):
        end_time = utc_now_ms()
        start_time = end_time - days * 24 * 60 * 60 * 1000
    
        fills = self.fetch_user_fills(start_time, end_time)
        symbols = list({f.get("coin") for f in fills if f.get("coin")})
        funding_map = self.fetch_all_funding(symbols, start_time, end_time)
    
        blocks = self.calculate_fifo_blocks(fills)
        closed = []
        for b in blocks:
            r = self.calculate_fifo_pnl(b)
            if not r:
                continue
            funding_total = self.calculate_funding_for_block(b, funding_map)
            realized_pnl = r["pnl"] - r["fee_total"] + funding_total
            notional = r["size"] * r["entry_price"]
    
            closed.append({
                "exchange": "hyperliquid",
                "symbol": r["symbol"],
                "side": r["side"],
                "size": r["size"],
                "entry_price": r["entry_price"],
                "close_price": r["close_price"],
                "open_time": r["open_time"] // 1000,
                "close_time": r["close_time"] // 1000,
                "pnl": r["pnl"],
                "realized_pnl": realized_pnl,
                "funding_total": funding_total,
                "fee_total": r["fee_total"],
                "notional": notional,
                "initial_margin": None,
                "leverage": None,
                "liquidation_price": None,
            })
        return closed

#============== fin de la reconstruccion

# Funciones públicas del adapter
def save_hyperliquid_closed_positions(db_path: str = "portfolio.db", days: int = 60, debug: bool = False) -> int:
    """Guarda posiciones cerradas de Hyperliquid usando reconstrucción FIFO"""
    if debug:
        print("🔍 [DEBUG] Iniciando reconstrucción FIFO Hyperliquid...")
    
    fifo = HyperliquidFIFO()
    closed_positions = fifo.reconstruct_closed_positions(days)
    
    saved_count = 0
    duplicate_count = 0
    
    for position in closed_positions:
        try:
            # Verificar si ya existe (mismo exchange, symbol, close_time)
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("""
                SELECT COUNT(*) FROM closed_positions 
                WHERE exchange = ? AND symbol = ? AND close_time = ?
            """, (position["exchange"], position["symbol"], position["close_time"]))
            
            if cur.fetchone()[0] == 0:
                save_closed_position(position)
                saved_count += 1
                if debug:
                    print(f"💾 Guardada posición {position['symbol']} - PnL: {position['pnl']:.4f}")
            else:
                duplicate_count += 1
                
            conn.close()
            
        except Exception as e:
            print(f"❌ Error guardando posición {position['symbol']}: {e}")
    
    print(f"✅ Hyperliquid: Guardadas {saved_count} posiciones, omitidas {duplicate_count} duplicadas")
    return saved_count

def debug_hyperliquid_fifo_reconstruction(symbol: str = None, days: int = 60):
    """Función de debug para reconstrucción FIFO"""
    print(f"🔍 DEBUG Hyperliquid FIFO - Símbolo: {symbol or 'Todos'} - Días: {days}")
    
    fifo = HyperliquidFIFO()
    end_time = utc_now_ms()
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    
    fills = fifo.fetch_user_fills(start_time, end_time)
    
    if symbol:
        fills = [f for f in fills if f.get("coin", "") == symbol]
    
    print(f"📦 Fills obtenidos: {len(fills)}")
    
    blocks = fifo.calculate_fifo_blocks(fills)
    print(f"🔍 Bloques FIFO identificados: {len(blocks)}")
    
    for i, block in enumerate(blocks):
        if symbol and block["symbol"] != symbol:
            continue
            
        fifo_result = fifo.calculate_fifo_pnl(block)
        if not fifo_result:
            continue
            
        print(f"\n🎯 Bloque {i+1}: {block['symbol']} {block['side']}")
        print(f"   📏 Size: {fifo_result['size']:.4f}")
        print(f"   💰 Entry: {fifo_result['entry_price']:.4f} | Close: {fifo_result['close_price']:.4f}")
        print(f"   📊 PnL FIFO: {fifo_result['pnl']:.4f} | Fees: {fifo_result['fee_total']:.4f}")
        print(f"   ⏰ Open: {datetime.fromtimestamp(block['open_time']/1000)}")
        print(f"   ⏰ Close: {datetime.fromtimestamp(block['close_time']/1000)}")
        print(f"   📈 Fills: {len(block['fills'])} trades")

def fetch_hyperliquid_open_positions():
    """Obtiene posiciones abiertas de Hyperliquid"""
    try:
        payload = {
            "type": "clearinghouseState",
            "user": HYPERLIQUID_ACCOUNT
        }
        
        response = requests.post(HYPERLIQUID_API_URL, json=payload, timeout=30)
        data = response.json()
        
        open_positions = []
        asset_positions = data.get("assetPositions", [])
        
        for asset_pos in asset_positions:
            position = asset_pos.get("position", {})
            if float(position.get("szi", 0)) != 0:  # Posición activa
                open_positions.append({
                    "exchange": "hyperliquid",
                    "symbol": position.get("coin", ""),
                    "side": "long" if float(position.get("szi", 0)) > 0 else "short",
                    "size": abs(float(position.get("szi", 0))),
                    "entry_price": float(position.get("entryPx", 0)),
                    "mark_price": 0,  # Necesitarías otro endpoint para mark price
                    "unrealized_pnl": float(position.get("unrealizedPnl", 0)),
                    "leverage": float(position.get("leverage", {}).get("value", 0)),
                    "liquidation_price": float(position.get("liquidationPx", 0)),
                    "notional": float(position.get("positionValue", 0))
                })
        
        return open_positions
        
    except Exception as e:
        print(f"❌ Error fetching Hyperliquid open positions: {e}")
        return []

def fetch_hyperliquid_funding_fees(limit: int = 50):
    """Obtiene historial de funding fees"""
    try:
        # Necesitaríamos saber los símbolos activos primero
        open_positions = fetch_hyperliquid_open_positions()
        symbols = list(set(pos["symbol"] for pos in open_positions))
        
        end_time = utc_now_ms()
        start_time = end_time - (7 * 24 * 60 * 60 * 1000)  # 1 semana
        
        funding_fees = []
        for symbol in symbols:
            payload = {
                "type": "fundingHistory",
                "coin": symbol,
                "startTime": start_time,
                "endTime": end_time
            }
            
            response = requests.post(HYPERLIQUID_API_URL, json=payload, timeout=30)
            symbol_funding = response.json()
            
            for funding_event in symbol_funding:
                funding_fees.append({
                    "exchange": "hyperliquid",
                    "symbol": symbol,
                    "payment": float(funding_event.get("fundingRate", 0)),
                    "timestamp": funding_event.get("time", 0) // 1000,
                    "payment_usdt": 0  # Se calcularía basado en la posición
                })
        
        return funding_fees[:limit]
        
    except Exception as e:
        print(f"❌ Error fetching Hyperliquid funding: {e}")
        return []

def fetch_hyperliquid_all_balances():
    """Balances correctos usando 'withdrawable' como disponible."""
    try:
        payload = {"type": "clearinghouseState", "user": HYPERLIQUID_ACCOUNT}
        r = requests.post(HYPERLIQUID_API_URL, json=payload, timeout=30)
        data = r.json()

        asset_positions = data.get("assetPositions", []) or []
        unreal = 0.0
        for ap in asset_positions:
            p = ap.get("position", {}) or {}
            unreal += float(p.get("unrealizedPnl", 0.0))

        ms = data.get("marginSummary", {}) or {}
        equity = float(ms.get("accountValue", 0.0))
        # ¡Clave!: el disponible real lo da 'withdrawable'
        available = float(data.get("withdrawable", equity))

        return {
            "exchange": "hyperliquid",
            "equity": equity,
            "balance": available,                 # antes te daba 0
            "unrealized_pnl": unreal,
            "spot": 0.0,
            "margin": float(ms.get("totalMarginUsed", 0.0)),
            "futures": equity,
        }
    except Exception as e:
        print(f"❌ Error fetching Hyperliquid balances: {e}")
        return None