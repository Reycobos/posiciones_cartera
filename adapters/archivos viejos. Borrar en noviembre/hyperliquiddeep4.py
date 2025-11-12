# adapters/hyperliquidv2.py
import os
import time
import sqlite3
import requests
from collections import defaultdict, deque
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

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
        
    def fetch_user_fills(self, start_time: int, end_time: int) -> List[Dict]:
        """Obtiene todos los fills del usuario en el rango de tiempo"""
        try:
            payload = {
                "type": "userFillsByTime",
                "user": HYPERLIQUID_ACCOUNT,
                "startTime": start_time,
                "endTime": end_time
            }
            
            response = requests.post(HYPERLIQUID_API_URL, json=payload, timeout=30)
            data = response.json()
            
            if isinstance(data, list):
                # Filtrar solo perpetuals (excluir spot que tienen formato @número)
                perpetual_fills = [fill for fill in data if not fill.get("coin", "").startswith("@")]
                return perpetual_fills
            return []
            
        except Exception as e:
            print(f"❌ Error fetching Hyperliquid fills: {e}")
            return []
    
    def fetch_funding_history(self, coin: str, start_time: int, end_time: int) -> List[Dict]:
        """Obtiene historial de funding para un símbolo"""
        try:
            payload = {
                "type": "fundingHistory",
                "coin": coin,
                "startTime": start_time,
                "endTime": end_time
            }
            
            response = requests.post(HYPERLIQUID_API_URL, json=payload, timeout=30)
            return response.json() if response.status_code == 200 else []
            
        except Exception as e:
            print(f"❌ Error fetching Hyperliquid funding for {coin}: {e}")
            return []
    
    def fetch_all_funding(self, symbols: List[str], start_time: int, end_time: int) -> Dict[str, List[Dict]]:
        """Obtiene funding para todos los símbolos"""
        funding_by_symbol = {}
        for symbol in symbols:
            funding_data = self.fetch_funding_history(symbol, start_time, end_time)
            funding_by_symbol[symbol] = funding_data
        return funding_by_symbol
    
    def calculate_fifo_blocks(self, fills: List[Dict]) -> List[Dict]:
        """Reconstruye bloques FIFO a partir de fills"""
        # Agrupar fills por símbolo
        fills_by_symbol = defaultdict(list)
        for fill in fills:
            symbol = fill.get("coin", "")
            if symbol:  # Excluir símbolos vacíos
                fills_by_symbol[symbol].append(fill)
        
        # Ordenar todos los fills por timestamp
        for symbol in fills_by_symbol:
            fills_by_symbol[symbol].sort(key=lambda x: x.get("time", 0))
        
        blocks = []
        
        for symbol, symbol_fills in fills_by_symbol.items():
            net_quantity = Decimal('0')
            open_time = None
            close_time = None
            current_block = {
                "symbol": symbol,
                "fills": [],
                "net_quantity_history": [],
                "side": None
            }
            
            for fill in symbol_fills:
                # Determinar cantidad firmada
                side = fill.get("side", "")
                dir_str = fill.get("dir", "")
                size = Decimal(str(fill.get("sz", "0")))
                
                # Determinar si es compra o venta
                if side == "B" or "Long" in dir_str or "Buy" in dir_str:
                    signed_size = size
                elif side == "A" or "Short" in dir_str or "Sell" in dir_str:
                    signed_size = -size
                else:
                    # Por defecto, usar side
                    signed_size = size if side == "B" else -size
                
                net_quantity += signed_size
                current_block["fills"].append(fill)
                current_block["net_quantity_history"].append(float(net_quantity))
                
                # Determinar side del bloque
                if current_block["side"] is None and abs(signed_size) > 0:
                    current_block["side"] = "long" if signed_size > 0 else "short"
                
                # Establecer open_time
                if open_time is None and len(current_block["fills"]) == 1:
                    open_time = fill.get("time", 0)
                
                # Si el net quantity vuelve a 0, cerrar el bloque
                if abs(net_quantity) < Decimal('0.0001'):  # Tolerancia para floating point
                    close_time = fill.get("time", 0)
                    
                    if len(current_block["fills"]) > 1:  # Bloque válido (mínimo 2 trades)
                        blocks.append({
                            "symbol": symbol,
                            "fills": current_block["fills"].copy(),
                            "open_time": open_time,
                            "close_time": close_time,
                            "side": current_block["side"],
                            "net_quantity_history": current_block["net_quantity_history"].copy(),
                            "max_position_size": max([abs(qty) for qty in current_block["net_quantity_history"]])
                        })
                    
                    # Resetear para el siguiente bloque
                    current_block = {
                        "symbol": symbol,
                        "fills": [],
                        "net_quantity_history": [],
                        "side": None
                    }
                    net_quantity = Decimal('0')
                    open_time = None
                    close_time = None
            
            # Manejar bloque final si no se cerró
            if current_block["fills"] and len(current_block["fills"]) >= 2:
                blocks.append({
                    "symbol": symbol,
                    "fills": current_block["fills"],
                    "open_time": open_time,
                    "close_time": current_block["fills"][-1].get("time", 0),
                    "side": current_block["side"],
                    "net_quantity_history": current_block["net_quantity_history"],
                    "max_position_size": max([abs(qty) for qty in current_block["net_quantity_history"]])
                })
        
        return blocks
    
    def calculate_fifo_pnl(self, block: Dict) -> Dict[str, Any]:
        """Calcula PnL FIFO para un bloque de trades usando el método correcto"""
        if not block["fills"]:
            return {}
        
        # Cola FIFO para lotes de entrada
        fifo_queue = deque()
        total_fees = Decimal('0')
        realized_pnl = Decimal('0')
        
        # Procesar todos los fills en orden cronológico
        for fill in block["fills"]:
            side = fill.get("side", "")
            dir_str = fill.get("dir", "")
            size = Decimal(str(fill.get("sz", "0")))
            price = Decimal(str(fill.get("px", "0")))
            fee = Decimal(str(fill.get("fee", "0")))
            
            total_fees += fee
            
            # Determinar dirección del trade
            is_buy = (side == "B" or "Long" in dir_str or "Buy" in dir_str)
            is_sell = (side == "A" or "Short" in dir_str or "Sell" in dir_str)
            
            # Para long: buys son entrada, sells son salida
            # Para short: sells son entrada, buys son salida
            if block["side"] == "long":
                if is_buy:
                    # Añadir a la cola FIFO como lote de entrada
                    fifo_queue.append({"size": size, "price": price})
                elif is_sell:
                    # Procesar venta contra lotes FIFO
                    remaining_sell_size = size
                    while remaining_sell_size > 0 and fifo_queue:
                        entry_lot = fifo_queue[0]
                        
                        if entry_lot["size"] <= remaining_sell_size:
                            # Usar todo el lote de entrada
                            pnl_for_lot = (price - entry_lot["price"]) * entry_lot["size"]
                            realized_pnl += pnl_for_lot
                            remaining_sell_size -= entry_lot["size"]
                            fifo_queue.popleft()
                        else:
                            # Usar parte del lote de entrada
                            pnl_for_lot = (price - entry_lot["price"]) * remaining_sell_size
                            realized_pnl += pnl_for_lot
                            entry_lot["size"] -= remaining_sell_size
                            remaining_sell_size = 0
            else:  # short
                if is_sell:
                    # Añadir a la cola FIFO como lote de entrada (short)
                    fifo_queue.append({"size": size, "price": price})
                elif is_buy:
                    # Procesar compra contra lotes FIFO (short)
                    remaining_buy_size = size
                    while remaining_buy_size > 0 and fifo_queue:
                        entry_lot = fifo_queue[0]
                        
                        if entry_lot["size"] <= remaining_buy_size:
                            # Usar todo el lote de entrada
                            pnl_for_lot = (entry_lot["price"] - price) * entry_lot["size"]
                            realized_pnl += pnl_for_lot
                            remaining_buy_size -= entry_lot["size"]
                            fifo_queue.popleft()
                        else:
                            # Usar parte del lote de entrada
                            pnl_for_lot = (entry_lot["price"] - price) * remaining_buy_size
                            realized_pnl += pnl_for_lot
                            entry_lot["size"] -= remaining_buy_size
                            remaining_buy_size = 0
        
        # Calcular precios promedio ponderados
        entry_size_total = Decimal('0')
        entry_value_total = Decimal('0')
        exit_size_total = Decimal('0')
        exit_value_total = Decimal('0')
        
        for fill in block["fills"]:
            side = fill.get("side", "")
            dir_str = fill.get("dir", "")
            size = Decimal(str(fill.get("sz", "0")))
            price = Decimal(str(fill.get("px", "0")))
            
            is_buy = (side == "B" or "Long" in dir_str or "Buy" in dir_str)
            is_sell = (side == "A" or "Short" in dir_str or "Sell" in dir_str)
            
            if block["side"] == "long":
                if is_buy:
                    entry_size_total += size
                    entry_value_total += size * price
                elif is_sell:
                    exit_size_total += size
                    exit_value_total += size * price
            else:  # short
                if is_sell:
                    entry_size_total += size
                    entry_value_total += size * price
                elif is_buy:
                    exit_size_total += size
                    exit_value_total += size * price
        
        entry_avg = entry_value_total / entry_size_total if entry_size_total > 0 else Decimal('0')
        exit_avg = exit_value_total / exit_size_total if exit_size_total > 0 else Decimal('0')
        
        return {
            "symbol": block["symbol"],
            "side": block["side"],
            "size": float(block.get("max_position_size", 0)),
            "entry_price": float(entry_avg),
            "close_price": float(exit_avg),
            "open_time": block["open_time"],
            "close_time": block["close_time"],
            "pnl": float(realized_pnl),  # PnL FIFO real
            "fee_total": float(total_fees),
            "fifo_queue_remaining": len(fifo_queue)
        }
    
    def calculate_funding_for_block(self, block: Dict, funding_data: Dict[str, List[Dict]]) -> float:
        """Calcula funding total para un bloque basado en su timeframe"""
        symbol = block["symbol"]
        open_time = block["open_time"]
        close_time = block["close_time"]
        
        if symbol not in funding_data:
            return 0.0
        
        total_funding = Decimal('0')
        symbol_funding = funding_data[symbol]
        
        for funding_event in symbol_funding:
            funding_time = funding_event.get("time", 0)
            if open_time <= funding_time <= close_time:
                funding_rate = Decimal(str(funding_event.get("fundingRate", "0")))
                # Usar el tamaño máximo de la posición para el cálculo del funding
                position_size = Decimal(str(block.get("max_position_size", "0")))
                
                # Calcular funding payment (tasa * tamaño)
                funding_payment = funding_rate * position_size
                total_funding += funding_payment
        
        return float(total_funding)
    
    def reconstruct_closed_positions(self, days: int = 60) -> List[Dict]:
        """Reconstruye todas las posiciones cerradas usando FIFO"""
        end_time = utc_now_ms()
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        print(f"🕐 Reconstruyendo posiciones Hyperliquid desde {datetime.fromtimestamp(start_time/1000)} hasta {datetime.fromtimestamp(end_time/1000)}")
        
        # Obtener datos
        fills = self.fetch_user_fills(start_time, end_time)
        print(f"📦 Obtenidos {len(fills)} fills de Hyperliquid")
        
        # Obtener símbolos únicos para funding
        symbols = list(set(fill.get("coin", "") for fill in fills if fill.get("coin", "")))
        funding_data = self.fetch_all_funding(symbols, start_time, end_time)
        
        # Reconstruir bloques FIFO
        blocks = self.calculate_fifo_blocks(fills)
        print(f"🔍 Identificados {len(blocks)} bloques FIFO")
        
        closed_positions = []
        
        for block in blocks:
            # Calcular PnL FIFO
            fifo_result = self.calculate_fifo_pnl(block)
            if not fifo_result:
                continue
            
            # Calcular funding
            funding_total = self.calculate_funding_for_block(block, funding_data)
            
            # Calcular realized PnL (PnL FIFO - fees + funding)
            realized_pnl = fifo_result["pnl"] - fifo_result["fee_total"] + funding_total
            
            # Calcular notional (usando precio de entrada)
            notional = fifo_result["size"] * fifo_result["entry_price"]
            
            closed_position = {
                "exchange": "hyperliquid",
                "symbol": fifo_result["symbol"],
                "side": fifo_result["side"],
                "size": fifo_result["size"],
                "entry_price": fifo_result["entry_price"],
                "close_price": fifo_result["close_price"],
                "open_time": fifo_result["open_time"] // 1000,  # Convertir a segundos
                "close_time": fifo_result["close_time"] // 1000,
                "pnl": fifo_result["pnl"],  # PnL FIFO de precio (sin fees ni funding)
                "realized_pnl": realized_pnl,  # PnL real incluyendo fees y funding
                "funding_total": funding_total,
                "fee_total": fifo_result["fee_total"],
                "notional": notional,
                "initial_margin": None,
                "leverage": None,
                "liquidation_price": None
            }
            
            closed_positions.append(closed_position)
        
        print(f"✅ Reconstruidas {len(closed_positions)} posiciones cerradas")
        return closed_positions

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
                    print(f"💾 Guardada posición {position['symbol']} - PnL: {position['realized_pnl']:.4f}")
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
    """Obtiene balances de Hyperliquid"""
    try:
        payload = {
            "type": "clearinghouseState", 
            "user": HYPERLIQUID_ACCOUNT
        }
        
        response = requests.post(HYPERLIQUID_API_URL, json=payload, timeout=30)
        data = response.json()
        
        margin_summary = data.get("marginSummary", {})
        account_value = float(margin_summary.get("accountValue", 0))
        total_margin_used = float(margin_summary.get("totalMarginUsed", 0))
        
        return {
            "exchange": "hyperliquid",
            "equity": account_value,
            "balance": account_value - total_margin_used,  # Balance disponible
            "unrealized_pnl": 0,  # Ya está incluido en accountValue
            "spot": 0,  # Hyperliquid es principalmente perpetuals
            "margin": 0,
            "futures": account_value
        }
        
    except Exception as e:
        print(f"❌ Error fetching Hyperliquid balances: {e}")
        return None