# adapters/hyperliquidv2.py
import os
import time
import sqlite3
import requests
from collections import defaultdict, deque
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd

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
    
    def parse_timestamp(self, timestamp_str: str) -> int:
        """Convierte timestamp de Hyperliquid a milisegundos"""
        try:
            # Formato: "16/10/2025 - 11:25:00"
            dt = datetime.strptime(timestamp_str, "%d/%m/%Y - %H:%M:%S")
            return int(dt.timestamp() * 1000)
        except Exception as e:
            print(f"❌ Error parsing timestamp {timestamp_str}: {e}")
            return 0

    def load_fills_from_csv(self, csv_path: str = "trade_history.csv") -> List[Dict]:
        """Carga fills desde CSV de Hyperliquid"""
        import csv
        
        fills = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row.get('time') or not row.get('coin'):
                        continue
                    
                    # Parsear timestamp
                    timestamp_ms = self.parse_timestamp(row['time'])
                    
                    # Determinar side y dirección
                    direction = row.get('dir', '')
                    if 'Open Short' in direction:
                        side = 'A'  # Ask = Short
                        action = 'open'
                    elif 'Close Short' in direction:
                        side = 'B'  # Bid = Buy (para cerrar short)
                        action = 'close'
                    else:
                        continue
                    
                    fill_data = {
                        'coin': row['coin'],
                        'side': side,
                        'dir': direction,
                        'px': float(row.get('px', 0)),
                        'sz': float(row.get('sz', 0)),
                        'time': timestamp_ms,
                        'fee': abs(float(row.get('fee', 0))),  # Fees siempre positivos en CSV
                        'closedPnl': float(row.get('closedPnl', 0)),
                        'ntl': float(row.get('ntl', 0))
                    }
                    fills.append(fill_data)
            
            print(f"📦 Cargados {len(fills)} fills desde CSV")
            return fills
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return []

    def calculate_fifo_blocks(self, fills: List[Dict]) -> List[Dict]:
        """Reconstruye bloques FIFO a partir de fills - VERSIÓN CORREGIDA"""
        # Agrupar fills por símbolo
        fills_by_symbol = defaultdict(list)
        for fill in fills:
            symbol = fill.get("coin", "")
            if symbol:
                fills_by_symbol[symbol].append(fill)
        
        # Ordenar todos los fills por timestamp
        for symbol in fills_by_symbol:
            fills_by_symbol[symbol].sort(key=lambda x: x.get("time", 0))
        
        blocks = []
        
        for symbol, symbol_fills in fills_by_symbol.items():
            print(f"🔍 Procesando símbolo {symbol}: {len(symbol_fills)} fills")
            
            # Usar deque para FIFO
            open_lots = deque()
            current_block = {
                "symbol": symbol,
                "fills": [],
                "open_time": None,
                "close_time": None,
                "side": None,
                "total_size": 0.0
            }
            
            for fill in symbol_fills:
                side = fill.get("side", "")
                size = float(fill.get("sz", 0))
                price = float(fill.get("px", 0))
                direction = fill.get("dir", "")
                
                # Determinar si es apertura o cierre
                is_open = "Open" in direction
                is_close = "Close" in direction
                
                if is_open:
                    # Es una apertura - agregar al lote
                    if current_block["side"] is None:
                        # Determinar side del bloque
                        if "Short" in direction:
                            current_block["side"] = "short"
                        elif "Long" in direction:
                            current_block["side"] = "long"
                    
                    if current_block["open_time"] is None:
                        current_block["open_time"] = fill.get("time", 0)
                    
                    # Agregar lote abierto
                    open_lots.append({
                        "size": size,
                        "price": price,
                        "time": fill.get("time", 0),
                        "fee": float(fill.get("fee", 0))
                    })
                    current_block["total_size"] += size
                    
                elif is_close:
                    # Es un cierre - procesar contra lotes abiertos
                    close_size_remaining = size
                    close_pnl = 0.0
                    
                    while close_size_remaining > 0 and open_lots:
                        current_lot = open_lots[0]
                        
                        if current_lot["size"] <= close_size_remaining:
                            # Cerrar lote completo
                            if current_block["side"] == "short":
                                # Para short: PnL = (entry - close) * size
                                close_pnl += (current_lot["price"] - price) * current_lot["size"]
                            else:  # long
                                # Para long: PnL = (close - entry) * size
                                close_pnl += (price - current_lot["price"]) * current_lot["size"]
                            
                            close_size_remaining -= current_lot["size"]
                            open_lots.popleft()
                        else:
                            # Cerrar parte del lote
                            if current_block["side"] == "short":
                                close_pnl += (current_lot["price"] - price) * close_size_remaining
                            else:  # long
                                close_pnl += (price - current_lot["price"]) * close_size_remaining
                            
                            current_lot["size"] -= close_size_remaining
                            close_size_remaining = 0
                    
                    # Actualizar close_time
                    current_block["close_time"] = fill.get("time", 0)
                    
                    # Agregar fill al bloque
                    current_block["fills"].append(fill)
                
                # Si no quedan lotes abiertos, cerrar el bloque
                if not open_lots and current_block["fills"]:
                    blocks.append(current_block.copy())
                    
                    # Iniciar nuevo bloque
                    current_block = {
                        "symbol": symbol,
                        "fills": [],
                        "open_time": None,
                        "close_time": None,
                        "side": None,
                        "total_size": 0.0
                    }
            
            # Manejar bloque final si hay lotes abiertos
            if current_block["fills"] and current_block["total_size"] > 0:
                blocks.append(current_block)
        
        print(f"🎯 Identificados {len(blocks)} bloques FIFO")
        return blocks

    def calculate_position_metrics(self, block: Dict) -> Dict[str, Any]:
        """Calcula métricas detalladas para un bloque - VERSIÓN CORREGIDA"""
        if not block["fills"]:
            return {}
        
        symbol = block["symbol"]
        side = block["side"]
        total_size = block["total_size"]
        
        # Separar fills de apertura y cierre
        open_fills = [f for f in block["fills"] if "Open" in f.get("dir", "")]
        close_fills = [f for f in block["fills"] if "Close" in f.get("dir", "")]
        
        # Calcular precios promedios ponderados
        if open_fills:
            entry_price = sum(f["px"] * f["sz"] for f in open_fills) / sum(f["sz"] for f in open_fills)
        else:
            entry_price = 0.0
            
        if close_fills:
            close_price = sum(f["px"] * f["sz"] for f in close_fills) / sum(f["sz"] for f in close_fills)
        else:
            close_price = 0.0
        
        # Calcular PnL de precio
        if side == "short":
            price_pnl = (entry_price - close_price) * total_size
        else:  # long
            price_pnl = (close_price - entry_price) * total_size
        
        # Calcular fees totales (siempre negativos)
        total_fees = -sum(abs(f.get("fee", 0)) for f in block["fills"])
        
        # Calcular funding (0 por ahora, ya que no tenemos datos de funding en CSV)
        total_funding = 0.0
        
        # Calcular realized PnL
        realized_pnl = price_pnl + total_funding + total_fees
        
        # Calcular notional
        notional = total_size * entry_price
        
        return {
            "exchange": "hyperliquid",
            "symbol": symbol,
            "side": side,
            "size": total_size,
            "entry_price": entry_price,
            "close_price": close_price,
            "open_time": block["open_time"] // 1000,  # Convertir a segundos
            "close_time": block["close_time"] // 1000,
            "pnl": price_pnl,
            "realized_pnl": realized_pnl,
            "funding_total": total_funding,
            "fee_total": total_fees,
            "notional": notional,
            "initial_margin": notional / 3.0,  # Asumir leverage 3x
            "leverage": 3.0,
            "liquidation_price": 0.0,
            "fills_count": len(block["fills"])
        }

    def reconstruct_closed_positions(self, days: int = 60, use_csv: bool = True) -> List[Dict]:
        """Reconstruye todas las posiciones cerradas usando FIFO - VERSIÓN CORREGIDA"""
        print("🚀 Iniciando reconstrucción FIFO de posiciones Hyperliquid...")
        
        if use_csv:
            # Usar datos del CSV
            fills = self.load_fills_from_csv()
        else:
            # Usar API (no implementado completamente para este ejemplo)
            end_time = utc_now_ms()
            start_time = end_time - (days * 24 * 60 * 60 * 1000)
            fills = self.fetch_user_fills(start_time, end_time)
        
        if not fills:
            print("❌ No se encontraron fills para procesar")
            return []
        
        # Reconstruir bloques FIFO
        blocks = self.calculate_fifo_blocks(fills)
        
        closed_positions = []
        
        for i, block in enumerate(blocks):
            print(f"📊 Procesando bloque {i+1}/{len(blocks)}: {block['symbol']} {block['side']}")
            
            position_data = self.calculate_position_metrics(block)
            if position_data and position_data["size"] > 0.001:
                closed_positions.append(position_data)
                
                # Debug detallado
                print(f"   ✅ Reconstruída:")
                print(f"      Size: {position_data['size']:.2f}")
                print(f"      Entry: ${position_data['entry_price']:.6f}")
                print(f"      Close: ${position_data['close_price']:.6f}")
                print(f"      PnL: ${position_data['pnl']:.6f}")
                print(f"      Realized: ${position_data['realized_pnl']:.6f}")
                print(f"      Fees: ${position_data['fee_total']:.6f}")
        
        print(f"🎉 Reconstrucción completada: {len(closed_positions)} posiciones")
        return closed_positions

# Funciones públicas del adapter
def save_hyperliquid_closed_positions(db_path: str = "portfolio.db", days: int = 60, debug: bool = False) -> int:
    """Guarda posiciones cerradas de Hyperliquid usando reconstrucción FIFO"""
    if debug:
        print("🔍 [DEBUG] Iniciando reconstrucción FIFO Hyperliquid...")
    
    fifo = HyperliquidFIFO()
    
    # Usar CSV para testing, luego cambiar a False para usar API
    closed_positions = fifo.reconstruct_closed_positions(days, use_csv=True)
    
    saved_count = 0
    duplicate_count = 0
    
    for position in closed_positions:
        try:
            # Verificar si ya existe
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("""
                SELECT COUNT(*) FROM closed_positions 
                WHERE exchange = ? AND symbol = ? AND close_time = ? AND ABS(size - ?) < 0.001
            """, (position["exchange"], position["symbol"], position["close_time"], position["size"]))
            
            if cur.fetchone()[0] == 0:
                # Marcar size como bloqueado para evitar recálculo
                position["_lock_size"] = True
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
    closed_positions = fifo.reconstruct_closed_positions(days, use_csv=True)
    
    for i, pos in enumerate(closed_positions):
        if symbol and pos["symbol"] != symbol:
            continue
            
        print(f"\n🎯 Posición {i+1}:")
        print(f"   Símbolo: {pos['symbol']}")
        print(f"   Side: {pos['side']}")
        print(f"   Size: {pos['size']:.2f}")
        print(f"   Entry: ${pos['entry_price']:.6f}")
        print(f"   Close: ${pos['close_price']:.6f}")
        print(f"   PnL (precio): ${pos['pnl']:.6f}")
        print(f"   Realized PnL: ${pos['realized_pnl']:.6f}")
        print(f"   Fees: ${pos['fee_total']:.6f}")
        print(f"   Funding: ${pos['funding_total']:.6f}")
        print(f"   Open: {datetime.fromtimestamp(pos['open_time'])}")
        print(f"   Close: {datetime.fromtimestamp(pos['close_time'])}")

# Funciones para el dashboard (placeholders por ahora)
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
    """Obtiene balances de Hyperliquid - VERSIÓN CORREGIDA"""
    try:
        payload = {
            "type": "clearinghouseState", 
            "user": HYPERLIQUID_ACCOUNT
        }
        
        response = requests.post(HYPERLIQUID_API_URL, json=payload, timeout=30)
        data = response.json()
        
        if not data:
            print("❌ Hyperliquid: Respuesta vacía de la API")
            return None
            
        margin_summary = data.get("marginSummary", {})
        cross_margin_summary = data.get("crossMarginSummary", {})
        
        # Usar crossMarginSummary si está disponible, sino marginSummary
        if cross_margin_summary and cross_margin_summary.get("accountValue"):
            account_value = float(cross_margin_summary.get("accountValue", 0))
            total_margin_used = float(cross_margin_summary.get("totalMarginUsed", 0))
        else:
            account_value = float(margin_summary.get("accountValue", 0))
            total_margin_used = float(margin_summary.get("totalMarginUsed", 0))
        
        # Calcular unrealized PnL desde las posiciones abiertas
        unrealized_pnl = 0.0
        try:
            open_positions = fetch_hyperliquid_open_positions()
            unrealized_pnl = sum(pos.get("unrealized_pnl", 0.0) for pos in open_positions)
        except Exception as e:
            print(f"⚠️ Error calculando unrealized PnL: {e}")
        
        available_balance = max(0, account_value - total_margin_used - unrealized_pnl)
        
        result = {
            "exchange": "hyperliquid",
            "equity": account_value,
            "balance": available_balance,
            "unrealized_pnl": unrealized_pnl,
            "initial_margin": total_margin_used,
            "spot": 0.0,  # Hyperliquid es principalmente perpetuals
            "margin": 0.0,
            "futures": account_value
        }
        
        print(f"✅ Hyperliquid balances - Equity: {account_value:.2f}, Available: {available_balance:.2f}")
        return result
        
    except Exception as e:
        print(f"❌ Error fetching Hyperliquid balances: {e}")
        return None
