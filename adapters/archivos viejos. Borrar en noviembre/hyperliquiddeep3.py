# adapters/hyperliquid_fixed.py
import os
import sys
import time
import sqlite3
import requests
from collections import defaultdict, deque
from decimal import Decimal
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import re

# =============================================
# CONFIGURACIÓN DE PATHS - CORREGIDA
# =============================================

# Agregar el directorio padre al path para importar desde la raíz
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Ahora importar los módulos desde la raíz
try:
    from db_manager import save_closed_position
    from symbols import normalize_symbol
    from money import D, usd, quant, normalize_fee, to_float
    
    # Para time.py, importar las funciones específicas
    try:
        from time import utc_now_ms, to_ms, to_s
    except ImportError:
        # Fallback si no existen
        def utc_now_ms(): return int(time.time() * 1000)
        def to_ms(ts): t = int(float(ts)); return t if t >= 10**12 else t * 1000
        def to_s(ts):  t = int(float(ts)); return t // 1000 if t >= 10**12 else t
        
    print("✅ Módulos importados correctamente desde la raíz")
    
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    print(f"📁 Current dir: {current_dir}")
    print(f"📁 Parent dir: {parent_dir}")
    print(f"📁 Sys.path: {sys.path}")
    
    # Fallback: definir funciones esenciales localmente
    def normalize_symbol(sym: str) -> str:
        if not sym: return ""
        s = sym.upper()
        s = re.sub(r'^PERP_', '', s)
        s = re.sub(r'(_|-)?(USDT|USDC|USD|PERP)$', '', s)
        s = re.sub(r'[_-]+$', '', s)
        s = re.split(r'[_/-]', s)[0]
        return s
        
    def utc_now_ms(): return int(time.time() * 1000)
    def to_ms(ts): t = int(float(ts)); return t if t >= 10**12 else t * 1000
    def to_s(ts):  t = int(float(ts)); return t // 1000 if t >= 10**12 else t

# =============================================
# CONFIGURACIÓN HYPERLIQUID
# =============================================

HYPERLIQUID_API_URL = "https://api.hyperliquid.xyz/info"
HYPERLIQUID_ACCOUNT = "0x981690Ec51Bb332Ec6eED511C27Df325104cb461"

# =============================================
# CLASE PRINCIPAL HYPERLIQUID FIFO
# =============================================

class HyperliquidFIFO:
    def __init__(self):
        self.fills = []
        
    def fetch_user_fills(self, start_time: int, end_time: int) -> List[Dict]:
        """Obtiene todos los fills del usuario"""
        try:
            if not HYPERLIQUID_ACCOUNT:
                print("❌ HYPERLIQUID_ACCOUNT no configurado")
                return []
                
            payload = {
                "type": "userFills",
                "user": HYPERLIQUID_ACCOUNT,
            }
            
            print(f"🔍 Obteniendo fills de Hyperliquid...")
            print(f"   Cuenta: {HYPERLIQUID_ACCOUNT}")
            
            response = requests.post(HYPERLIQUID_API_URL, json=payload, timeout=30)
            
            print(f"📡 Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ Error HTTP {response.status_code}: {response.text}")
                return []
            
            data = response.json()
            
            if isinstance(data, list):
                # Filtrar solo perpetuals
                perpetual_fills = [fill for fill in data if not fill.get("coin", "").startswith("@")]
                
                # Filtrar por rango de tiempo
                filtered_fills = []
                for fill in perpetual_fills:
                    fill_time = fill.get("time", 0)
                    if start_time <= fill_time <= end_time:
                        filtered_fills.append(fill)
                
                print(f"✅ {len(perpetual_fills)} fills totales, {len(filtered_fills)} en rango")
                
                # Mostrar primeros fills para debug
                if filtered_fills:
                    print("📋 Primeros 3 fills:")
                    for i, fill in enumerate(filtered_fills[:3]):
                        symbol = fill.get('coin', 'Unknown')
                        direction = fill.get('dir', 'Unknown')
                        size = fill.get('sz', 0)
                        fill_time = fill.get('time', 0)
                        time_str = datetime.fromtimestamp(fill_time/1000).strftime("%Y-%m-%d %H:%M:%S")
                        print(f"   {i+1}. {symbol} - {direction} - Size: {size} - Time: {time_str}")
                
                return filtered_fills
            else:
                print(f"❌ Respuesta inesperada: {data}")
                return []
            
        except Exception as e:
            print(f"❌ Error fetching fills: {e}")
            import traceback
            traceback.print_exc()
            return []

    def test_api_access(self):
        """Prueba si podemos acceder a la API"""
        try:
            if not HYPERLIQUID_ACCOUNT:
                return False
                
            payload = {
                "type": "clearinghouseState",
                "user": HYPERLIQUID_ACCOUNT
            }
            
            response = requests.post(HYPERLIQUID_API_URL, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                positions = data.get('assetPositions', [])
                print(f"✅ API accesible - {len(positions)} posiciones encontradas")
                return True
            else:
                print(f"❌ API no accesible - Status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing API: {e}")
            return False

    def reconstruct_closed_positions(self, days: int = 30) -> List[Dict]:
        """Reconstruye posiciones cerradas"""
        print("🚀 Iniciando reconstrucción FIFO...")
        
        # Probar acceso a API
        if not self.test_api_access():
            print("❌ No se puede acceder a la API de Hyperliquid")
            return []
        
        end_time = utc_now_ms()
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        fills = self.fetch_user_fills(start_time, end_time)
        
        if not fills:
            print("❌ No se encontraron fills")
            print("💡 Posibles causas:")
            print("   - La cuenta no tiene trades en el período")
            print("   - La cuenta es incorrecta")
            print("   - Problemas de red/API")
            return []
        
        print(f"🎯 Procesando {len(fills)} fills...")
        
        # Implementación simple de FIFO por ahora
        # Agrupar por símbolo
        fills_by_symbol = defaultdict(list)
        for fill in fills:
            symbol = fill.get("coin", "")
            if symbol:
                fills_by_symbol[symbol].append(fill)
        
        # Ordenar por tiempo
        for symbol in fills_by_symbol:
            fills_by_symbol[symbol].sort(key=lambda x: x.get("time", 0))
        
        positions = []
        
        for symbol, symbol_fills in fills_by_symbol.items():
            print(f"🔍 Procesando {symbol}: {len(symbol_fills)} fills")
            
            # Aquí iría la lógica FIFO completa
            # Por ahora, creamos una posición simple para testing
            if len(symbol_fills) >= 2:
                first_fill = symbol_fills[0]
                last_fill = symbol_fills[-1]
                
                position = {
                    "exchange": "hyperliquid",
                    "symbol": symbol,
                    "side": "short" if "Short" in first_fill.get("dir", "") else "long",
                    "size": float(first_fill.get("sz", 0)),
                    "entry_price": float(first_fill.get("px", 0)),
                    "close_price": float(last_fill.get("px", 0)),
                    "open_time": first_fill.get("time", 0) // 1000,
                    "close_time": last_fill.get("time", 0) // 1000,
                    "pnl": 0.0,  # Placeholder
                    "realized_pnl": 0.0,  # Placeholder
                    "funding_total": 0.0,
                    "fee_total": -abs(float(first_fill.get("fee", 0))),
                    "notional": float(first_fill.get("sz", 0)) * float(first_fill.get("px", 0)),
                    "initial_margin": 0.0,
                    "leverage": 1.0,
                    "liquidation_price": 0.0
                }
                
                positions.append(position)
                print(f"   ✅ Posición creada para {symbol}")
        
        print(f"🎉 Reconstrucción completada: {len(positions)} posiciones")
        return positions
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

# =============================================
# FUNCIONES PÚBLICAS
# =============================================

def save_hyperliquid_closed_positions(db_path: str = "portfolio.db", days: int = 30, debug: bool = False) -> int:
    """Guarda posiciones cerradas"""
    print("💾 Guardando posiciones Hyperliquid...")
    
    fifo = HyperliquidFIFO()
    positions = fifo.reconstruct_closed_positions(days)
    
    saved = 0
    for position in positions:
        try:
            save_closed_position(position)
            saved += 1
            print(f"✅ Guardada: {position['symbol']}")
        except Exception as e:
            print(f"❌ Error guardando {position['symbol']}: {e}")
    
    print(f"📊 Total guardadas: {saved}")
    return saved

def debug_hyperliquid():
    """Función de debug"""
    print("🔧 DEBUG Hyperliquid")
    print("=" * 50)
    
    if not HYPERLIQUID_ACCOUNT:
        print("❌ HYPERLIQUID_ACCOUNT no configurado")
        return
    
    fifo = HyperliquidFIFO()
    positions = fifo.reconstruct_closed_positions(days=30)
    
    for pos in positions:
        print(f"\n🎯 {pos['symbol']} {pos['side']}")
        print(f"   Size: {pos['size']}")
        print(f"   Entry: ${pos['entry_price']}")
        print(f"   Close: ${pos['close_price']}")
        print(f"   PnL: ${pos['pnl']}")

# =============================================
# EJECUCIÓN DIRECTA
# =============================================
if __name__ == "__main__":
    debug_hyperliquid()