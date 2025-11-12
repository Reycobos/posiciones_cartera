# adapters/mexc_websocket.py
from __future__ import annotations
import os
import time
import hmac
import hashlib
import json
import math
import asyncio
import websockets
import re  # <-- AÑADIDO
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, quote
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

__all__ = [
    "fetch_mexc_open_positions_websocket",
    "MEXCWebSocketClient",
    "debug_websocket_raw_data"
]

# =========================
# Configuración
# =========================
MEXC_WS_URL = "wss://contract.mexc.com/edge"
MEXC_API_KEY = "mx0vglOEFTy9klFKJo"
MEXC_API_SECRET = "1f45cf4ac48148419b59298352c45ef0"

# =========================
# Normalización de símbolo (misma lógica que el adapter original)
# =========================
SPECIAL_SYMBOL_MAP = {
    "OPENLEDGER": "OPEN",
}

def normalize_symbol(sym: str) -> str:
    if not sym:
        return ""
    s = sym.upper().strip()
    s = re.sub(r'^PERP_', '', s)
    s = re.sub(r'(_|-)?(USDT|USDC|USD|PERP)$', '', s)
    s = re.sub(r'[_-]+$', '', s)
    base = re.split(r'[_-]', s)[0]
    base = SPECIAL_SYMBOL_MAP.get(base, base)
    return base

# =========================
# Cliente WebSocket
# =========================
class MEXCWebSocketClient:
    def __init__(self):
        self.positions = {}
        self.connected = False
        self.websocket = None
        self.last_pong = None
        
    def _generate_signature(self) -> Tuple[str, str]:
        """Genera signature para autenticación WebSocket"""
        req_time = str(int(time.time() * 1000))
        sign_target = f"{MEXC_API_KEY}{req_time}"
        signature = hmac.new(
            MEXC_API_SECRET.encode("utf-8"),
            sign_target.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return req_time, signature
    
    async def connect(self):
        """Establece conexión WebSocket y autentica"""
        try:
            self.websocket = await websockets.connect(MEXC_WS_URL)
            self.connected = True
            
            # Autenticación
            req_time, signature = self._generate_signature()
            auth_msg = {
                "method": "login",
                "param": {
                    "apiKey": MEXC_API_KEY,
                    "reqTime": req_time,
                    "signature": signature
                }
            }
            await self.websocket.send(json.dumps(auth_msg))
            
            # Esperar respuesta de autenticación
            auth_response = await self.websocket.recv()
            auth_data = json.loads(auth_response)
            
            if auth_data.get("channel") == "rs.login":
                print("✅ Autenticación WebSocket exitosa")
            else:
                print(f"❌ Error en autenticación: {auth_data}")
                return False
            
            # Suscribir a posiciones
            subscribe_msg = {
                "method": "personal.filter",
                "param": {
                    "filters": [
                        {
                            "filter": "position"
                        }
                    ]
                }
            }
            await self.websocket.send(json.dumps(subscribe_msg))
            
            return True
            
        except Exception as e:
            print(f"❌ Error conectando WebSocket: {e}")
            self.connected = False
            return False
    
    async def listen_for_positions(self, timeout: int = 10):
        """Escucha mensajes de posiciones por WebSocket"""
        if not self.connected or not self.websocket:
            print("❌ WebSocket no conectado")
            return
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < timeout:
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    
                    # Procesar diferentes tipos de mensajes
                    channel = data.get("channel")
                    
                    if channel == "push.personal.position":
                        await self._handle_position_update(data)
                    elif channel == "pong":
                        self.last_pong = time.time()
                    elif channel == "rs.error":
                        print(f"❌ Error del servidor: {data}")
                    
                except asyncio.TimeoutError:
                    # Enviar ping para mantener conexión
                    ping_msg = {"method": "ping"}
                    await self.websocket.send(json.dumps(ping_msg))
                    
        except Exception as e:
            print(f"❌ Error escuchando WebSocket: {e}")
    
    async def _handle_position_update(self, data: Dict[str, Any]):
        """Procesa actualización de posición"""
        position_data = data.get("data", {})
        position_id = position_data.get("positionId")
        symbol = position_data.get("symbol", "")
        state = position_data.get("state")
        
        print(f"🔧 Posición recibida: {symbol} (ID: {position_id}, Estado: {state})")
        
        # Solo procesar posiciones abiertas (state=1: holding, state=2: system holding)
        if state in [1, 2]:
            self.positions[position_id] = position_data
            print(f"✅ Posición ABIERTA guardada: {symbol}")
        else:
            # Remover posición cerrada
            if position_id in self.positions:
                del self.positions[position_id]
                print(f"🗑️ Posición CERRADA removida: {symbol}")
    
    async def close(self):
        """Cierra conexión WebSocket"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Convierte posiciones raw a formato estandarizado"""
        open_positions = []
        
        for position_id, pos_data in self.positions.items():
            try:
                # Obtener precio mark desde API REST (una sola llamada por símbolo)
                raw_sym = pos_data.get("symbol", "")
                mark_price = self._get_mark_price_sync(raw_sym)
                
                # Datos básicos
                side = "long" if int(pos_data.get("positionType", 1)) == 1 else "short"
                size = abs(float(pos_data.get("holdVol", 0)))
                entry_price = float(pos_data.get("holdAvgPrice", pos_data.get("openAvgPrice", 0)))
                
                # Calcular PnL no realizado
                unrealized_pnl = self._calculate_unrealized_pnl(
                    entry_price, mark_price, size, side
                )
                
                # Construir posición estandarizada
                position = {
                    "exchange": "mexc",
                    "symbol": normalize_symbol(raw_sym),
                    "side": side,
                    "size": size,
                    "entry_price": entry_price,
                    "mark_price": mark_price,
                    "liquidation_price": float(pos_data.get("liquidatePrice", 0)),
                    "notional": size * entry_price,
                    "unrealized_pnl": unrealized_pnl,
                    "fee": 0.0,  # No disponible en WebSocket
                    "funding_fee": float(pos_data.get("holdFee", 0)),
                    "realized_pnl": float(pos_data.get("realised", 0)),
                    "leverage": float(pos_data.get("leverage", 1)),
                    "position_id": position_id
                }
                
                open_positions.append(position)
                print(f"📊 Posición procesada: {normalize_symbol(raw_sym)} {side} {size}")
                
            except Exception as e:
                print(f"❌ Error procesando posición {position_id}: {e}")
                continue
        
        return open_positions
    
    def _calculate_unrealized_pnl(self, entry: float, mark: float, size: float, side: str) -> float:
        """Calcula PnL no realizado"""
        if any(math.isnan(x) for x in (entry, mark, size)):
            return 0.0
        if side == "short":
            return (entry - mark) * size
        return (mark - entry) * size
    
    def _get_mark_price_sync(self, symbol: str) -> float:
        """Obtiene precio mark sincrónicamente (optimizado)"""
        try:
            import requests
            # Intentar endpoint más rápido primero
            url = "https://contract.mexc.com/api/v1/contract/ticker"
            params = {"symbol": symbol.upper()}
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    ticker_data = data.get("data", {})
                    # Priorizar fairPrice, luego lastPrice
                    mark_price = ticker_data.get("fairPrice") or ticker_data.get("lastPrice")
                    if mark_price:
                        return float(mark_price)
            
            # Fallback
            return float(pos_data.get("openAvgPrice", 0))
            
        except Exception as e:
            print(f"❌ Error obteniendo precio mark para {symbol}: {e}")
            return 0.0

# =========================
# Función principal
# =========================
async def fetch_mexc_open_positions_websocket(timeout: int = 8) -> List[Dict[str, Any]]:
    """
    Obtiene posiciones abiertas via WebSocket (MUCHO más rápido)
    
    Args:
        timeout: Tiempo máximo de escucha en segundos
        
    Returns:
        Lista de posiciones abiertas estandarizadas
    """
    client = MEXCWebSocketClient()
    
    try:
        # Conectar y autenticar
        connected = await client.connect()
        if not connected:
            return []
        
        print(f"🔍 Escuchando posiciones por WebSocket ({timeout}s)...")
        
        # Escuchar por actualizaciones
        await client.listen_for_positions(timeout)
        
        # Obtener y retornar posiciones procesadas
        positions = client.get_open_positions()
        print(f"✅ WebSocket: {len(positions)} posiciones abiertas encontradas")
        
        return positions
        
    except Exception as e:
        print(f"❌ Error en WebSocket: {e}")
        return []
    finally:
        await client.close()

# =========================
# Debug: Raw WebSocket Data
# =========================
async def debug_websocket_raw_data(timeout: int = 10):
    """
    Debug: Muestra todos los datos RAW recibidos por WebSocket
    Útil para entender la estructura de los mensajes
    """
    print("🚀 INICIANDO DEBUG WEBSOCKET - DATOS RAW")
    print("=" * 60)
    
    client = MEXCWebSocketClient()
    
    try:
        connected = await client.connect()
        if not connected:
            return
        
        start_time = time.time()
        message_count = 0
        
        while time.time() - start_time < timeout:
            try:
                message = await asyncio.wait_for(client.websocket.recv(), timeout=5.0)
                data = json.loads(message)
                message_count += 1
                
                print(f"\n📨 MENSAJE #{message_count}")
                print(f"⏰ Tiempo: {datetime.now().strftime('%H:%M:%S')}")
                print("📊 CONTENIDO RAW:")
                print(json.dumps(data, indent=2, ensure_ascii=False))
                
                # Análisis específico por canal
                channel = data.get("channel")
                if channel == "push.personal.position":
                    print("🎯 ANÁLISIS POSICIÓN:")
                    pos_data = data.get("data", {})
                    for key, value in pos_data.items():
                        print(f"   {key}: {value}")
                
                print("-" * 50)
                
            except asyncio.TimeoutError:
                # Ping para mantener conexión
                ping_msg = {"method": "ping"}
                await client.websocket.send(json.dumps(ping_msg))
                print("⏱️  Timeout - enviando ping...")
                
    except Exception as e:
        print(f"❌ Error en debug: {e}")
    finally:
        await client.close()
        print(f"🔚 Debug completado. Total mensajes: {message_count}")

# =========================
# Ejecución compatible con Spyder
# =========================
def run_websocket_debug():
    """Función para ejecutar en Spyder que maneja el event loop"""
    try:
        # Intentar usar el event loop existente
        loop = asyncio.get_event_loop()
        if loop.is_running():
            print("🔄 Event loop ya está ejecutándose - usando approach alternativo")
            # Para Spyder, crear un nuevo loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        # No hay event loop, crear uno nuevo
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Ejecutar el debug
    print("🕵️  EJECUTANDO DEBUG WEBSOCKET MEXC")
    print("1. Ejecutando debug WebSocket RAW...")
    result = loop.run_until_complete(debug_websocket_raw_data(timeout=15))
    
    print("\n2. Probando obtención de posiciones...")
    positions = loop.run_until_complete(fetch_mexc_open_positions_websocket(timeout=5))
    print(f"Posiciones encontradas: {len(positions)}")
    for pos in positions:
        print(f"  - {pos['symbol']} {pos['side']} {pos['size']}")
    
    return positions

def run_websocket_positions_only():
    """Solo obtiene posiciones sin debug"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    positions = loop.run_until_complete(fetch_mexc_open_positions_websocket(timeout=8))
    return positions

if __name__ == "__main__":
    import sys
    
    # Verificar si estamos en un entorno interactivo como Spyder
    in_spyder = 'spyder' in sys.modules or 'spyder_kernels' in str(sys.modules)
    
    if in_spyder:
        print("🕵️  SPYDER DETECTADO - Ejecutando debug automático...")
        positions = run_websocket_debug()
    else:
        # CLI normal
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--debug-raw", action="store_true", help="Mostrar datos RAW de WebSocket")
        parser.add_argument("--timeout", type=int, default=10, help="Timeout en segundos")
        args = parser.parse_args()
        
        if args.debug_raw:
            asyncio.run(debug_websocket_raw_data(timeout=args.timeout))
        else:
            positions = asyncio.run(fetch_mexc_open_positions_websocket(timeout=args.timeout))
            print(f"Posiciones abiertas: {len(positions)}")
            for pos in positions:
                print(json.dumps(pos, indent=2))