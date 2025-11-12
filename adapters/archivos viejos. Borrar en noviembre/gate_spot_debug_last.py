# gate_spot_debug.py
# -*- coding: utf-8 -*-
"""
Debug autónomo para Gate.io Spot Trades
Ejecutar directamente en Spyder para analizar datos raw del endpoint
"""

import os
import sys
import time
import requests
import hashlib
import hmac
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import defaultdict

# =========================
# 🎛️ TOGGLES DE CONFIGURACIÓN
# =========================
DEBUG_MODE = True
SHOW_COMPACT = True          # Mostrar filas compactas
MAX_ROWS_DISPLAY = 50        # Límite de filas a mostrar
DAYS_BACK = 60               # Días a consultar (máx 30 por ventana)
SPECIFIC_SYMBOLS = []        # Ejemplo: ["BTC_USDT", "ETH_USDT"] - vacío = todos
INCLUDE_BTC_ETH = False      # Incluir trades de BTC y ETH

# =========================
# 🔐 CONFIGURACIÓN API
# =========================
GATE_API_KEY = "fa605d48ce1fb0bd9c70a6f2fa517d9f"
GATE_SECRET_KEY = "a2aac7e8d847803b54df95c59b2a5e1290fd121ff855f9aadd09ac9d64aeaa99"
GATE_BASE_URL = "https://api.gateio.ws"
GATE_PREFIX = "/api/v4"

def gen_sign(method, url, query_string=None, payload_string=None):
    """Generar firma para autenticación Gate.io"""
    t = time.time()
    m = hashlib.sha512()
    m.update((payload_string or "").encode('utf-8'))
    hashed_payload = m.hexdigest()
    s = '%s\n%s\n%s\n%s\n%s' % (method, url, query_string or "", hashed_payload, t)
    sign = hmac.new(GATE_SECRET_KEY.encode('utf-8'), s.encode('utf-8'), hashlib.sha512).hexdigest()
    return {'KEY': GATE_API_KEY, 'Timestamp': str(t), 'SIGN': sign}

def gate_request(method: str, endpoint: str, params: Dict = None) -> List[Dict]:
    """Realizar petición autenticada a Gate.io"""
    if not GATE_API_KEY or not GATE_SECRET_KEY:
        print("❌ ERROR: Faltan API_KEY o SECRET_KEY en variables de entorno")
        return []
    
    url = GATE_PREFIX + endpoint
    query_string = '&'.join([f"{k}={v}" for k, v in (params or {}).items()])
    
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    sign_headers = gen_sign(method, url, query_string)
    headers.update(sign_headers)
    
    full_url = f"{GATE_BASE_URL}{url}?{query_string}" if query_string else f"{GATE_BASE_URL}{url}"
    
    try:
        response = requests.request(method, full_url, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            return []
    except Exception as e:
        print(f"❌ Error en petición: {e}")
        return []

@dataclass
class TradeRecord:
    """Estructura para registros de trades"""
    id: str
    create_time: str
    create_time_ms: str
    currency_pair: str
    side: str
    role: str
    amount: str
    price: str
    order_id: str
    fee: str
    fee_currency: str
    sequence_id: str
    
    def to_compact_row(self) -> str:
        """Convertir a fila compacta"""
        time_str = datetime.fromtimestamp(int(self.create_time), tz=timezone.utc).strftime("%m-%d %H:%M")
        return (f"│ {time_str:12} │ {self.currency_pair:12} │ {self.side:4} │ "
                f"{float(self.amount):>10.4f} │ {float(self.price):>10.6f} │ "
                f"{self.fee_currency:6} │ {float(self.fee):>8.6f} │ {self.id:12} │")

def fetch_trades_window(from_ts: int, to_ts: int, symbol: str = "") -> List[TradeRecord]:
    """Obtener trades en una ventana temporal específica"""
    params = {
        'limit': 1000,
        'from': from_ts,
        'to': to_ts,
    }
    
    if symbol:
        params['currency_pair'] = symbol
    
    print(f"   📥 Consultando: {_fmt_ts(from_ts)} -> {_fmt_ts(to_ts)} {f'[{symbol}]' if symbol else ''}")
    
    raw_data = gate_request('GET', '/spot/my_trades', params)
    trades = []
    
    for item in raw_data:
        trades.append(TradeRecord(
            id=item.get('id', ''),
            create_time=item.get('create_time', ''),
            create_time_ms=item.get('create_time_ms', ''),
            currency_pair=item.get('currency_pair', ''),
            side=item.get('side', ''),
            role=item.get('role', ''),
            amount=item.get('amount', '0'),
            price=item.get('price', '0'),
            order_id=item.get('order_id', ''),
            fee=item.get('fee', '0'),
            fee_currency=item.get('fee_currency', ''),
            sequence_id=item.get('sequence_id', '')
        ))
    
    return trades

def _fmt_ts(timestamp: int) -> str:
    """Formatear timestamp a string legible"""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%m-%d %H:%M")

def should_include_trade(trade: TradeRecord) -> bool:
    """Determinar si incluir el trade según los filtros"""
    if not INCLUDE_BTC_ETH:
        pair_upper = trade.currency_pair.upper()
        if 'BTC_' in pair_upper or '_BTC' in pair_upper:
            return False
        if 'ETH_' in pair_upper or '_ETH' in pair_upper:
            return False
    
    if SPECIFIC_SYMBOLS and trade.currency_pair not in SPECIFIC_SYMBOLS:
        return False
        
    return True

def print_compact_header():
    """Imprimir header compacto"""
    print("┌──────────────┬────────────┬──────┬────────────┬────────────┬────────┬──────────┬────────────┐")
    print("│     Time     │   Pair     │ Side │   Amount   │   Price    │ Fee Ccy│   Fee    │    ID      │")
    print("├──────────────┼────────────┼──────┼────────────┼────────────┼────────┼──────────┼────────────┤")

def print_compact_footer():
    """Imprimir footer compacto"""
    print("└──────────────┴────────────┴──────┴────────────┴────────────┴────────┴──────────┴────────────┘")

def main_debug():
    """Función principal de debug"""
    print("🔧 GATE.IO SPOT TRADES DEBUG")
    print("=" * 80)
    
    if not GATE_API_KEY:
        print("❌ ERROR: Configura GATEIO_API_KEY en variables de entorno")
        return
    
    # Calcular rangos de tiempo
    now_ts = int(time.time())
    target_from_ts = now_ts - (DAYS_BACK * 24 * 3600)
    
    print(f"📊 Configuración:")
    print(f"   • Días: {DAYS_BACK}")
    print(f"   • Símbolos: {SPECIFIC_SYMBOLS if SPECIFIC_SYMBOLS else 'TODOS'}")
    print(f"   • Incluir BTC/ETH: {INCLUDE_BTC_ETH}")
    print(f"   • Mostrar compacto: {SHOW_COMPACT}")
    print(f"   • Límite filas: {MAX_ROWS_DISPLAY}")
    print()
    
    all_trades = []
    
    # Estrategia: ventanas de 30 días (máximo permitido por API)
    window_days = 30
    current_to = now_ts
    
    symbols_to_query = SPECIFIC_SYMBOLS if SPECIFIC_SYMBOLS else [""]
    
    for symbol in symbols_to_query:
        print(f"🎯 Procesando símbolo: {symbol if symbol else 'TODOS'}")
        
        current_to = now_ts
        while current_to > target_from_ts:
            window_from = max(target_from_ts, current_to - (window_days * 24 * 3600))
            
            try:
                window_trades = fetch_trades_window(window_from, current_to, symbol)
                filtered_trades = [t for t in window_trades if should_include_trade(t)]
                all_trades.extend(filtered_trades)
                
                print(f"   ✅ Ventana: {len(window_trades)} trades → {len(filtered_trades)} filtrados")
                
            except Exception as e:
                print(f"   ❌ Error en ventana: {e}")
            
            # Mover ventana hacia atrás
            current_to = window_from - 1
            
            # Pequeña pausa
            time.sleep(0.1)
    
    # Ordenar por tiempo
    all_trades.sort(key=lambda x: int(x.create_time))
    
    print()
    print("📊 RESULTADOS:")
    print(f"   • Total trades obtenidos: {len(all_trades)}")
    
    # Estadísticas por símbolo
    symbol_stats = defaultdict(int)
    for trade in all_trades:
        symbol_stats[trade.currency_pair] += 1
    
    print(f"   • Distribución por símbolo:")
    for symbol, count in sorted(symbol_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"     {symbol}: {count} trades")
    
    print()
    
    # Mostrar trades
    if SHOW_COMPACT and all_trades:
        display_trades = all_trades[-MAX_ROWS_DISPLAY:]  # Los más recientes
        print(f"🔄 Mostrando {len(display_trades)} trades más recientes (compacto):")
        print_compact_header()
        
        for trade in display_trades:
            print(trade.to_compact_row())
        
        print_compact_footer()
    
    elif all_trades:
        # Mostrar formato extendido (limitado)
        display_trades = all_trades[-min(10, len(all_trades)):]  # Últimos 10
        print(f"🔄 Mostrando {len(display_trades)} trades (formato extendido):")
        for i, trade in enumerate(display_trades):
            print(f"\n[{i+1}] {trade.currency_pair} {trade.side.upper()}")
            print(f"   Time:    {_fmt_ts(int(trade.create_time))}")
            print(f"   Amount:  {trade.amount}")
            print(f"   Price:   {trade.price}")
            print(f"   Fee:     {trade.fee} {trade.fee_currency}")
            print(f"   ID:      {trade.id}")
    
    # Resumen final
    print()
    print("🎯 RESUMEN EJECUTIVO:")
    print(f"   • Período: {DAYS_BACK} días")
    print(f"   • Total trades: {len(all_trades)}")
    print(f"   • Símbolos únicos: {len(symbol_stats)}")
    
    if all_trades:
        first_trade = all_trades[0]
        last_trade = all_trades[-1]
        print(f"   • Rango temporal: {_fmt_ts(int(first_trade.create_time))} - {_fmt_ts(int(last_trade.create_time))}")
    
    print("=" * 80)

# =========================
# 🚀 EJECUCIÓN AUTOMÁTICA
# =========================
if __name__ == "__main__":
    print("🚀 Iniciando Debug de Gate.io Spot Trades...")
    main_debug()
    print("✅ Debug completado")

