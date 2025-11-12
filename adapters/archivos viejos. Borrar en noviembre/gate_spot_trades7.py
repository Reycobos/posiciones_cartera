# adapters/gate_spot_trades3.py
import sqlite3
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import os
import sys

# Agregar la carpeta utils al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.symbols import normalize_symbol
from utils.time import to_s
from utils.money import D, quant, normalize_fee, to_float

# Importar funciones de gate2 para la firma y balances
from adapters.gate2 import _request, _require_keys, _headers

def fetch_gate_spot_trades_batch(start_time: int, end_time: int, currency_pair: str = None, limit: int = 1000) -> List[Dict]:
    """
    Obtiene trades spot de Gate.io para un rango de tiempo específico
    """
    params = {
        "limit": min(limit, 1000),
        "from": start_time,
        "to": end_time
    }
    
    if currency_pair:
        params["currency_pair"] = currency_pair
    
    try:
        trades = _request("GET", "/spot/my_trades", params=params)
        return trades or []
    except Exception as e:
        print(f"❌ Error obteniendo trades spot para {start_time}-{end_time}: {e}")
        return []

def fetch_gate_spot_trades(currency_pair: str = None, limit: int = 1000, 
                          days_back: int = 60) -> List[Dict]:
    """
    Obtiene trades spot de Gate.io usando paginación por tiempo
    """
    print(f"🔍 Obteniendo trades spot de Gate.io (últimos {days_back} días)...")
    
    end_time = int(time.time())
    start_time = end_time - (days_back * 24 * 60 * 60)
    
    all_trades = []
    batch_days = 25
    current_start = start_time
    
    while current_start < end_time:
        current_end = min(current_start + (batch_days * 24 * 60 * 60), end_time)
        
        print(f"   📅 Lote: {time.strftime('%Y-%m-%d', time.gmtime(current_start))} a {time.strftime('%Y-%m-%d', time.gmtime(current_end))}")
        
        batch_trades = fetch_gate_spot_trades_batch(
            current_start, 
            current_end, 
            currency_pair, 
            limit
        )
        
        all_trades.extend(batch_trades)
        print(f"   📦 Lote: {len(batch_trades)} trades")
        
        current_start = current_end + 1
        time.sleep(0.1)
    
    print(f"📦 Total trades recibidos: {len(all_trades)}")
    return all_trades

def _should_ignore_symbol(symbol: str) -> bool:
    """
    Determina si un símbolo debe ser ignorado (BTC, ETH)
    """
    base_symbol = normalize_symbol(symbol)
    ignore_symbols = {"BTC", "ETH"}
    return base_symbol in ignore_symbols

def _is_stablecoin_pair(symbol: str) -> bool:
    """
    Determina si es un trade entre stablecoins
    """
    stable_pairs = {"USDT/USDC", "USDC/USDT", "USDT/USD", "USD/USDT", 
                   "USDC/USD", "USD/USDC"}
    return symbol.upper() in stable_pairs

def _is_stablecoin(currency: str) -> bool:
    """
    Determina si una moneda es stablecoin
    """
    stablecoins = {"USDT", "USDC", "USD"}
    return currency.upper() in stablecoins

def _convert_fee_to_usdt(fee: Decimal, fee_currency: str, price: Decimal = None) -> Decimal:
    """
    Convierte una fee a USDT
    """
    if _is_stablecoin(fee_currency):
        return fee
    elif price is not None:
        return fee * price
    else:
        return fee

def group_trades_by_position(trades: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Agrupa trades por posición (símbolo + ventana temporal)
    """
    # Ordenar trades por timestamp
    sorted_trades = sorted(trades, key=lambda x: to_s(x.get("create_time") or x.get("create_time_ms")))
    
    positions = defaultdict(list)
    current_positions = {}
    
    for trade in sorted_trades:
        symbol = trade.get("currency_pair", "")
        base_symbol = normalize_symbol(symbol)
        
        # Ignorar BTC y ETH
        if _should_ignore_symbol(symbol):
            continue
            
        timestamp = to_s(trade.get("create_time") or trade.get("create_time_ms"))
        side = trade.get("side", "").lower()
        
        # Para stablecoins, agrupar por símbolo y ventana de 10 minutos
        if _is_stablecoin_pair(symbol):
            position_key = f"stable_{symbol}"
            # Verificar si hay una posición activa dentro de 10 minutos
            if position_key in current_positions:
                last_trade = current_positions[position_key][-1]
                last_time = to_s(last_trade.get("create_time") or last_trade.get("create_time_ms"))
                if timestamp - last_time <= 600:  # 10 minutos
                    current_positions[position_key].append(trade)
                    continue
            
            # Nueva posición
            current_positions[position_key] = [trade]
            positions[position_key].append(trade)
            
        else:
            # Para tokens normales, agrupar por símbolo y dirección
            if side == "buy":
                position_key = f"buy_{base_symbol}"
            else:  # sell
                position_key = f"sell_{base_symbol}"
                
            # Verificar si hay una posición activa dentro de 30 minutos
            if position_key in current_positions:
                last_trade = current_positions[position_key][-1]
                last_time = to_s(last_trade.get("create_time") or last_trade.get("create_time_ms"))
                if timestamp - last_time <= 1800:  # 30 minutos
                    current_positions[position_key].append(trade)
                    continue
            
            # Nueva posición
            current_positions[position_key] = [trade]
            positions[position_key].append(trade)
    
    return positions

def calculate_stable_swap_position(trades: List[Dict]) -> Optional[Dict]:
    """
    Calcula una posición de stable swap
    """
    if not trades:
        return None
        
    symbol = trades[0].get("currency_pair", "")
    total_buy_amount = D('0')
    total_sell_amount = D('0')
    total_buy_value = D('0')
    total_sell_value = D('0')
    total_fees_usdt = D('0')
    
    open_time = None
    close_time = None
    
    for trade in trades:
        side = trade.get("side", "").lower()
        amount = D(trade.get("amount", "0"))
        price = D(trade.get("price", "0"))
        fee = D(trade.get("fee", "0"))
        fee_currency = trade.get("fee_currency", "")
        timestamp = to_s(trade.get("create_time") or trade.get("create_time_ms"))
        
        # Actualizar timestamps
        if open_time is None or timestamp < open_time:
            open_time = timestamp
        if close_time is None or timestamp > close_time:
            close_time = timestamp
        
        # Calcular valor del trade
        trade_value = amount * price
        
        if side == "buy":
            total_buy_amount += amount
            total_buy_value += trade_value
        else:  # sell
            total_sell_amount += amount
            total_sell_value += trade_value
        
        # Procesar fees
        fee_usdt = _convert_fee_to_usdt(fee, fee_currency, price)
        total_fees_usdt -= abs(fee_usdt)  # Fees siempre negativas
    
    # Calcular PnL para stable swap
    # PnL = (lo que recibí) - (lo que pagué)
    pnl = total_sell_value - total_buy_value
    realized_pnl = pnl + total_fees_usdt
    
    # Para stable swaps, usar el amount neto
    net_amount = abs(total_buy_amount - total_sell_amount)
    avg_price = total_buy_value / total_buy_amount if total_buy_amount > 0 else D('0')
    
    return {
        "exchange": "gate",
        "symbol": normalize_symbol(symbol),
        "side": "swapstable",
        "size": float(net_amount),
        "entry_price": float(avg_price),
        "close_price": float(avg_price),  # Para stables, es básicamente lo mismo
        "open_time": open_time,
        "close_time": close_time,
        "realized_pnl": float(realized_pnl),
        "pnl": float(pnl),
        "fees": float(total_fees_usdt),
        "funding_fee": 0,
        "notional": float(max(total_buy_value, total_sell_value)),
        "ignore_trade": 0
    }

def calculate_token_position(buy_trades: List[Dict], sell_trades: List[Dict]) -> Optional[Dict]:
    """
    Calcula una posición de token (compra + venta)
    """
    if not buy_trades or not sell_trades:
        return None
    
    symbol = buy_trades[0].get("currency_pair", "")
    base_symbol = normalize_symbol(symbol)
    
    # Procesar compras
    total_buy_amount = D('0')
    total_buy_value = D('0')
    total_buy_fees_usdt = D('0')
    buy_open_time = None
    
    for trade in buy_trades:
        amount = D(trade.get("amount", "0"))
        price = D(trade.get("price", "0"))
        fee = D(trade.get("fee", "0"))
        fee_currency = trade.get("fee_currency", "")
        timestamp = to_s(trade.get("create_time") or trade.get("create_time_ms"))
        
        # Si la fee es en el token, ajustar amount
        actual_amount = amount
        if fee_currency and fee_currency.upper() == base_symbol and fee > 0:
            actual_amount = amount - fee
        
        total_buy_amount += actual_amount
        total_buy_value += actual_amount * price
        
        # Procesar fees de compra
        fee_usdt = _convert_fee_to_usdt(fee, fee_currency, price)
        total_buy_fees_usdt -= abs(fee_usdt)
        
        if buy_open_time is None or timestamp < buy_open_time:
            buy_open_time = timestamp
    
    # Procesar ventas
    total_sell_amount = D('0')
    total_sell_value = D('0')
    total_sell_fees_usdt = D('0')
    sell_close_time = None
    
    for trade in sell_trades:
        amount = D(trade.get("amount", "0"))
        price = D(trade.get("price", "0"))
        fee = D(trade.get("fee", "0"))
        fee_currency = trade.get("fee_currency", "")
        timestamp = to_s(trade.get("create_time") or trade.get("create_time_ms"))
        
        total_sell_amount += amount
        total_sell_value += amount * price
        
        # Procesar fees de venta
        fee_usdt = _convert_fee_to_usdt(fee, fee_currency, price)
        total_sell_fees_usdt -= abs(fee_usdt)
        
        if sell_close_time is None or timestamp > sell_close_time:
            sell_close_time = timestamp
    
    # Verificar que tenemos suficiente inventory para vender
    if total_sell_amount > total_buy_amount:
        print(f"⚠️  VENTA excede compra para {base_symbol}: {float(total_sell_amount)} > {float(total_buy_amount)}")
        # Ajustar a la cantidad disponible
        total_sell_amount = total_buy_amount
        # Recalcular valor de venta proporcional
        total_sell_value = total_sell_value * (total_buy_amount / total_sell_amount)
    
    # Calcular métricas
    avg_entry_price = total_buy_value / total_buy_amount if total_buy_amount > 0 else D('0')
    avg_close_price = total_sell_value / total_sell_amount if total_sell_amount > 0 else D('0')
    
    # PnL de precio
    pnl_price = (avg_close_price - avg_entry_price) * total_sell_amount
    
    # Fees totales
    total_fees = total_buy_fees_usdt + total_sell_fees_usdt
    
    # PnL realizado
    realized_pnl = pnl_price + total_fees
    
    return {
        "exchange": "gate",
        "symbol": base_symbol,
        "side": "spotbuy",
        "size": float(total_sell_amount),
        "entry_price": float(avg_entry_price),
        "close_price": float(avg_close_price),
        "open_time": buy_open_time,
        "close_time": sell_close_time,
        "realized_pnl": float(realized_pnl),
        "pnl": float(pnl_price),
        "fees": float(total_fees),
        "funding_fee": 0,
        "notional": float(total_sell_value),
        "ignore_trade": 0
    }

def process_spot_positions(trades: List[Dict]) -> List[Dict]:
    """
    Procesa todos los trades y calcula posiciones
    """
    print("🔄 Procesando posiciones spot...")
    
    # Agrupar trades por posición
    grouped_trades = group_trades_by_position(trades)
    
    positions = []
    processed_symbols = set()
    
    # Procesar stable swaps primero
    for position_key, position_trades in grouped_trades.items():
        if position_key.startswith("stable_"):
            stable_position = calculate_stable_swap_position(position_trades)
            if stable_position:
                positions.append(stable_position)
                symbol = stable_position["symbol"]
                print(f"💱 Stable swap {symbol}: PnL {stable_position['realized_pnl']:.4f}")
    
    # Procesar tokens normales
    for position_key, position_trades in grouped_trades.items():
        if position_key.startswith("buy_"):
            base_symbol = position_key.replace("buy_", "")
            
            # Evitar procesar duplicados
            if base_symbol in processed_symbols:
                continue
                
            # Buscar ventas correspondientes
            sell_key = f"sell_{base_symbol}"
            sell_trades = grouped_trades.get(sell_key, [])
            
            if sell_trades:
                token_position = calculate_token_position(position_trades, sell_trades)
                if token_position:
                    positions.append(token_position)
                    processed_symbols.add(base_symbol)
                    print(f"💰 Token {base_symbol}: Size {token_position['size']:.2f}, PnL {token_position['realized_pnl']:.4f}")
            else:
                # Solo compras sin ventas - marcar como ignore_trade
                print(f"⚠️  Solo compras para {base_symbol}, marcando como ignore_trade")
                # Aquí podrías crear una posición ignore_trade si lo necesitas
    
    # Procesar ventas sin compras (deben ser ignore_trade)
    for position_key, position_trades in grouped_trades.items():
        if position_key.startswith("sell_"):
            base_symbol = position_key.replace("sell_", "")
            if base_symbol not in processed_symbols:
                print(f"⚠️  Ventas sin compras para {base_symbol}, marcando como ignore_trade")
                # Crear posición ignore_trade si es necesario
    
    print(f"✅ Procesadas {len(positions)} posiciones spot")
    return positions

def save_gate_spot_positions(db_path: str = "portfolio.db", days_back: int = 60) -> int:
    """
    Guarda posiciones spot de Gate.io en la base de datos
    """
    print("💾 Guardando posiciones spot de Gate.io...")
    
    # Obtener todos los trades
    all_trades = fetch_gate_spot_trades(days_back=days_back)
    
    if not all_trades:
        print("⚠️ No se obtuvieron trades spot de Gate.io")
        return 0
    
    # Calcular posiciones
    positions = process_spot_positions(all_trades)
    
    if not positions:
        print("⚠️ No se calcularon posiciones spot")
        return 0
    
    # Guardar en base de datos
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    saved = 0
    skipped = 0
    
    for pos in positions:
        try:
            # Verificar si ya existe (mismo exchange, symbol, close_time)
            if pos.get("close_time"):
                cur.execute("""
                    SELECT COUNT(*) FROM closed_positions 
                    WHERE exchange = ? AND symbol = ? AND close_time = ? AND side = ?
                """, (pos["exchange"], pos["symbol"], pos["close_time"], pos["side"]))
                
                if cur.fetchone()[0] > 0:
                    skipped += 1
                    continue
            
            # Insertar posición
            sql = """
                INSERT INTO closed_positions (
                    exchange, symbol, side, size, entry_price, close_price,
                    open_time, close_time, realized_pnl, pnl, funding_total, fee_total,
                    notional, ignore_trade
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            values = (
                pos["exchange"],
                pos["symbol"],
                pos["side"],
                pos["size"],
                pos["entry_price"],
                pos["close_price"],
                pos["open_time"],
                pos["close_time"],
                pos["realized_pnl"],
                pos.get("pnl", 0),
                pos["funding_fee"],
                pos["fees"],
                pos["notional"],
                pos["ignore_trade"]
            )
            
            cur.execute(sql, values)
            saved += 1
            
        except Exception as e:
            print(f"⚠️ Error guardando posición spot {pos.get('symbol')}: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    # Estadísticas
    normal_trades = len([p for p in positions if p.get("side") == "spotbuy"])
    stable_swaps = len([p for p in positions if p.get("side") == "swapstable"])
    ignore_trades = len([p for p in positions if p.get("ignore_trade", 0) == 1])
    
    print(f"✅ Gate spot guardadas: {saved} | omitidas: {skipped}")
    print(f"📊 Resumen: {normal_trades} tokens, {stable_swaps} stable swaps, {ignore_trades} ignorados")
    
    return saved

# Función principal para integración con portfoliov8.3
def sync_gate_spot_positions():
    """
    Función principal para sincronizar posiciones spot de Gate.io
    """
    from portfoliov8_3 import should_sync_spot
    
    if not should_sync_spot("gate"):
        print("⏭️  Sincronización de spot trades de Gate.io deshabilitada")
        return 0
    
    print("🚀 Sincronizando spot trades de Gate.io...")
    return save_gate_spot_positions()

if __name__ == "__main__":
    saved = save_gate_spot_positions()
    print(f"🎯 Prueba completada: {saved} posiciones guardadas")