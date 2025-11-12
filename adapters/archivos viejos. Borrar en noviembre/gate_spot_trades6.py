# adapters/gate_spot_trades.py
import sqlite3
import time
from collections import defaultdict, deque
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
from adapters.gate2 import _request, _require_keys, _headers, fetch_gate_spot_balances

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
    batch_days = 25  # Lotes de 25 días para estar seguros (menos del límite de 30)
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
        
        # Mover al siguiente lote
        current_start = current_end + 1  # +1 para evitar solapamientos
        
        # Pequeña pausa para no saturar la API
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
        # Si la fee es en stablecoin, el valor es directo
        return fee
    elif price is not None:
        # Si tenemos precio, convertir usando el precio del trade
        return fee * price
    else:
        # Si no tenemos precio, asumir 1:1 (mejor que nada)
        return fee

def _process_stable_swap(trade: Dict) -> Optional[Dict]:
    """
    Procesa swaps entre stablecoins
    """
    symbol = trade.get("currency_pair", "")
    if not _is_stablecoin_pair(symbol):
        return None
    
    side = trade.get("side", "").lower()
    amount = D(trade.get("amount", "0"))
    price = D(trade.get("price", "0"))
    fee = D(trade.get("fee", "0"))
    fee_currency = trade.get("fee_currency", "")
    
    # Calcular notional
    notional = amount * price
    
    # Para stable swaps, el PnL es la diferencia entre lo que entró y lo que salió
    # más las fees (que siempre son negativas)
    if side == "buy":
        # Comprando la segunda moneda con la primera
        # Ej: Comprar USDC con USDT -> "pierdes" USDT, "ganas" USDC
        pnl = -notional  # Lo que pagas en la moneda base
    else:  # sell
        # Vendiendo la primera moneda por la segunda  
        # Ej: Vender USDC por USDT -> "ganas" USDT, "pierdes" USDC
        pnl = notional   # Lo que recibes en la moneda quote
    
    # Convertir fee a USDT y restarla
    fee_usdt = _convert_fee_to_usdt(fee, fee_currency, price=price)
    realized_pnl = float(pnl - abs(fee_usdt))
    
    return {
        "exchange": "gate",
        "symbol": normalize_symbol(symbol),
        "side": "swapstable",
        "size": float(amount),
        "entry_price": float(price) if side == "buy" else 0,
        "close_price": float(price) if side == "sell" else 0,
        "open_time": to_s(trade.get("create_time") or trade.get("create_time_ms")),
        "close_time": to_s(trade.get("create_time") or trade.get("create_time_ms")),
        "realized_pnl": realized_pnl,
        "fees": float(-abs(fee_usdt)),
        "funding_fee": 0,
        "notional": float(notional),
        "ignore_trade": 0
    }

def get_current_balances() -> Dict[str, Decimal]:
    """
    Obtiene balances actuales de spot
    """
    try:
        balances = fetch_gate_spot_balances()
        balance_dict = {}
        for balance in balances:
            currency = balance.get("currency", "")
            available = D(balance.get("available", "0"))
            locked = D(balance.get("locked", "0"))
            balance_dict[currency.upper()] = available + locked
        return balance_dict
    except Exception as e:
        print(f"⚠️ Error obteniendo balances actuales: {e}")
        return {}

class FIFOProcessor:
    """
    Procesador FIFO para trades spot
    """
    
    def __init__(self):
        self.inventory = defaultdict(deque)  # symbol -> deque de (amount, price, timestamp, trade_id)
        self.fee_inventory = defaultdict(Decimal)  # symbol -> total fees en tokens
        self.positions = []
        self.current_balances = get_current_balances()
    
    def process_trade(self, trade: Dict) -> None:
        """
        Procesa un trade individual usando FIFO
        """
        symbol = trade.get("currency_pair", "")
        base_symbol = normalize_symbol(symbol)
        
        # Ignorar BTC y ETH
        if _should_ignore_symbol(symbol):
            print(f"⏭️  Ignorando trade de {symbol} (BTC/ETH)")
            return
        
        # Procesar stablecoins por separado
        if _is_stablecoin_pair(symbol):
            stable_trade = _process_stable_swap(trade)
            if stable_trade:
                self.positions.append(stable_trade)
            return
        
        side = trade.get("side", "").lower()
        amount = D(trade.get("amount", "0"))
        price = D(trade.get("price", "0"))
        timestamp = to_s(trade.get("create_time") or trade.get("create_time_ms"))
        trade_id = trade.get("id", "")
        fee = D(trade.get("fee", "0"))
        fee_currency = trade.get("fee_currency", "")
        
        if side == "buy":
            self._process_buy(base_symbol, amount, price, timestamp, trade_id, fee, fee_currency)
        elif side == "sell":
            self._process_sell(base_symbol, amount, price, timestamp, trade_id, trade, fee, fee_currency)
    
    def _process_buy(self, symbol: str, amount: Decimal, price: Decimal, 
                    timestamp: int, trade_id: str, fee: Decimal, fee_currency: str) -> None:
        """
        Procesa una compra - agrega al inventario
        """
        # Si la fee es en el token comprado, ajustar la cantidad
        actual_amount = amount
        if fee_currency and fee_currency.upper() == symbol and fee > 0:
            actual_amount = amount - fee
            self.fee_inventory[symbol] += fee
            print(f"   💸 Fee en {symbol}: {float(fee)} tokens")
        
        self.inventory[symbol].append((actual_amount, price, timestamp, trade_id))
        print(f"💰 COMPRA {symbol}: {float(actual_amount)} a {float(price)}")
    
    def _process_sell(self, symbol: str, amount: Decimal, price: Decimal,
                     timestamp: int, trade_id: str, original_trade: Dict, 
                     fee: Decimal, fee_currency: str) -> None:
        """
        Procesa una venta usando FIFO
        """
        remaining_sell = amount
        total_pnl = D('0')
        total_fees_usdt = D('0')
        
        # Verificar si hay inventory para vender
        if symbol not in self.inventory or not self.inventory[symbol]:
            print(f"⚠️  VENTA sin compra previa para {symbol}, marcando como ignore_trade")
            self._create_ignore_trade(original_trade, "sell_without_buy")
            return
        
        trades_used = []
        
        while remaining_sell > 0 and self.inventory[symbol]:
            buy_amount, buy_price, buy_timestamp, buy_trade_id = self.inventory[symbol][0]
            
            if buy_amount <= remaining_sell:
                # Usar toda esta compra
                used_amount = buy_amount
                self.inventory[symbol].popleft()
            else:
                # Usar parte de esta compra
                used_amount = remaining_sell
                self.inventory[symbol][0] = (buy_amount - used_amount, buy_price, 
                                           buy_timestamp, buy_trade_id)
            
            # Calcular PnL para esta porción
            pnl_portion = (price - buy_price) * used_amount
            total_pnl += pnl_portion
            
            trades_used.append({
                'buy_trade_id': buy_trade_id,
                'buy_price': float(buy_price),
                'buy_timestamp': buy_timestamp,
                'used_amount': float(used_amount)
            })
            
            remaining_sell -= used_amount
        
        # Procesar fees de la venta
        if fee > 0:
            # Convertir fee a USDT
            if _is_stablecoin(fee_currency):
                fee_usdt = fee
            else:
                fee_usdt = fee * price  # Asumir que la fee se cobra en USDT para ventas
            
            total_fees_usdt = -abs(fee_usdt)
            print(f"   💸 Fee de venta: {float(fee_usdt)} USDT")
        
        # Crear posición cerrada
        if trades_used:
            # Usar el timestamp de la primera compra como open_time
            open_time = min(trade['buy_timestamp'] for trade in trades_used)
            # Calcular precio de entrada promedio ponderado
            total_buy_amount = sum(D(str(trade['used_amount'])) for trade in trades_used)
            avg_entry_price = sum(D(str(trade['used_amount'])) * D(str(trade['buy_price'])) 
                               for trade in trades_used) / total_buy_amount
            
            position = {
                "exchange": "gate",
                "symbol": symbol,
                "side": "spotbuy",
                "size": float(amount - remaining_sell),
                "entry_price": float(avg_entry_price),
                "close_price": float(price),
                "open_time": open_time,
                "close_time": timestamp,
                "realized_pnl": float(total_pnl + total_fees_usdt),
                "pnl": float(total_pnl),
                "fees": float(total_fees_usdt),
                "funding_fee": 0,
                "notional": float(amount * price),
                "ignore_trade": 0
            }
            
            self.positions.append(position)
            print(f"📊 VENTA {symbol}: {float(amount)} - PnL: {float(total_pnl)}")
        
        # Si queda cantidad por vender sin inventory, marcar como ignore_trade
        if remaining_sell > 0:
            print(f"⚠️  VENTA parcial sin inventory para {symbol}, marcando resto como ignore_trade")
            self._create_ignore_trade(original_trade, "partial_sell_without_inventory")
    
    def _create_ignore_trade(self, trade: Dict, reason: str) -> None:
        """
        Crea un trade marcado para ignorar
        """
        symbol = trade.get("currency_pair", "")
        base_symbol = normalize_symbol(symbol)
        side = trade.get("side", "").lower()
        amount = D(trade.get("amount", "0"))
        price = D(trade.get("price", "0"))
        timestamp = to_s(trade.get("create_time") or trade.get("create_time_ms"))
        
        ignore_trade = {
            "exchange": "gate",
            "symbol": base_symbol,
            "side": f"spot{side}",
            "size": float(amount),
            "entry_price": float(price) if side == "buy" else 0,
            "close_price": float(price) if side == "sell" else 0,
            "open_time": timestamp if side == "buy" else 0,
            "close_time": timestamp if side == "sell" else 0,
            "realized_pnl": 0,
            "fees": 0,
            "funding_fee": 0,
            "notional": float(amount * price),
            "ignore_trade": 1,
            "ignore_reason": reason
        }
        
        self.positions.append(ignore_trade)
    
    def _should_ignore_remaining_inventory(self, symbol: str, remaining_amount: Decimal, max_trade_size: Decimal) -> bool:
        """
        Determina si el inventory restante debe ser ignorado
        """
        # Si el amount restante es muy pequeño comparado con el tamaño máximo de trade
        if max_trade_size > D('1000') and remaining_amount < D('0.1'):
            return True
        # Si no tenemos balance actual de este token
        if symbol not in self.current_balances or self.current_balances.get(symbol, D('0')) < D('0.001'):
            return True
        return False
    
    def get_final_inventory(self) -> List[Dict]:
        """
        Obtiene inventory final (tokens no vendidos)
        """
        for symbol, inventory in self.inventory.items():
            total_remaining = sum(amount for amount, _, _, _ in inventory)
            
            if total_remaining > D('0'):
                # Calcular el tamaño máximo de trade para este símbolo
                max_trade_size = D('0')
                for amount, _, _, _ in inventory:
                    if amount > max_trade_size:
                        max_trade_size = amount
                
                # Verificar si debemos ignorar este inventory
                if self._should_ignore_remaining_inventory(symbol, total_remaining, max_trade_size):
                    # Marcar como ignore_trade (token retirado o cantidad insignificante)
                    for amount, price, timestamp, trade_id in inventory:
                        if amount > D('0.001'):  # Solo si es una cantidad significativa
                            ignore_trade = {
                                "exchange": "gate",
                                "symbol": symbol,
                                "side": "spotbuy",
                                "size": float(amount),
                                "entry_price": float(price),
                                "close_price": 0,
                                "open_time": timestamp,
                                "close_time": 0,
                                "realized_pnl": 0,
                                "fees": 0,
                                "funding_fee": 0,
                                "notional": float(amount * price),
                                "ignore_trade": 1,
                                "ignore_reason": "unclosed_position"
                            }
                            self.positions.append(ignore_trade)
                            print(f"📦 Inventory no vendido {symbol}: {float(amount)} (ignore_trade)")
                else:
                    # El token todavía está en el exchange, no hacer nada
                    print(f"💼 Token {symbol} todavía en exchange: {float(total_remaining)}")
        
        # Procesar fees acumuladas en tokens
        for symbol, fee_amount in self.fee_inventory.items():
            if fee_amount > D('0.001'):
                print(f"💸 Fees totales en {symbol}: {float(fee_amount)}")
        
        return self.positions

def calculate_spot_positions_fifo(trades: List[Dict]) -> List[Dict]:
    """
    Calcula posiciones spot usando método FIFO
    """
    processor = FIFOProcessor()
    
    # Ordenar trades por timestamp (más antiguos primero)
    sorted_trades = sorted(trades, key=lambda x: to_s(x.get("create_time") or x.get("create_time_ms")))
    
    # Procesar cada trade
    for trade in sorted_trades:
        processor.process_trade(trade)
    
    # Obtener posiciones finales
    positions = processor.get_final_inventory()
    
    print(f"✅ Procesadas {len(positions)} posiciones spot")
    return positions

def save_gate_spot_positions(db_path: str = "portfolio.db", days_back: int = 60) -> int:
    """
    Guarda posiciones spot de Gate.io en la base de datos usando FIFO
    """
    print("💾 Guardando posiciones spot de Gate.io...")
    
    # Obtener todos los trades con paginación por tiempo
    all_trades = fetch_gate_spot_trades(days_back=days_back)
    
    if not all_trades:
        print("⚠️ No se obtuvieron trades spot de Gate.io")
        return 0
    
    # Calcular posiciones usando FIFO
    positions = calculate_spot_positions_fifo(all_trades)
    
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
                    WHERE exchange = ? AND symbol = ? AND close_time = ? AND side LIKE 'spot%'
                """, (pos["exchange"], pos["symbol"], pos["close_time"]))
                
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
    normal_trades = len([p for p in positions if p.get("ignore_trade", 0) == 0 and p.get("side") != "swapstable"])
    stable_swaps = len([p for p in positions if p.get("side") == "swapstable"])
    ignore_trades = len([p for p in positions if p.get("ignore_trade", 0) == 1])
    
    print(f"✅ Gate spot guardadas: {saved} | omitidas: {skipped}")
    print(f"📊 Resumen: {normal_trades} normales, {stable_swaps} stable swaps, {ignore_trades} ignorados")
    
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
    # Ejecutar prueba
    saved = save_gate_spot_positions()
    print(f"🎯 Prueba completada: {saved} posiciones guardadas")