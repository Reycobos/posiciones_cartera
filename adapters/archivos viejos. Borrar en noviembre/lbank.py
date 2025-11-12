"""
LBank Exchange Adapter - Solo Spot (sin futuros)
================================================
Implementa:
- Balances spot desde /v2/supplement/user_info.do
- Reconstrucción FIFO de posiciones cerradas desde trades (/v2/supplement/transaction_history.do)

LBank NO tiene futuros, solo spot trading.
"""

import hashlib
import hmac
import time
import requests
from collections import defaultdict
from typing import List, Dict, Any, Optional
import os
from datetime import datetime, timedelta

__all__ = [
    "fetch_lbank_all_balances",
    "save_lbank_closed_positions",
]

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
BASE_URL = "https://api.lbkex.com"
LBANK_API_KEY = "95b4f84f-7631-4286-9b25-5641f4fec5c3"
LBANK_SECRET_KEY = "F3F6CDC794DFC8248BE98B4DB538FDF9"

# Para debugging
DEBUG_REQUESTS = False


# =============================================================================
# AUTENTICACIÓN LBANK
# =============================================================================
def _generate_echostr(length=35):
    """Genera un echostr aleatorio (30-40 caracteres)"""
    import random
    import string
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def _sign_request(params: dict) -> str:
    """
    Firma los parámetros según la especificación de LBank:
    1. Ordena parámetros alfabéticamente
    2. Genera MD5 en mayúsculas
    3. Firma con HmacSHA256
    """
    # 1. Ordenar parámetros alfabéticamente (excluir 'sign')
    sorted_params = sorted(params.items())
    param_str = "&".join(f"{k}={v}" for k, v in sorted_params if k != "sign")
    
    if DEBUG_REQUESTS:
        print(f"[LBANK AUTH] Param string: {param_str}")
    
    # 2. MD5 digest en mayúsculas
    md5_digest = hashlib.md5(param_str.encode('utf-8')).hexdigest().upper()
    
    if DEBUG_REQUESTS:
        print(f"[LBANK AUTH] MD5 digest: {md5_digest}")
    
    # 3. HmacSHA256 con secret key
    signature = hmac.new(
        LBANK_SECRET_KEY.encode('utf-8'),
        md5_digest.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    if DEBUG_REQUESTS:
        print(f"[LBANK AUTH] Signature: {signature}")
    
    return signature


def _make_signed_request(endpoint: str, params: dict = None) -> dict:
    """
    Hace una request firmada a LBank
    Headers requeridos: contentType, timestamp, signature_method, echostr
    """
    if not LBANK_API_KEY or not LBANK_SECRET_KEY:
        raise ValueError("LBANK_API_KEY y LBANK_SECRET_KEY deben estar configurados")
    
    url = f"{BASE_URL}{endpoint}"
    
    # Parámetros base requeridos
    base_params = {
        "api_key": LBANK_API_KEY,
        "signature_method": "HmacSHA256",
        "timestamp": str(int(time.time() * 1000)),
        "echostr": _generate_echostr()
    }
    
    # Merge con parámetros adicionales
    if params:
        base_params.update(params)
    
    # Generar firma
    signature = _sign_request(base_params)
    base_params["sign"] = signature
    
    # Headers requeridos
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    if DEBUG_REQUESTS:
        print(f"[LBANK REQUEST] URL: {url}")
        print(f"[LBANK REQUEST] Params: {base_params}")
    
    try:
        response = requests.post(url, data=base_params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if DEBUG_REQUESTS:
            print(f"[LBANK RESPONSE] Status: {response.status_code}")
            print(f"[LBANK RESPONSE] Data: {data}")
        
        # Verificar respuesta de error de LBank
        if not data.get("result", False):
            error_code = data.get("error_code", "unknown")
            raise Exception(f"LBank API error: {error_code}")
        
        return data.get("data", {})
    
    except requests.exceptions.RequestException as e:
        print(f"[LBANK ERROR] Request failed: {e}")
        raise


# =============================================================================
# NORMALIZACIÓN DE SÍMBOLOS
# =============================================================================
def _normalize_symbol(raw_symbol: str) -> str:
    """
    Normaliza símbolos de LBank al formato estándar
    LBank usa formato: btc_usdt, eth_usdt, etc.
    Salida: BTC/USDT, ETH/USDT
    """
    if not raw_symbol:
        return ""
    
    # Convertir a mayúsculas y reemplazar _ por /
    parts = raw_symbol.upper().split("_")
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    
    return raw_symbol.upper()


# =============================================================================
# BALANCES
# =============================================================================
def fetch_lbank_all_balances(api_key: str = None, secret_key: str = None) -> dict:
    """
    Obtiene todos los balances spot de LBank
    
    Endpoint: POST /v2/supplement/user_info.do
    
    Returns:
        {
            "spot": {"BTC": 1.234, "USDT": 5000.0, ...},
            "margin": {},  # LBank no tiene margin
            "futures": {}, # LBank no tiene futures
            "total_usdt": 0.0,  # No calculamos valor en USDT
            "total_usd": 0.0
        }
    """
    global LBANK_API_KEY, LBANK_SECRET_KEY
    
    if api_key:
        LBANK_API_KEY = api_key
    if secret_key:
        LBANK_SECRET_KEY = secret_key
    
    try:
        # Llamada a la API de user info
        data = _make_signed_request("/v2/supplement/user_info.do")
        
        balances = {"spot": {}, "margin": {}, "futures": {}}
        
        # data es una lista de monedas con sus balances
        if isinstance(data, list):
            for coin_info in data:
                coin = coin_info.get("coin", "").upper()
                usable_amt = float(coin_info.get("usableAmt", 0))
                
                # Solo incluir si tiene balance positivo
                if usable_amt > 0:
                    balances["spot"][coin] = usable_amt
        
        # LBank solo tiene spot, no calculamos totales en USD
        balances["total_usdt"] = 0.0
        balances["total_usd"] = 0.0
        
        return balances
    
    except Exception as e:
        print(f"[LBANK] Error obteniendo balances: {e}")
        return {"spot": {}, "margin": {}, "futures": {}, "total_usdt": 0.0, "total_usd": 0.0}


# =============================================================================
# RECONSTRUCCIÓN FIFO DE POSICIONES CERRADAS
# =============================================================================
class FIFOQueue:
    """Cola FIFO para reconstruir posiciones"""
    
    def __init__(self):
        self.lots = []  # [(qty, price, timestamp, fee), ...]
    
    def add(self, qty: float, price: float, timestamp: int, fee: float):
        """Agrega un lote de entrada"""
        self.lots.append({
            "qty": qty,
            "price": price,
            "timestamp": timestamp,
            "fee": fee
        })
    
    def consume(self, qty_to_close: float) -> tuple:
        """
        Consume qty_to_close de la cola FIFO
        
        Returns:
            (avg_entry_price, total_qty_consumed, total_fees, first_timestamp)
        """
        remaining = qty_to_close
        total_value = 0.0
        total_qty = 0.0
        total_fees = 0.0
        first_ts = None
        
        while remaining > 0 and self.lots:
            lot = self.lots[0]
            
            if first_ts is None:
                first_ts = lot["timestamp"]
            
            if lot["qty"] <= remaining:
                # Consumir todo el lote
                total_value += lot["qty"] * lot["price"]
                total_qty += lot["qty"]
                total_fees += lot["fee"]
                remaining -= lot["qty"]
                self.lots.pop(0)
            else:
                # Consumir parcialmente
                consumed_qty = remaining
                total_value += consumed_qty * lot["price"]
                total_qty += consumed_qty
                # Fee proporcional
                fee_fraction = consumed_qty / lot["qty"]
                total_fees += lot["fee"] * fee_fraction
                
                # Actualizar el lote
                lot["qty"] -= consumed_qty
                lot["fee"] *= (1 - fee_fraction)
                remaining = 0
        
        avg_price = total_value / total_qty if total_qty > 0 else 0.0
        return avg_price, total_qty, total_fees, first_ts
    
    def is_empty(self) -> bool:
        return len(self.lots) == 0
    
    def total_qty(self) -> float:
        return sum(lot["qty"] for lot in self.lots)


def _fetch_trades(symbol: str = None, start_date: str = None, end_date: str = None, 
                  days: int = 30) -> List[dict]:
    """
    Obtiene trades históricos de LBank
    
    Endpoint: POST /v2/supplement/transaction_history.do
    
    Args:
        symbol: Par de trading (ej: "btc_usdt")
        start_date: Fecha inicio (yyyy-MM-dd o yyyy-MM-dd HH:mm:ss UTC+8)
        end_date: Fecha fin (yyyy-MM-dd o yyyy-MM-dd HH:mm:ss UTC+8)
        days: Días hacia atrás si no se especifican fechas
    
    Returns:
        Lista de trades con formato:
        [{
            "symbol": "lbk_usdt",
            "id": "trade-id",
            "orderId": "order-id",
            "price": "4.00000100",
            "qty": "12.00000000",
            "quoteQty": "48.000012",
            "commission": "10.10000000",
            "time": 1499865549590,
            "isBuyer": true,
            "isMaker": false
        }, ...]
    """
    all_trades = []
    
    # Si no hay fechas, usar los últimos N días
    if not end_date:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
    
    if not start_date:
        start = datetime.utcnow() - timedelta(days=days)
        start_date = start.strftime("%Y-%m-%d")
    
    # LBank limita a ventanas de 2 días, así que hay que hacer múltiples requests
    current_start = datetime.strptime(start_date, "%Y-%m-%d")
    final_end = datetime.strptime(end_date, "%Y-%m-%d")
    
    while current_start < final_end:
        current_end = min(current_start + timedelta(days=1), final_end)
        
        params = {
            "startTime": current_start.strftime("%Y-%m-%d"),
            "endTime": current_end.strftime("%Y-%m-%d"),
            "limit": "100"  # Máximo por página
        }
        
        if symbol:
            params["symbol"] = symbol.lower().replace("/", "_")
        
        try:
            trades = _make_signed_request("/v2/supplement/transaction_history.do", params)
            
            if isinstance(trades, list):
                all_trades.extend(trades)
                
                if DEBUG_REQUESTS:
                    print(f"[LBANK TRADES] {len(trades)} trades desde {current_start.date()} hasta {current_end.date()}")
        
        except Exception as e:
            print(f"[LBANK] Error obteniendo trades: {e}")
        
        current_start = current_end
    
    return all_trades


def _reconstruct_fifo_positions(trades: List[dict]) -> List[dict]:
    """
    Reconstruye posiciones cerradas usando FIFO desde trades
    
    Args:
        trades: Lista de trades ordenados por timestamp
    
    Returns:
        Lista de posiciones cerradas reconstruidas
    """
    # Agrupar por símbolo
    by_symbol = defaultdict(list)
    for trade in trades:
        symbol = _normalize_symbol(trade.get("symbol", ""))
        by_symbol[symbol].append(trade)
    
    closed_positions = []
    
    for symbol, symbol_trades in by_symbol.items():
        # Ordenar por timestamp
        symbol_trades.sort(key=lambda t: int(t.get("time", 0)))
        
        # Estado: colas separadas para long y short
        long_queue = FIFOQueue()
        short_queue = FIFOQueue()
        
        # Variables para rastrear el bloque actual
        current_block = None
        block_trades = []
        
        for trade in symbol_trades:
            price = float(trade.get("price", 0))
            qty = float(trade.get("qty", 0))
            is_buyer = trade.get("isBuyer", True)
            timestamp = int(trade.get("time", 0)) // 1000  # ms -> s
            fee = float(trade.get("commission", 0))
            
            # BUY = long, SELL = short
            side = "long" if is_buyer else "short"
            
            # Inicializar bloque si es el primero
            if current_block is None:
                current_block = {
                    "side": side,
                    "symbol": symbol,
                    "open_time": timestamp,
                    "close_time": timestamp,
                    "max_size": 0.0,
                    "total_fee": 0.0,
                    "entry_lots": [],
                    "exit_lots": []
                }
                block_trades = [trade]
            
            block_trades.append(trade)
            current_block["close_time"] = timestamp
            current_block["total_fee"] += fee
            
            # Lógica FIFO
            if side == current_block["side"]:
                # Mismo lado: aumentar posición
                if side == "long":
                    long_queue.add(qty, price, timestamp, fee)
                else:
                    short_queue.add(qty, price, timestamp, fee)
                
                current_block["entry_lots"].append({
                    "qty": qty,
                    "price": price,
                    "timestamp": timestamp,
                    "fee": fee
                })
                
                # Actualizar tamaño máximo
                current_size = long_queue.total_qty() if side == "long" else short_queue.total_qty()
                current_block["max_size"] = max(current_block["max_size"], current_size)
            
            else:
                # Lado opuesto: cerrar posición FIFO
                if current_block["side"] == "long":
                    # Cerrando longs con una venta
                    if not long_queue.is_empty():
                        avg_entry, consumed_qty, entry_fees, first_ts = long_queue.consume(qty)
                        
                        current_block["exit_lots"].append({
                            "qty": consumed_qty,
                            "price": price,
                            "timestamp": timestamp,
                            "fee": fee,
                            "entry_price": avg_entry,
                            "entry_fees": entry_fees
                        })
                        
                        # Si la cola quedó vacía, cerrar el bloque
                        if long_queue.is_empty():
                            closed_positions.append(_finalize_block(current_block))
                            current_block = None
                            block_trades = []
                
                else:
                    # Cerrando shorts con una compra
                    if not short_queue.is_empty():
                        avg_entry, consumed_qty, entry_fees, first_ts = short_queue.consume(qty)
                        
                        current_block["exit_lots"].append({
                            "qty": consumed_qty,
                            "price": price,
                            "timestamp": timestamp,
                            "fee": fee,
                            "entry_price": avg_entry,
                            "entry_fees": entry_fees
                        })
                        
                        # Si la cola quedó vacía, cerrar el bloque
                        if short_queue.is_empty():
                            closed_positions.append(_finalize_block(current_block))
                            current_block = None
                            block_trades = []
    
    return closed_positions


def _finalize_block(block: dict) -> dict:
    """
    Finaliza un bloque FIFO y calcula todas las métricas
    
    Returns:
        Dict con formato esperado por save_closed_position
    """
    symbol = block["symbol"]
    side = block["side"]
    
    # Calcular entry_price ponderado
    total_entry_value = 0.0
    total_entry_qty = 0.0
    for lot in block["entry_lots"]:
        total_entry_value += lot["qty"] * lot["price"]
        total_entry_qty += lot["qty"]
    
    entry_price = total_entry_value / total_entry_qty if total_entry_qty > 0 else 0.0
    
    # Calcular close_price ponderado
    total_exit_value = 0.0
    total_exit_qty = 0.0
    for lot in block["exit_lots"]:
        total_exit_value += lot["qty"] * lot["price"]
        total_exit_qty += lot["qty"]
    
    close_price = total_exit_value / total_exit_qty if total_exit_qty > 0 else 0.0
    
    # Size = máximo neto del bloque
    size = block["max_size"]
    
    # PnL de precio (según side)
    if side == "long":
        price_pnl = (close_price - entry_price) * size
    else:  # short
        price_pnl = (entry_price - close_price) * size
    
    # Fees (negativas)
    fee_total = -abs(block["total_fee"])
    
    # Funding (spot no tiene, dejamos en 0)
    funding_total = 0.0
    
    # Realized PnL
    realized_pnl = price_pnl + funding_total + fee_total
    
    # Notional
    notional = size * entry_price
    
    # Tiempos
    open_time = block["open_time"]
    close_time = block["close_time"]
    
    return {
        "exchange": "lbank",
        "symbol": symbol,
        "side": side,
        "size": size,
        "entry_price": entry_price,
        "close_price": close_price,
        "open_time": open_time,
        "close_time": close_time,
        "pnl": price_pnl,
        "realized_pnl": realized_pnl,
        "funding_total": funding_total,
        "fee_total": fee_total,
        "notional": notional,
        "leverage": 1.0,  # Spot = sin apalancamiento
        "initial_margin": notional,  # En spot, margin = notional
        "liquidation_price": 0.0,  # Spot no tiene liquidación
        "_lock_size": True  # No permitir que save_closed_position recalcule size
    }


def save_lbank_closed_positions(db_path: str = "portfolio.db", days: int = 30, 
                                 dry_run: bool = False, api_key: str = None, 
                                 secret_key: str = None) -> int:
    """
    Reconstruye posiciones cerradas desde trades usando FIFO y las guarda en DB
    
    Args:
        db_path: Ruta a la base de datos SQLite
        days: Días hacia atrás para obtener trades
        dry_run: Si True, solo imprime sin guardar
        api_key: API key de LBank (opcional, usa env var si no se provee)
        secret_key: Secret key de LBank (opcional, usa env var si no se provee)
    
    Returns:
        Número de posiciones guardadas
    """
    global LBANK_API_KEY, LBANK_SECRET_KEY
    
    if api_key:
        LBANK_API_KEY = api_key
    if secret_key:
        LBANK_SECRET_KEY = secret_key
    
    print(f"\n{'='*60}")
    print(f"🔄 Sincronizando posiciones cerradas de LBank (FIFO)")
    print(f"📅 Ventana: últimos {days} días")
    print(f"{'='*60}\n")
    
    try:
        # 1. Obtener trades
        print("📥 Obteniendo trades históricos...")
        trades = _fetch_trades(days=days)
        print(f"✅ {len(trades)} trades obtenidos\n")
        
        if not trades:
            print("⚠️  No hay trades para procesar")
            return 0
        
        # 2. Reconstruir posiciones FIFO
        print("🔄 Reconstruyendo posiciones con FIFO...")
        closed_positions = _reconstruct_fifo_positions(trades)
        print(f"✅ {len(closed_positions)} posiciones cerradas reconstruidas\n")
        
        if not closed_positions:
            print("⚠️  No se reconstruyeron posiciones cerradas")
            return 0
        
        # 3. Guardar en DB (si no es dry_run)
        if dry_run:
            print("🔍 DRY RUN - No se guardará en DB\n")
            for i, pos in enumerate(closed_positions, 1):
                print(f"\n📦 Posición {i}/{len(closed_positions)}:")
                print(f"   Symbol: {pos['symbol']}")
                print(f"   Side: {pos['side']}")
                print(f"   Size: {pos['size']:.6f}")
                print(f"   Entry: ${pos['entry_price']:.6f}")
                print(f"   Close: ${pos['close_price']:.6f}")
                print(f"   PnL: ${pos['realized_pnl']:.2f}")
                print(f"   Fees: ${pos['fee_total']:.2f}")
                print(f"   Open: {datetime.fromtimestamp(pos['open_time']).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Close: {datetime.fromtimestamp(pos['close_time']).strftime('%Y-%m-%d %H:%M:%S')}")
            return 0
        
        else:
            print("💾 Guardando en base de datos...")
            
            # Importar aquí para evitar dependencia circular
            try:
                from db_manager import save_closed_position
            except ImportError:
                print("⚠️  No se pudo importar db_manager. Asegúrate de que existe.")
                return 0
            
            saved_count = 0
            for pos in closed_positions:
                try:
                    save_closed_position(pos)
                    saved_count += 1
                except Exception as e:
                    print(f"❌ Error guardando posición {pos['symbol']}: {e}")
            
            print(f"✅ {saved_count}/{len(closed_positions)} posiciones guardadas\n")
            return saved_count
    
    except Exception as e:
        print(f"❌ Error en save_lbank_closed_positions: {e}")
        import traceback
        traceback.print_exc()
        return 0


# =============================================================================
# TESTING RÁPIDO
# =============================================================================
if __name__ == "__main__":
    print("LBank Adapter - Testing")
    print("=" * 60)
    
    # Test balances
    print("\n1. Testing balances...")
    try:
        balances = fetch_lbank_all_balances()
        print(f"Spot assets: {len(balances['spot'])}")
        for coin, amount in list(balances['spot'].items())[:5]:
            print(f"  {coin}: {amount}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test closed positions (dry run)
    print("\n2. Testing closed positions (dry run)...")
    try:
        save_lbank_closed_positions(days=7, dry_run=True)
    except Exception as e:
        print(f"Error: {e}")