import os
import time
import json
import requests
from typing import Any, Dict, List, Optional
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from db_manager import save_closed_position
from utils.symbols import normalize_symbol
from dotenv import load_dotenv
load_dotenv()
import json, os, sqlite3


# Load environment variables if needed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

HYPERLIQUID_API_URL = "https://api.hyperliquid.xyz/info"
HYPERLIQUID_PRIVATE_KEY = "0x5d5d8cef5e265430b15a8bee9264aa9d24985ee4da80008b0d82139f68ae38c8"
HYPERLIQUID_ACCOUNT = os.getenv("HYPERLIQUID_ACCOUNT") or "0x981690Ec51Bb332Ec6eED511C27Df325104cb461"

def _get_user_address() -> Optional[str]:
    """Devuelve la dirección de cuenta (pública) en formato hexadecimal 0x..."""
    global HYPERLIQUID_ACCOUNT
    if HYPERLIQUID_ACCOUNT:
        return HYPERLIQUID_ACCOUNT.lower()
    # Si no se proporcionó directamente, derivar de la clave privada
    if HYPERLIQUID_PRIVATE_KEY:
        try:
            # Interpretar la clave privada como bytes (hexadecimal)
            pk_bytes = bytes.fromhex(HYPERLIQUID_PRIVATE_KEY.replace("0x", ""))
        except Exception as e:
            print(f"❌ Error leyendo HYPERLIQUID_PRIVATE_KEY: {e}")
            return None
        try:
            sk = Ed25519PrivateKey.from_private_bytes(pk_bytes)
        except Exception as e:
            print(f"❌ Error construyendo clave Ed25519: {e}")
            return None
        pub_bytes = sk.public_key().public_bytes(encoding=None, format=None)
        # Usar Keccak-256 para derivar dirección Ethereum-like de 20 bytes
        try:
            import hashlib
            keccak_hash = hashlib.new("keccak256")
        except Exception:
            # Fallback a sha3_256 si keccak no disponible (muy similar a keccak256)
            keccak_hash = hashlib.sha3_256()
        keccak_hash.update(pub_bytes)
        addr = "0x" + keccak_hash.hexdigest()[-40:]
        HYPERLIQUID_ACCOUNT = addr.lower()
        return HYPERLIQUID_ACCOUNT
    return None

def _post_info(payload: Dict[str, Any]) -> Any:
    """Envía una solicitud POST al endpoint /info de Hyperliquid."""
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(HYPERLIQUID_API_URL, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"❌ Error en solicitud Hyperliquid /info: {e}")
        return None
    try:
        return resp.json()
    except Exception as e:
        print(f"⚠️ Respuesta no es JSON: {e}")
        return resp.text

def fetch_hyperliquid_open_positions() -> List[Dict[str, Any]]:
    """Obtiene las posiciones abiertas actuales en Hyperliquid en formato normalizado."""
    user_addr = _get_user_address()
    if not user_addr:
        print("⚠️ Dirección de usuario Hyperliquid no configurada.")
        return []
    data = _post_info({"type": "clearinghouseState", "user": user_addr})
    if data is None or "assetPositions" not in data:
        print("⚠️ No se pudieron obtener posiciones abiertas de Hyperliquid.")
        return []
    positions = []
    asset_positions = data.get("assetPositions", [])
    for pos_entry in asset_positions:
        pos = pos_entry.get("position", {})
        coin = normalize_symbol(pos.get("coin", "") or "")
        # Determinar lado (long/short) según el signo de rawUsd
        leverage_info = pos.get("leverage", {})
        raw_usd = float(leverage_info.get("rawUsd", 0.0))
        side = "long" if raw_usd >= 0 else "short"
        size = abs(float(pos.get("szi", 0.0)))
        entry_price = float(pos.get("entryPx", 0.0))
        # Calcular precio de marca a partir de positionValue si disponible
        position_value = float(pos.get("positionValue", 0.0))
        mark_price = 0.0
        if size > 0:
            mark_price = position_value / size
        unrealized_pnl = float(pos.get("unrealizedPnl", 0.0))
        # Funding realizado (acumulado desde que se abrió la posición)
        cum_funding = pos.get("cumFunding", {})
        realized_funding = float(cum_funding.get("sinceOpen", 0.0))
        # Preparar estructura de posición abierta
        positions.append({
            "exchange": "hyperliquid",
            "symbol": coin,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "mark_price": mark_price,
            "unrealized_pnl": unrealized_pnl,
            "realized_funding": realized_funding,
            "notional": size * entry_price if entry_price else 0.0,
            "leverage": float(leverage_info.get("value", 0.0)) if leverage_info.get("value") is not None else None,
            "liquidation_price": float(pos.get("liquidationPx", 0.0)) if pos.get("liquidationPx") is not None else None,
        })
    return positions

def fetch_hyperliquid_funding_fees(limit: int = 50, days: int = 7) -> List[Dict[str, Any]]:
    """Obtiene los últimos registros de pagos de funding de Hyperliquid."""
    user_addr = _get_user_address()
    if not user_addr:
        print("⚠️ Dirección de usuario Hyperliquid no configurada.")
        return []
    # Calcular rango de tiempo (últimos 'days' días)
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000
    data = _post_info({
        "type": "userFunding",
        "user": user_addr,
        "startTime": start_ms,
        "endTime": now_ms
    })
    if data is None:
        return []
    # Esperamos una lista de registros de funding
    events = []
    for entry in data:
        delta = entry.get("delta", {})
        if not delta or delta.get("type") != "funding":
            continue
        coin = normalize_symbol(delta.get("coin", "") or "")
        amount = float(delta.get("usdc", 0.0))
        frate = float(delta.get("fundingRate", 0.0))
        timestamp = int(entry.get("time", 0) // 1000)  # convertir ms a segundos
        # Normalizar signo: en funding, un valor negativo significa que pagamos funding (fee), positivo que recibimos
        events.append({
            "exchange": "hyperliquid",
            "symbol": coin,
            "funding_rate": frate,
            "funding_fee": amount,  # cantidad de USDC pagada/recibida (positiva o negativa)
            "time": timestamp
        })
    # Ordenar por tiempo descendente y limitar a 'limit'
    events.sort(key=lambda x: x["time"], reverse=True)
    return events[:limit]

def fetch_hyperliquid_all_balances() -> Optional[Dict[str, Any]]:
    """Obtiene balances de la cuenta de Hyperliquid (equidad total y separación spot/margin/futures)."""
    user_addr = _get_user_address()
    if not user_addr:
        print("⚠️ Dirección de usuario Hyperliquid no configurada.")
        return None
    data = _post_info({"type": "clearinghouseState", "user": user_addr})
    if data is None:
        return None
    # Extraer equity total y PnL no realizado
    margin_summary = data.get("marginSummary", {})
    equity = float(margin_summary.get("accountValue", margin_summary.get("totalRawUsd", 0.0)))
    # Sumar PnL no realizado de todas las posiciones
    unreal_pnl_total = 0.0
    for pos_entry in data.get("assetPositions", []):
        pos = pos_entry.get("position", {})
        unreal_pnl_total += float(pos.get("unrealizedPnl", 0.0))
    # Preparar objeto de balances
    balance_info = {
        "exchange": "hyperliquid",
        "equity": equity,
        "balance": equity,  # en Hyperliquid, consideramos todo en la cuenta de futures
        "unrealized_pnl": unreal_pnl_total,
        "spot": 0.0,
        "margin": 0.0,
        "futures": equity
    }
    return balance_info

def fetch_hyperliquid_closed_positions(days: int = 30) -> List[Dict[str, Any]]:
    """Obtiene y normaliza las posiciones cerradas en los últimos 'days' días."""
    user_addr = _get_user_address()
    if not user_addr:
        print("⚠️ Dirección de usuario Hyperliquid no configurada.")
        return []
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    # Obtener fills del usuario en el rango de tiempo
    fills_data = _post_info({
        "type": "userFillsByTime",
        "user": user_addr,
        "startTime": start_ms,
        "endTime": end_ms,
        "aggregate": True  # combinar fills parciales en una sola entrada
    })
    if fills_data is None:
        print("⚠️ No se pudieron obtener fills cerrados de Hyperliquid.")
        return []
    # Obtener eventos de funding en el rango (para asignar a posiciones)
    funding_events = _post_info({
        "type": "userFunding",
        "user": user_addr,
        "startTime": start_ms,
        "endTime": end_ms
    }) or []
    # Organizar eventos de funding por símbolo para acceso rápido
    funding_by_coin: Dict[str, List[Dict]] = {}
    for ev in funding_events:
        delta = ev.get("delta", {})
        if not delta or delta.get("type") != "funding":
            continue
        coin = normalize_symbol(delta.get("coin", "") or "")
        amt = float(delta.get("usdc", 0.0))
        t = int(ev.get("time", 0))
        funding_by_coin.setdefault(coin, []).append({"time": t, "amount": amt})
    for coin, ev_list in funding_by_coin.items():
        ev_list.sort(key=lambda x: x["time"])
    closed_positions: List[Dict[str, Any]] = []
    # Agrupar y procesar fills por símbolo
    fills_by_coin: Dict[str, List] = {}
    for fill in fills_data:
        coin = normalize_symbol(fill.get("coin", "") or "")
        fills_by_coin.setdefault(coin, []).append(fill)
    for coin, fills in fills_by_coin.items():
        fills.sort(key=lambda x: x.get("time", 0))
        # Simular posición para ese símbolo
        pos_size = 0.0
        avg_entry_price = 0.0
        pos_side = None
        cluster_start_time = None
        cum_fee = 0.0
        cum_funding = 0.0
        cum_price_pnl = 0.0
        for fill in fills:
            fill_time_ms = int(fill.get("time", 0))
            fill_price = float(fill.get("px", 0.0))
            fill_size = float(fill.get("sz", 0.0))
            fill_dir = fill.get("side") or fill.get("dir") or ""  # "Buy" o "Sell"
            fill_fee = float(fill.get("fee", 0.0))
            # Determine sign of fill (buy = +, sell = - for base asset)
            fill_sign = 1.0 if fill_dir.upper().startswith("B") else -1.0
            # If no open position, this fill opens a new one
            if pos_size == 0.0:
                pos_side = "long" if fill_sign > 0 else "short"
                pos_size = fill_sign * fill_size
                avg_entry_price = fill_price
                cluster_start_time = int(fill_time_ms // 1000)  # open time in seconds
                cum_fee = fill_fee
                cum_funding = 0.0
                cum_price_pnl = 0.0
            else:
                # Funding events that occurred up to this fill (but after last processed fill)
                if coin in funding_by_coin:
                    # Sumar todos los eventos de funding hasta el tiempo actual del fill
                    while funding_by_coin[coin] and funding_by_coin[coin][0]["time"] <= fill_time_ms:
                        ev = funding_by_coin[coin].pop(0)
                        cum_funding += ev["amount"]
                # Determine current position side sign
                current_sign = 1.0 if pos_size > 0 else -1.0
                # If fill is in same direction (increasing position)
                if current_sign == fill_sign:
                    # New weighted average entry price
                    new_size = pos_size + fill_sign * fill_size
                    abs_old = abs(pos_size)
                    abs_new = abs(new_size)
                    # Weighted average price calculation
                    avg_entry_price = ((abs_old * avg_entry_price) + (fill_size * fill_price)) / abs_new if abs_new > 0 else fill_price
                    pos_size = new_size
                    cum_fee += fill_fee
                    # No realized PnL in this case (still open)
                else:
                    # Fill reduces or closes position
                    if abs(fill_size) < abs(pos_size):
                        # Partial close
                        closed_size = fill_size  # this is positive quantity closed regardless of direction
                        closed_size_abs = fill_size
                        if current_sign > 0:  # closing long
                            price_pnl = (fill_price - avg_entry_price) * closed_size_abs
                        else:  # closing short
                            price_pnl = (avg_entry_price - fill_price) * closed_size_abs
                        cum_price_pnl += price_pnl
                        cum_fee += fill_fee
                        # Reduce position size
                        if current_sign > 0:
                            pos_size = pos_size - closed_size_abs  # still positive (long remaining)
                        else:
                            pos_size = pos_size + closed_size_abs  # still negative (short remaining), since fill_sign is opposite
                        # Position remains open, avg_entry_price remains the same (assuming FIFO for simplicity)
                    elif abs(fill_size) == abs(pos_size):
                        # Full close of position
                        closed_size_abs = abs(pos_size)
                        if current_sign > 0:  # closing long completely
                            price_pnl = (fill_price - avg_entry_price) * closed_size_abs
                        else:  # closing short completely
                            price_pnl = (avg_entry_price - fill_price) * closed_size_abs
                        cum_price_pnl += price_pnl
                        cum_fee += fill_fee
                        # Calculate close time and close price
                        close_time_sec = int(fill_time_ms // 1000)
                        close_price = fill_price
                        open_time_sec = cluster_start_time or close_time_sec
                        # Sumar cualquier funding restante hasta el cierre
                        if coin in funding_by_coin:
                            while funding_by_coin[coin] and funding_by_coin[coin][0]["time"] <= fill_time_ms:
                                ev = funding_by_coin[coin].pop(0)
                                cum_funding += ev["amount"]
                        # Construir posición cerrada normalizada
                        closed_position = {
                            "exchange": "hyperliquid",
                            "symbol": coin,
                            "side": pos_side or ("long" if current_sign > 0 else "short"),
                            "size": closed_size_abs,
                            "entry_price": avg_entry_price,
                            "close_price": close_price,
                            "open_time": open_time_sec,
                            "close_time": close_time_sec,
                            "pnl": cum_price_pnl,  # PnL de precio puro
                            "realized_pnl": cum_price_pnl + cum_funding - abs(cum_fee),  # neto incluyendo fees y funding
                            "funding_fee": cum_funding,
                            "fees": abs(cum_fee),
                            "notional": closed_size_abs * avg_entry_price,
                            "leverage": None,
                            "initial_margin": None,
                            "liquidation_price": None
                        }
                        closed_positions.append(closed_position)
                        # Reset position to zero (cluster closed)
                        pos_size = 0.0
                        avg_entry_price = 0.0
                        pos_side = None
                        cluster_start_time = None
                        cum_fee = 0.0
                        cum_funding = 0.0
                        cum_price_pnl = 0.0
                    else:
                        # Over-close (flip) scenario: fill_size > current position, closes and reverses
                        closed_size_abs = abs(pos_size)
                        if current_sign > 0:  # closing long fully and going short
                            price_pnl = (fill_price - avg_entry_price) * closed_size_abs
                        else:  # closing short fully and going long
                            price_pnl = (avg_entry_price - fill_price) * closed_size_abs
                        cum_price_pnl += price_pnl
                        # Split fee proportionally between closing part and opening part
                        proportion_closed = closed_size_abs / fill_size
                        fee_for_close = fill_fee * proportion_closed
                        fee_for_new = fill_fee - fee_for_close
                        cum_fee += fee_for_close
                        # Sumar funding hasta este momento
                        if coin in funding_by_coin:
                            while funding_by_coin[coin] and funding_by_coin[coin][0]["time"] <= fill_time_ms:
                                ev = funding_by_coin[coin].pop(0)
                                cum_funding += ev["amount"]
                        close_time_sec = int(fill_time_ms // 1000)
                        close_price = fill_price
                        open_time_sec = cluster_start_time or close_time_sec
                        closed_position = {
                            "exchange": "hyperliquid",
                            "symbol": coin,
                            "side": pos_side or ("long" if current_sign > 0 else "short"),
                            "size": closed_size_abs,
                            "entry_price": avg_entry_price,
                            "close_price": close_price,
                            "open_time": open_time_sec,
                            "close_time": close_time_sec,
                            "pnl": cum_price_pnl,
                            "realized_pnl": cum_price_pnl + cum_funding - abs(cum_fee),
                            "funding_fee": cum_funding,
                            "fees": abs(cum_fee),
                            "notional": closed_size_abs * avg_entry_price,
                            "leverage": None,
                            "initial_margin": None,
                            "liquidation_price": None
                        }
                        closed_positions.append(closed_position)
                        # New position opens with the remaining portion of the fill
                        remaining_size = abs(fill_size) - closed_size_abs
                        # Determine new side from the same fill
                        new_sign = fill_sign  # this is the direction of the fill (which is now the new position direction)
                        pos_side = "long" if new_sign > 0 else "short"
                        pos_size = new_sign * remaining_size
                        avg_entry_price = fill_price
                        cluster_start_time = int(fill_time_ms // 1000)
                        # Initialize PnL and fees for new position
                        cum_price_pnl = 0.0
                        cum_funding = 0.0
                        cum_fee = fee_for_new
        # (Fin del bucle de fills para este símbolo)
    return closed_positions

def save_hyperliquid_closed_positions(db_path: str = "portfolio.db", days: int = 30, debug: bool = False) -> int:
    """Guarda posiciones cerradas de Hyperliquid en la base de datos SQLite, evitando duplicados."""
    closed_positions = fetch_hyperliquid_closed_positions(days=days)
    if not closed_positions:
        if debug:
            print("⚠️ No se obtuvieron posiciones cerradas de Hyperliquid.")
        return 0
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return 0
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    saved = 0
    skipped = 0
    for pos in closed_positions:
        try:
            cur.execute("""
                SELECT COUNT(*) FROM closed_positions
                WHERE exchange = ? AND symbol = ? AND close_time = ?
            """, (pos["exchange"], pos["symbol"], pos["close_time"]))
            if cur.fetchone()[0]:
                skipped += 1
                continue
            # Mapear nombres al esquema de DB (funding_fee -> funding_total, fees -> fee_total)
            position_data = {
                "exchange": pos["exchange"],
                "symbol": pos["symbol"],
                "side": pos["side"],
                "size": pos["size"],
                "entry_price": pos["entry_price"],
                "close_price": pos["close_price"],
                "open_time": pos["open_time"],
                "close_time": pos["close_time"],
                "pnl": pos.get("pnl"),
                "realized_pnl": pos.get("realized_pnl"),
                "funding_total": pos.get("funding_fee", 0.0),
                "fee_total": pos.get("fees", 0.0),
                "notional": pos.get("notional", 0.0),
                "leverage": pos.get("leverage"),
                "liquidation_price": pos.get("liquidation_price"),
                "initial_margin": pos.get("initial_margin")
            }
            save_closed_position(position_data)
            saved += 1
            if debug:
                sym = pos["symbol"]
                pnl = pos.get("realized_pnl")
                print(f"✅ Guardada Hyperliquid {sym} - PnL: {pnl}")
        except Exception as e:
            print(f"⚠️ Error guardando posición {pos.get('symbol')}: {e}")
    conn.close()
    if debug:
        print(f"✅ Hyperliquid guardadas: {saved} | omitidas (duplicadas): {skipped}")
    else:
        if saved > 0:
            print(f"✅ Guardadas {saved} posiciones cerradas de Hyperliquid (omitidas {skipped} duplicadas).")
    return saved

# === Funciones auxiliares de depuración ===

def debug_preview_hyperliquid_closed(days: int = 3, symbol: str = None):
    """Imprime en consola un preview de posiciones cerradas recientes de Hyperliquid."""
    closed_positions = fetch_hyperliquid_closed_positions(days=days)
    if symbol:
        closed_positions = [p for p in closed_positions if p["symbol"].lower() == symbol.lower()]
    print(f"📦 DEBUG: Se recibieron {len(closed_positions)} registros de posiciones cerradas")
    for pos in closed_positions:
        sym = pos["symbol"].upper()
        side = pos["side"]
        size = pos["size"]
        entry = pos["entry_price"]
        close = pos["close_price"]
        pnl = pos["realized_pnl"]
        fee = pos.get("fees", 0.0)
        funding = pos.get("funding_fee", 0.0)
        open_ts = pos["open_time"]
        close_ts = pos["close_time"]
        print(f"🔎 {sym}")
        print(f"   📏 Size: {size:.4f} | 🎯 Side: {side}")
        print(f"   💰 Entry: {entry} | Close: {close}")
        print(f"   📊 PnL: {pnl} | Fee: {fee} | Funding: {funding}")
        print(f"   ⏰ Open sec: {open_ts} | Close sec: {close_ts}")
        print(f"   ✅ Normalizada: {sym} - PnL: {pnl}")

def debug_dump_hyperliquid_opens():
    """Muestra las posiciones abiertas actuales de Hyperliquid."""
    positions = fetch_hyperliquid_open_positions()
    print(f"📈 Hyperliquid: {len(positions)} posiciones abiertas")
    for pos in positions:
        sym = pos["symbol"]
        qty = pos["size"]
        entry = pos["entry_price"]
        mark = pos["mark_price"]
        unreal = pos["unrealized_pnl"]
        rf = pos.get("realized_funding")
        unsettled = pos.get("total_unsettled")
        print(f"   🔎 {sym}")
        print(f"      📦 Quantity: {qty}")
        print(f"      💰 Entry: {entry} | Mark: {mark}")
        print(f"      📉 Unrealized PnL: {unreal}")
        if rf is not None:
            print(f"      💵 Realized Funding: {rf}")
        if unsettled is not None:
            print(f"      🧮 Total (API Unsettled): {unsettled}")

def debug_dump_hyperliquid_funding(limit: int = 50):
    """Muestra los últimos registros de funding de Hyperliquid."""
    events = fetch_hyperliquid_funding_fees(limit=limit)
    print(f"💵 Hyperliquid funding: {len(events)} registros")
    for ev in events:
        sym = ev["symbol"]
        amount = ev["funding_fee"]
        rate = ev.get("funding_rate")
        ts = ev["time"]
        direction = "received" if amount > 0 else "paid"
        print(f"   ⏰ {ts} | {sym}: {amount} ({direction}, rate {rate})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Solo imprimir datos (por defecto)", default=True)
    parser.add_argument("--save-closed", action="store_true", help="Guardar en SQLite y luego leer")
    parser.add_argument("--opens", action="store_true", help="Mostrar posiciones abiertas")
    parser.add_argument("--funding", type=int, nargs="?", const=50, help="Mostrar últimos N fundings (defecto 50)")
    args = parser.parse_args()
    if args.opens:
        debug_dump_hyperliquid_opens()
    if args.funding is not None:
        debug_dump_hyperliquid_funding(limit=int(args.funding))
    if args.save_closed:
        # Preview, guardar en DB y luego mostrar JSON simulado de /api/closed_positions
        preview = fetch_hyperliquid_closed_positions(days=30)
        print(json.dumps({"preview_closed": preview}, indent=2))
        saved = save_hyperliquid_closed_positions("portfolio.db", days=30, debug=True)
        # Leer de la BD las posiciones cerradas de Hyperliquid guardadas
        try:
            conn = sqlite3.connect("portfolio.db")
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""
                SELECT * FROM closed_positions
                WHERE exchange = 'hyperliquid'
                ORDER BY close_time ASC
            """)
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            print(json.dumps({"closed_positions": rows}, indent=2))
        except Exception as e:
            print(f"⚠️ Error al leer de la base de datos: {e}")
    elif not (args.opens or args.funding):
        # Por defecto (--dry-run): solo imprimir preview de posiciones cerradas recientes
        debug_preview_hyperliquid_closed(days=3)
