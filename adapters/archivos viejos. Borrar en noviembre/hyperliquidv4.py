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

def fetch_hyperliquid_funding_fees(limit: int = 200, days: int = 30):
    """
    Devuelve eventos de funding de Hyperliquid normalizados para la UI:
      - income: importe real en USDC (signado) -> delta.usdc
      - asset:  "USDC"
      - symbol: coin
      - timestamp: ms (la UI toMs acepta s o ms)
    """
    try:
        end_time = utc_now_ms()
        start_time = end_time - days * 24 * 60 * 60 * 1000

        payload = {
            "type": "userFunding",
            "user": HYPERLIQUID_ACCOUNT,
            "startTime": start_time,
            "endTime": end_time,
        }
        r = requests.post(HYPERLIQUID_API_URL, json=payload, timeout=30)
        data = r.json() if r.ok else []

        out = []
        for ev in (data or []):
            d = ev.get("delta", {})
            if d.get("type") != "funding":
                continue
            out.append({
                "exchange": "hyperliquid",
                "symbol": d.get("coin"),
                "income": float(d.get("usdc") or 0.0),          # ✅ UI usa 'income'
                "asset": "USDC",                                 # ✅ UI muestra 'asset'
                "fundingRate": float(d.get("fundingRate") or 0.0),
                "szi": float(d.get("szi") or 0.0),
                "timestamp": int(ev.get("time") or 0),           # ms; la UI lo maneja
            })

        out.sort(key=lambda x: x["timestamp"], reverse=True)
        return out[:limit]
    except Exception as e:
        print(f"❌ Error fetching Hyperliquid funding (userFunding): {e}")
        return []

def fetch_hyperliquid_all_balances():
    """
    Devuelve balances de Hyperliquid combinando:
      - PERPS  -> clearinghouseState
      - SPOT   -> spotClearinghouseState (+ spotMetaAndAssetCtxs para precios en USDC)

    Salida normalizada para la UI:
      {
        exchange: "hyperliquid",
        equity:  spot_value + perps_equity,
        balance: spot_value + withdrawable_perps,  # consistente con otros adapters
        unrealized_pnl: unrealized_perps_only,
        spot: spot_value,
        margin: totalMarginUsed_perps,
        futures: perps_equity
      }
    """
    try:
        # -------- PERPS --------
        perps_equity = 0.0
        perps_withdrawable = 0.0
        perps_unreal = 0.0
        perps_margin_used = 0.0

        p = {"type": "clearinghouseState", "user": HYPERLIQUID_ACCOUNT}
        r = requests.post(HYPERLIQUID_API_URL, json=p, timeout=30)
        if r.ok:
            data = r.json() or {}
            ms = (data.get("marginSummary") or {}) or (data.get("crossMarginSummary") or {})
            perps_equity = float(ms.get("accountValue") or 0.0)
            wd = data.get("withdrawable")
            perps_withdrawable = float(wd) if wd not in (None, "") else perps_equity
            perps_margin_used = float(ms.get("totalMarginUsed") or 0.0)

            for ap in (data.get("assetPositions") or []):
                pos = ap.get("position") or {}
                try:
                    perps_unreal += float(pos.get("unrealizedPnl") or 0.0)
                except Exception:
                    pass

        # -------- SPOT (balances por token) --------
        spot_value = 0.0

        # 1) Balances spot del usuario
        s = {"type": "spotClearinghouseState", "user": HYPERLIQUID_ACCOUNT}
        rs = requests.post(HYPERLIQUID_API_URL, json=s, timeout=30)
        spot_balances = (rs.json() or {}).get("balances", []) if rs.ok else []

        if spot_balances:
            # 2) Meta + contextos de precios para valorar tokens frente a USDC
            m = {"type": "spotMetaAndAssetCtxs"}
            rm = requests.post(HYPERLIQUID_API_URL, json=m, timeout=30)
            usdc_index = None
            price_map = {}

            if rm.ok:
                spot_payload = rm.json() or []
                # Respuesta esperada: [ meta, [ctxs...] ]
                if isinstance(spot_payload, list) and len(spot_payload) == 2:
                    meta = spot_payload[0] or {}
                    ctxs = spot_payload[1] or []

                    tokens = meta.get("tokens", []) or []
                    universe = meta.get("universe", []) or []

                    # localizar índice de USDC
                    for t in tokens:
                        if (t.get("name") or "").upper() == "USDC":
                            usdc_index = t.get("index")
                            break
                    if usdc_index is None:
                        usdc_index = 0  # fallback típico

                    # construir mapa token_base_index -> precio_en_USDC
                    for i, u in enumerate(universe):
                        toks = u.get("tokens", [])
                        if len(toks) == 2 and toks[1] == usdc_index:
                            ctx = ctxs[i] if i < len(ctxs) else {}
                            px = ctx.get("midPx") or ctx.get("markPx") or ctx.get("prevDayPx")
                            try:
                                px = float(px or 0.0)
                            except Exception:
                                px = 0.0
                            if px > 0:
                                base_idx = toks[0]
                                price_map[base_idx] = px

            # 3) Valorar cada token spot en USDC
            for bal in spot_balances:
                try:
                    coin = bal.get("coin")
                    total = float(bal.get("total") or 0.0)
                    token_idx = bal.get("token")

                    # USDC directo
                    if coin == "USDC" or (usdc_index is not None and token_idx == usdc_index):
                        spot_value += total
                        continue

                    # Precio contra USDC si lo tenemos
                    px = price_map.get(token_idx)
                    if px:
                        spot_value += total * px
                        continue

                    # Fallback: usar entryNtl si el precio no está disponible
                    entry_ntl = bal.get("entryNtl")
                    if entry_ntl is not None:
                        spot_value += float(entry_ntl or 0.0)
                except Exception:
                    pass

        # -------- Componer respuesta combinada --------
        return {
            "exchange": "hyperliquid",
            "equity": perps_equity + spot_value,
            "balance": perps_withdrawable + spot_value,   # consistente con Gate/otros (suma todo)
            "unrealized_pnl": perps_unreal,               # sólo perps; spot no es margen
            "spot": spot_value,
            "margin": perps_margin_used,
            "futures": perps_equity,
        }

    except Exception as e:
        print(f"❌ Error fetching Hyperliquid balances: {e}")
        return None


def _debug_hl_balances_raw():
    try:
        p = {"type": "clearinghouseState", "user": HYPERLIQUID_ACCOUNT}
        r = requests.post(HYPERLIQUID_API_URL, json=p, timeout=30)
        print(f"[clearinghouseState] status={r.status_code}")
        data = r.json() if r.ok else {}
        print(" keys:", list(data.keys()))
        print(" withdrawable:", data.get("withdrawable"))
        ms = data.get("marginSummary") or {}
        print(" marginSummary:", {k: ms.get(k) for k in ("accountValue","totalMarginUsed","totalNtlPos")})
        aps = data.get("assetPositions") or []
        print(" assetPositions:", len(aps))
        if aps:
            p0 = aps[0].get("position", {})
            print(" sample position:", {k: p0.get(k) for k in ["coin","szi","entryPx","unrealizedPnl"]})
    except Exception as e:
        print(f"❌ _debug_hl_balances_raw error: {e}")

def _debug_hl_balances_view():
    print("=== Adapter output ===")
    print(fetch_hyperliquid_all_balances())

if __name__ == "__main__":
    _debug_hl_balances_raw()
    _debug_hl_balances_view()

def _debug_hl_spot(days: int = 0):
    try:
        print("=== SPOT RAW ===")
        s = {"type": "spotClearinghouseState", "user": HYPERLIQUID_ACCOUNT}
        rs = requests.post(HYPERLIQUID_API_URL, json=s, timeout=30)
        print("[spotClearinghouseState] status=", rs.status_code)
        print("balances:", (rs.json() or {}).get("balances") if rs.ok else None)

        print("\n=== SPOT META/PRICES ===")
        m = {"type": "spotMetaAndAssetCtxs"}
        rm = requests.post(HYPERLIQUID_API_URL, json=m, timeout=30)
        print("[spotMetaAndAssetCtxs] status=", rm.status_code)
        if rm.ok:
            payload = rm.json() or []
            print("payload_type:", type(payload), "len:", len(payload))
            if isinstance(payload, list) and len(payload) == 2:
                meta, ctxs = payload
                print("tokens:", len(meta.get("tokens", [])))
                print("universe:", len(meta.get("universe", [])))
                print("ctxs:", len(ctxs))
                if ctxs:
                    print("sample ctx:", ctxs[0])

        print("\n=== MIXED VIEW (adapter output) ===")
        print(fetch_hyperliquid_all_balances())
    except Exception as e:
        print("❌ _debug_hl_spot error:", e)


if __name__ == "__main__":
    _debug_hl_spot()

    
# def _debug_hl_raw(days: int = 7):
#     """Imprime RAW de clearinghouseState y userFunding para diagnosticar conexión/clave."""
#     try:
#         print("=== Hyperliquid RAW DEBUG ===")
#         end = utc_now_ms()
#         start = end - days * 24 * 60 * 60 * 1000

#         # --- clearinghouseState
#         p = {"type": "clearinghouseState", "user": HYPERLIQUID_ACCOUNT}
#         r = requests.post(HYPERLIQUID_API_URL, json=p, timeout=30)
#         print(f"[clearinghouseState] status={r.status_code}")
#         data = r.json() if r.ok else {}
#         print(" keys:", list(data.keys()))
#         print(" withdrawable:", data.get("withdrawable"))
#         ms = data.get("marginSummary") or {}
#         print(" marginSummary.accountValue:", ms.get("accountValue"))
#         print(" marginSummary.totalMarginUsed:", ms.get("totalMarginUsed"))
#         aps = data.get("assetPositions") or []
#         print(" assetPositions:", len(aps))
#         if aps:
#             p0 = aps[0].get("position", {})
#             print(" sample position:", {k: p0.get(k) for k in ["coin", "szi", "entryPx", "unrealizedPnl"]})

#         # --- userFunding
#         p = {"type": "userFunding", "user": HYPERLIQUID_ACCOUNT, "startTime": start, "endTime": end}
#         r = requests.post(HYPERLIQUID_API_URL, json=p, timeout=30)
#         print(f"[userFunding] status={r.status_code}")
#         arr = r.json() if r.ok else []
#         print(" events:", len(arr))
#         for ev in arr[:5]:
#             d = ev.get("delta", {})
#             print("  ", ev.get("time"), d.get("coin"), d.get("fundingRate"), d.get("szi"), d.get("usdc"))

#     except Exception as e:
#         print(f"❌ _debug_hl_raw error: {e}")

# def _debug_funding_ui_view():
#     rows = fetch_hyperliquid_funding_fees(limit=10, days=7)
#     print("=== Sample rows (UI-normalized) ===")
#     for r in rows[:5]:
#         print({k: r[k] for k in ("exchange", "symbol", "income", "asset", "timestamp")})

# if __name__ == "__main__":
#     _debug_funding_ui_view()

# def _debug_adapter_views():
#     """Imprime el output de las funciones del adapter tal cual las usa el servidor."""
#     try:
#         print("\n--- Adapter-level outputs ---")
#         print("Balances:", fetch_hyperliquid_all_balances())
#         ff = fetch_hyperliquid_funding_fees(limit=10)
#         print("Funding (latest 10):")
#         for x in ff:
#             print(" ", x)
#     except Exception as e:
#         print(f"❌ _debug_adapter_views error: {e}")


# if __name__ == "__main__":
#     # Ejecuta el archivo directamente para probar credenciales/datos sin Flask
#     _debug_hl_raw(days=7)
#     _debug_adapter_views()   
