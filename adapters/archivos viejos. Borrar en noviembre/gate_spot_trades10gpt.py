# adapters/gate_spot_trades_fifo.py
# -*- coding: utf-8 -*-
"""
Gate.io — Spot trades → closed positions (FIFO)

Qué hace
--------
- Descarga fills de /spot/my_trades sin requerir currency_pair.
- Ignora TODO trade que involucre BTC o ETH (en base o quote).
- Detecta swaps USDT<->USDC y los guarda como side = "swapstable"
  calculando realized_pnl = diferencia de valor - fees.
- Para el resto de tokens, calcula PnL con FIFO, agregando rondas
  (compras seguidas de ventas) y generando una única posición cerrada por ronda.
- Si el primer fill de un símbolo es un SELL, lo guarda con ignore_trade=1
  (porque proviene de depósito/transfer) y NO afecta PnL agregado.
- Si quedan compras sin vender y el token NO existe en balances spot,
  se considera retirada → ignora_trade=1.

Dependencias
------------
- adapters/gate2.py  → _request(), fetch_gate_spot_balances()
- utils/symbols.py   → normalize_symbol()
- utils/time.py      → to_s()
- db_manager.py      → init_db() (para migración), pero aquí insertamos manualmente
                       para poder setear ignore_trade.

Cómo usar
---------
- Desde portfoliov8.3 importa:
    from adapters.gate_spot_trades_fifo import save_gate_spot_positions
- (Opcional) Ejecuta como script en Spyder:
    runfile('.../adapters/gate_spot_trades_fifo.py')
  Acepta argumentos: --days_back 30 --db portfolio.db

Notas
-----
- Tolerancia de cierre: si tras una venta el remanente de la posición < 0.01
  *y* la ronda superó 500 unidades, se considera cerrada.
- Fees: si vienen en moneda base, se convierten a quote multiplicando por price.
  Si vienen en quote, se usan tal cual. (En swaps stables, todo en "USD" notional.)
"""

from __future__ import annotations
import os
import sys
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

# === Path utils ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

from utils.symbols import normalize_symbol  # utils/symbols.py
from utils.time import to_s  # utils/time.py (convierte ms↔s robustamente)

# === Gate.auth helpers ===
try:
    from adapters.gate2 import _request, fetch_gate_spot_balances  # firma v4 cuando se importa como paquete
except ImportError:
    from gate2 import _request, fetch_gate_spot_balances  # compat al ejecutar como script desde /adapters

DB_PATH_DEFAULT = os.path.join(BASE_DIR, 'portfolio.db')

# ---------- Helpers ----------
STABLES = {"USDT", "USDC"}
IGNORE_BASES = {"BTC", "ETH"}  # ignora cualquier trade que las involucre


def _num(x: Any, d: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return d


def _split_pair(cp: str) -> Tuple[str, str]:
    """ 'AAA_USDT' | 'AAA/USDT' -> (base, quote) en MAYÚSCULAS. """
    s = (cp or '').replace('/', '_').upper()
    parts = s.split('_')
    if len(parts) >= 2:
        return parts[0], parts[1]
    # fallback: todo como base
    return s, ''


@dataclass
class Fill:
    ts: int           # epoch seconds
    pair: str         # e.g., "AAA_USDT"
    side: str         # buy|sell
    amount: float     # base amount
    price: float      # quote per base
    fee: float        # fee amount in fee_ccy units
    fee_ccy: str      # currency of fee

    @property
    def base_quote(self) -> Tuple[str, str]:
        return _split_pair(self.pair)

    def fee_in_quote(self) -> float:
        """Convierte la fee a QUOTE; si ya es quote, se usa tal cual."""
        base, quote = self.base_quote
        if self.fee_ccy == quote:
            return self.fee
        if self.fee_ccy == base:
            return self.fee * self.price
        # Desconocida: asume quote (mejor que 0)
        return self.fee
def _fmt_ms(ms) -> str:
    """Convierte ms/seg a 'YYYY-MM-DD HH:MM:SS UTC'."""
    from datetime import datetime, timezone
    try:
        ms = int(ms or 0)
        if ms and ms < 1_000_000_000_000:  # venía en segundos
            ms *= 1000
        return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(ms)

# ---------- Fetch layer ----------

def fetch_spot_trades(days_back: int = 40, limit: int = 1000) -> List[Fill]:
    """Descarga fills usando ventanas de 30 días + paginación."""
    import time as _t
    
    all_fills = []
    to_ts = int(_t.time())
    from_ts = to_ts - max(1, int(days_back)) * 24 * 3600
    
    # Dividir en ventanas de máximo 30 días
    window_days = 30
    current_to = to_ts
    
    while current_to > from_ts:
        current_from = max(from_ts, current_to - (window_days * 24 * 3600))
        
        page = 1
        max_pages = 10  # Límite de páginas por ventana
        
        while page <= max_pages:
            params = {
                'limit': max(1, min(int(limit), 1000)),
                'page': page,
                'from': current_from,
                'to': current_to,
            }
            
            try:
                print(f"📥 Ventana {_fmt_ms(current_from * 1000)} - {_fmt_ms(current_to * 1000)}, página {page}")
                
                rows = _request('GET', '/spot/my_trades', params=params) or []
                
                if not rows:
                    break
                    
                for r in rows:
                    ts_raw = r.get('create_time') or r.get('create_time_ms') or 0
                    ts = to_s(ts_raw)
                    pair = (r.get('currency_pair') or '').replace('/', '_').upper()
                    side = (r.get('side') or '').lower()
                    amt = _num(r.get('amount'))
                    px  = _num(r.get('price'))
                    fee = _num(r.get('fee'))
                    fee_ccy = (r.get('fee_currency') or '').upper()
                    all_fills.append(Fill(ts, pair, side, amt, px, fee, fee_ccy))
                
                print(f"✅ Página {page}: {len(rows)} trades")
                
                # Si obtenemos menos trades que el límite, es la última página
                if len(rows) < limit:
                    break
                    
                page += 1
                _t.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Error en página {page}: {e}")
                break
        
        # Mover la ventana hacia atrás
        current_to = current_from - 1
        _t.sleep(0.2)

    # Orden por tiempo ascendente para FIFO estable
    all_fills.sort(key=lambda f: f.ts)
    print(f"📊 Total trades descargados: {len(all_fills)}")
    return all_fills

# ---------- FIFO engine ----------
@dataclass
class RoundAgg:
    qty: float = 0.0
    cost_quote: float = 0.0     # suma (buy_qty * buy_price)
    proceeds_quote: float = 0.0 # suma (sell_qty * sell_price)
    fees_quote: float = 0.0
    open_time: Optional[int] = None
    close_time: Optional[int] = None

    def merge_buy(self, qty: float, px: float, fee_q: float, ts: int):
        self.qty += qty
        self.cost_quote += qty * px
        self.fees_quote += fee_q
        self.open_time = ts if self.open_time is None else min(self.open_time, ts)

    def merge_sell(self, qty: float, px: float, fee_q: float, ts: int):
        self.proceeds_quote += qty * px
        self.fees_quote += fee_q
        self.close_time = ts if self.close_time is None else max(self.close_time, ts)

    def is_valid(self) -> bool:
        return self.qty > 0 and self.close_time is not None

    def finalize(self) -> Dict[str, Any]:
        size = self.qty
        entry_price = self.cost_quote / max(size, 1e-12)
        close_price = self.proceeds_quote / max(size, 1e-12)
        pnl_price = (self.proceeds_quote - self.cost_quote)
        fee_total = -abs(self.fees_quote)  # negativo en DB
        realized = pnl_price + fee_total   # no hay funding en spot
        return {
            'size': size,
            'entry_price': entry_price,
            'close_price': close_price,
            'pnl': pnl_price,
            'realized_pnl': realized,
            'fee_total': fee_total,
            'funding_total': 0.0,
            'open_time': int(self.open_time or 0),
            'close_time': int(self.close_time or 0),
            'notional': self.cost_quote,
        }


def _should_ignore_pair(pair: str) -> bool:
    b, q = _split_pair(pair)
    return (b in IGNORE_BASES) or (q in IGNORE_BASES)


def _is_stable_swap(pair: str) -> bool:
    b, q = _split_pair(pair)
    return b in STABLES and q in STABLES


# ---------- DB insert (incluye ignore_trade) ----------
# ---------- DB insert (incluye ignore_trade) ----------
INSERT_SQL = (
    "INSERT OR IGNORE INTO closed_positions (" \
    "exchange, symbol, side, size, entry_price, close_price, " \
    "open_time, close_time, pnl, realized_pnl, funding_total, fee_total, " \
    "pnl_percent, apr, initial_margin, notional, leverage, liquidation_price, ignore_trade" \
    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)

def _insert_row(conn: sqlite3.Connection, row: Dict[str, Any]):
    # Métricas derivadas como en save_closed_position
    size = _num(row.get('size'))
    entry = _num(row.get('entry_price'))
    close = _num(row.get('close_price'))
    pnl   = _num(row.get('pnl'))
    realized = _num(row.get('realized_pnl'))
    notional = _num(row.get('notional')) or abs(size) * entry
    fee_total = _num(row.get('fee_total'))
    open_s = int(row.get('open_time') or 0)
    close_s = int(row.get('close_time') or 0)

    # pnl_percent / apr sobre realized
    base_cap = notional if notional > 0 else max(abs(size) * entry, 1e-9)
    pnl_percent = 100.0 * (realized / base_cap) if base_cap else 0.0
    days = max((close_s - open_s) / 86400.0, 1e-9) if (open_s and close_s) else 0.0
    apr = pnl_percent * (365.0 / days) if days > 0 else 0.0

    vals = (
        row.get('exchange', 'gate'),
        row.get('symbol'),
        row.get('side'),
        size,
        entry,
        close,
        open_s,
        close_s,
        pnl,
        realized,
        0.0,            # funding_total
        fee_total,
        pnl_percent,
        apr,
        None,           # initial_margin
        notional,
        0.0,            # leverage
        None,           # liquidation_price
        int(bool(row.get('ignore_trade', False)))
    )
    cur = conn.cursor()
    cur.execute(INSERT_SQL, vals)
    
    # Opcional: mostrar si se insertó o se ignoró
    if cur.rowcount == 0:
        print(f"⚠️  Duplicado ignorado: {row.get('symbol')} {row.get('side')} {close_s}")
# ---------- Core ----------
DUST_RATIO = 0.001  # 0.1% del pico de inventario; usa 0.0005 si quieres ser más estricto
def save_gate_spot_positions(db_path: str = "portfolio.db", days_back: int = 30, debug: bool = True) -> Tuple[int, int]:
    """
    Descarga fills spot, calcula FIFO y guarda rondas en closed_positions.
    Reglas:
      - Ignora pares con BTC/ETH.
      - USDT<->USDC: side='swapstable', PnL = (USDT recibido - USDC neto entregado) - fee_quote.
      - Si primer fill es SELL -> ignore_trade=1 (depósito/transfer).
      - Cierre de ronda por "polvo": si remanente <= max(0.01, DUST_RATIO * pico_inventario) y total_qty>=500.
      - Si quedan compras sin vender:
          * si hubo ventas y remanente es polvo -> cerrar ronda (no ignorar)
          * si NO hubo ventas o remanente > polvo y no está en balances -> ignore_trade=1 (retiro)
    Devuelve: (guardadas, ignoradas)
    """
    # --- umbral de "polvo" para cierre (0.1% del pico); mínimo absoluto 0.01 ---
    DUST_RATIO = 0.001

    # Asegura DB/migración mínima (mantén tus funciones como en tu proyecto)
    from db_manager import init_db, migrate_spot_support
    init_db(); migrate_spot_support()

    fills = fetch_spot_trades(days_back=days_back, limit=1000)

    if debug:
        print(f"🔎 GATE spot fills recibidos: {len(fills)}")

    # Balances spot para detectar retiros
    try:
        balances = fetch_gate_spot_balances()
        spot_have = { (b.get('currency') or '').upper(): (float(b.get('available', 0) or 0) + float(b.get('locked', 0) or 0)) for b in balances }
    except Exception:
        spot_have = {}

    # Agrupa por par y filtra pares ignorados
    by_pair: Dict[str, List[Fill]] = defaultdict(list)
    for f in fills:
        if _should_ignore_pair(f.pair):
            continue
        by_pair[f.pair].append(f)

    saved = 0
    ignored = 0

    conn = sqlite3.connect(db_path)

    for pair, trades in by_pair.items():
        base, quote = _split_pair(pair)

        # -------- 1) Swaps estables USDC/USDT --------
        if _is_stable_swap(pair):
            # Inserta un registro por fill con PnL de precio + fees
            for f in trades:
                fee_q = f.fee_in_quote()
                received_quote = f.amount * f.price                         # USDT recibido (según trade)
                net_base_out   = f.amount - (f.fee if f.fee_ccy.upper() == base else 0.0)  # USDC neto entregado
                price_pnl      = received_quote - net_base_out             # referencia 1:1
                fee_quote      = fee_q if f.fee_ccy.upper() == quote else 0.0
                realized       = price_pnl - fee_quote

                row = {
                    'exchange': 'gate',
                    'symbol': f"{base}{quote}",
                    'side': 'swapstable',
                    'size': abs(net_base_out),
                    'entry_price': 1.0,
                    'close_price': 1.0,
                    'pnl': price_pnl,
                    'realized_pnl': realized,
                    'fee_total': -abs(fee_q),          # fees totales llevadas a QUOTE
                    'open_time': f.ts,
                    'close_time': f.ts,
                    'notional': max(received_quote, net_base_out),
                    'ignore_trade': 0,
                }
                _insert_row(conn, row)
                saved += 1
            continue

        # -------- 2) Tokens normales → FIFO por rondas --------
        trades.sort(key=lambda x: x.ts)

        # Si el primer fill(s) es SELL → ignorar (depósito/transfer)
        idx = 0
        while idx < len(trades) and trades[idx].side == 'sell':
            f = trades[idx]
            fee_q = f.fee_in_quote()
            row = {
                'exchange': 'gate',
                'symbol': normalize_symbol(f"{base}{quote}"),
                'side': 'spotsell',
                'size': abs(f.amount),
                'entry_price': f.price,
                'close_price': f.price,
                'pnl': 0.0,
                'realized_pnl': 0.0,
                'fee_total': -abs(fee_q),
                'open_time': f.ts,
                'close_time': f.ts,
                'notional': abs(f.amount) * f.price,
                'ignore_trade': 1,
            }
            _insert_row(conn, row)
            ignored += 1
            idx += 1

        # Estado de la ronda FIFO
        lot_q = deque()     # (qty_recibida_base, price, fee_per_unit_quote, ts)
        round_agg = RoundAgg()
        round_started = False
        total_qty_in_round = 0.0

        # Inventario y pico para size real
        inventory_base = 0.0
        peak_inventory_base = 0.0

        # Flag para saber si hubo ventas en la ronda (importa para cierre final)
        sells_occurred = False

        # Helper de cierre
        def _flush_round():
            nonlocal saved, round_agg, round_started, total_qty_in_round, peak_inventory_base
            if not round_started:
                return
            if not round_agg.is_valid():
                # no hay ventas → nada que cerrar
                round_agg = RoundAgg(); round_started = False
                total_qty_in_round = 0.0; peak_inventory_base = 0.0
                return

            data = round_agg.finalize()
            # size real = pico máximo alcanzado en la ronda (no el remanente)
            data['size'] = peak_inventory_base

            row = {
                'exchange': 'gate',
                'symbol': normalize_symbol(f"{base}{quote}"),
                'side': 'spotbuy',     # convención para UI/estrategia
                'ignore_trade': 0,
                **data,
            }
            _insert_row(conn, row)
            saved += 1

            # reset
            round_agg = RoundAgg(); round_started = False
            total_qty_in_round = 0.0; peak_inventory_base = 0.0

        # Recorre los fills restantes
        for f in trades[idx:]:
            if f.side == 'buy':
                round_started = True
                fee_q = f.fee_in_quote()

                # cantidad REAL recibida en base (si fee es en BASE, descuéntala)
                if f.fee_ccy.upper() == base:
                    received_base = max(f.amount - f.fee, 0.0)
                else:
                    received_base = f.amount

                # fee prorrateada por unidad recibida (en QUOTE)
                fee_per_unit_q = fee_q / max(received_base, 1e-12)

                # lote FIFO: solo lo recibido (evita restos por fees en base)
                lot_q.append([received_base, f.price, fee_per_unit_q, f.ts])

                # coste y fees de la ronda se computan con el amount pagado
                round_agg.merge_buy(f.amount, f.price, fee_q, f.ts)
                total_qty_in_round += f.amount

                # inventario y pico
                inventory_base += received_base
                if inventory_base > peak_inventory_base:
                    peak_inventory_base = inventory_base

            else:  # sell
                sells_occurred = True
                fee_q = f.fee_in_quote()
                sell_qty = f.amount
                sell_left = sell_qty

                # match FIFO
                while sell_left > 1e-12 and lot_q:
                    q, p, fee_u, tsb = lot_q[0]
                    take = min(q, sell_left)

                    # proceeds y fees de venta prorrateadas
                    round_agg.merge_sell(
                        take,
                        f.price,
                        fee_q * (take / sell_qty) if sell_qty > 0 else 0.0,
                        f.ts
                    )

                    # reducir lote
                    q -= take
                    sell_left -= take
                    if q <= 1e-12:
                        lot_q.popleft()
                    else:
                        lot_q[0][0] = q

                # inventario vivo tras vender
                inventory_base = sum(q for q, *_ in lot_q)

                # Criterio de cierre con polvo
                dust = max(0.01, DUST_RATIO * peak_inventory_base)
                if (inventory_base <= dust and total_qty_in_round >= 500) or (not lot_q and sell_left <= 1e-12):
                    _flush_round()

        # Al terminar el símbolo, ¿quedan lotes?
        if lot_q:
            rem_base = sum(q for q, *_ in lot_q)
            dust = max(0.01, DUST_RATIO * peak_inventory_base)

            if sells_occurred and rem_base <= dust:
                # prácticamente cerrada (solo polvo) → ciérrala
                _flush_round()
            else:
                # Heurística de retiro solo si NO hubo ventas
                # o si queda más que polvo y además no está en balances
                bal_base = spot_have.get(base, 0.0)
                if (not sells_occurred) or (rem_base > dust and bal_base < rem_base * 0.5):
                    # Retiro → ignorado
                    notional = sum(q * p for q, p, *_ in lot_q)
                    ts_open = min(ts for *_a, ts in lot_q)
                    row = {
                        'exchange': 'gate',
                        'symbol': normalize_symbol(f"{base}{quote}"),
                        'side': 'spotbuy',
                        'size': rem_base,
                        'entry_price': notional / max(rem_base, 1e-12),
                        'close_price': notional / max(rem_base, 1e-12),
                        'pnl': 0.0,
                        'realized_pnl': 0.0,
                        'fee_total': 0.0,
                        'open_time': ts_open,
                        'close_time': ts_open,
                        'notional': notional,
                        'ignore_trade': 1,
                    }
                    _insert_row(conn, row)
                    ignored += 1
                # si hubo ventas y rem_base > dust → ronda queda abierta (para futuras ventas)

    conn.commit(); conn.close()

    if debug:
        print(f"✅ Spot FIFO Gate: guardadas={saved}, ignoradas={ignored}")
    return saved, ignored


# ---------- CLI ----------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Gate.io Spot FIFO → closed_positions')
    parser.add_argument('--db', type=str, default=DB_PATH_DEFAULT, help='Ruta a portfolio.db')
    parser.add_argument('--days_back', type=int, default=30, help='Ventana de histórico (días)')
    parser.add_argument('--debug', action='store_true', help='Logs verbosos')
    args = parser.parse_args()

    save_gate_spot_positions(db_path=args.db, days_back=args.days_back, debug=args.debug or True)
