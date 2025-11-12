# adapters/bitget_spot_trades.py
# -*- coding: utf-8 -*-
"""
Bitget — Spot trades → closed_positions (FIFO por rondas)

- Descarga fills de /api/v2/spot/trade/fills en ventanas ≤90d + paginación (idLessThan).
- Ignora TODO trade que involucre BTC o ETH (en base o quote).
- Detecta swaps USDT<->USDC → side="swapstable" con PnL 1:1 menos fees.
- Tokens normales: FIFO por rondas (compras→ventas). Cierre por polvo (DUST_RATIO).
- Primer SELL de un símbolo sin inventario → side="spotsell" con ignore_trade=1.
- Si quedan compras sin vender: heurística de retiro consultando balances spot.
- Inserta en SQLite (tabla closed_positions) calculando pnl_percent y apr antes de escribir.

Compatibilidad y estilo alineado con tu adapter Gate.  :contentReference[oaicite:5]{index=5}
"""

from __future__ import annotations
import os, sys, time, sqlite3
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple, Optional

# === Path utils ===
# --- rutas y imports de utilidades del proyecto ---
import os, sys, sqlite3, time as pytime
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

UTILS_DIR = os.path.join(BASE_DIR, "utils")
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

# Importa SIEMPRE con el prefijo utils.* para evitar choques con stdlib
from utils.symbols import normalize_symbol
from utils.time import to_s



# === Auth / request helpers (Bitget) ===
try:
    # Firma y request centralizado (ya usado en tu proyecto).  :contentReference[oaicite:8]{index=8}
    from adapters.bitgetv4 import _bitget_request as _request
except Exception:
    from bitget4 import _bitget_request as _request

DB_PATH_DEFAULT = os.path.join(BASE_DIR, "portfolio.db")

# ---------- Constantes / helpers ----------
STABLES = {"USDT", "USDC"}
IGNORE_BASES = {"BTC", "ETH"}    # ignora pares con BTC/ETH

DUST_RATIO = 0.001               # 0.1% del pico de inventario
MIN_DUST_ABS = 0.01

def _f(x: Any, d: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return d

def _split_pair(s: str) -> Tuple[str, str]:
    """'AAA_USDT' | 'AAA/USDT' | 'AAAUSDT' -> (base, quote) en MAYÚSCULAS."""
    t = (s or "").upper().replace("/", "_")
    if "_" in t:
        b, q = t.split("_", 1)
        return b, q
    # Sin separador: intentar heurística USDT/USDC/USD
    for q in ("USDT", "USDC", "USD"):
        if t.endswith(q):
            return t[:-len(q)], q
    return t, ""

def _pair_upper(pair: str) -> str:
    return pair.replace("/", "_").upper()

def _fmt_ms(ms) -> str:
    from datetime import datetime, timezone
    try:
        ms = int(ms or 0)
        if ms and ms < 1_000_000_000_000:
            ms *= 1000
        return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(ms)

def _should_ignore_pair(pair: str) -> bool:
    b, q = _split_pair(pair)
    return (b in IGNORE_BASES) or (q in IGNORE_BASES)

def _is_stable_swap(pair: str) -> bool:
    b, q = _split_pair(pair)
    return b in STABLES and q in STABLES

# ---------- Estructuras ----------
@dataclass
class Fill:
    ts: int            # epoch seconds
    pair: str          # e.g., "AAA_USDT"
    side: str          # buy|sell
    amount: float      # base amount
    price: float       # quote per base
    fee: float         # fee amount in fee_ccy units (positivo)
    fee_ccy: str       # fee currency

    @property
    def base_quote(self) -> Tuple[str, str]:
        return _split_pair(self.pair)

    def fee_in_quote(self) -> float:
        """Fee en QUOTE (si fee en base ⇒ fee*price)."""
        base, quote = self.base_quote
        if self.fee_ccy == quote:
            return self.fee
        if self.fee_ccy == base:
            return self.fee * self.price
        return self.fee  # fallback: tratar como quote

# ---------- Fetch: fills con ventanas + paginación ----------
def fetch_spot_trades(days_back: int = 30, limit: int = 100) -> List[Fill]:
    """
    Ventanas de hasta 90 días (API limita <=90d).
    Paginación hacia atrás con idLessThan (tradeId).
    Orden final ascendente para FIFO estable.

    Bitget API: GET /api/v2/spot/trade/fills
      params: symbol?, startTime(ms), endTime(ms), limit<=100, idLessThan (tradeId anterior).
    """
    all_fills: List[Fill] = []
    now_ms = int(time.time() * 1000)
    from_ms = now_ms - max(1, int(days_back)) * 24 * 3600 * 1000

    # Ventanas de 90 días como máximo
    max_win_days = 90
    win_ms = max_win_days * 24 * 3600 * 1000

    end = now_ms
    while end > from_ms:
        start = max(from_ms, end - win_ms)
        print(f"📥 Ventana {_fmt_ms(start)} - {_fmt_ms(end)}")

        # paginación con idLessThan (página conceptual)
        id_less = None
        page_idx = 1
        while True:
            params = {
                "startTime": str(start),
                "endTime": str(end),
                "limit": str(min(100, max(1, int(limit)))),
            }
            if id_less:
                params["idLessThan"] = id_less

            print(f"   ⏩ Página {page_idx} (idLessThan={id_less})")
            data = _request("GET", "/api/v2/spot/trade/fills", params=params)
            # data: {"code":"00000","data":[...]}
            rows = (data or {}).get("data") or []
            print(f"   ✅ Página {page_idx}: {len(rows)} trades")

            if not rows:
                break

            # Adaptar schema Bitget → Fill
            for r in rows:
                try:
                    pair = _pair_upper(r.get("symbol") or r.get("symbolName") or "")
                    side = (r.get("side") or "").lower()
                    px = _f(r.get("priceAvg") or r.get("price"))
                    sz = _f(r.get("size") or r.get("baseSize") or r.get("fillSize"))
                    ts = to_s(r.get("cTime") or r.get("uTime") or r.get("time"))
                    # feeDetail.totalFee (negativo en la API) y feeCoin
                    fee_d = r.get("feeDetail") or {}
                    fee_coin = (fee_d.get("feeCoin") or "").upper()
                    fee_raw = _f(fee_d.get("totalFee") or 0.0)
                    fee = abs(fee_raw)  # trabajamos fee como magnitud positiva

                    all_fills.append(Fill(ts=ts, pair=pair, side=side, amount=sz,
                                          price=px, fee=fee, fee_ccy=fee_coin))
                except Exception as e:
                    print("      · skip fill por error:", e)
                    continue

            # preparar siguiente página
            last_trade_id = (rows[-1].get("tradeId") if rows else None)
            if not last_trade_id or len(rows) < int(params["limit"]):
                break
            id_less = str(last_trade_id)
            page_idx += 1
            time.sleep(0.05)  # RL ~10/s

        end = start - 1
        time.sleep(0.1)

    # Orden ascendente para FIFO estable
    all_fills.sort(key=lambda x: x.ts)
    print(f"📊 Total trades descargados: {len(all_fills)}")
    return all_fills

# ---------- FIFO engine ----------
@dataclass
class RoundAgg:
    qty: float = 0.0
    cost_quote: float = 0.0
    proceeds_quote: float = 0.0
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
        sz = self.qty
        entry_price = self.cost_quote / max(sz, 1e-12)
        close_price = self.proceeds_quote / max(sz, 1e-12)
        pnl_price = (self.proceeds_quote - self.cost_quote)
        fee_total = -abs(self.fees_quote)         # negativo en DB
        realized = pnl_price + fee_total          # funding=0 en spot
        return {
            "size": sz,
            "entry_price": entry_price,
            "close_price": close_price,
            "pnl": pnl_price,
            "realized_pnl": realized,
            "fee_total": fee_total,
            "funding_total": 0.0,
            "open_time": int(self.open_time or 0),
            "close_time": int(self.close_time or 0),
            "notional": self.cost_quote,
        }

# ---------- Balances spot (para heurística de retiro) ----------
def fetch_bitget_spot_balances() -> Dict[str, float]:
    """
    Devuelve {ASSET: total} con disponibles+congelados en Spot.
    Endpoint Bitget: /api/v2/spot/account/assets (estructura típica).
    """
    out: Dict[str, float] = {}
    try:
        data = _request("GET", "/api/v2/spot/account/assets", params={}) or {}
        rows = data.get("data") or []
        for a in rows:
            ccy = (a.get("coin") or a.get("symbol") or a.get("currency") or "").upper()
            total = _f(a.get("available"), 0.0) + _f(a.get("frozen"), 0.0)
            if ccy:
                out[ccy] = out.get(ccy, 0.0) + total
    except Exception as e:
        print(f"⚠️ No se pudieron leer balances spot Bitget: {e}")
    return out

# ---------- DB insert (calcula pnl_percent / apr e idempotencia con colisión) ----------
INSERT_SQL = (
    "INSERT OR IGNORE INTO closed_positions ("
    "exchange, symbol, side, size, entry_price, close_price, "
    "open_time, close_time, pnl, realized_pnl, funding_total, fee_total, "
    "pnl_percent, apr, initial_margin, notional, leverage, liquidation_price, ignore_trade"
    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)

def _insert_row(conn: sqlite3.Connection, row: Dict[str, Any]):
    # Derivados (igual filosofía que tu db_manager.save_closed_position).  :contentReference[oaicite:9]{index=9}
    size = _f(row.get("size"))
    entry = _f(row.get("entry_price"))
    close = _f(row.get("close_price"))
    pnl = _f(row.get("pnl"))
    realized = _f(row.get("realized_pnl"))
    fee_total = _f(row.get("fee_total"))
    notional = _f(row.get("notional")) or abs(size) * entry
    open_s = int(row.get("open_time") or 0)
    close_s = int(row.get("close_time") or 0)

    base_cap = notional if notional > 0 else max(abs(size) * entry, 1e-9)
    pnl_percent = 100.0 * (realized / base_cap) if base_cap else 0.0
    days = max((close_s - open_s) / 86400.0, 1e-9) if (open_s and close_s) else 0.0
    apr = pnl_percent * (365.0 / days) if days > 0 else 0.0

    vals = (
        row.get("exchange", "bitget"),
        row.get("symbol"),
        row.get("side"),
        size,
        entry,
        close,
        open_s,
        close_s,
        pnl,
        realized,
        0.0,                 # funding_total (spot)
        fee_total,
        pnl_percent,
        apr,
        None,                # initial_margin
        notional,
        0.0,                 # leverage
        None,                # liquidation_price
        int(bool(row.get("ignore_trade", False))),
    )

    cur = conn.cursor()
    cur.execute(INSERT_SQL, vals)
    if cur.rowcount == 0:
        # posible colisión UNIQUE(exchange,symbol,side,close_time) → desplazar 1s y reintentar hasta 3 veces
        for i in range(1, 4):
            cur.execute(INSERT_SQL, vals[:7] + (close_s + i,) + vals[8:])
            if cur.rowcount > 0:
                print(f"⚠️ close_time desplazado +{i}s por colisión UNIQUE")
                break
        if cur.rowcount == 0:
            print(f"⚠️ Duplicado ignorado: {vals[1]} {vals[2]} {close_s}")

# ---------- Core ----------
def save_bitget_spot_positions(db_path: str = "portfolio.db", days_back: int = 30, debug: bool = True) -> Tuple[int, int]:
    """
    Descarga fills spot, calcula FIFO y guarda rondas en closed_positions.

    Reglas:
      - Ignora pares con BTC/ETH.
      - USDT<->USDC: side='swapstable'.
      - Primer SELL → spotsell ignore_trade=1.
      - Cierre por polvo si remanente ≤ max(0.01, DUST_RATIO * pico) y ronda ≥500 unidades,
        o inventario queda 0 exacto.
      - Si quedan compras sin vender:
          * hubo ventas y remanente ≤ polvo ⇒ forzar cierre (no ignorar)
          * si NO hubo ventas o remanente > polvo y no está en balances spot ⇒ ignore_trade=1
    Devuelve: (guardadas, ignoradas)
    """
    # Asegura DB con columna ignore_trade.  :contentReference[oaicite:10]{index=10}
    from db_manager import init_db, migrate_spot_support
    init_db(); migrate_spot_support()

    fills = fetch_spot_trades(days_back=days_back, limit=100)
    if debug:
        print(f"🔎 Bitget spot fills: {len(fills)}")

    # Balances spot para heurística de retiro
    spot_bal = fetch_bitget_spot_balances()

    # Agrupar por par y filtrar ignorados
    by_pair: Dict[str, List[Fill]] = defaultdict(list)
    for f in fills:
        if _should_ignore_pair(f.pair):
            continue
        by_pair[_pair_upper(f.pair)].append(f)

    saved = 0
    ignored = 0
    conn = sqlite3.connect(db_path)

    for pair, trades in by_pair.items():
        base, quote = _split_pair(pair)

        # -------- 1) Swaps estables USDT/USDC --------
        if _is_stable_swap(pair):
            for f in trades:
                fee_q = f.fee_in_quote()
                received_quote = f.amount * f.price
                net_base_out = f.amount - (f.fee if f.fee_ccy == base else 0.0)
                price_pnl = received_quote - net_base_out
                fee_quote_only = fee_q if f.fee_ccy == quote else 0.0
                realized = price_pnl - fee_quote_only

                row = {
                    "exchange": "bitget",
                    "symbol": f"{base}{quote}",
                    "side": "swapstable",
                    "size": abs(net_base_out),
                    "entry_price": 1.0,
                    "close_price": 1.0,
                    "pnl": price_pnl,
                    "realized_pnl": realized,
                    "fee_total": -abs(fee_q),
                    "open_time": f.ts,
                    "close_time": f.ts,
                    "notional": max(received_quote, net_base_out),
                    "ignore_trade": 0,
                }
                _insert_row(conn, row)
                saved += 1
            continue

        # -------- 2) Tokens normales → FIFO por rondas --------
        trades.sort(key=lambda x: x.ts)

        # Primeros SELL sin inventario ⇒ depósito/transfer ignorado
        idx = 0
        while idx < len(trades) and trades[idx].side == "sell":
            f = trades[idx]
            fee_q = f.fee_in_quote()
            row = {
                "exchange": "bitget",
                "symbol": normalize_symbol(f"{base}{quote}"),
                "side": "spotsell",
                "size": abs(f.amount),
                "entry_price": f.price,
                "close_price": f.price,
                "pnl": 0.0,
                "realized_pnl": 0.0,
                "fee_total": -abs(fee_q),
                "open_time": f.ts,
                "close_time": f.ts,
                "notional": abs(f.amount) * f.price,
                "ignore_trade": 1,
            }
            _insert_row(conn, row)
            ignored += 1
            idx += 1

        # Estado ronda
        lot_q = deque()  # [qty_recibida_base, price, fee_per_unit_quote, ts]
        round_agg = RoundAgg()
        round_started = False
        total_qty_in_round = 0.0
        inventory_base = 0.0
        peak_inventory_base = 0.0
        sells_occurred = False

        def _flush_round():
            nonlocal saved, round_agg, round_started, total_qty_in_round, peak_inventory_base
            if not round_started or not round_agg.is_valid():
                round_agg = RoundAgg(); round_started = False
                total_qty_in_round = 0.0; peak_inventory_base = 0.0
                return
            data = round_agg.finalize()
            data["size"] = peak_inventory_base  # size = pico de inventario
            row = {
                "exchange": "bitget",
                "symbol": normalize_symbol(f"{base}{quote}"),
                "side": "spotbuy",
                "ignore_trade": 0,
                **data,
            }
            _insert_row(conn, row)
            saved += 1
            round_agg = RoundAgg(); round_started = False
            total_qty_in_round = 0.0; peak_inventory_base = 0.0

        for f in trades[idx:]:
            if f.side == "buy":
                round_started = True
                fee_q = f.fee_in_quote()

                # cantidad REAL recibida en base si fee en BASE
                received_base = max(f.amount - (f.fee if f.fee_ccy == base else 0.0), 0.0)
                fee_per_unit_q = (fee_q / max(received_base, 1e-12)) if received_base > 0 else 0.0

                lot_q.append([received_base, f.price, fee_per_unit_q, f.ts])

                # coste (con amount) + fees en quote se llevan en la ronda
                round_agg.merge_buy(f.amount, f.price, fee_q, f.ts)
                total_qty_in_round += f.amount

                inventory_base += received_base
                peak_inventory_base = max(peak_inventory_base, inventory_base)

            else:  # sell
                sells_occurred = True
                fee_q = f.fee_in_quote()
                sell_qty = f.amount
                sell_left = sell_qty

                while sell_left > 1e-12 and lot_q:
                    q, p, fee_u, tsb = lot_q[0]
                    take = min(q, sell_left)

                    round_agg.merge_sell(
                        take, f.price,
                        fee_q * (take / sell_qty) if sell_qty > 0 else 0.0,
                        f.ts
                    )

                    q -= take
                    sell_left -= take
                    if q <= 1e-12:
                        lot_q.popleft()
                    else:
                        lot_q[0][0] = q

                inventory_base = sum(q for q, *_ in lot_q)

                dust = max(MIN_DUST_ABS, DUST_RATIO * peak_inventory_base)
                if (inventory_base <= dust and total_qty_in_round >= 500) or (not lot_q and sell_left <= 1e-12):
                    _flush_round()

        # Final del símbolo
        if lot_q:
            rem_base = sum(q for q, *_ in lot_q)
            dust = max(MIN_DUST_ABS, DUST_RATIO * peak_inventory_base)

            if sells_occurred and rem_base <= dust:
                _flush_round()
            else:
                bal_base = spot_bal.get(base, 0.0)
                if (not sells_occurred) or (rem_base > dust and bal_base < rem_base * 0.5):
                    notional = sum(q * p for q, p, *_ in lot_q)
                    ts_open = min(ts for *_a, ts in lot_q)
                    row = {
                        "exchange": "bitget",
                        "symbol": normalize_symbol(f"{base}{quote}"),
                        "side": "spotbuy",
                        "size": rem_base,
                        "entry_price": notional / max(rem_base, 1e-12),
                        "close_price": notional / max(rem_base, 1e-12),
                        "pnl": 0.0,
                        "realized_pnl": 0.0,
                        "fee_total": 0.0,
                        "open_time": ts_open,
                        "close_time": ts_open,
                        "notional": notional,
                        "ignore_trade": 1,
                    }
                    _insert_row(conn, row)
                    ignored += 1
                # si hubo ventas y rem_base > polvo ⇒ queda abierta

    conn.commit(); conn.close()
    if debug:
        print(f"✅ Spot FIFO Bitget: guardadas={saved}, ignoradas={ignored}")
    return saved, ignored

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bitget Spot FIFO → closed_positions")
    parser.add_argument("--db", type=str, default=DB_PATH_DEFAULT, help="Ruta a portfolio.db")
    parser.add_argument("--days_back", type=int, default=30, help="Histórico hacia atrás (días, ≤90 por ventana)")
    parser.add_argument("--debug", action="store_true", help="Logs verbosos")
    args = parser.parse_args()

    save_bitget_spot_positions(db_path=args.db, days_back=args.days_back, debug=(args.debug or True))

