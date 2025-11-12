# adapters/hyperliquid.py
"""
Hyperliquid adapter for delta-neutral dashboard.

Public API (exported via __all__):
- fetch_hyperliquid_open_positions()
- fetch_hyperliquid_funding_fees(limit=50, ...)
- fetch_hyperliquid_all_balances(...)
- save_hyperliquid_closed_positions(db_path="portfolio.db", days=30, debug=False)

Key info endpoints (POST https://api.hyperliquid.xyz/info):
- Perps account + open positions: {"type":"clearinghouseState","user":<addr>}
- Perps meta + asset contexts (includes markPx): {"type":"metaAndAssetCtxs"}
- User funding history: {"type":"userFunding","user":<addr>,"startTime":ms,"endTime":ms}
- User fills by time (for reconstruction of closed): {"type":"userFillsByTime","user":<addr>,"startTime":ms,"endTime":ms,"aggregateByTime":true}

Notes:
- All addresses should be lowercased (recommended by official docs).
- Funding "usdc" is positive when collected, negative when paid.
- Spot balances are fetched via {"type":"spotClearinghouseState","user":<addr>} and valued in USDC.
"""
from __future__ import annotations

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from http import get_session  # ahora trae tu http.py del proyecto

from dotenv import load_dotenv
load_dotenv()
import os, time, json, math, re
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from money import D, usd, quant, normalize_fee, to_float
from symbols import normalize_symbol

import importlib.util as _ilu
_time_spec = _ilu.spec_from_file_location("project_time", os.path.join(ROOT, "time.py"))
_project_time = _ilu.module_from_spec(_time_spec); _time_spec.loader.exec_module(_project_time)
utc_now_ms = getattr(_project_time, "utc_now_ms", lambda: int(__import__("time").time()*1000))
to_ms      = getattr(_project_time, "to_ms",      lambda ts: int(ts if ts >= 10**12 else ts*1000))
to_s       = getattr(_project_time, "to_s",       lambda ts: int(ts//1000 if ts >= 10**12 else ts))

from time import time as _time
from time import sleep as _sleep

from time import time as _time_py
from time import sleep as _sleep_py

# Project utils (epoch conversions)
from time import time as _time  # stdlib alias to avoid confusion
import time as _tmod
try:
    from time import time as _time  # ensure available
except Exception:
    pass

try:
    from time import time as _time
except Exception:
    pass

# Our helper utils (project-local)
try:
    from time import utc_now_ms, to_ms, to_s  # project helper (time.py)
except Exception:
    # Safe fallbacks if helpers are not available at import time
    import time as _t
    def utc_now_ms(): return int(_t.time() * 1000)
    def to_ms(ts): t=int(float(ts)); return t if t>=10**12 else t*1000
    def to_s(ts):  t=int(float(ts)); return t//1000 if t>=10**12 else t

# Optional pretty printers: don't import from main to avoid circular imports
PRINT_CLOSED_DEBUG   = bool(os.getenv("PRINT_CLOSED_DEBUG", "0") == "1")
PRINT_OPEN_POSITIONS = bool(os.getenv("PRINT_OPEN_POSITIONS", "0") == "1")
PRINT_FUNDING        = bool(os.getenv("PRINT_FUNDING", "0") == "1")
PRINT_BALANCES       = bool(os.getenv("PRINT_BALANCES", "0") == "1")

def _p(*a, **k):
    print(*a, **k)

def p_open_summary(exchange: str, count: int):
    if PRINT_OPEN_POSITIONS: _p(f"📈 {exchange.capitalize()}: {count} posiciones abiertas")

def p_open_block(exchange: str, symbol: str, qty: float, entry: float, mark: float,
                 unrealized: float, realized_funding: float | None, total_unsettled: float | None,
                 notional: float | None, extra_verification: bool = False):
    if not PRINT_OPEN_POSITIONS: return
    _p(f"   🔎 {symbol.upper()}")
    _p(f"      📦 Quantity: {qty}")
    _p(f"      💰 Entry: {entry} | Mark: {mark}")
    _p(f"      📉 Unrealized PnL: {unrealized}")
    if realized_funding is not None: _p(f"      💵 Realized Funding: {realized_funding}")
    if notional is not None: _p(f"      🏦 Notional: {notional}")

def p_funding_fetching(exchange: str):
    if PRINT_FUNDING: _p(f"🔍 DEBUG: Obteniendo FUNDING FEES (USDC) de {exchange.capitalize()}...")

def p_funding_count(exchange: str, n: int):
    if PRINT_FUNDING: _p(f"📦 DEBUG: Se recibieron {n} registros de funding")

def p_balance_equity(exchange: str, equity: float):
    if PRINT_BALANCES: _p(f"💼 {exchange.capitalize()} equity total: {equity:.2f}")

# --------------------------- Config ---------------------------
EXCHANGE = "hyperliquid"
BASE_URL = os.getenv("HYPERLIQUID_BASE_URL", "https://api.hyperliquid.xyz")
INFO_URL = BASE_URL.rstrip("/") + "/info"
USER_ADDR = "0x981690Ec51Bb332Ec6eED511C27Df325104cb461"
DEX_NAME = os.getenv("HYPERLIQUID_DEX", "")  # empty = first dex

SESSION = get_session(timeout=20)

def _post_info(payload: Dict, retries: int = 3):
    """
    Wrapper with simple backoff for POST /info
    """
    for i in range(retries):
        r = SESSION.post(INFO_URL, json=payload)
        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                # Some endpoints return arrays directly (not JSON object)
                return json.loads(r.text)
        _sleep(0.3 * (2 ** i))
    r.raise_for_status()

def _require_addr() -> str:
    if not USER_ADDR or not USER_ADDR.startswith("0x") or len(USER_ADDR) != 42:
        raise RuntimeError("HYPERLIQUID_ADDRESS (0x...) no configurada en entorno.")
    return USER_ADDR

# ---------------------- Helper: markPx map --------------------
def _fetch_perp_contexts() -> Tuple[Dict[str, Dict], List[str]]:
    """
    Returns:
      - ctx_by_coin: {"ETH": {"markPx": "...", "funding": "...", ...}, ...}
      - universe_order: ["BTC","ETH",...]
    """
    data = _post_info({"type": "metaAndAssetCtxs"})
    if isinstance(data, list) and len(data) == 2:
        meta, ctxs = data
        universe = [x["name"] for x in meta.get("universe", [])]
        ctx_by_coin = {}
        for i, coin in enumerate(universe):
            if i < len(ctxs):
                ctx_by_coin[coin] = ctxs[i]
        return ctx_by_coin, universe
    return {}, []

# ---------------------- Public: Open positions ----------------
def fetch_hyperliquid_open_positions() -> List[Dict]:
    """
    Map Hyperliquid open perp positions to dashboard shape.

    Row shape:
    {
      "exchange": "hyperliquid",
      "symbol": "<NORMALIZED>",
      "side": "long" | "short",
      "size": float,
      "entry_price": float,
      "mark_price": float,
      "liquidation_price": float | 0.0,
      "notional": float,
      "unrealized_pnl": float,     # ONLY price (API value)
      "fee": float,                # accumulated; negative if cost (unknown -> 0.0)
      "funding_fee": float,        # + receive / - pay, since open
      "realized_pnl": float        # fee + funding_fee (for opens)
    }
    """
    addr = _require_addr()
    # Contexts for markPx
    ctx_by_coin, _ = _fetch_perp_contexts()

    state = _post_info({"type": "clearinghouseState", "user": addr, "dex": DEX_NAME})
    positions = []
    asset_positions = state.get("assetPositions") or []

    for ap in asset_positions:
        pos = ap.get("position") or {}
        coin = pos.get("coin") or ""
        szi = float(pos.get("szi") or 0.0)
        if abs(szi) <= 0:
            continue
        entry = float(pos.get("entryPx") or 0.0)
        mark = float((ctx_by_coin.get(coin) or {}).get("markPx") or pos.get("markPx") or 0.0)
        liq = float(pos.get("liquidationPx") or 0.0) if pos.get("liquidationPx") is not None else 0.0
        unreal = float(pos.get("unrealizedPnl") or 0.0)
        funding_since_open = float(((pos.get("cumFunding") or {}).get("sinceOpen")) or 0.0)

        side = "long" if szi > 0 else "short"
        size = abs(szi)
        notional = entry * size

        row = {
            "exchange": EXCHANGE,
            "symbol": normalize_symbol(coin),
            "side": side,
            "size": float(size),
            "entry_price": float(entry),
            "mark_price": float(mark),
            "liquidation_price": float(liq or 0.0),
            "notional": float(notional),
            "unrealized_pnl": float(unreal),      # price-only API field
            "fee": 0.0,                            # unknown at HL API for opens
            "funding_fee": float(funding_since_open),
            "realized_pnl": float(funding_since_open + 0.0),
        }
        positions.append(row)
        p_open_block(EXCHANGE, row["symbol"], row["size"], row["entry_price"], row["mark_price"],
                     row["unrealized_pnl"], row["funding_fee"], None, row["notional"], False)

    p_open_summary(EXCHANGE, len(positions))
    return positions

# ---------------------- Public: Funding fees ------------------
def fetch_hyperliquid_funding_fees(limit: int = 50, days: int = 14,
                                   start_time: Optional[int] = None,
                                   end_time: Optional[int] = None) -> List[Dict]:
    """
    Fetch user funding deltas and normalize.
    Output row shape:
    {
      "exchange": "hyperliquid",
      "symbol": "<NORMALIZED>",
      "income": float,       # + receive / - pay
      "asset": "USDC",
      "timestamp": int,      # epoch ms
      "funding_rate": float, # rate at that time (if present)
      "type": "FUNDING_FEE"
    }
    """
    addr = _require_addr()
    p_funding_fetching(EXCHANGE)

    now_ms = utc_now_ms()
    start_ms = int(start_time or (now_ms - days * 24 * 60 * 60 * 1000))
    end_ms = int(end_time or now_ms)

    payload = {"type": "userFunding", "user": addr, "startTime": start_ms, "endTime": end_ms}
    data = _post_info(payload)
    out = []
    for ev in (data or []):
        ts = int(ev.get("time") or 0)
        delta = ev.get("delta") or {}
        coin = delta.get("coin") or ""
        rate = float(delta.get("fundingRate") or 0.0)
        usdc = float(delta.get("usdc") or 0.0)  # positive when received
        out.append({
            "exchange": EXCHANGE,
            "symbol": normalize_symbol(coin),
            "income": usdc,
            "asset": "USDC",
            "timestamp": ts,
            "funding_rate": rate,
            "type": "FUNDING_FEE",
        })
    # Newest first
    out.sort(key=lambda x: x["timestamp"], reverse=True)
    p_funding_count(EXCHANGE, len(out))
    return out[:limit] if limit else out

# ---------------------- Balances (Spot + Perps) ---------------
def fetch_hyperliquid_all_balances(db_path: str = "portfolio.db") -> Dict:
    """
    Compose equity/buckets across perps + spot.

    Return dict (exact shape expected by Balances tab):
    {
      "exchange": "hyperliquid",
      "equity": float,
      "balance": float,
      "unrealized_pnl": float,   # 0.0 if N/A
      "initial_margin": float,   # total margin used on perps
      "spot": float,
      "margin": 0.0,
      "futures": float
    }
    """
    addr = _require_addr()

    # Perps account
    state = _post_info({"type": "clearinghouseState", "user": addr, "dex": DEX_NAME}) or {}
    # Totals (perps)
    margin_summary = state.get("marginSummary") or {}
    cross_summary = state.get("crossMarginSummary") or {}
    withdrawable = float(state.get("withdrawable") or 0.0)
    unrealized_total = 0.0
    for ap in (state.get("assetPositions") or []):
        pos = ap.get("position") or {}
        unrealized_total += float(pos.get("unrealizedPnl") or 0.0)

    perps_account_value = float(margin_summary.get("accountValue") or 0.0)
    initial_margin = float(margin_summary.get("totalMarginUsed") or 0.0)

    # Spot balances
    spot_state = _post_info({"type": "spotClearinghouseState", "user": addr}) or {}
    spot_balances = spot_state.get("balances") or []

    # Value spot in USDC: USDC total + entryNtl of non-USDC balances
    spot_usdc = 0.0
    spot_others_ntl = 0.0
    for b in spot_balances:
        coin = b.get("coin") or ""
        total = float(b.get("total") or 0.0)
        if coin.upper() in ("USDC", "USD"):
            spot_usdc += total
        else:
            # entry notionals for other tokens are returned in USDC terms
            spot_others_ntl += float(b.get("entryNtl") or 0.0)

    spot_value = spot_usdc + spot_others_ntl

    # Compose totals
    equity = perps_account_value + spot_value
    balance = withdrawable + spot_value  # utilizable
    res = {
        "exchange": EXCHANGE,
        "equity": float(equity),
        "balance": float(balance),
        "unrealized_pnl": float(unrealized_total),
        "initial_margin": float(initial_margin),
        "spot": float(spot_value),
        "margin": 0.0,
        "futures": float(initial_margin),
    }
    p_balance_equity(EXCHANGE, res["equity"])
    return res

# ------------- Closed positions: reconstruction from fills -------------
def _iter_user_fills(addr: str, start_ms: int, end_ms: int, aggregate: bool = True) -> List[Dict]:
    """
    Pages user fills by time until end_ms.
    Returns a flat, time-ascending list of perps fills (skips spot).
    """
    out = []
    cursor = start_ms
    while cursor <= end_ms:
        data = _post_info({
            "type": "userFillsByTime",
            "user": addr,
            "startTime": int(cursor),
            "endTime": int(end_ms),
            "aggregateByTime": bool(aggregate),
        })
        if not data:
            break
        # filter only perps: coin like "BTC","ETH" etc. Skip spot with '@' or 'TOKEN/USDC' style.
        perps = [f for f in data if isinstance(f.get("coin"), str) and not (str(f["coin"]).startswith("@") or "/" in str(f["coin"]))]
        out.extend(perps)
        last_ts = max(int(x.get("time") or cursor) for x in data)
        next_cursor = last_ts + 1
        if next_cursor <= cursor:  # safety
            break
        cursor = next_cursor
        if len(data) < 2000:
            break  # API returned fewer than max; likely finished
        _sleep(0.15)  # gentle backoff
    # ascending
    out.sort(key=lambda x: int(x.get("time") or 0))
    return out

def _group_roundtrips_from_fills(fills: List[Dict]) -> List[Dict]:
    """
    Build closed position cycles (open->flat) for each coin.

    Returns list of dicts ready for save_closed_position(...) minimal fields.
    """
    by_coin = defaultdict(list)
    for f in fills:
        by_coin[f["coin"]].append(f)

    closed = []
    for coin, fs in by_coin.items():
        # State for current cycle
        pos = 0.0
        cycle = None  # dict to accumulate
        for f in fs:
            ts = int(f.get("time") or 0)
            side = (f.get("side") or "B").upper()  # "B" buy, "A" sell
            sz = float(f.get("sz") or 0.0)
            px = float(f.get("px") or 0.0)
            fee = float(f.get("fee") or 0.0)
            closed_pnl_piece = float(f.get("closedPnl") or 0.0)

            delta = sz if side == "B" else -sz
            prev_pos = pos
            next_pos = prev_pos + delta

            # Start new cycle when moving from zero to non-zero
            if prev_pos == 0.0 and next_pos != 0.0:
                cycle = {
                    "coin": coin,
                    "start_time": ts,
                    "end_time": ts,
                    "side_sign": 1.0 if delta > 0 else -1.0,
                    "open_legs": [],   # list of (sz, px)
                    "close_legs": [],  # list of (sz, px)
                    "fee_total": 0.0,
                    "funding_total": 0.0,  # to be populated later
                    "price_pnl": 0.0,
                }

            if cycle is None:
                # In case we start in the middle of an existing position, treat until it closes
                cycle = {
                    "coin": coin,
                    "start_time": ts,
                    "end_time": ts,
                    "side_sign": 1.0 if (next_pos if next_pos!=0 else delta) > 0 else -1.0,
                    "open_legs": [],
                    "close_legs": [],
                    "fee_total": 0.0,
                    "funding_total": 0.0,
                    "price_pnl": 0.0,
                }

            # Classify leg vs side_sign
            if (delta > 0 and cycle["side_sign"] > 0) or (delta < 0 and cycle["side_sign"] < 0):
                cycle["open_legs"].append((abs(delta), px))
            else:
                cycle["close_legs"].append((abs(delta), px))

            cycle["fee_total"] += -abs(fee)  # always negative
            cycle["price_pnl"] += closed_pnl_piece  # HL provides closed PnL per fill
            cycle["end_time"] = ts

            pos = next_pos

            # Cycle closes when pos returns to zero
            if pos == 0.0 and cycle is not None:
                # Aggregate
                open_sz = sum(sz for sz, _ in cycle["open_legs"])
                close_sz = sum(sz for sz, _ in cycle["close_legs"])
                size = float(min(open_sz, close_sz))  # matched size
                # Weighted avg prices
                def wavg(legs):
                    num = sum(sz*px for sz, px in legs)
                    den = sum(sz for sz, _ in legs)
                    return (num/den) if den > 0 else 0.0
                entry_px = wavg(cycle["open_legs"])
                close_px = wavg(cycle["close_legs"])
                side = "long" if cycle["side_sign"] > 0 else "short"

                # Price pnl: prefer HL closedPnL sum; else compute
                if abs(cycle["price_pnl"]) < 1e-12 and size > 0 and entry_px > 0 and close_px > 0:
                    price_pnl = (close_px - entry_px) * size if side == "long" else (entry_px - close_px) * size
                else:
                    price_pnl = cycle["price_pnl"]

                # Save structure; funding_total will be filled by caller joining userFunding by time range
                closed.append({
                    "exchange": EXCHANGE,
                    "symbol": normalize_symbol(coin),
                    "side": side if size > 0 else "closed",
                    "size": float(size),
                    "entry_price": float(entry_px),
                    "close_price": float(close_px),
                    "open_time": int(to_s(cycle["start_time"])),
                    "close_time": int(to_s(cycle["end_time"])),
                    "realized_pnl": 0.0,       # fill later after funding join
                    "funding_total": 0.0,      # fill later via userFunding
                    "fee_total": float(cycle["fee_total"]),   # negative
                    "pnl": float(price_pnl),    # price-only
                    "notional": float(entry_px * size),
                    "leverage": None,
                    "initial_margin": None,
                    "liquidation_price": None,
                })
                cycle = None

    return closed

def _join_funding(addr: str, rows: List[Dict], fudge_ms: int = 60_000) -> None:
    """
    For each closed row, sum funding deltas within [open_time, close_time] (± fudge).
    Modifies rows in-place: sets funding_total and realized_pnl.
    """
    if not rows:
        return
    start_ms = min((r["open_time"] or 0) for r in rows) * 1000 - fudge_ms
    end_ms   = max((r["close_time"] or 0) for r in rows) * 1000 + fudge_ms
    data = _post_info({"type": "userFunding", "user": addr, "startTime": start_ms, "endTime": end_ms}) or []
    # group by (coin) for faster match
    fund_by_coin: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for ev in data:
        ts = int(ev.get("time") or 0)
        delta = ev.get("delta") or {}
        coin = delta.get("coin") or ""
        usdc = float(delta.get("usdc") or 0.0)
        fund_by_coin[normalize_symbol(coin)].append((ts, usdc))

    for r in rows:
        sym = r["symbol"]
        open_ms = (r.get("open_time") or 0) * 1000
        close_ms = (r.get("close_time") or 0) * 1000
        fsum = 0.0
        for ts, amt in fund_by_coin.get(sym, []):
            if open_ms - fudge_ms <= ts <= close_ms + fudge_ms:
                fsum += amt
        r["funding_total"] = float(fsum)
        r["realized_pnl"] = float((r.get("pnl") or 0.0) + fsum + (r.get("fee_total") or 0.0))

# ---------------------- Public: Save closed -------------------
def save_hyperliquid_closed_positions(db_path: str = "portfolio.db", days: int = 30, debug: bool = False) -> int:
    """
    Reconstruct closed positions from fills and persist to SQLite via db_manager.save_closed_position().
    Returns number of rows saved.
    """
    from db_manager import save_closed_position  # local import

    addr = _require_addr()
    now_ms = utc_now_ms()
    start_ms = now_ms - int(days) * 24 * 60 * 60 * 1000

    fills = _iter_user_fills(addr, start_ms, now_ms, aggregate=True)
    rows = _group_roundtrips_from_fills(fills)
    _join_funding(addr, rows, fudge_ms=60_000)

    saved = 0
    dup = 0
    for r in rows:
        try:
            ok = save_closed_position(db_path, r)
            if ok: saved += 1
            else: dup += 1
            if debug or PRINT_CLOSED_DEBUG:
                _p(f"🔎 {r['symbol']} size={r['size']} entry={r['entry_price']} close={r['close_price']} "
                   f"pnl={r['pnl']} fees={r['fee_total']} funding={r['funding_total']} realized={r['realized_pnl']} "
                   f"open={r['open_time']} close={r['close_time']}")
        except Exception as e:
            _p(f"❌ Error guardando {EXCHANGE} closed: {e}")
    if debug:
        _p(f"✅ Guardadas {saved} (omitidas {dup}).")
    return saved

# ---------------------- Debug utilities ----------------------
def debug_preview_hyperliquid_closed(days: int = 3, symbol: Optional[str] = None):
    addr = _require_addr()
    now_ms = utc_now_ms()
    start_ms = now_ms - days * 24 * 60 * 60 * 1000
    fills = _iter_user_fills(addr, start_ms, now_ms, aggregate=True)
    rows = _group_roundtrips_from_fills(fills)
    _join_funding(addr, rows)
    for r in rows:
        if symbol and normalize_symbol(symbol) != r["symbol"]:
            continue
        price_pnl = float((r.get("pnl") or 0.0))
        recomputed = float(r["realized_pnl"] - r["funding_total"] - r["fee_total"])
        _p(json.dumps({**r, "price_pnl_check": recomputed}, ensure_ascii=False))

def debug_dump_hyperliquid_opens():
    addr = _require_addr()
    return _post_info({"type": "clearinghouseState", "user": addr, "dex": DEX_NAME})

def debug_dump_hyperliquid_funding(days: int = 7):
    addr = _require_addr()
    now_ms = utc_now_ms()
    start_ms = now_ms - days * 24 * 60 * 60 * 1000
    return _post_info({"type": "userFunding", "user": addr, "startTime": start_ms, "endTime": now_ms})

__all__ = [
    "fetch_hyperliquid_open_positions",
    "fetch_hyperliquid_funding_fees",
    "fetch_hyperliquid_all_balances",
    "save_hyperliquid_closed_positions",
    "debug_preview_hyperliquid_closed",
    "debug_dump_hyperliquid_opens",
    "debug_dump_hyperliquid_funding",
]

# ---------------------- CLI ----------------------
if __name__ == "__main__":
    import argparse, sqlite3

    ap = argparse.ArgumentParser("Hyperliquid adapter debug")
    ap.add_argument("--dry-run", action="store_true", default=True, help="No persist (default)")
    ap.add_argument("--save-closed", action="store_true", help="Persist reconstructed closed positions")
    ap.add_argument("--opens", action="store_true", help="Print normalized open positions and raw")
    ap.add_argument("--funding", type=int, default=50, help="Fetch N funding records")
    ap.add_argument("--days", type=int, default=7, help="Days back for fills/funding")
    args = ap.parse_args()

    if args.opens:
        raw = debug_dump_hyperliquid_opens()
        norm = fetch_hyperliquid_open_positions()
        _p("=== RAW clearinghouseState ===")
        _p(json.dumps(raw, indent=2))
        _p("=== NORMALIZED opens ===")
        _p(json.dumps(norm, indent=2))

    if args.funding:
        items = fetch_hyperliquid_funding_fees(limit=args.funding, days=args.days)
        _p(f"Funding items: {len(items)}")
        _p(json.dumps(items[:min(5, len(items))], indent=2))

    if args.save_closed:
        if args.dry_run:
            debug_preview_hyperliquid_closed(days=args.days)
        else:
            save_hyperliquid_closed_positions("portfolio.db", days=args.days, debug=True)
            _p("✅ Saved closed positions.")


