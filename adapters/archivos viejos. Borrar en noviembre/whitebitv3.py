# adapters/whitebit.py
# WhiteBIT adapter — reconstrucción por positionId usando SOLO el último trade
# Sin dependencias del backend. Requiere:
#   - WHITEBIT_API_KEY
#   - WHITEBIT_API_SECRET

import os, time, hmac, hashlib, base64, json, re
from typing import Any, Dict, List, Optional, Tuple
import requests

WHITEBIT_API_KEY   = os.getenv("WHITEBIT_API_KEY", "").strip()
WHITEBIT_API_SECRET= os.getenv("WHITEBIT_API_SECRET", "").strip()
WHITEBIT_BASE_URL  = "https://whitebit.com"
TIMEOUT            = 30

__all__ = [
    "fetch_whitebit_open_positions",
    "fetch_whitebit_funding_fees",
    "fetch_whitebit_all_balances",
    "save_whitebit_closed_positions",
    "debug_whitebit_position_last_trade",
]

# ========= Normalización símbolo =========
def normalize_symbol(sym: str) -> str:
    if not sym:
        return ""
    s = sym.upper()
    s = re.sub(r'^PERP_', '', s)
    s = re.sub(r'(_|-)?(USDT|USDC|USD|PERP)$', '', s)
    s = re.sub(r'[_-]+$', '', s)
    s = re.split(r'[_-]', s)[0]
    return s

# ========= Auth privada =========
def _now_ms() -> int:
    return int(time.time() * 1000)

def _auth_headers(path: str, payload: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    if not WHITEBIT_API_KEY or not WHITEBIT_API_SECRET:
        raise RuntimeError("Faltan WHITEBIT_API_KEY / WHITEBIT_API_SECRET en el entorno.")
    body = dict(payload)
    body["request"] = path
    body.setdefault("nonce", _now_ms())
    body_json = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
    payload_b64 = base64.b64encode(body_json.encode("utf-8"))
    signature = hmac.new(WHITEBIT_API_SECRET.encode("utf-8"), payload_b64, hashlib.sha512).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-TXC-APIKEY": WHITEBIT_API_KEY,
        "X-TXC-PAYLOAD": payload_b64.decode(),
        "X-TXC-SIGNATURE": signature,
    }
    return headers, body

def _post(path: str, payload: Dict[str, Any]) -> Any:
    headers, body = _auth_headers(path, payload)
    r = requests.post(f"{WHITEBIT_BASE_URL}{path}", data=json.dumps(body), headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

# ========= Utilidades =========
def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _side_from_amounts(amts: List[float]) -> str:
    # regla: si hay ALGÚN amount < 0 ⇒ short; si hay ALGÚN amount > 0 ⇒ long; fallback long
    has_neg = any(a < 0 for a in amts)
    has_pos = any(a > 0 for a in amts)
    if has_neg:
        return "short"
    if has_pos:
        return "long"
    return "long"

# ========= OPEN POSITIONS (sin cambios de lógica) =========
def fetch_whitebit_open_positions(market: Optional[str] = None) -> List[Dict[str, Any]]:
    payload = {}
    if market:
        payload["market"] = market
    try:
        data = _post("/api/v4/collateral-account/positions/open", payload)
    except Exception as e:
        print(f"❌ WhiteBIT open positions error: {e}")
        return []

    out = []
    for row in data or []:
        try:
            mkt = row.get("market") or ""
            sym = normalize_symbol(mkt)
            # amount en abiertas es neto con signo (sirve para side)
            side = "short" if _f(row.get("amount")) < 0 else "long"
            size = abs(_f(row.get("amount")))  # posición neta actual
            entry = _f(row.get("basePrice"))
            liq = _f(row.get("liquidationPrice")) if row.get("liquidationPrice") is not None else 0.0
            # pnl en dinero (no mark), notional aproximado con entry
            unreal = _f(row.get("pnl"))
            notional = abs(size * (entry if entry > 0 else 0.0))

            out.append({
                "exchange": "whitebit",
                "symbol": sym,
                "side": side,
                "size": size,
                "entry_price": entry,
                "mark_price": entry,           # WhiteBIT no da mark: dejamos entry (front lo acepta)
                "liquidation_price": liq,
                "notional": notional,
                "unrealized_pnl": unreal,
                "fee": 0.0,
                "funding_fee": 0.0,
                "realized_pnl": 0.0,
            })
        except Exception as e:
            print(f"❌ parse open row: {e}")
    return out

# ========= FUNDING (histórico) =========
def fetch_whitebit_funding_fees(limit: int = 50, market: Optional[str] = None,
                                start_ts: Optional[int] = None, end_ts: Optional[int] = None) -> List[Dict[str, Any]]:
    mkts: List[str] = [market] if market else []
    if not mkts:
        try:
            opens = _post("/api/v4/collateral-account/positions/open", {})
            mkts = [r.get("market") for r in opens if r.get("market")]
        except Exception:
            pass
    if not mkts:
        mkts = ["BTC_PERP", "ETH_PERP", "SOL_PERP", "BNB_PERP"]

    out: List[Dict[str, Any]] = []
    for m in list(dict.fromkeys(mkts)):
        payload = {"market": m, "limit": max(1, min(100, int(limit)))}
        if start_ts: payload["startDate"] = int(start_ts)
        if end_ts:   payload["endDate"] = int(end_ts)
        try:
            res = _post("/api/v4/collateral-account/funding-history", payload) or {}
            for rec in res.get("records", []):
                sym = normalize_symbol(rec.get("market") or "")
                ts_sec = int(_f(rec.get("fundingTime"), 0.0))
                out.append({
                    "exchange": "whitebit",
                    "symbol": sym,
                    "income": _f(rec.get("fundingAmount")),
                    "asset": "USDT",
                    "timestamp": ts_sec * 1000,
                    "funding_rate": _f(rec.get("fundingRate", 0.0)),
                    "type": "FUNDING_FEE",
                })
        except Exception as e:
            print(f"❌ funding error {m}: {e}")
            continue
    return out

# ========= BALANCES =========
def fetch_whitebit_all_balances() -> Dict[str, Any]:
    spot_total, futures_total, unreal, initial_margin = 0.0, 0.0, 0.0, 0.0
    try:
        tr = _post("/api/v4/trade-account/balance", {}) or {}
        for t, v in tr.items():
            if t in ("USDT", "USDC", "USD"):
                spot_total += _f(v.get("available")) + _f(v.get("freeze"))
    except Exception as e:
        print(f"⚠️ spot balance: {e}")
    try:
        coll = _post("/api/v4/collateral-account/balance", {}) or {}
        for t, amt in coll.items():
            if t in ("USDT", "USDC", "USD"):
                futures_total += _f(amt)
        opens = _post("/api/v4/collateral-account/positions/open", {}) or []
        for r in opens:
            unreal += _f(r.get("pnl"))
            initial_margin += _f(r.get("margin"))
    except Exception as e:
        print(f"⚠️ futures balance: {e}")
    equity = spot_total + futures_total + unreal
    return {
        "exchange": "whitebit",
        "equity": float(equity),
        "balance": float(spot_total + futures_total),
        "unrealized_pnl": float(unreal),
        "initial_margin": float(initial_margin) if initial_margin > 0 else 0.0,
        "spot": float(spot_total),
        "margin": 0.0,
        "futures": float(futures_total),
    }

# ========= CLOSED (último trade) =========
def _group_by_position(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    d: Dict[int, List[Dict[str, Any]]] = {}
    for st in rows:
        try:
            pid = int(st.get("positionId"))
        except Exception:
            continue
        d.setdefault(pid, []).append(st)
    for pid in d:
        d[pid].sort(key=lambda r: _f(r.get("modifyDate")))  # ascendente
    return d

def _od(st: Dict[str, Any]) -> Dict[str, Any]:
    return st.get("orderDetail") or {}

def _entry_close_prices(legs: List[Dict[str, Any]]) -> Tuple[float, float]:
    first, last = legs[0], legs[-1]
    ep = _f(_od(first).get("price")) or _f(first.get("basePrice"))
    cp = _f(_od(last).get("price"))  or _f(last.get("basePrice")) or ep
    return ep, cp

def _fees_sum(legs: List[Dict[str, Any]]) -> float:
    return sum(_f(_od(st).get("tradeFee")) for st in legs)

def _funding_sum(legs: List[Dict[str, Any]]) -> float:
    return sum(_f(_od(st).get("fundingFee")) for st in legs)

def _max_abs_amount(legs: List[Dict[str, Any]]) -> float:
    return max((abs(_f(st.get("amount"))) for st in legs), default=0.0)

def _side_from_legs(legs: List[Dict[str, Any]]) -> str:
    amts = [_f(st.get("amount")) for st in legs if st.get("amount") is not None]
    return _side_from_amounts(amts)

def save_whitebit_closed_positions(db_path: str = "portfolio.db", days: int = 50, debug: bool = False) -> Tuple[int, int]:
    try:
        from db_manager import save_closed_position
    except Exception as e:
        print(f"❌ db_manager.save_closed_position no disponible: {e}")
        return (0, 0)

    # descarga positions/history con ventana
    end_ts = int(time.time())
    start_ts = end_ts - days * 86400 if days else None
    limit, offset = 100, 0
    all_rows: List[Dict[str, Any]] = []

    while True:
        payload = {"limit": limit, "offset": offset}
        if start_ts: payload["startDate"] = int(start_ts)
        if end_ts:   payload["endDate"] = int(end_ts)
        try:
            chunk = _post("/api/v4/collateral-account/positions/history", payload) or []
        except Exception as e:
            print(f"❌ history error: {e}")
            break
        if not chunk:
            break
        all_rows.extend(chunk)
        if len(chunk) < limit:
            break
        offset += limit

    if not all_rows:
        if debug: print("⚠️ WhiteBIT: sin histórico en la ventana.")
        return (0, 0)

    grouped = _group_by_position(all_rows)

    # conexión sqlite para evitar duplicados
    import sqlite3, os as _os
    if not _os.path.exists(db_path):
        print(f"❌ DB no encontrada: {db_path}")
        return (0, 0)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    saved = skipped = 0
    for pid, legs in grouped.items():
        try:
            mkt  = (legs[-1].get("market") or "").upper()
            sym  = normalize_symbol(mkt)
            side = _side_from_legs(legs)
            size = _max_abs_amount(legs)

            open_s  = int(_f(legs[0].get("openDate")))
            close_s = int(_f(legs[-1].get("modifyDate")))

            # evita duplicados
            cur.execute("SELECT COUNT(*) FROM closed_positions WHERE exchange=? AND symbol=? AND close_time=?",
                        ("whitebit", sym, close_s))
            if cur.fetchone()[0] > 0:
                skipped += 1
                continue

            entry_price, close_price = _entry_close_prices(legs)
            fees_pos_sum     = _fees_sum(legs)             # positivo (API)
            fee_total_db     = -abs(fees_pos_sum)          # siempre negativo en DB
            funding_total    = _funding_sum(legs)          # signo natural (API)
            realized_last    = _f(_od(legs[-1]).get("realizedPnl"))

            # precio puro: realized(último) + fees_pos_sum
            price_pnl        = realized_last + fees_pos_sum
            # realized DB: realized(último) + funding_total (fees ya incluidas en realized)
            realized_db      = realized_last + funding_total

            notional         = abs(size) * entry_price

            if debug:
                print(f"\n📋 PID {pid} {mkt} -> {sym}")
                print(f" side={side} | size(max|amount|)={size}")
                print(f" entry/close: {entry_price} / {close_price}")
                print(f" fees_sum(+)={fees_pos_sum}  -> fee_total(DB)={fee_total_db}")
                print(f" funding_total={funding_total}")
                print(f" realized_last={realized_last}")
                print(f" pnl(precio)={price_pnl} | realized(DB)={realized_db}")
                print(f" open/close ts: {open_s} / {close_s}")

            save_closed_position({
                "exchange": "whitebit",
                "symbol": sym,
                "side": side,
                "size": abs(size),
                "entry_price": float(entry_price),
                "close_price": float(close_price),
                "open_time": open_s,
                "close_time": close_s,
                "pnl": float(price_pnl),
                "realized_pnl": float(realized_db),
                "funding_total": float(funding_total),
                "fee_total": float(fee_total_db),
                "notional": float(notional),
                "leverage": None,
                "initial_margin": None,
                "liquidation_price": None,
            })
            saved += 1
        except Exception as e:
            print(f"❌ build/save error pid={pid}: {e}")
            continue

    conn.close()
    print(f"✅ WhiteBIT guardadas={saved} | omitidas={skipped}")
    return (saved, skipped)

# ========= DEBUG: ver un positionId (último trade y objeto final) =========
def debug_whitebit_position_last_trade(position_id: int, days: int = 50):
    payload = {"limit": 100, "offset": 0}
    end_ts = int(time.time())
    start_ts = end_ts - days * 86400 if days else None
    if start_ts: payload["startDate"] = int(start_ts)
    if end_ts:   payload["endDate"] = int(end_ts)
    payload["positionId"] = int(position_id)
    rows = _post("/api/v4/collateral-account/positions/history", payload) or []
    if not rows:
        print(f"⚠️ Sin filas para positionId={position_id}")
        return
    rows.sort(key=lambda r: _f(r.get("modifyDate")))
    mkt = (rows[-1].get("market") or "").upper()
    sym = normalize_symbol(mkt)
    side = _side_from_legs(rows)
    size = _max_abs_amount(rows)
    ep, cp = _entry_close_prices(rows)
    fees_pos_sum  = _fees_sum(rows)
    funding_total = _funding_sum(rows)
    realized_last = _f(_od(rows[-1]).get("realizedPnl"))
    price_pnl     = realized_last + fees_pos_sum
    realized_db   = realized_last + funding_total
    open_s  = int(_f(rows[0].get("openDate")))
    close_s = int(_f(rows[-1].get("modifyDate")))

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print("\n— CÁLCULO —")
    print(f"symbol={sym} side={side} size={size}")
    print(f"entry/close={ep} / {cp}")
    print(f"fees_sum(+)={fees_pos_sum} -> fee_total(DB)={-abs(fees_pos_sum)}")
    print(f"funding_total={funding_total}")
    print(f"realized_last={realized_last}")
    print(f"pnl(precio)={price_pnl} | realized(DB)={realized_db}")
    print(f"open/close ts: {open_s} / {close_s}")

# ========= CLI =========
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("whitebit adapter (último trade)")
    ap.add_argument("--opens", action="store_true")
    ap.add_argument("--funding", type=int, default=0)
    ap.add_argument("--market", type=str, default=None)
    ap.add_argument("--save-closed", action="store_true")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--debug-pid", type=int, default=None, help="imprime raw + cálculo de ese positionId")
    args = ap.parse_args()

    if args.opens:
        for r in fetch_whitebit_open_positions(market=args.market):
            print(r)
    if args.funding > 0:
        for r in fetch_whitebit_funding_fees(limit=args.funding, market=args.market):
            print(r)
    if args.debug_pid is not None:
        debug_whitebit_position_last_trade(args.debug_pid, days=args.days)
    if args.save_closed:
        save_whitebit_closed_positions(days=args.days, debug=True)


