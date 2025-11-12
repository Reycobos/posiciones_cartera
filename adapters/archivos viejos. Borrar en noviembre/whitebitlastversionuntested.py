# -*- coding: utf-8 -*-
import base64
import hashlib
import hmac
import json
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import requests
import sqlite3


WB_BASE = "https://whitebit.com"
ENDPOINT_POS_HISTORY = "/api/v4/collateral-account/positions/history"  # WhiteBIT docs :contentReference[oaicite:2]{index=2}

# ========== Firma WhiteBIT V4 ==========
# Body JSON con:
#   - request: path del endpoint
#   - nonce: entero creciente (recomendado: timestamp ms)
# Headers:
#   - Content-Type: application/json
#   - X-TXC-APIKEY: <api_key>
#   - X-TXC-PAYLOAD: base64(body_json)
#   - X-TXC-SIGNATURE: hex(HMAC_SHA512(X-TXC-PAYLOAD, api_secret))
# Referencia oficial de autenticación. :contentReference[oaicite:3]{index=3}
def _wb_signed_post(endpoint: str, api_key: str, api_secret: str, body: Dict[str, Any]) -> requests.Response:
    body_full = dict(body or {})
    body_full.setdefault("request", endpoint)
    body_full.setdefault("nonce", int(time.time() * 1000))

    raw = json.dumps(body_full, separators=(",", ":"), ensure_ascii=False)
    payload_b64 = base64.b64encode(raw.encode()).decode()
    sign = hmac.new(api_secret.encode(), payload_b64.encode(), hashlib.sha512).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "X-TXC-APIKEY": api_key,
        "X-TXC-PAYLOAD": payload_b64,
        "X-TXC-SIGNATURE": sign,
    }
    return requests.post(WB_BASE + endpoint, data=raw, headers=headers, timeout=30)


def _f(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _first_not_none(seq: List[Any], key: str) -> Optional[Any]:
    for s in seq:
        v = s.get(key)
        if v is not None:
            return v
    return None


def _last_not_none(seq: List[Any], key: str) -> Optional[Any]:
    for s in reversed(seq):
        v = s.get(key)
        if v is not None:
            return v
    return None


def _side_norm(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.upper()
    if s in ("LONG", "SHORT"):
        return s.lower()
    return s.lower()  # BOTH -> both (raro) o lo que devuelva


def _is_closed(group: List[Dict[str, Any]]) -> bool:
    last = group[-1]
    amt = last.get("amount")
    liq = last.get("liquidationState")
    # Cerrada si amount == 0 (string/num) o liquidada.
    try:
        if float(amt) == 0.0:
            return True
    except Exception:
        pass
    return (liq or "").lower() == "liquidation"


def _pick_prices(states: List[Dict[str, Any]]) -> Tuple[float, float]:
    # entry: primer basePrice no nulo
    entry = _first_not_none(states, "basePrice")
    # close: último orderDetail.price no nulo; fallback: último basePrice
    last_order_price = None
    for st in reversed(states):
        od = st.get("orderDetail") or {}
        if od.get("price") is not None:
            last_order_price = od["price"]
            break
    close = last_order_price if last_order_price is not None else _last_not_none(states, "basePrice")
    return _f(entry), _f(close)


def _calc_funding(states: List[Dict[str, Any]]) -> float:
    # Regla: usa realizedFunding del último estado (acumulado de toda la vida).
    last_rf = _last_not_none(states, "realizedFunding")
    if last_rf is not None:
        return _f(last_rf)
    # Fallback: suma de orderDetail.fundingFee por evento si existiera
    s = 0.0
    for st in states:
        od = st.get("orderDetail") or {}
        if od.get("fundingFee") is not None:
            s += _f(od["fundingFee"])
    return s


def _calc_trade_fees(states: List[Dict[str, Any]]) -> float:
    # Suma positiva de tradeFee; la DB la guarda negativa.
    s = 0.0
    for st in states:
        od = st.get("orderDetail") or {}
        if od.get("tradeFee") is not None:
            s += _f(od["tradeFee"])
    return s


def _calc_price_realized(states: List[Dict[str, Any]]) -> float:
    # Suma de orderDetail.realizedPnl (cuando exista). Si WhiteBIT no lo rellena, queda 0.
    s = 0.0
    for st in states:
        od = st.get("orderDetail") or {}
        if od.get("realizedPnl") is not None:
            s += _f(od["realizedPnl"])
    return s


def _max_abs_amount(states: List[Dict[str, Any]]) -> float:
    mx = 0.0
    for st in states:
        mx = max(mx, abs(_f(st.get("amount"))))
    return mx


def _symbol_from_market(mkt: str) -> str:
    # En WhiteBIT, futures suelen ser XXX_PERP; margen/spot: XXX_USDT, etc.
    return mkt  # mantén el naming nativo; ajusta aquí si necesitas normalizar.


def _insert_many(sqlite_path: str, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    cols = [
        "exchange", "symbol", "side", "size", "entry_price", "close_price",
        "open_time", "close_time", "pnl", "realized_pnl", "funding_total",
        "fee_total", "pnl_percent", "apr", "initial_margin", "notional",
        "leverage", "liquidation_price", "created_at"
    ]
    placeholders = ",".join("?" * len(cols))
    with sqlite3.connect(sqlite_path) as con:
        cur = con.cursor()
        # Evita duplicados básicos por (exchange, symbol, close_time, side)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS closed_positions (
                exchange TEXT, symbol TEXT, side TEXT, size REAL,
                entry_price REAL, close_price REAL,
                open_time INTEGER, close_time INTEGER,
                pnl REAL, realized_pnl REAL, funding_total REAL,
                fee_total REAL, pnl_percent REAL, apr REAL,
                initial_margin REAL, notional REAL, leverage REAL,
                liquidation_price REAL, created_at INTEGER
            )
        """)
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_closed_unique
            ON closed_positions (exchange, symbol, close_time, side)
        """)
        inserted = 0
        for r in rows:
            vals = [r.get(c) for c in cols]
            try:
                cur.execute(f"INSERT OR IGNORE INTO closed_positions ({','.join(cols)}) VALUES ({placeholders})", vals)
                inserted += cur.rowcount  # 1 si inserta, 0 si ignora
            except sqlite3.IntegrityError:
                pass
        con.commit()
    return inserted


def fetch_whitebit_closed_positions(
    db_path: str,
    api_key: str,
    api_secret: str,
    market: Optional[str] = None,
    start_date: Optional[int] = None,  # unix seconds
    end_date: Optional[int] = None,    # unix seconds
    limit: int = 100,                  # 1..100 (doc)
    max_pages: int = 50,
    debug: bool = False,
) -> int:
    """
    Descarga posiciones (histórico) desde WhiteBIT y guarda SOLO las cerradas
    en la tabla closed_positions.
    - Endpoint oficial: /api/v4/collateral-account/positions/history  :contentReference[oaicite:4]{index=4}
    - Auth requerida con X-TXC-* headers (ver docs).                 :contentReference[oaicite:5]{index=5}

    Retorna: número de filas insertadas (nuevas).
    """
    assert 1 <= limit <= 100, "limit fuera de rango (1..100)"
    offset = 0
    all_states: List[Dict[str, Any]] = []

    for _ in range(max_pages):
        body = {
            "limit": limit,
            "offset": offset,
        }
        if market:
            body["market"] = market
        if start_date:
            body["startDate"] = int(start_date)
        if end_date:
            body["endDate"] = int(end_date)

        resp = _wb_signed_post(ENDPOINT_POS_HISTORY, api_key, api_secret, body)
        if resp.status_code != 200:
            # Devuelve 422 en validaciones internas, según docs.
            msg = f"WhiteBIT {ENDPOINT_POS_HISTORY} {resp.status_code}: {resp.text}"
            raise RuntimeError(msg)

        page = resp.json()
        if not isinstance(page, list):
            raise RuntimeError(f"Respuesta inesperada (no es lista): {page}")

        if not page:
            break

        all_states.extend(page)
        if len(page) < limit:
            break
        offset += limit

    # Agrupa por (positionId, market, positionSide)
    groups: Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for st in all_states:
        key = (st.get("positionId"), st.get("market"), st.get("positionSide"))
        groups[key].append(st)

    # Ordena cada grupo por modifyDate
    for k in groups:
        groups[k].sort(key=lambda s: _f(s.get("modifyDate")))

    # Construye filas solo para cerradas
    rows: List[Dict[str, Any]] = []
    now_ts = int(time.time())

    for (pid, mkt, pos_side), states in groups.items():
        if not _is_closed(states):
            continue

        entry_price, close_price = _pick_prices(states)
        size = _max_abs_amount(states)

        open_time = int(_f(states[0].get("openDate")))
        close_time = int(_f(states[-1].get("modifyDate")))
        duration = max(1, close_time - open_time)

        funding_total = _calc_funding(states)                  # acumulado (API)
        trade_fees_pos = _calc_trade_fees(states)              # suma positiva
        fee_total_db = -abs(trade_fees_pos)                    # convención DB: negativo
        price_realized = _calc_price_realized(states)          # PnL precio puro
        realized_pnl = price_realized + funding_total          # total realized

        notional = abs(size) * entry_price if entry_price else 0.0
        pnl_percent = (realized_pnl / notional) if notional > 0 else 0.0
        apr = (pnl_percent * (365 * 24 * 3600) / duration) if duration > 0 else 0.0

        liquidation_price = states[-1].get("liquidationPrice")
        symbol = _symbol_from_market(mkt)
        side = _side_norm(pos_side)

        row = {
            "exchange": "whitebit",
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "close_price": close_price,
            "open_time": open_time,
            "close_time": close_time,
            "pnl": price_realized,
            "realized_pnl": realized_pnl,
            "funding_total": funding_total,
            "fee_total": fee_total_db,
            "pnl_percent": pnl_percent,
            "apr": apr,
            "initial_margin": None,       # No lo expone este endpoint
            "notional": notional,
            "leverage": None,             # No lo expone este endpoint
            "liquidation_price": _f(liquidation_price) if liquidation_price is not None else None,
            "created_at": now_ts,
        }
        if debug:
            print(json.dumps({
                "pid": pid, "market": mkt, "side": side,
                "size": size, "entry": entry_price, "close": close_price,
                "funding_total": funding_total,
                "trade_fees_sum(+)": trade_fees_pos,
                "fee_total(DB)": fee_total_db,
                "price_realized": price_realized,
                "realized_pnl": realized_pnl,
                "open_time": open_time, "close_time": close_time,
            }, ensure_ascii=False, indent=2))
        rows.append(row)

    inserted = _insert_many(db_path, rows)
    if debug:
        print(f"[whitebit] closed_positions guardadas: {inserted} nuevas / {len(rows)} calculadas")
    return inserted


if __name__ == "__main__":
    # Ejemplo rápido (rellena con tus credenciales y DB)
    import os
    DB = os.getenv("PORTFOLIO_DB", "portfolio.db")
    KEY = os.getenv("WHITEBIT_API_KEY", "")
    SEC = os.getenv("WHITEBIT_API_SECRET", "")
    # Filtra, por ejemplo, solo futuros BTC_PERP en los últimos 90 días:
    # start = int(time.time()) - 90*24*3600
    start = None
    end = None
    print(fetch_whitebit_closed_positions(DB, KEY, SEC, market=None, start_date=start, end_date=end, limit=100, debug=True))

