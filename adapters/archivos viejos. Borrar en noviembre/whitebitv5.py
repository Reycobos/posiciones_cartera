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
                                start_ts: Optional[int] = None, end_ts: Optional[int] = None,
                                days_discovery: int = 50) -> List[Dict[str, Any]]:
    """
    Descubre mercados válidos desde TU CUENTA (abiertas + history últimos N días)
    para evitar 422 en markets inexistentes. Si se pasa 'market', usa solo ese.
    """

    def _discover_markets(days: int = 50) -> List[str]:
        found: List[str] = []
        # 1) Posiciones abiertas
        try:
            opens = _post("/api/v4/collateral-account/positions/open", {}) or []
            found.extend([str(r.get("market")).upper() for r in opens if r.get("market")])
        except Exception:
            pass
        # 2) Positions history (últimos N días)
        try:
            end_ = int(time.time())
            start_ = end_ - max(1, int(days)) * 86400
            payload = {"limit": 100, "offset": 0, "startDate": start_, "endDate": end_}
            hist = _post("/api/v4/collateral-account/positions/history", payload) or []
            found.extend([str(r.get("market")).upper() for r in hist if r.get("market")])
        except Exception:
            pass
        # Normaliza y filtra a *_PERP
        uniq = []
        for m in found:
            if not m or not isinstance(m, str):
                continue
            m = m.upper()
            if not m.endswith("_PERP"):
                continue
            if m not in uniq:
                uniq.append(m)
        return uniq

    mkts: List[str] = [market.upper()] if market else _discover_markets(days_discovery)
    # Si no se descubrió ninguno, devolvemos vacío para no provocar 422
    if not mkts:
        return []
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
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code == 422:
                # Mercados no válidos o no soportados -> saltar sin ruido
                print(f"⏭️ WhiteBIT funding: mercado no válido o sin soporte, se omite: {m}")
                continue
            else:
                print(f"❌ WhiteBIT funding HTTP error {m}: {e}")
                continue
        except Exception as e:
            print(f"❌ WhiteBIT funding error {m}: {e}")
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


# =============== POSICIONES CERRADAS (WhiteBIT) — SIMPLE, ESTILO GATE ===============

from typing import Any, Dict, List, Optional, Tuple
import time, sqlite3

def fetch_whitebit_closed_positions(days_back: int = 30,
                                    market: Optional[str] = None,
                                    limit: int = 100,
                                    max_pages: int = 50) -> List[Dict[str, Any]]:
    """
    Descarga posiciones desde /collateral-account/positions/history y devuelve
    una lista normalizada (como Gate), SIN guardar aún.
    - Funding: usa realizedFunding del ÚLTIMO estado; si no viene, suma orderDetail.fundingFee.
    - Fees: suma de orderDetail.tradeFee (positivo).
    - PnL precio: suma de orderDetail.realizedPnl.
    - realized_pnl = pnl_precio + funding_fee.
    """
    print(f"🔍 WhiteBIT CLOSED: últimos {days_back} días" + (f" | market={market}" if market else ""))

    end_ts = int(time.time())
    start_ts = end_ts - days_back * 24 * 3600
    offset = 0
    limit = max(1, min(int(limit), 100))

    # 1) Paginación simple
    rows: List[Dict[str, Any]] = []
    for _ in range(max_pages):
        payload = {"limit": limit, "offset": offset, "startDate": start_ts, "endDate": end_ts}
        if market:
            payload["market"] = market
        page = _post("/api/v4/collateral-account/positions/history", payload) or []  # usa tu _post probado
        if not page:
            break
        rows.extend(page)
        if len(page) < limit:
            break
        offset += limit

    if not rows:
        print("⚠️ WhiteBIT CLOSED: sin datos en el rango.")
        return []

    # 2) Agrupar por positionId y ordenar por modifyDate
    by_pid: Dict[int, List[Dict[str, Any]]] = {}
    for st in rows:
        try:
            pid = int(st.get("positionId"))
        except Exception:
            continue
        by_pid.setdefault(pid, []).append(st)
    for pid in by_pid:
        by_pid[pid].sort(key=lambda s: float(s.get("modifyDate") or 0.0))

    # 3) Normalizar cada posición cerrada
    out: List[Dict[str, Any]] = []
    for pid, states in by_pid.items():
        last = states[-1]

        # Cerrada si último amount == 0 o liquidation
        closed_flag = False
        try:
            if float(states[-1].get("amount") or 0) == 0.0:
                closed_flag = True
        except Exception:
            pass
        if (last.get("liquidationState") or "").lower() == "liquidation":
            closed_flag = True
        if not closed_flag:
            continue

        mkt = (last.get("market") or "").upper()
        symbol = normalize_symbol(mkt)  # usa tu normalizador interno
        # side por cantidades observadas (usa tu helper _side_from_amounts)
        amts = []
        for st in states:
            try:
                amts.append(float(st.get("amount") or 0))
            except Exception:
                pass
        side = _side_from_amounts(amts)  # "long" | "short"

        # entry: primer basePrice no nulo (o primer orderDetail.price)
        entry_price = None
        for st in states:
            if st.get("basePrice") is not None:
                entry_price = float(st["basePrice"])
                break
            od = st.get("orderDetail") or {}
            if od.get("price") is not None:
                entry_price = float(od["price"])
                break
        entry_price = entry_price or 0.0

        # close: último orderDetail.price; si no hay, último basePrice; fallback: entry
        close_price = None
        for st in reversed(states):
            od = st.get("orderDetail") or {}
            if od.get("price") is not None:
                close_price = float(od["price"])
                break
            if st.get("basePrice") is not None and close_price is None:
                close_price = float(st["basePrice"])
        if close_price is None:
            close_price = entry_price

        # size = máximo |amount|
        size = 0.0
        for st in states:
            try:
                size = max(size, abs(float(st.get("amount") or 0)))
            except Exception:
                pass

        # tiempos
        try:
            open_time = int(float(states[0].get("openDate") or 0))
        except Exception:
            open_time = None
        try:
            close_time = int(float(states[-1].get("modifyDate") or 0))
        except Exception:
            close_time = None

        # fees (positivo)
        fees = 0.0
        # funding: preferir realizedFunding del último estado
        rf_last = last.get("realizedFunding")
        funding_fee = float(rf_last) if rf_last is not None else 0.0
        # PnL precio puro (suma realizedPnl en orderDetail)
        pnl_price = 0.0

        for st in states:
            od = st.get("orderDetail") or {}
            if od.get("tradeFee") is not None:
                try:
                    fees += float(od["tradeFee"])
                except Exception:
                    pass
            if od.get("realizedPnl") is not None:
                try:
                    pnl_price += float(od["realizedPnl"])
                except Exception:
                    pass
            # fallback funding por evento
            if rf_last is None and od.get("fundingFee") is not None:
                try:
                    funding_fee += float(od["fundingFee"])
                except Exception:
                    pass

        realized_pnl = pnl_price + funding_fee
        notional = abs(size) * (entry_price if entry_price else 0.0)

        # liq (si viene)
        liq = last.get("liquidationPrice")
        try:
            liquidation_price = float(liq) if liq is not None else None
        except Exception:
            liquidation_price = None

        out.append({
            "exchange": "whitebit",
            "symbol": symbol,
            "side": side,
            "size": abs(size),
            "entry_price": entry_price,
            "close_price": close_price,
            "notional": notional,
            "fees": fees,                 # positivo (save_closed_position ya estandariza)
            "funding_fee": funding_fee,   # funding neto
            "pnl": pnl_price,             # solo precio (opcional)
            "realized_pnl": realized_pnl, # total
            "open_time": open_time,
            "close_time": close_time,
            "leverage": None,
            "liquidation_price": liquidation_price,
        })

    print(f"✅ WhiteBIT CLOSED normalizadas: {len(out)}")
    return out


def save_whitebit_closed_positions(db_path: str = "portfolio.db",
                                   days: int = 30,debug=False,
                                   market: Optional[str] = None) -> int:
    """
    Guarda posiciones cerradas en SQLite evitando duplicados por (exchange, symbol, close_time),
    mapeando campos igual que Gate.
    """
    try:
        from db_manager import save_closed_position
    except Exception as e:
        print(f"❌ db_manager.save_closed_position no disponible: {e}")
        return 0

    # Trae ya normalizadas (no toca firmas ni helpers nuevos)
    closed_positions = fetch_whitebit_closed_positions(days_back=days, market=market)
    if not closed_positions:
        print("⚠️ No hay posiciones cerradas que guardar.")
        return 0

    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        print(f"❌ No se pudo abrir DB {db_path}: {e}")
        return 0

    cur = conn.cursor()
    saved = skipped = 0

    for pos in closed_positions:
        try:
            # Dedupe idéntico a Gate: (exchange, symbol, close_time)
            cur.execute("""
                SELECT COUNT(*) FROM closed_positions
                WHERE exchange = ? AND symbol = ? AND close_time = ?
            """, (pos["exchange"], pos["symbol"], pos["close_time"]))
            if cur.fetchone()[0]:
                skipped += 1
                continue

            position_data = {
                "exchange": pos["exchange"],
                "symbol": pos["symbol"],
                "side": pos["side"],
                "size": pos["size"],
                "entry_price": pos["entry_price"],
                "close_price": pos["close_price"],
                "open_time": pos["open_time"],
                "close_time": pos["close_time"],
                "realized_pnl": pos["realized_pnl"],
                "funding_total": pos.get("funding_fee", 0),
                "pnl": pos.get("pnl"),
                "fee_total": pos.get("fees", 0),
                "notional": pos.get("notional", 0),
                "leverage": pos.get("leverage"),
                "liquidation_price": pos.get("liquidation_price"),
            }

            save_closed_position(position_data)
            saved += 1

        except Exception as e:
            print(f"⚠️ Error guardando {pos.get('symbol')} @ {pos.get('close_time')}: {e}")

    conn.close()
    print(f"✅ WhiteBIT guardadas: {saved} | omitidas (duplicadas): {skipped}")
    return saved



