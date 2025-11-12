# adapters/whitebit.py
# WhiteBIT adapter — standalone (sin dependencias del backend)
# Requiere: WHITEBIT_API_KEY, WHITEBIT_API_SECRET en entorno.
# Endpoints usados (privados, POST):
#   - /api/v4/trade-account/balance                   (spot/trade)
#   - /api/v4/main-account/balance                    (main opcional)
#   - /api/v4/collateral-account/balance              (futuros - assets)
#   - /api/v4/collateral-account/balance-summary      (futuros - resumen)
#   - /api/v4/collateral-account/positions/open       (posiciones abiertas)
#   - /api/v4/collateral-account/positions/history    (histórico posiciones)
#   - /api/v4/collateral-account/funding-history      (funding realizado)

import os
import time
import hmac
import hashlib
import base64
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
load_dotenv()

WHITEBIT_API_KEY = os.getenv("WHITEBIT_API_KEY", "")
WHITEBIT_API_SECRET = os.getenv("WHITEBIT_API_SECRET", "")
WHITEBIT_BASE_URL = "https://whitebit.com"
TIMEOUT = 30

__all__ = [
    "fetch_whitebit_open_positions",
    "fetch_whitebit_funding_fees",
    "fetch_whitebit_all_balances",
    "save_whitebit_closed_positions",
]

# =========================
# Normalización de símbolos (contrato A)
# =========================
def normalize_symbol(sym: str) -> str:
    if not sym:
        return ""
    s = sym.upper()
    s = re.sub(r'^PERP_', '', s)
    s = re.sub(r'(_|-)?(USDT|USDC|PERP)$', '', s)
    s = re.sub(r'[_-]+$', '', s)
    s = re.split(r'[_-]', s)[0]
    return s

# =========================
# Helpers privados de auth
# =========================
def _now_ms() -> int:
    return int(time.time() * 1000)

def _wb_headers(path: str, payload: Dict[str, Any]) -> Dict[str, str]:
    """
    WhiteBIT auth:
      - Body JSON con 'request' = path y 'nonce' = timestamp ms
      - X-TXC-PAYLOAD = base64(JSON)
      - X-TXC-SIGNATURE = hex(HMAC_SHA512(payload_b64, api_secret))
    """
    if not WHITEBIT_API_KEY or not WHITEBIT_API_SECRET:
        raise ValueError("Faltan credenciales WHITEBIT_API_KEY / WHITEBIT_API_SECRET")

    body = dict(payload)
    body["request"] = path
    body.setdefault("nonce", _now_ms())

    body_json = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
    payload_b64 = base64.b64encode(body_json.encode("utf-8"))
    signature = hmac.new(
        WHITEBIT_API_SECRET.encode("utf-8"), payload_b64, hashlib.sha512
    ).hexdigest()

    return {
        "Content-Type": "application/json",
        "X-TXC-APIKEY": WHITEBIT_API_KEY,
        "X-TXC-PAYLOAD": payload_b64.decode(),
        "X-TXC-SIGNATURE": signature,
    }, body

def _post(path: str, payload: Dict[str, Any]) -> Any:
    headers, body = _wb_headers(path, payload)
    url = f"{WHITEBIT_BASE_URL}{path}"
    r = requests.post(url, data=json.dumps(body), headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

# =========================
# Utilidades
# =========================
def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _side_map(position_side: Optional[str]) -> str:
    s = (position_side or "").upper()
    if s == "SHORT":
        return "short"
    return "long"  # por defecto

def _mark_from_unrealized(entry: float, pnl_money: float, size: float, side: str) -> float:
    """
    WhiteBIT entrega 'pnl' en dinero para abiertas; aproximamos mark:
      long : mark = entry + pnl/size
      short: mark = entry - pnl/size
    """
    sz = abs(_safe_float(size))
    if sz <= 0:
        return entry
    if side == "short":
        return entry - (pnl_money / sz)
    return entry + (pnl_money / sz)
# =========================================================
# B) OPEN POSITIONS — shape EXACTO para /api/positions
# =========================================================
def fetch_whitebit_open_positions(market: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Devuelve lista de posiciones abiertas normalizadas:
    {
      exchange,symbol,side,size,entry_price,mark_price,liquidation_price,
      notional,unrealized_pnl,fee,funding_fee,realized_pnl
    }
    """
    payload: Dict[str, Any] = {}
    if market:
        payload["market"] = market

    try:
        data = _post("/api/v4/collateral-account/positions/open", payload)
    except Exception as e:
        print(f"❌ WhiteBIT open positions error: {e}")
        return []

    out: List[Dict[str, Any]] = []
    for row in data or []:
        try:
            market_name = row.get("market") or ""
            sym = normalize_symbol(market_name)
            side = _side_map(row.get("positionSide"))
            size = _safe_float(row.get("amount"))
            entry = _safe_float(row.get("basePrice"))
            liq = _safe_float(row.get("liquidationPrice")) if row.get("liquidationPrice") is not None else 0.0
            pnl_money = _safe_float(row.get("pnl"))  # unrealized PnL (dinero)
            mark = _mark_from_unrealized(entry, pnl_money, size, side)

            # notional: preferimos mark por consistencia visual, fallback entry
            notional = abs(size) * (mark if mark > 0 else entry)

            # WhiteBIT no expone fee/funding acumulados a nivel "open" (realizados)
            fee_total = 0.0
            funding_fee = 0.0
            realized_pnl = fee_total + funding_fee

            unreal = (mark - entry) * size if side == "long" else (entry - mark) * size

            out.append({
                "exchange": "whitebit",
                "symbol": sym,
                "side": side,
                "size": abs(size),
                "entry_price": float(entry),
                "mark_price": float(mark),
                "liquidation_price": float(liq or 0.0),
                "notional": float(notional),
                "unrealized_pnl": float(unreal),
                "fee": float(fee_total),          # NEGATIVO si coste (aquí 0.0)
                "funding_fee": float(funding_fee),# + cobro / - pago (aquí 0.0)
                "realized_pnl": float(realized_pnl),
            })
        except Exception as e:
            print(f"❌ WhiteBIT parse open row error: {e}")
            continue

    return out

# =========================================================
# C) FUNDING FEES — shape EXACTO para /api/funding
# =========================================================
def fetch_whitebit_funding_fees(
    limit: int = 50,
    market: Optional[str] = None,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Funding realizados (histórico). Si no se pasa 'market', intenta deducirlo de abiertas,
    y si no hay, usa un pequeño fallback (BTC_PERP/ETH_PERP/...).
    """
    # compone lista de mercados a consultar
    markets: List[str] = []
    if market:
        markets = [market]
    else:
        try:
            opens = _post("/api/v4/collateral-account/positions/open", {})
            markets.extend([r.get("market") for r in opens if r.get("market")])
        except Exception:
            pass
        if not markets:
            markets = ["BTC_PERP", "ETH_PERP", "SOL_PERP", "BNB_PERP"]

    limit = max(1, min(100, int(limit)))
    out: List[Dict[str, Any]] = []

    for m in list(dict.fromkeys(markets)):
        payload: Dict[str, Any] = {"market": m, "limit": limit}
        if start_ts: payload["startDate"] = int(start_ts)
        if end_ts:   payload["endDate"] = int(end_ts)

        try:
            res = _post("/api/v4/collateral-account/funding-history", payload) or {}
            for rec in res.get("records", []):
                sym = normalize_symbol(rec.get("market") or "")
                ts_sec = int(float(rec.get("fundingTime", 0)) or 0)
                out.append({
                    "exchange": "whitebit",
                    "symbol": sym,
                    "income": _safe_float(rec.get("fundingAmount")),  # + cobro / - pago
                    "asset": "USDT",
                    "timestamp": ts_sec * 1000,  # ms
                    "funding_rate": _safe_float(rec.get("fundingRate", 0.0)),
                    "type": "FUNDING_FEE",
                })
        except Exception as e:
            print(f"❌ WhiteBIT funding error ({m}): {e}")
            continue

    return out

# =========================================================
# D) BALANCES — shape EXACTO para /api/balances
# =========================================================
def fetch_whitebit_all_balances() -> Dict[str, Any]:
    """
    Devuelve:
    {
      "exchange": "whitebit",
      "equity": float, "balance": float, "unrealized_pnl": float,
      "initial_margin": 0.0, "spot": float, "margin": 0.0, "futures": float
    }
    """
    spot_total = 0.0
    futures_total = 0.0
    unreal = 0.0
    initial_margin = 0.0

    # Spot/trade balance (en USDT equivalente si el endpoint lo proporciona)
    try:
        tr = _post("/api/v4/trade-account/balance", {}) or {}
        # No viene en USDT value; sumamos available+freeze por cada asset como "unidades" (~proxy).
        # Si prefieres, limita a USDT/USDC/USD.
        for ticker, vals in tr.items():
            available = _safe_float(vals.get("available"))
            freeze = _safe_float(vals.get("freeze"))
            # si es USDT/USDC, cuenta directo; otros podrías ignorarlos o mapearlos vía cotización (no disponible aquí)
            if ticker in ("USDT", "USDC", "USD"):
                spot_total += available + freeze
    except Exception as e:
        print(f"⚠️ WhiteBIT spot balance error: {e}")

    # Collateral/futuros
    try:
        coll = _post("/api/v4/collateral-account/balance", {}) or {}
        for asset, amt in coll.items():
            if asset in ("USDT", "USDC", "USD"):
                futures_total += _safe_float(amt)
        # PnL no realizado sumando 'pnl' de abiertas
        try:
            opens = _post("/api/v4/collateral-account/positions/open", {}) or []
            for r in opens:
                unreal += _safe_float(r.get("pnl"))
                # Estimación simple de initial margin: sumar 'margin' por posición si existe
                initial_margin += _safe_float(r.get("margin"))
        except Exception:
            pass
    except Exception as e:
        print(f"⚠️ WhiteBIT futures balance error: {e}")

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

# ===== util interno =====
def _signed_trade_amount(st: Dict[str, Any]) -> Optional[float]:
    od = st.get("orderDetail") or {}
    ta = od.get("tradeAmount")
    if ta is None:
        return None
    try:
        return float(ta)
    except Exception:
        return None

def _state_side(st: Dict[str, Any]) -> Optional[str]:
    s = (st.get("positionSide") or "").upper()
    if s in ("LONG", "SHORT"):
        return s.lower()
    return None

def _deduce_side_and_size(legs: List[Dict[str, Any]]) -> Tuple[str, float, List[float]]:
    """
    Lógica de deducción:
      1) Si CUALQUIER estado trae SHORT => 'short'
      2) Si no, si existe ALGÚN tradeAmount < 0 => 'short'; si >0 => 'long'
      3) Si no, si existe ALGÚN amount y un side explícito => usa ese side
      4) Fallback 'long'
    Size = máx |posición neta| reconstruyendo con tradeAmount (con signo).
           Si tradeAmount viene vacío en algún tramo, se usa 'amount' con el signo del side deducido.
    """
    # candidato por estados
    any_short = any((_state_side(st) == "short") for st in legs)
    any_long  = any((_state_side(st) == "long") for st in legs)

    # candidato por fills (signo)
    ta_vals = [v for v in (_signed_trade_amount(st) for st in legs) if v is not None]
    has_neg = any(v < 0 for v in ta_vals)
    has_pos = any(v > 0 for v in ta_vals)

    if any_short:
        ded_side = "short"
    elif any_long and not has_neg:
        ded_side = "long"
    elif has_neg and not has_pos:
        ded_side = "short"
    elif has_pos and not has_neg:
        ded_side = "long"
    else:
        # si hay mezcla o nada claro: último side explícito si existe
        explicit = [_state_side(st) for st in legs if _state_side(st)]
        ded_side = (explicit[-1] if explicit else "long") or "long"

    # reconstrucción neta
    net = 0.0
    path = []
    for st in legs:
        ta = _signed_trade_amount(st)
        if ta is None:
            # usa amount con el signo del side deducido (mejor que perder el tramo)
            amt = st.get("amount")
            try:
                amt = float(amt)
            except Exception:
                amt = 0.0
            ta = (-abs(amt)) if ded_side == "short" else abs(amt)
        net += ta
        path.append(net)
    max_abs_sz = max((abs(x) for x in path), default=0.0)

    return ded_side, max_abs_sz, path

def _last_realized_funding(legs: List[Dict[str, Any]]) -> float:
    # preferir acumulado del último estado
    last = legs[-1] if legs else {}
    rf = last.get("realizedFunding")
    if rf is not None:
        try:
            return float(rf)
        except Exception:
            pass
    # fallback: suma fundingFee por orderDetail
    tot = 0.0
    for st in legs:
        od = st.get("orderDetail") or {}
        try:
            tot += float(od.get("fundingFee") or 0.0)
        except Exception:
            pass
    return tot

def _sum_trade_fees(legs: List[Dict[str, Any]]) -> float:
    tot = 0.0
    for st in legs:
        od = st.get("orderDetail") or {}
        try:
            tot += float(od.get("tradeFee") or 0.0)
        except Exception:
            pass
    return -abs(tot)  # siempre negativo

def _sum_realized_pnl(legs: List[Dict[str, Any]]) -> Tuple[float, bool]:
    """
    Devuelve (realized_total, tiene_realized_por_leg?)
    Si todos vienen None, retornará (0.0, False)
    """
    tot = 0.0
    has_any = False
    for st in legs:
        od = st.get("orderDetail") or {}
        rp = od.get("realizedPnl")
        if rp is not None:
            try:
                tot += float(rp)
                has_any = True
            except Exception:
                pass
    return tot, has_any

def _entry_close_prices(legs: List[Dict[str, Any]]) -> Tuple[float, float]:
    first, last = legs[0], legs[-1]
    def _price_from(st: Dict[str, Any]) -> float:
        od = st.get("orderDetail") or {}
        p = od.get("price")
        if p is None:
            p = st.get("basePrice")
        try:
            return float(p)
        except Exception:
            return 0.0
    entry = _price_from(first)
    close = _price_from(last) or entry
    return entry, close

# ===== reemplaza la función save_whitebit_closed_positions por ésta =====
def save_whitebit_closed_positions(
    db_path: str = "portfolio.db",
    days: int = 30,
    debug: bool = False,
):
    try:
        from db_manager import save_closed_position
    except Exception as e:
        print(f"❌ db_manager.save_closed_position no disponible: {e}")
        return (0, 0)

    # descarga paginada
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

    # agrupa por positionId
    by_id: Dict[int, List[Dict[str, Any]]] = {}
    for st in all_rows:
        try:
            pid = int(st.get("positionId"))
        except Exception:
            continue
        by_id.setdefault(pid, []).append(st)
    for pid in by_id:
        by_id[pid].sort(key=lambda r: float(r.get("modifyDate") or 0.0))

    import sqlite3, os as _os
    if not _os.path.exists(db_path):
        print(f"❌ DB no encontrada: {db_path}")
        return (0, 0)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    saved = skipped = 0
    for pid, legs in by_id.items():
        try:
            market = (legs[-1].get("market") or "").upper()
            symbol = normalize_symbol(market)

            side, size, path = _deduce_side_and_size(legs)
            open_s  = int(float(legs[0].get("openDate") or 0))
            close_s = int(float(legs[-1].get("modifyDate") or 0))

            # evita duplicados
            cur.execute(
                "SELECT COUNT(*) FROM closed_positions WHERE exchange=? AND symbol=? AND close_time=?",
                ("whitebit", symbol, close_s),
            )
            if cur.fetchone()[0] > 0:
                skipped += 1
                continue

            entry_price, close_price = _entry_close_prices(legs)
            fee_total = _sum_trade_fees(legs)                       # negativo
            funding_total = _last_realized_funding(legs)            # acumulado
            realized_sum, has_realized = _sum_realized_pnl(legs)    # incluye fees si lo da el exchange

            # Si el exchange da realized por leg, lo usamos tal cual (suma)
            # y calculamos price_pnl derivado para coherencia:
            if has_realized:
                price_pnl = realized_sum - funding_total - fee_total
            else:
                # fallback: aproximación con entry/close y tamaño neto máximo (no perfecto en scale-in/out)
                signed_sz = -abs(size) if side == "short" else abs(size)
                price_pnl = (close_price - entry_price) * signed_sz
                realized_sum = price_pnl + funding_total + (-fee_total)  # si fees vienen 0, cuadra

            notional = abs(size) * entry_price

            if debug:
                print(f"\n📋 WB#{pid} {market} -> {symbol}")
                print(f"   sides_in_states = {[ _state_side(st) for st in legs ]}")
                print(f"   tradeAmounts    = {[ _signed_trade_amount(st) for st in legs ]}")
                print(f"   net_path        = {path}")
                print(f"   DEDUCED side={side} size={size}")
                print(f"   entry/close     = {entry_price} / {close_price}")
                print(f"   fee_total       = {fee_total}")
                print(f"   funding_total   = {funding_total}")
                print(f"   realized_sum    = {realized_sum} (has_realized={has_realized})")
                print(f"   price_pnl(der)  = {price_pnl}")
                print("   -> objeto a guardar:")

            save_closed_position({
                "exchange": "whitebit",
                "symbol": symbol,
                "side": side,
                "size": abs(size),
                "entry_price": float(entry_price),
                "close_price": float(close_price),
                "open_time": open_s,
                "close_time": close_s,
                "realized_pnl": float(realized_sum),
                "funding_total": float(funding_total),
                "fee_total": float(fee_total),
                "pnl": float(price_pnl),
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

# ===== DEBUG CRUDO: imprime tal cual lo que devuelve la API + trazas =====
def debug_dump_whitebit_history(market: Optional[str] = None, position_id: Optional[int] = None,
                                days: int = 7, raw: bool = True, limit: int = 100, offset: int = 0):
    end_ts = int(time.time())
    start_ts = end_ts - days * 86400 if days else None
    payload = {"limit": max(1, min(100, limit)), "offset": max(0, offset)}
    if start_ts: payload["startDate"] = int(start_ts)
    if end_ts:   payload["endDate"] = int(end_ts)
    if market:   payload["market"] = market
    if position_id is not None: payload["positionId"] = int(position_id)

    res = _post("/api/v4/collateral-account/positions/history", payload) or []
    if raw:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    # modo analítico por positionId
    by_id: Dict[int, List[Dict[str, Any]]] = {}
    for st in res:
        try:
            pid = int(st.get("positionId"))
        except Exception:
            continue
        by_id.setdefault(pid, []).append(st)
    for pid in by_id:
        by_id[pid].sort(key=lambda r: float(r.get("modifyDate") or 0.0))

    for pid, legs in by_id.items():
        side, size, path = _deduce_side_and_size(legs)
        entry_price, close_price = _entry_close_prices(legs)
        fee_total = _sum_trade_fees(legs)
        funding_total = _last_realized_funding(legs)
        realized_sum, has_realized = _sum_realized_pnl(legs)
        price_pnl = realized_sum - funding_total - fee_total if has_realized else (close_price - entry_price) * ((-abs(size)) if side=="short" else abs(size))

        print(f"\n[HIST] pid={pid} market={legs[-1].get('market')}")
        print(f" sides      : {[ _state_side(st) for st in legs ]}")
        print(f" tradeAmnts : {[ _signed_trade_amount(st) for st in legs ]}")
        print(f" net_path   : {path}  (size={size})")
        print(f" entry/close: {entry_price} -> {close_price}")
        print(f" fee_total  : {fee_total} | funding_total: {funding_total}")
        print(f" realized   : {realized_sum} (has_realized={has_realized})")
        print(f" price_pnl  : {price_pnl}")
