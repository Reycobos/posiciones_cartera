# adapters/xt.py
# XT adapter (Futures + Spot) usando el SDK pyxt (perp.py/spot.py locales si no hay pip).
from __future__ import annotations

import os
import json
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

# ============ SDK (pip o archivos locales) ============
try:
    from pyxt.perp import Perp  # pip install pyxt
except Exception:
    from perp import Perp       # /mnt/data/perp.py

try:
    from pyxt.spot import Spot  # pip install pyxt
except Exception:
    from spot import Spot       # /mnt/data/spot.py

# ============ Helpers de impresión del backend (no-op si no existen) ============
def _noop(*a, **k): pass
try:
    from portfoliov7 import (
        p_balance_equity, p_balance_fetching, p_balance_done,
        p_funding_fetching, p_funding_count,
        p_open_fetching, p_open_count,
        p_closed_sync_start, p_closed_sync_saved, p_closed_sync_done, p_closed_sync_none,
    )
except Exception:
    p_balance_equity = p_balance_fetching = p_balance_done = _noop
    p_funding_fetching = p_funding_count = _noop
    p_open_fetching = p_open_count = _noop
    p_closed_sync_start = p_closed_sync_saved = p_closed_sync_done = p_closed_sync_none = _noop

# ============ Normalización de símbolos ============
try:
    from symbols import normalize_symbol
except Exception:
    import re
    def normalize_symbol(sym: str) -> str:
        if not sym: return ""
        s = sym.upper()
        s = re.sub(r'^PERP_', '', s)
        s = re.sub(r'(_|-)?(USDT|USDC|USD|PERP)$', '', s)
        s = re.sub(r'[_-]+$', '', s)
        s = re.split(r'[_/-]', s)[0]
        return s

# ============ DB manager ============
try:
    from db_manager import save_closed_position
except Exception:
    # Fallback que imprime lo que guardaríamos si el módulo no está.
    def save_closed_position(position: dict):
        print("⚠️ db_manager.save_closed_position no disponible; payload:")
        print(json.dumps(position, indent=2, ensure_ascii=False))

# ============ Config/ENV ============
EXCHANGE = "xt"
XT_API_KEY = "96b1c999-b9df-42ed-9d4f-a4491aac7a1d"
XT_API_SECRET = "5f60a0a147d82db2117a45da129a6f4480488234"
XT_FAPI_HOST = os.getenv("XT_FAPI_HOST", "https://fapi.xt.com")
XT_SAPI_HOST = os.getenv("XT_SAPI_HOST", "https://sapi.xt.com")

DEFAULT_DAYS_TRADES = int(os.getenv("XT_DAYS_TRADES", "14"))

# ============ Utils ============
def to_float(x) -> float:
    try: return float(x)
    except Exception: return 0.0

def utc_now_ms() -> int:
    return int(time.time() * 1000)

def _unwrap_result(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "result" in obj: return obj["result"]
        if "data" in obj: return obj["data"]
        if "items" in obj: return obj["items"]
        if "list" in obj: return obj["list"]
    return obj

def _fee_for_trade(order_type: Optional[str], price: float, qty: float) -> float:
    """
    Calcula la comisión por trade según orderType:
      MARKET -> 0.0588%
      LIMIT  -> 0.038%
    Devuelve SIEMPRE negativa (costo).
    """
    t = (order_type or "").upper()
    if "MARKET" in t:
        rate = 0.000588
    elif "LIMIT" in t:
        rate = 0.000380
    else:
        # si no llega el tipo, asumimos LIMIT por defecto (según tu instrucción)
        rate = 0.000380
    return -abs(rate * price * qty)
# =========================================================
#                      BALANCES (COMBINADO)
#   Futures:  /future/user/v1/balance/list   -> walletBalance
#   Spot:     /v4/balances
# =========================================================
_spot_cli: Optional[Spot] = None
_perp_cli: Optional[Perp] = None

def _get_spot() -> Spot:
    global _spot_cli
    if _spot_cli is None:
        _spot_cli = Spot(host=XT_SAPI_HOST, access_key=XT_API_KEY, secret_key=XT_API_SECRET)
    return _spot_cli

def _get_perp() -> Perp:
    global _perp_cli
    if _perp_cli is None:
        _perp_cli = Perp(host=XT_FAPI_HOST, access_key=XT_API_KEY, secret_key=XT_API_SECRET)
    return _perp_cli

def fetch_xt_all_balances(db_path: str = "portfolio.db") -> Dict[str, Any]:
    """
    Devuelve estructura EXACTA para /api/balances combinando:
    - spot: /v4/balances
    - futures: /future/user/v1/balance/list (get_account_capital)
    """
    p_balance_fetching(EXCHANGE)

    # -------- Spot --------
    spot_equity = 0.0
    spot_avail = 0.0
    try:
        cli_s = _get_spot()
        res_s = cli_s.balances(currencies=None)  # GET /v4/balances
        assets = (res_s or {}).get("assets") if isinstance(res_s, dict) else res_s
        if isinstance(assets, list):
            for a in assets:
                if not isinstance(a, dict): continue
                avail = to_float(a.get("availableAmount") or 0.0)
                total = to_float(a.get("totalAmount") or 0.0)
                # No tenemos conversión a USDT aquí; usamos total.
                spot_equity += total
                spot_avail += avail
    except Exception:
        # Spot puede no estar habilitado → seguimos con futuros
        pass

    # -------- Futuros --------
    cli_f = _get_perp()
    code, success, error = cli_f.get_account_capital()   # GET /future/user/v1/balance/list
    if error or code != 200 or success is None:
        raise RuntimeError(f"XT futures balance error: {error or code}")
    res_f = _unwrap_result(success)

    futures_equity = 0.0
    futures_unreal = 0.0
    if isinstance(res_f, list):
        for it in res_f:
            if isinstance(it, dict):
                futures_equity += to_float(it.get("walletBalance") or 0.0)
                futures_unreal += to_float(it.get("notProfit") or it.get("unrealizedProfit") or 0.0)
    elif isinstance(res_f, dict):
        arr = res_f.get("items") or res_f.get("list") or []
        for it in arr or []:
            if isinstance(it, dict):
                futures_equity += to_float(it.get("walletBalance") or 0.0)
                futures_unreal += to_float(it.get("notProfit") or it.get("unrealizedProfit") or 0.0)

    total_equity = spot_equity + futures_equity
    out = {
        "exchange": EXCHANGE,
        "equity": float(total_equity),
        "balance": float(total_equity),
        "unrealized_pnl": float(futures_unreal),
        "initial_margin": 0.0,
        "spot": float(spot_equity),
        "margin": 0.0,
        "futures": float(futures_equity),
    }
    p_balance_equity(EXCHANGE, out["equity"])
    p_balance_done(EXCHANGE)
    return out

# =========================================================
#                    FUNDING FEES
#   GET /future/user/v1/balance/funding-rate-list
# =========================================================
def fetch_xt_funding_fees(limit: int = 50,
                          start_ms: Optional[int] = None,
                          end_ms: Optional[int] = None,
                          symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    p_funding_fetching(EXCHANGE)
    cli = _get_perp()

    if end_ms is None:
        end_ms = utc_now_ms()
    if start_ms is None:
        start_ms = end_ms - 14 * 24 * 60 * 60 * 1000

    out: List[Dict[str, Any]] = []
    next_id: Optional[int] = None
    direction = "NEXT"
    path = "/future/user" + "/v1/balance/funding-rate-list"

    while len(out) < limit:
        page_size = min(100, max(1, limit - len(out)))
        params: Dict[str, Any] = {
            "limit": page_size,
            "direction": direction,
            "startTime": int(start_ms),
            "endTime": int(end_ms),
        }
        if symbol:
            params["symbol"] = symbol

        header = cli._create_sign(XT_API_KEY, XT_API_SECRET,
                                  path=path, bodymod="application/x-www-form-urlencoded",
                                  params=params)
        header["Content-Type"] = "application/x-www-form-urlencoded"
        url = cli.host + path
        code, success, error = cli._fetch(method="GET", url=url, headers=header, params=params, timeout=cli.timeout)
        if error or code != 200 or success is None:
            raise RuntimeError(f"XT funding error: {error or code}")

        res = _unwrap_result(success)
        items = []
        if isinstance(res, dict):
            items = res.get("items") or res.get("list") or []
        if not isinstance(items, list):
            items = []

        for it in items:
            sym_raw = str(it.get("symbol") or "")
            base = normalize_symbol(sym_raw)
            income = to_float(it.get("cast") or 0.0)
            asset = (it.get("coin") or "USDT").upper()
            ts = int(it.get("createdTime") or 0)
            ts = ts if ts > 10**12 else ts * 1000
            out.append({
                "exchange": EXCHANGE,
                "symbol": base,
                "income": float(income),
                "asset": "USDT" if asset not in ("USDT", "USDC", "USD") else asset,
                "timestamp": ts,
                "funding_rate": 0.0,
                "type": "FUNDING_FEE",
            })
            if len(out) >= limit:
                break

        if isinstance(res, dict) and res.get("hasNext") and items:
            next_id = items[-1].get("id")
            direction = "NEXT"
        else:
            break

    p_funding_count(EXCHANGE, len(out))
    return out

# ========= WebSocket: user trades (pyxt) =========
try:
    from pyxt.websocket.perp import PerpWebsocketStreamClient
except Exception:
    # si tienes el árbol pyxt dentro del proyecto
    from pyxt.websocket.perp import PerpWebsocketStreamClient  # fallback igual; ajusta el import si hiciera falta

import threading
from queue import Queue, Empty

def _parse_ws_user_trade(msg: dict) -> Optional[dict]:
    """
    Intenta extraer un 'fill' de un mensaje user_trade del WS.
    Devuelve dict con shape para nuestro FIFO:
      {"price": float, "qty": float, "side": "BUY"/"SELL", "fee": float<=0,
       "symbol": str, "timestamp": int(ms), "orderType": str}
    Si no reconoce el mensaje, devuelve None.
    """
    if not isinstance(msg, dict):
        return None

    # Algunos WS envuelven en {"data": {...}} o listas
    data = msg.get("data", msg)
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return None

    # Extracción flexible de campos (nombres típicos)
    sym = data.get("symbol") or data.get("s") or ""
    side = (data.get("side") or data.get("S") or "").upper()
    otype = (data.get("orderType") or data.get("ot") or data.get("type") or "").upper()

    # precio y cantidad con claves alternativas
    price = data.get("price", data.get("p"))
    if price is None:
        price = data.get("avgPrice") or data.get("ap") or data.get("dealPrice")
    qty = data.get("executedQty", data.get("q"))
    if qty is None:
        qty = data.get("size") or data.get("dealQty")

    # timestamps (ms)
    ts = data.get("createdTime") or data.get("T") or data.get("transactTime") or data.get("t")
    if ts is None:
        ts = data.get("eventTime")

    try:
        price = float(price)
        qty = abs(float(qty))
        ts = int(ts)
        if ts < 10**12:  # s → ms
            ts *= 1000
    except Exception:
        return None

    # fee por tu tabla (MARKET 0.0588%, LIMIT 0.038%) → SIEMPRE negativa
    fee = _fee_for_trade(otype, price, qty)

    return {
        "price": float(price),
        "qty": float(qty),
        "side": "BUY" if side in ("BUY", "LONG", "BID") else "SELL",
        "fee": float(fee),
        "symbol": sym,
        "timestamp": ts,
        "orderType": otype,
    }


def stream_xt_user_trades_ws(listen_key: str = "",
                             seconds: int = 30,
                             symbol_filter: Optional[str] = None,
                             write_path: Optional[str] = None) -> List[dict]:
    """
    Conecta al WS auth de XT Perp y suscribe a 'user_trade'.
    - listen_key: si tu cuenta lo requiere; si no, déjalo "" (como en el ejemplo oficial).
    - seconds: cuánto tiempo mantener la suscripción (bloqueante).
    - symbol_filter: si lo pasas (ej. 'btc_usdt'), filtra por ese símbolo.
    - write_path: si lo pasas, guarda JSONL con cada trade crudo recibido.

    Devuelve lista de trades NORMALIZADOS para tu FIFO.
    También imprime DEBUG crudo y un resumen final.
    """
    q_raw: "Queue[dict]" = Queue()
    normalized: List[dict] = []
    raw_count = 0

    def on_message(_, message):
        nonlocal raw_count
        raw_count += 1
        try:
            # El SDK suele entregar ya como dict; si fuese string, intenta parsear:
            if isinstance(message, str):
                obj = json.loads(message)
            else:
                obj = message
        except Exception:
            obj = {"raw": message}

        # DEBUG crudo (muestra los primeros 5)
        if raw_count <= 5:
            print(f"[WS RAW #{raw_count}] {json.dumps(obj, ensure_ascii=False)}")
        q_raw.put(obj)

    client = PerpWebsocketStreamClient(on_message=on_message, is_auth=True)

    # heartbeat en hilo aparte
    hb_thread = threading.Thread(target=client.heartbeat, daemon=True)
    hb_thread.start()

    # suscripción
    print("→ Subscribiendo a user_trade...")
    client.user_trade(listen_key=listen_key, action=PerpWebsocketStreamClient.ACTION_SUBSCRIBE)

    t0 = time.time()
    f = None
    if write_path:
        f = open(write_path, "a", encoding="utf-8")

    try:
        while (time.time() - t0) < seconds:
            try:
                msg = q_raw.get(timeout=1.0)
            except Empty:
                continue

            # opcional: persistencia cruda JSONL
            if f:
                try:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")
                    f.flush()
                except Exception:
                    pass

            parsed = _parse_ws_user_trade(msg)
            if not parsed:
                continue

            if symbol_filter and (parsed["symbol"] or "").lower() != symbol_filter.lower():
                continue

            normalized.append(parsed)

    finally:
        # desuscribirse y cerrar
        print("→ Desubscribiendo y cerrando WS...")
        try:
            client.user_trade(listen_key=listen_key, action=PerpWebsocketStreamClient.ACTION_UNSUBSCRIBE)
        except Exception:
            pass
        try:
            client.stop()
        except Exception:
            pass
        if f:
            f.close()

    # DEBUG final
    print(f"\n=== WS user_trade resumen ===")
    print(f"mensajes crudos recibidos: {raw_count}")
    print(f"fills normalizados: {len(normalized)}")
    if normalized:
        print("muestra normalizada (hasta 10):")
        print(json.dumps(normalized[:10], indent=2, ensure_ascii=False))
    return normalized


def debug_ws_xt_trades(seconds: int = 20, symbol: Optional[str] = None, listen_key: str = "", out: Optional[str] = None):
    """
    Debug directo: conecta, escucha N segundos y muestra crudo + normalizado.
    """
    print("=== DEBUG WS user_trade (XT) ===")
    print(f"seconds={seconds} symbol={symbol or '*'} listen_key={'<empty>' if not listen_key else '<provided>'} out={out or '-'}")
    fills = stream_xt_user_trades_ws(listen_key=listen_key, seconds=seconds, symbol_filter=symbol, write_path=out)
    print("\n--- resumen ---")
    # suma fees y tamaños por símbolo
    by = defaultdict(lambda: {"qty": 0.0, "fee": 0.0, "trades": 0})
    for f in fills:
        s = normalize_symbol(f["symbol"])
        by[s]["qty"] += float(f["qty"])
        by[s]["fee"] += float(f["fee"])
        by[s]["trades"] += 1
    for s, agg in by.items():
        print(f"{s}: trades={agg['trades']} qty={agg['qty']:.8f} fee={agg['fee']:.8f}")
