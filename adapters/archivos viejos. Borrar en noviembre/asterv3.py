# ---------- adapters/asterv2.py ----------
import os, time, hmac, hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
import sqlite3
from db_manager import save_closed_position

import requests
from requests.exceptions import RequestException

from utils.symbols import normalize_symbol  # único import interno que pediste

# ========== Config y hosts ==========
# Host principal según la documentación
_DEFAULT_HOST = "https://fapi.asterdex.com"

# Si alguien setea ASTER_HOST mal, probamos una ronda de fallbacks razonables
_FALLBACK_HOSTS = [
    "https://fapi.asterdex.com",
    "https://fapi.aster.finance",
    "https://api.asterdex.com",
    "https://api.aster.finance",
]

ASTER_API_KEY = os.getenv("ASTER_API_KEY") or ""
ASTER_API_SECRET = os.getenv("ASTER_API_SECRET") or ""
# Si el usuario configuró ASTER_HOST, lo ponemos al frente de la lista; si no, usamos default
_user_host = (os.getenv("ASTER_HOST") or _DEFAULT_HOST).rstrip("/")
_HOSTS = [h.rstrip("/") for h in ([_user_host] + [x for x in _FALLBACK_HOSTS if x.rstrip("/") != _user_host])]

def _require_keys():
    if not ASTER_API_KEY or not ASTER_API_SECRET:
        raise RuntimeError("Faltan ASTER_API_KEY / ASTER_API_SECRET en el entorno.")

def _sign(params: Dict[str, Any]) -> Dict[str, Any]:
    # Respeta el orden de inserción para el query string
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    sig = hmac.new(ASTER_API_SECRET.encode("utf-8"), qs.encode("utf-8"), hashlib.sha256).hexdigest()
    params["signature"] = sig
    return params

def aster_signed_request(path: str, params: Optional[Dict[str, Any]] = None, timeout=30) -> Any:
    """
    GET firmado estilo MBX. Rota entre hosts hasta que uno responda.
    Lanza excepción con el resumen de errores si todos fallan.
    """
    _require_keys()
    base = {"timestamp": int(time.time() * 1000), "recvWindow": 5000}
    if params:
        base.update(params)
    signed = _sign(base)
    headers = {"X-MBX-APIKEY": ASTER_API_KEY, "User-Agent": "python-requests"}

    last_errs = []
    for host in _HOSTS:
        url = f"{host}{path}"
        try:
            r = requests.get(url, params=signed, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except RequestException as e:
            # Guardamos el error y probamos el siguiente host
            last_errs.append(f"{host}: {repr(e)}")
            continue

    raise ConnectionError("Todos los hosts fallaron para "
                          f"{path}. Intentados: {', '.join(_HOSTS)}. "
                          f"Errores: {' | '.join(last_errs[-3:])}")


def fetch_account_aster():
    """
    Aster account info (TotalEquity, Wallet, etc).
    Endpoint: GET //api/v3/account
    """
    try:
        data = aster_signed_request("/fapi/v4/account")
        
        if not data:
            return None

        # Extraer totales directamente
        total_wallet_balance = float(data.get("totalWalletBalance", 0))
        total_unrealized_pnl = float(data.get("totalUnrealizedProfit", 0))
        total_equity = float(data.get("totalMarginBalance", 0))  # equivale a wallet + PnL
        ASTER_EQUITY = total_equity
        
   
 
        # print(f"[DEBUG] Aster - Wallet Balance: {total_wallet_balance}, Equity: {total_equity}")
        
        
        
        return {
            "exchange": "aster",
            "equity": total_equity,
            "balance": total_wallet_balance,
            "unrealized_pnl": total_unrealized_pnl,
            "initial_margin": float(data.get("totalPositionInitialMargin", 0))
        }

    except Exception as e:
        print(f"[ERROR] Failed to fetch Aster account: {e}")
        return None
    
ASTER_EQUITY = 0.0  

def calc_liq_price(entry_price, position_amt, notional, leverage, wallet_balance, maint_rate=0.004):
    """
    Estima el precio de liquidación en cross margin.
    Usa equity (wallet + PnL no realizado) en lugar de solo wallet.
    """
    try:
        if position_amt == 0 or entry_price == 0 or notional == 0 or leverage == 0:
            return None

        maintenance_margin = notional * maint_rate

        if position_amt > 0:  # long
            liq = entry_price * (1 - 1/leverage + (wallet_balance - maintenance_margin) / notional)
        else:  # short
            liq = entry_price * (1 + 1/leverage - (wallet_balance - maintenance_margin) / notional)

        return round(liq, 6) if liq > 0 else None
    except Exception as e:
        print(f"[WARNING] Error calculating liquidation price: {e}")
        return None


def fetch_aster_open_positions():
    """
    Get current open positions from Aster.
    Endpoint: GET /api/v2/positionRisk
    """
    try:
        data = aster_signed_request("/fapi/v2/positionRisk")
        if not data:
            return []

        # ⚠️ Aquí deberías traer el wallet balance de la cuenta Aster
        # si ya lo calculas en otra función, puedes pasarlo como parámetro o almacenarlo global
        wallet_balance = 0.0  

        positions = []
        for position in data:
            try:
                position_amt = float(position.get("positionAmt", 0) or 0.0)
                if position_amt == 0:
                    continue

                unrealized_pnl = float(position.get("unRealizedProfit", 0) or 0.0)
                entry_price = float(position.get("entryPrice", 0) or 0.0)
                mark_price = float(position.get("markPrice", 0) or 0.0)
                notional = float(position.get("notional", 0) or 0.0)

                leverage = float(position.get("leverage", 0) or 0.0)
                if leverage == 0 and entry_price and position_amt:
                    leverage = abs(notional / (position_amt * entry_price)) if (position_amt * entry_price) else 10

                if position_amt > 0:
                    side = "long"
                elif position_amt < 0:
                    side = "short"
                else:
                    side = "flat"

                # 🔧 Liquidation Price
                liq_raw = float(position.get("liquidationPrice", 0) or 0.0)
                liquidation_price = liq_raw if liq_raw > 0 else calc_liq_price(
                    entry_price=entry_price,
                    position_amt=position_amt,
                    notional=notional,
                    leverage=leverage,
                    wallet_balance= ASTER_EQUITY,  
                    maint_rate=0.004  # valor por defecto, puedes ajustarlo
                )

                positions.append({
                    "exchange": "aster",
                    "symbol": position.get("symbol", ""),
                    "side": side,
                    "size": abs(position_amt),
                    "entry_price": entry_price,
                    "mark_price": mark_price,
                    "unrealized_pnl": unrealized_pnl,
                    "notional": notional,
                    "liquidation_price": liquidation_price,
                    "leverage": leverage
                })

                print("[DEBUG][Aster] Raw position:", position)
                print("[DEBUG][Aster] Calculated liq price:", liquidation_price)

            except Exception as e:
                print(f"[WARNING] Error processing Aster position: {e}")
                continue

        return positions

    except Exception as e:
        print(f"[ERROR] Failed to fetch Aster positions: {e}")
        return []

# ========== Funding del usuario ==========
def fetch_funding_aster(
    limit: int = 1000,
    startTime: Optional[int] = None,
    endTime: Optional[int] = None,
    symbol: Optional[str] = None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Funding del usuario (incomeType=FUNDING_FEE)
    Endpoint correcto: GET /fapi/v1/income
    """
    params: Dict[str, Any] = {
        "incomeType": "FUNDING_FEE",
        "limit": min(int(limit), 1000),
    }
    if startTime is not None: params["startTime"] = int(startTime)
    if endTime   is not None: params["endTime"]   = int(endTime)
    if symbol:                params["symbol"]    = symbol  # crudo, p.ej. BTCUSDT

    data = aster_signed_request("/fapi/v1/income", params=params) or []
    out: List[Dict[str, Any]] = []
    for it in data:
        try:
            out.append({
                "exchange": "aster",
                "symbol": it.get("symbol", ""),  # crudo (BTCUSDT)
                "income": float(it.get("income", 0) or 0.0),
                "asset": it.get("asset", "USDT") or "USDT",
                "timestamp": int(it.get("time") or it.get("timestamp") or it.get("tranTime") or 0),
                "funding_rate": None,
                "type": "FUNDING_FEE",
            })
        except Exception:
            continue
    if debug:
        print(f"[Aster] funding items: {len(out)}")
    return out

# ========== Reconstrucción de posiciones cerradas ==========
def fetch_closed_positions_aster(days: int = 30, limit: int = 1000, debug: bool = False) -> List[Dict[str, Any]]:
    """
    Reconstruye cerradas por símbolo:
      - detecta símbolos desde funding (rango days)
      - descarga trades con GET /fapi/v1/userTrades en ventanas de 7 días
      - cierra bloques cuando el neto de cantidad vuelve a cero
      - suma fees y funding dentro del rango del bloque
    """
    now_utc = datetime.utcnow()
    start_utc = now_utc - timedelta(days=days)
    start_ms = int(start_utc.timestamp() * 1000)
    end_ms   = int(now_utc.timestamp() * 1000)

    f_all = fetch_funding_aster(limit=1000, startTime=start_ms, endTime=end_ms, debug=debug) or []
    # Mapa base->raw para llamar a la API con el símbolo correcto
    base2raw: Dict[str, str] = {}
    for f in f_all:
        raw = f.get("symbol") or ""
        if not raw:
            continue
        base = normalize_symbol(raw)
        if base:
            base2raw.setdefault(base, raw)

    bases = sorted(base2raw.keys())
    if debug:
        print(f"[Aster] símbolos detectados por funding (base): {bases}")
    if not bases:
        if debug:
            print("[Aster] No se detectaron símbolos en el rango pedido.")
        return []

    # Index funding por símbolo base para sumar por rango
    f_by_base: Dict[str, List[Dict[str, Any]]] = {}
    for f in f_all:
        b = normalize_symbol(f.get("symbol", ""))
        f_by_base.setdefault(b, []).append(f)

    results: List[Dict[str, Any]] = []

    for base in bases:
        raw_sym = base2raw[base]   # BTCUSDT para la API
        # Descargar trades en chunks de 7 días
        all_trades: List[Dict[str, Any]] = []
        cursor = start_utc

        while cursor < now_utc:
            c0 = cursor
            c1 = min(cursor + timedelta(days=7), now_utc)
            params = {
                "symbol": raw_sym,
                "limit": int(limit),
                "startTime": int(c0.timestamp() * 1000),
                "endTime": int(c1.timestamp() * 1000),
            }
            try:
                page = aster_signed_request("/fapi/v1/userTrades", params=params)
                items = page if isinstance(page, list) else (page.get("data") or [])
                if items:
                    all_trades.extend(items)
                    if debug:
                        print(f"[Aster] {raw_sym}: +{len(items)} trades {c0:%Y-%m-%d} → {c1:%Y-%m-%d}")
            except Exception as e:
                if debug:
                    print(f"[Aster] userTrades error {raw_sym} @ {c0:%Y-%m-%d}: {e}")
            cursor = c1
            time.sleep(0.20)

        if not all_trades:
            if debug:
                print(f"[Aster] {raw_sym}: sin trades en {days} días.")
            continue

        # Normalizar y ordenar
        norm: List[Dict[str, Any]] = []
        for t in all_trades:
            try:
                side = (t.get("side") or "").upper()  # BUY/SELL
                qty = float(t.get("qty") or t.get("quantity") or 0.0)
                price = float(t.get("price") or 0.0)
                fee = abs(float(t.get("commission", 0) or 0.0))
                realized = float(t.get("realizedPnl") or 0.0)
                ts = int(t.get("time") or 0)
                signed = qty if side == "BUY" else -qty
                norm.append({"qty": qty, "price": price, "fee": fee, "realized": realized, "signed": signed, "ts": ts})
            except Exception:
                continue

        if not norm:
            continue
        norm.sort(key=lambda x: x["ts"])

        # Funding del símbolo base
        fnd = f_by_base.get(base, [])

        # Reconstrucción por bloques neto=0
        net = 0.0
        block: List[Dict[str, Any]] = []

        def _close_block(bl: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not bl:
                return None
            buys = [x for x in bl if x["signed"] > 0]
            sells = [x for x in bl if x["signed"] < 0]
            if not buys or not sells:
                return None

            buy_qty = sum(x["qty"] for x in buys)
            sell_qty = sum(x["qty"] for x in sells)
            if buy_qty <= 0 or sell_qty <= 0:
                return None

            avg_buy = sum(x["qty"] * x["price"] for x in buys) / buy_qty
            avg_sell = sum(x["qty"] * x["price"] for x in sells) / sell_qty

            is_short = bl[0]["signed"] < 0
            side = "short" if is_short else "long"
            entry_avg = avg_sell if is_short else avg_buy
            close_avg = avg_buy if is_short else avg_sell

            size = min(buy_qty, sell_qty)
            fees = sum(x["fee"] for x in bl)
            pnl_trades = sum(x["realized"] for x in bl)
            open_ts = min(x["ts"] for x in bl)
            close_ts = max(x["ts"] for x in bl)

            # funding en el rango
            f_sum = 0.0
            for r in fnd:
                ts_f = int(r.get("timestamp") or 0)
                if open_ts <= ts_f <= close_ts:
                    f_sum += float(r.get("income") or 0.0)

            total = pnl_trades - fees + f_sum
            return {
                "exchange": "aster",
                "symbol": base,  # ya normalizado
                "side": side,
                "size": size,
                "entry_price": entry_avg,
                "close_price": close_avg,
                "notional": entry_avg * size,
                "fees": fees,
                "funding_fee": f_sum,
                "realized_pnl": total,
                "open_date": datetime.fromtimestamp(open_ts / 1000).strftime("%Y-%m-%d %H:%M"),
                "close_date": datetime.fromtimestamp(close_ts / 1000).strftime("%Y-%m-%d %H:%M"),
            }

        for tr in norm:
            net += tr["signed"]
            block.append(tr)
            if abs(net) < 1e-9:
                rec = _close_block(block)
                if rec:
                    results.append(rec)
                    if debug:
                        print(f"  ✅ [{base}] {rec['side'].upper()} size={rec['size']:.6f} "
                              f"entry={rec['entry_price']:.6f} close={rec['close_price']:.6f} "
                              f"pnl={rec['realized_pnl']:.6f} fees={rec['fees']:.6f} funding={rec['funding_fee']:.6f}")
                block, net = [], 0.0

    if debug:
        print(f"[Aster] cerradas reconstruidas: {len(results)}")
    return results


def save_aster_closed_positions(db_path="portfolio.db", days=30, debug=False):
    """
    Guarda las posiciones cerradas de Aster en la base de datos SQLite.
    """
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return

    closed_positions = fetch_closed_positions_aster(debug=debug)
    if not closed_positions:
        print("⚠️ No closed positions returned from Aster.")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    saved = 0
    skipped = 0

    for pos in closed_positions:
        try:
            def to_ts(dt_str):
                try:
                    return int(datetime.fromisoformat(dt_str).timestamp())
                except Exception:
                    return None

            open_ts = to_ts(pos["open_date"])
            close_ts = to_ts(pos["close_date"])

            cur.execute("""
                SELECT COUNT(*) FROM closed_positions
                WHERE exchange = ? AND symbol = ? AND close_time = ?
            """, (pos["exchange"], pos["symbol"], close_ts))
            if cur.fetchone()[0]:
                skipped += 1
                continue

            save_closed_position({
                "exchange": pos["exchange"],
                "symbol": pos["symbol"],
                "side": pos["side"],
                "size": pos["size"],
                "entry_price": pos["entry_price"],
                "close_price": pos["close_price"],
                "open_time": open_ts,
                "close_time": close_ts,
                "realized_pnl": pos["realized_pnl"],
                "funding_total": pos.get("funding_fee", 0.0),
                "fee_total": pos.get("fees", 0.0),
                "notional": pos["notional"],
                "leverage": None,
                "liquidation_price": None
            })
            saved += 1

        except Exception as e:
            print(f"⚠️ Error guardando posición {pos.get('symbol')} (Aster): {e}")

    conn.close()
    print(f"✅ Guardadas {saved} posiciones cerradas de Aster (omitidas {skipped} duplicadas).")

# def save_aster_closed_positions(db_path="portfolio.db", days=30, debug=False):
#     """
#     Reconstruye las posiciones cerradas de Aster y las guarda en SQLite,
#     deduplicando por (exchange, symbol, close_time).

#     Requisitos:
#       - adapters.asterv2.fetch_closed_positions_aster (o asterv1 como fallback)
#       - db_manager.save_closed_position y db_manager.init_db
#     """
#     import os
#     import sqlite3
#     from datetime import datetime, timezone

#     # 1) Imports limpios sin circulares
#     try:
#         from adapters.asterv2 import fetch_closed_positions_aster
#     except Exception:
#         from adapters.asterv1 import fetch_closed_positions_aster  # fallback si aún usas v1

#     from db_manager import save_closed_position, init_db, DB_PATH  # escribe en DB_PATH propio  # noqa

#     # 2) Asegura esquema/table
#     init_db()  # crea 'closed_positions' con columnas nuevas si faltan

#     # 3) Aviso amistoso si estás leyendo una DB y escribiendo en otra
#     if os.path.abspath(DB_PATH) != os.path.abspath(db_path):
#         print(f"⚠️ Aviso: db_manager guarda en '{DB_PATH}', pero tú estás deduplicando sobre '{db_path}'. "
#               f"Mejor usa el mismo archivo para no perseguir fantasmas.")

#     # 4) Reconstruir cerradas
#     rows = fetch_closed_positions_aster(days=days, debug=debug) or []
#     if not rows:
#         print("⚠️ No closed positions returned from Aster.")
#         return 0

#     # 5) Deduplicación y guardado
#     if not os.path.exists(db_path):
#         # si no existe, la creamos abriendo conexión (init_db ya creó la tabla en DB_PATH)
#         open(db_path, "a").close()

#     conn = sqlite3.connect(db_path)
#     cur = conn.cursor()

#     saved = 0
#     skipped = 0

#     def _to_ts(dt_str: str):
#         try:
#             return int(datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
#                        .replace(tzinfo=timezone.utc).timestamp())
#         except Exception:
#             return None

#     for pos in rows:
#         try:
#             open_ts = _to_ts(pos.get("open_date"))
#             close_ts = _to_ts(pos.get("close_date"))

#             # dedup (exchange, symbol base, close_time)
#             cur.execute("""
#                 SELECT COUNT(*) FROM closed_positions
#                 WHERE exchange = ? AND symbol = ? AND close_time = ?
#             """, (pos["exchange"], pos["symbol"], close_ts))
#             if cur.fetchone()[0]:
#                 skipped += 1
#                 continue

#             # insertar usando la función central que calcula métricas y respeta el esquema
#             save_closed_position({
#                 "exchange": pos["exchange"],
#                 "symbol": pos["symbol"],                 # ya viene normalizado en el adapter
#                 "side": pos["side"],
#                 "size": float(pos["size"]),
#                 "entry_price": float(pos["entry_price"]),
#                 "close_price": float(pos["close_price"]),
#                 "open_time": open_ts,
#                 "close_time": close_ts,
#                 "realized_pnl": float(pos["realized_pnl"]),
#                 "funding_total": float(pos.get("funding_fee", 0.0)),
#                 "fee_total": float(pos.get("fees", 0.0)),
#                 "notional": float(pos["notional"]),
#                 "leverage": None,
#                 "liquidation_price": None
#             })
#             saved += 1

#         except Exception as e:
#             print(f"⚠️ Error guardando posición {pos.get('symbol')} (Aster): {e}")

#     conn.close()
#     print(f"✅ Guardadas {saved} posiciones cerradas de Aster (omitidas {skipped} duplicadas).")
#     return saved


# ========== Diagnóstico rápido ==========
def diagnose_aster_hosts():
    """
    Intenta /fapi/v1/time en todos los hosts para ver cuál responde.
    """
    base = {"timestamp": int(time.time() * 1000), "recvWindow": 5000}
    headers = {"X-MBX-APIKEY": ASTER_API_KEY or "dummy"}
    ok = []
    bad = []
    for host in _HOSTS:
        url = f"{host}/fapi/v1/time"
        try:
            r = requests.get(url, params=base, headers=headers, timeout=8)
            r.raise_for_status()
            ok.append(host)
        except Exception as e:
            bad.append((host, str(e)))
    return {"ok": ok, "bad": bad, "order": _HOSTS}


