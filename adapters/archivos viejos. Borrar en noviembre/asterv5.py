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
def _get_step_size(raw_sym: str) -> float:
    """
    Busca stepSize del símbolo en /fapi/v1/exchangeInfo.
    Si falla, devuelve 0.0 (luego usaremos un fallback 1e-6).
    """
    try:
        info = aster_signed_request("/fapi/v1/exchangeInfo")
        syms = info.get("symbols") or []
        for s in syms:
            if s.get("symbol") == raw_sym:
                for f in s.get("filters", []):
                    if f.get("filterType") == "LOT_SIZE":
                        return float(f.get("stepSize") or 0.0)
    except Exception:
        pass
    return 0.0


def _load_position_risk_map() -> dict:
    """
    Mapa raw_symbol -> positionAmt (float). Si falla, {}.
    """
    try:
        arr = aster_signed_request("/fapi/v2/positionRisk")
        out = {}
        for p in arr or []:
            rs = p.get("symbol", "")
            out[rs] = float(p.get("positionAmt") or 0.0)
        return out
    except Exception:
        return {}

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

def fetch_funding_aster_windowed(
    days: Optional[int] = None,
    since_ms: Optional[int] = None,
    until_ms: Optional[int] = None,
    symbol: Optional[str] = None,
    step_days: int = 7,
    per_req_limit: int = 1000,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Barrido por ventanas de 'step_days' hasta cubrir [since_ms, until_ms] o 'days' hacia atrás.
    Devuelve funding (incomeType=FUNDING_FEE) deduplicado por tranId si está disponible.
    """
    now_ms = int(time.time() * 1000)
    if until_ms is None:
        until_ms = now_ms
    if since_ms is None:
        if days is None:
            days = 7
        since_ms = max(0, until_ms - int(days) * 24 * 3600 * 1000)

    out: List[Dict[str, Any]] = []
    seen: set = set()
    step_ms = int(step_days) * 24 * 3600 * 1000

    start = int(since_ms)
    while start <= until_ms:
        end = min(start + step_ms - 1, until_ms)

        params: Dict[str, Any] = {
            "incomeType": "FUNDING_FEE",
            "startTime": start,
            "endTime": end,
            "limit": min(int(per_req_limit), 1000),
        }
        if symbol:
            params["symbol"] = symbol

        data = aster_signed_request("/fapi/v1/income", params=params) or []
        if debug:
            print(f"[Aster][{datetime.utcfromtimestamp(start/1000):%Y-%m-%d}→{datetime.utcfromtimestamp(end/1000):%Y-%m-%d}] "
                  f"items={len(data)}")

        for it in data:
            try:
                ts = int(it.get("time") or it.get("timestamp") or it.get("tranTime") or 0)
                sym_raw = it.get("symbol", "") or ""
                tran_id = str(it.get("tranId") or it.get("id") or f"{sym_raw}|{ts}|{it.get('income',0)}")
                if tran_id in seen:
                    continue
                seen.add(tran_id)

                out.append({
                    "exchange": "aster",
                    # Si quieres que siempre salga sin sufijo, descomenta la línea de normalize_symbol:
                    # "symbol": normalize_symbol(sym_raw),
                    "symbol": sym_raw,  # crudo (p.ej. BTCUSDT). Déjalo así si tu normalizador está en otra capa.
                    "income": float(it.get("income", 0) or 0.0),
                    "asset": it.get("asset", "USDT") or "USDT",
                    "timestamp": ts,
                    "funding_rate": None,
                    "type": "FUNDING_FEE",
                    "external_id": tran_id,
                })
            except Exception:
                continue

        start = end + 1
        time.sleep(0.05)  # mimos al RL

    # orden cronológico (por si acaso)
    out.sort(key=lambda x: x["timestamp"] or 0)
    if debug and out:
        first, last = out[0]["timestamp"], out[-1]["timestamp"]
        print(f"[Aster] total={len(out)}  range=({datetime.utcfromtimestamp(first/1000):%Y-%m-%d %H:%M} .. "
              f"{datetime.utcfromtimestamp(last/1000):%Y-%m-%d %H:%M})")
    return out


def pull_funding_aster(**kwargs) -> List[Dict[str, Any]]:
    """
    Wrapper tolerante para el sync:
      - acepta since (ms) y/o force_days
      - ignora kwargs desconocidos (evita TypeError)
    """
    now_ms = int(time.time() * 1000)
    force_days = kwargs.get("force_days", None)
    since = kwargs.get("since", None)
    symbol = kwargs.get("symbol", None)
    debug = kwargs.get("debug", False)

    if isinstance(force_days, int):
        return fetch_funding_aster_windowed(days=int(force_days), symbol=symbol, debug=debug)
    if since is not None:
        try:
            return fetch_funding_aster_windowed(since_ms=int(since), until_ms=now_ms, symbol=symbol, debug=debug)
        except Exception:
            # si 'since' viene raro (str 'None', etc.), cae al default 7 días
            return fetch_funding_aster_windowed(days=7, symbol=symbol, debug=debug)
    # default: últimos 7 días
    return fetch_funding_aster_windowed(days=7, symbol=symbol, debug=debug)

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
    posrisk_map = _load_position_risk_map()

    for base in bases:
        raw_sym = base2raw[base] 
        step = _get_step_size(raw_sym)
        eps_qty = max(1e-6, (step / 2.0) if step > 0 else 0.0)
        if debug:
            print(f"[Aster] {raw_sym}: step={step} → eps_qty={eps_qty}")# BTCUSDT para la API
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
            if abs(net) <= eps_qty:
                rec = _close_block(block)
                if rec:
                    results.append(rec)
                    if debug:
                        print(f"  ✅ [{base}] {rec['side'].upper()} size={rec['size']:.6f} "
                              f"entry={rec['entry_price']:.6f} close={rec['close_price']:.6f} "
                              f"pnl={rec['realized_pnl']:.6f} fees={rec['fees']:.6f} funding={rec['funding_fee']:.6f}")
                block, net = [], 0.0
                if block:
                    pos_amt_now = float(posrisk_map.get(raw_sym, 0.0))
                    if debug:
                        print(f"[Aster] {raw_sym}: flush final → net={net}, posAmtNow={pos_amt_now}")
                    if abs(net) <= eps_qty or abs(pos_amt_now) <= eps_qty:
                        rec = _close_block(block)
                        if rec:
                            results.append(rec)
                            if debug:
                                print(f"  ✅ [FLUSH {base}] {rec['side'].upper()} size={rec['size']:.6f} "
                                      f"entry={rec['entry_price']:.6f} close={rec['close_price']:.6f} "
                                      f"pnl={rec['realized_pnl']:.6f} fees={rec['fees']:.6f} funding={rec['funding_fee']:.6f}")
                    # Si no cumple, dejamos el bloque abierto (quedará para la próxima corrida)

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

# ─────────────────────────────────────────────────────────────────────────────
# 1) FETCH CLOSED: logs de ventana + fallback opcional por símbolos forzados
#    (añade el parámetro force_bases: list[str] | None = None)
# ─────────────────────────────────────────────────────────────────────────────
from typing import Optional, List, Dict, Any  # ya lo tendrás importado arriba
# ...

def fetch_closed_positions_aster(
    days: int = 30, 
    limit: int = 1000, 
    debug: bool = False,
    force_bases: Optional[List[str]] = None,   # ← NUEVO
) -> List[Dict[str, Any]]:
    now_utc = datetime.utcnow()
    start_utc = now_utc - timedelta(days=days)
    start_ms = int(start_utc.timestamp() * 1000)
    end_ms   = int(now_utc.timestamp() * 1000)

    if debug:
        print(f"[Aster] ventana cerradas: {start_utc:%Y-%m-%d %H:%M} → {now_utc:%Y-%m-%d %H:%M} UTC")

    f_all = fetch_funding_aster(limit=1000, startTime=start_ms, endTime=end_ms, debug=debug) or []
    if debug:
        print(f"[Aster] funding items en ventana: {len(f_all)}")

    # Mapa base->raw detectado desde funding
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

    # Fallback opcional: si no hubo funding (o quieres forzar), resuelve desde /exchangeInfo
    if (not bases) and force_bases:
        try:
            info = aster_signed_request("/fapi/v1/exchangeInfo")
            syms = info.get("symbols") or []
            fbases = [b.strip().upper() for b in force_bases if b.strip()]
            for b in fbases:
                for s in syms:
                    raw = s.get("symbol", "")
                    if normalize_symbol(raw) == b:
                        base2raw[b] = raw
                        break
            bases = sorted(base2raw.keys())
            if debug:
                print(f"[Aster] (fallback) símbolos desde exchangeInfo: {bases}")
        except Exception as e:
            if debug:
                print(f"[Aster] fallback exchangeInfo falló: {e}")

    if not bases:
        if debug:
            print("[Aster] No se detectaron símbolos en el rango pedido.")
        return []

    # ... (resto de la función SIN cambios) ...
    # Asegúrate de que más abajo sigues usando 'raw_sym = base2raw[base]'

# ─────────────────────────────────────────────────────────────────────────────
# 2) SAVE CLOSED: respeta el parámetro days (antes no lo pasabas)
# ─────────────────────────────────────────────────────────────────────────────
def save_aster_closed_positions(db_path="portfolio.db", days=30, debug=False):
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return

    # ANTES: closed_positions = fetch_closed_positions_aster(debug=debug)
    closed_positions = fetch_closed_positions_aster(days=days, debug=debug)  # ← FIX

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

# ─────────────────────────────────────────────────────────────────────────────
# 3) DEBUG AUTOEJECUTABLE con toggles por entorno
#    - ASTER_DEBUG_DAYS: ventana (días)
#    - ASTER_DEBUG_DB:   ruta DB (por defecto 'portfolio.db')
#    - ASTER_DEBUG_SYMBOLS: coma-separados para forzar símbolos (base, p.ej. 'OM,AVAX')
# ─────────────────────────────────────────────────────────────────────────────
ASTER_DEBUG_DAYS = int(os.getenv("ASTER_DEBUG_DAYS", "7"))
ASTER_DEBUG_DB = os.getenv("ASTER_DEBUG_DB", "portfolio.db")
ASTER_DEBUG_SYMBOLS = [s.strip().upper() for s in os.getenv("ASTER_DEBUG_SYMBOLS", "").split(",") if s.strip()]

def _print_tail(rows, n=10):
    for r in rows[-n:]:
        try:
            print(f"   · {r['symbol']} {r['side']} size={float(r['size']):.6f} "
                  f"entry={float(r['entry_price']):.6f} close={float(r['close_price']):.6f} "
                  f"open={r['open_date']} close={r['close_date']} "
                  f"pnl={float(r['realized_pnl']):.6f} fee={float(r.get('fees',0.0)):.6f} "
                  f"funding={float(r.get('funding_fee',0.0)):.6f}")
        except Exception:
            print("   ·", r)

if __name__ == "__main__":
    print("🧪 DEBUG Aster — closed/save")
    print(f"🔧 ASTER_DEBUG_DAYS={ASTER_DEBUG_DAYS} | ASTER_DEBUG_DB='{ASTER_DEBUG_DB}'")
    print(f"🔧 ASTER_DEBUG_SYMBOLS={ASTER_DEBUG_SYMBOLS or '—'}")

    try:
        diag = diagnose_aster_hosts()
        print(f"🌐 Hosts OK: {diag.get('ok')} | Orden de prueba: {diag.get('order')}")
    except Exception as e:
        print(f"⚠️ diagnose_aster_hosts falló: {e}")

    print("\n🔎 fetch_closed_positions_aster(...)")
    rows = fetch_closed_positions_aster(days=ASTER_DEBUG_DAYS, debug=True,
                                        force_bases=ASTER_DEBUG_SYMBOLS or None)
    print(f"📦 fetch_closed_positions_aster → {len(rows)} filas")
    _print_tail(rows, n=10)

    print("\n💾 save_aster_closed_positions(...)")
    save_aster_closed_positions(db_path=ASTER_DEBUG_DB, days=ASTER_DEBUG_DAYS, debug=True)



