from __future__ import annotations
import os, time, hmac, hashlib, json, math, sqlite3, re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, quote
import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timezone
import time, hmac, hashlib, requests
from urllib.parse import urlencode
from collections import defaultdict
from datetime import datetime, timezone

import requests

UA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
BINANCE_BASE_URL = "https://fapi.binance.com"






def binance_server_offset_ms():
    t0 = int(time.time() * 1000)
    r = requests.get(f"{BINANCE_BASE_URL}/fapi/v1/time", headers=UA_HEADERS, timeout=10)
    r.raise_for_status()
    server = r.json()["serverTime"]
    t1 = int(time.time() * 1000)
    return server - ((t0 + t1) // 2)

def binance_signed_get(path, params=None, off=0):
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        raise RuntimeError("Missing BINANCE_API_KEY/BINANCE_API_SECRET")
    params = dict(params or {})
    params["timestamp"] = int(time.time() * 1000) + off
    qs = urlencode(params, doseq=True)
    sig = hmac.new(BINANCE_API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY, **UA_HEADERS}
    url = f"{BINANCE_BASE_URL}{path}?{qs}&signature={sig}"
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()

def fetch_account_binance(off=0):
    """
    Binance account info adaptado para este proyecto (dict en vez de DataFrame).
    Combina futuros + spot.
    """
    try:
        # -------- FUTUROS --------
        path = "/fapi/v2/account"
        params = {
            "timestamp": int(time.time() * 1000) + off,
            "recvWindow": 5000
        }
        qs = urlencode(params, doseq=True)
        sig = hmac.new(BINANCE_API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
        url = f"{BINANCE_BASE_URL}{path}?{qs}&signature={sig}"
        headers = {"X-MBX-APIKEY": BINANCE_API_KEY, **UA_HEADERS}

        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        data_futures = r.json() or {}

        futures_wallet_balance = float(data_futures.get("totalWalletBalance", 0))
        futures_margin_balance = float(data_futures.get("totalMarginBalance", 0))
        futures_unrealized = float(data_futures.get("totalUnrealizedProfit", 0))

        # -------- SPOT --------
        url_spot = "https://api.binance.com/api/v3/account"
        params_spot = {
            "timestamp": int(time.time() * 1000) + off,
            "recvWindow": 5000
        }
        qs_spot = urlencode(params_spot, doseq=True)
        sig_spot = hmac.new(BINANCE_API_SECRET.encode(), qs_spot.encode(), hashlib.sha256).hexdigest()
        url_spot = f"{url_spot}?{qs_spot}&signature={sig_spot}"

        r_spot = requests.get(url_spot, headers=headers, timeout=30)
        r_spot.raise_for_status()
        data_spot = r_spot.json() or {}

        # Traer precios para valuar balances spot
        prices = {p["symbol"]: float(p["price"]) for p in requests.get("https://api.binance.com/api/v3/ticker/price").json()}
        total_spot_usdt = 0.0
        for bal in data_spot.get("balances", []):
            asset = bal["asset"]
            free = float(bal["free"])
            locked = float(bal["locked"])
            amount = free + locked
            if amount == 0:
                continue
            if asset == "USDT":
                total_spot_usdt += amount
            else:
                symbol = asset + "USDT"
                if symbol in prices:
                    total_spot_usdt += amount * prices[symbol]

        # -------- FORMATO NORMALIZADO --------
        return {
            "exchange": "binance",
            "equity": futures_margin_balance + total_spot_usdt,   # equity total (spot + futures)
            "balance": futures_wallet_balance + total_spot_usdt,  # wallet futures + spot
            "unrealized_pnl": futures_unrealized,
            "initial_margin": float(data_futures.get("totalPositionInitialMargin", 0))
        }

    except Exception as e:
        print(f"❌ Binance account error: {e}")
        return None
    
def fetch_positions_binance(off=0):
    """
    Posiciones abiertas en Binance Futures
    """
    try:
        data = binance_signed_get("/fapi/v2/positionRisk", {}, off)
        rows = [d for d in data if float(d.get("positionAmt", "0")) != 0.0]
        positions = []
        for pos in rows:
            qty = float(pos["positionAmt"])
            side = "long" if qty > 0 else "short"
            positions.append({
                "exchange": "binance",
                "symbol": pos["symbol"],
                "side": side,
                "size": abs(qty),
                "entry_price": float(pos["entryPrice"]),
                "mark_price": float(pos["markPrice"]),
                "unrealized_pnl": float(pos["unRealizedProfit"]),
                "notional": float(pos["notional"]),
                "liquidation_price": float(pos["liquidationPrice"]),
                "leverage": float(pos["leverage"]),
            })
        return positions
    except Exception as e:
        print(f"❌ Binance positions error: {e}")
        return []
    


# === Helpers internos ===

def _iso_ms(ms):
    try:
        return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ms)

def _bn_user_trades_range(symbol, start_ms, end_ms, off=0, debug=False, position_side=None):
    """
    Descarga todos los userTrades de [start_ms, end_ms] para un símbolo con paginación por fromId.
    Filtra por position_side si viene (LONG/SHORT) cuando el trade lo trae.
    """
    all_trades = []
    last_id = None
    while True:
        params = {
            "symbol": symbol,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
        if last_id is not None:
            params["fromId"] = last_id + 1
        try:
            page = binance_signed_get("/fapi/v1/userTrades", params, off)
        except Exception as e:
            if debug:
                print(f"[WARN] userTrades fallo {symbol}: {e}")
            break
        if not page:
            break
        # filtra por positionSide si aplica
        if position_side:
            page = [t for t in page if (t.get("positionSide") or "").upper() == position_side.upper()]
        all_trades.extend(page)
        if len(page) < 1000:
            break
        last_id = int(page[-1]["id"])
    # ordenados por tiempo
    all_trades.sort(key=lambda x: x.get("time", 0))
    return all_trades

def _bn_income_range(symbol, start_ms, end_ms, types=("COMMISSION","FUNDING_FEE"), off=0, debug=False):
    """
    Descarga income en el rango para un símbolo. Paginación por 'page'.
    """
    incomes = []
    page = 1
    while True:
        params = {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000, "page": page}
        try:
            arr = binance_signed_get("/fapi/v1/income", params, off)
        except Exception as e:
            if debug:
                print(f"[WARN] income fallo {symbol}: {e}")
            break
        if not arr:
            break
        for i in arr:
            t = i.get("incomeType")
            if t in types:
                incomes.append(i)
        if len(arr) < 1000:
            break
        page += 1
    # filtra por símbolo explícito (defensivo)
    incomes = [i for i in incomes if (i.get("symbol") or "") == symbol]
    return incomes

def _find_open_block_start(trades, current_qty, debug=False):
    """
    Dado el listado cronológico de trades y el qty actual (signed),
    devuelve el índice del inicio del bloque abierto y su timestamp.
    Si no cuadra (histórico insuficiente), retorna (None, None).
    """
    if not trades:
        return None, None

    net = 0.0
    last_flat_idx = -1
    # recorre cronológico
    for idx, t in enumerate(trades):
        q = float(t["qty"])
        side = t["side"].upper()  # BUY / SELL
        qty_signed = q if side == "BUY" else -q
        net += qty_signed
        if abs(net) < 1e-12:
            last_flat_idx = idx

    # net al final debería ≈ current_qty
    if abs(net - float(current_qty)) > 1e-8:
        if debug:
            print(f"⚠️ net({net}) != current_qty({current_qty}) → histórico insuficiente")
        return None, None

    open_idx = last_flat_idx + 1  # siguiente a la última vez que net volvió a 0
    if open_idx >= len(trades):
        return None, None
    open_time = int(trades[open_idx]["time"])
    return open_idx, open_time

def _sum_fees_and_funding_for_block(symbol, trades_block, open_time_ms, now_ms, incomes, debug=False):
    """
    Suma fees y funding del bloque abierto:
    - Fees: PRIMARIO desde userTrades.commission (por trade) → negativo
            FALLBACK desde income.COMMISSION (por tradeId y rango +/-60s)
    - Funding: desde income.FUNDING_FEE (por rango +/-60s)
    Devuelve (fee_final, fnd_sum, dbg) donde dbg trae métricas de debug.
    """
    # --- 1) Fees desde TRADES (fiable) ---
    fees_by_asset = {}
    for t in trades_block:
        if t.get("commission") is None:
            continue
        asset = (t.get("commissionAsset") or "USDT").upper()
        try:
            val = float(t.get("commission", 0.0))
        except Exception:
            val = 0.0
        fees_by_asset[asset] = fees_by_asset.get(asset, 0.0) + val

    # Binance reporta commission como cantidad POSITIVA del asset cobrado → conviértelo a NEGATIVO en tu PnL
    fee_from_trades = -sum(fees_by_asset.values()) if fees_by_asset else 0.0

    # --- 2) Fees desde INCOME (fallback) ---
    t0 = open_time_ms - 60_000    # -60s de margen
    t1 = now_ms + 60_000          # +60s de margen
    commissions_income = [i for i in incomes if i.get("incomeType") == "COMMISSION"]

    block_ids = {str(int(t["id"])) for t in trades_block if "id" in t}
    comm_by_tid  = [i for i in commissions_income if (i.get("tradeId") is not None and str(i["tradeId"]) in block_ids)]
    comm_by_time = [i for i in commissions_income if t0 <= int(i.get("time", 0)) <= t1]

    fee_from_income_tid  = sum(float(i.get("income", 0.0)) for i in comm_by_tid)
    fee_from_income_time = sum(float(i.get("income", 0.0)) for i in comm_by_time)

    # Selección final de fees: trades > income por tradeId > income por rango
    if abs(fee_from_trades) > 1e-12:
        fee_final = fee_from_trades
        fee_source = "trades"
    elif comm_by_tid:
        fee_final = fee_from_income_tid
        fee_source = "income.tradeId"
    else:
        fee_final = fee_from_income_time
        fee_source = "income.timerange"

    # --- 3) Funding desde INCOME (rango con margen) ---
    fundings = [i for i in incomes if i.get("incomeType") == "FUNDING_FEE" and t0 <= int(i.get("time", 0)) <= t1]
    fnd_sum = sum(float(i.get("income", 0.0)) for i in fundings)

    # --- 4) Debug detallado ---
    dbg = {
        "fee_source": fee_source,
        "fee_from_trades": fee_from_trades,
        "fees_by_asset": fees_by_asset,    # ej. {"USDT": 3.7123}
        "fee_income_tid": fee_from_income_tid,
        "fee_income_time": fee_from_income_time,
        "comm_tid_count": len(comm_by_tid),
        "comm_time_count": len(comm_by_time),
        "funding_count": len(fundings),
        "time_window": (t0, t1),
    }

    if debug:
        print(f"      FEES:")
        if fees_by_asset:
            for a, v in fees_by_asset.items():
                print(f"        from trades: {a}={v:.6f} → as pnl: {-abs(v):.6f}")
        else:
            print(f"        from trades: (none)")
        print(f"        from income[tradeId]: {fee_from_income_tid:.6f} (items={len(comm_by_tid)})")
        print(f"        from income[time ±60s]: {fee_from_income_time:.6f} (items={len(comm_by_time)})")
        print(f"        >> fee_final = {fee_final:.6f} (source={fee_source})")

        print(f"      FUNDING:")
        print(f"        funding_sum: {fnd_sum:.6f} (items={len(fundings)})")

    return fee_final, fnd_sum, dbg



# === Función principal: posiciones abiertas enriquecidas ===

def fetch_positions_binance_enriched(days=60, off=0, debug=False):
    """
    Devuelve posiciones abiertas de Binance con 'fees', 'funding_fee' y 'realized_pnl' reconstruidos.
    NO modifica tus funciones de funding ni de closed; usa su propia lógica aquí.
    - days: cuántos días máximo mirar hacia atrás para reconstruir el bloque abierto actual.
    """
    try:
        # 1) posiciones abiertas básicas (tal cual ya tienes)
        base = binance_signed_get("/fapi/v2/positionRisk", {}, off)
        rows = [d for d in base if float(d.get("positionAmt", "0")) != 0.0]
        if debug:
            print(f"[Binance] posiciones abiertas: {len(rows)}")

        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        start_limit_ms = now_ms - int(days*24*60*60*1000)

        enriched = []

        for pos in rows:
            sym = pos["symbol"]
            qty = float(pos["positionAmt"])
            side = "long" if qty > 0 else "short"
            pos_side_field = (pos.get("positionSide") or "").upper() or None  # LONG/SHORT en hedge
            entry_price = float(pos["entryPrice"])
            mark_price  = float(pos["markPrice"])
            unrealized  = float(pos["unRealizedProfit"])
            notional    = float(pos.get("notional", entry_price * abs(qty)))
            liq_price   = float(pos["liquidationPrice"])
            lev         = float(pos["leverage"])
            update_time = int(pos.get("updateTime") or 0)  # fallback si no encontramos el inicio

            if debug:
                print(f"\n[Symbol] {sym} side={side} qty={abs(qty)} entry={entry_price} mark={mark_price}")

            # 2) trae trades suficientes para hallar el inicio del bloque abierto actual
            #    (ventanas de 7 días desde start_limit_ms → now)
            trades = []
            t0 = start_limit_ms
            while t0 < now_ms:
                t1 = min(t0 + 7*24*60*60*1000, now_ms)
                tpage = _bn_user_trades_range(sym, t0, t1, off=off, debug=False, position_side=pos_side_field)
                trades.extend(tpage)
                t0 = t1
                # micro-optimización: si ya tenemos muchísimos trades y encontramos luego el inicio, salimos

            # 3) localizar inicio del bloque actual
            open_idx, open_time_ms = _find_open_block_start(trades, current_qty=qty, debug=debug)

            if open_time_ms is None:
                # histórico insuficiente → usar updateTime como aproximación para incomes
                if debug:
                    print(f"   ⚠️ No se pudo determinar el inicio exacto. Fallback a updateTime={_iso_ms(update_time)}")
                open_time_ms = update_time if update_time else start_limit_ms
                # para fees por tradeId necesitamos trades del "bloque"; si no sabemos el bloque exacto,
                # usamos los trades de [open_time_ms..now] como aproximación.
                trades_block = [t for t in trades if int(t.get("time", 0)) >= open_time_ms]
            else:
                trades_block = trades[open_idx:]

            # 4) ingresos en rango y sumas
            incomes = _bn_income_range(sym, open_time_ms, now_ms, types=("COMMISSION","FUNDING_FEE"), off=off, debug=debug)
            fee_sum, fnd_sum, dbg = _sum_fees_and_funding_for_block(sym, trades_block, open_time_ms, now_ms, incomes, debug=debug)
            
            realized_total = fee_sum + fnd_sum  # solo OPEN: realized = fees + funding
            
            if debug:
                print(f"   open_time={_iso_ms(open_time_ms)}  trades_in_block={len(trades_block)}")
                print(f"   fees={fee_sum:.6f} (src={dbg['fee_source']}) | funding={fnd_sum:.6f} | realized(open)={realized_total:.6f}")
                # sanity check
                recomposed = fee_sum + fnd_sum
                if abs(recomposed - realized_total) > 1e-9:
                    print(f"   ⚠️ mismatch realized: {realized_total:.6f} vs fees+funding {recomposed:.6f}")
            enriched.append({
                "exchange": "binance",
                "symbol": sym,
                "side": side,
                "size": abs(qty),
                "entry_price": entry_price,
                "mark_price": mark_price,
                "unrealized_pnl": unrealized,
                "notional": notional,
                "liquidation_price": liq_price,
                "leverage": lev,
                # añadidos:
                "fee": fee_sum,                 # suele ser negativo
                "funding_fee": fnd_sum,          # + cobro / − pago
                "realized_pnl": realized_total,  # = fees + funding
                "open_time": int(open_time_ms/1000),
                "update_time": int((update_time or now_ms)/1000),
            })

        return enriched

    except Exception as e:
        print(f"❌ Binance positions enriched error: {e}")
        return []

    
def fetch_funding_binance(limit=100, off=0):
    """
    Funding payments en Binance
    """
    try:
        data = binance_signed_get("/fapi/v1/income",
                                  {"incomeType": "FUNDING_FEE", "limit": limit},
                                  off)
        funding = []
        for f in data:
            funding.append({
                "exchange": "binance",
                "symbol": f.get("symbol", ""),
                "income": float(f.get("income", 0)),
                "asset": f.get("asset", "USDT"),
                "timestamp": f.get("time"),
                "funding_rate": None,
                "type": "FUNDING_FEE",
                "external_id": f.get("tranId"),    # 👈 incluir si viene
            })
        return funding
    except Exception as e:
        print(f"❌ Binance funding error: {e}")
        return []
    
# ========= Funding Binance: ventanas 7d =========

def fetch_funding_binance_windowed(
    days: Optional[int] = None,
    since_ms: Optional[int] = None,
    until_ms: Optional[int] = None,
    symbol: Optional[str] = None,
    step_days: int = 7,
    per_req_limit: int = 1000,
    off: int = 0,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Barrido por ventanas de 'step_days' usando /fapi/v1/income para traer FUNDING_FEE.
    Dedup por tranId (si existe).
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

        try:
            data = binance_signed_get("/fapi/v1/income", params, off=off)
        except Exception as e:
            if debug:
                print(f"[Binance] income window {start}→{end} error: {e}")
            data = []

        if debug:
            print(f"[Binance] window {datetime.utcfromtimestamp(start/1000):%Y-%m-%d}→"
                  f"{datetime.utcfromtimestamp(end/1000):%Y-%m-%d} items={len(data)}")

        for it in data or []:
            if it.get("incomeType") != "FUNDING_FEE":
                continue
            ts = int(it.get("time") or 0)
            sym_raw = it.get("symbol", "") or ""
            tran_id = str(it.get("tranId") or f"{sym_raw}|{ts}|{it.get('income',0)}")
            if tran_id in seen:
                continue
            seen.add(tran_id)

            out.append({
                "exchange": "binance",
                "symbol": sym_raw,  # si quieres sin sufijo, normalízalo en la capa de UI/global
                "income": float(it.get("income", 0) or 0.0),
                "asset": it.get("asset", "USDT") or "USDT",
                "timestamp": ts,
                "funding_rate": None,
                "type": "FUNDING_FEE",
                "external_id": tran_id,
            })

        start = end + 1
        time.sleep(0.05)  # mimo al RL

    out.sort(key=lambda x: x["timestamp"] or 0)
    if debug and out:
        first, last = out[0]["timestamp"], out[-1]["timestamp"]
        print(f"[Binance] total={len(out)}  range=({datetime.utcfromtimestamp(first/1000):%Y-%m-%d %H:%M} .. "
              f"{datetime.utcfromtimestamp(last/1000):%Y-%m-%d %H:%M})")
    return out


def pull_funding_binance(**kwargs) -> List[Dict[str, Any]]:
    """
    Wrapper tolerante para sync_all_funding:
      - acepta since (ms), force_days (int), symbol, debug, off
      - ignora kwargs desconocidos (evita TypeError)
    """
    now_ms  = int(time.time() * 1000)
    force_d = kwargs.get("force_days", None)
    since   = kwargs.get("since", None)
    symbol  = kwargs.get("symbol", None)
    debug   = kwargs.get("debug", False)
    off     = int(kwargs.get("off", 0) or 0)

    if isinstance(force_d, int):
        return fetch_funding_binance_windowed(days=int(force_d), symbol=symbol, off=off, debug=debug)
    if since is not None:
        try:
            return fetch_funding_binance_windowed(since_ms=int(since), until_ms=now_ms, symbol=symbol, off=off, debug=debug)
        except Exception:
            return fetch_funding_binance_windowed(days=7, symbol=symbol, off=off, debug=debug)
    # default: últimos 7 días
    return fetch_funding_binance_windowed(days=7, symbol=symbol, off=off, debug=debug)




def fmt_time(ms):
    return datetime.fromtimestamp(ms/1000).strftime("%Y-%m-%d %H:%M")
# Guardar lista de símbolos válidos en Binance Futures
# ====== DEBUG BINANCE TRADES: HELPERS ======


def fetch_closed_positions_binance(days=30, off=0):
    """
    Reconstruye posiciones cerradas de Binance usando userTrades + income.
    Cada vez que net_qty vuelve a 0 → nueva posición cerrada.
    """
    try:
        now = int(time.time() * 1000)
        start_time = now - days*24*60*60*1000

        # 1) Income global
        income = binance_signed_get("/fapi/v1/income", {
            "limit": 1000,
            "startTime": start_time,
            "endTime": now
        }, off)

        income_by_symbol = defaultdict(list)
        for inc in income:
            if inc["incomeType"] in ("REALIZED_PNL", "COMMISSION", "FUNDING_FEE"):
                income_by_symbol[inc["symbol"]].append(inc)

        # 2) Determinar símbolos activos desde income
        symbols = [s for s in income_by_symbol.keys() if s]

        results = []

        for sym in symbols:
            # 3) Traer trades del símbolo
            try:
                trades = binance_signed_get("/fapi/v1/userTrades", {
                    "symbol": sym,
                    "limit": 1000,
                    "startTime": start_time,
                    "endTime": now
                }, off)
            except Exception as e:
                print(f"❌ userTrades error {sym}: {e}")
                continue

            if not trades:
                continue

            trades_sorted = sorted(trades, key=lambda x: x["time"])

            net_qty = 0.0
            block = []
            for t in trades_sorted:
                qty = float(t["qty"]) if t["side"] == "BUY" else -float(t["qty"])
                net_qty += qty
                block.append(t)

                # posición cerrada
                if abs(net_qty) < 1e-8:
                    open_date = fmt_time(block[0]["time"])
                    close_date = fmt_time(block[-1]["time"])

                    buys = [b for b in block if b["side"] == "BUY"]
                    sells = [s for s in block if s["side"] == "SELL"]

                    def avg_price(lst):
                        total_qty = sum(float(x["qty"]) for x in lst)
                        notional = sum(float(x["qty"]) * float(x["price"]) for x in lst)
                        return notional / total_qty if total_qty else 0.0

                    entry_price = avg_price(buys)
                    close_price = avg_price(sells)
                    size = sum(float(b["qty"]) for b in buys)

                    # 4) PnL y fees dentro de ese rango
                    start_ts, end_ts = block[0]["time"], block[-1]["time"]
                    incs = [i for i in income_by_symbol[sym] if start_ts <= i["time"] <= end_ts]
                    realized_pnl = sum(float(i["income"]) for i in incs if i["incomeType"] == "REALIZED_PNL")
                    fees = sum(float(i["income"]) for i in incs if i["incomeType"] == "COMMISSION")
                    funding = sum(float(i["income"]) for i in incs if i["incomeType"] == "FUNDING_FEE")

                    results.append({
                        "exchange": "binance",
                        "symbol": sym,
                        "side": "closed",
                        "size": size,
                        "entry_price": entry_price,
                        "close_price": close_price,
                        "notional": entry_price * size,
                        "fees": fees,
                        "funding_fee": funding,
                        "pnl": realized_pnl,
                        "realized_pnl": realized_pnl,
                        "open_date": open_date,
                        "close_date": close_date,
                    })

                    # reset
                    block = []

        print(f"✅ Binance closed positions reconstruidas: {len(results)} en {days} días")
        return results

    except Exception as e:
        print(f"❌ Binance closed positions error: {e}")
        return []
    


def fetch_closed_positions_binance(days=30, off=0, debug=False):
    """
    Reconstruye posiciones cerradas de Binance Futures en los últimos `days`.
    - Ventanas de 7 días (limitación API) con estado 'carry-over' por símbolo.
    - side correcto (long/short) por neto del bloque.
    - entry/close correctos (para short se invierte).
    - income asociado por tradeId cuando esté disponible; si no, por rango de tiempo.
    - realized_pnl = SOLO precio (tu UI ya muestra fees y funding aparte).
    """

    def _iso(ms):
        try:
            return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(ms)



    def signed_get(path, params=None):
        if not BINANCE_API_KEY or not BINANCE_API_SECRET:
            raise RuntimeError("Missing BINANCE_API_KEY/BINANCE_API_SECRET")       
        params = dict(params or {})
        params["timestamp"] = int(time.time() * 1000) + int(off)
        qs = urlencode(params, doseq=True)
        sig = hmac.new(BINANCE_API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
        url = f"{BINANCE_BASE_URL}{path}?{qs}&signature={sig}"
        headers = {"X-MBX-APIKEY": BINANCE_API_KEY, **UA_HEADERS}
        if debug:
            print(f"[GET] {path} {params}")
        r = requests.get(url, headers=headers, timeout=25)
        r.raise_for_status()
        return r.json()

    try:
        now = int(time.time() * 1000)
        start_time = now - days * 24 * 60 * 60 * 1000

        # 1) INCOME de todo el rango, paginado por 'page'
        income_by_symbol = defaultdict(list)
        page = 1
        while True:
            inc = signed_get("/fapi/v1/income", {
                "limit": 1000, "startTime": start_time, "endTime": now, "page": page
            })
            if not inc:
                break
            for i in inc:
                t = i.get("incomeType")
                sym = i.get("symbol") or ""
                if t in ("REALIZED_PNL", "COMMISSION", "FUNDING_FEE") and sym:
                    income_by_symbol[sym].append(i)
            if len(inc) < 1000:
                break
            page += 1

        if debug:
            print("[Income] resumen:")
            for sym, arr in income_by_symbol.items():
                s_pnl = sum(float(x["income"]) for x in arr if x["incomeType"]=="REALIZED_PNL")
                s_fee = sum(float(x["income"]) for x in arr if x["incomeType"]=="COMMISSION")
                s_fnd = sum(float(x["income"]) for x in arr if x["incomeType"]=="FUNDING_FEE")
                print(f"  {sym}: pnl={s_pnl:.2f} fee={s_fee:.2f} fnd={s_fnd:.2f} items={len(arr)}")

        # 2) exchangeInfo (opcional)
        try:
            exi = signed_get("/fapi/v1/exchangeInfo")
            valid_symbols = {s["symbol"] for s in exi.get("symbols", [])}
        except Exception:
            valid_symbols = set(income_by_symbol.keys())

        results = []

        # 3) Estado carry-over por símbolo (bloque abierto y net_qty acumulado)
        carry_block_by_sym = {}   # sym -> list of trades
        carry_net_by_sym   = {}   # sym -> float

        # 4) Procesar símbolos
        for sym in list(income_by_symbol.keys()):
            if sym not in valid_symbols and debug:
                print(f"[SKIP] {sym} no en exchangeInfo")
            if debug:
                print(f"\n[Symbol] {sym} ventanas de 7d {_iso(start_time)} → {_iso(now)}")

            end_time = start_time
            # inicializar carry si existe
            block = carry_block_by_sym.get(sym, [])
            net_qty = carry_net_by_sym.get(sym, 0.0)

            while end_time < now:
                chunk_start = end_time
                chunk_end = min(chunk_start + 7*24*60*60*1000, now)
                end_time = chunk_end
                if debug:
                    print(f"  [Window] {_iso(chunk_start)} → {_iso(chunk_end)} (carry net={net_qty:.6f}, block={len(block)})")

                # Paginación userTrades por fromId
                trades_all = []
                last_id = None
                while True:
                    params = {
                        "symbol": sym,
                        "startTime": chunk_start,
                        "endTime": chunk_end,
                        "limit": 1000,
                    }
                    if last_id is not None:
                        params["fromId"] = last_id + 1
                    try:
                        tpage = signed_get("/fapi/v1/userTrades", params)
                    except Exception as e:
                        if debug:
                            print(f"    [WARN] userTrades fallo: {e}")
                        break
                    if not tpage:
                        break
                    trades_all.extend(tpage)
                    if len(tpage) < 1000:
                        break
                    last_id = int(tpage[-1]["id"])

                if debug:
                    print(f"    trades nuevos: {len(trades_all)}")

                if trades_all:
                    trades_all.sort(key=lambda x: x["time"])

                # 4.1 Prepend del carry al inicio de la ventana
                # (block ya contiene lo anterior, net_qty ya acumulado)
                for t in trades_all:
                    q = float(t["qty"])
                    qty_signed = q if t["side"] == "BUY" else -q
                    net_qty += qty_signed
                    block.append(t)

                    # ¿se cerró el bloque?
                    if abs(net_qty) < 1e-10:
                        open_t  = block[0]["time"]
                        close_t = block[-1]["time"]
                        buys  = [b for b in block if b["side"] == "BUY"]
                        sells = [s for s in block if s["side"] == "SELL"]

                        def avg_price(lst):
                            qsum = sum(float(x["qty"]) for x in lst)
                            nsum = sum(float(x["qty"]) * float(x["price"]) for x in lst)
                            return nsum / qsum if qsum else 0.0

                        long_qty  = sum(float(b["qty"]) for b in buys)
                        short_qty = sum(float(s["qty"]) for s in sells)
                        # ✅ REGLA ROBUSTA: side = primera trade del bloque
                        first_trade_side = block[0]["side"]
                        side = "long" if first_trade_side == "BUY" else "short"
                        # Para depurar, también calculamos la “dominancia” por cantidades
                        dominance_side = "long" if long_qty >= short_qty else "short"

                        if side == "long":
                            entry = avg_price(buys)
                            close = avg_price(sells)
                            size  = min(long_qty, short_qty)
                        else:
                            entry = avg_price(sells)
                            close = avg_price(buys)
                            size  = min(long_qty, short_qty)

                
                        # --- Asociar income al bloque ---
                        
                        block_trade_ids = {str(int(x["id"])) for x in block if "id" in x}
                        
                        # 1) PNL y COMMISSION: prioriza emparejar por tradeId; si no hay, cae a time-range
                        incs_pnl_fee = [
                            i for i in income_by_symbol[sym]
                            if i.get("incomeType") in ("REALIZED_PNL", "COMMISSION")
                                and i.get("tradeId") and str(i["tradeId"]) in block_trade_ids
                        ]
                        
                        if not incs_pnl_fee:
                            incs_pnl_fee = [
                                i for i in income_by_symbol[sym]
                                if i.get("incomeType") in ("REALIZED_PNL", "COMMISSION")
                                    and open_t <= i.get("time", 0) <= close_t
                            ]
                        
                        # 2) FUNDING_FEE: SIEMPRE por rango temporal (no tiene tradeId)
                        incs_funding = [
                            i for i in income_by_symbol[sym]
                            if i.get("incomeType") == "FUNDING_FEE"
                                and open_t <= i.get("time", 0) <= close_t
                        ]
                        
                        pnl     = sum(float(i["income"]) for i in incs_pnl_fee if i["incomeType"] == "REALIZED_PNL")
                        fees    = sum(float(i["income"]) for i in incs_pnl_fee if i["incomeType"] == "COMMISSION")
                        funding = sum(float(i["income"]) for i in incs_funding)
                        
                        if debug:
                            link_mode = "tradeId" if any(i.get("tradeId") for i in incs_pnl_fee) else "time-range"
                            print(f"    [BLOCK] {sym} side={side.upper()} (first={first_trade_side}, dom={dominance_side}) "
                                  f"size={size:.4f} entry={entry:.6f} close={close:.6f}")
                            print(f"      Buys={len(buys)}({long_qty:.4f}) Sells={len(sells)}({short_qty:.4f}) "
                                  f"open={_iso(open_t)} close={_iso(close_t)}")
                            print(f"      Income link: PnL/Fees={'tradeId' if any(i.get('tradeId') for i in incs_pnl_fee) else 'time-range'}, Funding=time-range")
                            print(f"      Totals → pnl={pnl:.6f} fee={fees:.6f} funding={funding:.6f}")
                        realized_total = pnl + fees + funding  # ✅ incluye todo

                        results.append({
                            "exchange": "binance",
                            "symbol": sym,
                            "side": side,
                            "size": size,
                            "entry_price": entry,
                            "close_price": close,
                            "notional": entry * size,
                            "fees": fees,                   # (normalmente negativas)
                            "funding_fee": funding,         # (+ cobro / - pago)
                            "pnl": pnl,            # SOLO precio
                            "realized_pnl": realized_total,
                            "open_time": int(open_t/1000),  # epoch s
                            "close_time": int(close_t/1000),
                            # por si tu frontend todavía usa strings:
                            "open_date":  datetime.fromtimestamp(open_t/1000).strftime("%Y-%m-%d %H:%M:%S"),
                            "close_date": datetime.fromtimestamp(close_t/1000).strftime("%Y-%m-%d %H:%M:%S"),
                        })

                        # reset bloque para el siguiente
                        block = []
                        net_qty = 0.0

                # 4.2 Guardar carry para la siguiente ventana
                carry_block_by_sym[sym] = block[:]    # copia
                carry_net_by_sym[sym]   = net_qty

            # fin ventanas

        if debug:
            print(f"\n✅ Binance closed positions totales: {len(results)}")
        return results

    except Exception as e:
        print(f"❌ Binance closed positions error: {e}")
        return []

def save_binance_closed_positions(db_path="portfolio.db", days=30, debug=False):
    """
    Guarda posiciones cerradas de Binance en SQLite delegando el cálculo derivado
    (pnl_percent, apr, initial_margin) a db_manager.save_closed_position.

    - Dedupe por (exchange, symbol, close_time, size)
    - pnl (precio) = realized_pnl - fees - funding
    - fee_total se normaliza dentro de db_manager a negativo
    """
    import sqlite3
    import db_manager as dm
    from db_manager import save_closed_position as save_cp

    # Asegura que db_manager use el DB que le pasamos
    dm.DB_PATH = db_path

    positions = fetch_closed_positions_binance(days=days, debug=debug)
    if not positions:
        print("⚠️ No se encontraron posiciones cerradas en Binance.")
        return 0

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    saved = 0
    skipped = 0

    for pos in positions:
        try:
            exchange = pos.get("exchange", "binance")
            symbol   = pos["symbol"]
            close_ts = int(pos.get("close_time") or 0)
            size     = float(pos["size"])

            # Dedupe por exchange + symbol + close_time + size
            cur.execute("""
                SELECT 1 FROM closed_positions
                WHERE exchange = ? AND symbol = ? AND close_time = ? AND ABS(size - ?) < 1e-8
            """, (exchange, symbol, close_ts, size))
            if cur.fetchone():
                if debug:
                    print(f"↩️  Skip duplicado: {exchange} {symbol} t={close_ts} size={size}")
                skipped += 1
                continue

            # Asegurar PNL de precio: realized - fees - funding
            realized_total = float(pos.get("realized_pnl", 0.0))
            fees           = float(pos.get("fees", 0.0))           # db_manager lo volverá negativo
            funding        = float(pos.get("funding_fee", 0.0))    # positivo si recibes, negativo si pagas
            pnl_price      = pos.get("pnl")
            if pnl_price is None:
                pnl_price = realized_total - fees - funding

            entry_price = float(pos["entry_price"])
            close_price = float(pos["close_price"])
            notional    = float(pos.get("notional", entry_price * size))

            payload = {
                "exchange": exchange,
                "symbol": symbol,
                "side": pos["side"],                    # "long" o "short"
                "size": size,
                "entry_price": entry_price,
                "close_price": close_price,
                "open_time": int(pos["open_time"]),
                "close_time": close_ts,

                # Núcleo de PnL
                "pnl": float(pnl_price),                # SOLO precio
                "realized_pnl": realized_total,         # total con fees+funding
                "funding_total": funding,
                "fee_total": fees,

                # Soportes para db_manager
                "notional": notional,
                "leverage": pos.get("leverage"),
                "liquidation_price": pos.get("liquidation_price"),
            }

            # Guarda con lógica de db_manager (calcula pnl_percent, apr, initial_margin, etc.)
            save_cp(payload)
            saved += 1

        except Exception as e:
            print(f"⚠️ Error guardando {pos.get('symbol','?')}: {e}")

    conn.close()
    if debug:
        print(f"✅ Guardadas {saved} | ⏭️ omitidas {skipped} (duplicadas).")
    return saved
    


##========= inicio del  odigo viejo 
# def fetch_closed_positions_binance(days=30, off=0, debug=False):
#     """
#     Reconstruye posiciones cerradas de Binance Futures en los últimos `days`.
#     - Ventanas de 7 días (limitación API) con estado 'carry-over' por símbolo.
#     - side correcto (long/short) por neto del bloque.
#     - entry/close correctos (para short se invierte).
#     - income asociado por tradeId cuando esté disponible; si no, por rango de tiempo.
#     - realized_pnl = SOLO precio (tu UI ya muestra fees y funding aparte).
#     """

#     def _iso(ms):
#         try:
#             return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
#         except Exception:
#             return str(ms)


#     def signed_get(path, params=None):
#         if not BINANCE_API_KEY or not BINANCE_API_SECRET:
#             raise RuntimeError("Missing BINANCE_API_KEY/BINANCE_API_SECRET")       
#         params = dict(params or {})
#         params["timestamp"] = int(time.time() * 1000) + int(off)
#         qs = urlencode(params, doseq=True)
#         sig = hmac.new(BINANCE_API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
#         url = f"{BINANCE_BASE_URL}{path}?{qs}&signature={sig}"
#         headers = {"X-MBX-APIKEY": BINANCE_API_KEY, **UA_HEADERS}
#         if debug:
#             print(f"[GET] {path} {params}")
#         r = requests.get(url, headers=headers, timeout=25)
#         r.raise_for_status()
#         return r.json()

#     try:
#         now = int(time.time() * 1000)
#         start_time = now - days * 24 * 60 * 60 * 1000

#         # 1) INCOME de todo el rango, paginado por 'page'
#         income_by_symbol = defaultdict(list)
#         page = 1
#         while True:
#             inc = signed_get("/fapi/v1/income", {
#                 "limit": 1000, "startTime": start_time, "endTime": now, "page": page
#             })
#             if not inc:
#                 break
#             for i in inc:
#                 t = i.get("incomeType")
#                 sym = i.get("symbol") or ""
#                 if t in ("REALIZED_PNL", "COMMISSION", "FUNDING_FEE") and sym:
#                     income_by_symbol[sym].append(i)
#             if len(inc) < 1000:
#                 break
#             page += 1

#         if debug:
#             print("[Income] resumen:")
#             for sym, arr in income_by_symbol.items():
#                 s_pnl = sum(float(x["income"]) for x in arr if x["incomeType"]=="REALIZED_PNL")
#                 s_fee = sum(float(x["income"]) for x in arr if x["incomeType"]=="COMMISSION")
#                 s_fnd = sum(float(x["income"]) for x in arr if x["incomeType"]=="FUNDING_FEE")
#                 print(f"  {sym}: pnl={s_pnl:.2f} fee={s_fee:.2f} fnd={s_fnd:.2f} items={len(arr)}")

#         # 2) exchangeInfo (opcional)
#         try:
#             exi = signed_get("/fapi/v1/exchangeInfo")
#             valid_symbols = {s["symbol"] for s in exi.get("symbols", [])}
#         except Exception:
#             valid_symbols = set(income_by_symbol.keys())

#         results = []

#         # 3) Estado carry-over por símbolo (bloque abierto y net_qty acumulado)
#         carry_block_by_sym = {}   # sym -> list of trades
#         carry_net_by_sym   = {}   # sym -> float

#         # 4) Procesar símbolos
#         for sym in list(income_by_symbol.keys()):
#             if sym not in valid_symbols and debug:
#                 print(f"[SKIP] {sym} no en exchangeInfo")
#             if debug:
#                 print(f"\n[Symbol] {sym} ventanas de 7d {_iso(start_time)} → {_iso(now)}")

#             end_time = start_time
#             # inicializar carry si existe
#             block = carry_block_by_sym.get(sym, [])
#             net_qty = carry_net_by_sym.get(sym, 0.0)

#             while end_time < now:
#                 chunk_start = end_time
#                 chunk_end = min(chunk_start + 7*24*60*60*1000, now)
#                 end_time = chunk_end
#                 if debug:
#                     print(f"  [Window] {_iso(chunk_start)} → {_iso(chunk_end)} (carry net={net_qty:.6f}, block={len(block)})")

#                 # Paginación userTrades por fromId
#                 trades_all = []
#                 last_id = None
#                 while True:
#                     params = {
#                         "symbol": sym,
#                         "startTime": chunk_start,
#                         "endTime": chunk_end,
#                         "limit": 1000,
#                     }
#                     if last_id is not None:
#                         params["fromId"] = last_id + 1
#                     try:
#                         tpage = signed_get("/fapi/v1/userTrades", params)
#                     except Exception as e:
#                         if debug:
#                             print(f"    [WARN] userTrades fallo: {e}")
#                         break
#                     if not tpage:
#                         break
#                     trades_all.extend(tpage)
#                     if len(tpage) < 1000:
#                         break
#                     last_id = int(tpage[-1]["id"])

#                 if debug:
#                     print(f"    trades nuevos: {len(trades_all)}")

#                 if trades_all:
#                     trades_all.sort(key=lambda x: x["time"])

#                 # 4.1 Prepend del carry al inicio de la ventana
#                 # (block ya contiene lo anterior, net_qty ya acumulado)
#                 for t in trades_all:
#                     q = float(t["qty"])
#                     qty_signed = q if t["side"] == "BUY" else -q
#                     net_qty += qty_signed
#                     block.append(t)

#                     # ¿se cerró el bloque?
#                     if abs(net_qty) < 1e-10:
#                         open_t  = block[0]["time"]
#                         close_t = block[-1]["time"]
#                         buys  = [b for b in block if b["side"] == "BUY"]
#                         sells = [s for s in block if s["side"] == "SELL"]

#                         def avg_price(lst):
#                             qsum = sum(float(x["qty"]) for x in lst)
#                             nsum = sum(float(x["qty"]) * float(x["price"]) for x in lst)
#                             return nsum / qsum if qsum else 0.0

#                         long_qty  = sum(float(b["qty"]) for b in buys)
#                         short_qty = sum(float(s["qty"]) for s in sells)
#                         # ✅ REGLA ROBUSTA: side = primera trade del bloque
#                         first_trade_side = block[0]["side"]
#                         side = "long" if first_trade_side == "BUY" else "short"
#                         # Para depurar, también calculamos la “dominancia” por cantidades
#                         dominance_side = "long" if long_qty >= short_qty else "short"

#                         if side == "long":
#                             entry = avg_price(buys)
#                             close = avg_price(sells)
#                             size  = min(long_qty, short_qty)
#                         else:
#                             entry = avg_price(sells)
#                             close = avg_price(buys)
#                             size  = min(long_qty, short_qty)

                
#                         # --- Asociar income al bloque ---
                        
#                         block_trade_ids = {str(int(x["id"])) for x in block if "id" in x}
                        
#                         # 1) PNL y COMMISSION: prioriza emparejar por tradeId; si no hay, cae a time-range
#                         incs_pnl_fee = [
#                             i for i in income_by_symbol[sym]
#                             if i.get("incomeType") in ("REALIZED_PNL", "COMMISSION")
#                                and i.get("tradeId") and str(i["tradeId"]) in block_trade_ids
#                         ]
                        
#                         if not incs_pnl_fee:
#                             incs_pnl_fee = [
#                                 i for i in income_by_symbol[sym]
#                                 if i.get("incomeType") in ("REALIZED_PNL", "COMMISSION")
#                                    and open_t <= i.get("time", 0) <= close_t
#                             ]
                        
#                         # 2) FUNDING_FEE: SIEMPRE por rango temporal (no tiene tradeId)
#                         incs_funding = [
#                             i for i in income_by_symbol[sym]
#                             if i.get("incomeType") == "FUNDING_FEE"
#                                and open_t <= i.get("time", 0) <= close_t
#                         ]
                        
#                         pnl     = sum(float(i["income"]) for i in incs_pnl_fee if i["incomeType"] == "REALIZED_PNL")
#                         fees    = sum(float(i["income"]) for i in incs_pnl_fee if i["incomeType"] == "COMMISSION")
#                         funding = sum(float(i["income"]) for i in incs_funding)
                        
#                         if debug:
#                             link_mode = "tradeId" if any(i.get("tradeId") for i in incs_pnl_fee) else "time-range"
#                             print(f"    [BLOCK] {sym} side={side.upper()} (first={first_trade_side}, dom={dominance_side}) "
#                                   f"size={size:.4f} entry={entry:.6f} close={close:.6f}")
#                             print(f"      Buys={len(buys)}({long_qty:.4f}) Sells={len(sells)}({short_qty:.4f}) "
#                                   f"open={_iso(open_t)} close={_iso(close_t)}")
#                             print(f"      Income link: PnL/Fees={'tradeId' if any(i.get('tradeId') for i in incs_pnl_fee) else 'time-range'}, Funding=time-range")
#                             print(f"      Totals → pnl={pnl:.6f} fee={fees:.6f} funding={funding:.6f}")
#                         realized_total = pnl + fees + funding  # ✅ incluye todo

#                         results.append({
#                             "exchange": "binance",
#                             "symbol": sym,
#                             "side": side,
#                             "size": size,
#                             "entry_price": entry,
#                             "close_price": close,
#                             "notional": entry * size,
#                             "fees": fees,                   # (normalmente negativas)
#                             "funding_fee": funding,         # (+ cobro / - pago)
#                             "pnl": pnl,            # SOLO precio
#                             "realized_pnl": realized_total,
#                             "open_time": int(open_t/1000),  # epoch s
#                             "close_time": int(close_t/1000),
#                             # por si tu frontend todavía usa strings:
#                             "open_date":  datetime.fromtimestamp(open_t/1000).strftime("%Y-%m-%d %H:%M:%S"),
#                             "close_date": datetime.fromtimestamp(close_t/1000).strftime("%Y-%m-%d %H:%M:%S"),
#                         })

#                         # reset bloque para el siguiente
#                         block = []
#                         net_qty = 0.0

#                 # 4.2 Guardar carry para la siguiente ventana
#                 carry_block_by_sym[sym] = block[:]    # copia
#                 carry_net_by_sym[sym]   = net_qty

#             # fin ventanas

#         if debug:
#             print(f"\n✅ Binance closed positions totales: {len(results)}")
#         return results

#     except Exception as e:
#         print(f"❌ Binance closed positions error: {e}")
#         return []
  
    
# def save_binance_closed_positions(db_path="portfolio.db", days=30, debug=False):
#     """
#     Guarda posiciones cerradas de Binance en SQLite.
#     Dedupe por (symbol, close_time, size) para evitar falsos duplicados.
#     """
#     import sqlite3
#     positions = fetch_closed_positions_binance(days=days, debug=debug)
#     if not positions:
#         print("⚠️ No se encontraron posiciones cerradas en Binance.")
#         return

#     conn = sqlite3.connect(db_path)
#     cur = conn.cursor()
#     saved = 0
#     skipped = 0

#     for pos in positions:
#         try:
#             close_ts = pos.get("close_time")
#             symbol   = pos["symbol"]
#             size     = float(pos["size"])

#             # dedupe por símbolo + close_time + size
#             cur.execute("""
#                 SELECT COUNT(*) FROM closed_positions
#                 WHERE symbol = ? AND close_time = ? AND ABS(size - ?) < 1e-8
#             """, (symbol, close_ts, size))
#             if cur.fetchone()[0]:
#                 skipped += 1
#                 continue

#             cur.execute("""
#                 INSERT INTO closed_positions (
#                     exchange, symbol, side, size, entry_price, close_price,
#                     open_time, close_time, realized_pnl, funding_total,
#                     fee_total, notional, leverage, liquidation_price
#                 )
#                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#             """, (
#                 pos["exchange"],
#                 symbol,
#                 pos["side"],
#                 size,
#                 pos["entry_price"],
#                 pos["close_price"],
#                 pos["open_time"],
#                 close_ts,
#                 pos["realized_pnl"],           # SOLO precio
#                 pos.get("funding_fee", 0.0),
#                 -abs(pos.get("fees", 0.0)),    # fees siempre negativas
#                 pos["notional"],
#                 None,
#                 None
#             ))
#             saved += 1
#         except Exception as e:
#             print(f"⚠️ Error guardando {symbol}: {e}")

#     conn.commit()
#     conn.close()
#     print(f"✅ Guardadas {saved} posiciones cerradas de Binance (omitidas {skipped} duplicadas).")    
    
    ## fin del codigo viejo 