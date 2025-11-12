# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 19:37:41 2025

@author: aleja
"""

# debug_positionRisk_aster.py
# Ejecuta: Run en Spyder (sin argumentos).
# Requiere: variables de entorno ASTER_API_KEY y ASTER_API_SECRET (y opcional ASTER_HOST).
import os, time, hmac, hashlib, json
from datetime import datetime, timedelta, timezone
import requests

HOSTS = [
    (os.getenv("ASTER_HOST") or "https://fapi.asterdex.com").rstrip("/"),
    "https://fapi.aster.finance",
    "https://api.asterdex.com",
    "https://api.aster.finance",
]

API_KEY = "8491e7b39a2782f066fa8355a3afe1883345b5228007ab11609290fdde314853"
API_SECRET = "67c1601e4bfb7ae48f1513e165a4caa711b2566aeebab98a8e0d22a47e6c4138"

def _require_keys():
    if not API_KEY or not API_SECRET:
        raise SystemExit("❌ Faltan ASTER_API_KEY / ASTER_API_SECRET en variables de entorno.")

def _sign(params: dict) -> dict:
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    sig = hmac.new(API_SECRET.encode("utf-8"), qs.encode("utf-8"), hashlib.sha256).hexdigest()
    params["signature"] = sig
    return params

def signed_get(path: str, params: dict | None = None, timeout=30):
    _require_keys()
    base = {"timestamp": int(time.time() * 1000), "recvWindow": 5000}
    if params:
        base.update(params)
    signed = _sign(base)
    headers = {"X-MBX-APIKEY": API_KEY, "User-Agent": "python-requests"}
    last_err = None
    for host in HOSTS:
        url = f"{host}{path}"
        try:
            r = requests.get(url, params=signed, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Todos los hosts fallaron para {path}. Último error: {last_err!r}")

def pretty(dt_ms: int | None):
    if not dt_ms:
        return "-"
    return datetime.fromtimestamp(dt_ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def sum_income(symbol: str, income_type: str, start_ms: int, end_ms: int) -> float:
    data = signed_get("/fapi/v1/income", {
        "symbol": symbol,
        "incomeType": income_type,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000
    }) or []
    total = 0.0
    for it in data:
        try:
            total += float(it.get("income", 0) or 0.0)
        except Exception:
            pass
    return round(total, 8)

def main():
    print("🔧 Debug Aster — positionRisk + income + trades")
    print(f"🌐 Hosts: {', '.join(HOSTS)}")
    # 1) RAW positionRisk
    arr = signed_get("/fapi/v2/positionRisk")
    print(f"\n=== RAW /fapi/v2/positionRisk (items={len(arr) if isinstance(arr, list) else 'n/a'}) ===")
    print(json.dumps(arr, indent=2, ensure_ascii=False))

    # 2) Resumen por símbolos con positionAmt != 0
    non_zero = []
    for p in (arr or []):
        try:
            amt = float(p.get("positionAmt") or 0.0)
            if abs(amt) > 0:
                non_zero.append(p)
        except Exception:
            continue

    if not non_zero:
        print("\nℹ️ No hay posiciones abiertas (positionAmt = 0 en todos los símbolos).")
        return

    end_ms = int(time.time() * 1000)
    start_7d = end_ms - 7*24*3600*1000
    start_24h = end_ms - 24*3600*1000

    print("\n=== RESUMEN SÍMBOLOS ABIERTOS ===")
    for p in non_zero:
        sym = p.get("symbol", "")
        side = "LONG" if float(p.get("positionAmt") or 0) > 0 else "SHORT"
        entry = float(p.get("entryPrice") or 0)
        mark = float(p.get("markPrice") or 0)
        liq  = float(p.get("liquidationPrice") or 0)
        unpnl = float(p.get("unRealizedProfit") or 0)
        lev  = float(p.get("leverage") or 0)
        upd  = int(p.get("updateTime") or 0)
        notional = float(p.get("notional", 0) or (abs(float(p.get("positionAmt") or 0))*entry))

        # ingresos / gastos
        funding_24h   = sum_income(sym, "FUNDING_FEE",   start_24h, end_ms)
        commission_7d = sum_income(sym, "COMMISSION",    start_7d,  end_ms)
        realized_7d   = sum_income(sym, "REALIZED_PNL",  start_7d,  end_ms)

        print(f"\n— {sym} — {side}")
        print(f"  entry={entry}  mark={mark}  liq={liq}  lev={lev}  notional≈{notional}")
        print(f"  unRealizedPnL={unpnl}  updateTime={pretty(upd)}")
        print(f"  funding(24h)={funding_24h}  commission(7d)={commission_7d}  realizedPnL(7d)={realized_7d}")

        # Trades recientes (conteo y ventana)
        try:
            trades = signed_get("/fapi/v1/userTrades", {
                "symbol": sym, "startTime": start_7d, "endTime": end_ms, "limit": 1000
            }) or []
            realized_from_trades = 0.0
            for t in trades:
                rp = t.get("realizedPnl")
                if rp is not None:
                    try: realized_from_trades += float(rp or 0.0)
                    except: pass
            print(f"  trades(7d)={len(trades)}  realizedPnL_from_trades(7d)≈{round(realized_from_trades,8)}")
        except Exception as e:
            print(f"  ⚠️ userTrades error: {e}")

if __name__ == "__main__":
    main()
