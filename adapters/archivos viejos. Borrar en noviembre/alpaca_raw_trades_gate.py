#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alpaca_raw_trades_gate.py
Descarga las transacciones RAW del endpoint privado de Gate.io para un par spot (por defecto ALPACA_USDT)
y las imprime en filas compactas (una línea por trade), ideales para pantalla estrecha.

Prioridad de backend:
  1) Si existe adapters.gate2._request en tu proyecto, lo usa directamente.
  2) Si no, usa una implementación HTTP firmada (API v4) con GATE_API_KEY / GATE_API_SECRET en variables de entorno.

Ejemplos:
  python alpaca_raw_trades_gate.py
  python alpaca_raw_trades_gate.py --pair ALPACA_USDT --days 7 --limit 200
  python alpaca_raw_trades_gate.py --since "2025-10-01 00:00:00" --until "2025-10-10 00:00:00" --limit 100
"""

import os
import sys
import hmac
import json
import time
import hashlib
import argparse
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

API_HOST = "https://api.gateio.ws"
API_PREFIX = "/api/v4"


def _dt_to_epoch_s(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _try_import_gate2_request():
    try:
        from adapters.gate2 import _request as gate_request
        return gate_request
    except Exception:
        return None


def _sign_v4(secret: str, method: str, path: str, query: str, body: str, ts: str) -> str:
    payload = "\n".join([method.upper(), path, query, body, ts])
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha512).hexdigest()


def _http_request_signed(method: str, path: str, params: Optional[Dict[str, Any]] = None, body: Optional[Dict[str, Any]] = None):
    """
    Fallback HTTP client for Gate.io API v4, signing with HMAC SHA512.
    Requires env vars: GATE_API_KEY, GATE_API_SECRET
    """
    import requests
    key = os.environ.get("GATE_API_KEY")
    secret = os.environ.get("GATE_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Faltan credenciales: establece GATE_API_KEY y GATE_API_SECRET en el entorno.")

    url = API_HOST + API_PREFIX + path
    params = params or {}
    body = body or {}

    # Build query string (sorted for consistency)
    from urllib.parse import urlencode
    query = urlencode(params, doseq=True)

    # Serialize body (JSON for non-GET)
    body_json = "" if method.upper() == "GET" else json.dumps(body, separators=(",", ":"))

    ts = str(int(time.time()))
    sign = _sign_v4(secret, method, API_PREFIX + path, query, body_json, ts)

    headers = {
        "KEY": key,
        "Timestamp": ts,
        "SIGN": sign,
        "Content-Type": "application/json"
    }

    if method.upper() == "GET":
        resp = requests.get(url, headers=headers, params=params, timeout=30)
    else:
        resp = requests.request(method.upper(), url, headers=headers, params=params, data=body_json, timeout=30)

    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    try:
        return resp.json()
    except Exception:
        return resp.text


def _fetch_raw_trades(pair: str, since: Optional[datetime], until: Optional[datetime], limit: int, page: Optional[int]) -> List[Dict[str, Any]]:
    """
    Llama a /spot/my_trades para un par concreto.
    Gate soporta filtros de tiempo 'from' y 'to' en epoch segundos (algunos clientes soportan 'page').
    Probamos con 'from'/'to'; si falla, reintenta sin tiempo.
    """
    params: Dict[str, Any] = {"currency_pair": pair, "limit": limit}
    if page is not None:
        params["page"] = page
    if since:
        params["from"] = _dt_to_epoch_s(since)
    if until:
        params["to"] = _dt_to_epoch_s(until)

    # Backend preferente: adapters.gate2._request
    gate_request = _try_import_gate2_request()
    try:
        if gate_request:
            return gate_request("GET", "/spot/my_trades", params=params) or []
        # Fallback firmado
        return _http_request_signed("GET", "/spot/my_trades", params=params) or []
    except Exception as e:
        # Reintento sin filtros de tiempo si hubo error de parámetros
        if "INVALID_PARAM_VALUE" in str(e) or "422" in str(e):
            params.pop("from", None)
            params.pop("to", None)
            return (gate_request("GET", "/spot/my_trades", params=params) if gate_request
                    else _http_request_signed("GET", "/spot/my_trades", params=params)) or []
        raise


def _compact_line(t: Dict[str, Any]) -> str:
    # Campos típicos devueltos por Gate: id, create_time, create_time_ms, side, amount, price, fee, fee_currency, role, order_id
    ts = t.get("create_time")
    try:
        ts_str = datetime.utcfromtimestamp(int(float(ts))).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        ts_str = str(ts)

    fee_val = t.get("fee")
    fee_ccy = t.get("fee_currency")
    fee_part = f"{fee_val} {fee_ccy}" if fee_val is not None and fee_ccy else f"{fee_val}"

    # Una línea pequeña y legible
    return (
        f"{ts_str} | id={t.get('id')} | {t.get('side')} | px={t.get('price')} | "
        f"qty={t.get('amount')} | fee={fee_part} | total≈{float(t.get('amount',0))*float(t.get('price',0)):.6f}"
    )


def main():
    ap = argparse.ArgumentParser(description="Imprime trades RAW (Gate spot my_trades) en filas compactas.")
    ap.add_argument("--pair", type=str, default="ALPACA_USDT", help="Par spot (ej: ALPACA_USDT)")
    ap.add_argument("--days", type=int, default=3, help="Ventana en días hacia atrás (ignorado si usas --since/--until)")
    ap.add_argument("--since", type=str, default=None, help="Inicio (UTC) 'YYYY-MM-DD HH:MM:SS'")
    ap.add_argument("--until", type=str, default=None, help="Fin (UTC) 'YYYY-MM-DD HH:MM:SS'")
    ap.add_argument("--limit", type=int, default=1000, help="Límite por llamada (max ~1000)")
    ap.add_argument("--page", type=int, default=None, help="Página (opcional, si tu cuenta lo soporta)")
    ap.add_argument("--json", action="store_true", help="Imprime JSON crudo en vez de filas compactas")
    args = ap.parse_args()

    since_dt = None
    until_dt = None
    if args.since:
        since_dt = datetime.strptime(args.since, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    if args.until:
        until_dt = datetime.strptime(args.until, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    if not since_dt and not until_dt and args.days:
        until_dt = datetime.now(timezone.utc)
        since_dt = until_dt - timedelta(days=args.days)

    trades = _fetch_raw_trades(args.pair, since_dt, until_dt, args.limit, args.page)

    if args.json:
        print(json.dumps(trades, ensure_ascii=False, indent=2))
        return

    # Ordenamos por tiempo asc y mostramos filas cortas
    try:
        trades = sorted(trades, key=lambda x: int(float(x.get("create_time", 0))))
    except Exception:
        pass

    if not trades:
        print("(sin trades)")
        return

    # salida compacta
    for t in trades:
        print(_compact_line(t))


if __name__ == "__main__":
    main()
