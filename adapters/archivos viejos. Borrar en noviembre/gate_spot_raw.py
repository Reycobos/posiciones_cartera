#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gate_spot_raw.py
Obtiene *RAW* de tus transacciones spot desde Gate.io usando *exclusivamente*
la función `_request` del módulo `gate2` (firma incluida allí).

Imprime en filas pequeñas por defecto (una línea por trade). Con --json imprime el JSON crudo.
Por defecto consulta ALPACA_USDT en una ventana reciente.

Uso:
  python gate_spot_raw.py
  python gate_spot_raw.py --pair ALPACA_USDT --since "2025-10-06 00:00:00" --until "2025-10-08 00:00:00"
  python gate_spot_raw.py --pair ALPACA_USDT --days 3 --limit 1000 --max-pages 5
  python gate_spot_raw.py --pair ALPACA_USDT --days 3 --json

Requisitos:
  - Tener gate2.py accesible en PYTHONPATH (mismo proyecto/carpeta) con `_request` ya operativo.
"""

import sys
import json
import argparse
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# Usamos SOLO gate2._request (firma, headers, etc. resueltos allí)
from gate2 import _request

EP = "/spot/my_trades"


def _dt_to_epoch_s(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # asumimos UTC si viene naive
    return int(dt.timestamp())


def _fetch_all(pair: str, since: Optional[datetime], until: Optional[datetime],
               limit: int, max_pages: int) -> List[Dict[str, Any]]:
    """Pagina sobre /spot/my_trades devolviendo lista acumulada."""
    out: List[Dict[str, Any]] = []
    page = 1
    while page <= max_pages:
        params: Dict[str, Any] = {"currency_pair": pair, "limit": limit, "page": page}
        if since:
            params["from"] = _dt_to_epoch_s(since)
        if until:
            params["to"] = _dt_to_epoch_s(until)

        rows = _request("GET", EP, params=params) or []
        if not isinstance(rows, list):
            # Algunos entornos devuelven dict con 'data'
            rows = rows.get("data", []) if isinstance(rows, dict) else []
        if not rows:
            break
        out.extend(rows)
        # Si ya vino menos que el límite, no hace falta seguir
        if len(rows) < limit:
            break
        page += 1
    return out


def _compact_line(t: Dict[str, Any]) -> str:
    # Campos típicos: id, create_time, side, amount, price, fee, fee_currency, role, order_id, currency_pair
    # Pequeña, legible y con lo esencial
    def _ts_str(x):
        try:
            return datetime.utcfromtimestamp(int(float(x))).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(x)

    ts = _ts_str(t.get("create_time"))
    side = t.get("side")
    px = t.get("price")
    qty = t.get("amount")
    fee = t.get("fee")
    fccy = t.get("fee_currency")
    pair = t.get("currency_pair")
    tid = t.get("id")

    total = None
    try:
        total = float(qty) * float(px)
    except Exception:
        total = None
    total_str = f"{total:.6f}" if total is not None else "?"

    # Línea compacta (≤ ~120 chars)
    return f"{ts} | {pair} | id={tid} | {side} | px={px} | qty={qty} | fee={fee} {fccy} | total≈{total_str}"


def main():
    ap = argparse.ArgumentParser(description="RAW spot trades desde Gate.io (filas compactas o JSON).")
    ap.add_argument("--pair", type=str, default="ALPACA_USDT", help="Par spot Gate (ej: ALPACA_USDT)")
    ap.add_argument("--since", type=str, default=None, help="Inicio UTC 'YYYY-MM-DD HH:MM:SS' (naive=UTC)")
    ap.add_argument("--until", type=str, default=None, help="Fin UTC 'YYYY-MM-DD HH:MM:SS' (naive=UTC)")
    ap.add_argument("--days", type=int, default=30, help="Ventana hacia atrás si no usas --since/--until")
    ap.add_argument("--limit", type=int, default=1000, help="Límite por página (1..1000)")
    ap.add_argument("--max-pages", type=int, default=5, help="Número máximo de páginas a recuperar")
    ap.add_argument("--json", action="store_true", help="Imprime JSON crudo en lugar de filas compactas")
    args = ap.parse_args()

    # Resolución de ventana temporal
    since_dt = None
    until_dt = None
    if args.since:
        since_dt = datetime.strptime(args.since, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    if args.until:
        until_dt = datetime.strptime(args.until, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    if not since_dt and not until_dt and args.days:
        until_dt = datetime.now(timezone.utc)
        since_dt = until_dt - timedelta(days=args.days)

    # Fetch paginado usando gate2._request
    try:
        rows = _fetch_all(args.pair, since_dt, until_dt, max(1, min(args.limit, 1000)), max(1, args.max_pages))
    except Exception as e:
        print(f"❌ Error pidiendo {EP}: {e}")
        sys.exit(2)

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    if not rows:
        print("(sin trades)")
        return

    # Orden ascendente por create_time si existe
    try:
        rows = sorted(rows, key=lambda x: int(float(x.get('create_time', 0))))
    except Exception:
        pass

    # Imprimir filas muy cortas
    for t in rows:
        print(_compact_line(t))


if __name__ == "__main__":
    main()
