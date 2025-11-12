# debug_whitebit_history.py
# RAW de /api/v4/collateral-account/positions/history + conversión de fechas legibles
# Requiere: WHITEBIT_API_KEY y WHITEBIT_API_SECRET en el entorno.

import os
import time
import hmac
import hashlib
import base64
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import requests
from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # Py>=3.9
except Exception:
    ZoneInfo = None  # fallback

BASE_URL = "https://whitebit.com"
TIMEOUT = 30

WHITEBIT_API_KEY = os.getenv("WHITEBIT_API_KEY", "").strip()
WHITEBIT_API_SECRET = os.getenv("WHITEBIT_API_SECRET", "").strip()

# ---- Auth helpers ----
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
    signature = hmac.new(
        WHITEBIT_API_SECRET.encode("utf-8"),
        payload_b64,
        hashlib.sha512
    ).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-TXC-APIKEY": WHITEBIT_API_KEY,
        "X-TXC-PAYLOAD": payload_b64.decode(),
        "X-TXC-SIGNATURE": signature,
    }
    return headers, body

def wb_post(path: str, payload: Dict[str, Any]) -> Any:
    headers, body = _auth_headers(path, payload)
    url = f"{BASE_URL}{path}"
    r = requests.post(url, data=json.dumps(body), headers=headers, timeout=TIMEOUT)
    try:
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"HTTP {r.status_code} error: {r.text}") from e
    return r.json()

# ---- Fetch RAW ----
def fetch_positions_history_raw(
    market: Optional[str] = None,
    position_id: Optional[int] = None,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    limit = max(1, min(100, int(limit)))
    payload: Dict[str, Any] = {"limit": limit, "offset": max(0, int(offset))}
    if market:
        payload["market"] = market
    if position_id is not None:
        payload["positionId"] = int(position_id)
    if start_ts is not None:
        payload["startDate"] = int(start_ts)
    if end_ts is not None:
        payload["endDate"] = int(end_ts)
    return wb_post("/api/v4/collateral-account/positions/history", payload) or []

# ---- Date helpers ----
DATE_KEYS = {
    "openDate", "modifyDate", "closeDate",            # positions/history
    "fundingTime", "rateCalculatedTime",              # funding-history
    "ctime", "mtime", "time", "timestamp"             # genéricos
}

def _to_epoch_seconds(value: Any) -> Optional[float]:
    """
    Convierte value a epoch seconds (float). Detecta ms vs s.
    Acepta int/float/str numérica. Devuelve None si no es válido.
    """
    try:
        if isinstance(value, (int, float)):
            ts = float(value)
        elif isinstance(value, str):
            # admite "1734451200" o "1734451200123" o "1650400589.882613"
            ts = float(value.strip())
        else:
            return None
    except Exception:
        return None

    # Heurística ms -> s
    if ts > 1e12:
        ts /= 1000.0
    return ts

def ts_to_str(ts_seconds: float, tz_name: str = "Europe/Zurich") -> str:
    """
    Devuelve 'YYYY-MM-DD HH:MM:SS TZ' en la zona indicada.
    Si ZoneInfo no está disponible, usa hora local sin TZ.
    """
    try:
        if ZoneInfo is not None and tz_name:
            dt = datetime.fromtimestamp(ts_seconds, tz=ZoneInfo(tz_name))
            return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        else:
            dt = datetime.fromtimestamp(ts_seconds)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts_seconds)

def decorate_dates(obj: Any, tz_name: str = "Europe/Zurich") -> Any:
    """
    Recorre dict/list y añade claves '*_readable' para fechas conocidas.
    No elimina las originales.
    """
    if isinstance(obj, list):
        return [decorate_dates(x, tz_name) for x in obj]
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            new[k] = decorate_dates(v, tz_name)
        # añade *_readable para claves candidatas
        for k, v in obj.items():
            if k in DATE_KEYS:
                ts = _to_epoch_seconds(v)
                if ts is not None:
                    new[f"{k}_readable"] = ts_to_str(ts, tz_name)
        # también mira campos anidados que contengan "Time"/"Date" al final
        for k, v in obj.items():
            if isinstance(k, str) and (k.lower().endswith("time") or k.lower().endswith("date")):
                ts = _to_epoch_seconds(v)
                if ts is not None and f"{k}_readable" not in new:
                    new[f"{k}_readable"] = ts_to_str(ts, tz_name)
        return new
    return obj

# ---- CLI ----
def main():
    ap = argparse.ArgumentParser(description="RAW debug WhiteBIT positions/history (+ fechas legibles)")
    ap.add_argument("--market", type=str, default=None)
    ap.add_argument("--position-id", type=int, default=None)
    ap.add_argument("--start", type=int, default=None, help="epoch seconds")
    ap.add_argument("--end", type=int, default=None, help="epoch seconds")
    ap.add_argument("--days", type=int, default=None, help="ventana relativa si no pasas start/end")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--pages", type=int, default=1)
    ap.add_argument("--sleep-ms", type=int, default=250)
    ap.add_argument("--raw-file", type=str, default=None)
    ap.add_argument("--no-pretty", action="store_true")
    ap.add_argument("--decorate-dates", action="store_true", help="añade *_readable con fechas humanas")
    ap.add_argument("--tz", type=str, default="Europe/Zurich", help="Zona horaria para *_readable")
    args = ap.parse_args()

    # Ventana relativa
    start_ts, end_ts = args.start, args.end
    if (start_ts is None or end_ts is None) and args.days:
        end_ts = int(time.time())
        start_ts = end_ts - args.days * 86400

    all_rows: List[Dict[str, Any]] = []
    off = args.offset
    for _ in range(max(1, args.pages)):
        rows = fetch_positions_history_raw(
            market=args.market,
            position_id=args.position_id,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=args.limit,
            offset=off,
        )
        all_rows.extend(rows)
        if len(rows) < args.limit:
            break
        off += args.limit
        time.sleep(max(0, args.sleep_ms) / 1000.0)

    out = decorate_dates(all_rows, args.tz) if args.decorate_dates else all_rows

    print(json.dumps(out, ensure_ascii=False, indent=None if args.no_pretty else 2))

    if args.raw_file:
        with open(args.raw_file, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2 if not args.no_pretty else None)
        print(f"\n💾 Guardado en: {args.raw_file}")

if __name__ == "__main__":
    # Modo rápido para Spyder (edita estos defaults si prefieres no usar CLI)
    _MARKET = None         # p.ej. "OMNI_USDT"
    _PID = None            # p.ej. 12345
    _DAYS = None           # p.ej. 7
    _START = None          # epoch s
    _END = None            # epoch s
    _DECORATE = True       # activa por defecto conversión a *_readable
    _TZ = "Europe/Zurich"

    import sys
    sys.argv = [sys.argv[0]]
    if _MARKET: sys.argv += ["--market", _MARKET]
    if _PID is not None: sys.argv += ["--position-id", str(_PID)]
    if _DAYS is not None: sys.argv += ["--days", str(_DAYS)]
    if _START is not None: sys.argv += ["--start", str(_START)]
    if _END is not None: sys.argv += ["--end", str(_END)]
    if _DECORATE: sys.argv += ["--decorate-dates"]
    if _TZ: sys.argv += ["--tz", _TZ]
    sys.argv += ["--limit", "100", "--pages", "1"]
    # sys.argv += ["--raw-file", "wb_history_readable.json"]
    main()
