# adapters/okx.py
# API pública exportada:
#   - fetch_okx_open_positions(...)
#   - fetch_okx_funding_fees(limit: int = 50, ...)
#   - fetch_okx_all_balances(...)
#   - save_okx_closed_positions(db_path: str = "portfolio.db", ...)
#   - purge_okx_partial_closes(db_path: str = "portfolio.db", ...)
#
# Requisitos del proyecto:
# - Shapes EXACTOS para abiertas, funding y balances.
# - Persistencia de cerradas vía db_manager.save_closed_position(...)
# - **Opción recomendada**: SOLO guardar en DB ciclos totalmente cerrados
#   (OKX type in {2: close all, 3: liquidation}).
#   Además, limpiar de la DB cierres parciales previos (types {1,4,5}).
# - Respeta signos: fee_total siempre NEGATIVO; funding con su signo natural.
# - Normaliza símbolo en MAYÚSCULAS (normalize_symbol).
#
# Endpoints OKX usados (REST v5):
# - GET /api/v5/account/positions
# - GET /api/v5/account/positions-history
# - GET /api/v5/account/balance
# - GET /api/v5/account/account-position-risk (opcional, snapshot)
# - GET /api/v5/account/bills  (type=8 => funding fees)
#
# Auth: OK-ACCESS-KEY / OK-ACCESS-SIGN / OK-ACCESS-TIMESTAMP / OK-ACCESS-PASSPHRASE
# Firma: Base64(HMAC_SHA256(timestamp + method + requestPath + body, secret))

from __future__ import annotations

import os
import re
import hmac
import json
import time
import base64
import hashlib
import sqlite3
import argparse
from typing import Any, Dict, List, Optional, Tuple, Iterable
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

import requests

# Persistencia del proyecto
from db_manager import save_closed_position  # guarda una fila en closed_positions

__all__ = [
    "fetch_okx_open_positions",
    "fetch_okx_funding_fees",
    "fetch_okx_all_balances",
    "save_okx_closed_positions",
    "purge_okx_partial_closes",
    # utils de debug
    "debug_preview_okx_closed",
    "debug_dump_okx_opens",
    "debug_dump_okx_funding",
]

# =========================
#   Config / ENV
# =========================
OKX_HOST = os.getenv("OKX_HOST", "https://www.okx.com")
OKX_API_KEY = os.getenv("OKX_API_KEY", "")
OKX_API_SECRET = os.getenv("OKX_API_SECRET", "")
OKX_API_PASSPHRASE = "Sudafric4.12"

# ============== Helpers de impresión ==============
def _log(msg: str) -> None:
    print(msg)

# =========================
#  Normalización de símbolo
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
#     Firmas / Requests
# =========================
def _iso_ts() -> str:
    # formato: 2020-12-08T09:08:57.715Z
    now = datetime.now(timezone.utc)
    return now.isoformat(timespec="milliseconds").replace("+00:00", "Z")

def _require_keys():
    if not (OKX_API_KEY and OKX_API_SECRET and OKX_API_PASSPHRASE):
        raise RuntimeError("Faltan OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE.")

def _sign_okx(ts: str, method: str, request_path: str, body: str) -> str:
    prehash = f"{ts}{method.upper()}{request_path}{body}"
    mac = hmac.new(OKX_API_SECRET.encode(), prehash.encode(), hashlib.sha256).digest()
    return base64.b64encode(mac).decode()

def _okx_request(method: str, path: str, params: Optional[Dict[str, Any]] = None,
                 body: Optional[Dict[str, Any]] = None, timeout=20, retries=3, backoff=0.6) -> Any:
    """
    Hace request firmado a OKX. Para GET, los query params cuentan dentro del path.
    Retrys con backoff simple en 429/5xx.
    """
    _require_keys()
    url = f"{OKX_HOST}{path}"
    q = ""
    if method.upper() == "GET" and params:
        # Los params van en el path para la firma
        from urllib.parse import urlencode
        q = "?" + urlencode(params, doseq=True)
        url = url + q

    body_str = "" if body is None else json.dumps(body, separators=(",", ":"), ensure_ascii=False)

    for attempt in range(1, retries + 1):
        ts = _iso_ts()
        sign = _sign_okx(ts, method, path + q, body_str)
        headers = {
            "OK-ACCESS-KEY": OKX_API_KEY,
            "OK-ACCESS-SIGN": sign,
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": OKX_API_PASSPHRASE,
            "Content-Type": "application/json",
        }
        try:
            resp = requests.request(
                method.upper(), url, headers=headers,
                data=(None if body is None else body_str),
                timeout=timeout,
            )
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(backoff * attempt)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == retries:
                resp.raise_for_status()
            time.sleep(backoff * attempt)
            continue

        data = resp.json()
        if data.get("code") != "0":
            # errores OKX estilo {"code":"60009","msg":"Login failed"}
            if attempt == retries:
                raise RuntimeError(f"OKX error {data.get('code')}: {data.get('msg')}")
            time.sleep(backoff * attempt)
            continue

        return data.get("data", [])

    return []

# ============== utils numéricos ==============

def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _i_ms(ms: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(int(float(ms)) / 1000)
    except Exception:
        return default

# =========================
#      Open Positions
# =========================

def fetch_okx_open_positions(inst_type: str = "SWAP") -> List[Dict[str, Any]]:
    """
    GET /api/v5/account/positions?instType=SWAP
    Shape EXACTO (por fila):
    {
      "exchange": "okx",
      "symbol": "<NORMALIZADO>",
      "side": "long" | "short",
      "size": float (positivo),
      "entry_price": float,
      "mark_price": float,
      "liquidation_price": float | 0.0,
      "notional": float,
      "unrealized_pnl": float,    # SOLO precio (mark - entry) * size, ajustado por side
      "fee": float,               # acumulado; NEGATIVO si costo
      "funding_fee": float,       # acumulado +/-
      "realized_pnl": float       # fee + funding_fee (solo abiertas)
    }
    """
    params = {"instType": inst_type}
    rows = _okx_request("GET", "/api/v5/account/positions", params=params) or []
    out: List[Dict[str, Any]] = []

    _log(f"🔍 DEBUG OKX: {len(rows)} posiciones abiertas")

    for r in rows:
        inst_id = r.get("instId", "")  # p.ej. BTC-USDT-SWAP
        sym = normalize_symbol(inst_id)
        pos_side = (r.get("posSide") or r.get("posSide", "") or "net").lower()
        pos = _f(r.get("pos"))
        side = "long"
        if pos_side in ("long", "short"):
            side = pos_side
        else:
            side = "long" if pos >= 0 else "short"

        margin = _f(r.get("margin"))
        entry = _f(r.get("avgPx") or r.get("openAvgPx"))
        mark = _f(r.get("markPx") or r.get("last"))
        liq = _f(r.get("liqPx") or 0.0)
        size = abs(pos)
        notional = abs(_f(r.get("notionalUsd"))) or (abs(size * mark) if mark > 0 else abs(size * entry))

        # PnL no realizado directo de OKX si viene disponible
        unreal = _f(r.get("upl") or r.get("uplLastPx"))
        if unreal == 0.0 and entry and mark and size:
            unreal = (entry - mark) * size if side == "short" else (mark - entry) * size

        fee_acc = _f(r.get("fee") or 0.0)           # acumulado (negativo costo)
        funding_acc = _f(r.get("fundingFee") or 0.0)
        realized_open = fee_acc + funding_acc

        out.append({
            "exchange": "okx",
            "symbol": sym,
            "side": side,
            "size": size,
            "margin": margin,
            "entry_price": entry,
            "mark_price": mark,
            "liquidation_price": liq or 0.0,
            "notional": notional,
            "unrealized_pnl": unreal,
            "fee": fee_acc,
            "funding_fee": funding_acc,
            "realized_pnl": realized_open,
        })

    _log(f"✅ OKX abiertas normalizadas: {len(out)}")
    return out

# =========================
#    Funding (account)
# =========================

def fetch_okx_funding_fees(limit: int = 50, before: Optional[str] = None, after: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    GET /api/v5/account/bills?type=8
      - type=8 => Funding fee
    Shape EXACTO:
    {
      "exchange": "okx",
      "symbol": "<NORMALIZADO>",
      "income": float,           # + cobro / - pago
      "asset": "USDT"|"USDC"|"USD",
      "timestamp": int,          # epoch ms
      "funding_rate": 0.0,
      "type": "FUNDING_FEE"
    }
    """
    params = {"type": "8", "limit": str(max(1, min(int(limit), 100)))}
    if before: params["before"] = before
    if after:  params["after"] = after

    rows = _okx_request("GET", "/api/v5/account/bills", params=params) or []
    out: List[Dict[str, Any]] = []

    _log(f"🔍 DEBUG OKX: {len(rows)} registros funding")

    for r in rows:
        inst_id = r.get("instId") or ""
        ccy = (r.get("ccy") or "USDT").upper()
        ts_ms = int(r.get("ts")) if r.get("ts") else None
        sym = normalize_symbol(inst_id) if inst_id else ""
        # En bills, 'balChg' es el cambio neto de balance. Fallbacks seguros.
        income = _f(r.get("balChg") if r.get("balChg") not in ("", None)
                    else r.get("amt") if r.get("amt") not in ("", None)
                    else r.get("pnl"))

        out.append({
            "exchange": "okx",
            "symbol": sym,
            "income": income,          # + cobro / - pago
            "asset": ccy,
            "timestamp": int(ts_ms) if ts_ms else None,
            "funding_rate": 0.0,
            "type": "FUNDING_FEE",
        })

    return out

# =========================
#        Balances
# =========================

def fetch_okx_all_balances(db_path: str = "portfolio.db") -> Dict[str, Any]:
    """
    GET /api/v5/account/balance
    Devuelve un único dict con los buckets solicitados:
    {
      "exchange": "okx",
      "equity": float,            # totalEq (USD)
      "balance": float,           # availEq (USD)
      "unrealized_pnl": float,    # upl (USD)
      "initial_margin": float,    # imr (USD)
      "spot": 0.0,
      "margin": 0.0,
      "futures": float            # mapea derivados como en el front
    }
    """
    rows = _okx_request("GET", "/api/v5/account/balance") or []
    if not rows:
        return {
            "exchange": "okx",
            "equity": 0.0,
            "balance": 0.0,
            "unrealized_pnl": 0.0,
            "initial_margin": 0.0,
            "spot": 0.0,
            "margin": 0.0,
            "futures": 0.0,
        }

    acc = rows[0] or {}
    equity = _f(acc.get("totalEq"))
    avail_eq = _f(acc.get("availEq"))
    upl = _f(acc.get("upl"))
    imr = _f(acc.get("imr"))

    result = {
        "exchange": "okx",
        "equity": equity,
        "balance": avail_eq,
        "unrealized_pnl": upl,
        "initial_margin": imr,
        "spot": 0.0,
        "margin": 0.0,
        "futures": equity,  # como hacen Aden/Extended: todo el equity a derivados
    }
    _log(f"💼 OKX equity total: {equity:.2f}")
    return result

# =========================
#     Closed Positions
# =========================

# Mapa de tipos OKX relevantes
_CLOSE_ALL_TYPES = {"2", "3"}      # 2: close all, 3: liquidation
_PARTIAL_TYPES   = {"1", "4", "5"}  # 1: partial close, 4: partial liquidation, 5: ADL not fully closed


def _normalize_closed_from_history(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Mapea una fila de /account/positions-history al input de DB:
    {
      exchange, symbol, side, size, entry_price, close_price,
      open_time, close_time, realized_pnl, funding_total, fee_total, pnl,
      notional, leverage, initial_margin, liquidation_price
    }
    """
    try:
        inst_id = row.get("instId", "")
        sym = normalize_symbol(inst_id)
        side = (row.get("posSide") or row.get("direction") or "net").lower()
        if side not in ("long", "short"):
            side = "long"

        # tamaños/precios
        size = abs(_f(row.get("closeTotalPos") or row.get("openMaxPos")))
        entry = _f(row.get("openAvgPx") or row.get("nonSettleAvgPx") or 0.0)
        close = _f(row.get("closeAvgPx") or 0.0)

        # pnl
        realized = _f(row.get("realizedPnl"))
        fee = _f(row.get("fee"))
        funding = _f(row.get("fundingFee"))
        price_pnl = _f(row.get("pnl"))  # “Profit and loss (excluding the fee)”

        # tiempos en SEGUNDOS
        open_s = _i_ms(row.get("cTime"))
        close_s = _i_ms(row.get("uTime"))

        lev = _f(row.get("lever") or 0.0) or None
        notional = abs(size * entry) if entry > 0 else abs(size * close)
        initial_margin = (notional / lev) if lev and notional else None

        return {
            "exchange": "okx",
            "symbol": sym,
            "side": side,
            "size": size,
            "entry_price": entry,
            "close_price": close,
            "open_time": open_s,
            "close_time": close_s,
            "realized_pnl": realized,
            "funding_total": funding,
            "fee_total": fee,
            "pnl": price_pnl,
            "notional": notional,
            "leverage": lev,
            "initial_margin": initial_margin,
            "liquidation_price": None,
        }
    except Exception as e:
        _log(f"⚠️ normalize_closed_from_history error: {e}")
        return None


def _fetch_positions_history(inst_type: str = "SWAP", limit: int = 100) -> List[Dict[str, Any]]:
    params = {"instType": inst_type, "limit": str(max(1, min(int(limit), 100)))}
    rows = _okx_request("GET", "/api/v5/account/positions-history", params=params) or []
    _log(f"🔍 DEBUG OKX: {len(rows)} filas de positions-history")
    return rows


def _split_history_by_type(rows: Iterable[Dict[str, Any]]):
    allowed, partial, ignored = [], [], []
    for r in rows:
        t = str(r.get("type") or "").strip()
        if t in _CLOSE_ALL_TYPES:
            allowed.append(r)
        elif t in _PARTIAL_TYPES:
            partial.append(r)
        else:
            # desconocidos o vacíos
            ignored.append(r)
    return allowed, partial, ignored


def purge_okx_partial_closes(db_path: str = "portfolio.db", inst_type: str = "SWAP", limit: int = 100) -> int:
    """
    Elimina de SQLite cualquier fila que corresponda a cierres **parciales** de OKX
    (types {1,4,5}) previamente guardados en 'closed_positions'.
    Coincidencia por (exchange, symbol, close_time).
    Devuelve el número de filas borradas.
    """
    if not os.path.exists(db_path):
        _log(f"❌ Database not found: {db_path}")
        return 0

    hist = _fetch_positions_history(inst_type=inst_type, limit=limit)
    _, partial_rows, _ = _split_history_by_type(hist)

    if not partial_rows:
        return 0

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    removed = 0
    for r in partial_rows:
        sym = normalize_symbol(r.get("instId", ""))
        close_s = _i_ms(r.get("uTime"))
        if not sym or not close_s:
            continue
        cur.execute(
            "DELETE FROM closed_positions WHERE exchange=? AND symbol=? AND close_time=?",
            ("okx", sym, close_s),
        )
        removed += cur.rowcount
    conn.commit()
    conn.close()
    if removed:
        _log(f"🧹 OKX parciales eliminados de DB: {removed}")
    return removed


def fetch_okx_closed_positions(inst_type: str = "SWAP", limit: int = 100) -> List[Dict[str, Any]]:
    """
    GET /api/v5/account/positions-history
    Retorna filas normalizadas aptas para DB **FILTRADAS** a cierres completos:
    - type in {2 (close all), 3 (liquidation)}.
    """
    rows = _fetch_positions_history(inst_type=inst_type, limit=limit)
    allowed, _, _ = _split_history_by_type(rows)

    out: List[Dict[str, Any]] = []
    for r in allowed:
        mapped = _normalize_closed_from_history(r)
        if mapped:
            out.append(mapped)
    _log(f"✅ OKX cerradas normalizadas (solo completas): {len(out)}")
    return out


def save_okx_closed_positions(db_path: str = "portfolio.db", days: int = 30, debug: bool = False,
                              inst_type: str = "SWAP", limit: int = 100,
                              cleanup_partials: bool = True) -> int:
    """
    Guarda en SQLite **solo** cierres completos de OKX.
    - Filtra por type in {2: close all, 3: liquidation}.
    - (opcional) Elimina antes de guardar los cierres parciales previos (types {1,4,5}).
    - Deduplicación por (exchange, symbol, close_time).
    """
    if not os.path.exists(db_path):
        _log(f"❌ Database not found: {db_path}")
        return 0

    # 1) Limpieza de parciales guardados por error (si se desea)
    if cleanup_partials:
        purge_okx_partial_closes(db_path=db_path, inst_type=inst_type, limit=limit)

    # 2) Cierres completos filtrados
    closed = fetch_okx_closed_positions(inst_type=inst_type, limit=limit)
    if not closed:
        _log("⚠️ No se obtuvieron posiciones cerradas completas de OKX.")
        return 0

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    saved = 0
    skipped = 0

    for pos in closed:
        try:
            cur.execute(
                """
                SELECT COUNT(*) FROM closed_positions
                WHERE exchange=? AND symbol=? AND close_time=?
                """,
                (pos["exchange"], pos["symbol"], pos["close_time"]),
            )
            if cur.fetchone()[0]:
                skipped += 1
                continue

            # save_closed_position aplica reglas: fee_total negativa, pnl_percent, apr, etc.
            save_closed_position({
                "exchange": pos["exchange"],
                "symbol": pos["symbol"],
                "side": pos["side"],
                "size": pos["size"],
                "entry_price": pos["entry_price"],
                "close_price": pos["close_price"],
                "open_time": pos["open_time"],
                "close_time": pos["close_time"],
                "realized_pnl": pos["realized_pnl"],
                "funding_total": pos.get("funding_total", 0.0),
                "fee_total": pos.get("fee_total", 0.0),
                "pnl": pos.get("pnl", 0.0),
                "notional": pos.get("notional", 0.0),
                "leverage": pos.get("leverage"),
                "initial_margin": pos.get("initial_margin"),
                "liquidation_price": pos.get("liquidation_price"),
            })
            saved += 1
            if debug:
                price_pnl = pos.get("realized_pnl", 0.0) - pos.get("funding_total", 0.0) - pos.get("fee_total", 0.0)
                _log(f"   💾 OKX saved {pos['symbol']} | price_pnl≈{price_pnl:.6f}")
        except Exception as e:
            _log(f"⚠️ Error guardando {pos.get('symbol')}: {e}")

    conn.close()
    _log(f"✅ OKX guardadas: {saved} | omitidas (duplicadas): {skipped}")
    return saved

# =========================
#         Debug
# =========================

def debug_preview_okx_closed(days: int = 3, symbol: Optional[str] = None, inst_type: str = "SWAP", limit: int = 100) -> None:
    rows = _fetch_positions_history(inst_type=inst_type, limit=limit)
    allowed, partial, ignored = _split_history_by_type(rows)
    _log(f"📦 DEBUG: total={len(rows)} | completas={len(allowed)} | parciales={len(partial)} | otras={len(ignored)}")
    for r in allowed[:50]:
        if symbol and normalize_symbol(r.get("instId", "")) != normalize_symbol(symbol):
            continue
        mapped = _normalize_closed_from_history(r) or {}
        price_pnl = (mapped.get("realized_pnl") or 0.0) - (mapped.get("funding_total") or 0.0) - (mapped.get("fee_total") or 0.0)
        _log(
            f"🔎 {mapped.get('symbol')} type={r.get('type')} side={mapped.get('side')} size={mapped.get('size')} "
            f"entry={mapped.get('entry_price')} close={mapped.get('close_price')} "
            f"realized={mapped.get('realized_pnl')} fee={mapped.get('fee_total')} funding={mapped.get('funding_total')} "
            f"price_pnl≈{price_pnl:.6f} open={mapped.get('open_time')} close={mapped.get('close_time')}"
        )


def debug_dump_okx_opens() -> List[Dict[str, Any]]:
    params = {"instType": "SWAP"}
    return _okx_request("GET", "/api/v5/account/positions", params=params)


def debug_dump_okx_funding(limit: int = 50) -> List[Dict[str, Any]]:
    params = {"type": "8", "limit": str(limit)}
    return _okx_request("GET", "/api/v5/account/bills", params=params)

# =========================
#          CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OKX adapter debug/CLI")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Solo imprimir (default).")
    parser.add_argument("--save-closed", action="store_true", help="Guardar cerradas en portfolio.db")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--funding", type=int, default=0, help="Mostrar N registros de funding")
    parser.add_argument("--opens", action="store_true", help="Mostrar abiertas normalizadas y RAW")
    parser.add_argument("--limit", type=int, default=100, help="Límite de history (máx. 100)")
    args = parser.parse_args()

    if args.opens:
        raw = debug_dump_okx_opens()
        _log(f"🧾 RAW positions: {len(raw)}")
        norm = fetch_okx_open_positions()
        _log(json.dumps(norm[:5], indent=2))

    if args.funding > 0:
        raw_f = debug_dump_okx_funding(limit=args.funding)
        _log(f"🧾 RAW funding: {len(raw_f)}")
        norm_f = fetch_okx_funding_fees(limit=args.funding)
        _log(json.dumps(norm_f[:5], indent=2))

    # Preview de history con breakdown por tipo
    debug_preview_okx_closed(days=args.days, limit=args.limit)

    if args.save_closed:
        _log("🧹 Limpiando parciales y guardando cierres completos en DB...")
        save_okx_closed_positions("portfolio.db", days=args.days, debug=True, limit=args.limit, cleanup_partials=True)

