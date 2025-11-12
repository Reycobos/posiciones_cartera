# adapters/mexc.py
from __future__ import annotations
import os, time, hmac, hashlib, json, math, sqlite3, re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, quote
import os
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timezone

import requests
# === Añade cerca de los imports / helpers ===
_MARK_CACHE = {}          # { "SYMBOL": (price_float, ts_seconds) }
_MARK_TTL   = 3.0         # segundos: cache cortita para la misma pasada
__all__ = [
    "fetch_mexc_open_positions",
    "fetch_mexc_funding_fees",
    "fetch_mexc_all_balances",
    "save_mexc_closed_positions",
    # util/debug
    # "debug_preview_mexc_closed",
]
# #====== Imports para prints
# from pp import (
#     p_closed_debug_header, p_closed_debug_count, p_closed_debug_norm_size,
#     p_closed_debug_prices, p_closed_debug_pnl, p_closed_debug_times, p_closed_debug_normalized,
#     p_open_summary, p_open_block,
#     p_funding_fetching, p_funding_count,
#     p_balance_equity
# )
# #===========================
# =========================
# Config & credenciales
# =========================
MEXC_BASE_URL = os.getenv("MEXC_BASE_URL", "https://contract.mexc.com")
MEXC_API_KEY = "mx0vglC1hxM1TTXJiO"
MEXC_API_SECRET = "294cf219e5bd407e82bbea19dee1baa5"
MEXC_RECV_WINDOW = os.getenv("MEXC_RECV_WINDOW", "10000")  # ms, <= 60000 recomendado

# =========================
# Normalización de símbolo (Regla A)
# =========================
# 👇 Añade este diccionario cerca de normalize_symbol
SPECIAL_SYMBOL_MAP = {
    "OPENLEDGER": "OPEN",   # Unifica el ticker a OPEN
    # agrega aquí otros alias si los necesitas...
}

def normalize_symbol(sym: str) -> str:
    if not sym:
        return ""
    s = sym.upper().strip()
    s = re.sub(r'^PERP_', '', s)
    s = re.sub(r'(_|-)?(USDT|USDC|USD|PERP)$', '', s)   # quita sufijos + separador
    s = re.sub(r'[_-]+$', '', s)                        # guiones finales
    base = re.split(r'[_-]', s)[0]                      # primera parte (KAITO_USDC -> KAITO)
    # 👇 Aplica alias especiales
    base = SPECIAL_SYMBOL_MAP.get(base, base)
    return base
# =========================
# Helpers HTTP / firma / backoff
# =========================




def _now_ms() -> int:
    return int(time.time() * 1000)

def _has_creds() -> bool:
    return bool(MEXC_API_KEY and MEXC_API_SECRET)

def _param_str_for_get(params: Optional[Dict[str, Any]]) -> str:
    """GET/DELETE → concatenación en orden lexicográfico k=v con URL-encode y &."""
    if not params:
        return ""
    # params con valores None no participan
    items = [(k, "" if v is None else str(v)) for k, v in params.items() if v is not None]
    items.sort(key=lambda kv: kv[0])
    return "&".join(f"{kv[0]}={quote(kv[1], safe='')}" for kv in items)

def _mexc_signature(param_str: str, ts: str) -> str:
    """
    Regla oficial:
      sign_target = accessKey + reqTime + requestParamString
      signature = HMAC_SHA256(secret, sign_target) → hex lower
    """
    target = f"{MEXC_API_KEY}{ts}{param_str}"
    digest = hmac.new(MEXC_API_SECRET.encode("utf-8"), target.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest

def _headers(ts: str, signature: Optional[str]) -> Dict[str, str]:
    hdrs = {
        "Content-Type": "application/JSON",
        "Request-Time": ts,
        "ApiKey": MEXC_API_KEY or "",
    }
    if MEXC_RECV_WINDOW:
        hdrs["Recv-Window"] = MEXC_RECV_WINDOW
    if signature:
        hdrs["Signature"] = signature
    return hdrs

def _request(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    private: bool = False,
    timeout: int = 25,
    max_retries: int = 3,
    retry_backoff: float = 0.75
) -> Dict[str, Any]:
    """
    Cliente con firma y backoff. GET params se pasan también en la URL.
    POST debe enviarse como JSON (no usado aquí).
    """
    url = f"{MEXC_BASE_URL}{path}"
    params = dict(params or {})

    # Firma
    ts = str(_now_ms())
    sign_str = ""
    if private:
        if method.upper() in ("GET", "DELETE"):
            sign_str = _param_str_for_get(params)
        else:
            # POST → JSON string (sin ordenar) (no lo usamos por ahora)
            sign_str = json.dumps(params, separators=(",", ":"))
        sig = _mexc_signature(sign_str, ts)
        headers = _headers(ts, sig)
    else:
        headers = _headers(ts, None)

    # Query en URL para GET/DELETE
    if method.upper() in ("GET", "DELETE") and params:
        qs = urlencode(params)
        url = f"{url}?{qs}"

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            if method.upper() == "GET":
                r = requests.get(url, headers=headers, timeout=timeout)
            elif method.upper() == "DELETE":
                r = requests.delete(url, headers=headers, timeout=timeout)
            else:
                r = requests.post(url, headers=headers, data=json.dumps(params) if params else "{}", timeout=timeout)
            r.raise_for_status()
            data = r.json() if r.text else {}
            # Protocolo MEXC common
            if isinstance(data, dict) and not data.get("success", True):
                # algunos endpoints devuelven success=false + code/message
                raise RuntimeError(f"MEXC error: code={data.get('code')} msg={data.get('message')}")
            return data
        except Exception as e:
            last_err = e
            if attempt >= max_retries:
                raise
            time.sleep(retry_backoff * attempt)
    # si sale del loop
    raise last_err or RuntimeError("MEXC request failed")

# ============
# Precios mark
# ============


def _get_mark_price_fast(symbol: str) -> Optional[float]:
    """
    Mark/last price con política ultra-rápida:
    - Una sola llamada: /api/v1/contract/ticker?symbol=SYMBOL
    - timeout bajo y sin reintentos (queremos velocidad > exactitud)
    - cache efímera para evitar llamadas repetidas dentro de la misma petición
    """
    sym = (symbol or "").upper()
    now = time.time()
    cached = _MARK_CACHE.get(sym)
    if cached and now - cached[1] < _MARK_TTL:
        return cached[0]

    price = None
    try:
        data = _request(
            "GET",
            "/api/v1/contract/ticker",
            {"symbol": sym},
            private=False,
            timeout=2,          # << muy agresivo
            max_retries=1       # << sin reintentos
        )
        d = data.get("data")
        if isinstance(d, dict):
            for k in ("fairPrice", "lastPrice", "last"):
                if d.get(k) is not None:
                    price = float(d[k]); break
        elif isinstance(d, list) and d:
            x = d[0]
            for k in ("fairPrice", "lastPrice", "last"):
                if x.get(k) is not None:
                    price = float(x[k]); break
    except Exception:
        price = None

    if price is not None:
        _MARK_CACHE[sym] = (price, now)
    return price


def _pnl_unrealized(entry: float, mark: float, size: float, side: str) -> float:
    if any(math.isnan(x) for x in (entry, mark, size)):
        return 0.0
    if side == "short":
        # (entry - mark) * size
        return (entry - mark) * size
    # long
    return (mark - entry) * size

def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default

# =========================
# BALANCES (Regla D exacta)
# =========================
def fetch_mexc_all_balances() -> Dict[str, Any]:
    """
    GET api/v1/private/account/assets

    Respuesta(s) vistas:
    - data: { currency, positionMargin, frozenBalance, availableBalance, cashBalance, equity, unrealized }
    - o data: [ { ... }, ... ] (por currency)
    Mapeo UI:
      {
        "exchange": "mexc",
        "equity": float,
        "balance": float,           # saldo utilizable TOTAL
        "unrealized_pnl": float,    # 0.0 si no aplica
        "initial_margin": float | 0.0,
        "spot": 0.0, "margin": 0.0, "futures": float
      }
    """
    if not _has_creds():
        return {
            "exchange": "mexc",
            "equity": 0.0,
            "balance": 0.0,
            "unrealized_pnl": 0.0,
            "initial_margin": 0.0,
            "spot": 0.0,
            "margin": 0.0,
            "futures": 0.0,
        }
    try:
        resp = _request("GET", "/api/v1/private/account/assets", private=True)
        data = resp.get("data")
        rows = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
        eq = bal = unrl = init_m = 0.0
        futures_bucket = 0.0
        for row in rows:
            cur = (row.get("currency") or "").upper()
            # contrato MEXC suele ser USDT; sumo solo stables visibles
            if cur in ("USDT", "USDC", "USD", ""):
                eq      += _safe_float(row.get("equity", 0))
                bal     += _safe_float(row.get("availableBalance", row.get("cashBalance", 0)))
                unrl    += _safe_float(row.get("unrealized", 0))
                init_m  += _safe_float(row.get("positionMargin", 0))
        futures_bucket = eq
        return {
            "exchange": "mexc",
            "equity": float(eq),
            "balance": float(bal),
            "unrealized_pnl": float(unrl),
            "initial_margin": float(init_m),
            "spot": 0.0,
            "margin": 0.0,
            "futures": float(futures_bucket),
        }
    except Exception as e:
        print(f"❌ MEXC balances error: {e}")
        return {
            "exchange": "mexc",
            "equity": 0.0,
            "balance": 0.0,
            "unrealized_pnl": 0.0,
            "initial_margin": 0.0,
            "spot": 0.0,
            "margin": 0.0,
            "futures": 0.0,
        }

# =========================
# OPEN POSITIONS (Regla B exacta)
# =========================
def fetch_mexc_open_positions(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    GET api/v1/private/position/open_positions
    Campos de ejemplo:
      positionId, symbol, positionType(1 long/2 short), holdVol, openAvgPrice,
      closeAvgPrice, liquidatePrice, im, holdFee, realised, leverage, updateTime

    Salida EXACTA por fila (front):
      {
        "exchange": "mexc",
        "symbol": "<normalize_symbol>",
        "side": "long" | "short",
        "size": float,
        "entry_price": float,
        "mark_price": float,
        "liquidation_price": float | 0.0,
        "notional": float,
        "unrealized_pnl": float,    # SOLO precio
        "fee": float,               # acumulado fees (negativo si costo) - no disponible → 0.0
        "funding_fee": float,       # + cobro / - pago (holdFee)
        "realized_pnl": float       # = fee + funding_fee (abiertas)
      }
    """
    if not _has_creds():
        return []
    try:
        params = {}
        if symbol:
            params["symbol"] = symbol
        resp = _request("GET", "/api/v1/private/position/open_positions", params=params, private=True)
        rows = resp.get("data", []) if isinstance(resp, dict) else []
        out: List[Dict[str, Any]] = []
        for r in rows:
            raw_sym = r.get("symbol", "")
            side = "long" if int(r.get("positionType", 1)) == 1 else "short"
            size = _safe_float(r.get("holdVol", 0))
            entry = _safe_float(r.get("openAvgPrice", r.get("holdAvgPrice", 0)))
            # mark = _get_mark_price(raw_sym) or _safe_float(r.get("closeAvgPrice", entry)) or entry
            mark = _get_mark_price_fast(raw_sym)
            if mark is None:
                # fallback ultrarrápido: si la API no da mark, no nos bloqueamos
                mark = _safe_float(r.get("closeAvgPrice", 0)) or entry
            liq  = _safe_float(r.get("liquidatePrice", 0))
            notional = abs(size) * entry
            funding = _safe_float(r.get("holdFee", 0))  # holding fee = funding (+ / -)
            fee_acc = 0.0  # MEXC no expone fee acumulada por pos en este endpoint
            unreal = _pnl_unrealized(entry, mark, abs(size), side)
            realized_open = fee_acc + funding
            out.append({
                "exchange": "mexc",
                "symbol": normalize_symbol(raw_sym),
                "side": side,
                "size": abs(size),
                "entry_price": entry,
                "mark_price": mark,
                "liquidation_price": liq,
                "notional": notional,
                "unrealized_pnl": unreal,
                "fee": fee_acc,                # mantener NEGATIVO si costo (0.0 aquí)
                "funding_fee": funding,
                "realized_pnl": realized_open,
            })
        return out
    except Exception as e:
        print(f"❌ MEXC open positions error: {e}")
        return []

# =========================
# FUNDING HISTORY (Regla C exacta)
# =========================
def fetch_mexc_funding_fees(limit: int = 1000,
                            symbol: Optional[str] = None,
                            since: Optional[int] = None,
                            max_pages: int = 100) -> List[Dict[str, Any]]:
    """
    GET /api/v1/private/position/funding_records (rate: 20 req / 2s)
    - Si 'since' está definido (ms epoch), pagina hacia atrás hasta cruzar ese corte temporal.
    - Usa page_size=100 (máximo permitido por API) para minimizar llamadas.
    - 'limit' es un límite duro por conteo, por si quieres recortar.
    """
    if not _has_creds():
        return []
    acc: List[Dict[str, Any]] = []
    try:
        cutoff = int(since) if since else 0
        page_num = 1
        page_size = 100  # máximo soportado por API

        while page_num <= max_pages:
            params = {"page_num": page_num, "page_size": page_size}
            if symbol:
                params["symbol"] = symbol

            resp = _request("GET", "/api/v1/private/position/funding_records",
                            params=params, private=True)
            d = resp.get("data", {}) if isinstance(resp, dict) else {}
            rows = d.get("resultList", []) or []
            if not rows:
                break

            # Se asume orden descendente por settleTime
            for it in rows:
                ts = int(_safe_float(it.get("settleTime", 0)))
                if cutoff and ts < cutoff:
                    # Es más viejo que el corte → saltamos este item
                    continue
                acc.append({
                    "exchange": "mexc",
                    "symbol": normalize_symbol(it.get("symbol", "")),
                    "income": _safe_float(it.get("funding", 0)),
                    "asset": "USDT",
                    "timestamp": ts,  # ms
                    "funding_rate": _safe_float(it.get("rate", 0)),
                    "type": "FUNDING_FEE",
                })
                if len(acc) >= int(limit or 10**9):
                    return acc

            # Si la última fila de la página ya es anterior al corte temporal, no hace falta seguir
            last_ts = int(_safe_float(rows[-1].get("settleTime", 0)))
            if cutoff and last_ts < cutoff:
                break

            total_page = int(_safe_float(d.get("totalPage", 0)))
            if total_page and page_num >= total_page:
                break

            page_num += 1
            # (opcional) respetar un pequeño backoff para no chocar rate-limit
            # time.sleep(0.05)

        return acc
    except Exception as e:
        print(f"❌ MEXC funding error: {e}")
        return acc

#=========Closed positions

def _adjust_size_scale(size_raw: float, entry: float, close: float,
                       target_price_component: float, side: str,
                       rel_tol: float = 0.05):
    """
    Ajusta size por potencias de 10 para aproximar el PnL de precio
    al 'target_price_component' (≈ realised - funding - fee_explicit).
    Devuelve (size_ajustado, factor_aplicado, error_relativo).
    """
    if size_raw <= 0 or entry == close or target_price_component == 0:
        return size_raw, 1.0, 1.0

    tgt = abs(target_price_component)
    candidates = [1000, 100, 10, 1, 0.1, 0.01, 0.001]
    best = (float("inf"), size_raw, 1.0)
    for f in candidates:
        s = size_raw * f
        p = abs(_price_pnl(side, entry, close, s))
        err = abs(p - tgt) / max(1.0, tgt)
        if err < best[0]:
            best = (err, s, f)
    return best[1], best[2], best[0]

def _price_pnl(side: str, entry: float, close: float, size: float) -> float:
    side = (side or "").lower()
    if side == "short":
        return (entry - close) * abs(size)
    return (close - entry) * abs(size)

def _row_to_closed_payload(r: Dict[str, Any]) -> Dict[str, Any]:
    raw_sym = r.get("symbol", "")

    # Precios y lado
    entry = _safe_float(r.get("openAvgPrice", r.get("holdAvgPrice", 0)))
    close = _safe_float(r.get("closeAvgPrice", entry))
    side = "long" if int(_safe_float(r.get("positionType", 1))) == 1 else "short"

    # ✅ CALCULAR SIZE USANDO LA LÓGICA DEL DEBUG - PRIORIDAD CORREGIDA
    size = 0.0
    
    # Obtener los valores base
    close_vol = _safe_float(r.get("closeVol", 0.0))
    hold_vol = _safe_float(r.get("holdVol", 0.0))
    
    # Funding y realized TOTAL (neto)
    funding_total = _safe_float(r.get("holdFee", 0))
    realized_total = _safe_float(r.get("realised", 0))

    # Buscar fee explícito
    fee_explicit = None
    for k in ("closeFee", "fee", "poundage", "realisedFee", "commission"):
        if r.get(k) is not None:
            fee_explicit = _safe_float(r.get(k))
            break

    # Calcular target_price_component (≈ realised - funding - fee_explicit)
    target_price_component = realized_total - funding_total - (fee_explicit if fee_explicit is not None else 0.0)

    # Elegir el size base: preferir closeVol, luego holdVol
    size_base = close_vol if close_vol > 0 else hold_vol
    
    # Aplicar ajuste de escala (la clave del debug)
    if size_base > 0 and abs(target_price_component) > 1e-8 and abs(close - entry) > 1e-8:
        size_scaled, scale_factor, error = _adjust_size_scale(
            size_base, entry, close, target_price_component, side, rel_tol=0.05
        )
        
        # Usar el size escalado si el error es razonable
        if error < 0.2:  # 20% de tolerancia
            size = size_scaled
            print(f"         📏 Usando size_after_scale: {size} (factor: {scale_factor}, error: {error:.2%})")
        else:
            size = size_base
            print(f"         📏 Usando size base (error muy alto): {size} (error: {error:.2%})")
    else:
        # Si no podemos calcular el scale, usar el size base
        size = size_base
        if size > 0:
            print(f"         📏 Usando size base: {size}")
    
    # Si aún no tenemos tamaño válido, usar reconstrucción básica
    if size <= 0:
        diff = abs(close - entry)
        if diff > 1e-8 and abs(target_price_component) > 1e-8:
            size = abs(target_price_component) / diff
            print(f"         🔧 Size reconstruido básico: {size}")

    size = abs(size)

    # PnL de precio con el size final
    price_pnl = (entry - close) * size if side == "short" else (close - entry) * size

    # Fees por identidad: realized = price_pnl + funding + fee_total
    fee_total = realized_total - price_pnl - funding_total
    if abs(fee_total) < 1e-8:
        fee_total = 0.0

    # Notional a entry
    notional = size * entry if (entry and size) else 0.0

    # tiempos ms → s
    open_ms = int(_safe_float(r.get("createTime", 0)))
    close_ms = int(_safe_float(r.get("updateTime", 0)))
    open_s = int(open_ms / 1000) if open_ms else 0
    close_s = int(close_ms / 1000) if close_ms else 0

    lev = _safe_float(r.get("leverage", 0)) or None
    liq = _safe_float(r.get("liquidatePrice", 0)) or None

    return {
        "exchange": "mexc",
        "symbol": normalize_symbol(raw_sym),
        "side": side,
        "size": float(size),                 # 👈 Tamaño corregido con lógica del debug
        "entry_price": float(entry),
        "close_price": float(close),
        "open_time": open_s,
        "close_time": close_s,

        "realized_pnl": float(realized_total),
        "funding_total": float(funding_total),
        "fee_total": float(fee_total),

        "pnl": float(price_pnl),            
        "notional": float(notional),
        "leverage": float(lev) if lev else None,
        "liquidation_price": float(liq) if liq else None,
    }



def _iter_history_positions(days: int = 60, symbol: Optional[str] = None, max_pages: int = 10) -> List[Dict[str, Any]]:
    """GET api/v1/private/position/list/history_positions con paginación defensiva."""
    acc: List[Dict[str, Any]] = []
    page_num = 1
    page_size = 100
    cutoff_ms = _now_ms() - days * 24 * 60 * 60 * 1000 if days and days > 0 else 0
    while page_num <= max_pages:
        params = {"page_num": page_num, "page_size": page_size}
        if symbol:
            params["symbol"] = symbol
        resp = _request("GET", "/api/v1/private/position/list/history_positions", params=params, private=True)
        
        # DEBUG TEMPORAL: Ver estructura de la respuesta
        if page_num == 1 and not acc:
            print("🔍 DEBUG: Estructura de respuesta MEXC:")
            print(f"   Tipo de data: {type(resp.get('data'))}")
            if isinstance(resp.get('data'), dict):
                sample_row = resp['data'].get('resultList', [{}])[0] if resp['data'].get('resultList') else {}
                print(f"   Campos disponibles: {list(sample_row.keys())}")
                if 'size_after_scale' in sample_row:
                    print(f"   ✅ size_after_scale disponible: {sample_row['size_after_scale']}")
                else:
                    print("   ❌ size_after_scale NO disponible")
        # Formatos posibles: {"success":true,"data":[...]} o data:{resultList:[...]}
        d = resp.get("data", [])
        rows = []
        if isinstance(d, list):
            rows = d
        elif isinstance(d, dict):
            rows = d.get("resultList", [])
        for r in rows:
            if cutoff_ms and int(_safe_float(r.get("updateTime", 0))) < cutoff_ms:
                continue
            acc.append(r)
        # fin paginación
        total_page = 0
        if isinstance(d, dict):
            total_page = int(_safe_float(d.get("totalPage", 0)))
        if total_page and page_num >= total_page:
            break
        if not rows:
            break
        page_num += 1
    return acc
 

# def _iter_history_positions(days: int = 60, symbol: Optional[str] = None, max_pages: int = 10) -> List[Dict[str, Any]]:
#     """GET api/v1/private/position/list/history_positions con paginación defensiva."""
#     acc: List[Dict[str, Any]] = []
#     page_num = 1
#     page_size = 100
#     cutoff_ms = _now_ms() - days * 24 * 60 * 60 * 1000 if days and days > 0 else 0
#     while page_num <= max_pages:
#         params = {"page_num": page_num, "page_size": page_size}
#         if symbol:
#             params["symbol"] = symbol
#         resp = _request("GET", "/api/v1/private/position/list/history_positions", params=params, private=True)
#         # Formatos posibles: {"success":true,"data":[...]} o data:{resultList:[...]}
#         d = resp.get("data", [])
#         rows = []
#         if isinstance(d, list):
#             rows = d
#         elif isinstance(d, dict):
#             rows = d.get("resultList", [])
#         for r in rows:
#             if cutoff_ms and int(_safe_float(r.get("updateTime", 0))) < cutoff_ms:
#                 continue
#             acc.append(r)
#         # fin paginación
#         total_page = 0
#         if isinstance(d, dict):
#             total_page = int(_safe_float(d.get("totalPage", 0)))
#         if total_page and page_num >= total_page:
#             break
#         if not rows:
#             break
#         page_num += 1
#     return acc


def save_mexc_closed_positions(db_path: str = "portfolio.db", days: int = 10, debug: bool = True) -> int:
    try:
        import sqlite3
        from db_manager import save_closed_position

        rows = _iter_history_positions(days=days)
        if not rows:
            if debug:
                print("⚠️ No se encontraron posiciones cerradas en MEXC.")
            return 0

        if not os.path.exists(db_path):
            print(f"❌ Database not found: {db_path}")
            return 0

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        saved = 0
        replaced = 0

        for r in rows:
            try:
                pos = _row_to_closed_payload(r)

                # ¿Existe ya? (clave natural)
                cur.execute("""
                    SELECT id FROM closed_positions
                    WHERE exchange = ? AND symbol = ? AND close_time = ?
                """, (pos["exchange"], pos["symbol"], pos["close_time"]))
                row = cur.fetchone()

                if row:
                    # 🔁 Reemplazar para aplicar el size corregido
                    cur.execute("""
                        DELETE FROM closed_positions
                        WHERE id = ?
                    """, (row[0],))
                    conn.commit()
                    replaced += 1
                    if debug:
                        print(f"🔁 Reemplazando duplicado: {pos['symbol']} close_time={pos['close_time']}")

                # Insert centralizado (db_manager recalcula métricas)
                save_closed_position(pos)
                saved += 1

                if debug:
                    src = "size_after_scale" if r.get("size_after_scale") not in (None, "", 0, "0") else \
                          ("closeVol" if r.get("closeVol") else ("holdVol" if r.get("holdVol") else "reconstructed"))
                    print(f"✅ MEXC cerrada: {pos['symbol']} {pos['side']} size={pos['size']} (src={src}) t={pos['close_time']}")

            except Exception as e:
                print(f"❌ Error guardando posición MEXC {r.get('symbol', '')}: {e}")
                continue

        conn.close()
        print(f"✅ MEXC guardadas: {saved} | reemplazadas: {replaced}")
        return saved

    except Exception as e:
        print(f"❌ MEXC closed positions error: {e}")
        return 0




# =========================
# Smoke tests CLI
# =========================

# === DEBUG TOGGLE ===
MEXC_DEBUG = str(os.getenv("MEXC_DEBUG", "0")).lower() in ("1","true","yes")
def _dbg(*a):
    if MEXC_DEBUG:
        print("[MEXC DEBUG]", *a)

def debug_dump_mexc_history_raw(days: int = 60, symbol: Optional[str] = None, limit: int = 50):
    """
    Imprime filas crudas de /position/list/history_positions con todos los campos relevantes
    y compara tamaños: closeVol, holdVol y reconstrucción por PnL.
    """
    rows = _iter_history_positions(days=days, symbol=symbol, max_pages=200)
    count = 0
    for r in rows:
        if count >= limit:
            break
        # Crudos
        raw_sym = r.get("symbol","")
        entry = _safe_float(r.get("openAvgPrice", r.get("holdAvgPrice", 0)))
        close = _safe_float(r.get("closeAvgPrice", entry))
        side  = "long" if int(_safe_float(r.get("positionType", 1))) == 1 else "short"
        closeVol = _safe_float(r.get("closeVol", 0))
        holdVol  = _safe_float(r.get("holdVol", 0))
        realised = _safe_float(r.get("realised", 0))
        holdFee  = _safe_float(r.get("holdFee", 0))
        lev      = _safe_float(r.get("leverage", 0))
        im       = _safe_float(r.get("im", r.get("oim", 0)))
        fee_explicit = None
        for k in ("closeFee","fee","poundage","realisedFee","commission"):
            if r.get(k) is not None:
                fee_explicit = _safe_float(r.get(k))
                break

        # Objetivo de PnL precio (si hay fee explícita, restarla)
        approx_price_component = realised - holdFee - (fee_explicit if fee_explicit is not None else 0.0)
        diff = abs(close - entry)

        def price_pnl(sz):
            if side == "short":
                return (entry - close) * abs(sz)
            return (close - entry) * abs(sz)

        pp_close = price_pnl(closeVol) if closeVol else 0.0
        pp_hold  = price_pnl(holdVol)  if holdVol  else 0.0

        _dbg("---- ROW ----")
        _dbg("symbol:", raw_sym, "side:", side)
        _dbg("prices:", {"entry": entry, "close": close, "Δ": diff})
        _dbg("volumes:", {"closeVol": closeVol, "holdVol": holdVol})
        _dbg("wallet:", {"realised": realised, "holdFee": holdFee, "fee_explicit": fee_explicit, "lev": lev, "im": im})
        _dbg("price_pnl_from_closeVol:", pp_close, "price_pnl_from_holdVol:", pp_hold, "target≈", approx_price_component)

        print(json.dumps({
            "symbol": raw_sym,
            "side": side,
            "entry": entry,
            "close": close,
            "closeVol": closeVol,
            "holdVol": holdVol,
            "realised": realised,
            "holdFee": holdFee,
            "fee_explicit": fee_explicit,
            "leverage": lev,
            "im": im,
            "price_pnl_closeVol": pp_close,
            "price_pnl_holdVol": pp_hold,
            "target_price_component": approx_price_component,
            "createTime": r.get("createTime"),
            "updateTime": r.get("updateTime"),
        }, ensure_ascii=False))
        count += 1
def _ts_iso(ms: int) -> str:
    try:
        return datetime.fromtimestamp(int(ms)/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return str(ms)

def debug_dump_mexc_funding(days: int = 60,
                            symbol: Optional[str] = None,
                            max_pages: int = 15,
                            page_size: int = 100,
                            print_rows: int = 3,
                            raw: bool = False) -> None:
    """
    Imprime respuesta cruda y un resumen por página del endpoint:
      GET /api/v1/private/position/funding_records

    - days: ventana retrospectiva solo para mostrar un 'since' de referencia (informativo).
            OJO: este debugger NO filtra por fecha; muestra TODO lo que devuelva la API.
    - symbol: filtra por símbolo MEXC (p.ej. 'BTC_USDT'); si None, trae todos.
    - max_pages/page_size: controlan la paginación (MEXC máximo page_size=100).
    - raw: si True, imprime el JSON 'data' de la primera página.
    """
    if not _has_creds():
        print("⚠️  No hay credenciales MEXC configuradas.")
        return

    print("============== MEXC FUNDING DEBUG ==============")
    print(f"params: days={days}, symbol={symbol}, max_pages={max_pages}, page_size={page_size}, raw={raw}")
    now_ms = int(time.time() * 1000)
    since_ms = now_ms - days * 24 * 3600 * 1000
    print(f"now:   {_ts_iso(now_ms)}")
    print(f"since: {_ts_iso(since_ms)} (informativo)\n")

    total_seen = 0
    earliest = None
    latest = None

    for page_num in range(1, max_pages + 1):
        params = {"page_num": page_num, "page_size": min(100, max(1, int(page_size)))}
        if symbol:
            params["symbol"] = symbol

        try:
            resp = _request("GET", "/api/v1/private/position/funding_records",
                            params=params, private=True)
        except Exception as e:
            print(f"❌ request error page={page_num}: {e}")
            break

        d = resp.get("data", {}) if isinstance(resp, dict) else {}
        lst = d.get("resultList", []) or []
        total_page = int(_safe_float(d.get("totalPage", 0)))
        current_page = int(_safe_float(d.get("currentPage", page_num)))
        page_sz = int(_safe_float(d.get("pageSize", 0)))
        total_count = int(_safe_float(d.get("totalCount", 0)))

        print(f"--- PAGE {current_page}/{total_page or '?'}  count={len(lst)}  (api.pageSize={page_sz}  api.totalCount={total_count})")

        if raw and page_num == 1:
            print("RAW first page data:", json.dumps(d, ensure_ascii=False)[:2000], "...\n")

        # imprime una muestra corta de filas
        for it in lst[:print_rows]:
            ts = int(_safe_float(it.get("settleTime", 0)))
            s  = it.get("symbol", "")
            inc = _safe_float(it.get("funding", 0))
            rt  = _safe_float(it.get("rate", 0))
            print(f"   · {s:<16}  funding={inc:<12} rate={rt:<10} settleTime={_ts_iso(ts)}")

            # actualiza min/max
            if ts:
                latest = ts if (latest is None or ts > latest) else latest
                earliest = ts if (earliest is None or ts < earliest) else earliest

        total_seen += len(lst)

        # criterio de fin por totalPage
        if total_page and current_page >= total_page:
            print("↳ fin: alcanzado totalPage")
            break

        # corte si la página viene vacía
        if not lst:
            print("↳ fin: página vacía")
            break

        # pequeño respiro para no pegarse con el rate-limit (20 req / 2 s)
        time.sleep(0.1)

    print("\n============== RESUMEN ==============")
    print(f"total_vistos={total_seen}")
    if earliest or latest:
        print(f"rango_en_resultado: {_ts_iso(earliest)}  →  {_ts_iso(latest)}")
    else:
        print("rango_en_resultado: (sin timestamps)")
    print("====================================\n")
        
        
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--symbol", type=str, default=None)

    # 🔎 nuevos flags de debug funding
    ap.add_argument("--funding-debug", action="store_true", help="Imprime debug de funding por páginas")
    ap.add_argument("--f-days", type=int, default=60, help="Ventana informativa de días hacia atrás")
    ap.add_argument("--f-symbol", type=str, default=None, help="Símbolo MEXC (p.ej. BTC_USDT)")
    ap.add_argument("--f-max-pages", type=int, default=100, help="Máximo de páginas a recorrer")
    ap.add_argument("--f-page-size", type=int, default=100, help="Tamaño de página (max 100)")
    ap.add_argument("--f-raw", action="store_true", help="Mostrar JSON crudo de la primera página")
    args = ap.parse_args()

    if args.funding_debug:
        debug_dump_mexc_funding(days=args.f_days,
                                symbol=args.f_symbol,
                                max_pages=args.f_max_pages,
                                page_size=args.f_page_size,
                                raw=args.f_raw)
        raise SystemExit(0)

    print("== balances ==")
    print(fetch_mexc_all_balances())

    print("== open positions ==")
    print(fetch_mexc_open_positions(symbol=args.symbol))

    print("== funding (10) ==")
    print(fetch_mexc_funding_fees(limit=100, symbol=args.symbol))

    if not args.dry_run:
        print("== saving closed ==")
        n = save_mexc_closed_positions(days=args.days, symbol=args.symbol, debug=True)
        print(f"saved: {n}")