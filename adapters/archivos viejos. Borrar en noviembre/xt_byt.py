# adapters/xt_sapi.py
# XT SAPI (Spot) – Firma exacta X+Y con #METHOD#PATH#QUERY#BODY y debug detallado
from __future__ import annotations

import os, json, time, hmac, hashlib, argparse
from typing import Any, Dict, Optional
from urllib.parse import parse_qsl
import requests

SAPI_HOST = os.getenv("XT_SAPI_HOST", "https://sapi.xt.com")

DEFAULT_RECVWINDOW = int(os.getenv("XT_RECVWINDOW_MS", "5000"))
EXCHANGE = "xt"
XT_PERP_HOST = os.getenv("XT_PERP_HOST", "https://sapi.xt.com")
XT_API_KEY = "96b1c999-b9df-42ed-9d4f-a4491aac7a1d"
XT_API_SECRET = "5f60a0a147d82db2117a45da129a6f4480488234"


def _now_ms() -> int:
    return int(time.time() * 1000)

def _canon_query(params: Optional[Dict[str, Any]]) -> str:
    """Devuelve k=v&k2=v2 con claves en orden ASCII ascendente (lexicográfico)."""
    if not params:
        return ""
    items = sorted((str(k), "" if v is None else str(v)) for k, v in params.items())
    return "&".join(f"{k}={v}" for k, v in items)

def _json_body_str(body: Any) -> str:
    """Para BODY JSON: usa la cadena RAW JSON, sin ordenar claves, compacta (no pretty)."""
    if body is None or body == "":
        return ""
    if isinstance(body, str):
        # suponemos que ya es JSON string crudo
        return body
    # serializa sin ordenar claves (sort_keys=False), compacto
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=False)

def _build_signature_headers(
    method: str,
    path: str,
    query_params: Optional[Dict[str, Any]] = None,
    body_json: Any = None,
    recvwindow_ms: int = DEFAULT_RECVWINDOW,
    algorithms: str = "HmacSHA256",
) -> Dict[str, str]:
    """
    X = validate-algorithms=...&validate-appkey=...&validate-recvwindow=...&validate-timestamp=...
    Y = #METHOD#PATH[#QUERY][#BODY]
    original = X + Y
    signature = HMAC_SHA256(secretKey, original)
    """
    if not XT_API_KEY or not XT_API_SECRET:
        raise RuntimeError("Faltan XT_API_KEY / XT_API_SECRET en el entorno.")

    ts = _now_ms()
    method_u = (method or "GET").upper()
    if not path.startswith("/"):
        path = "/" + path

    # --- X (ORDEN ALFABÉTICO NATURAL de claves) ---
    # Nota: mantenemos exactamente las claves indicadas y en orden ASCII ascendente
    X = (
        f"validate-algorithms={algorithms}"
        f"&validate-appkey={XT_API_KEY}"
        f"&validate-recvwindow={recvwindow_ms}"
        f"&validate-timestamp={ts}"
    )

    # --- Y ---
    q = _canon_query(query_params)
    body_str = _json_body_str(body_json) if method_u != "GET" else ""
    if q and body_str:
        Y = f"#{method_u}#{path}#{q}#{body_str}"
    elif q:
        Y = f"#{method_u}#{path}#{q}"
    elif body_str:
        Y = f"#{method_u}#{path}#{body_str}"
    else:
        Y = f"#{method_u}#{path}"

    original = f"{X}{Y}"
    signature = hmac.new(
        XT_API_SECRET.encode("utf-8"),
        original.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "validate-algorithms": algorithms,
        "validate-appkey": XT_API_KEY,
        "validate-recvwindow": str(recvwindow_ms),
        "validate-timestamp": str(ts),
        "validate-signature": signature,
    }
    # devolvemos también los componentes para debug
    headers["_X"] = X
    headers["_Y"] = Y
    headers["_original"] = original
    headers["_signature"] = signature
    headers["_ts"] = str(ts)
    return headers

def xt_sapi_request(
    method: str,
    path: str,
    query: Optional[Dict[str, Any]] = None,
    body: Any = None,
    recvwindow_ms: int = DEFAULT_RECVWINDOW,
    debug: bool = True,
    timeout: int = 20,
) -> Dict[str, Any]:
    """Hace la petición firmada a sapi.xt.com con debug opcional."""
    method_u = (method or "GET").upper()
    if not path.startswith("/"):
        path = "/" + path
    headers = _build_signature_headers(method_u, path, query, body, recvwindow_ms)
    # extrae y no envíes los campos de debug en headers
    debug_fields = {k: headers.pop(k) for k in list(headers.keys()) if k.startswith("_")}

    url = SAPI_HOST.rstrip("/") + path
    if debug:
        print("┌─ XT SAPI REQUEST")
        print(f"│ HOST     : {SAPI_HOST}")
        print(f"│ METHOD   : {method_u}")
        print(f"│ PATH     : {path}")
        print(f"│ URL      : {url}")
        print(f"│ QUERY    : {json.dumps(query or {}, ensure_ascii=False)}")
        if method_u != "GET":
            print(f"│ BODY(JSON): { _json_body_str(body) }")
        print("│ --- SIGNATURE DEBUG ---")
        print(f"│ X        : {debug_fields.get('_X')}")
        print(f"│ Y        : {debug_fields.get('_Y')}")
        print(f"│ ORIGINAL : {debug_fields.get('_original')}")
        print(f"│ SIGNATURE: {debug_fields.get('_signature')}")
        print(f"│ HEADERS  : {{")
        for hk, hv in headers.items():
            if hk.lower() == "validate-signature":
                print(f"│   {hk}: {hv}")
            else:
                print(f"│   {hk}: {hv}")
        print("│ }")
        print("└────────────────────────")

    if method_u == "GET":
        resp = requests.get(url, headers=headers, params=(query or {}), timeout=timeout)
    elif method_u == "POST":
        resp = requests.post(url, headers=headers, params=(query or {}), data=_json_body_str(body), timeout=timeout)
    elif method_u == "DELETE":
        resp = requests.delete(url, headers=headers, params=(query or {}), timeout=timeout)
    elif method_u == "PUT":
        resp = requests.put(url, headers=headers, params=(query or {}), data=_json_body_str(body), timeout=timeout)
    else:
        raise ValueError("Método HTTP no soportado")

    try:
        resp.raise_for_status()
    except Exception as e:
        print(f"❌ HTTP {resp.status_code}: {getattr(resp, 'text', '')}")
        raise

    try:
        data = resp.json()
    except Exception:
        data = {"_raw_text": resp.text}

    if debug:
        print("🧾 RESPONSE RAW:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    return data

# ---------- Caso de prueba: GET /v4/balances (Spot) ----------
def xt_spot_get_balances(currencies: Optional[str] = None, debug: bool = True) -> Dict[str, Any]:
    """
    GET /v4/balances
    params: currencies='usdt,btc' (opcional)
    """
    path = "/v4/balances"
    query = {}
    if currencies:
        query["currencies"] = currencies  # e.g., 'usdt,btc'
    return xt_sapi_request("GET", path, query=query, body=None, debug=debug)

# ------------- CLI -------------
def _parse_query(q: Optional[str]) -> Dict[str, str]:
    if not q:
        return {}
    return {k: v for k, v in parse_qsl(q, keep_blank_values=True)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser("XT SAPI signer & /v4/balances debug")
    ap.add_argument("--balances", action="store_true", help="Prueba GET /v4/balances")
    ap.add_argument("--currencies", type=str, default=None, help="Lista separada por comas, p.ej. 'usdt,btc'")
    ap.add_argument("--method", type=str, default="GET", help="Para pruebas arbitrarias")
    ap.add_argument("--path", type=str, default="/v4/balances", help="Ruta p.ej. /v4/balances")
    ap.add_argument("--query", type=str, default=None, help="Query en formato k=v&k2=v2")
    ap.add_argument("--body", type=str, default=None, help="Cuerpo JSON como string crudo")
    ap.add_argument("--no-debug", action="store_true", help="No imprimir trazas de firma")
    ap.add_argument("--recvwindow", type=int, default=DEFAULT_RECVWINDOW)
    args = ap.parse_args()

    if not XT_API_KEY or not XT_API_SECRET:
        print("⚠️  Define XT_API_KEY y XT_API_SECRET en el entorno antes de probar.")
        print("    Ej.: export XT_API_KEY='...' ; export XT_API_SECRET='...'")
        exit(1)

    if args.balances:
        xt_spot_get_balances(args.currencies, debug=(not args.no_debug))
    else:
        # Runner genérico para cualquier endpoint de SAPI firmado
        q = _parse_query(args.query)
        body = args.body
        xt_sapi_request(
            method=args.method,
            path=args.path,
            query=q,
            body=body,
            recvwindow_ms=args.recvwindow,
            debug=(not args.no_debug),
        )


