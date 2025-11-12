#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEXC Futures — WebSocket listener (positions + asset) con HTTP endpoints
------------------------------------------------------------------------
- WS privado: wss://contract.mexc.com/edge
- Login HMAC-SHA256 (signature = HMAC(secret, apiKey + reqTime))
- Cancelamos el push por defecto (subscribe:false) y luego pedimos TODO con
  personal.filter vacío; opcionalmente restringimos (position/asset/order/deal)
- Maneja frames BINARIOS (gzip/deflate) además de texto (on_data)
- Mantiene snapshots en memoria de:
    • posiciones abiertas (state 1/2) → /api/mexc_ws_positions
    • asset por moneda (USDT, etc.) → /api/mexc_ws_asset
    • health básico → /api/mexc_ws_health
- Servidor HTTP de prueba integrado (puerto 8787) con ruta /mexc-ws-test

Requisitos:
  pip install websocket-client requests python-dotenv flask

ENV (opcional):
  MEXC_WS_URL=wss://contract.mexc.com/edge
  MEXC_API_KEY=...
  MEXC_API_SECRET=...
  MEXC_PING_SEC=20
  MEXC_MARK_TTL=5

Uso (standalone):
  python mexcwebsocket.py
  → http://127.0.0.1:8787/mexc-ws-test

Integración en app principal (Flask):
  from adapters.mexcwebsocket import mexc_ws_bp, get_listener
  app.register_blueprint(mexc_ws_bp)
  get_listener()  # arranca el WS en segundo plano

Notas:
- Si no ves pushes, puede ser por falta de actividad, permisos o compresión.
  Este cliente ya soporta binario (gzip/deflate).
"""

from __future__ import annotations
import os, time, hmac, hashlib, json, math, threading, re, gzip, zlib
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone

# --- dependencias externas ---
try:
    import websocket  # websocket-client
    from websocket import ABNF
except Exception as e:  # pragma: no cover
    raise SystemExit("\n⚠️ Falta 'websocket-client'. Instala con: pip install websocket-client\n")

try:
    import requests
except Exception:
    raise SystemExit("\n⚠️ Falta 'requests'. Instala con: pip install requests\n")

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- configuración ---
WS_URL = os.getenv("MEXC_WS_URL", "wss://contract.mexc.com/edge")
REST_BASE = os.getenv("MEXC_BASE_URL", "https://contract.mexc.com")
API_KEY = "mx0vglOEFTy9klFKJo"
API_SECRET = "1f45cf4ac48148419b59298352c45ef0"
PING_INTERVAL_SEC = int(os.getenv("MEXC_PING_SEC", "20"))  # doc recomienda 10-20s
MARK_TTL = int(os.getenv("MEXC_MARK_TTL", "5"))

# -------------------- util --------------------

def _now_ms() -> int:
    return int(time.time() * 1000)

def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default

# Normaliza el símbolo para UI
SPECIAL_SYMBOL_MAP = {"OPENLEDGER": "OPEN"}

def normalize_symbol(sym: str) -> str:
    if not sym:
        return ""
    s = sym.upper().strip()
    s = re.sub(r'^PERP_', '', s)
    s = re.sub(r'(_|-)?(USDT|USDC|USD|PERP)$', '', s)
    s = re.sub(r'[_-]+$', '', s)
    base = re.split(r'[_-]', s)[0]
    return SPECIAL_SYMBOL_MAP.get(base, base)

# --- mark price helper (rápido) ---
class MarkPriceCache:
    def __init__(self, ttl_sec: int = 5):
        self.ttl = ttl_sec
        self._data: Dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def get(self, raw_symbol: str) -> Optional[float]:
        now = time.time()
        k = (raw_symbol or '').upper()
        with self._lock:
            v = self._data.get(k)
            if v and now - v[1] <= self.ttl:
                return v[0]
        p = self._fetch_mark_price(k)
        if p is not None:
            with self._lock:
                self._data[k] = (p, now)
        return p

    @staticmethod
    def _fetch_mark_price(symbol: str) -> Optional[float]:
        # 1) fair_price
        try:
            r = requests.get(f"{REST_BASE}/api/v1/contract/fair_price", params={"symbol": symbol}, timeout=5)
            d = r.json().get("data", {}) if r.text else {}
            if isinstance(d, dict) and d.get("fairPrice") is not None:
                return float(d["fairPrice"])
        except Exception:
            pass
        # 2) ticker
        try:
            r = requests.get(f"{REST_BASE}/api/v1/contract/ticker", params={"symbol": symbol}, timeout=5)
            d = r.json().get("data")
            if isinstance(d, dict):
                for k in ("fairPrice", "lastPrice", "last"):
                    if d.get(k) is not None:
                        return float(d[k])
            elif isinstance(d, list) and d:
                x = d[0]
                for k in ("fairPrice", "lastPrice", "last"):
                    if x.get(k) is not None:
                        return float(x[k])
        except Exception:
            pass
        return None

MARK = MarkPriceCache(ttl_sec=MARK_TTL)

# --- pnl unrealized ---
def _pnl_unrealized(entry: float, mark: float, size: float, side: str) -> float:
    if any(math.isnan(x) for x in (entry, mark, size)):
        return 0.0
    if side == "short":
        return (entry - mark) * size
    return (mark - entry) * size

# --- firma de login WS ---
def mexc_ws_signature(api_key: str, api_secret: str, ms: int) -> str:
    target = f"{api_key}{ms}"
    return hmac.new(api_secret.encode(), target.encode(), hashlib.sha256).hexdigest()

# ===========================
#       Listener WS
# ===========================
class MexcWSPositions:
    def __init__(self, api_key: str, api_secret: str, debug_raw: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.ws: Optional[websocket.WebSocketApp] = None
        self.debug_raw = debug_raw
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._positions: Dict[int, Dict[str, Any]] = {}
        self._asset: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._ping_thread: Optional[threading.Thread] = None
        self._thread: Optional[threading.Thread] = None

    # ---------- lifecycle ----------
    def start(self):
        self.ws = websocket.WebSocketApp(
            WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_data=self._on_data,  # soporta binario (gzip/deflate)
        )
        self._thread = threading.Thread(target=self.ws.run_forever, kwargs={"ping_interval": 0}, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError("Timeout esperando login de MEXC WS")
        self._ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        self._ping_thread.start()

    def stop(self):
        self._stop.set()
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass

    # ---------- handlers ----------
    def _on_open(self, *_):
        ts = _now_ms()
        sig = mexc_ws_signature(self.api_key, self.api_secret, ts)
        # Cancelamos default push y controlamos con personal.filter
        login_msg = {
            "method": "login",
            "subscribe": False,
            "param": {
                "apiKey": self.api_key,
                "reqTime": str(ts),
                "signature": sig,
            },
        }
        self._send(login_msg)

    def _on_message(self, _, message: str):
        try:
            msg = json.loads(message)
        except Exception:
            return

        ch = msg.get("channel") or msg.get("topic")
        if self.debug_raw and ch in ("rs.login", "rs.error", "push.personal.position", "push.personal.asset", "rs.personal.filter"):
            print("RAW:", json.dumps(msg, ensure_ascii=False))

        # login ok → pedimos TODO y luego restringimos
        if ch == "rs.login":
            self._send({"method": "personal.filter"})  # filtros vacíos = todos
            # restringe un poco tras 200ms
            def _restrict():
                self._send({
                    "method": "personal.filter",
                    "param": {"filters": [
                        {"filter": "position"},
                        {"filter": "asset"},
                        {"filter": "order"},
                        {"filter": "order.deal"},
                    ]},
                })
            threading.Timer(0.2, _restrict).start()
            self._ready.set()
            return

        if ch == "rs.error":
            print("❌ WS error:", msg.get("data") or msg)
            return

        if ch == "push.personal.position":
            data = msg.get("data", {})
            self._handle_position(data)
            return

        if ch == "push.personal.asset":
            data = msg.get("data", {})
            self._handle_asset(data)
            return

    def _on_error(self, _, err):
        print("❌ WS error:", err)
        # en errores, para no spamear pings a socket caído
        self._stop.set()

    def _on_close(self, *_):
        print("🔌 WS cerrado")
        self._stop.set()

    def _on_data(self, _, message, data_type, cont):
        # Maneja tanto texto como binario (gzip/deflate)
        try:
            if data_type == ABNF.OPCODE_TEXT:
                if isinstance(message, (bytes, bytearray)):
                    message = message.decode("utf-8", "ignore")
                self._on_message(self.ws, message)
                return
            if data_type == ABNF.OPCODE_BINARY:
                payload = None
                try:
                    payload = gzip.decompress(message)
                except Exception:
                    try:
                        payload = zlib.decompress(message, -zlib.MAX_WBITS)  # raw deflate
                    except Exception:
                        pass
                if payload is None:
                    try:
                        payload = bytes(message).decode("utf-8", "ignore").encode()
                    except Exception:
                        return
                self._on_message(self.ws, payload.decode("utf-8", "ignore"))
        except Exception as e:
            print("❌ on_data error:", e)

    def _send(self, payload: dict):
        try:
            if self.ws and getattr(self.ws, "sock", None) and self.ws.sock and self.ws.sock.connected:
                self.ws.send(json.dumps(payload))
        except Exception as e:
            print("❌ send error:", e)

    def _ping_loop(self):
        while not self._stop.is_set():
            try:
                self._send({"method": "ping"})
            except Exception:
                pass
            self._stop.wait(PING_INTERVAL_SEC)

    # ---------- domain ----------
    def _handle_position(self, p: dict):
        pid = int(_safe_float(p.get("positionId"), 0))
        if pid <= 0:
            return
        state = int(_safe_float(p.get("state", 0)))
        if state == 3:  # cerrada
            with self._lock:
                self._positions.pop(pid, None)
            return

        raw_sym = p.get("symbol", "")
        side = "long" if int(_safe_float(p.get("positionType", 1))) == 1 else "short"
        size = abs(_safe_float(p.get("holdVol", 0)))
        entry = _safe_float(p.get("openAvgPrice", p.get("holdAvgPrice", 0)))
        liq = _safe_float(p.get("liquidatePrice", 0))
        funding = _safe_float(p.get("holdFee", 0))
        fee_acc = 0.0
        mark = MARK.get(raw_sym) or _safe_float(p.get("closeAvgPrice", entry)) or entry
        unreal = _pnl_unrealized(entry, mark, size, side)
        notional = size * entry
        realized_open = fee_acc + funding

        normalized = {
            "exchange": "mexc",
            "symbol": normalize_symbol(raw_sym),
            "side": side,
            "size": size,
            "entry_price": entry,
            "mark_price": mark,
            "liquidation_price": liq,
            "notional": notional,
            "unrealized_pnl": unreal,
            "fee": fee_acc,
            "funding_fee": funding,
            "realized_pnl": realized_open,
            "_raw": p,
        }
        with self._lock:
            self._positions[pid] = normalized

    def _handle_asset(self, a: dict):
        cur = str(a.get("currency", "")).upper() or "USDT"
        asset = {
            "currency": cur,
            "available_balance": float(a.get("availableBalance", 0) or 0),
            "frozen_balance": float(a.get("frozenBalance", 0) or 0),
            "position_margin": float(a.get("positionMargin", 0) or 0),
            "cash_balance": float(a.get("cashBalance", a.get("availableBalance", 0)) or 0),
            "ts": int(a.get("updateTime") or 0),
        }
        with self._lock:
            self._asset[cur] = asset

    # ---------- API pública ----------
    def get_positions_snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._positions.values())

    def get_asset_snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._asset)


# ===========================
#    Blueprint HTTP (Flask)
# ===========================
from flask import Blueprint, jsonify

LISTENER: Optional[MexcWSPositions] = None

def get_listener() -> MexcWSPositions:
    global LISTENER
    if LISTENER is None:
        if not API_KEY or not API_SECRET:
            raise RuntimeError("Configura MEXC_API_KEY y MEXC_API_SECRET para iniciar el WS")
        L = MexcWSPositions(API_KEY, API_SECRET, debug_raw=False)
        L.start()
        LISTENER = L
    return LISTENER

mexc_ws_bp = Blueprint("mexc_ws", __name__)

@mexc_ws_bp.route("/api/mexc_ws_positions")
def api_mexc_ws_positions():
    L = get_listener()
    snap = L.get_positions_snapshot()
    return jsonify({"ok": True, "source": "ws:mexc", "ts": int(time.time()*1000), "positions": snap})

@mexc_ws_bp.route("/api/mexc_ws_asset")
def api_mexc_ws_asset():
    L = get_listener()
    return jsonify({"ok": True, "ts": int(time.time()*1000), "asset": L.get_asset_snapshot()})

@mexc_ws_bp.route("/api/mexc_ws_health")
def api_mexc_ws_health():
    L = get_listener()
    return jsonify({
        "ok": True,
        "positions_count": len(L.get_positions_snapshot()),
        "asset_currencies": list(L.get_asset_snapshot().keys()),
        "ts": int(time.time()*1000),
    })


# ===========================
#  Servidor de prueba (main)
# ===========================
if __name__ == "__main__":
    from flask import Flask, send_from_directory, Response

    app = Flask(__name__)
    app.register_blueprint(mexc_ws_bp)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    @app.route("/mexc-ws-test")
    def mexc_ws_test_page():
        # Sirve el archivo si existe; si no, devuelve una página mínima embebida
        fname = os.path.join(BASE_DIR, "mexc_ws_test.html")
        if os.path.exists(fname):
            return send_from_directory(BASE_DIR, "mexc_ws_test.html")
        # HTML mínimo inline (fallback)
        html = f"""
<!doctype html><html><head><meta charset='utf-8'><title>MEXC WS Test</title>
<style>body{{font:14px system-ui;background:#0f1115;color:#e7e9ee}}.box{{max-width:980px;margin:24px auto;padding:16px;background:#151826;border-radius:12px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid #22283e}}small{{color:#9aa4b2}}</style></head>
<body><div class='box'>
<h2>MEXC WS — Positions & Asset (test)</h2>
<p><small>Este fallback se sirve inline. Para UI completa pon <b>mexc_ws_test.html</b> junto al .py</small></p>
<p>
  <button onclick="loadPos()">Refresh positions</button>
  <button onclick="loadAst()">Refresh asset</button>
</p>
<h3>Positions</h3>
<pre id='pos'>(vacío)</pre>
<h3>Asset</h3>
<pre id='ast'>(vacío)</pre>
<script>
async function loadPos(){{let r=await fetch('/api/mexc_ws_positions');let j=await r.json();document.getElementById('pos').textContent=JSON.stringify(j,null,2);}}
async function loadAst(){{let r=await fetch('/api/mexc_ws_asset');let j=await r.json();document.getElementById('ast').textContent=JSON.stringify(j,null,2);}}
</script>
</div></body></html>
        """
        return Response(html, mimetype="text/html")

    print("🔧 Bootstrapping MEXC WS listener…")
    # arranca el WS una vez al levantar el server HTTP
    try:
        get_listener()
    except Exception as e:
        print("⚠️ No se pudo iniciar el WS:", e)

    print("🌐 Serving endpoints en http://127.0.0.1:8787 …")
    print("   • /api/mexc_ws_positions  • /api/mexc_ws_asset  • /api/mexc_ws_health")
    print("   • /mexc-ws-test (UI de prueba)")
    app.run(host="127.0.0.1", port=8787, debug=False, use_reloader=False)
