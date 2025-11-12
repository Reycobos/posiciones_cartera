#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEXC WebSocket — Open Positions (private)
-----------------------------------------
- Conecta al WS privado de MEXC futures: wss://contract.mexc.com/edge
- Hace login (HMAC-SHA256: apiKey+reqTime firmado con secret)
- Filtra sólo el canal de posiciones: push.personal.position
- Mantiene un snapshot en memoria de las posiciones abiertas (state 1/2)
- Normaliza la salida para tu UI (exchange, symbol, side, size, entry_price,
  mark_price, liquidation_price, notional, unrealized_pnl, fee, funding_fee,
  realized_pnl).

Incluye además un DEBUG autoejecutable para Spyder que:
  1) Imprime cada payload crudo recibido para "push.personal.position"
  2) Muestra el snapshot normalizado actual cada N segundos

Requisitos:
  pip install websocket-client requests python-dotenv

ENV esperadas (o pasar por CLI):
  MEXC_API_KEY
  MEXC_API_SECRET

Ejemplos:
  # Sólo debug 20s con raw payloads
  python mexc_ws_positions.py --seconds 20 --raw 1

  # Sin raw, pero al final imprime snapshot normalizado
  python mexc_ws_positions.py --seconds 30

Integración:
  - Este archivo es independiente. Para integrarlo en Flask, crea un proceso/hilo
    que deje vivo el listener y expón el snapshot por donde lo necesites.
"""

from __future__ import annotations
from flask import Blueprint, jsonify
import os, time, hmac, hashlib, json, math, threading, re
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone
import gzip, zlib
from websocket import ABNF


# --- dependencias de red ---
try:
    import websocket  # websocket-client
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

WS_URL = os.getenv("MEXC_WS_URL", "wss://contract.mexc.com/edge")
REST_BASE = os.getenv("MEXC_BASE_URL", "https://contract.mexc.com")
API_KEY = "mx0vglOEFTy9klFKJo"
API_SECRET = "1f45cf4ac48148419b59298352c45ef0"
PING_INTERVAL_SEC = int(os.getenv("MEXC_PING_SEC", "20"))  # doc recomienda 10-20s


# === Singleton del listener (usa tus API_KEY/SECRET ya definidos en el archivo) ===
LISTENER = None

def get_listener():
    global LISTENER
    if LISTENER is None:
        # arranca el listener en segundo plano una única vez
        L = MexcWSPositions(API_KEY, API_SECRET, debug_raw=False)
        L.start()
        LISTENER = L
    return LISTENER

# === Blueprint para integrarlo en tu Flask principal ===
mexc_ws_bp = Blueprint("mexc_ws", __name__)


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



# Normaliza el símbolo (similar a adapters/mexcv2.py)
SPECIAL_SYMBOL_MAP = {
    "OPENLEDGER": "OPEN",
}

@mexc_ws_bp.route("/api/mexc_ws_asset")
def api_mexc_ws_asset():
    L = get_listener()
    return jsonify({"ok": True, "ts": int(time.time()*1000), "asset": L.get_asset_snapshot()})

@mexc_ws_bp.route("/api/mexc_ws_health")
def api_mexc_ws_health():
    L = get_listener()
    snap = L.get_snapshot()
    asset = L.get_asset_snapshot()
    return jsonify({
        "ok": True,
        "positions_count": len(snap),
        "asset_keys": list(asset.keys()),
        "ts": int(time.time()*1000),
    })

def normalize_symbol(sym: str) -> str:
    if not sym:
        return ""
    s = sym.upper().strip()
    s = re.sub(r'^PERP_', '', s)
    s = re.sub(r'(_|-)?(USDT|USDC|USD|PERP)$', '', s)
    s = re.sub(r'[_-]+$', '', s)
    base = re.split(r'[_-]', s)[0]
    return SPECIAL_SYMBOL_MAP.get(base, base)


# --- mark price helper (rápido, con pequeños fallbacks) ---
class MarkPriceCache:
    def __init__(self, ttl_sec: int = 5):
        self._asset = {}  # currency -> dict con balances

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
        # fetch
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
        # 2) ticker (lista o dict)
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

MARK = MarkPriceCache(ttl_sec=5)

# --- pnl unrealized ---
def _pnl_unrealized(entry: float, mark: float, size: float, side: str) -> float:
    if any(math.isnan(x) for x in (entry, mark, size)):
        return 0.0
    if side == "short":
        return (entry - mark) * size
    return (mark - entry) * size

# --- firma de login WS ---
def mexc_ws_signature(api_key: str, api_secret: str, ms: int) -> str:
    # doc: signature target string = accessKey + timestamp ; HMAC-SHA256(secret)
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
            on_data=self._on_data, 
        )
        self._thread = threading.Thread(target=self.ws.run_forever, kwargs={"ping_interval": 0}, daemon=True)
        self._thread.start()
        # Espera login OK o timeout
        if not self._ready.wait(timeout=10):
            raise RuntimeError("Timeout esperando login de MEXC WS")
        # keepalive pings
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
        # login
        ts = _now_ms()
        sig = mexc_ws_signature(self.api_key, self.api_secret, ts)
        login_msg = {
            "method": "login",
            "subscribe": False, 
            "param": {
                "apiKey": self.api_key,
                "reqTime": str(ts),
                "signature": sig,
            },
            # Si quisieras cancelar la suscripción por defecto:
            # "subscribe": False,
        }
        self._send(login_msg)

        # filtra sólo posiciones (tras login success)
        # lo enviaremos cuando recibamos "rs.login"
        
    def _on_data(self, _, message, data_type, cont):
        # websocket-client: data_type = ABNF.OPCODE_TEXT o ABNF.OPCODE_BINARY
        if data_type == ABNF.OPCODE_TEXT:
            try:
                self._on_message(self.ws, message.decode() if isinstance(message, (bytes,bytearray)) else message)
            except Exception as e:
                print("❌ on_data text error:", e)
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
            if payload is not None:
                try:
                    self._on_message(self.ws, payload.decode("utf-8"))
                except Exception as e:
                    print("❌ on_data bin->text error:", e)
            else:
                # último intento: tratar como texto
                try:
                    self._on_message(self.ws, message.decode("utf-8", "ignore"))
                except Exception as e:
                    print("❌ on_data bin error:", e)

    def _on_message(self, _, message: str):
        try:
            msg = json.loads(message)
        except Exception:
            return

        ch = msg.get("channel") or msg.get("topic")
        if self.debug_raw and (ch == "push.personal.position" or ch is None):
            print("RAW:", json.dumps(msg, ensure_ascii=False))

        if ch == "rs.login":
            # 1) pide TODO (snapshot inicial de todos los canales privados)
            self._send({"method": "personal.filter"})  # igual a filtros vacíos
            # 2) (opcional) 200 ms después, restringe a lo que te interese
            def _restrict():
                self._send({
                    "method": "personal.filter",
                    "param": {"filters": [
                        {"filter": "position"},
                        {"filter": "asset"},
                        {"filter": "order"},        # si quieres ver órdenes también
                        {"filter": "order.deal"},
                    ]}
                })
            threading.Timer(0.2, _restrict).start()
            self._ready.set()
            return
        if ch == "push.personal.asset":
            self._handle_asset(msg.get("data", {}))
            return

        # errores
        if ch == "rs.error":
            err = msg.get("data") or msg
            print("❌ WS error:", err)
            return

        # posiciones privadas
        if ch == "push.personal.position":
            data = msg.get("data", {})
            self._handle_position(data)

    def _on_error(self, _, err):
        print("❌ WS error:", err)

    def _send(self, payload: dict):
        try:
            if self.ws and getattr(self.ws, "sock", None) and self.ws.sock and self.ws.sock.connected:
                self.ws.send(json.dumps(payload))
        except Exception as e:
            print("❌ send error:", e)
    
    def _on_close(self, *_):
        print("🔌 WS cerrado")
        # detén el ping loop para que no intente enviar a un socket cerrado
        self._stop.set()

    def _ping_loop(self):
        while not self._stop.is_set():
            try:
                self._send({"method": "ping"})
            except Exception:
                pass
            self._stop.wait(PING_INTERVAL_SEC)

    # ---------- domain ----------
    def _handle_position(self, p: dict):
        # Campos esperados (doc): positionId, symbol, holdVol, positionType (1 long / 2 short),
        # openAvgPrice, holdAvgPrice, closeAvgPrice, liquidatePrice, realised, holdFee, state (1/2=abierta, 3=cerrada)
        pid = int(_safe_float(p.get("positionId"), 0))
        if pid <= 0:
            return

        state = int(_safe_float(p.get("state", 0)))
        if state == 3:  # cerrada → elimina del snapshot
            with self._lock:
                self._positions.pop(pid, None)
            return

        # normaliza/guarda
        raw_sym = p.get("symbol", "")
        side = "long" if int(_safe_float(p.get("positionType", 1))) == 1 else "short"
        size = abs(_safe_float(p.get("holdVol", 0)))
        entry = _safe_float(p.get("openAvgPrice", p.get("holdAvgPrice", 0)))
        liq = _safe_float(p.get("liquidatePrice", 0))
        funding = _safe_float(p.get("holdFee", 0))  # acumulado + / -
        fee_acc = 0.0  # no disponible en este canal
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
            # opcionalmente, guarda crudo para inspección
            "_raw": p,
        }

        with self._lock:
            self._positions[pid] = normalized

    # ---------- API pública ----------
    def get_snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._positions.values())

def _handle_asset(self, a: dict):
    cur = str(a.get("currency","")).upper() or "USDT"
    asset = {
        "currency": cur,
        "available_balance": float(a.get("availableBalance") or 0),
        "frozen_balance": float(a.get("frozenBalance") or 0),
        "position_margin": float(a.get("positionMargin") or 0),
        "cash_balance": float(a.get("cashBalance") or a.get("availableBalance") or 0),
        "ts": int(a.get("updateTime") or 0),
    }
    with self._lock:
        self._asset[cur] = asset

def get_asset_snapshot(self):
    with self._lock:
        return dict(self._asset)


# helper opcional si quieres timestamp del mensaje
def msg_ts(d):
    try:
        return int(d.get("ts") or 0)
    except Exception:
        return 0

# ===========================
#      DEBUG autoejecutable
# ===========================

def _iso(ms: int) -> str:
    try:
        return datetime.fromtimestamp(int(ms)/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return str(ms)


def run_debug(seconds: int = 20, raw: bool = True):
    key = API_KEY
    sec = API_SECRET
    if not key or not sec:
        raise SystemExit("⚠️  Configura MEXC_API_KEY y MEXC_API_SECRET en el entorno o pásalos por CLI.")

    print(f"🌐 Conectando a {WS_URL} (debug {seconds}s, raw={raw})…")
    client = MexcWSPositions(key, sec, debug_raw=raw)
    client.start()

    t0 = time.time()
    try:
        while time.time() - t0 < max(5, seconds):
            snap = client.get_snapshot()
            print(f"\n📦 Snapshot posiciones abiertas: {len(snap)} items")
            for i, pos in enumerate(snap[:20], 1):
                p = pos
                print(
                    f"  {i:02d}. {p['symbol']} {p['side']:>5} size={p['size']} entry={p['entry_price']} "
                    f"mark={p['mark_price']} unrl={p['unrealized_pnl']} liq={p['liquidation_price']}"
                )
            time.sleep(3)
    finally:
        client.stop()
        snap = client.get_snapshot()
        print("\n====== SNAPSHOT FINAL NORMALIZADO ======")
        print(json.dumps(snap, ensure_ascii=False, indent=2))
        print("======================================\n")



if __name__ == "__main__":
    from flask import Flask, send_from_directory
    import os

    app = Flask(__name__)
    app.register_blueprint(mexc_ws_bp)  # /api/mexc_ws_positions
    

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    @app.route("/mexc-ws-test")
    def mexc_ws_test_page():
        # Asegúrate de tener 'mexc_ws_test.html' en la MISMA carpeta que este .py
        return send_from_directory(BASE_DIR, "mexc_ws_test.html")

    print("🔧 Bootstrapping MEXC WS listener…")
    get_listener()  # arranca el WS una vez

    print("🌐 Serving /api/mexc_ws_positions en http://127.0.0.1:8787 …")
    print("🌐 Test page: http://127.0.0.1:8787/mexc-ws-test")
    app.run(host="127.0.0.1", port=8787, debug=False, use_reloader=False)


