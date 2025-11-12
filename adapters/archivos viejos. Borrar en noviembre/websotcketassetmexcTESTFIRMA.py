#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEXC WebSocket — AUTH DEBUG (private)
-------------------------------------
Objetivo: comprobar de forma explícita que el login WS se está firmando bien
 y que el servidor lo acepta o rechaza como corresponde.

Qué hace:
 1) Calcula y MUESTRA por pantalla el payload de login, incluidos apiKey (en claro),
    reqTime y signature (hash). ⚠️ Úsalo sólo en entorno local seguro.
 2) Abre el WS y envía el login correcto -> espera ver "rs.login".
 3) Envia un login INTENCIONADAMENTE MALO (con signature falsa) a la misma conexión
    y comprueba si llega "rs.error".
 4) Si el login correcto llega, manda un personal.filter minimal y muestra todo
    lo que empuje el servidor (rs.login, rs.error, push.*).
 5) Imprime cada frame con marca de tiempo + latencia estimada.

Uso (Spyder / consola):
  set MEXC_API_KEY=...
  set MEXC_API_SECRET=...
  python mexc_ws_auth_debug.py --seconds 20 --raw 1

Requisitos:
  pip install websocket-client python-dotenv
"""
from __future__ import annotations
import os, time, json, hmac, hashlib, threading
from datetime import datetime, timezone
from typing import Optional

try:
    import websocket  # websocket-client
except Exception:
    raise SystemExit("\n⚠️ Falta 'websocket-client'. Instala con: pip install websocket-client\n")

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

WS_URL = os.getenv("MEXC_WS_URL", "wss://contract.mexc.com/edge")
API_KEY = "mx0vglOEFTy9klFKJo"
API_SECRET = "1f45cf4ac48148419b59298352c45ef0"
PING_INTERVAL_SEC = int(os.getenv("MEXC_PING_SEC", "20"))

# -------------- helpers --------------

def now_ms() -> int:
    return int(time.time() * 1000)


def iso_ms(ms: int) -> str:
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f %Z")
    except Exception:
        return str(ms)


def sign_login(api_key: str, api_secret: str, req_ms: int) -> str:
    target = f"{api_key}{req_ms}"
    return hmac.new(api_secret.encode(), target.encode(), hashlib.sha256).hexdigest()


class MexcWSAuthDebug:
    def __init__(self, api_key: str, api_secret: str, debug_raw: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.debug_raw = debug_raw
        self.ws: Optional[websocket.WebSocketApp] = None
        self._stop = threading.Event()
        self._opened = threading.Event()
        self._ping_thread: Optional[threading.Thread] = None
        self._t0 = time.time()
        self._last_server_ts = None

    # ---------- lifecycle ----------
    def start(self):
        self.ws = websocket.WebSocketApp(
            WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        t = threading.Thread(target=self.ws.run_forever, kwargs={"ping_interval": 0}, daemon=True)
        t.start()
        if not self._opened.wait(timeout=10):
            raise RuntimeError("Timeout abriendo WS")
        self._ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        self._ping_thread.start()

    def stop(self):
        self._stop.set()
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass

    # ---------- ws handlers ----------
    def _on_open(self, *_):
        print(f"🌐 Conectado a {WS_URL}")
        self._opened.set()
        # 1) LOGIN CORRECTO
        ts = now_ms()
        sig = sign_login(self.api_key, self.api_secret, ts)
        login = {"method": "login", "param": {"apiKey": self.api_key, "reqTime": str(ts), "signature": sig}}
        print("\n🧾 LOGIN CORRECTO (payload):")
        print(json.dumps(login, ensure_ascii=False, indent=2))
        self._send(login)

        # 2) LOGIN INCORRECTO (firma mala) tras 1s para observar el comportamiento
        def send_bad_login():
            bad_ts = now_ms()
            bad_sig = sign_login(self.api_key + "x", self.api_secret, bad_ts)  # objetivo: firma inválida
            bad = {"method": "login", "param": {"apiKey": self.api_key, "reqTime": str(bad_ts), "signature": bad_sig}}
            print("\n🧾 LOGIN INCORRECTO (payload intencionadamente malo):")
            print(json.dumps(bad, ensure_ascii=False, indent=2))
            self._send(bad)
        threading.Timer(1.0, send_bad_login).start()

    def _on_message(self, _, message: str):
        now = now_ms()
        try:
            msg = json.loads(message)
        except Exception:
            print(f"[{iso_ms(now)}] (non-json)", message)
            return
        ch = msg.get("channel") or msg.get("topic")
        ts = msg.get("ts")
        if ts:
            self._last_server_ts = ts
        if self.debug_raw:
            print(f"[{iso_ms(now)}] <- {ch}")
            print(json.dumps(msg, ensure_ascii=False, indent=2))
            if ts:
                try:
                    rtt_ms = now - int(ts)
                    print(f"   ⏱️  latency ~ {rtt_ms} ms (client_now - server_ts)")
                except Exception:
                    pass

        if ch == "rs.login":
            # tras login correcto, pide sólo eco mínimo para ver si hay permiso
            filt = {"method": "personal.filter", "param": {"filters": [{"filter": "asset"}, {"filter": "order"}, {"filter": "position"}]}}
            print("\n📡 Enviando personal.filter (asset, order, position)…")
            self._send(filt)
        elif ch == "rs.error":
            print("❌ ERROR del servidor (posible auth fallida o filtro inválido)")

    def _on_error(self, _, err):
        print("❌ WS error:", err)

    def _on_close(self, *_):
        print("🔌 WS cerrado")

    def _send(self, payload: dict):
        try:
            if self.ws:
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


# -------------- debug runner --------------

def run(seconds: int = 20, raw: bool = True):
    if not API_KEY or not API_SECRET:
        raise SystemExit("⚠️  Configura MEXC_API_KEY y MEXC_API_SECRET en el entorno o pásalos por CLI.")

    print(f"🌐 Conectando a {WS_URL} (debug {seconds}s, raw={raw})…")
    c = MexcWSAuthDebug(API_KEY, API_SECRET, debug_raw=raw)
    c.start()
    t0 = time.time()
    try:
        while time.time() - t0 < max(5, seconds):
            time.sleep(1.0)
    finally:
        c.stop()
        print("\nℹ️  Fin del AUTH DEBUG. Si viste 'rs.login' y luego 'rs.error' para el login malo, la firma correcta está OK y el servidor valida la auth.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--raw", type=int, default=1)
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--api-secret", type=str, default=None)
    args = ap.parse_args()

    if args.api_key:
        API_KEY = args.api_key
    if args.api_secret:
        API_SECRET = args.api_secret

    run(seconds=args.seconds, raw=bool(args.raw))


