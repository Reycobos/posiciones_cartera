#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEXC WebSocket — Orders & Asset (private)
-----------------------------------------
- Conecta al WS privado de MEXC futures: wss://contract.mexc.com/edge
- Login HMAC-SHA256 (target = apiKey + reqTime)
- Se suscribe (via personal.filter) a:
    * push.personal.order
    * push.personal.asset
- Mantiene dos snapshots en memoria:
    * asset por currency (USDT, etc.)
    * cola de últimos N orders normalizados
- Incluye DEBUG autoejecutable para Spyder:
    * imprime mensajes RAW (opcional)
    * muestra snapshot periódico y vuelca JSON final

Requisitos:
  pip install websocket-client requests python-dotenv

ENV esperadas (o pasar por CLI):
  MEXC_API_KEY
  MEXC_API_SECRET
"""

from __future__ import annotations
import os, time, hmac, hashlib, json, threading, collections
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

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
PING_INTERVAL_SEC = int(os.getenv("MEXC_PING_SEC", "20"))  # doc recomienda 10-20s
MAX_ORDERS = int(os.getenv("MEXC_DEBUG_MAX_ORDERS", "200"))

# -------------------- utils --------------------

def now_ms() -> int:
    return int(time.time() * 1000)


def iso_ms(ms: int) -> str:
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return str(ms)


def ws_signature(api_key: str, api_secret: str, ms: int) -> str:
    target = f"{api_key}{ms}"
    return hmac.new(api_secret.encode(), target.encode(), hashlib.sha256).hexdigest()


SIDE_MAP = {
    1: ("open", "long"),   # open long
    2: ("close", "short"),  # close short
    3: ("open", "short"),  # open short
    4: ("close", "long"),  # close long
}

STATE_MAP = {
    1: "uninformed",
    2: "partial",
    3: "completed",
    4: "cancelled",
    5: "invalid",
}

CATEGORY_MAP = {
    1: "limit",
    2: "system_takeover",
    3: "close_delegate",
    4: "adl_reduce",
}


class MexcWSOrdersAssets:
    def __init__(self, api_key: str, api_secret: str, debug_raw: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.debug_raw = debug_raw
        self.ws: Optional[websocket.WebSocketApp] = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._ping_thread: Optional[threading.Thread] = None
        self._thread: Optional[threading.Thread] = None
        # snapshots
        self._asset: Dict[str, Dict[str, Any]] = {}  # currency -> asset dict
        self._orders = collections.deque(maxlen=MAX_ORDERS)
        self._lock = threading.Lock()

    # -------- lifecycle --------
    def start(self):
        self.ws = websocket.WebSocketApp(
            WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
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

    # -------- handlers --------
    def _on_open(self, *_):
        ts = now_ms()
        sig = ws_signature(self.api_key, self.api_secret, ts)
        login = {
            "method": "login",
            "param": {
                "apiKey": self.api_key,
                "reqTime": str(ts),
                "signature": sig,
            },
            # "subscribe": False,  # descomentar si quieres cancelar la suscripción por defecto
        }
        self._send(login)

    def _on_message(self, _, message: str):
        try:
            msg = json.loads(message)
        except Exception:
            return
        ch = msg.get("channel") or msg.get("topic")

        if self.debug_raw and ch in ("push.personal.order", "push.personal.asset", "rs.error", "rs.login"):
            print("RAW:", json.dumps(msg, ensure_ascii=False))

        if ch == "rs.login":
            # nos quedamos solo con orders + asset
            filt = {
                "method": "personal.filter",
                "param": {"filters": [
                    {"filter": "order"},
                    {"filter": "asset"},
                ]},
            }
            self._send(filt)
            self._ready.set()
            return

        if ch == "rs.error":
            print("❌ WS error:", msg.get("data") or msg)
            return

        if ch == "push.personal.asset":
            self._handle_asset(msg.get("data", {}))
            return

        if ch == "push.personal.order":
            self._handle_order(msg.get("data", {}))
            return

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

    # -------- domain --------
    def _handle_asset(self, a: dict):
        cur = str(a.get("currency", "")).upper() or "USDT"
        asset = {
            "currency": cur,
            "available_balance": float(a.get("availableBalance", 0) or 0),
            "frozen_balance": float(a.get("frozenBalance", 0) or 0),
            "position_margin": float(a.get("positionMargin", 0) or 0),
            "cash_balance": float(a.get("cashBalance", a.get("availableBalance", 0)) or 0),
            "ts": int(a.get("updateTime") or 0),  # no siempre presente
        }
        with self._lock:
            self._asset[cur] = asset

    def _handle_order(self, o: dict):
        side_raw = int(o.get("side", 0) or 0)
        side_t = SIDE_MAP.get(side_raw, ("unknown", "unknown"))
        state = STATE_MAP.get(int(o.get("state", 0) or 0), str(o.get("state")))
        category = CATEGORY_MAP.get(int(o.get("category", 0) or 0), str(o.get("category")))

        norm = {
            "order_id": str(o.get("orderId", "")),
            "symbol": str(o.get("symbol", "")),
            "position_id": int(o.get("positionId", 0) or 0),
            "price": float(o.get("price", 0) or 0),
            "vol": float(o.get("vol", 0) or 0),
            "deal_avg_price": float(o.get("dealAvgPrice", 0) or 0),
            "deal_vol": float(o.get("dealVol", 0) or 0),
            "order_margin": float(o.get("orderMargin", 0) or 0),
            "used_margin": float(o.get("usedMargin", 0) or 0),
            "taker_fee": float(o.get("takerFee", 0) or 0),
            "maker_fee": float(o.get("makerFee", 0) or 0),
            "fee_currency": str(o.get("feeCurrency", "")),
            "profit": float(o.get("profit", 0) or 0),
            "open_type": int(o.get("openType", 0) or 0),
            "leverage": int(o.get("leverage", 0) or 0),
            "state": state,
            "category": category,
            "side_action": side_t[0],
            "side_direction": side_t[1],
            "create_time": int(o.get("createTime", 0) or 0),
            "update_time": int(o.get("updateTime", 0) or 0),
            "external_oid": str(o.get("externalOid", "")),
            "version": int(o.get("version", 0) or 0),
        }
        with self._lock:
            self._orders.append(norm)

    # -------- public API --------
    def get_asset_snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._asset)

    def get_orders_snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._orders)


# ===========================
#      DEBUG autoejecutable
# ===========================

def run_debug(seconds: int = 20, raw: bool = True):
    key = API_KEY
    sec = API_SECRET
    if not key or not sec:
        raise SystemExit("⚠️  Configura MEXC_API_KEY y MEXC_API_SECRET en el entorno o pásalos por CLI.")

    print(f"🌐 Conectando a {WS_URL} (debug {seconds}s, raw={raw})…")
    client = MexcWSOrdersAssets(key, sec, debug_raw=raw)
    client.start()

    t0 = time.time()
    try:
        while time.time() - t0 < max(5, seconds):
            assets = client.get_asset_snapshot()
            orders = client.get_orders_snapshot()
            print("\n💰 Asset snapshot:")
            for ccy, a in assets.items():
                print(f"  {ccy}: avail={a['available_balance']} frozen={a['frozen_balance']} posMargin={a['position_margin']} cash={a['cash_balance']}")
            print(f"🧾 Orders snapshot: {len(orders)} items (últimos {MAX_ORDERS})")
            for o in orders[-5:]:  # últimos 5
                print(
                    f"  • {o['symbol']} {o['side_action']}/{o['side_direction']} state={o['state']} cat={o['category']} vol={o['vol']} dealVol={o['deal_vol']} avg={o['deal_avg_price']} fee(t/m)={o['taker_fee']}/{o['maker_fee']} profit={o['profit']} time={iso_ms(o['update_time'])}"
                )
            time.sleep(3)
    finally:
        client.stop()
        print("\n====== SNAPSHOT FINAL ======")
        print("ASSET:")
        print(json.dumps(client.get_asset_snapshot(), ensure_ascii=False, indent=2))
        print("ORDERS:")
        print(json.dumps(client.get_orders_snapshot(), ensure_ascii=False, indent=2))
        print("======================================\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=20, help="Segundos a dejar el WS abierto para debug")
    ap.add_argument("--raw", type=int, default=1, help="Imprimir payloads crudos (1/0)")
    ap.add_argument("--api-key", type=str, default=None, help="Override MEXC_API_KEY")
    ap.add_argument("--api-secret", type=str, default=None, help="Override MEXC_API_SECRET")
    args = ap.parse_args()

    if args.api_key:
        API_KEY = args.api_key
    if args.api_secret:
        API_SECRET = args.api_secret

    run_debug(seconds=args.seconds, raw=bool(args.raw))
