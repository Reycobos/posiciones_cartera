@@
-load_dotenv()
-BINGX_BASE = "https://open-api.bingx.com"
+load_dotenv()
+BINGX_BASE = "https://open-api.bingx.com"
@@
-# Cache de símbolos activos
-SYMBOL_CACHE_FILE = "bingx_active_symbols_cache.json"
-SYMBOL_CACHE_TTL = 7 * 24 * 60 * 60  # 7 días
+# Cache de símbolos activos (ruta absoluta, anclada al módulo; permite override por ENV)
+MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
+CACHE_DIR = os.path.join(MODULE_DIR, ".cache")
+os.makedirs(CACHE_DIR, exist_ok=True)
+SYMBOL_CACHE_FILE = os.getenv("BINGX_SYMBOL_CACHE_FILE",
+                              os.path.join(CACHE_DIR, "bingx_active_symbols_cache.json"))
+SYMBOL_CACHE_TTL = 7 * 24 * 60 * 60  # 7 días
@@
-def _load_symbol_cache() -> dict:
-    """Carga el caché de símbolos desde archivo"""
-    try:
-        if os.path.exists(SYMBOL_CACHE_FILE):
-            with open(SYMBOL_CACHE_FILE, 'r') as f:
-                return json.load(f)
-    except Exception:
-        pass
-    return {}
+def _load_symbol_cache() -> dict:
+    """Carga el caché (con fallbacks para compatibilidad de versiones/CWD)."""
+    candidates = [
+        SYMBOL_CACHE_FILE,  # nueva ruta (absoluta)
+        os.path.join(os.getcwd(), "bingx_active_symbols_cache.json"),       # legacy CWD
+        os.path.join(MODULE_DIR, "bingx_active_symbols_cache.json"),        # legacy junto al adapter
+    ]
+    for path in candidates:
+        try:
+            if os.path.exists(path):
+                with open(path, "r") as f:
+                    data = json.load(f)
+                    # compat: si alguna vez fue lista, normalízalo a dict {sym: ts}
+                    if isinstance(data, list):
+                        now = time.time()
+                        return {s: now for s in data}
+                    return data
+        except Exception:
+            continue
+    return {}
@@
-def _update_symbol_cache(symbols: list):
-    """Actualiza el caché con nuevos símbolos activos"""
-    try:
-        current_cache = _load_symbol_cache()
-        current_time = time.time()
-        
-        for symbol in symbols:
-            current_cache[symbol] = current_time
-        
-        # Eliminar expirados
-        current_cache = {sym: ts for sym, ts in current_cache.items() 
-                        if current_time - ts <= SYMBOL_CACHE_TTL}
-        
-        with open(SYMBOL_CACHE_FILE, 'w') as f:
-            json.dump(current_cache, f)
-    except Exception as e:
-        print(f"⚠️ Error actualizando caché: {e}")
+def _update_symbol_cache(symbols: list):
+    """Actualiza el caché con nuevos símbolos activos (siempre en la ruta absoluta)."""
+    try:
+        current_cache = _load_symbol_cache()
+        current_time = time.time()
+        for symbol in symbols or []:
+            current_cache[symbol] = current_time
+        # limpia expirados
+        current_cache = {sym: ts for sym, ts in current_cache.items()
+                         if current_time - ts <= SYMBOL_CACHE_TTL}
+        os.makedirs(os.path.dirname(SYMBOL_CACHE_FILE), exist_ok=True)
+        with open(SYMBOL_CACHE_FILE, "w") as f:
+            json.dump(current_cache, f)
+    except Exception as e:
+        print(f"⚠️ Error actualizando caché: {e}")


