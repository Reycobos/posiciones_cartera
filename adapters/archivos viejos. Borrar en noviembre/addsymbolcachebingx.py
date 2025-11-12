# add_symbols_simple.py
import json
import time
import os

SYMBOL_CACHE_FILE = "bingx_active_symbols_cache.json"

# Los símbolos que quieres agregar (con USDT como requiere la API)
symbols_to_add = [
    "MYX-USDT", 
    "AT-USDT", 
    "ENSO-USDT", 
    "KAITO-USDT", 
    "GIGGLE-USDT"
]

# Cargar caché existente o crear uno nuevo
cache_data = {}
if os.path.exists(SYMBOL_CACHE_FILE):
    with open(SYMBOL_CACHE_FILE, 'r') as f:
        cache_data = json.load(f)

# Agregar símbolos con timestamp actual
current_time = time.time()
for symbol in symbols_to_add:
    cache_data[symbol] = current_time
    print(f"✅ {symbol} agregado al caché")

# Guardar caché actualizado
with open(SYMBOL_CACHE_FILE, 'w') as f:
    json.dump(cache_data, f, indent=2)

print(f"\n🎯 Caché actualizado con {len(cache_data)} símbolos")
print("¡Los símbolos estarán disponibles por 7 días!")

