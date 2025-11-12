# 1) claves (si no las tienes en tu entorno)
import os
os.environ['96b1c999-b9df-42ed-9d4f-a4491aac7a1d'] = '<TU_API_KEY>'
os.environ['5f60a0a147d82db2117a45da129a6f4480488234'] = '<TU_API_SECRET>'

# 2) importar y (si editaste el archivo) recargar
from adapters import xtv5 as xtad
import importlib; xtad = importlib.reload(xtad)

# 3) llamar a los debugs directamente:

# # Abiertas normalizadas
# xtad.debug_dump_xt_opens()

# Funding normalizado
xtad.debug_dump_xt_funding(limit=50, days=14, symbol=None)

 

# # Preview de cerradas por FIFO (NO guarda en DB)
# xtad.debug_preview_xt_closed(days=30, symbol="btc_usdt", with_funding=True)

# Guardar cerradas en SQLite (usa tu db_manager.save_closed_position)
xtad.save_xt_closed_positions(days=30, symbol=None, inject_funding=True)
