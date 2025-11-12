import pandas as pd
import requests
import time
import hashlib
from dotenv import load_dotenv
import hmac
import os
from urllib.parse import urlencode
import json


BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
BYBIT_BASE_URL = "https://api.bybit.com"







#=======================
def fetch_account_bybit():
    """Obtener balance de Bybit - VERSIÓN CORREGIDA Y SIMPLIFICADA"""
    try:
        # Configuración básica
        url = "https://api.bybit.com/v5/account/wallet-balance"
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        
        # Parámetros simples - solo accountType
        params = {"accountType": "UNIFIED"}
        
        # Crear string de parámetros para la firma (ordenado alfabéticamente)
        param_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        
        # String para firma (formato exacto que Bybit espera)
        sign_string = timestamp + BYBIT_API_KEY + recv_window + param_string
        
        #print(f"[DEBUG] Bybit sign string: '{sign_string}'")
        
        # Generar firma
        signature = hmac.new(
            BYBIT_API_SECRET.encode("utf-8"),
            sign_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
       # print(f"[DEBUG] Bybit signature: {signature}")

        # Headers
        headers = {
            "X-BAPI-API-KEY": BYBIT_API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
        }

        # Hacer el request
        full_url = f"{url}?{param_string}"
       # print(f"[DEBUG] Bybit final URL: {full_url}")
        
        r = requests.get(full_url, headers=headers, timeout=30)
       # print(f"[DEBUG] Bybit response status: {r.status_code}")
        
        data = r.json()
        #print(f"[DEBUG] Bybit full response: {data}")
        
        # Verificar respuesta
        if data.get("retCode") != 0:
            error_msg = data.get('retMsg', 'Unknown error')
            print(f"[ERROR] Bybit API error: {error_msg}")
            return None
        
        # Procesar datos
        balance_data = data.get("result", {}).get("list", [])
        if not balance_data:
            print("[WARNING] Bybit returned empty balance data")
            return None
            
        account_balance = balance_data[0]
        total_equity = float(account_balance.get("totalEquity", 0))
        total_balance = float(account_balance.get("totalWalletBalance", 0))
        
        print(f"[SUCCESS] Bybit - Equity: {total_equity}, Balance: {total_balance}")
        
        return {
            "exchange": "bybit", 
            "equity": total_equity,
            "balance": total_balance,
            "unrealized_pnl": total_equity - total_balance,
            "initial_margin": 0
        }
        
    except Exception as e:
        print(f"[ERROR] Bybit failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    