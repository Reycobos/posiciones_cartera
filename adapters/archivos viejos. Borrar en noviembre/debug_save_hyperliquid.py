#!/usr/bin/env python3
# debug_save_hyperliquid.py

import os
import sys
sys.path.append('.')

from adapters.hyperliquiddeep4 import debug_hyperliquid_fifo_reconstruction

if __name__ == "__main__":
    print("🔧 Debug Hyperliquid FIFO Reconstruction")
    
    # Debug para todos los símbolos
    debug_hyperliquid_fifo_reconstruction(days=7)
    
    # Debug para símbolo específico
    # debug_hyperliquid_fifo_reconstruction(symbol="BTC", days=30)