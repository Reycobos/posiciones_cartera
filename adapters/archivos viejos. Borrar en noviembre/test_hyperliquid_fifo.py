# tests/test_hyperliquid_fifo.py
import pytest
from adapters.hyperliquiddeep import HyperliquidFIFO

def test_fifo_block_detection():
    """Test que valida la detección correcta de bloques FIFO"""
    fifo = HyperliquidFIFO()
    
    # Mock fills que forman un bloque completo
    mock_fills = [
        {"coin": "BTC", "side": "B", "sz": "0.1", "px": "45000", "time": 1000, "fee": "0.5"},
        {"coin": "BTC", "side": "B", "sz": "0.1", "px": "45100", "time": 2000, "fee": "0.5"},
        {"coin": "BTC", "side": "A", "sz": "0.2", "px": "45500", "time": 3000, "fee": "0.5"}
    ]
    
    blocks = fifo.calculate_fifo_blocks(mock_fills)
    assert len(blocks) == 1
    assert blocks[0]["symbol"] == "BTC"
    assert blocks[0]["side"] == "long"

def test_fifo_pnl_calculation():
    """Test que valida el cálculo FIFO del PnL"""
    fifo = HyperliquidFIFO()
    
    mock_block = {
        "symbol": "BTC",
        "side": "long", 
        "fills": [
            {"side": "B", "sz": "0.1", "px": "45000", "fee": "0.5"},
            {"side": "B", "sz": "0.1", "px": "45100", "fee": "0.5"},
            {"side": "A", "sz": "0.2", "px": "45500", "fee": "0.5"}
        ],
        "open_time": 1000,
        "close_time": 3000
    }
    
    result = fifo.calculate_fifo_pnl(mock_block)
    assert result["pnl"] > 0  # PnL positivo
    assert result["fee_total"] == -1.5  # Fees negativos
    assert result["size"] == 0.2  # Tamaño correcto