from adapters.whitebit import normalize_symbol, _mark_from_unrealized

def test_normalize_symbol():
    assert normalize_symbol("BTC_USDT") == "BTC"
    assert normalize_symbol("ETH-USDT") == "ETH"
    assert normalize_symbol("PERP_KAITOUSDC-PERP") == "KAITO"
    assert normalize_symbol("sol_perp") == "SOL"

def test_mark_from_unrealized_long():
    assert _mark_from_unrealized(100.0, 10.0, 1.0, "long") == 110.0

def test_mark_from_unrealized_short():
    assert _mark_from_unrealized(100.0, 10.0, 1.0, "short") == 90.0

def test_mark_from_unrealized_size_zero():
    assert _mark_from_unrealized(100.0, 50.0, 0.0, "long") == 100.0

