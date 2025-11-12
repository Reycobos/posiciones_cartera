# debug_save_xt.py
from adapters.xt2 import save_xt_closed_positions, debug_dump_xt_opens, debug_dump_xt_funding

if __name__ == "__main__":
    debug_dump_xt_opens()
    debug_dump_xt_funding()
    save_xt_closed_positions(days=7, debug=True)


