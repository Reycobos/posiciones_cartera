from adapters.bybitv2 import (
    debug_dump_bybit_opens,
    debug_dump_bybit_funding,
    debug_preview_bybit_closed,
    save_bybit_closed_positions,
)
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--opens", action="store_true", help="Mostrar posiciones abiertas normalizadas")
    ap.add_argument("--funding", type=int, default=0, help="Descargar funding N días y listar")
    ap.add_argument("--save-closed", action="store_true", help="Guardar closed positions (DB)")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.opens:
        debug_dump_bybit_opens()

    if args.funding > 0:
        debug_dump_bybit_funding(days=args.funding)

    if args.save_closed:
        if args.dry_run:
            debug_preview_bybit_closed(days=args.days)
        else:
            saved, dup = save_bybit_closed_positions(days=args.days, debug=False)
            print(f"✅ Closed saved: {saved} (dup omitidas: {dup})")
