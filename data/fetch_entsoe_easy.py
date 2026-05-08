"""
fetch_entsoe_easy.py
====================
Versione semplificata che usa la libreria 'entsoe-py' (pip install entsoe-py).
E' il modo piu' comodo per scaricare i prezzi IT_NORD 2024.

SETUP
-----
    pip install entsoe-py pandas

UTILIZZO
--------
    python fetch_entsoe_easy.py --api-key TUA_API_KEY

L'API key la ottieni registrandoti gratis su:
    https://transparency.entsoe.eu  →  My Account → API Security Token

OUTPUT
------
    data/it_nord_day_ahead_2024.json

NOTA: 2024 e' bisestile (8784 ore). Lo script rimuove il 29 feb
per ottenere 8760 ore compatibili con il motore BESS.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="API key ENTSO-E")
    parser.add_argument("--keep-leap-day", action="store_true")
    parser.add_argument("--output", default="data/it_nord_day_ahead_2024.json")
    args = parser.parse_args()

    try:
        import pandas as pd
        from entsoe import EntsoePandasClient
    except ImportError:
        print("Installa le dipendenze:")
        print("  pip install entsoe-py pandas")
        sys.exit(1)

    print("Connessione a ENTSO-E...")
    client = EntsoePandasClient(api_key=args.api_key)

    start = pd.Timestamp("2024-01-01", tz="Europe/Rome")
    end   = pd.Timestamp("2025-01-01", tz="Europe/Rome")

    print("Download prezzi IT_NORD 2024 (puo' richiedere 30-60 secondi)...")
    try:
        prices = client.query_day_ahead_prices(
            country_code="IT_NORD",
            start=start,
            end=end,
        )
    except Exception as e:
        print(f"Errore API: {e}")
        sys.exit(1)

    print(f"Scaricati {len(prices)} punti orari.")
    print(f"Primo: {prices.index[0]}  →  {prices.iloc[0]:.2f} EUR/MWh")
    print(f"Ultimo: {prices.index[-1]}  →  {prices.iloc[-1]:.2f} EUR/MWh")

    # Converti in EUR/MWh (ENTSO-E restituisce gia' EUR/MWh)
    price_list = prices.tolist()

    # Rimuovi 29 febbraio se richiesto (default: si')
    if not args.keep_leap_day and len(price_list) == 8784:
        # 744 ore gennaio + 672 ore febbraio 1-28
        leap_start = 744 + 672
        price_list = price_list[:leap_start] + price_list[leap_start + 24:]
        print(f"Rimosso 29 febbraio. Ore finali: {len(price_list)}")
    elif len(price_list) not in (8760, 8784):
        print(f"ATTENZIONE: numero ore inatteso = {len(price_list)}")

    output = {
        "_source": "ENTSO-E Transparency Platform",
        "_zone":   "IT_NORD",
        "_year":   2024,
        "_unit":   "EUR_MWh",
        "hours":   price_list,
    }

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    print(f"\nSalvato: {path}")
    print(f"Ore:     {len(price_list)}")
    print(f"Min:     {min(price_list):.2f} EUR/MWh")
    print(f"Max:     {max(price_list):.2f} EUR/MWh")
    print(f"Media:   {sum(price_list)/len(price_list):.2f} EUR/MWh")


if __name__ == "__main__":
    main()
