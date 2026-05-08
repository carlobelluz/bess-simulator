"""
fetch_entsoe_prices.py
======================
Scarica i prezzi day-ahead IT_NORD 2024 da ENTSO-E Transparency Platform
e li salva nel formato richiesto dal motore di simulazione BESS.

UTILIZZO
--------
Opzione A — scarica via API ENTSO-E (richiede API key gratuita):
    1. Registrati su https://transparency.entsoe.eu e richiedi la API key
    2. Lancia:  python fetch_entsoe_prices.py --api-key TUA_API_KEY

Opzione B — converti un CSV scaricato manualmente:
    1. Vai su https://transparency.entsoe.eu
       Market Data → Transmission → Day-Ahead Prices
       Area: BZN|IT_NORD, Periodo: 01.01.2024 → 31.12.2024, PT60M
       Esporta come CSV
    2. Lancia:  python fetch_entsoe_prices.py --csv percorso/al/file.csv

OUTPUT
------
Genera: data/it_nord_day_ahead_2024.json
Con struttura:
{
  "_source": "ENTSO-E Transparency Platform",
  "_zone": "IT_NORD",
  "_year": 2024,
  "_unit": "EUR_MWh",
  "hours": [val_ora_0, val_ora_1, ..., val_ora_8759]
}

NOTA SU 2024 (anno bisestile)
------------------------------
2024 ha 366 giorni = 8784 ore.
Il motore di simulazione BESS usa 8760 ore = 365 giorni.
Per default questo script esclude il 29 febbraio 2024 (24 ore)
per ottenere esattamente 8760 valori.
Puoi cambiare questo comportamento con --keep-leap-day (produce 8784 valori).
"""

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# --------------------------------------------------------------------------- #
#  OPZIONE A: download via API ENTSO-E                                        #
# --------------------------------------------------------------------------- #

def fetch_via_api(api_key: str) -> list[float]:
    """
    Scarica i prezzi orari IT_NORD 2024 via API REST ENTSO-E.
    Richiede: pip install requests lxml
    """
    try:
        import requests
        from lxml import etree
    except ImportError:
        print("Installa le dipendenze: pip install requests lxml")
        sys.exit(1)

    ENTSOE_API = "https://web-api.tp.entsoe.eu/api"
    ZONE_EIC   = "10Y1001A1001A73I"   # IT_NORD

    # L'API vuole periodi mensili o brevi; facciamo richieste mensili per evitare timeout
    months = [
        ("202401010000", "202401312300"),
        ("202402010000", "202402292300"),
        ("202403010000", "202403312300"),
        ("202404010000", "202404302300"),
        ("202405010000", "202405312300"),
        ("202406010000", "202406302300"),
        ("202407010000", "202407312300"),
        ("202408010000", "202408312300"),
        ("202409010000", "202409302300"),
        ("202410010000", "202410312300"),
        ("202411010000", "202411302300"),
        ("202412010000", "202412312300"),
    ]

    all_prices: dict[datetime, float] = {}

    for start, end in months:
        params = {
            "securityToken": api_key,
            "documentType": "A44",          # Day-ahead prices
            "in_Domain":     ZONE_EIC,
            "out_Domain":    ZONE_EIC,
            "periodStart":   start,
            "periodEnd":     end,
        }
        print(f"  Scarico {start[:6]}...", end=" ", flush=True)
        resp = requests.get(ENTSOE_API, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"ERRORE HTTP {resp.status_code}")
            print(resp.text[:500])
            sys.exit(1)

        root = etree.fromstring(resp.content)
        ns = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0"}

        for ts in root.findall(".//ns:TimeSeries", ns):
            period = ts.find(".//ns:Period", ns)
            if period is None:
                continue
            start_el = period.find("ns:timeInterval/ns:start", ns)
            if start_el is None:
                continue
            # Parsing timestamp UTC
            dt_start = datetime.strptime(start_el.text, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
            for point in period.findall("ns:Point", ns):
                pos   = int(point.find("ns:position", ns).text)
                price = float(point.find("ns:price.amount", ns).text)
                dt    = dt_start + timedelta(hours=pos - 1)
                all_prices[dt] = price

        print(f"OK ({len(all_prices)} ore totali finora)")

    return _dict_to_ordered_list(all_prices)


# --------------------------------------------------------------------------- #
#  OPZIONE B: conversione CSV scaricato manualmente da ENTSO-E                #
# --------------------------------------------------------------------------- #

def parse_entsoe_csv(csv_path: str) -> list[float]:
    """
    Legge un CSV esportato da ENTSO-E Transparency Platform.
    Il formato tipico ha queste colonne:
      MTU (CET/CEST) | Day-ahead Price [EUR/MWh]
    oppure
      DateTime (UTC) | IT_NORD [EUR/MWh]
    """
    import csv

    prices: dict[datetime, float] = {}
    path = Path(csv_path)
    if not path.exists():
        print(f"File non trovato: {csv_path}")
        sys.exit(1)

    with open(path, encoding="utf-8-sig") as f:
        # Detect separator
        sample = f.read(2048)
        f.seek(0)
        sep = "\t" if "\t" in sample else ","
        reader = csv.reader(f, delimiter=sep)
        header = next(reader)
        print(f"Colonne trovate: {header}")

        # Trova la colonna del prezzo
        price_col = None
        for i, col in enumerate(header):
            if "price" in col.lower() or "eur" in col.lower() or "nord" in col.lower():
                price_col = i
                break
        if price_col is None:
            print("Impossibile trovare la colonna del prezzo. Colonne disponibili:")
            print(header)
            sys.exit(1)

        print(f"Colonna prezzo: '{header[price_col]}' (indice {price_col})")

        for row in reader:
            if not row or not row[0].strip():
                continue
            raw_dt = row[0].strip()
            raw_price = row[price_col].strip().replace(",", ".")
            if not raw_price or raw_price == "N/A" or raw_price == "-":
                continue
            try:
                price = float(raw_price)
            except ValueError:
                continue

            # Prova vari formati datetime
            dt = None
            for fmt in (
                "%d.%m.%Y %H:%M",        # ENTSO-E CET: 01.01.2024 00:00
                "%Y-%m-%dT%H:%MZ",        # ISO UTC
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
            ):
                try:
                    dt = datetime.strptime(raw_dt[:16], fmt[:len(raw_dt[:16])])
                    break
                except ValueError:
                    continue

            if dt is None:
                print(f"  Formato data non riconosciuto: '{raw_dt}' — riga saltata")
                continue

            prices[dt] = price

    print(f"Letti {len(prices)} punti orari dal CSV.")
    return _dict_to_ordered_list(prices)


# --------------------------------------------------------------------------- #
#  Funzioni di utilità                                                         #
# --------------------------------------------------------------------------- #

def _dict_to_ordered_list(prices: dict) -> list[float]:
    """Ordina per timestamp e restituisce lista."""
    sorted_keys = sorted(prices.keys())
    return [prices[k] for k in sorted_keys]


def filter_to_8760(hours: list[float], drop_leap_day: bool = True) -> list[float]:
    """
    2024 è bisestile → 8784 ore.
    Se drop_leap_day=True, rimuove le 24 ore del 29 febbraio
    per ottenere esattamente 8760 valori.
    """
    if len(hours) == 8760:
        print("Già 8760 valori — nessuna modifica necessaria.")
        return hours

    if len(hours) == 8784 and drop_leap_day:
        # Jan = 31*24=744, Feb 1-28 = 28*24=672 → offset = 744+672 = 1416
        leap_day_start = 744 + 672   # ora 1416 = inizio 29 feb
        leap_day_end   = leap_day_start + 24
        filtered = hours[:leap_day_start] + hours[leap_day_end:]
        print(f"Rimosso 29 febbraio ({leap_day_start}:{leap_day_end}). "
              f"Risultato: {len(filtered)} ore.")
        return filtered

    if len(hours) == 8784 and not drop_leap_day:
        print(f"Mantengo tutte le 8784 ore (anno bisestile).")
        return hours

    print(f"ATTENZIONE: numero ore inatteso = {len(hours)}. "
          f"Attesi 8760 o 8784. Continuo comunque.")
    return hours


def save_json(hours: list[float], output_path: str):
    data = {
        "_source": "ENTSO-E Transparency Platform",
        "_zone":   "IT_NORD",
        "_year":   2024,
        "_unit":   "EUR_MWh",
        "hours":   hours,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"\nSalvato: {path}")
    print(f"Ore totali: {len(hours)}")
    if hours:
        print(f"Min:  {min(hours):.2f} EUR/MWh")
        print(f"Max:  {max(hours):.2f} EUR/MWh")
        print(f"Media:{sum(hours)/len(hours):.2f} EUR/MWh")


# --------------------------------------------------------------------------- #
#  MAIN                                                                        #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Scarica/converte prezzi day-ahead IT_NORD 2024 per il simulatore BESS"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--api-key",
        metavar="KEY",
        help="API key ENTSO-E (registrati su transparency.entsoe.eu)"
    )
    group.add_argument(
        "--csv",
        metavar="FILE",
        help="Percorso al CSV scaricato manualmente da ENTSO-E o GME"
    )
    parser.add_argument(
        "--keep-leap-day",
        action="store_true",
        default=False,
        help="Mantieni il 29 febbraio (produce 8784 ore invece di 8760)"
    )
    parser.add_argument(
        "--output",
        default="data/it_nord_day_ahead_2024.json",
        help="Percorso output JSON (default: data/it_nord_day_ahead_2024.json)"
    )
    args = parser.parse_args()

    print("=== Fetch prezzi IT_NORD 2024 ===\n")

    if args.api_key:
        print("Modalita': download via API ENTSO-E")
        hours_raw = fetch_via_api(args.api_key)
    else:
        print(f"Modalita': conversione CSV  →  {args.csv}")
        hours_raw = parse_entsoe_csv(args.csv)

    hours = filter_to_8760(hours_raw, drop_leap_day=not args.keep_leap_day)
    save_json(hours, args.output)


if __name__ == "__main__":
    main()
