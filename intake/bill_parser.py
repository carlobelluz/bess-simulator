"""
intake/bill_parser.py
Estrae campi strutturati da bollette elettriche in formato PDF.

Usa Google Gemini 2.5 Flash via API per leggere il PDF (testo o scansione)
e restituire i campi nel formato atteso da bill_extractor.expand_parsed_bill().

Public API:
  parse_bill_pdf(pdf_bytes) -> dict
      Restituisce {"site": {...}, "pricing": {...}, "periods": [...]}.
      Se l'estrazione fallisce, restituisce {"parse_error": "..."}.

  is_available() -> bool
      True se GOOGLE_API_KEY è configurata e la libreria è installata.
"""

from __future__ import annotations
import json
import os
import re

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Prompt di estrazione ───────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """
Sei un assistente specializzato nell'estrazione dati da bollette elettriche italiane.
Analizza questa bolletta e restituisci SOLO un oggetto JSON con la struttura descritta sotto.

═══════════════════════════════════════════════════════
STRUTTURA JSON RICHIESTA
═══════════════════════════════════════════════════════

{
  "site": { ... },        ← dati fissi del sito/contratto
  "pricing": { ... },     ← tariffe e prezzi dell'offerta
  "periods": [ ... ]      ← TUTTI i periodi mensili trovati nel documento
}

───────────────────────────────────────────────────────
SEZIONE "site" — dati del sito/contratto
───────────────────────────────────────────────────────
- nome_cliente          : ragione sociale o nome del cliente (stringa)
- codice_pod            : codice POD (es. "IT001E18709731")
- indirizzo_fornitura   : indirizzo del punto di fornitura (non quello di fatturazione)
- potenza_contrattuale_kw : potenza impegnata/contrattuale in kW (numero)
- potenza_disponibile_kw  : potenza disponibile in kW (numero, se diversa da contrattuale)
- consumo_annuo_kwh     : consumo annuo aggiornato totale in kWh, se esplicitamente indicato in bolletta (numero)
- spesa_annua_eur       : spesa annua sostenuta in EUR, se esplicitamente indicata (numero)
- cosfi                 : fattore di potenza (cosfi/cosφ), se presente (numero decimale es. 0.92)

───────────────────────────────────────────────────────
SEZIONE "pricing" — tariffe e prezzi offerta
───────────────────────────────────────────────────────
- tipo_prezzo           : "variabile_orario" (PUN/IPEX orario) | "fisso_fasce" (F1/F2/F3 fissi) | "fisso_monorario"
- indice                : nome dell'indice di riferimento se variabile (es. "PUN orario GME")
- f0_eur_kwh            : prezzo energia F0 unico in EUR/kWh, solo se prezzo variabile orario (numero)
- f1_eur_kwh            : prezzo energia fascia F1 in EUR/kWh, solo se tariffazione a fasce (numero)
- f2_eur_kwh            : prezzo energia fascia F2 in EUR/kWh (numero)
- f3_eur_kwh            : prezzo energia fascia F3 in EUR/kWh (numero)
- quota_fissa_eur_mese  : quota fissa di commercializzazione in EUR/mese (numero)
- quota_potenza_eur_kw_mese : tariffa totale di potenza in EUR/kW/mese (RATE, non il totale mensile)
  ATTENZIONE: cerca il valore unitario EUR/kW/mese negli "Elementi di dettaglio" o nello "Scontrino".
  Se ci sono più componenti (rete + ASOS + ARIM), somma i rate. Esempio: 2.9376 + 1.2037 + 0.3056 = 4.447

───────────────────────────────────────────────────────
SEZIONE "periods" — lista di TUTTI i periodi mensili
───────────────────────────────────────────────────────
Includi DUE tipi di record:

TIPO 1 — "fatturato": il mese oggetto di questa bolletta (ha costi completi)
TIPO 2 — "storico": ogni riga della tabella dei consumi storici passati

Per OGNI record:
- periodo     : formato "YYYY-MM" (es. MAG-2024 → "2024-05", OTT-2025 → "2025-10")
- tipo        : "fatturato" oppure "storico"
- consumo_kwh : energia attiva totale in kWh (numero)
- f1_kwh      : consumo fascia F1 in kWh (numero, null se non disponibile)
- f2_kwh      : consumo fascia F2 in kWh (numero)
- f3_kwh      : consumo fascia F3 in kWh (numero)
- picco_kw    : potenza massima registrata in kW nel mese (numero, null se non disponibile)
- costo_energia_eur  : costo componente energia in EUR (numero, null per "storico")
- costo_potenza_eur  : costo componente potenza in EUR (numero, null per "storico")
- costo_totale_eur   : importo totale bolletta in EUR (numero, null per "storico")

REGOLA CRITICA per i periodi storici:
Molte bollette contengono una tabella "Informazioni storiche" o "Consumi in kWh degli ultimi N mesi"
con una riga per ogni mese passato. Ogni riga è un period record separato con tipo "storico".
NON aggregare i mesi storici. Ogni mese = 1 record.

═══════════════════════════════════════════════════════
REGOLE GENERALI
═══════════════════════════════════════════════════════
- Restituisci SOLO il JSON, senza testo prima o dopo
- Se un campo non è presente, usa null
- I numeri non devono avere unità di misura, solo il valore numerico
- Usa il punto come separatore decimale (non la virgola)
- Se trovi "." come separatore migliaia (es. 29.000 kWh), trattalo come intero (29000)
- Non inventare valori: null se non trovi il dato con certezza

═══════════════════════════════════════════════════════
ESEMPIO OUTPUT (bolletta con 3 mesi storici + 1 mese fatturato)
═══════════════════════════════════════════════════════
{
  "site": {
    "nome_cliente": "OMC S.R.L.",
    "codice_pod": "IT001E18709731",
    "indirizzo_fornitura": "VIA VAL CHIAMPO, SNC - 36050 MONTORSO VICENTINO (VI)",
    "potenza_contrattuale_kw": 70.0,
    "potenza_disponibile_kw": 70.0,
    "consumo_annuo_kwh": 152906,
    "spesa_annua_eur": 46078.36,
    "cosfi": 0.924864
  },
  "pricing": {
    "tipo_prezzo": "variabile_orario",
    "indice": "PUN orario GME",
    "f0_eur_kwh": 0.132630,
    "f1_eur_kwh": null,
    "f2_eur_kwh": null,
    "f3_eur_kwh": null,
    "quota_fissa_eur_mese": 12.00,
    "quota_potenza_eur_kw_mese": 4.447
  },
  "periods": [
    {
      "periodo": "2024-05", "tipo": "storico",
      "consumo_kwh": 6333, "f1_kwh": 1887, "f2_kwh": 1820, "f3_kwh": 2626,
      "picco_kw": 40.20,
      "costo_energia_eur": null, "costo_potenza_eur": null, "costo_totale_eur": null
    },
    {
      "periodo": "2024-06", "tipo": "storico",
      "consumo_kwh": 6114, "f1_kwh": 1318, "f2_kwh": 1907, "f3_kwh": 2889,
      "picco_kw": 41.70,
      "costo_energia_eur": null, "costo_potenza_eur": null, "costo_totale_eur": null
    },
    {
      "periodo": "2025-10", "tipo": "fatturato",
      "consumo_kwh": 10112, "f1_kwh": 4015, "f2_kwh": 2856, "f3_kwh": 3241,
      "picco_kw": 90.70,
      "costo_energia_eur": 2218.25, "costo_potenza_eur": 403.34, "costo_totale_eur": 3059.28
    }
  ]
}
"""

# ── Costanti di validazione ────────────────────────────────────────────────────

_SITE_FIELDS_NUMERIC = {
    "potenza_contrattuale_kw", "potenza_disponibile_kw",
    "consumo_annuo_kwh", "spesa_annua_eur", "cosfi",
}
_SITE_FIELDS_STRING = {
    "nome_cliente", "codice_pod", "indirizzo_fornitura",
}

_PRICING_FIELDS_NUMERIC = {
    "f0_eur_kwh", "f1_eur_kwh", "f2_eur_kwh", "f3_eur_kwh",
    "quota_fissa_eur_mese", "quota_potenza_eur_kw_mese",
}
_PRICING_FIELDS_STRING = {
    "tipo_prezzo", "indice",
}

_PERIOD_FIELDS_NUMERIC = {
    "consumo_kwh", "f1_kwh", "f2_kwh", "f3_kwh", "picco_kw",
    "costo_energia_eur", "costo_potenza_eur", "costo_totale_eur",
}
_PERIOD_FIELDS_STRING = {
    "periodo", "tipo",
}


# ── Public API ─────────────────────────────────────────────────────────────────

def is_available() -> bool:
    """True se GOOGLE_API_KEY è configurata e google-genai è installato."""
    if not os.environ.get("GOOGLE_API_KEY"):
        return False
    try:
        import google.genai
        return True
    except ImportError:
        return False


def parse_bill_pdf(pdf_bytes: bytes) -> dict:
    """
    Estrae dati strutturati da una bolletta elettrica PDF.

    Restituisce:
      {"site": {...}, "pricing": {...}, "periods": [...]}

    In caso di errore:
      {"parse_error": "descrizione errore"}
    """
    if not is_available():
        return {"parse_error": "GOOGLE_API_KEY non configurata o libreria mancante."}

    try:
        raw_json = _call_gemini(pdf_bytes)
        extracted = _parse_json_response(raw_json)
        return extracted
    except Exception as e:
        return {"parse_error": str(e)}


# ── Chiamata Gemini ────────────────────────────────────────────────────────────

_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


def _call_gemini(pdf_bytes: bytes) -> str:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError
    import time

    client  = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

    last_error = None
    for model in _GEMINI_MODELS:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[pdf_part, _EXTRACTION_PROMPT],
                    config=types.GenerateContentConfig(temperature=0),
                )
                return response.text
            except ClientError as e:
                last_error = e
                code = e.code if hasattr(e, "code") else 0
                if code == 503:
                    time.sleep(5 * (attempt + 1))
                    continue
                if code == 429:
                    break
                raise

    raise RuntimeError(
        f"Tutti i modelli Gemini non disponibili. Ultimo errore: {last_error}. "
        "Riprova tra qualche minuto."
    ) from last_error


# ── Parse risposta JSON ────────────────────────────────────────────────────────

def _parse_json_response(text: str) -> dict:
    """
    Estrae e valida il JSON dalla risposta di Gemini.
    Gestisce blocchi markdown e normalizza tutti i valori numerici.
    Restituisce {"site": {...}, "pricing": {...}, "periods": [...]}.
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()

    data = json.loads(text)

    site    = _normalize_section(data.get("site",    {}), _SITE_FIELDS_NUMERIC,    _SITE_FIELDS_STRING)
    pricing = _normalize_section(data.get("pricing", {}), _PRICING_FIELDS_NUMERIC, _PRICING_FIELDS_STRING)

    raw_periods = data.get("periods", [])
    if not isinstance(raw_periods, list):
        raise ValueError("Campo 'periods' non è una lista")

    periods = [
        _normalize_section(p, _PERIOD_FIELDS_NUMERIC, _PERIOD_FIELDS_STRING)
        for p in raw_periods
        if isinstance(p, dict)
    ]

    # Almeno 1 period obbligatorio
    if not periods:
        raise ValueError("Nessun periodo trovato nella risposta")

    return {"site": site, "pricing": pricing, "periods": periods}


def _normalize_section(
    raw: dict,
    numeric_fields: set[str],
    string_fields: set[str],
) -> dict:
    """Normalizza una sezione del JSON: converte numeri e stringhe."""
    result = {}
    for field in numeric_fields | string_fields:
        val = raw.get(field)
        if val is None:
            result[field] = None
        elif field in string_fields:
            result[field] = str(val).strip() if val else None
        else:
            result[field] = _parse_number(val)
    return result


def _parse_number(val) -> float | None:
    """Converte un valore in float, gestendo formati italiani (1.234,56 → 1234.56)."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        import math
        return float(val) if math.isfinite(float(val)) else None
    try:
        cleaned = str(val).strip()
        if "." in cleaned and "," in cleaned:
            # formato italiano: 1.234,56
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", ".")
        return float(cleaned)
    except (TypeError, ValueError):
        return None


# ── Test rapido ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if not is_available():
        print("GOOGLE_API_KEY non configurata. Aggiungi al file .env")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Uso: python intake/bill_parser.py percorso/bolletta.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    print(f"Invio {pdf_path} a Gemini...")
    result = parse_bill_pdf(pdf_bytes)

    if "parse_error" in result:
        print(f"\nERRORE: {result['parse_error']}")
        sys.exit(1)

    print("\n── SITE ──")
    for k, v in result["site"].items():
        if v is not None:
            print(f"  {k:30s} = {v}")

    print("\n── PRICING ──")
    for k, v in result["pricing"].items():
        if v is not None:
            print(f"  {k:30s} = {v}")

    print(f"\n── PERIODS ({len(result['periods'])} trovati) ──")
    for p in result["periods"]:
        tipo = p.get("tipo", "?")
        periodo = p.get("periodo", "?")
        kwh = p.get("consumo_kwh")
        picco = p.get("picco_kw")
        f1 = p.get("f1_kwh")
        f2 = p.get("f2_kwh")
        f3 = p.get("f3_kwh")
        costo = p.get("costo_totale_eur")
        def _fmt(v, decimals=0):
            return f"{v:,.{decimals}f}" if v is not None else "—"
        print(f"  [{tipo:10s}] {periodo}  {_fmt(kwh):>10} kWh  "
              f"F1:{_fmt(f1)}  F2:{_fmt(f2)}  F3:{_fmt(f3)}  "
              f"picco:{_fmt(picco,1)} kW  "
              f"totale:{_fmt(costo,2)} EUR")
