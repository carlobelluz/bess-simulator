"""
intake/bill_extractor.py
Normalizes raw bill entries and derives tariff fields.

A "raw bill" is a dict of manually entered (or future OCR-extracted) fields.
This module produces normalized bill dicts and aggregates them into the
billing section of site_diagnostic_input.json.

Public API:
  normalize_bill(raw)                    → dict   single bill normalized + derived fields
  aggregate_bills(bills, form)           → dict   billing section for site_diagnostic
  derive_tariff_fields(billing, form)    → dict   tariff_context section for site_diagnostic

Precedence rule for consumo_annuo_kwh:
  Bills always take precedence over user-provided input.
  The form value (consumo_annuo_kwh) is used ONLY when no bills are provided (n == 0).
  When at least one bill is present, the annualized bill value is used regardless
  of what was typed into the form.
"""

from __future__ import annotations
import math


# ── Module-level stub constant ─────────────────────────────────────────────────
#
# Used in derive_tariff_fields() to estimate supplier spread from costo_medio.
# This is a temporary approximation based on the IT_NORD 2024 annual average
# (spread-free, before adding supplier margin).
#
# TODO: replace with a lookup of the actual mean from the selected price series
# file (e.g. prices/it_nord_2024.json → _stats.mean / 1000).  Until then this
# constant is intentionally visible here and must not be buried or silenced.
#
_IT_NORD_2024_MARKET_AVG_EUR_KWH: float = 0.107   # STUB — see TODO above


# ── Public API ─────────────────────────────────────────────────────────────────

def normalize_bill(raw: dict) -> dict:
    """
    Takes a raw bill dict (from form input or future OCR) and returns a
    normalized dict with all derivable fields computed.

    Required raw keys:
      consumo_kwh  : float — energy consumed in the billing period
      periodo      : str   — billing period identifier (e.g. "2024-01")

    Optional raw keys (all default to None if absent):
      potenza_contrattuale_kw, costo_energia_eur, costo_potenza_eur,
      costo_totale_eur, f1_kwh, f2_kwh, f3_kwh,
      potenza_disponibile_kw, picco_kw (MT only),
      source: "manual_entry" | "bill_ocr" | "bill_structured"

    Derived fields computed when source data is available:
      costo_medio_energia_eur_kwh = costo_energia_eur / consumo_kwh
      costo_totale_eur            = costo_energia_eur + costo_potenza_eur
                                    (only if costo_totale_eur is not provided)
    """
    consumo  = float(raw.get("consumo_kwh") or 0)
    periodo  = str(raw.get("periodo", ""))
    source   = raw.get("source", "manual_entry")

    pot_contr = _float_or_none(raw.get("potenza_contrattuale_kw"))
    costo_en  = _float_or_none(raw.get("costo_energia_eur"))
    costo_pot = _float_or_none(raw.get("costo_potenza_eur"))
    costo_tot = _float_or_none(raw.get("costo_totale_eur"))
    f1        = _float_or_none(raw.get("f1_kwh"))
    f2        = _float_or_none(raw.get("f2_kwh"))
    f3        = _float_or_none(raw.get("f3_kwh"))
    pot_disp  = _float_or_none(raw.get("potenza_disponibile_kw"))
    picco_kw  = _float_or_none(raw.get("picco_kw"))

    # Derive costo medio energia
    costo_medio = None
    if costo_en is not None and consumo > 0:
        costo_medio = round(costo_en / consumo, 4)

    # Derive costo totale only if not provided
    if costo_tot is None and costo_en is not None and costo_pot is not None:
        costo_tot = round(costo_en + costo_pot, 2)

    return {
        "periodo":                     periodo,
        "consumo_kwh":                 consumo,
        "potenza_contrattuale_kw":     pot_contr,
        "potenza_disponibile_kw":      pot_disp,
        "picco_kw":                    picco_kw,
        "costo_energia_eur":           costo_en,
        "costo_potenza_eur":           costo_pot,
        "costo_totale_eur":            costo_tot,
        "costo_medio_energia_eur_kwh": costo_medio,
        "f1_kwh":                      f1,
        "f2_kwh":                      f2,
        "f3_kwh":                      f3,
        "source":                      source,
    }


def expand_parsed_bill(parsed: dict, file_name: str = "") -> tuple[list[dict], dict]:
    """
    Converte l'output del parser multi-periodo in (lista raw bills, site_meta).

    Input: {"site": {...}, "pricing": {...}, "periods": [...]}
    Output: (bills, site_meta)
      - bills     : lista di dict compatibili con normalize_bill(), uno per periodo
      - site_meta : dict con metadati del sito e tariffe estratti dalla bolletta

    I periodi "storico" vengono inclusi: hanno consumo/picco ma costi null.
    La distinzione storico/fatturato è mantenuta in bill["source"] non influisce
    sull'aggregazione — il motore usa tutti i record indistintamente.
    """
    site    = parsed.get("site",    {}) or {}
    pricing = parsed.get("pricing", {}) or {}
    periods = parsed.get("periods", []) or []

    site_meta = {
        "nome_cliente":              site.get("nome_cliente"),
        "codice_pod":                site.get("codice_pod"),
        "indirizzo_fornitura":       site.get("indirizzo_fornitura"),
        "potenza_contrattuale_kw":   site.get("potenza_contrattuale_kw"),
        "potenza_disponibile_kw":    site.get("potenza_disponibile_kw"),
        "consumo_annuo_kwh":         site.get("consumo_annuo_kwh"),
        "spesa_annua_eur":           site.get("spesa_annua_eur"),
        "cosfi":                     site.get("cosfi"),
        "quota_potenza_eur_kw_mese": pricing.get("quota_potenza_eur_kw_mese"),
        "f1_eur_kwh":                pricing.get("f1_eur_kwh"),
        "f2_eur_kwh":                pricing.get("f2_eur_kwh"),
        "f3_eur_kwh":                pricing.get("f3_eur_kwh"),
        "f0_eur_kwh":                pricing.get("f0_eur_kwh"),
        "tipo_prezzo":               pricing.get("tipo_prezzo"),
        "quota_fissa_eur_mese":      pricing.get("quota_fissa_eur_mese"),
    }

    bills = []
    seen_periodi: set[str] = set()
    pot_contr = site_meta["potenza_contrattuale_kw"]
    for p in periods:
        raw = dict(p)

        # Propaga potenza contrattuale dal sito se non presente nel periodo
        if not raw.get("potenza_contrattuale_kw") and pot_contr:
            raw["potenza_contrattuale_kw"] = pot_contr

        # Distingui source per l'UI
        raw["source"] = "bill_fatturato" if raw.get("tipo") == "fatturato" else "bill_storico"

        # Deriva consumo_kwh da fasce se mancante ma F1+F2+F3 disponibili
        if not raw.get("consumo_kwh"):
            f1 = raw.get("f1_kwh")
            f2 = raw.get("f2_kwh")
            f3 = raw.get("f3_kwh")
            if f1 is not None and f2 is not None and f3 is not None:
                raw["consumo_kwh"] = round(f1 + f2 + f3, 1)

        # Filtra periodi completamente vuoti (nessun dato energetico)
        has_energy = raw.get("consumo_kwh") or raw.get("f1_kwh") or raw.get("f2_kwh") or raw.get("f3_kwh")
        if not has_energy:
            continue

        # Deduplicazione intra-bolletta: fatturato vince su storico per lo stesso periodo
        periodo = raw.get("periodo", "")
        if periodo and periodo in seen_periodi:
            # Sostituisci il precedente se questo è fatturato e il precedente era storico
            if raw["source"] == "bill_fatturato":
                bills = [b for b in bills if b.get("periodo") != periodo]
            else:
                continue
        if periodo:
            seen_periodi.add(periodo)

        bills.append(normalize_bill(raw))

    return bills, site_meta


def aggregate_bills(bills: list[dict], form: dict) -> dict:
    """
    Aggregates a list of normalized bills into the billing section of
    site_diagnostic_input.json.

    Precedence rule:
      If bills is non-empty, consumo_annuo is derived from the bills.
      The form's consumo_annuo_kwh is used ONLY when bills is empty (n == 0).

    Confidence assignment:
      n == 0  : "low"    (manual input, no bill)
      n == 1–2: "low"    (extrapolated × 12/n, high uncertainty)
      n == 3–11: "medium" (partial-year extrapolation)
      n >= 12 : "high"   (full-year sum, no extrapolation needed)

    form must contain at minimum:
      consumo_annuo_kwh  (fallback used only when n == 0)
    """
    n = len(bills)
    mesi_coperti = [b["periodo"] for b in bills if b.get("periodo")]

    if n == 0:
        return {
            "n_bills":      0,
            "mesi_coperti": [],
            "consumo_annuo_derivato_kwh": {
                "value":      form.get("consumo_annuo_kwh", 0),
                "source":     "user_input",
                "confidence": "low",
                "note":       "Inserito manualmente — nessuna bolletta disponibile",
            },
            "mensili_kwh":                 None,
            "f1_kwh":                      None,
            "f2_kwh":                      None,
            "f3_kwh":                      None,
            "costo_medio_energia_eur_kwh": None,
            "potenza_contrattuale_kw":     None,
            "raw_bills":                   [],
        }

    # Annualize from bills (always preferred over form input)
    total_kwh = sum(b.get("consumo_kwh", 0) for b in bills)

    if n >= 12:
        consumo_annuo = round(total_kwh)
        source_cons   = "bill_sum"
        confidence    = "high"
        note          = f"Somma di {n} bollette"
    else:
        consumo_annuo = round(total_kwh * 12 / n)
        source_cons   = f"extrapolated_x{round(12 / n, 1)}"
        confidence    = "medium" if n >= 3 else "low"
        note          = f"Extrapolato da {n} bollette × {12 / n:.1f}"

    # Contracted power: use most recent non-null value across bills
    pot_contr = next(
        (b["potenza_contrattuale_kw"] for b in reversed(bills)
         if b.get("potenza_contrattuale_kw")),
        None,
    )

    # Weighted average energy cost (weighted by consumption)
    costo_medio = _weighted_avg_cost(bills)

    # Monthly kWh breakdown: only if exactly 12 bills present
    mensili_kwh = [b.get("consumo_kwh", 0) for b in bills] if n == 12 else None

    # F1/F2/F3 annual totals: only if ALL bills provide each band
    f1 = _sum_band(bills, "f1_kwh")
    f2 = _sum_band(bills, "f2_kwh")
    f3 = _sum_band(bills, "f3_kwh")

    # Monthly breakdowns for intake charts — available for any n (not only n==12)
    sorted_bills = sorted(bills, key=lambda b: b.get("periodo") or "")

    def _periodo_to_mese(p):
        if isinstance(p, int):
            return p
        try:
            return int(str(p)[5:7])
        except (TypeError, ValueError, IndexError):
            return None

    mensili_per_fascia = [
        {
            "mese":        _periodo_to_mese(b.get("periodo")),
            "consumo_kwh": b.get("consumo_kwh"),
            "f1_kwh":      b.get("f1_kwh"),
            "f2_kwh":      b.get("f2_kwh"),
            "f3_kwh":      b.get("f3_kwh"),
        }
        for b in sorted_bills
    ]

    picchi_mensili_kw = [
        {"mese": b.get("periodo"), "picco_kw": b.get("picco_kw")}
        for b in sorted_bills
        if b.get("picco_kw") is not None
    ]

    return {
        "n_bills":      n,
        "mesi_coperti": mesi_coperti,
        "consumo_annuo_derivato_kwh": {
            "value":      consumo_annuo,
            "source":     source_cons,
            "confidence": confidence,
            "note":       note,
        },
        "mensili_kwh":                 mensili_kwh,
        "mensili_per_fascia":          mensili_per_fascia,
        "picchi_mensili_kw":           picchi_mensili_kw,
        "f1_kwh":                      f1,
        "f2_kwh":                      f2,
        "f3_kwh":                      f3,
        "costo_medio_energia_eur_kwh": costo_medio,
        "potenza_contrattuale_kw":     pot_contr,
        "raw_bills":                   bills,
    }


def derive_tariff_fields(billing: dict, form: dict) -> dict:
    """
    Derives tariff context from billing data, falling back to form inputs.

    Returns the tariff_context section of site_diagnostic_input.json.

    Args:
      billing : output of aggregate_bills()
      form    : original form dict — used for user-provided fallback values
                and for market_price_series selection

    Supplier spread derivation:
      If costo_medio_energia_eur_kwh is available from billing, spread is
      estimated as: costo_medio − _IT_NORD_2024_MARKET_AVG_EUR_KWH (stub constant).
      If spread_derived falls outside [0.0, 0.30], falls back to form value.
      Source is tagged "bill_derived" when derived, "user_input" when from form.

    Quota potenza derivation:
      If costo_potenza_eur is available in bills AND potenza_contrattuale_kw
      is known, quota is estimated as:
        total_costo_potenza / (potenza_contrattuale_kw × n_bills)
      ASSUMPTION: this assumes all billing periods are roughly the same length
      (monthly). If bills mix monthly and quarterly periods, the derived value
      will be inaccurate — confidence remains "medium" in all cases.
      Source is tagged "bill_derived" when derived, "user_input" when from form.
    """
    price_series = form.get("market_price_series", "prices/it_nord_2024.json")

    # ── Supplier spread ───────────────────────────────────────────────────────
    spread_val    = form.get("spread_eur_kwh")
    spread_source = "user_input"
    spread_conf   = "medium"

    costo_medio = billing.get("costo_medio_energia_eur_kwh")
    if costo_medio is not None:
        spread_derived = round(costo_medio - _IT_NORD_2024_MARKET_AVG_EUR_KWH, 4)
        if 0.0 <= spread_derived <= 0.30:
            spread_val    = spread_derived
            spread_source = "bill_derived"
            spread_conf   = "medium"
        # If out of range, fall through to form value without overwriting

    # ── Quota potenza ─────────────────────────────────────────────────────────
    quota_val    = form.get("quota_potenza_eur_kw_mese")
    quota_source = "user_input"
    quota_conf   = "medium"

    pot_contr = billing.get("potenza_contrattuale_kw")
    n_bills   = billing.get("n_bills", 0)
    if pot_contr and n_bills > 0:
        total_costo_pot = sum(
            b.get("costo_potenza_eur", 0) or 0
            for b in billing.get("raw_bills", [])
        )
        if total_costo_pot > 0:
            # Assumes homogeneous billing period length (see docstring)
            quota_derived = round(total_costo_pot / (pot_contr * n_bills), 2)
            if 0.5 <= quota_derived <= 50.0:
                quota_val    = quota_derived
                quota_source = "bill_derived"
                quota_conf   = "medium"

    return {
        "market_price_series": {
            "value":      price_series,
            "source":     "user_selected",
            "confidence": "medium",
        },
        "supplier_spread_eur_kwh": {
            "value":      spread_val,
            "source":     spread_source,
            "confidence": spread_conf,
        },
        "quota_potenza_eur_kw_mese": {
            "value":      quota_val,
            "source":     quota_source,
            "confidence": quota_conf,
            "note":       (
                "Derivato assumendo periodi di fatturazione omogenei (mensili). "
                "Inaffidabile se le bollette hanno durate diverse."
            ) if quota_source == "bill_derived" else None,
        },
        "tipo_contratto": form.get("tipo_contratto"),
    }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _float_or_none(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _weighted_avg_cost(bills: list[dict]) -> float | None:
    """Consumption-weighted average of costo_medio_energia_eur_kwh across bills."""
    total_cost = 0.0
    total_kwh  = 0.0
    for b in bills:
        kwh  = b.get("consumo_kwh") or 0
        cost = b.get("costo_energia_eur") or 0
        if kwh > 0 and cost > 0:
            total_cost += cost
            total_kwh  += kwh
    return round(total_cost / total_kwh, 4) if total_kwh > 0 else None


def _sum_band(bills: list[dict], key: str) -> float | None:
    """
    Sums a time-band field across all bills.
    Returns None if ANY bill is missing the field — no partial sums.
    """
    values = [b.get(key) for b in bills]
    if any(v is None for v in values):
        return None
    return round(sum(values), 1)
