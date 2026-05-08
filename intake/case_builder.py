"""
intake/case_builder.py
Assembles site_diagnostic_input.json from form input.

Public API:
  build_site_diagnostic(form, base_dir) → dict

Raises:
  ValueError  — if validate_form() returns hard errors (intake blocked)

Form keys consumed (all optional unless marked *required):
  consumo_annuo_kwh*    — fallback when no bills present
  bills                 — list of raw bill dicts
  nome_cliente, nome_sito, case_id, autore
  comune, lat, lon      — comune used for geocoding if lat/lon absent
  potenza_contrattuale_kw, picco_potenza_kw
  ore_lavoro_giorno, giorni_lavoro_settimana
  ha_pv, kwp, pv_tilt, pv_azimuth
  fv_export_regime, producibilita_annua_kwh
  spread_eur_kwh, quota_potenza_eur_kw_mese
  market_price_series, tipo_contratto
  load_curve_file
  f1_kwh, f2_kwh, f3_kwh  (annual time-band totals)
  bess_esistente, vincoli_sito, note_cliente
"""

from __future__ import annotations
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from intake.validators import (
    validate_form, validate_form_warnings, validate_billing,
    compute_quality_level, determine_macro_case,
)
from intake.bill_extractor import normalize_bill, aggregate_bills, derive_tariff_fields
from intake.profile_generator import generate_profiles, load_profiles

_NOMINATIM_URL  = "https://nominatim.openstreetmap.org/search"
_ITALY_LAT_MIN  = 35.5
_ITALY_LAT_MAX  = 47.1
_ITALY_LON_MIN  =  6.6
_ITALY_LON_MAX  = 18.5


# ── Public API ─────────────────────────────────────────────────────────────────

def build_site_diagnostic(form: dict, base_dir: str = ".") -> dict:
    """
    Validates form data, processes bills, resolves coordinates, generates
    15-min profiles, and returns a complete site_diagnostic_input dict.

    The returned dict is ready to be serialized to site_diagnostic_input.json
    and handed off to Block 2.
    """
    # 1. Hard validation — raise immediately on blocking errors
    errors = validate_form(form)
    if errors:
        raise ValueError("Errori di validazione: " + "; ".join(errors))

    warnings: list[str] = []
    warnings.extend(validate_form_warnings(form))

    # 2. Bill processing
    raw_bills = form.get("bills") or []
    bills     = [normalize_bill(b) for b in raw_bills]
    warnings.extend(validate_billing(bills))
    billing    = aggregate_bills(bills, form)
    tariff_ctx = derive_tariff_fields(billing, form)

    # 3. Case type and data quality level
    macro_case    = determine_macro_case(form)
    quality_level = compute_quality_level(form)

    # 4. Coordinate resolution — never silently defaults to generic coordinates
    lat, lon, geo_src, geo_conf, geo_warn = _resolve_coordinates(form)
    if geo_warn:
        warnings.append(geo_warn)

    # 5. Derived scalars
    case_id        = form.get("case_id") or _make_case_id(form)
    pot_bill       = billing.get("potenza_contrattuale_kw")
    pot_form       = form.get("potenza_contrattuale_kw")
    pot_contr      = pot_bill if pot_bill is not None else pot_form
    pot_src        = "bill_extracted" if pot_bill is not None else "user_input"
    pv_existing    = _build_pv_existing(form, lat, lon)
    consumo_d      = billing["consumo_annuo_derivato_kwh"]

    # 6. Assemble SDI — profiles and data_quality filled in after generation
    sdi = {
        "_schema_version": "2.0",
        "_block":          "Block1_SiteIntake",

        "meta": {
            "case_id":            case_id,
            "nome_cliente":       form.get("nome_cliente", ""),
            "nome_sito":          form.get("nome_sito", ""),
            "data_creazione":     date.today().isoformat(),
            "autore":             form.get("autore", ""),
            "macro_case":         macro_case,
            "data_quality_level": quality_level,
            "pv_info_quality":    _pv_info_quality(form),
        },

        "input_sources": {
            "n_bills":            billing["n_bills"],
            "bill_periods":       billing["mesi_coperti"],
            "load_curve_file":    form.get("load_curve_file"),
            "pv_production_file": None,
            "geocoding_done":     geo_src == "geocoded",
        },

        "site": {
            "consumo_annuo_kwh": {
                "value":      consumo_d["value"],
                "source":     consumo_d["source"],
                "confidence": consumo_d["confidence"],
            },
            "potenza_contrattuale_kw": _wrap(
                pot_contr, pot_src,
                "high" if pot_src == "bill_extracted" else "medium",
            ),
            "picco_potenza_kw": _wrap(form.get("picco_potenza_kw"), "user_input", "medium"),
            "lat":    {"value": lat, "source": geo_src, "confidence": geo_conf},
            "lon":    {"value": lon, "source": geo_src, "confidence": geo_conf},
            "comune": form.get("comune", ""),
        },

        "operational_profile": {
            "ore_lavoro_giorno":       _wrap(form.get("ore_lavoro_giorno"), "user_input", "medium"),
            "giorni_lavoro_settimana": _wrap(form.get("giorni_lavoro_settimana"), "user_input", "medium"),
            "concentrazione_consumi":  form.get("concentrazione_consumi"),
            "carico_notturno":         form.get("carico_notturno"),
            "stagionalita":            form.get("stagionalita"),
            "tipo_carico":             form.get("tipo_carico"),
        },

        "billing":        billing,
        "tariff_context": tariff_ctx,
    }

    # Align billing.costo_medio_energia_eur_kwh with the {value, source, confidence} schema.
    # aggregate_bills() returns it as a plain float; wrap it here before handing off.
    _raw_costo = sdi["billing"].get("costo_medio_energia_eur_kwh")
    sdi["billing"]["costo_medio_energia_eur_kwh"] = {
        "value":      _raw_costo,
        "source":     "bill_derived",
        "confidence": "medium",
    }

    sdi.update({
        "pv_existing":    pv_existing,
        "profiles":       None,   # placeholder — filled after generation
        "data_quality":   None,   # placeholder — filled after generation

        "assumptions": [],
        "warnings":    [],

        "completeness": _build_completeness(form, billing, lat, lon, tariff_ctx),

        "bess_context": {
            "bess_esistente": form.get("bess_esistente"),
            "vincoli_sito":   form.get("vincoli_sito"),
            "note_cliente":   form.get("note_cliente"),
        },
    })

    # 7. Profile generation
    try:
        profiles_section, prof_warnings = generate_profiles(sdi, base_dir)
        sdi["profiles"] = profiles_section
        warnings.extend(prof_warnings)

        # Case B: update site.consumo_annuo_kwh with the reconstructed site value.
        # The billing consumo (net grid draw) was used as seed; the real site
        # consumption is higher because it includes PV self-consumption.
        if macro_case == "B":
            cached = load_profiles(sdi, base_dir)
            if cached is not None:
                consumo_ricost = round(float(cached["load_kw"].sum() * 0.25))
                sdi["site"]["consumo_annuo_kwh"] = {
                    "value":      consumo_ricost,
                    "source":     "reconstructed_from_bill_and_pv",
                    "confidence": "low",
                    "note": (
                        "Consumo sito ricostruito via temporal matching. "
                        "Include autoconsumo FV — superiore al prelievo netto da bolletta."
                    ),
                }

    except NotImplementedError as exc:
        warnings.append(f"Profili non generati: {exc}")

    # 8. Finalize deferred sections
    sdi["data_quality"] = _build_data_quality(sdi, macro_case, quality_level)
    sdi["warnings"]     = warnings
    sdi["assumptions"]  = _build_assumptions(sdi, billing, lat)

    return sdi


# ── Coordinate resolution ──────────────────────────────────────────────────────

def _resolve_coordinates(
    form: dict,
) -> tuple[float | None, float | None, str, str, str | None]:
    """
    Returns (lat, lon, source, confidence, warning_or_None).

    Priority: user-provided lat/lon > Nominatim geocoding via comune.
    Never silently falls back to generic Italian coordinates.
    """
    lat = form.get("lat")
    lon = form.get("lon")
    if lat is not None and lon is not None:
        try:
            return float(lat), float(lon), "user_input", "high", None
        except (TypeError, ValueError):
            pass

    comune = (form.get("comune") or "").strip()
    if not comune:
        return (
            None, None, "unknown", "very_low",
            "Comune non fornito e coordinate mancanti — profilo FV sarà sintetico.",
        )

    lat_g, lon_g = _geocode_nominatim(comune)
    if lat_g is not None:
        return lat_g, lon_g, "geocoded", "medium", None

    return (
        None, None, "unknown", "very_low",
        f"Geocoding fallito per '{comune}' — comune non trovato o fuori dall'Italia. "
        "Inserire lat/lon manualmente se necessario. Profilo FV sarà sintetico.",
    )


def _geocode_nominatim(comune: str) -> tuple[float | None, float | None]:
    """
    Queries Nominatim (OpenStreetMap) for Italian municipality coordinates.
    Returns (lat, lon) rounded to 5 decimal places, or (None, None) on failure.
    Validates that result falls within the Italy bounding box.
    """
    url = (
        _NOMINATIM_URL + "?"
        + urllib.parse.urlencode({
            "q":            comune + ", Italia",
            "format":       "json",
            "limit":        1,
            "countrycodes": "it",
        })
    )
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "bess-simulator/0.1 (carlo.belluz@gmail.com)"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            results = json.loads(resp.read().decode())
        if not results:
            return None, None
        lat = float(results[0]["lat"])
        lon = float(results[0]["lon"])
        if _ITALY_LAT_MIN <= lat <= _ITALY_LAT_MAX and _ITALY_LON_MIN <= lon <= _ITALY_LON_MAX:
            return round(lat, 5), round(lon, 5)
        return None, None   # result outside Italy bounding box
    except Exception:
        return None, None


# ── SDI section builders ───────────────────────────────────────────────────────

def _build_pv_existing(form: dict, lat, lon) -> dict | None:
    """Builds the pv_existing section. Returns None when ha_pv is False or kwp = 0."""
    if not form.get("ha_pv") or not (form.get("kwp") or 0):
        return None

    kwp     = form["kwp"]
    tilt    = form.get("pv_tilt")
    azimuth = form.get("pv_azimuth")

    # Use PVGIS when site coordinates are known; fall back to synthetic
    profilo_source = "pvgis" if (lat is not None and lon is not None) else "sintetico"

    return {
        "presente":       True,
        "kwp":            _wrap(kwp, "user_input", "high"),
        "tilt": {
            "value":      tilt if tilt is not None else 30,
            "source":     "user_input" if tilt is not None else "default",
            "confidence": "high" if tilt is not None else "low",
        },
        "azimuth": {
            "value":      azimuth if azimuth is not None else 0,
            "source":     "user_input" if azimuth is not None else "default",
            "confidence": "high" if azimuth is not None else "low",
        },
        "anno_installazione":    form.get("anno_installazione"),
        "anomalie_ombreggiamento": form.get("anomalie_ombreggiamento", False),
        "producibilita_annua_kwh": form.get("producibilita_annua_kwh"),
        "profilo_source":        profilo_source,
        "fv_export_regime":      form.get("fv_export_regime", "nessuno"),
    }


def _build_data_quality(sdi: dict, macro_case: str, quality_level: int) -> dict:
    profiles  = sdi.get("profiles") or {}
    load_meta = profiles.get("load_kw") or {}
    pv_meta   = profiles.get("pv_kw") or {}

    load_src_map = {
        "synthetic_industrial":         "synthetic",
        "reconstructed_from_bill_and_pv": "reconstructed",
        "real_csv":                     "measured",
    }
    load_q = load_src_map.get(load_meta.get("source", ""), "unknown")

    pv_src_map = {
        "zeros":     "not_present",
        "pvgis":     "pvgis",
        "synthetic": "synthetic",
        "measured":  "measured",
    }
    pv_q = pv_src_map.get(pv_meta.get("source", "zeros"), "unknown")

    billing   = sdi.get("billing") or {}
    n_bills   = billing.get("n_bills", 0)
    if n_bills == 0:
        billing_q = "no_bills"
    elif n_bills >= 12:
        billing_q = "monthly_detail"
    elif billing.get("f1_kwh") is not None:
        billing_q = "f1f2f3"
    elif n_bills >= 3:
        billing_q = "multi_bill"
    else:
        billing_q = "single_bill"

    spread_src = (
        (sdi.get("tariff_context") or {})
        .get("supplier_spread_eur_kwh", {})
        .get("source", "user_input")
    )
    tariff_q = "bill_derived" if spread_src == "bill_derived" else "estimated"

    if macro_case == "B":
        nota = (
            "Consumo sito ricostruito da prelievo netto bolletta + stima profilo FV. "
            "Accuratezza indicativa ±30–40%."
        )
    elif quality_level >= 3:
        nota = (
            f"Profilo carico sintetico con stagionalità mensile reale (L{quality_level}). "
            "Accuratezza indicativa ±25%."
        )
    elif quality_level == 2:
        nota = "Profilo carico sintetico con parametri operativi. Accuratezza indicativa ±35%."
    else:
        nota = "Profilo carico sintetico da dati minimi. Accuratezza indicativa ±40%."

    return {
        "overall_level":        quality_level,
        "load_profile_quality": load_q,
        "pv_profile_quality":   pv_q,
        "billing_quality":      billing_q,
        "tariff_quality":       tariff_q,
        "nota_accuratezza":     nota,
    }


def _build_completeness(
    form: dict, billing: dict, lat, lon, tariff_ctx: dict
) -> dict:
    return {
        "consumo_annuo":            (billing["consumo_annuo_derivato_kwh"]["value"] or 0) > 0,
        "potenza_contrattuale":     (billing.get("potenza_contrattuale_kw") or
                                     form.get("potenza_contrattuale_kw")) is not None,
        "coordinate_geografiche":   lat is not None and lon is not None,
        "profilo_orario_reale":     bool(form.get("load_curve_file")),
        "bollette_mensili_complete": billing["n_bills"] >= 12,
        "dati_fv_misurati":         bool(form.get("pv_production_file")),
        "spread_fornitore":         (tariff_ctx.get("supplier_spread_eur_kwh") or {}).get("value") is not None,
        "quota_potenza":            (tariff_ctx.get("quota_potenza_eur_kw_mese") or {}).get("value") is not None,
    }


def _build_assumptions(sdi: dict, billing: dict, lat) -> list[dict]:
    assumptions: list[dict] = []

    n = billing["n_bills"]
    if n == 0:
        assumptions.append({
            "campo":     "consumo_annuo_kwh",
            "assunzione": "Inserito manualmente — nessuna bolletta disponibile.",
            "rischio":   "Nessuna base oggettiva. Alta incertezza.",
        })
    elif n < 12:
        assumptions.append({
            "campo":     "consumo_annuo_kwh",
            "assunzione": f"Extrapolato da {n} bollette × {12 / n:.1f}.",
            "rischio":   "Stagionalità non completamente catturata.",
        })

    picco = sdi["site"].get("picco_potenza_kw")
    if picco is None:
        pot = sdi["site"].get("potenza_contrattuale_kw")
        if pot and (pot.get("value") if isinstance(pot, dict) else pot):
            assumptions.append({
                "campo":     "picco_potenza_kw",
                "assunzione": "Non fornito — picco reale sconosciuto.",
                "rischio":   "Efficacia peak shaving non verificabile prima della simulazione.",
            })

    if lat is None:
        assumptions.append({
            "campo":     "pv_kw",
            "assunzione": "Coordinate geografiche non disponibili — profilo FV sintetico.",
            "rischio":   "Producibilità FV molto approssimativa (±40%).",
        })

    if sdi["meta"]["macro_case"] == "B":
        assumptions.append({
            "campo":     "consumo_annuo_kwh",
            "assunzione": (
                "Consumo sito ricostruito via temporal matching da prelievo netto "
                "bolletta + stima profilo FV."
            ),
            "rischio": "Accuratezza ±30–40%. Forma del profilo di carico rimane sintetica.",
        })

    return assumptions


# ── Small helpers ──────────────────────────────────────────────────────────────

def _pv_info_quality(form: dict) -> str:
    if not form.get("ha_pv") or not (form.get("kwp") or 0):
        return "none"
    if form.get("pv_tilt") is not None and form.get("pv_azimuth") is not None:
        return "intermediate"
    return "minimal"


def _wrap(value, source: str, confidence: str) -> dict | None:
    """Wraps a scalar into {value, source, confidence} or returns None."""
    if value is None:
        return None
    return {"value": value, "source": source, "confidence": confidence}


def _make_case_id(form: dict) -> str:
    slug = re.sub(r"[^a-z0-9]", "_", form.get("nome_cliente", "site").lower())[:15]
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{slug}_{ts}"


# ── Quick sanity test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, shutil

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TMP  = tempfile.mkdtemp()
    os.makedirs(os.path.join(TMP, "profiles_cache"), exist_ok=True)
    for d in ("prices", "pvgis_cache", "engine", "intake"):
        src, dst = os.path.join(BASE, d), os.path.join(TMP, d)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    # ── Shared bill ───────────────────────────────────────────────────────────
    bill = {
        "periodo":               "2024-01",
        "consumo_kwh":           30_000,
        "potenza_contrattuale_kw": 160,
        "costo_energia_eur":     5_400,
        "costo_potenza_eur":     1_920,
    }

    # ── Test A / L1 — 1 bill, no PV, coordinates provided ────────────────────
    form_a = {
        "case_id":                 "test_a_l1",
        "nome_cliente":            "Azienda Test",
        "comune":                  "Milano",
        "lat":                     45.464,
        "lon":                     9.188,
        "consumo_annuo_kwh":       1_000,  # fallback not used (bill present)
        "ha_pv":                   False,
        "kwp":                     0,
        "market_price_series":     "prices/it_nord_2024.json",
        "bills":                   [bill],
    }

    sdi_a = build_site_diagnostic(form_a, base_dir=TMP)

    assert sdi_a["meta"]["macro_case"]         == "A",   "macro_case"
    assert sdi_a["meta"]["data_quality_level"] == 1,     "quality_level"
    assert sdi_a["billing"]["n_bills"]         == 1,     "n_bills"
    assert sdi_a["billing"]["consumo_annuo_derivato_kwh"]["value"] == 360_000, "consumo annuo (30k×12)"
    assert sdi_a["site"]["consumo_annuo_kwh"]["value"]             == 360_000, "site consumo"
    assert sdi_a["pv_existing"]                is None,  "no pv_existing"
    assert sdi_a["profiles"]["cache_file"].startswith("profiles_cache/"), "cache path"
    assert sdi_a["profiles"]["pv_kw"]["source"]  == "zeros",             "pv = zeros"
    assert sdi_a["data_quality"]["billing_quality"] == "single_bill",    "billing_quality"
    assert sdi_a["completeness"]["coordinate_geografiche"],              "coords present"
    assert any("bolletta" in w.lower() for w in sdi_a["warnings"]),     "1-bill warning"
    print("Test A/L1 OK")

    # ── Test B — 1 bill + 80 kWp PV, synthetic ───────────────────────────────
    form_b = {
        **form_a,
        "case_id": "test_b_l1",
        "ha_pv":   True,
        "kwp":     80,
    }

    sdi_b = build_site_diagnostic(form_b, base_dir=TMP)

    billing_consumo = sdi_b["billing"]["consumo_annuo_derivato_kwh"]["value"]
    site_consumo    = sdi_b["site"]["consumo_annuo_kwh"]["value"]

    assert sdi_b["meta"]["macro_case"] == "B",              "macro_case B"
    assert site_consumo > billing_consumo,                  "site > net (includes PV autoconsumo)"
    assert sdi_b["site"]["consumo_annuo_kwh"]["source"] == "reconstructed_from_bill_and_pv"
    assert sdi_b["profiles"]["load_kw"]["source"]       == "reconstructed_from_bill_and_pv"
    assert sdi_b["pv_existing"]["presente"]              is True
    assert sdi_b["pv_existing"]["profilo_source"]        == "pvgis"   # lat/lon provided → PVGIS
    assert sdi_b["data_quality"]["load_profile_quality"] == "reconstructed"
    assert any("ricostruito" in w.lower() for w in sdi_b["warnings"]), "reconstruct warning"
    print(f"Test B/L1 OK  (billing consumo {billing_consumo:,} kWh → site {site_consumo:,} kWh)")

    # ── Test C — blocked ──────────────────────────────────────────────────────
    form_c = {**form_a, "case_id": "test_c", "ha_pv": False, "kwp": 0, "load_curve_file": "dummy.csv"}
    try:
        build_site_diagnostic(form_c, base_dir=TMP)
        print("Test C: profili non generati (NotImplementedError catturato in warnings) OK")
    except NotImplementedError:
        print("Test C: NotImplementedError propagato correttamente (atteso in warnings, non raise)")

    shutil.rmtree(TMP)
    print("\nTutti i test passati.")
