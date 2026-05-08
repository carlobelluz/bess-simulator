"""
intake/profile_generator.py
Generates and caches annual 15-min profiles for a site.

Public API:
  generate_profiles(sdi, base_dir) -> tuple[dict, list[str]]
      dict  = profiles section to store in sdi["profiles"]
      list  = profile-specific warnings (caller merges into sdi["warnings"])
  load_profiles(sdi, base_dir)     -> dict | None
      Returns arrays from cache, or None if cache file is missing.

Supported macro-cases:
  A — reconcile on synthetic F1/F2/F3 derived from annual consumption
  B — reconcile on real monthly F1/F2/F3 + peaks from billing
  C/D — NotImplementedError (CSV upload not yet implemented)

Internal engine: site_reconstruction.reconcile() — the only source of truth
for load profiles and self-consumption calculations.
"""

from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_SLOT_H = 0.25


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_profiles(sdi: dict, base_dir: str = ".") -> tuple[dict, list[str]]:
    """
    Generates annual 15-min profiles for the site described in sdi.

    Saves all profiles to profiles_cache/<case_id>_profiles.json.
    Returns (profiles_metadata_section, warnings_list).

    profiles_metadata_section is the dict to store at sdi["profiles"].
    warnings_list contains profile-specific warnings; the caller (case_builder)
    is responsible for merging them into sdi["warnings"].
    """
    macro_case = sdi["meta"]["macro_case"]
    case_id    = sdi["meta"]["case_id"]
    warnings: list[str] = []

    if macro_case in ("C", "D"):
        raise NotImplementedError(
            f"Case {macro_case}: il supporto alla curva di carico reale da CSV "
            "non è ancora implementato in Block 1."
        )

    # 1. Traduci sdi → SiteEnergyState
    state = _sdi_to_site_energy_state(sdi, base_dir)

    # 2. Riconcilia — unico motore di calcolo del profilo
    from intake.site_reconstruction import reconcile
    state = reconcile(state)

    # 3. Profilo prezzi (motore invariato)
    price_qh = _build_price_profile(sdi, base_dir)

    # 4. Metadati temporali per il cache
    time_arrays = _build_time_index_arrays(state.anno_riferimento)

    # 5. Salva cache nel formato atteso da load_profiles() e case_builder
    cache_rel = f"profiles_cache/{case_id}_profiles.json"
    cache_abs = os.path.join(base_dir, cache_rel)
    os.makedirs(os.path.dirname(cache_abs), exist_ok=True)
    _save_cache(
        {
            "load_kw":       state.load_profile_qh_kw,
            "pv_kw":         state.fv_profile_qh_kw,
            "price_eur_kwh": price_qh,
            **time_arrays,
            "slot_hours": 0.25,
            "n_slots":    35040,
        },
        case_id,
        cache_abs,
    )

    # 6. Sezione metadata nel formato che case_builder si aspetta
    profiles_section = {
        "cache_file":    cache_rel,
        "load_kw":       _load_meta_from_state(state),
        "pv_kw":         _pv_meta(sdi),
        "price_eur_kwh": _price_meta(sdi, base_dir),
    }

    # 7. Warning diagnostici
    avg_err = _avg_band_error(state)
    if state.overall_confidence == "low":
        warnings.append(
            f"Ricostruzione carico a confidenza bassa "
            f"(archetipo: {state.archetype_inferred}, "
            f"errore F1/F2/F3 medio: {avg_err:.1f}%)."
        )
    elif state.overall_confidence == "medium":
        warnings.append(
            f"Ricostruzione carico a confidenza media. "
            f"Archetipo inferito: {state.archetype_inferred}."
        )

    if state.has_fv and state.fabbisogno_annuo_kwh:
        if state.reconcile_mode in ("constrained_annual", "constrained_monthly"):
            warnings.append(
                f"Modalità vincolata ({state.reconcile_mode}): autoconsumo dichiarato dall'utente. "
                f"Fabbisogno annuo: {state.fabbisogno_annuo_kwh.value:,.0f} kWh "
                f"(autoconsumo: {state.autoconsumo_fv_annuo_kwh.value:,.0f} kWh, "
                f"surplus: {state.surplus_fv_annuo_kwh.value:,.0f} kWh)."
            )
        else:
            warnings.append(
                f"Consumo sito ricostruito da prelievo bolletta + autoconsumo FV stimato. "
                f"Fabbisogno annuo stimato: {state.fabbisogno_annuo_kwh.value:,.0f} kWh "
                f"(autoconsumo FV: {state.autoconsumo_fv_annuo_kwh.value:,.0f} kWh, "
                f"surplus FV: {state.surplus_fv_annuo_kwh.value:,.0f} kWh)."
            )

    # 8. Aggiorna sdi["site"]["consumo_annuo_kwh"] per Case B con il fabbisogno ricostruito.
    # (spostato da case_builder.py a qui per tenere il valore in un unico posto)
    if macro_case == "B" and state.fabbisogno_annuo_kwh:
        sdi["site"]["consumo_annuo_kwh"] = {
            "value":      round(state.fabbisogno_annuo_kwh.value),
            "source":     "reconstructed_from_bill_and_pv",
            "confidence": state.overall_confidence or "low",
            "note": (
                f"Fabbisogno energetico del sito ricostruito da bolletta + FV. "
                f"Include autoconsumo FV "
                f"({state.autoconsumo_fv_annuo_kwh.value:,.0f} kWh) "
                f"oltre al prelievo netto da bolletta."
            ),
        }

    return profiles_section, warnings


def load_profiles(sdi: dict, base_dir: str = ".") -> dict | None:
    """
    Loads profiles from the cache file referenced in sdi["profiles"]["cache_file"].
    Returns a dict with numpy arrays for each profile key, or None if missing.
    """
    cache_rel = (sdi.get("profiles") or {}).get("cache_file")
    if not cache_rel:
        return None
    cache_abs = os.path.join(base_dir, cache_rel)
    if not os.path.exists(cache_abs):
        return None
    with open(cache_abs) as f:
        raw = json.load(f)
    return {
        k: np.array(v) if isinstance(v, list) else v
        for k, v in raw.items()
        if not k.startswith("_")
    }


# ── SDI → SiteEnergyState translation ─────────────────────────────────────────

def _sdi_to_site_energy_state(sdi: dict, base_dir: str):
    """Estrae i dati dal sdi e costruisce un SiteEnergyState per reconcile()."""
    from intake.site_reconstruction import SiteEnergyState

    billing = sdi.get("billing") or {}
    site    = sdi.get("site")    or {}
    pv_ex   = sdi.get("pv_existing") or {}

    # F1/F2/F3 mensili
    f1_m, f2_m, f3_m = _extract_monthly_bands(billing)

    # Picchi mensili
    picchi_m = _extract_monthly_peaks(billing, site)

    # FV
    has_fv = bool(pv_ex.get("presente", False)) and _v(pv_ex.get("kwp"), 0) > 0
    fv_kwp     = float(_v(pv_ex.get("kwp"), 0)) if has_fv else 0.0
    fv_tilt    = float(_v(pv_ex.get("tilt"), 30)) if has_fv else 30.0
    fv_azimuth = float(_v(pv_ex.get("azimuth"), 0)) if has_fv else 0.0

    fv_oraria_kw   = None
    fv_mensile_kwh = None
    if has_fv:
        fv_oraria_kw   = _load_or_build_fv_hourly(sdi, base_dir, fv_kwp, fv_tilt, fv_azimuth)
        fv_mensile_kwh = _monthly_from_hourly(fv_oraria_kw)

    anno_rif  = _extract_reference_year(billing)
    quota_pot = float(_v_nested(sdi, "tariff_context", "quota_potenza_eur_kw_mese", default=0.0))

    # Estrai vincoli utente dal SDI (Brief 6)
    constraints = (sdi.get("pv_existing") or {}).get("user_constraints") or {}
    ctype  = constraints.get("type")
    cscope = constraints.get("scope")
    user_ac_pct_annuo    = None
    user_ac_pct_mensile  = None
    user_sup_kwh_annuo   = None
    user_sup_kwh_mensile = None
    if ctype == "autoconsumo_pct" and cscope == "annuo":
        user_ac_pct_annuo = float(constraints.get("annual_value") or 0)
    elif ctype == "autoconsumo_pct" and cscope == "mensile":
        user_ac_pct_mensile = [float(v) for v in constraints.get("monthly_values") or [0] * 12]
    elif ctype == "surplus_kwh" and cscope == "annuo":
        user_sup_kwh_annuo = float(constraints.get("annual_value") or 0)
    elif ctype == "surplus_kwh" and cscope == "mensile":
        user_sup_kwh_mensile = [float(v) for v in constraints.get("monthly_values") or [0] * 12]

    return SiteEnergyState(
        prelievo_f1_mensile=f1_m,
        prelievo_f2_mensile=f2_m,
        prelievo_f3_mensile=f3_m,
        picchi_mensili_kw=picchi_m,
        quota_potenza_eur_kw_mese=quota_pot,
        has_fv=has_fv,
        fv_kwp=fv_kwp,
        fv_tilt=fv_tilt,
        fv_azimuth=fv_azimuth,
        fv_oraria_pvgis_kw=fv_oraria_kw,
        fv_mensile_pvgis_kwh=fv_mensile_kwh,
        anno_riferimento=anno_rif,
        user_autoconsumo_pct_annuo=user_ac_pct_annuo,
        user_autoconsumo_pct_mensile=user_ac_pct_mensile,
        user_surplus_kwh_annuo=user_sup_kwh_annuo,
        user_surplus_kwh_mensile=user_sup_kwh_mensile,
    )


def _extract_monthly_bands(
    billing: dict,
) -> tuple[list[float], list[float], list[float]]:
    """
    Estrae i prelievi mensili F1/F2/F3 dal billing dict.

    Priorità:
    1. mensili_per_fascia (per-month breakdown da bollette)
    2. Aggregati annuali f1_kwh, f2_kwh, f3_kwh distribuiti su 12 mesi
    3. consumo_annuo_derivato_kwh con split tipico industriale 50/25/25
    """
    f1 = [0.0] * 12
    f2 = [0.0] * 12
    f3 = [0.0] * 12

    # Try 1: mensili_per_fascia
    fascia = billing.get("mensili_per_fascia") or []
    if fascia and isinstance(fascia, list):
        for entry in fascia:
            if not isinstance(entry, dict):
                continue
            m = entry.get("mese")
            if m is None:
                continue
            try:
                m_idx = int(m) - 1
                if not (0 <= m_idx < 12):
                    continue
                f1[m_idx] = float(entry.get("f1_kwh") or 0.0)
                f2[m_idx] = float(entry.get("f2_kwh") or 0.0)
                f3[m_idx] = float(entry.get("f3_kwh") or 0.0)
            except (TypeError, ValueError):
                continue
        months_with_data = sum(1 for i in range(12) if f1[i] > 0 or f2[i] > 0 or f3[i] > 0)
        if months_with_data >= 10:
            return f1, f2, f3
        f1 = [0.0] * 12
        f2 = [0.0] * 12
        f3 = [0.0] * 12

    # Try 2: aggregati annuali → distribuzione uniforme
    f1_tot = float(billing.get("f1_kwh") or 0.0)
    f2_tot = float(billing.get("f2_kwh") or 0.0)
    f3_tot = float(billing.get("f3_kwh") or 0.0)

    if f1_tot + f2_tot + f3_tot > 0:
        n_bills = int(billing.get("n_bills") or 0)
        if 0 < n_bills < 12:
            factor = 12.0 / n_bills
            f1_tot *= factor
            f2_tot *= factor
            f3_tot *= factor
        return [f1_tot / 12.0] * 12, [f2_tot / 12.0] * 12, [f3_tot / 12.0] * 12

    # Try 3: consumo_annuo con split tipico industriale 50/25/25
    annual = billing.get("consumo_annuo_derivato_kwh") or {}
    if isinstance(annual, dict):
        annual_kwh = float(annual.get("value") or 0.0)
    else:
        annual_kwh = float(annual or 0.0)

    if annual_kwh > 0:
        return (
            [annual_kwh * 0.50 / 12.0] * 12,
            [annual_kwh * 0.25 / 12.0] * 12,
            [annual_kwh * 0.25 / 12.0] * 12,
        )

    return f1, f2, f3


def _extract_monthly_peaks(billing: dict, site: dict) -> list[float]:
    """Estrae picchi mensili kW da billing o usa il picco di sito come fallback."""
    peaks = [0.0] * 12

    picchi = billing.get("picchi_mensili_kw") or []
    if isinstance(picchi, list):
        for entry in picchi:
            if not isinstance(entry, dict):
                continue
            m = entry.get("mese")
            if m is None:
                continue
            try:
                m_idx = int(m) - 1
                if not (0 <= m_idx < 12):
                    continue
                pk = entry.get("picco_kw")
                if pk is not None:
                    peaks[m_idx] = float(pk or 0.0)
            except (TypeError, ValueError):
                continue

    if not any(peaks):
        site_peak = float(_v(site.get("picco_potenza_kw"), 0) or 0.0)
        if site_peak > 0:
            peaks = [site_peak] * 12

    return peaks


def _load_or_build_fv_hourly(
    sdi: dict, base_dir: str, kwp: float, tilt: float, azimuth: float
) -> np.ndarray:
    """
    Ritorna profilo FV orario (8760 valori, kW).
    Usa build_pv_profile (engine.profile_builder) e aggrega da 35040 → 8760.
    """
    from engine.profile_builder import build_pv_profile
    from engine import make_time_index

    pv_ex = sdi.get("pv_existing") or {}
    site  = sdi.get("site")        or {}

    pv_dict = {
        "presente":         True,
        "kwp":              kwp,
        "profilo_source":   pv_ex.get("profilo_source", "sintetico"),
        "fv_export_regime": pv_ex.get("fv_export_regime", "nessuno"),
        "pvgis_year":       2020,
        "pvgis_tilt":       tilt,
        "pvgis_azimuth":    azimuth,
        "pvgis_losses_pct": 14,
    }
    site_dict = {
        "lat": _v(site.get("lat"), None),
        "lon": _v(site.get("lon"), None),
    }

    ti       = make_time_index()
    pv_qh_kw = build_pv_profile(pv_dict, ti, site_dict, base_dir)   # 35040 slot

    # Aggrega in orario: media dei 4 slot quartorari per ogni ora
    pv_hourly = pv_qh_kw[:35040].reshape(8760, 4).mean(axis=1)
    deg_factor = float(_v(pv_ex.get("degradation_factor"), 1.0))
    if deg_factor != 1.0:
        pv_hourly = pv_hourly * deg_factor
    return pv_hourly


def _monthly_from_hourly(fv_hourly_kw: np.ndarray) -> list[float]:
    """
    Calcola kWh mensili da un profilo orario (8760 valori, kW).
    Anno non bisestile: 365 giorni, 8760 ore.
    """
    _days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    result = []
    h = 0
    for d in _days:
        hours = d * 24
        result.append(float(fv_hourly_kw[h: h + hours].sum()))   # kW × 1h = kWh
        h += hours
    return result


def _extract_reference_year(billing: dict) -> int:
    """Inferisce l'anno di riferimento dai periodi coperti dalle bollette."""
    mesi = billing.get("mesi_coperti") or []
    if mesi and isinstance(mesi, list):
        years = []
        for p in mesi:
            if isinstance(p, str) and len(p) >= 4:
                try:
                    years.append(int(p[:4]))
                except ValueError:
                    pass
        if years:
            return max(years)

    from datetime import date
    return date.today().year - 1


# ── Helpers di supporto ────────────────────────────────────────────────────────

def _v(field, default=None):
    """Unwraps {value,...} objects or returns the scalar directly."""
    if isinstance(field, dict):
        return field.get("value", default)
    return field if field is not None else default


def _v_nested(sdi: dict, *keys, default=None):
    """Walks nested dicts; unwraps {value,...} at each step."""
    node = sdi
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k)
        if isinstance(node, dict) and "value" in node:
            node = node["value"]
    return node if node is not None else default


def _build_price_profile(sdi: dict, base_dir: str) -> np.ndarray:
    """Costruisce il profilo prezzi usando engine.profile_builder."""
    from engine.profile_builder import build_price_profile
    from engine import make_time_index

    tariff_s = sdi.get("tariff_context") or {}
    tariffs = {
        "market_price_series":     _v(tariff_s.get("market_price_series"), ""),
        "supplier_spread_eur_kwh": _v(tariff_s.get("supplier_spread_eur_kwh"), 0.0),
    }
    ti = make_time_index()
    return build_price_profile(tariffs, base_dir, ti)


def _build_time_index_arrays(year: int) -> dict:
    """Genera gli array di metadati temporali richiesti dal formato cache."""
    from engine import make_time_index
    ti = make_time_index()   # già 35040 slot, anno standard
    return {
        "month":       ti["month"],
        "hour":        ti["hour"],
        "dow":         ti["dow"],
        "slot_in_day": ti["slot_in_day"],
    }


# ── Metadata builders ──────────────────────────────────────────────────────────

def _load_meta_from_state(state) -> dict:
    """Costruisce il dizionario metadata di load_kw dal SiteEnergyState."""
    source = "reconstructed_from_bill_and_pv" if state.has_fv else "reconstructed_from_bill"

    arch_label = {
        "industrial_single_shift": "industriale singolo turno",
        "industrial_double_shift": "industriale due turni",
        "industrial_continuous":   "industriale continuo 24/7",
        "commercial_office":       "commerciale/uffici",
        "mixed":                   "carico misto",
    }.get(state.archetype_inferred, state.archetype_inferred or "non determinato")

    score_pct = f"{state.archetype_confidence_score:.0%}" if state.archetype_confidence_score else "n/d"
    method_note = (
        f"Profilo ricostruito via site_reconstruction.reconcile() in modalità auto. "
        f"Archetipo inferito: {arch_label} (confidenza {score_pct}). "
        f"Calibrazione su 36 vincoli (12 mesi × 3 fasce F1/F2/F3)."
    )

    avg_err = _avg_band_error(state)
    accuracy_note = (
        f"Errore medio sui prelievi modellati vs bolletta: {avg_err:.1f}%. "
        f"Confidenza globale: {state.overall_confidence}."
    )

    return {
        "source":        source,
        "confidence":    state.overall_confidence or "low",
        "method_note":   method_note,
        "accuracy_note": accuracy_note,
        "f1f2f3_used":   True,
        "band_draws_kwh": state.band_match_error_pct,  # dict {f1, f2, f3} → errore %
        "fabbisogno_annuo_kwh": (
            state.fabbisogno_annuo_kwh.value if state.fabbisogno_annuo_kwh else None
        ),
        "autoconsumo_fv_annuo_kwh": (
            state.autoconsumo_fv_annuo_kwh.value if state.autoconsumo_fv_annuo_kwh else None
        ),
        "surplus_fv_annuo_kwh": (
            state.surplus_fv_annuo_kwh.value if state.surplus_fv_annuo_kwh else None
        ),
        "archetype_inferred": state.archetype_inferred,
        "assumptions_active": state.assumptions_active,
        "reconcile_mode":     state.reconcile_mode,
        "autoconsumo_source": (
            state.autoconsumo_fv_annuo_kwh.source
            if state.autoconsumo_fv_annuo_kwh else "estimated"
        ),
    }


def _avg_band_error(state) -> float:
    """Errore medio assoluto % su F1/F2/F3."""
    if state.band_match_error_pct:
        errs = [abs(state.band_match_error_pct.get(b, 0.0)) for b in ("f1", "f2", "f3")]
        return float(np.mean(errs))
    return 0.0


def _pv_meta(sdi: dict) -> dict:
    """Costruisce la sezione metadata pv_kw per profiles_section."""
    pv_ex = sdi.get("pv_existing") or {}
    if not pv_ex.get("presente", False):
        return {"source": "zeros", "confidence": "high", "pvgis_params": None}

    profilo_source = pv_ex.get("profilo_source", "sintetico")
    if profilo_source == "pvgis":
        site = sdi.get("site") or {}
        return {
            "source":      "pvgis",
            "confidence":  "medium",
            "pvgis_params": {
                "year":       2020,
                "tilt":       _v(pv_ex.get("tilt"), 30),
                "azimuth":    _v(pv_ex.get("azimuth"), 0),
                "losses_pct": 14,
                "lat":        _v(site.get("lat"), None),
                "lon":        _v(site.get("lon"), None),
            },
        }
    return {"source": "synthetic", "confidence": "low", "pvgis_params": None}


def _price_meta(sdi: dict, base_dir: str) -> dict:
    series  = _v_nested(sdi, "tariff_context", "market_price_series", default="")
    has_file = bool(series) and os.path.exists(os.path.join(base_dir, series))
    if has_file:
        label = os.path.splitext(os.path.basename(series))[0]
        return {"source": label, "series_file": series, "confidence": "high"}
    return {"source": "synthetic_it_nord", "series_file": None, "confidence": "medium"}


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _save_cache(profiles: dict, case_id: str, cache_abs: str) -> None:
    out: dict = {
        "_case_id":   case_id,
        "_generated": datetime.now(timezone.utc).isoformat(),
    }
    for k, v in profiles.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    with open(cache_abs, "w") as f:
        json.dump(out, f)
