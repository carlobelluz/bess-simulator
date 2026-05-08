"""
intake/diagnostic.py
Trasforma SDI + cache profili in strutture diagnostiche pronte per il frontend.

Non renderizza nulla. Solo calcoli e aggregazioni.

Public API:
  build_diagnostic(sdi, base_dir=".") -> dict
      Carica i profili dalla cache e produce 7 sezioni diagnostiche.
      Ogni sezione include un flag "available" che indica se i dati erano presenti.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Import load_profiles senza toccare profile_generator direttamente
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from intake.profile_generator import load_profiles

SLOT_H = 0.25  # ogni slot = 15 minuti = 0.25 ore


# ── Helpers interni ───────────────────────────────────────────────────────────

def _v(field, default=None):
    """Estrae .value da un campo SDI oppure restituisce il valore diretto."""
    if isinstance(field, dict):
        return field.get("value", default)
    return field if field is not None else default


def _source(field) -> str | None:
    if isinstance(field, dict):
        return field.get("source")
    return None


def _confidence(field) -> str | None:
    if isinstance(field, dict):
        return field.get("confidence")
    return None


def _reliability_label(level: int) -> str:
    return {1: "Molto bassa", 2: "Bassa", 3: "Media", 4: "Alta", 5: "Alta"}.get(level, "N/D")


def _data_basis_text(sdi: dict) -> str:
    billing = sdi.get("billing") or {}
    n = billing.get("n_bills", 0)
    macro = (sdi.get("meta") or {}).get("macro_case", "A")
    dq = (sdi.get("meta") or {}).get("data_quality_level", 1)
    parts = []
    if n == 0:
        parts.append("nessuna bolletta caricata")
    elif n == 1:
        parts.append("1 bolletta mensile")
    else:
        parts.append(f"{n} bollette mensili")
    if macro in ("B", "D"):
        parts.append("FV esistente")
    if dq >= 4:
        parts.append("curva di carico reale")
    return " · ".join(parts) if parts else "input manuale"


def _profile_origin_text(load_meta: dict) -> str:
    src = load_meta.get("source", "")
    if src == "synthetic_industrial":
        return "Sintetico industriale (stimato)"
    if "reconstructed" in src:
        return "Ricostruito da bolletta + profilo FV"
    if src == "real_csv":
        return "Curva di carico reale (CSV)"
    return src or "N/D"


# ── Sezione 1: Executive Snapshot ─────────────────────────────────────────────

def _build_executive_snapshot(sdi: dict, profiles: dict | None) -> dict:
    meta    = sdi.get("meta") or {}
    site_s  = sdi.get("site") or {}
    billing = sdi.get("billing") or {}
    pv_ex   = sdi.get("pv_existing") or {}
    tariff  = sdi.get("tariff_context") or {}

    dq_level = meta.get("data_quality_level", 1)
    has_pv   = bool(pv_ex.get("presente", False))
    kwp_val  = _v(pv_ex.get("kwp")) if has_pv else None

    # consumo e prelievo
    consumo_kwh = _v(site_s.get("consumo_annuo_kwh"))
    net_draw    = _v(billing.get("consumo_annuo_derivato_kwh"))

    # costo da billing (1 bolletta → ×12)
    annual_cost = None
    avg_cost    = None
    raw_bills   = billing.get("raw_bills") or []
    if raw_bills:
        costs = [b.get("costo_totale_eur") for b in raw_bills if b.get("costo_totale_eur")]
        if costs:
            n_bills = billing.get("n_bills", 1) or 1
            annual_cost = round(sum(costs) / n_bills * 12)
        costo_medio = _v(billing.get("costo_medio_energia_eur_kwh"))
        if costo_medio:
            avg_cost = round(float(costo_medio), 4)

    # produzione FV dai profili
    pv_prod_kwh = None
    if has_pv and profiles:
        _pv_raw = profiles.get("pv_kw")
        if _pv_raw is not None:
            pv_arr = np.asarray(_pv_raw)
            if len(pv_arr) > 0:
                pv_prod_kwh = round(float(pv_arr.sum()) * SLOT_H)

    return {
        "macro_case":              meta.get("macro_case", "A"),
        "data_quality_level":      dq_level,
        "reliability_label":       _reliability_label(dq_level),
        "nome_cliente":            meta.get("nome_cliente", ""),
        "comune":                  _v(site_s.get("comune")) or site_s.get("comune") or "",
        "lat":                     _v(site_s.get("lat")),
        "lon":                     _v(site_s.get("lon")),
        "annual_consumption_kwh":  consumo_kwh,
        "annual_net_draw_kwh":     net_draw,
        "has_pv":                  has_pv,
        "pv_kwp":                  float(kwp_val) if kwp_val is not None else None,
        "pv_annual_production_kwh": pv_prod_kwh,
        "contracted_power_kw":     _v(site_s.get("potenza_contrattuale_kw")),
        "peak_kw":                 _v(site_s.get("picco_potenza_kw")),
        "annual_energy_cost_eur":  annual_cost,
        "avg_cost_eur_kwh":        avg_cost,
        "spread_eur_kwh":          _v(tariff.get("supplier_spread_eur_kwh")),
    }


# ── Sezione 2: Energy Identity ────────────────────────────────────────────────

def _build_energy_identity(sdi: dict, profiles: dict | None) -> dict:
    site_s  = sdi.get("site") or {}
    billing = sdi.get("billing") or {}
    pv_ex   = sdi.get("pv_existing") or {}
    prof_s  = sdi.get("profiles") or {}
    has_pv  = bool(pv_ex.get("presente", False))

    consumo_field  = site_s.get("consumo_annuo_kwh") or {}
    consumo_kwh    = _v(consumo_field)
    grid_draw      = _v(billing.get("consumo_annuo_derivato_kwh"))

    pv_prod_kwh       = None
    pv_sc_kwh         = None
    pv_export_kwh     = None
    pv_coverage_pct   = None

    if profiles:
        _load_raw = profiles.get("load_kw")
        _pv_raw   = profiles.get("pv_kw")
        load = np.asarray(_load_raw) if _load_raw is not None else np.array([])
        pv   = np.asarray(_pv_raw)   if _pv_raw   is not None else np.array([])
        if len(load) > 0 and len(pv) > 0:
            pv_prod_kwh   = round(float(pv.sum()) * SLOT_H)
            pv_sc_kwh     = round(float(np.minimum(load, pv).sum()) * SLOT_H)
            pv_export_kwh = round(float(np.maximum(0.0, pv - load).sum()) * SLOT_H)
            if has_pv and consumo_kwh and consumo_kwh > 0:
                pv_coverage_pct = round(pv_sc_kwh / consumo_kwh * 100, 1)

    load_meta = prof_s.get("load_kw") or {}
    return {
        "site_consumption_kwh":    consumo_kwh,
        "site_consumption_source": _source(consumo_field),
        "site_consumption_conf":   _confidence(consumo_field),
        "grid_draw_kwh":           grid_draw,
        "pv_production_kwh":       pv_prod_kwh if has_pv else None,
        "pv_self_consumption_kwh": pv_sc_kwh   if has_pv else None,
        "pv_export_kwh":           pv_export_kwh if has_pv else None,
        "pv_coverage_pct":         pv_coverage_pct,
        "load_profile_origin":     _profile_origin_text(load_meta),
        "data_basis":              _data_basis_text(sdi),
    }


# ── Sezione 3: Load Behavior ──────────────────────────────────────────────────

def _build_load_behavior(profiles: dict | None) -> dict:
    if not profiles:
        return {"available": False}

    _load_raw   = profiles.get("load_kw")
    _pv_raw     = profiles.get("pv_kw")
    _month_raw  = profiles.get("month")
    load       = np.asarray(_load_raw)  if _load_raw  is not None else np.array([])
    pv         = np.asarray(_pv_raw)    if _pv_raw    is not None else np.array([])
    months_arr = np.asarray(_month_raw) if _month_raw is not None else np.array([])

    if len(load) == 0 or len(load) % 96 != 0:
        return {"available": False}

    n_days = len(load) // 96
    load_2d   = load.reshape(n_days, 96)
    _pv_full  = pv if len(pv) == len(load) else np.zeros_like(load)
    grid_2d   = np.maximum(0.0, load - _pv_full).reshape(n_days, 96)
    pv_2d     = _pv_full.reshape(n_days, 96)
    months_d  = months_arr.reshape(n_days, 96)[:, 0] if len(months_arr) == len(load) else None

    slot_labels = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 15)]

    avg_daily_load = load_2d.mean(axis=0).tolist()
    avg_daily_grid = grid_2d.mean(axis=0).tolist()
    avg_daily_pv   = pv_2d.mean(axis=0).tolist()

    seasonal = {}
    if months_d is not None:
        for name, mo in [
            ("winter", [12, 1, 2]),
            ("spring", [3, 4, 5]),
            ("summer", [6, 7, 8]),
            ("autumn", [9, 10, 11]),
        ]:
            mask = np.isin(months_d, mo)
            if mask.sum() == 0:
                seasonal[name] = {"load": avg_daily_load, "grid": avg_daily_grid, "pv": avg_daily_pv}
            else:
                seasonal[name] = {
                    "load": load_2d[mask].mean(axis=0).tolist(),
                    "grid": grid_2d[mask].mean(axis=0).tolist(),
                    "pv":   pv_2d[mask].mean(axis=0).tolist(),
                }

    return {
        "available":       True,
        "slot_labels":     slot_labels,
        "avg_daily_load":  avg_daily_load,
        "avg_daily_grid":  avg_daily_grid,
        "avg_daily_pv":    avg_daily_pv,
        "seasonal":        seasonal,
    }


# ── Sezione 4: Power Peaks ────────────────────────────────────────────────────

def _build_power_peaks(sdi: dict, profiles: dict | None) -> dict:
    site_s = sdi.get("site") or {}
    contracted = _v(site_s.get("potenza_contrattuale_kw"))

    if not profiles:
        return {
            "available":          False,
            "contracted_power_kw": contracted,
            "max_peak_kw":         _v(site_s.get("picco_potenza_kw")),
        }

    _load_raw = profiles.get("load_kw")
    load = np.asarray(_load_raw) if _load_raw is not None else np.array([])
    if len(load) == 0:
        return {"available": False, "contracted_power_kw": contracted}

    max_kw = float(load.max())
    p95    = float(np.percentile(load, 95))
    p99    = float(np.percentile(load, 99))

    hours_above_80 = None
    if contracted and contracted > 0:
        thresh = contracted * 0.8
        hours_above_80 = round(float((load > thresh).sum()) * SLOT_H, 1)

    ratio = round(max_kw / contracted, 2) if contracted and contracted > 0 else None

    return {
        "available":                 True,
        "max_peak_kw":               round(max_kw, 1),
        "contracted_power_kw":       contracted,
        "peak_to_contracted_ratio":  ratio,
        "p95_kw":                    round(p95, 1),
        "p99_kw":                    round(p99, 1),
        "hours_above_80pct_contracted": hours_above_80,
    }


# ── Sezione 5: Tariff Bands ────────────────────────────────────────────────────

def _build_tariff_bands(sdi: dict) -> dict:
    billing  = sdi.get("billing") or {}
    prof_s   = sdi.get("profiles") or {}
    load_meta = prof_s.get("load_kw") or {}

    f1 = billing.get("f1_kwh")
    f2 = billing.get("f2_kwh")
    f3 = billing.get("f3_kwh")

    if f1 is None and f2 is None and f3 is None:
        return {
            "available":                    False,
            "f1f2f3_used_in_reconstruction": load_meta.get("f1f2f3_used", False),
        }

    tot = (f1 or 0) + (f2 or 0) + (f3 or 0)
    f1_pct = round((f1 or 0) / tot * 100, 1) if tot > 0 else None
    f2_pct = round((f2 or 0) / tot * 100, 1) if tot > 0 else None
    f3_pct = round((f3 or 0) / tot * 100, 1) if tot > 0 else None

    band_draws   = load_meta.get("band_draws_kwh")
    band_conf    = load_meta.get("band_coherence_confidence") if load_meta.get("f1f2f3_used") else None
    band_warning = None
    if load_meta.get("f1f2f3_used") and load_meta.get("confidence") == "low" and band_draws:
        band_warning = "Distribuzione per fascia non perfettamente coerente con la bolletta."

    return {
        "available":                    True,
        "f1_kwh":                       f1,
        "f2_kwh":                       f2,
        "f3_kwh":                       f3,
        "f1_pct":                       f1_pct,
        "f2_pct":                       f2_pct,
        "f3_pct":                       f3_pct,
        "daytime_pct":                  f1_pct,
        "f1f2f3_used_in_reconstruction": load_meta.get("f1f2f3_used", False),
        "band_coherence_confidence":    band_conf,
        "band_mismatch_warning":        band_warning,
        "band_draws_kwh":               band_draws,
    }


# ── Sezione 6: Economic Picture ────────────────────────────────────────────────

def _build_economic_picture(sdi: dict) -> dict:
    billing = sdi.get("billing") or {}
    tariff  = sdi.get("tariff_context") or {}
    raw_bills = billing.get("raw_bills") or []

    if not raw_bills:
        return {
            "available":             False,
            "spread_eur_kwh":        _v(tariff.get("supplier_spread_eur_kwh")),
            "quota_potenza":         _v(tariff.get("quota_potenza_eur_kw_mese")),
        }

    n = billing.get("n_bills", 1) or 1

    # Somma su tutte le bollette disponibili, poi annualizza
    e_costs  = [b.get("costo_energia_eur") for b in raw_bills if b.get("costo_energia_eur")]
    p_costs  = [b.get("costo_potenza_eur")  for b in raw_bills if b.get("costo_potenza_eur")]
    t_costs  = [b.get("costo_totale_eur")   for b in raw_bills if b.get("costo_totale_eur")]

    annual_e    = round(sum(e_costs) / n * 12) if e_costs else None
    annual_p    = round(sum(p_costs) / n * 12) if p_costs else None
    annual_tot  = round(sum(t_costs) / n * 12) if t_costs else None

    e_share = round(annual_e / annual_tot * 100, 1) if annual_e and annual_tot else None
    p_share = round(annual_p / annual_tot * 100, 1) if annual_p and annual_tot else None

    avg_kwh = _v(billing.get("costo_medio_energia_eur_kwh"))

    return {
        "available":            True,
        "annual_cost_eur":      annual_tot,
        "avg_eur_kwh":          round(float(avg_kwh), 4) if avg_kwh else None,
        "energy_cost_eur":      annual_e,
        "power_cost_eur":       annual_p,
        "energy_share_pct":     e_share,
        "power_share_pct":      p_share,
        "spread_eur_kwh":       _v(tariff.get("supplier_spread_eur_kwh")),
        "spread_source":        _source(tariff.get("supplier_spread_eur_kwh")),
        "quota_potenza":        _v(tariff.get("quota_potenza_eur_kw_mese")),
        "quota_source":         _source(tariff.get("quota_potenza_eur_kw_mese")),
    }


# ── Sezione 7: Data Quality Report ────────────────────────────────────────────

_OBSERVED_SOURCES = {"bill_extracted", "user_input", "real_csv", "bill_sum"}
_DERIVED_SOURCES  = {"bill_derived", "geocoded", "extrapolated", "pvgis"}
_ESTIMATED_SOURCES = {"estimated", "estimated_from_contracted", "default",
                      "synthetic_industrial", "zeros", "synthetic"}


def _classify_source(src: str | None) -> str:
    if src is None:
        return "estimated"
    if src in _OBSERVED_SOURCES:
        return "observed"
    if src in _DERIVED_SOURCES or (src and any(src.startswith(p) for p in ("bill_derived", "geocoded", "reconstructed", "pvgis", "extrapolated"))):
        return "derived"
    if src in _ESTIMATED_SOURCES or (src and any(src.startswith(p) for p in ("synthetic", "estimated", "default", "zeros"))):
        return "estimated"
    return "derived"


def _build_data_quality_report(sdi: dict) -> dict:
    dq   = sdi.get("data_quality") or {}
    comp = sdi.get("completeness") or {}
    warn = sdi.get("warnings") or []
    assu = sdi.get("assumptions") or []
    meta = sdi.get("meta") or {}

    # Classifica i campi principali in base alla source
    key_fields = {
        "consumo_annuo_kwh":     (sdi.get("site") or {}).get("consumo_annuo_kwh"),
        "potenza_contrattuale":  (sdi.get("site") or {}).get("potenza_contrattuale_kw"),
        "picco_potenza":         (sdi.get("site") or {}).get("picco_potenza_kw"),
        "coordinate":            (sdi.get("site") or {}).get("lat"),
        "spread":                (sdi.get("tariff_context") or {}).get("supplier_spread_eur_kwh"),
        "quota_potenza":         (sdi.get("tariff_context") or {}).get("quota_potenza_eur_kw_mese"),
        "profilo_carico":        (sdi.get("profiles") or {}).get("load_kw"),
        "profilo_fv":            (sdi.get("profiles") or {}).get("pv_kw"),
    }

    observed  = []
    derived   = []
    estimated = []

    for field_name, field_val in key_fields.items():
        src = _source(field_val) if isinstance(field_val, dict) else None
        cls = _classify_source(src)
        if cls == "observed":
            observed.append(field_name)
        elif cls == "derived":
            derived.append(field_name)
        else:
            estimated.append(field_name)

    return {
        "overall_level":       meta.get("data_quality_level", 1),
        "nota_accuratezza":    dq.get("nota_accuratezza", ""),
        "load_profile_quality": dq.get("load_profile_quality", "N/D"),
        "pv_profile_quality":  dq.get("pv_profile_quality", "N/D"),
        "billing_quality":     dq.get("billing_quality", "N/D"),
        "tariff_quality":      dq.get("tariff_quality", "N/D"),
        "observed_fields":     observed,
        "derived_fields":      derived,
        "estimated_fields":    estimated,
        "completeness":        comp,
        "warnings":            warn,
        "assumptions":         assu,
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def build_diagnostic(sdi: dict, base_dir: str = ".") -> dict:
    """
    Carica i profili dalla cache e costruisce le 7 sezioni diagnostiche.

    Returns un dict con chiavi:
      executive_snapshot, energy_identity, load_behavior,
      power_peaks, tariff_bands, economic_picture, data_quality_report
    """
    profiles = load_profiles(sdi, base_dir)

    return {
        "executive_snapshot":  _build_executive_snapshot(sdi, profiles),
        "energy_identity":     _build_energy_identity(sdi, profiles),
        "load_behavior":       _build_load_behavior(profiles),
        "power_peaks":         _build_power_peaks(sdi, profiles),
        "tariff_bands":        _build_tariff_bands(sdi),
        "economic_picture":    _build_economic_picture(sdi),
        "data_quality_report": _build_data_quality_report(sdi),
    }
