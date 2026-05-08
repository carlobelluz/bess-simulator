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
  A — synthetic load, zeros PV
  B — temporal matching reconstruction (binary search)
  C/D — NotImplementedError (CSV upload not yet implemented)
"""

from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

# Ensure project root is on sys.path so `engine` is importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.profile_builder import build_all_profiles
from intake.tariff_bands import build_band_masks, validate_band_reconstruction

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

    pb_case = _sdi_to_pb_case(sdi)

    if macro_case == "B":
        profiles, load_meta, recon_warnings = _reconstruct_case_b(pb_case, sdi, base_dir)
        warnings.append(
            "Case B: consumo sito ricostruito da prelievo netto bolletta + stima profilo FV. "
            "Accuratezza indicativa ±30–40%."
        )
        warnings.extend(recon_warnings)
    else:
        # Case A (and any future extension): standard synthetic path
        profiles = build_all_profiles(pb_case, base_dir)
        load_meta = _load_meta_case_a(pb_case)

    # Save cache
    cache_rel = f"profiles_cache/{case_id}_profiles.json"
    cache_abs = os.path.join(base_dir, cache_rel)
    os.makedirs(os.path.dirname(cache_abs), exist_ok=True)
    _save_cache(profiles, case_id, cache_abs)

    profiles_section = {
        "cache_file":    cache_rel,
        "load_kw":       load_meta,
        "pv_kw":         _pv_meta(sdi, pb_case),
        "price_eur_kwh": _price_meta(sdi, base_dir),
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


# ── Case B reconstruction ──────────────────────────────────────────────────────

def _reconstruct_case_b(
    pb_case: dict, sdi: dict, base_dir: str
) -> tuple[dict, dict, list[str]]:
    """
    Temporal matching reconstruction for Case B (existing PV, no real curve).

    Algorithm:
      1. Call build_all_profiles to get the synthetic load shape and PV profile.
         The load is initially scaled to consumo_netto (net grid draw from billing),
         which is used as the seed shape — magnitude will be corrected in step 3.
      2. Binary-search for scaling factor k such that:
             Σ max(0, k·load_raw[t] − pv[t]) × 0.25h = consumo_netto_bolletta
      3. Override load_kw = k × load_raw.
      4. If billing F1/F2/F3 values are present: validate the reconstructed
         per-band grid draw against billed values and update confidence accordingly.

    Returns (profiles_dict, load_meta_dict, extra_warnings_list).
    """
    consumo_netto   = _get_consumo_netto(sdi)
    extra_warnings: list[str] = []

    # Single build_all_profiles call: gets load shape + PV + price + time index
    full     = build_all_profiles(pb_case, base_dir)
    load_raw = full["load_kw"]  # scaled to consumo_netto (good seed shape)
    pv_kw    = full["pv_kw"]

    # Binary search for k
    k = _find_k_bisect(load_raw, pv_kw, consumo_netto)

    # Override load_kw with reconstructed version
    full["load_kw"] = k * load_raw

    load_meta = {
        "source":        "reconstructed_from_bill_and_pv",
        "confidence":    "low",
        "f1f2f3_used":   False,
        "method_note": (
            f"Temporal matching: fattore k={k:.4f} × load sintetico. "
            f"Target prelievo netto da bolletta: {consumo_netto:,.0f} kWh/anno. "
            f"Consumo sito ricostruito: {k * consumo_netto:,.0f} kWh/anno."
        ),
        "accuracy_note": "Accuratezza indicativa ±30–40%. Consumo sito include autoconsumo FV.",
    }

    # F1/F2/F3 band validation (if available from billing)
    billing = sdi.get("billing") or {}
    f1 = _get_band_kwh(billing, "f1_kwh")
    f2 = _get_band_kwh(billing, "f2_kwh")
    f3 = _get_band_kwh(billing, "f3_kwh")

    # Annualize: billing stores raw sum across n bills (same logic as consumo_annuo)
    if f1 is not None and f2 is not None and f3 is not None:
        n_bills = billing.get("n_bills", 0)
        if 0 < n_bills < 12:
            factor = 12.0 / n_bills
            f1, f2, f3 = f1 * factor, f2 * factor, f3 * factor

    if f1 is not None and f2 is not None and f3 is not None:
        year  = _infer_profile_year(len(full["load_kw"]))
        masks = build_band_masks(year)
        confidence, band_warning, band_draws = validate_band_reconstruction(
            full["load_kw"], pv_kw, masks, f1, f2, f3
        )
        load_meta["confidence"]     = confidence
        load_meta["f1f2f3_used"]    = True
        load_meta["band_draws_kwh"] = band_draws
        if band_warning:
            extra_warnings.append(band_warning)

    return full, load_meta, extra_warnings


def _find_k_bisect(
    load_raw: np.ndarray,
    pv_kw: np.ndarray,
    target_kwh: float,
    n_iter: int = 50,
) -> float:
    """
    Returns k ≥ 1 such that sum(max(0, k·load_raw − pv_kw)) × 0.25h ≈ target_kwh.

    k_lo = 1.0: when k=1, load_raw sums to target_kwh (profile_builder scaling).
    With any PV, grid_draw(k=1) ≤ target_kwh → k > 1 for Case B.
    k_hi starts at 4.0 and doubles until grid_draw(k_hi) > target_kwh.
    """
    if target_kwh <= 0:
        return 1.0

    def _grid(k: float) -> float:
        return float(np.maximum(k * load_raw - pv_kw, 0.0).sum() * _SLOT_H)

    k_lo = 1.0
    k_hi = 4.0
    while _grid(k_hi) < target_kwh:
        k_hi *= 2.0
        if k_hi > 200.0:
            # PV is enormous relative to load — return best upper bound
            return k_hi

    for _ in range(n_iter):
        k_mid = (k_lo + k_hi) / 2.0
        if _grid(k_mid) < target_kwh:
            k_lo = k_mid
        else:
            k_hi = k_mid

    return (k_lo + k_hi) / 2.0


def _get_consumo_netto(sdi: dict) -> float:
    """Net grid draw from billing (target for Case B binary search)."""
    annual = (sdi.get("billing") or {}).get("consumo_annuo_derivato_kwh") or {}
    if isinstance(annual, dict):
        return float(annual.get("value") or 0.0)
    return float(annual or 0.0)


def _get_band_kwh(billing: dict, key: str) -> float | None:
    """Extracts a F1/F2/F3 value from billing dict. Returns None if missing or zero."""
    v = billing.get(key)
    if v is None:
        return None
    return float(v) if float(v) > 0 else None


def _infer_profile_year(n_slots: int) -> int:
    """
    Infers a representative year for tariff-band masks from profile length.
    35040 slots → 365 days → 2023 (non-leap).
    35136 slots → 366 days → 2024 (leap).
    Falls back to 2023 for any unexpected length.
    """
    if n_slots == 35136:
        return 2024
    return 2023


# ── SDI → profile_builder case format ─────────────────────────────────────────

def _sdi_to_pb_case(sdi: dict) -> dict:
    """
    Translates a site_diagnostic_input dict into the profile_builder case format.

    SDI fields follow the {value, source, confidence} pattern.
    profile_builder expects plain scalar values.
    pv_existing.presente and pv_existing.profilo_source are plain values (not wrapped).
    pv_existing.kwp, .tilt, .azimuth are {value,...} objects.
    """
    def _v(field, default=None):
        if isinstance(field, dict):
            return field.get("value", default)
        return field if field is not None else default

    site_s   = sdi.get("site") or {}
    op_s     = sdi.get("operational_profile") or {}
    tariff_s = sdi.get("tariff_context") or {}
    pv_ex    = sdi.get("pv_existing") or {}

    has_pv = bool(pv_ex.get("presente")) and (_v(pv_ex.get("kwp"), 0) or 0) > 0

    pb_site = {
        "consumo_annuo_kwh":      _v(site_s.get("consumo_annuo_kwh"), 0),
        "lat":                     _v(site_s.get("lat")),
        "lon":                     _v(site_s.get("lon")),
        "ore_lavoro_giorno":       _v(op_s.get("ore_lavoro_giorno"), 10),
        "giorni_lavoro_settimana": _v(op_s.get("giorni_lavoro_settimana"), 5),
    }

    pb_pv = {
        "presente":         has_pv,
        "kwp":              _v(pv_ex.get("kwp"), 0) if has_pv else 0,
        "profilo_source":   pv_ex.get("profilo_source", "sintetico") if has_pv else "sintetico",
        "fv_export_regime": pv_ex.get("fv_export_regime", "nessuno") if has_pv else "nessuno",
        "pvgis_year":       2020,
        "pvgis_tilt":       _v(pv_ex.get("tilt"), 30) if has_pv else 30,
        "pvgis_azimuth":    _v(pv_ex.get("azimuth"), 0) if has_pv else 0,
        "pvgis_losses_pct": 14,
    }

    pb_tariffs = {
        "market_price_series":     _v(tariff_s.get("market_price_series"), ""),
        "supplier_spread_eur_kwh": _v(tariff_s.get("supplier_spread_eur_kwh"), 0.0),
    }

    return {"site": pb_site, "pv": pb_pv, "tariffs": pb_tariffs}


# ── Metadata builders ──────────────────────────────────────────────────────────

def _load_meta_case_a(pb_case: dict) -> dict:
    site = pb_case["site"]
    return {
        "source":        "synthetic_industrial",
        "confidence":    "low",
        "method_note": (
            f"Profilo sintetico industriale — "
            f"{site['consumo_annuo_kwh']:,.0f} kWh/anno, "
            f"{site.get('ore_lavoro_giorno', 10)}h/gg, "
            f"{site.get('giorni_lavoro_settimana', 5)}gg/sett."
        ),
        "accuracy_note":  "Accuratezza indicativa ±40%.",
        "f1f2f3_used":    False,
    }


def _pv_meta(sdi: dict, pb_case: dict) -> dict:
    pv_ex = sdi.get("pv_existing") or {}
    if not pv_ex.get("presente", False):
        return {"source": "zeros", "confidence": "high", "pvgis_params": None}

    profilo_source = pv_ex.get("profilo_source", "sintetico")
    if profilo_source == "pvgis":
        pv = pb_case["pv"]
        site = pb_case["site"]
        return {
            "source":      "pvgis",
            "confidence":  "medium",
            "pvgis_params": {
                "year":       pv.get("pvgis_year", 2020),
                "tilt":       pv.get("pvgis_tilt", 30),
                "azimuth":    pv.get("pvgis_azimuth", 0),
                "losses_pct": pv.get("pvgis_losses_pct", 14),
                "lat":        site.get("lat"),
                "lon":        site.get("lon"),
            },
        }
    return {"source": "synthetic", "confidence": "low", "pvgis_params": None}


def _price_meta(sdi: dict, base_dir: str) -> dict:
    series = _v_nested(sdi, "tariff_context", "market_price_series", default="")
    has_file = bool(series) and os.path.exists(os.path.join(base_dir, series))
    if has_file:
        label = os.path.splitext(os.path.basename(series))[0]  # e.g. "it_nord_2024"
        return {"source": label, "series_file": series, "confidence": "high"}
    return {"source": "synthetic_it_nord", "series_file": None, "confidence": "medium"}


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


# ── Quick sanity test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, shutil

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TMP  = tempfile.mkdtemp()

    # Create profiles_cache inside tmp
    os.makedirs(os.path.join(TMP, "profiles_cache"), exist_ok=True)
    # Symlink prices/ and pvgis_cache/ so the engine can find them
    for subdir in ("prices", "pvgis_cache", "engine"):
        src = os.path.join(BASE, subdir)
        dst = os.path.join(TMP, subdir)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    # ── Test A ────────────────────────────────────────────────────────────────
    sdi_a = {
        "meta":    {"case_id": "test_a", "macro_case": "A"},
        "site":    {"consumo_annuo_kwh": {"value": 200_000}},
        "operational_profile": {
            "ore_lavoro_giorno":       {"value": 10},
            "giorni_lavoro_settimana": {"value": 5},
        },
        "billing": {"consumo_annuo_derivato_kwh": {"value": 200_000}},
        "tariff_context": {
            "market_price_series":     {"value": "prices/it_nord_2024.json"},
            "supplier_spread_eur_kwh": {"value": 0.05},
        },
        "pv_existing": None,
    }

    meta_a, warns_a = generate_profiles(sdi_a, base_dir=TMP)
    cached_a = load_profiles({"profiles": meta_a}, base_dir=TMP)

    load_sum_a = cached_a["load_kw"].sum() * _SLOT_H
    pv_sum_a   = cached_a["pv_kw"].sum() * _SLOT_H

    print("=== Test A ===")
    print(f"  load sum : {load_sum_a:,.0f} kWh  (atteso ~200,000)")
    print(f"  pv sum   : {pv_sum_a:,.0f} kWh  (atteso 0)")
    print(f"  source   : {meta_a['load_kw']['source']}")
    print(f"  warnings : {warns_a}")
    assert abs(load_sum_a - 200_000) < 1, f"load sum mismatch: {load_sum_a}"
    assert pv_sum_a == 0.0, f"PV non zero in Case A: {pv_sum_a}"
    print("  OK\n")

    # ── Test B ────────────────────────────────────────────────────────────────
    sdi_b = {
        "meta":    {"case_id": "test_b", "macro_case": "B"},
        "site": {
            "consumo_annuo_kwh": {"value": 150_000},
            "lat":               {"value": None},
            "lon":               {"value": None},
        },
        "operational_profile": {
            "ore_lavoro_giorno":       {"value": 10},
            "giorni_lavoro_settimana": {"value": 5},
        },
        "billing": {"consumo_annuo_derivato_kwh": {"value": 150_000}},
        "tariff_context": {
            "market_price_series":     {"value": "prices/it_nord_2024.json"},
            "supplier_spread_eur_kwh": {"value": 0.05},
        },
        "pv_existing": {
            "presente":       True,
            "kwp":            {"value": 80},
            "profilo_source": "sintetico",
        },
    }

    meta_b, warns_b = generate_profiles(sdi_b, base_dir=TMP)
    cached_b = load_profiles({"profiles": meta_b}, base_dir=TMP)

    load_sum_b = cached_b["load_kw"].sum() * _SLOT_H
    pv_sum_b   = cached_b["pv_kw"].sum() * _SLOT_H
    grid_sum_b = float(
        np.maximum(cached_b["load_kw"] - cached_b["pv_kw"], 0.0).sum() * _SLOT_H
    )

    print("=== Test B ===")
    print(f"  load sum  : {load_sum_b:,.0f} kWh  (atteso > 150,000)")
    print(f"  pv sum    : {pv_sum_b:,.0f} kWh")
    print(f"  grid draw : {grid_sum_b:,.1f} kWh  (atteso ~150,000)")
    print(f"  source    : {meta_b['load_kw']['source']}")
    print(f"  warnings  : {warns_b}")
    assert load_sum_b > 150_000, f"load sum deve superare 150,000 kWh: {load_sum_b}"
    assert abs(grid_sum_b - 150_000) < 2, f"grid draw non converge: {grid_sum_b}"
    print("  OK\n")

    shutil.rmtree(TMP)
    print("Tutti i test passati.")
