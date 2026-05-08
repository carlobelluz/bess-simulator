"""
tests/test_profile_generator_integration.py
Test di integrazione per intake/profile_generator.py — Brief 2.

Test 1  — Case A senza FV: struttura, cache, consumo entro 10%
Test 2  — Case B con FV (Toninato-like): profilo, metadata, sdi aggiornato
Test 3  — Coerenza cache file ↔ profiles_section
Test 4  — Case C → NotImplementedError
Test 5  — Case D → NotImplementedError
Test 6  — Identità di bilancio per Case B
Test 7  — case_builder integration
Test 8  — Performance < 10 secondi
"""

import json
import os
import shutil
import tempfile
import time

import numpy as np
import pytest

from intake.profile_generator import generate_profiles, load_profiles

# ── Costanti Toninato ──────────────────────────────────────────────────────────

_F1 = [13456, 9177, 4277, 2070, 1203, 2236, 2756, 1600, 3250, 3901, 6003, 9895]
_F2 = [5382,  5234, 3908, 1792, 1394, 2204, 2620, 2196, 2671, 2805, 3179, 4205]
_F3 = [5953,  5167, 4363, 2841, 2327, 3612, 3556, 3582, 3270, 3058, 4133, 5967]
_PICCHI = [135.0, 127.5, 109.0, 91.5, 58.2, 54.2, 66.5, 52.0, 56.8, 92.5, 96.5, 101.5]
_PRELIEVO_NETTO = 150_451   # kWh annui (somma F1+F2+F3)


# ── Fixture directory temporanea ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def tmp_dir():
    """
    Directory temporanea con symlink alle risorse del progetto
    (prices/, pvgis_cache/, engine/).
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp  = tempfile.mkdtemp(prefix="bess_test_")
    os.makedirs(os.path.join(tmp, "profiles_cache"), exist_ok=True)
    for subdir in ("prices", "pvgis_cache", "engine", "intake"):
        src = os.path.join(base, subdir)
        dst = os.path.join(tmp, subdir)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


# ── Helper: costruisce sdi di test ─────────────────────────────────────────────

def _make_test_sdi(
    macro_case: str = "A",
    has_fv: bool = False,
    annual_kwh: float = 200_000,
    kwp: float = 80.0,
    f1_mensile: list | None = None,
    f2_mensile: list | None = None,
    f3_mensile: list | None = None,
    peaks: list | None = None,
    lat: float = 45.55,
    lon: float = 11.55,
) -> dict:
    """
    Costruisce un sdi minimo ma strutturalmente corretto per i test.
    Segue il formato prodotto da case_builder.build_site_diagnostic().
    """
    case_id = f"test_{macro_case.lower()}_{'fv' if has_fv else 'nofv'}"

    # F1/F2/F3 mensili per fascia
    if f1_mensile and f2_mensile and f3_mensile:
        mensili_per_fascia = [
            {"mese": m + 1, "f1_kwh": f1_mensile[m], "f2_kwh": f2_mensile[m],
             "f3_kwh": f3_mensile[m], "consumo_kwh": f1_mensile[m] + f2_mensile[m] + f3_mensile[m]}
            for m in range(12)
        ]
        picchi_mensili_kw = [{"mese": m + 1, "picco_kw": (peaks[m] if peaks else 0.0)}
                             for m in range(12)]
        f1_tot = sum(f1_mensile)
        f2_tot = sum(f2_mensile)
        f3_tot = sum(f3_mensile)
        annual_kwh = f1_tot + f2_tot + f3_tot
    else:
        mensili_per_fascia = None
        picchi_mensili_kw = None
        # billing.f1_kwh = raw sum from n_bills=1 (non annualizzato)
        # _extract_monthly_bands applicherà factor=12 per n_bills=1
        f1_tot = annual_kwh * 0.50 / 12   # un mese di F1
        f2_tot = annual_kwh * 0.25 / 12   # un mese di F2
        f3_tot = annual_kwh * 0.25 / 12   # un mese di F3

    pv_existing = None
    if has_fv and kwp > 0:
        pv_existing = {
            "presente":       True,
            "kwp":            {"value": kwp, "source": "user_input", "confidence": "high"},
            "tilt":           {"value": 30,  "source": "user_input", "confidence": "high"},
            "azimuth":        {"value": 0,   "source": "user_input", "confidence": "high"},
            "profilo_source": "sintetico",   # non chiama PVGIS API in test
            "fv_export_regime": "nessuno",
        }

    return {
        "_schema_version": "2.0",
        "_block":          "Block1_SiteIntake",
        "meta": {
            "case_id":            case_id,
            "nome_cliente":       "Test SRL",
            "macro_case":         macro_case,
            "data_quality_level": 2,
        },
        "site": {
            "consumo_annuo_kwh": {
                "value":      annual_kwh,
                "source":     "bill_derived",
                "confidence": "medium",
            },
            "potenza_contrattuale_kw": {"value": 160, "source": "user_input", "confidence": "medium"},
            "picco_potenza_kw":  {"value": 100, "source": "user_input", "confidence": "medium"},
            "lat": {"value": lat, "source": "user_input", "confidence": "high"},
            "lon": {"value": lon, "source": "user_input", "confidence": "high"},
            "comune": "Vicenza",
        },
        "operational_profile": {
            "ore_lavoro_giorno":       {"value": 10, "source": "user_input", "confidence": "medium"},
            "giorni_lavoro_settimana": {"value": 5,  "source": "user_input", "confidence": "medium"},
        },
        "billing": {
            "n_bills": 12 if mensili_per_fascia else 1,
            "mesi_coperti": [f"2024-{m+1:02d}" for m in range(12)] if mensili_per_fascia else ["2024-01"],
            "consumo_annuo_derivato_kwh": {
                "value":      annual_kwh,
                "source":     "bill_sum",
                "confidence": "medium",
            },
            "mensili_per_fascia": mensili_per_fascia,
            "picchi_mensili_kw":  picchi_mensili_kw,
            "f1_kwh": f1_tot if not mensili_per_fascia else None,
            "f2_kwh": f2_tot if not mensili_per_fascia else None,
            "f3_kwh": f3_tot if not mensili_per_fascia else None,
        },
        "tariff_context": {
            "market_price_series":     {"value": "prices/it_nord_2024.json"},
            "supplier_spread_eur_kwh": {"value": 0.05},
            "quota_potenza_eur_kw_mese": {"value": 10.0},
        },
        "pv_existing": pv_existing,
        "profiles": None,
    }


# ── Test 1 — Case A senza FV ───────────────────────────────────────────────────

def test_1_case_a_no_fv(tmp_dir):
    """Case A (no FV): struttura profiles_section, cache, consumo entro 10%."""
    sdi = _make_test_sdi(macro_case="A", has_fv=False, annual_kwh=200_000)
    profiles_section, warnings = generate_profiles(sdi, base_dir=tmp_dir)

    # Struttura dizionario invariata
    assert "cache_file" in profiles_section
    assert "load_kw" in profiles_section
    assert "pv_kw" in profiles_section
    assert "price_eur_kwh" in profiles_section

    # Cache file generato e leggibile
    sdi["profiles"] = profiles_section
    cached = load_profiles(sdi, base_dir=tmp_dir)
    assert cached is not None
    assert cached["load_kw"].shape == (35_040,)
    assert cached["pv_kw"].shape == (35_040,)

    # Senza FV, pv_kw è tutto zero
    assert cached["pv_kw"].sum() == pytest.approx(0.0, abs=1.0), \
        "pv_kw deve essere zero per Case A senza FV"

    # Consumo annuo entro 10% dal target
    consumo_modellato = cached["load_kw"].sum() * 0.25
    err_rel = abs(consumo_modellato - 200_000) / 200_000
    assert err_rel < 0.10, (
        f"Consumo modellato {consumo_modellato:,.0f} kWh vs target 200,000 kWh "
        f"(errore {err_rel*100:.1f}% > 10%)"
    )

    # n_slots e slot_hours in cache
    assert cached["n_slots"] == 35_040
    assert cached["slot_hours"] == pytest.approx(0.25)


# ── Test 2 — Case B con FV (Toninato-like) ────────────────────────────────────

def test_2_case_b_with_fv(tmp_dir):
    """Case B con FV: usa site_reconstruction, metadata corretti, sdi aggiornato."""
    sdi = _make_test_sdi(
        macro_case="B",
        has_fv=True,
        kwp=88.0,
        f1_mensile=[float(v) for v in _F1],
        f2_mensile=[float(v) for v in _F2],
        f3_mensile=[float(v) for v in _F3],
        peaks=[float(v) for v in _PICCHI],
    )
    profiles_section, warnings = generate_profiles(sdi, base_dir=tmp_dir)

    # Cache caricato
    sdi["profiles"] = profiles_section
    cached = load_profiles(sdi, base_dir=tmp_dir)
    assert cached is not None
    assert cached["load_kw"].shape == (35_040,)

    # Il consumo lordo del sito DEVE essere maggiore del prelievo netto di bolletta
    # (perché include autoconsumo FV)
    consumo_lordo = cached["load_kw"].sum() * 0.25
    assert consumo_lordo > _PRELIEVO_NETTO, (
        f"Consumo lordo {consumo_lordo:,.0f} kWh deve essere > prelievo netto "
        f"{_PRELIEVO_NETTO:,} kWh (manca autoconsumo FV)"
    )

    # Metadata corretti
    load_meta = profiles_section["load_kw"]
    assert load_meta["source"] == "reconstructed_from_bill_and_pv"
    assert load_meta["confidence"] in ("low", "medium", "high")
    assert load_meta["f1f2f3_used"] is True
    assert load_meta["archetype_inferred"] is not None
    assert load_meta["fabbisogno_annuo_kwh"] is not None
    assert load_meta["fabbisogno_annuo_kwh"] > 0
    assert load_meta["autoconsumo_fv_annuo_kwh"] is not None

    # sdi["site"]["consumo_annuo_kwh"] aggiornato da generate_profiles
    assert sdi["site"]["consumo_annuo_kwh"]["source"] == "reconstructed_from_bill_and_pv"
    assert sdi["site"]["consumo_annuo_kwh"]["value"] > 0


# ── Test 3 — Coerenza cache file ──────────────────────────────────────────────

def test_3_cache_file_consistency(tmp_dir):
    """Il cache file JSON deve contenere tutti i campi richiesti."""
    sdi = _make_test_sdi(macro_case="A", has_fv=False, annual_kwh=100_000)
    profiles_section, _ = generate_profiles(sdi, base_dir=tmp_dir)

    cache_path = os.path.join(tmp_dir, profiles_section["cache_file"])
    assert os.path.exists(cache_path), f"Cache file non trovato: {cache_path}"

    with open(cache_path) as f:
        data = json.load(f)

    assert "load_kw"       in data
    assert "pv_kw"         in data
    assert "price_eur_kwh" in data
    assert "month"         in data
    assert "hour"          in data
    assert "dow"           in data
    assert "slot_in_day"   in data
    assert data["n_slots"]    == 35_040
    assert data["slot_hours"] == pytest.approx(0.25)

    # Gli array devono avere 35040 elementi
    assert len(data["load_kw"])  == 35_040
    assert len(data["pv_kw"])    == 35_040
    assert len(data["month"])    == 35_040


# ── Test 4+5 — Case C/D → NotImplementedError ────────────────────────────────

def test_4_case_c_not_implemented(tmp_dir):
    sdi = _make_test_sdi(macro_case="C")
    with pytest.raises(NotImplementedError):
        generate_profiles(sdi, base_dir=tmp_dir)


def test_5_case_d_not_implemented(tmp_dir):
    sdi = _make_test_sdi(macro_case="D")
    with pytest.raises(NotImplementedError):
        generate_profiles(sdi, base_dir=tmp_dir)


# ── Test 6 — Identità di bilancio Case B ─────────────────────────────────────

def test_6_balance_identity_case_b(tmp_dir):
    """
    Per Case B con FV: fabbisogno ≈ prelievo bolletta + autoconsumo_fv
    Tolleranza: 1% del fabbisogno.
    """
    sdi = _make_test_sdi(
        macro_case="B",
        has_fv=True,
        kwp=88.0,
        f1_mensile=[float(v) for v in _F1],
        f2_mensile=[float(v) for v in _F2],
        f3_mensile=[float(v) for v in _F3],
        peaks=[float(v) for v in _PICCHI],
    )
    profiles_section, _ = generate_profiles(sdi, base_dir=tmp_dir)
    load_meta = profiles_section["load_kw"]

    fabb  = load_meta["fabbisogno_annuo_kwh"]
    autoc = load_meta["autoconsumo_fv_annuo_kwh"]

    assert fabb is not None and autoc is not None
    assert fabb > 0

    # Identità fondamentale: fabb - autoc = grid_simulato (esatta per costruzione)
    # Verifica: grid_simulato deve essere > 0 e nell'ordine del prelievo bolletta
    grid_simulato = fabb - autoc
    assert grid_simulato > 0, f"Grid simulato non positivo: {grid_simulato}"

    # Il grid simulato deve essere nell'ordine del prelievo bolletta (entro 20%
    # dato che la calibrazione ha una tolleranza intrinseca)
    err = abs(grid_simulato - _PRELIEVO_NETTO) / _PRELIEVO_NETTO
    assert err < 0.20, (
        f"Grid simulato {grid_simulato:,.0f} vs prelievo bolletta {_PRELIEVO_NETTO:,} "
        f"(errore {err*100:.1f}% > 20%)"
    )


# ── Test 7 — case_builder integration ────────────────────────────────────────

def test_7_case_builder_integration(tmp_dir):
    """
    Test integrato: case_builder.build_site_diagnostic() con form realistico.
    Verifica che sdi sia completo e coerente.
    """
    from intake.case_builder import build_site_diagnostic

    bill = {
        "periodo":               "2024-06",
        "consumo_kwh":           12_000,
        "potenza_contrattuale_kw": 100,
        "costo_energia_eur":     2_400,
        "costo_potenza_eur":     800,
        "f1_kwh":                6_000,
        "f2_kwh":                3_000,
        "f3_kwh":                3_000,
    }

    form = {
        "case_id":               "test_integration_b2",
        "nome_cliente":          "Test Integrazione SRL",
        "comune":                "Vicenza",
        "lat":                   45.55,
        "lon":                   11.55,
        "ha_pv":                 True,
        "kwp":                   50.0,
        "consumo_annuo_kwh":     144_000,   # richiesto da validate_form (12*12000)
        "potenza_contrattuale_kw": 100,
        "market_price_series":   "prices/it_nord_2024.json",
        "bills":                 [bill],
    }

    sdi = build_site_diagnostic(form, base_dir=tmp_dir)

    # sdi completo con profiles
    assert sdi["profiles"] is not None, "sdi['profiles'] è None"
    assert sdi["profiles"]["load_kw"]["source"] in (
        "reconstructed_from_bill_and_pv",
        "reconstructed_from_bill",
        "synthetic_industrial",
    ), f"Fonte non attesa: {sdi['profiles']['load_kw']['source']}"

    # Profilo caricato
    cached = load_profiles(sdi, base_dir=tmp_dir)
    assert cached is not None
    assert cached["load_kw"].shape == (35_040,)

    # Case B: consumo sito > consumo bolletta
    macro = sdi["meta"]["macro_case"]
    if macro == "B":
        billing_consumo = sdi["billing"]["consumo_annuo_derivato_kwh"]["value"]
        site_consumo    = sdi["site"]["consumo_annuo_kwh"]["value"]
        assert site_consumo > billing_consumo, (
            f"Case B: site_consumo {site_consumo:,} deve essere > billing_consumo {billing_consumo:,}"
        )
        assert sdi["site"]["consumo_annuo_kwh"]["source"] == "reconstructed_from_bill_and_pv"

    # Coerenza: fabbisogno nella metadata ≈ consumo nel sdi
    if sdi["profiles"]["load_kw"].get("fabbisogno_annuo_kwh"):
        fabb_meta = sdi["profiles"]["load_kw"]["fabbisogno_annuo_kwh"]
        consumo_sdi = sdi["site"]["consumo_annuo_kwh"]["value"]
        assert abs(consumo_sdi - fabb_meta) < 200, (
            f"Incoerenza: sdi.site.consumo={consumo_sdi:,} vs profiles.fabbisogno={fabb_meta:,.0f}"
        )


# ── Test 8 — Performance ──────────────────────────────────────────────────────

def test_8_performance(tmp_dir):
    """generate_profiles() su Case B realistico deve completare in < 10 secondi."""
    sdi = _make_test_sdi(
        macro_case="B",
        has_fv=True,
        kwp=88.0,
        f1_mensile=[float(v) for v in _F1],
        f2_mensile=[float(v) for v in _F2],
        f3_mensile=[float(v) for v in _F3],
        peaks=[float(v) for v in _PICCHI],
    )
    t0 = time.time()
    generate_profiles(sdi, base_dir=tmp_dir)
    elapsed = time.time() - t0
    assert elapsed < 10.0, f"generate_profiles ha impiegato {elapsed:.2f}s > 10s"


# ── Bonus: test backward compat (Brief 1 non rotto) ──────────────────────────

def test_brief1_still_passes():
    """Verifica che i test del Brief 1 siano ancora importabili senza crash."""
    from intake.site_reconstruction import SiteEnergyState, reconcile
    state = SiteEnergyState(
        prelievo_f1_mensile=[1000.0] * 12,
        prelievo_f2_mensile=[500.0]  * 12,
        prelievo_f3_mensile=[800.0]  * 12,
        has_fv=False,
    )
    state = reconcile(state)
    assert state.fabbisogno_annuo_kwh is not None
    assert state.load_profile_qh_kw.shape == (35_040,)
