"""
tests/test_tariff_bands.py
Regression tests for intake/tariff_bands.py MVP.

Run with:  python -m pytest tests/test_tariff_bands.py -v
       or: python tests/test_tariff_bands.py
"""

from datetime import datetime
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intake.tariff_bands import (
    italian_holidays,
    classify_slot,
    build_band_masks,
    validate_band_reconstruction,
)


# ── Classification sanity checks ───────────────────────────────────────────────

def test_classify_monday_peak():
    # Mon 2024-01-08 08:00 → F1 (Mon-Fri 8-19 non-holiday)
    assert classify_slot(datetime(2024, 1, 8, 8, 0)) == "F1"

def test_classify_monday_offpeak_morning():
    # Mon 2024-01-08 07:00 → F2 (Mon-Fri 7-8)
    assert classify_slot(datetime(2024, 1, 8, 7, 0)) == "F2"

def test_classify_monday_offpeak_evening():
    # Mon 2024-01-08 19:00 → F2 (Mon-Fri 19-23)
    assert classify_slot(datetime(2024, 1, 8, 19, 0)) == "F2"

def test_classify_monday_night():
    # Mon 2024-01-08 00:00 → F3 (Mon-Fri 0-7)
    assert classify_slot(datetime(2024, 1, 8, 0, 0)) == "F3"

def test_classify_monday_late_night():
    # Mon 2024-01-08 23:00 → F3 (Mon-Fri 23-24)
    assert classify_slot(datetime(2024, 1, 8, 23, 0)) == "F3"

def test_classify_saturday_day():
    # Sat 2024-01-06 12:00 — BUT Jan 6 is Epifania (holiday) → F3
    # Use Sat 2024-01-13 instead (non-holiday Saturday)
    assert classify_slot(datetime(2024, 1, 13, 12, 0)) == "F2"

def test_classify_saturday_night():
    # Sat 2024-01-13 06:00 → F3 (Sat 0-7)
    assert classify_slot(datetime(2024, 1, 13, 6, 0)) == "F3"

def test_classify_sunday():
    # Sun 2024-01-14 10:00 → F3 (Sunday all day)
    assert classify_slot(datetime(2024, 1, 14, 10, 0)) == "F3"

def test_classify_epifania():
    # Mon 2024-01-06 (Epifania — national holiday) 10:00 → F3 (holiday all day)
    assert classify_slot(datetime(2024, 1, 6, 10, 0)) == "F3"

def test_classify_easter_monday_2024():
    # Easter Monday 2024 = 2024-04-01, 09:00 → F3 (holiday all day)
    assert classify_slot(datetime(2024, 4, 1, 9, 0)) == "F3"

def test_classify_day_after_easter_monday_is_f1():
    # Tue 2024-04-02 10:00 → F1 (normal workday, non-holiday)
    assert classify_slot(datetime(2024, 4, 2, 10, 0)) == "F1"

def test_classify_republic_day():
    # Sun 2024-06-02 (Festa della Repubblica — also a Sunday) → F3
    assert classify_slot(datetime(2024, 6, 2, 10, 0)) == "F3"

def test_classify_repubblica_on_weekday():
    # 2025-06-02 is Monday — national holiday → F3 all day
    assert classify_slot(datetime(2025, 6, 2, 10, 0)) == "F3"


# ── Band mask sanity checks ─────────────────────────────────────────────────────

def test_masks_exhaustive_2023():
    masks = build_band_masks(2023)
    f1, f2, f3 = masks["f1"], masks["f2"], masks["f3"]
    # Every slot is in exactly one band
    combined = f1.astype(int) + f2.astype(int) + f3.astype(int)
    assert np.all(combined == 1), "Masks are not mutually exclusive and exhaustive"

def test_masks_length_2023():
    masks = build_band_masks(2023)
    assert len(masks["f1"]) == 35_040  # 365 × 96

def test_masks_length_2024():
    masks = build_band_masks(2024)
    assert len(masks["f1"]) == 35_136  # 366 × 96

def test_masks_exhaustive_2024():
    masks = build_band_masks(2024)
    combined = masks["f1"].astype(int) + masks["f2"].astype(int) + masks["f3"].astype(int)
    assert np.all(combined == 1)

def test_masks_f1_fraction_plausible():
    # F1 covers Mon-Fri 8-19 = 11h/day. Roughly 11/24 × 5/7 ≈ 32.7% max,
    # minus holidays. Expect 28–34%.
    masks = build_band_masks(2023)
    f1_frac = masks["f1"].mean()
    assert 0.28 <= f1_frac <= 0.34, f"F1 fraction unexpected: {f1_frac:.3f}"

def test_masks_f3_majority():
    # F3 (off-peak) covers nights + weekends + holidays — always the largest band
    masks = build_band_masks(2023)
    assert masks["f3"].mean() > 0.40, "F3 should be the largest band"


# ── validate_band_reconstruction ───────────────────────────────────────────────

def _make_test_arrays(n_slots=35_040, pv_fraction=0.0):
    """
    Builds flat synthetic load and PV arrays for validation tests.
    load_kw = constant 1.0 kW across all slots.
    pv_kw   = constant pv_fraction kW across all slots.
    """
    load = np.ones(n_slots, dtype=float)
    pv   = np.full(n_slots, pv_fraction, dtype=float)
    return load, pv


def _get_band_totals(load, pv, masks, slot_h=0.25):
    """Returns per-band grid draw totals (kWh) from load and PV arrays."""
    grid = np.maximum(load - pv, 0.0)
    return {b: float(grid[masks[b]].sum() * slot_h) for b in ("f1", "f2", "f3")}


def test_coherent_bands_give_medium():
    """When billing bands match reconstructed draws exactly, confidence = medium."""
    masks = build_band_masks(2023)
    load, pv = _make_test_arrays(n_slots=35_040, pv_fraction=0.0)

    # Use the exact reconstructed values as "billing" → perfect match
    draws = _get_band_totals(load, pv, masks)
    confidence, warning, band_draws = validate_band_reconstruction(
        load, pv, masks,
        billing_f1=draws["f1"],
        billing_f2=draws["f2"],
        billing_f3=draws["f3"],
    )
    assert confidence == "medium", f"Expected medium, got {confidence}"
    assert warning is None, f"Expected no warning, got: {warning}"


def test_coherent_bands_within_tolerance():
    """Bands within 10% tolerance (< 15%) still give medium confidence."""
    masks = build_band_masks(2023)
    load, pv = _make_test_arrays(n_slots=35_040, pv_fraction=0.0)
    draws = _get_band_totals(load, pv, masks)

    # Billing is 10% higher than reconstructed — within 15% tolerance
    confidence, warning, _ = validate_band_reconstruction(
        load, pv, masks,
        billing_f1=draws["f1"] * 1.10,
        billing_f2=draws["f2"] * 1.10,
        billing_f3=draws["f3"] * 1.10,
    )
    assert confidence == "medium"
    assert warning is None


def test_incoherent_bands_keep_low():
    """When one band diverges > 15%, confidence stays low and a warning is returned."""
    masks = build_band_masks(2023)
    load, pv = _make_test_arrays(n_slots=35_040, pv_fraction=0.0)
    draws = _get_band_totals(load, pv, masks)

    # F1 billing is 40% higher than reconstructed — exceeds 15% tolerance
    confidence, warning, band_draws = validate_band_reconstruction(
        load, pv, masks,
        billing_f1=draws["f1"] * 1.40,
        billing_f2=draws["f2"],
        billing_f3=draws["f3"],
    )
    assert confidence == "low", f"Expected low, got {confidence}"
    assert warning is not None, "Expected a warning for incoherent F1"
    assert "F1" in warning


def test_incoherent_f2_band_triggers_warning():
    """F2 divergence > 15% also triggers low confidence and named warning."""
    masks = build_band_masks(2023)
    load, pv = _make_test_arrays(n_slots=35_040, pv_fraction=0.0)
    draws = _get_band_totals(load, pv, masks)

    confidence, warning, _ = validate_band_reconstruction(
        load, pv, masks,
        billing_f1=draws["f1"],
        billing_f2=draws["f2"] * 2.0,   # 100% divergence
        billing_f3=draws["f3"],
    )
    assert confidence == "low"
    assert "F2" in warning


def test_zero_billing_band_skipped():
    """Bands with billing value = 0 are skipped; other coherent bands still give medium."""
    masks = build_band_masks(2023)
    load, pv = _make_test_arrays(n_slots=35_040, pv_fraction=0.0)
    draws = _get_band_totals(load, pv, masks)

    # F3 billing = 0 → skipped; F1 and F2 match exactly → medium
    confidence, warning, _ = validate_band_reconstruction(
        load, pv, masks,
        billing_f1=draws["f1"],
        billing_f2=draws["f2"],
        billing_f3=0.0,
    )
    assert confidence == "medium"
    assert warning is None


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed.")
    if failed:
        sys.exit(1)
