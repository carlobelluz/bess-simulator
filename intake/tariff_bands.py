"""
intake/tariff_bands.py
Italian F1/F2/F3 tariff-band classification engine.

Scope: diagnostic reconstruction aid for Block 1 load reconstruction.
       Not for economic simulation — Block 2 uses slot-level ENTSO-E prices.

Public API:
  italian_holidays(year)                     → frozenset[date]
  classify_slot(dt, holidays=None)           → "F1" | "F2" | "F3"
  build_band_masks(year, slot_minutes=15)    → {"f1": ndarray, "f2": ndarray, "f3": ndarray}
  validate_band_reconstruction(...)          → (confidence, warning|None, band_draws)

ARERA rules (Delibera ARG/elt 199/11 and subsequent updates):
  F1 — peak:          Mon-Fri non-holiday, 08:00–19:00
  F2 — intermediate:  Mon-Fri non-holiday, 07:00–08:00 and 19:00–23:00
                      Saturday non-holiday, 07:00–23:00
  F3 — off-peak:      Mon-Fri, 00:00–07:00 and 23:00–24:00
                      Saturday, 00:00–07:00 and 23:00–24:00
                      Sunday, all day
                      National holidays, all day

Slot classification uses the slot START time.
Datetimes must be in Italian local time (naive, no DST adjustment).
build_band_masks results are cached per (year, slot_minutes).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Literal

import numpy as np


# ── Public API ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def italian_holidays(year: int) -> frozenset:
    """
    Italian national holidays for year (frozenset of date objects).
    Includes Easter Monday (Anonymous Gregorian algorithm).
    Does not include local patron-saint days (vary by municipality).
    """
    fixed = [
        date(year,  1,  1),  # Capodanno
        date(year,  1,  6),  # Epifania
        date(year,  4, 25),  # Liberazione
        date(year,  5,  1),  # Festa del Lavoro
        date(year,  6,  2),  # Repubblica
        date(year,  8, 15),  # Ferragosto
        date(year, 11,  1),  # Ognissanti
        date(year, 12,  8),  # Immacolata
        date(year, 12, 25),  # Natale
        date(year, 12, 26),  # Santo Stefano
    ]
    return frozenset(fixed + [_easter_monday(year)])


def classify_slot(
    dt: datetime,
    holidays: frozenset | None = None,
) -> Literal["F1", "F2", "F3"]:
    """
    Classifies a single datetime into F1, F2, or F3 per ARERA rules.

    dt       : naive datetime in Italian local time; slot is classified
               by its START time.
    holidays : pass italian_holidays(year) when calling in bulk to avoid
               recomputing for every slot.
    """
    if holidays is None:
        holidays = italian_holidays(dt.year)

    d  = dt.date()
    h  = dt.hour
    wd = dt.weekday()   # 0 = Monday … 6 = Sunday

    # Sunday or national holiday → F3 all day
    if wd == 6 or d in holidays:
        return "F3"

    # Saturday
    if wd == 5:
        return "F2" if 7 <= h < 23 else "F3"

    # Monday–Friday (non-holiday)
    if 8 <= h < 19:
        return "F1"
    if 7 <= h < 8 or 19 <= h < 23:
        return "F2"
    return "F3"   # 00:00–07:00 and 23:00–24:00


@lru_cache(maxsize=8)
def build_band_masks(year: int, slot_minutes: int = 15) -> dict[str, np.ndarray]:
    """
    Builds boolean F1/F2/F3 masks for every time slot in year.

    Returns dict with keys "f1", "f2", "f3" — boolean numpy arrays.
    Masks are mutually exclusive and exhaustive (exactly one True per slot).
    Length: n_days × 24 × (60 // slot_minutes).
      365 × 96 = 35 040 for non-leap years (15-min slots).
      366 × 96 = 35 136 for leap years.

    Result is cached — callers must not mutate the returned arrays.
    """
    slots_per_day = 24 * (60 // slot_minutes)
    n_days  = 366 if _is_leap(year) else 365
    n_slots = n_days * slots_per_day

    holidays = italian_holidays(year)
    start    = datetime(year, 1, 1)

    # Build integer band array: 1=F1, 2=F2, 3=F3
    bands = np.empty(n_slots, dtype=np.uint8)
    for i in range(n_slots):
        dt   = start + timedelta(minutes=i * slot_minutes)
        band = classify_slot(dt, holidays)
        bands[i] = 1 if band == "F1" else (2 if band == "F2" else 3)

    return {"f1": bands == 1, "f2": bands == 2, "f3": bands == 3}


def validate_band_reconstruction(
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    masks: dict[str, np.ndarray],
    billing_f1: float,
    billing_f2: float,
    billing_f3: float,
    slot_h: float = 0.25,
    tolerance: float = 0.15,
) -> tuple[str, str | None, dict[str, float]]:
    """
    Post-reconstruction validation: compares per-band reconstructed grid draw
    against billed F1/F2/F3 net-draw values.

    Algorithm:
      1. Compute grid_draw[t] = max(0, load_kw[t] - pv_kw[t]) for each slot.
      2. Sum grid_draw by band using masks.
      3. Compare each band's sum against the billed value.
      4. Upgrade confidence to "medium" if all bands are within tolerance.
         Keep "low" and emit a warning if any band diverges > tolerance.

    Returns:
      confidence  — "medium" if coherent, "low" if not
      warning     — None if coherent; descriptive message otherwise
      band_draws  — {"f1": kWh, "f2": kWh, "f3": kWh} reconstructed draws
    """
    grid  = np.maximum(load_kw - pv_kw, 0.0)
    draws = {b: float(grid[masks[b]].sum() * slot_h) for b in ("f1", "f2", "f3")}
    billed = {"f1": billing_f1, "f2": billing_f2, "f3": billing_f3}

    mismatches = []
    for band, bval in billed.items():
        if bval <= 0:
            continue
        err = abs(draws[band] - bval) / bval
        if err > tolerance:
            mismatches.append(
                f"{band.upper()}: stimato {draws[band]:,.0f} kWh "
                f"vs bolletta {bval:,.0f} kWh ({err * 100:.0f}% scarto)"
            )

    if mismatches:
        warning = (
            f"Fasce tariffarie: divergenza > {tolerance * 100:.0f}% su "
            f"{'; '.join(mismatches)}. "
            "Il profilo sintetico non riflette la distribuzione oraria reale del carico."
        )
        return "low", warning, draws

    return "medium", None, draws


# ── Internal helpers ───────────────────────────────────────────────────────────

def _easter_monday(year: int) -> date:
    """Easter Sunday + 1 day, Anonymous Gregorian algorithm."""
    a    = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f    = (b + 8) // 25
    g    = (b - f + 1) // 3
    h    = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l    = (32 + 2 * e + 2 * i - h - k) % 7
    m    = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day   = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day) + timedelta(days=1)


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
