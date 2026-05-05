"""
profile_builder.py
Generates the three annual arrays needed by the simulation engine:
  load_kw, pv_kw, price_eur_kwh  —  all at 15-min resolution (35,040 slots).

v1: fully synthetic. Each function has a SWAP marker showing where a real
data source plugs in later:
  SWAP_LOAD  → CSV reader with measured 15-min load curve
  SWAP_PV    → PVGIS API client
  SWAP_PRICE → ENTSO-E real data loader (already wired; just populate the JSON)
"""

import json
import math
import os
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

N_SLOTS     = 35040   # 365 × 24 × 4
N_HOURS     = 8760    # 365 × 24
SLOTS_PER_H = 4
SLOT_H      = 0.25    # hours per slot (15 min)
_WORK_START = 7       # default working day start — 07:00

# Slot index at start of each month boundary (non-leap 365-day year).
# Stored at slot granularity so both hourly and 15-min loops can use it.
_MONTH_DAY_STARTS = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

# ── Public API ────────────────────────────────────────────────────────────────

def build_all_profiles(case: dict, base_dir: str = ".") -> dict:
    """
    Master entry point for the engine. Takes the loaded case dict and
    returns every array the engine needs in a single dict.

    base_dir: project root — used to resolve the market price file path.

    Returned keys:
      load_kw       (35040,) kW    — site consumption, slot by slot
      pv_kw         (35040,) kW    — PV production, slot by slot
      price_eur_kwh (35040,) €/kWh — customer energy price, slot by slot
      month         (35040,) int   — 1-12
      hour          (35040,) int   — 0-23
      dow           (35040,) int   — 0-6, 0=Monday
      slot_in_day   (35040,) int   — 0-95
      slot_hours    float          — 0.25
      n_slots       int            — 35040
    """
    ti = _make_time_index()
    return {
        "load_kw":       build_load_profile(case["site"], ti),
        "pv_kw":         build_pv_profile(case["pv"], ti),
        "price_eur_kwh": build_price_profile(case["tariffs"], base_dir, ti),
        "slot_hours":    SLOT_H,
        "n_slots":       N_SLOTS,
        **ti,
    }


def build_load_profile(site: dict, ti: dict | None = None) -> np.ndarray:
    """
    Returns annual site consumption profile in kW (35040 slots).
    SWAP_LOAD: replace _synthetic_load() body with a CSV reader.
    """
    if ti is None:
        ti = _make_time_index()
    return _synthetic_load(site, ti)


def build_pv_profile(pv: dict, ti: dict | None = None) -> np.ndarray:
    """
    Returns annual PV production profile in kW (35040 slots).
    Returns zeros if pv.presente is false.
    SWAP_PV: replace _synthetic_pv() body with PVGIS client.
    """
    if not pv.get("presente", False):
        return np.zeros(N_SLOTS)
    if ti is None:
        ti = _make_time_index()
    return _synthetic_pv(pv)


def build_price_profile(tariffs: dict, base_dir: str = ".", ti: dict | None = None) -> np.ndarray:
    """
    Returns annual customer price in €/kWh (35040 slots).
    Loads the ENTSO-E file if available; falls back to synthetic.
    SWAP_PRICE: the real-data path is already wired — populate the JSON.
    """
    if ti is None:
        ti = _make_time_index()
    spread    = tariffs["supplier_spread_eur_kwh"]
    market_h  = _load_market_prices(tariffs.get("market_price_series", ""), base_dir)
    # Upsample hourly → 15-min. Price is constant within each hour.
    market_slot = np.repeat(market_h, SLOTS_PER_H) / 1000.0   # €/MWh → €/kWh
    return market_slot + spread


# ── Time index ────────────────────────────────────────────────────────────────

def _make_time_index() -> dict:
    """
    Builds the four time-coordinate arrays for all 35,040 slots.
    Reference year: Jan 1, 2024 = slot 0 = Monday (dow=0).
    Using a fixed 365-day year — no leap day — for simplicity in v1.
    """
    day         = np.arange(N_SLOTS) // 96
    slot_in_day = np.arange(N_SLOTS) % 96
    hour        = slot_in_day // SLOTS_PER_H    # 0-23
    dow         = day % 7                        # 0=Mon, Jan 1 2024 is Monday

    month = np.empty(N_SLOTS, dtype=np.int8)
    for m in range(12):
        lo = _MONTH_DAY_STARTS[m] * 96
        hi = _MONTH_DAY_STARTS[m + 1] * 96
        month[lo:hi] = m + 1

    return {"month": month, "hour": hour, "dow": dow, "slot_in_day": slot_in_day}


# ── Synthetic load ─────────────────────────────────────────────────────────────

# Index 0 unused; positions 1-12 = Jan-Dec.
# Italian industrial calendar: August closure (-45%), Christmas dip (-12%),
# summer slowdown in June-July, winter uplift from heating and lighting.
_LOAD_SEASONAL = np.array([
    0.00,                                                 # [0] unused
    1.08, 1.05, 1.00, 0.97, 0.95, 0.92,                 # Jan-Jun
    0.78, 0.55, 0.95, 1.00, 1.05, 0.88,                 # Jul-Dec
])

def _synthetic_load(site: dict, ti: dict) -> np.ndarray:
    """
    Generates a block-shaped industrial load profile.

    Shape logic:
      - Working day + working hour  → full load (1.0)
      - Anything else               → base load (0.12)
    Working window starts at _WORK_START (07:00) and spans ore_lavoro_giorno hours.
    Working days: Monday through (giorni_lavoro_settimana - 1).

    After shaping, the profile is scaled so that the annual sum equals
    consumo_annuo_kwh exactly. The simulated peak may therefore differ
    from picco_potenza_kw if the stated load factor does not match the
    synthetic shape — this is expected and acceptable in v1.

    Small reproducible slot-level noise (σ=3%, seed=42) avoids perfectly
    flat blocks and creates more realistic 15-min variance.
    """
    consumo_kwh = site["consumo_annuo_kwh"]
    ore_lavoro  = site.get("ore_lavoro_giorno", 10)
    giorni_lav  = site.get("giorni_lavoro_settimana", 5)
    work_end    = _WORK_START + ore_lavoro

    BASE_LOAD = 0.12   # security lighting, HVAC, standby equipment

    is_workday  = ti["dow"] < giorni_lav
    is_workhour = (ti["hour"] >= _WORK_START) & (ti["hour"] < work_end)

    shape    = np.where(is_workday & is_workhour, 1.0, BASE_LOAD)
    seasonal = _LOAD_SEASONAL[ti["month"]]
    shape    = shape * seasonal

    # Scale to target annual energy
    shape *= consumo_kwh / (shape.sum() * SLOT_H)

    # Mild noise for 15-min realism, then re-normalise to keep annual total exact
    rng   = np.random.default_rng(42)
    noise = np.clip(rng.normal(1.0, 0.03, N_SLOTS), 0.85, 1.15)
    load  = shape * noise
    load *= consumo_kwh / (load.sum() * SLOT_H)

    return np.maximum(load, 0.0)


# ── Synthetic PV ──────────────────────────────────────────────────────────────

def _synthetic_pv(pv: dict) -> np.ndarray:
    """
    Generates PV production using a simplified astronomical model.

    For each day of the year, computes the solar elevation angle at each hour
    using the Spencer declination formula. Production is proportional to
    sin(elevation) when the sun is above the horizon — a clear-sky
    approximation with no cloud or diffuse irradiance model.

    Latitude defaults to 45.0°N (northern Italy). SWAP_PV will use
    site["lat"] once PVGIS is integrated.

    The resulting shape is scaled so that annual production equals
    producibilita_annua_kwh (or kwp × 1200 kWh if that field is absent).
    """
    kwp        = pv["kwp"]
    target_kwh = pv.get("producibilita_annua_kwh") or kwp * 1200.0
    lat_rad    = math.radians(45.0)   # SWAP_PV: math.radians(site["lat"])

    pv_h = np.zeros(N_HOURS)

    for d in range(365):
        # Solar declination via Spencer approximation
        B    = math.radians(360 / 365 * (d - 81))
        decl = math.radians(23.45 * math.sin(B))

        for h in range(24):
            # Hour angle: 0° at solar noon, ±15° per hour from noon
            omega   = math.radians((h + 0.5 - 12.0) * 15.0)
            sin_alt = (math.sin(lat_rad) * math.sin(decl)
                       + math.cos(lat_rad) * math.cos(decl) * math.cos(omega))
            if sin_alt > 0.0:
                pv_h[d * 24 + h] = sin_alt

    # Scale so annual production matches target
    raw_kwh = pv_h.sum()   # each hourly value × 1 h = kWh
    if raw_kwh > 0:
        pv_h *= target_kwh / raw_kwh

    # Upsample to 15-min — production is uniform within each hour
    return np.maximum(np.repeat(pv_h, SLOTS_PER_H), 0.0)


# ── Market price loader ────────────────────────────────────────────────────────

def _load_market_prices(series_path: str, base_dir: str) -> np.ndarray:
    """
    Tries to load 8760 hourly market prices (€/MWh) from the JSON file
    referenced in tariffs.market_price_series.
    Falls back to _synthetic_market_prices() if the file is missing,
    unreadable, or still contains the placeholder null value.
    """
    if series_path:
        full_path = os.path.join(base_dir, series_path)
        if os.path.exists(full_path):
            with open(full_path) as f:
                data = json.load(f)
            hours = data.get("hours")
            if hours is not None and len(hours) == N_HOURS:
                return np.array(hours, dtype=float)

    # Real file not yet available — use synthetic approximation
    return _synthetic_market_prices()


def _synthetic_market_prices() -> np.ndarray:
    """
    Approximate synthetic IT_NORD day-ahead price profile (€/MWh).

    Structure: monthly base price × hourly shape × weekend discount × mild noise.
    Monthly bases reflect approximate 2024 IT_NORD averages.
    Annual average ≈ 105 €/MWh; range ≈ 40-180 €/MWh.

    The daily shape captures the typical Italian price pattern:
    cheap overnight (00-06), plateau during working hours (08-19),
    with a late-afternoon peak (17-19) driven by thermal demand.
    Weekends trade ~13% below weekdays (lower industrial demand).

    This is a calibration-quality approximation, not a forecast.
    Replace with real ENTSO-E data as soon as it is available.
    """
    # Index 0 unused; 1-12 = Jan-Dec (approximate IT_NORD 2024, €/MWh)
    MONTHLY_BASE = np.array([
        0,
        128, 115,  98,  88,  82,  92,    # Jan-Jun
        108, 102,  95, 100, 118, 125,    # Jul-Dec
    ], dtype=float)

    # Relative hourly shape within a day (multiplied by monthly base)
    HOURLY_SHAPE = np.array([
        0.63, 0.59, 0.56, 0.55, 0.57, 0.62,   # 00-05  cheap night
        0.73, 0.88, 1.00, 1.06, 1.09, 1.11,   # 06-11  morning ramp
        1.08, 1.06, 1.09, 1.15, 1.21, 1.27,   # 12-17  afternoon rise
        1.29, 1.22, 1.10, 0.96, 0.82, 0.69,   # 18-23  evening decline
    ])

    days_h   = np.arange(N_HOURS) // 24
    hours_h  = np.arange(N_HOURS) % 24
    dows_h   = days_h % 7

    months_h = np.zeros(N_HOURS, dtype=int)
    for m in range(12):
        lo = _MONTH_DAY_STARTS[m] * 24
        hi = _MONTH_DAY_STARTS[m + 1] * 24
        months_h[lo:hi] = m + 1

    base    = MONTHLY_BASE[months_h]
    shape   = HOURLY_SHAPE[hours_h]
    weekend = np.where(dows_h >= 5, 0.87, 1.0)

    prices = base * shape * weekend

    # Mild noise to avoid perfectly smooth artificial profiles
    rng   = np.random.default_rng(99)
    noise = np.clip(rng.normal(1.0, 0.06, N_HOURS), 0.75, 1.30)
    return prices * noise


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    case_path = sys.argv[1] if len(sys.argv) > 1 else "cases/example_case.json"
    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(base_dir, case_path)) as f:
        case = json.load(f)

    p = build_all_profiles(case, base_dir)

    print(f"load_kw       — sum={p['load_kw'].sum() * SLOT_H:,.0f} kWh/yr  "
          f"peak={p['load_kw'].max():.1f} kW  min={p['load_kw'].min():.1f} kW")
    print(f"pv_kw         — sum={p['pv_kw'].sum() * SLOT_H:,.0f} kWh/yr  "
          f"peak={p['pv_kw'].max():.1f} kW")
    print(f"price_eur_kwh — mean={p['price_eur_kwh'].mean():.4f} €/kWh  "
          f"min={p['price_eur_kwh'].min():.4f}  max={p['price_eur_kwh'].max():.4f}")
    print(f"time index    — months {p['month'].min()}-{p['month'].max()}  "
          f"hours {p['hour'].min()}-{p['hour'].max()}  "
          f"dow {p['dow'].min()}-{p['dow'].max()}")
    print("OK")
