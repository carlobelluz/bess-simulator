"""
bess_engine.py
Slot-by-slot BESS simulation for scenarios S1-S4.

Convention — all 'kw' arrays are AC-side:
  bess_charge_kw[t]    : AC power drawn by charger from bus  (kW)
  bess_discharge_kw[t] : AC power delivered by inverter to bus (kW)
  grid_kw[t]           : net grid draw (+import / −export, kW)

SOC update:
  charge    : ΔSOC = +bess_charge_kw[t] * eta_c * SLOT_H   (DC kWh stored)
  discharge : ΔSOC = −bess_discharge_kw[t] / eta_d * SLOT_H (DC kWh removed)

Grid identity:
  grid_kw[t] = (load_kw[t] − pv_kw[t]) + bess_charge_kw[t] − bess_discharge_kw[t]
"""

import json
import math
import os

import numpy as np

SLOT_H  = 0.25    # hours per 15-min slot
N_SLOTS = 35040


# ── Public API ────────────────────────────────────────────────────────────────

def simulate(case: dict, profiles: dict) -> dict:
    """
    Runs all scenarios listed in case["simulation"]["scenari"].

    Args:
      case     : loaded case dict (example_case.json)
      profiles : output of profile_builder.build_all_profiles()

    Returns:
      { "S1": result, "S2": result, "S3": result, "S4": result }

    Each result is a dict with 7 arrays of shape (35 040,):
      grid_kw              kW   — net grid draw (+import / −export)
      bess_charge_kw       kW   — AC power into charger
      bess_discharge_kw    kW   — AC power from inverter
      soc_kwh              kWh  — state of charge at END of slot
      discharged_fv_kwh    kWh  — DC energy discharged from PV-sourced SOC
      discharged_grid_kwh  kWh  — DC energy discharged from grid-sourced SOC
      charged_grid_kwh     kWh  — AC energy drawn from grid for charging
    """
    bess    = case["bess"]
    sim     = case["simulation"]
    scenari = sim["scenari"]

    load_kw = profiles["load_kw"]
    pv_kw   = profiles["pv_kw"]
    price   = profiles["price_eur_kwh"]
    month   = profiles["month"]

    eta_rt      = bess["efficienza_roundtrip"]
    eta_c       = math.sqrt(eta_rt)
    eta_d       = math.sqrt(eta_rt)
    cap_nom     = bess["capacita_nominale_kwh"]
    soc_min_kwh = bess["soc_min"] * cap_nom
    soc_max_kwh = bess["soc_max"] * cap_nom
    p_c_max     = bess["potenza_carica_kw"]
    p_d_max     = bess["potenza_scarica_kw"]

    bp = {
        "eta_c": eta_c, "eta_d": eta_d,
        "soc_min_kwh": soc_min_kwh, "soc_max_kwh": soc_max_kwh,
        "p_c_max": p_c_max, "p_d_max": p_d_max,
    }

    results = {}

    if "S1" in scenari:
        results["S1"] = _run_s1(load_kw)

    if "S2" in scenari:
        results["S2"] = _run_s2(load_kw, pv_kw)

    if "S3" in scenari:
        results["S3"] = _run_s3(load_kw, pv_kw, bp)

    if "S4" in scenari:
        s2_grid = (results["S2"]["grid_kw"]
                   if "S2" in results
                   else load_kw - pv_kw)
        soglia  = sim.get("soglia_critical_day_pct", 0.85)
        results["S4"] = _run_s4(load_kw, pv_kw, price, month,
                                 s2_grid, soglia, bp)

    return results


# ── Scenario runners ───────────────────────────────────────────────────────────

def _run_s1(load_kw: np.ndarray) -> dict:
    """S1: no PV, no BESS. Pure grid draw."""
    return _make_result(grid_kw=load_kw.copy())


def _run_s2(load_kw: np.ndarray, pv_kw: np.ndarray) -> dict:
    """S2: PV present, no BESS. Grid draw = load − PV (negative = export)."""
    return _make_result(grid_kw=load_kw - pv_kw)


def _run_s3(load_kw: np.ndarray, pv_kw: np.ndarray, bp: dict) -> dict:
    """
    S3: BESS for self-consumption only.

    - Charges from PV surplus; discharges to cover load deficit.
    - No grid charging — charged_grid_kwh is always zero.
    - Tracks FV/grid SOC split so economics can attribute
      discharged energy to the correct source.
    """
    eta_c, eta_d     = bp["eta_c"], bp["eta_d"]
    soc_min, soc_max = bp["soc_min_kwh"], bp["soc_max_kwh"]
    p_c_max, p_d_max = bp["p_c_max"], bp["p_d_max"]

    grid_kw             = np.empty(N_SLOTS)
    bess_charge_kw      = np.zeros(N_SLOTS)
    bess_discharge_kw   = np.zeros(N_SLOTS)
    soc_arr             = np.empty(N_SLOTS)
    discharged_fv_kwh   = np.zeros(N_SLOTS)
    discharged_grid_kwh = np.zeros(N_SLOTS)
    charged_grid_kwh    = np.zeros(N_SLOTS)   # always 0 in S3

    soc    = soc_min
    soc_fv = 0.0   # kWh of SOC that originated from PV

    for t in range(N_SLOTS):
        net = load_kw[t] - pv_kw[t]   # >0 deficit, <0 surplus
        p_c = p_d = 0.0

        if net < 0:   # PV surplus → charge battery
            surplus = -net
            p_c     = min(surplus, p_c_max,
                          (soc_max - soc) / (eta_c * SLOT_H))
            p_c     = max(p_c, 0.0)
            e_in    = p_c * eta_c * SLOT_H
            soc    += e_in
            soc_fv += e_in

        elif net > 0:   # load deficit → discharge battery
            p_d_avail = (soc - soc_min) * eta_d / SLOT_H
            p_d       = min(net, p_d_max, p_d_avail)
            p_d       = max(p_d, 0.0)
            if p_d > 0:
                e_dc       = p_d / eta_d * SLOT_H   # DC kWh removed from SOC
                fv_ratio   = soc_fv / soc if soc > 0 else 0.0
                discharged_fv_kwh[t]   = e_dc * fv_ratio
                discharged_grid_kwh[t] = e_dc * (1.0 - fv_ratio)
                soc    -= e_dc
                soc_fv  = max(soc_fv - discharged_fv_kwh[t], 0.0)

        bess_charge_kw[t]    = p_c
        bess_discharge_kw[t] = p_d
        grid_kw[t]           = net + p_c - p_d
        soc_arr[t]           = soc

    return _make_result(
        grid_kw=grid_kw,
        bess_charge_kw=bess_charge_kw,
        bess_discharge_kw=bess_discharge_kw,
        soc_kwh=soc_arr,
        discharged_fv_kwh=discharged_fv_kwh,
        discharged_grid_kwh=discharged_grid_kwh,
        charged_grid_kwh=charged_grid_kwh,
    )


def _run_s4(
    load_kw: np.ndarray,
    pv_kw:   np.ndarray,
    price:   np.ndarray,
    month:   np.ndarray,
    s2_grid: np.ndarray,
    soglia:  float,
    bp:      dict,
) -> dict:
    """
    S4: BESS multilayer — self-consumption + peak shaving + shifting.

    ── Pre-computation ──────────────────────────────────────────────────────────
    Monthly peak reference: max daily import peak (S2 grid, clipped to ≥0) per month.
    Critical day : daily peak ≥ soglia × monthly_peak.
    Shaving target: soglia × monthly_peak — the draw level we aim to stay below.

    ── Dispatch on CRITICAL days (peak shaving priority) ────────────────────────
    - Charge from PV surplus normally.
    - Discharge ONLY when net > shaving_target:
        remove just enough SOC to bring grid draw down to the target.
    - When 0 ≤ net ≤ shaving_target: battery holds back (no action).
    - No grid charging on critical days.

    ── Dispatch on NON-CRITICAL days (self-consumption + shifting) ──────────────
    - S3-style: charge from PV surplus, discharge to cover load deficit.
    - [PROVISIONAL — v1] Grid charging for arbitrage:
        triggered when no charge/discharge is already active
        AND price[t] < daily mean price
        AND soc < 80 % of soc_max.
        Rationale: buy cheap electricity, sell back later as avoided peak import.
        This rule is intentionally naive; it will be replaced in v2 with a
        look-ahead or optimization-based charging schedule.

    ── SOC source tracking ──────────────────────────────────────────────────────
    Same FV/grid proportion tracking as S3.
    Grid-charged energy does NOT increment soc_fv.
    """
    eta_c, eta_d     = bp["eta_c"], bp["eta_d"]
    soc_min, soc_max = bp["soc_min_kwh"], bp["soc_max_kwh"]
    p_c_max, p_d_max = bp["p_c_max"], bp["p_d_max"]

    n_days = N_SLOTS // 96

    # Daily import peak from S2 (reference without BESS, clip exports to 0)
    daily_peak_s2 = (s2_grid.clip(min=0)
                     .reshape(n_days, 96)
                     .max(axis=1))                  # (365,) kW

    day_month = month[::96][:n_days]                # (365,) values 1-12

    # Monthly peak = max of all daily import peaks in that month (1-indexed)
    monthly_peak_s2 = np.zeros(13)
    for m in range(1, 13):
        mask = day_month == m
        if mask.any():
            monthly_peak_s2[m] = daily_peak_s2[mask].max()

    # Per-day shaving target and critical flag (vectorised)
    shaving_target = soglia * monthly_peak_s2[day_month]   # (365,) kW
    critical_day   = daily_peak_s2 >= shaving_target        # (365,) bool

    # Daily mean price used by the provisional grid-charging rule
    daily_mean_price = price.reshape(n_days, 96).mean(axis=1)   # (365,)

    # ── Slot loop ─────────────────────────────────────────────────────────────
    grid_kw             = np.empty(N_SLOTS)
    bess_charge_kw      = np.zeros(N_SLOTS)
    bess_discharge_kw   = np.zeros(N_SLOTS)
    soc_arr             = np.empty(N_SLOTS)
    discharged_fv_kwh   = np.zeros(N_SLOTS)
    discharged_grid_kwh = np.zeros(N_SLOTS)
    charged_grid_kwh    = np.zeros(N_SLOTS)

    soc    = soc_min
    soc_fv = 0.0

    for t in range(N_SLOTS):
        d   = t // 96
        net = load_kw[t] - pv_kw[t]   # >0 deficit, <0 surplus
        p_c = p_d = 0.0

        if critical_day[d]:
            # ── Peak shaving priority: hold SOC for high-demand moments ──────
            if net < 0:                              # PV surplus — top up battery
                surplus = -net
                p_c     = min(surplus, p_c_max,
                              (soc_max - soc) / (eta_c * SLOT_H))
                p_c     = max(p_c, 0.0)
                e_in    = p_c * eta_c * SLOT_H
                soc    += e_in
                soc_fv += e_in

            elif net > shaving_target[d]:            # grid draw exceeds target
                excess    = net - shaving_target[d]
                p_d_avail = (soc - soc_min) * eta_d / SLOT_H
                p_d       = min(excess, p_d_max, p_d_avail)
                p_d       = max(p_d, 0.0)
                if p_d > 0:
                    e_dc       = p_d / eta_d * SLOT_H
                    fv_ratio   = soc_fv / soc if soc > 0 else 0.0
                    discharged_fv_kwh[t]   = e_dc * fv_ratio
                    discharged_grid_kwh[t] = e_dc * (1.0 - fv_ratio)
                    soc    -= e_dc
                    soc_fv  = max(soc_fv - discharged_fv_kwh[t], 0.0)
            # 0 ≤ net ≤ shaving_target: battery holds back, no action

        else:
            # ── Non-critical day: self-consumption ───────────────────────────
            if net < 0:                              # PV surplus → charge
                surplus = -net
                p_c     = min(surplus, p_c_max,
                              (soc_max - soc) / (eta_c * SLOT_H))
                p_c     = max(p_c, 0.0)
                e_in    = p_c * eta_c * SLOT_H
                soc    += e_in
                soc_fv += e_in

            elif net > 0:                            # load deficit → discharge
                p_d_avail = (soc - soc_min) * eta_d / SLOT_H
                p_d       = min(net, p_d_max, p_d_avail)
                p_d       = max(p_d, 0.0)
                if p_d > 0:
                    e_dc       = p_d / eta_d * SLOT_H
                    fv_ratio   = soc_fv / soc if soc > 0 else 0.0
                    discharged_fv_kwh[t]   = e_dc * fv_ratio
                    discharged_grid_kwh[t] = e_dc * (1.0 - fv_ratio)
                    soc    -= e_dc
                    soc_fv  = max(soc_fv - discharged_fv_kwh[t], 0.0)

            # [PROVISIONAL — v1] Grid charging for shifting arbitrage.
            # Will be replaced with a look-ahead or optimised rule in v2.
            if (p_c == 0 and p_d == 0
                    and price[t] < daily_mean_price[d]
                    and soc < 0.8 * soc_max):
                p_c  = min(p_c_max, (soc_max - soc) / (eta_c * SLOT_H))
                p_c  = max(p_c, 0.0)
                e_in = p_c * eta_c * SLOT_H
                soc += e_in
                # Grid-charged energy does NOT increment soc_fv
                charged_grid_kwh[t] = p_c * SLOT_H   # AC kWh drawn from grid

        bess_charge_kw[t]    = p_c
        bess_discharge_kw[t] = p_d
        grid_kw[t]           = net + p_c - p_d
        soc_arr[t]           = soc

    return _make_result(
        grid_kw=grid_kw,
        bess_charge_kw=bess_charge_kw,
        bess_discharge_kw=bess_discharge_kw,
        soc_kwh=soc_arr,
        discharged_fv_kwh=discharged_fv_kwh,
        discharged_grid_kwh=discharged_grid_kwh,
        charged_grid_kwh=charged_grid_kwh,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_result(
    grid_kw:             np.ndarray,
    bess_charge_kw:      np.ndarray | None = None,
    bess_discharge_kw:   np.ndarray | None = None,
    soc_kwh:             np.ndarray | None = None,
    discharged_fv_kwh:   np.ndarray | None = None,
    discharged_grid_kwh: np.ndarray | None = None,
    charged_grid_kwh:    np.ndarray | None = None,
) -> dict:
    """Packs scenario arrays into a uniform result dict; absent arrays → zeros."""
    z = np.zeros(N_SLOTS)
    return {
        "grid_kw":             np.asarray(grid_kw,             dtype=float),
        "bess_charge_kw":      z.copy() if bess_charge_kw      is None else np.asarray(bess_charge_kw,      dtype=float),
        "bess_discharge_kw":   z.copy() if bess_discharge_kw   is None else np.asarray(bess_discharge_kw,   dtype=float),
        "soc_kwh":             z.copy() if soc_kwh             is None else np.asarray(soc_kwh,             dtype=float),
        "discharged_fv_kwh":   z.copy() if discharged_fv_kwh   is None else np.asarray(discharged_fv_kwh,   dtype=float),
        "discharged_grid_kwh": z.copy() if discharged_grid_kwh is None else np.asarray(discharged_grid_kwh, dtype=float),
        "charged_grid_kwh":    z.copy() if charged_grid_kwh    is None else np.asarray(charged_grid_kwh,    dtype=float),
    }


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine.profile_builder import build_all_profiles  # noqa: E402

    case_path = sys.argv[1] if len(sys.argv) > 1 else "cases/example_case.json"
    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(base_dir, case_path)) as f:
        case = json.load(f)

    profiles = build_all_profiles(case, base_dir)
    results  = simulate(case, profiles)

    cap_nom = case["bess"]["capacita_nominale_kwh"]
    print(f"{'Scenario':<6}  {'Grid import':>14}  {'Peak kW':>8}  "
          f"{'Throughput':>12}  {'Cycles/yr':>10}  {'Grid export':>12}")
    print("-" * 72)
    for sc, r in results.items():
        g         = r["grid_kw"]
        import_kwh = g.clip(min=0).sum() * SLOT_H
        export_kwh = (-g).clip(min=0).sum() * SLOT_H
        peak_kw    = g.max()
        throughput = r["bess_discharge_kw"].sum() * SLOT_H
        cycles     = throughput / cap_nom if cap_nom > 0 else 0.0
        print(f"{sc:<6}  {import_kwh:>14,.0f}  {peak_kw:>8.1f}  "
              f"{throughput:>12,.0f}  {cycles:>10.0f}  {export_kwh:>12,.0f}")
    print("OK")
