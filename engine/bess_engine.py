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
    charged_fv_ac_kwh   = np.zeros(N_SLOTS)

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
            charged_fv_ac_kwh[t] = p_c * SLOT_H

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
        charged_fv_ac_kwh=charged_fv_ac_kwh,
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
    S4: BESS multilayer — self-consumption + peak shaving + overnight grid charging.

    ── Peak shaving: look-ahead dispatch ────────────────────────────────────────
    At the start of each critical day three cases are handled:

    Case 1 — SOC already sufficient:
      No grid charging needed. P_target is achievable as-is.

    Case 2 — SOC insufficient but battery large enough when full:
      Grid charging is planned for the overnight window (slots 0–27, 00:00–06:45).
      Cheapest slots are filled first (sorted by price). This brings SOC to the
      level needed to achieve the original P_target.

    Case 3 — Even a full battery cannot shave to P_target:
      Grid charging fills the battery to soc_max, then a binary search finds
      the best achievable P_target* given full capacity.

    ── Overnight charging ────────────────────────────────────────────────────────
    day_grid_c_plan[96] holds planned AC charge power (kW) for each slot.
    Slots 0–27 only (00:00–06:45). Sorted by price ascending to minimise cost.
    In the slot loop these slots take priority over all other dispatch.
    Cost is tracked in charged_grid_kwh and deducted from Layer 2 in economics.py.

    ── SOC reservation on critical days ─────────────────────────────────────────
    SC discharge is only allowed when:
        soc - soc_min - day_soc_res_dc > 0
    This makes the peak-shaving reservation inviolable.

    ── Non-critical days ─────────────────────────────────────────────────────────
    Pure S3-style self-consumption. No grid charging.
    """
    eta_c, eta_d     = bp["eta_c"], bp["eta_d"]
    soc_min, soc_max = bp["soc_min_kwh"], bp["soc_max_kwh"]
    p_c_max, p_d_max = bp["p_c_max"], bp["p_d_max"]

    n_days   = N_SLOTS // 96
    net_load = load_kw - pv_kw   # signed: >0 import deficit, <0 PV surplus

    # Monthly reference peak (S2 import, no BESS)
    daily_peak_s2 = s2_grid.clip(min=0).reshape(n_days, 96).max(axis=1)
    day_month     = month[::96][:n_days]
    monthly_peak_s2 = np.zeros(13)
    for m in range(1, 13):
        mask = day_month == m
        if mask.any():
            monthly_peak_s2[m] = daily_peak_s2[mask].max()

    shaving_target = soglia * monthly_peak_s2[day_month]   # (n_days,) kW
    critical_day   = daily_peak_s2 >= shaving_target        # (n_days,) bool

    # Output arrays
    grid_kw             = np.empty(N_SLOTS)
    bess_charge_kw      = np.zeros(N_SLOTS)
    bess_discharge_kw   = np.zeros(N_SLOTS)
    soc_arr             = np.empty(N_SLOTS)
    discharged_fv_kwh   = np.zeros(N_SLOTS)
    discharged_grid_kwh = np.zeros(N_SLOTS)
    charged_grid_kwh    = np.zeros(N_SLOTS)
    charged_fv_ac_kwh   = np.zeros(N_SLOTS)

    soc    = soc_min
    soc_fv = 0.0

    # Day-level look-ahead state (reset at t_in_day == 0)
    day_p_d_plan      = np.zeros(96)   # planned AC discharge per slot (kW)
    day_grid_c_plan   = np.zeros(96)   # planned AC grid charge per slot (kW)
    day_p_target_star = 0.0            # achievable P_target for today
    day_soc_res_dc    = 0.0            # remaining DC kWh reserved for peak slots
    day_is_critical   = False

    for t in range(N_SLOTS):
        d        = t // 96
        t_in_day = t % 96
        net      = net_load[t]
        p_c = p_d = 0.0

        # ── Plan the current day at its first slot ──────────────────────────
        if t_in_day == 0:
            day_is_critical = bool(critical_day[d])
            if day_is_critical:
                day_net        = net_load[d * 96 : d * 96 + 96]
                p_target       = float(shaving_target[d])
                soc_usable_now = soc - soc_min          # DC kWh available now
                soc_avail_max  = soc_max - soc_min      # DC kWh if battery full

                # DC energy needed to shave all peaks to p_target
                e_needed_dc = float(
                    np.maximum(0.0, day_net - p_target).sum() * SLOT_H / eta_d)

                if e_needed_dc <= soc_usable_now:
                    # Case 1: enough SOC already — no grid charging
                    p_target_star  = p_target
                    soc_deficit_dc = 0.0

                elif e_needed_dc <= soc_avail_max:
                    # Case 2: full battery sufficient — charge deficit from grid
                    p_target_star  = p_target
                    soc_deficit_dc = e_needed_dc - soc_usable_now

                else:
                    # Case 3: even full battery can't reach p_target —
                    # charge to max, binary-search best achievable target
                    soc_deficit_dc = soc_avail_max - soc_usable_now
                    lo = p_target
                    hi = float(max(day_net.max(), p_target * 1.001))
                    for _ in range(20):
                        mid   = (lo + hi) * 0.5
                        e_mid = float(
                            np.maximum(0.0, day_net - mid).sum() * SLOT_H / eta_d)
                        if e_mid <= soc_avail_max:
                            hi = mid
                        else:
                            lo = mid
                    p_target_star = hi

                day_p_d_plan   = np.minimum(
                    np.maximum(0.0, day_net - p_target_star), p_d_max)
                day_soc_res_dc = float(day_p_d_plan.sum() * SLOT_H / eta_d)
                day_p_target_star = p_target_star

                # ── Plan overnight grid charging (slots 0–27, 00:00–06:45) ──
                day_grid_c_plan = np.zeros(96)
                if soc_deficit_dc > 0:
                    energy_ac_needed = soc_deficit_dc / eta_c
                    ov_prices = price[d * 96 : d * 96 + 28]
                    sort_idx  = np.argsort(ov_prices)   # cheapest slots first
                    remaining = energy_ac_needed
                    for si in sort_idx:
                        if remaining <= 0.0:
                            break
                        e_slot = min(p_c_max * SLOT_H, remaining)
                        day_grid_c_plan[int(si)] = e_slot / SLOT_H
                        remaining -= e_slot

            else:
                day_p_d_plan      = np.zeros(96)
                day_grid_c_plan   = np.zeros(96)
                day_soc_res_dc    = 0.0
                day_p_target_star = 0.0

        # ── Slot dispatch ───────────────────────────────────────────────────
        if day_is_critical:

            if day_grid_c_plan[t_in_day] > 0:
                # Overnight grid charging — capped so total grid draw stays
                # below p_target_star (charging must not create new peaks)
                headroom = max(0.0, day_p_target_star - net)
                p_c = min(day_grid_c_plan[t_in_day],
                          headroom,
                          (soc_max - soc) / (eta_c * SLOT_H))
                p_c = max(p_c, 0.0)
                if p_c > 0:
                    e_in = p_c * eta_c * SLOT_H
                    soc += e_in
                    # soc_fv unchanged — this energy is from the grid
                    charged_grid_kwh[t] = p_c * SLOT_H

            elif net < 0:   # PV surplus → charge (always allowed)
                surplus = -net
                p_c     = min(surplus, p_c_max,
                              (soc_max - soc) / (eta_c * SLOT_H))
                p_c     = max(p_c, 0.0)
                e_in    = p_c * eta_c * SLOT_H
                soc    += e_in
                soc_fv += e_in
                charged_fv_ac_kwh[t] = p_c * SLOT_H

            elif net > day_p_target_star:   # peak window: execute planned discharge
                p_d = min(day_p_d_plan[t_in_day],
                          (soc - soc_min) * eta_d / SLOT_H,
                          p_d_max)
                p_d = max(p_d, 0.0)
                if p_d > 0:
                    e_dc       = p_d / eta_d * SLOT_H
                    fv_ratio   = soc_fv / soc if soc > 0 else 0.0
                    discharged_fv_kwh[t]   = e_dc * fv_ratio
                    discharged_grid_kwh[t] = e_dc * (1.0 - fv_ratio)
                    soc    -= e_dc
                    soc_fv  = max(soc_fv - discharged_fv_kwh[t], 0.0)
                # Decrement reservation by planned slot energy (regardless of actual)
                day_soc_res_dc = max(
                    0.0, day_soc_res_dc - day_p_d_plan[t_in_day] * SLOT_H / eta_d)

            elif net > 0:   # 0 < net ≤ p_target_star: SC if above reservation
                soc_above = soc - soc_min - day_soc_res_dc
                if soc_above > 0:
                    p_d = min(net, p_d_max, soc_above * eta_d / SLOT_H)
                    p_d = max(p_d, 0.0)
                    if p_d > 0:
                        e_dc       = p_d / eta_d * SLOT_H
                        fv_ratio   = soc_fv / soc if soc > 0 else 0.0
                        discharged_fv_kwh[t]   = e_dc * fv_ratio
                        discharged_grid_kwh[t] = e_dc * (1.0 - fv_ratio)
                        soc    -= e_dc
                        soc_fv  = max(soc_fv - discharged_fv_kwh[t], 0.0)

        else:   # non-critical day: S3-style SC, no grid charging
            if net < 0:   # PV surplus → charge
                surplus = -net
                p_c     = min(surplus, p_c_max,
                              (soc_max - soc) / (eta_c * SLOT_H))
                p_c     = max(p_c, 0.0)
                e_in    = p_c * eta_c * SLOT_H
                soc    += e_in
                soc_fv += e_in
                charged_fv_ac_kwh[t] = p_c * SLOT_H

            elif net > 0:   # load deficit → discharge for SC
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
        charged_fv_ac_kwh=charged_fv_ac_kwh,
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
    charged_fv_ac_kwh:   np.ndarray | None = None,
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
        "charged_fv_ac_kwh":   z.copy() if charged_fv_ac_kwh   is None else np.asarray(charged_fv_ac_kwh,   dtype=float),
    }

