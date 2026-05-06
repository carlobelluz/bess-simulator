"""
economics.py
Computes all economic KPIs from simulation results.

Three saving layers (always kept separate):
  Layer 1 — FV self-consumption saving     (S3 and S4)
  Layer 2 — Quota potenza / peak shaving   (S4 only in v1)
  Layer 3 — Energy shifting margin         (S4 only: grid charging arbitrage)

Multi-year model:
  Year 0  : −investment
  Year y  : (base_saving × (1 − degradazione_annua)^(y−1)) − O&M_annual
"""

import numpy as np
import numpy_financial as npf

SLOT_H  = 0.25
N_SLOTS = 35040


# ── Public API ────────────────────────────────────────────────────────────────

def compute_economics(case: dict, profiles: dict, results: dict) -> dict:
    """
    Computes economic KPIs for all simulated scenarios.

    S1 / S2 : annual energy cost and peak summary (no BESS investment).
    S3 / S4 : full economics — savings by layer, payback, NPV, IRR.

    Returns a dict keyed by scenario ID.
    """
    tariffs = case["tariffs"]
    bess    = case["bess"]
    sim     = case["simulation"]

    price        = profiles["price_eur_kwh"]
    month        = profiles["month"]

    quota_kw_mese   = tariffs["quota_potenza_eur_kw_mese"]
    investment      = bess["capacita_nominale_kwh"] * bess["costo_installato_eur_kwh"]
    anni            = sim["anni_analisi"]
    discount_rate   = sim["tasso_sconto"]
    om_rate         = sim["om_rate"]
    degradazione    = sim["degradazione_annua"]
    anni_vita       = bess.get("anni_vita", anni + 1)
    cap_nom         = bess["capacita_nominale_kwh"]
    eta_d           = np.sqrt(bess["efficienza_roundtrip"])

    load_kw        = np.asarray(profiles["load_kw"])
    pv_kw          = np.asarray(profiles["pv_kw"])
    load_total_kwh = float(load_kw.sum() * SLOT_H)
    pv_total_kwh   = float(pv_kw.sum() * SLOT_H)
    direct_sc_kwh  = float(np.minimum(load_kw, pv_kw).sum() * SLOT_H)

    out = {}

    # S1 and S2 — reference cost summaries
    if "S1" in results:
        out["S1"] = _reference_kpis(results["S1"]["grid_kw"], price, month,
                                    quota_kw_mese)

    s2_grid = None
    if "S2" in results:
        s2_grid = results["S2"]["grid_kw"]
        out["S2"] = _reference_kpis(s2_grid, price, month, quota_kw_mese,
                                    direct_sc_kwh, pv_total_kwh, load_total_kwh)

    # S3 and S4 — BESS economic analysis
    for sc in ["S3", "S4"]:
        if sc not in results:
            continue

        r_sc    = results[sc]
        grid_kw = r_sc["grid_kw"]

        # ── Layer 1: FV self-consumption saving ───────────────────────────────
        # Gross value: DC kWh discharged × η_d (→ AC kWh) × price at discharge
        # Opportunity cost: AC kWh taken from FV surplus for charging ×
        #   export value foregone (0 if no incentive scheme, price if SSP, etc.)
        export_value = _get_export_value(case["pv"], price)
        saving_fv = float(
            (r_sc["discharged_fv_kwh"] * eta_d * price).sum()
            - (r_sc["charged_fv_ac_kwh"] * export_value).sum()
        )

        # ── Layer 2: quota potenza saving — S4 only ───────────────────────────
        # Reduction in monthly demand peak × monthly power tariff.
        # In S3 the peak effect is incidental; only S4 explicitly targets it.
        if sc == "S4" and s2_grid is not None:
            saving_quota = _quota_potenza_saving(
                s2_grid, grid_kw, month, quota_kw_mese
            )
        else:
            saving_quota = 0.0

        # ── Layer 3: shifting margin — S4 only ───────────────────────────────
        # Revenue = avoided import when discharging grid-sourced energy
        # Cost    = grid energy purchased during cheap-price charging
        # Zeroed when no grid charging occurred (disabled in MVP)
        if sc == "S4" and np.asarray(r_sc["charged_grid_kwh"]).sum() > 0:
            revenue_shifting = (r_sc["discharged_grid_kwh"] * eta_d * price).sum()
            cost_grid_charge = (r_sc["charged_grid_kwh"] * price).sum()
            shifting_margin  = revenue_shifting - cost_grid_charge
        else:
            shifting_margin = 0.0

        annual_saving = saving_fv + saving_quota + shifting_margin

        # ── Battery technical KPIs ────────────────────────────────────────────
        throughput = r_sc["bess_discharge_kw"].sum() * SLOT_H
        eq_cycles  = throughput / cap_nom if cap_nom > 0 else 0.0

        # ── Multi-year cash flows ─────────────────────────────────────────────
        om_annual         = investment * om_rate
        replacement_capex = investment if anni_vita <= anni else 0.0

        cash_flows    = np.empty(anni + 1)
        cash_flows[0] = -investment
        for y in range(1, anni + 1):
            # After battery replacement, degradation exponent resets to 0
            exponent      = y - 1 if (anni_vita > anni or y <= anni_vita) else y - anni_vita - 1
            saving_y      = annual_saving * (1.0 - degradazione) ** exponent
            cash_flows[y] = saving_y - om_annual
        # Replacement capex at end of year anni_vita (same year as last operating year)
        if anni_vita <= anni:
            cash_flows[anni_vita] -= replacement_capex

        # ── Self-consumption KPIs ─────────────────────────────────────────────
        bess_sc_ac_kwh = float(np.asarray(r_sc["discharged_fv_kwh"]).sum()) * eta_d
        total_sc_kwh   = direct_sc_kwh + bess_sc_ac_kwh
        scr_pct        = total_sc_kwh / pv_total_kwh * 100 if pv_total_kwh > 0 else 0.0
        ssr_pct        = total_sc_kwh / load_total_kwh * 100 if load_total_kwh > 0 else 0.0

        # ── Demand charge under this scenario ────────────────────────────────
        n_d    = N_SLOTS // 96
        d_mon  = month[::96][:n_d]
        d_pk   = np.asarray(grid_kw).clip(min=0).reshape(n_d, 96).max(axis=1)
        annual_demand_charge = sum(
            float(d_pk[d_mon == m].max()) * quota_kw_mese
            if (d_mon == m).any() else 0.0
            for m in range(1, 13)
        )

        payback = _payback(cash_flows)
        npv_val = float(npf.npv(discount_rate, cash_flows))
        try:
            irr_raw = npf.irr(cash_flows)
            irr_val = round(float(irr_raw) * 100.0, 2) if np.isfinite(irr_raw) else None
        except Exception:
            irr_val = None

        out[sc] = {
            # Savings breakdown
            "saving_fv_eur":       round(saving_fv, 2),
            "saving_quota_eur":    round(saving_quota, 2),
            "shifting_margin_eur": round(shifting_margin, 2),
            "annual_saving_eur":   round(annual_saving, 2),
            # Self-consumption KPIs
            "direct_selfcons_kwh": round(direct_sc_kwh, 1),
            "bess_sc_kwh":         round(bess_sc_ac_kwh, 1),
            "total_selfcons_kwh":  round(total_sc_kwh, 1),
            "scr_pct":             round(scr_pct, 1),
            "ssr_pct":             round(ssr_pct, 1),
            # Demand charge
            "annual_demand_charge_eur": round(annual_demand_charge, 2),
            # Battery KPIs
            "throughput_kwh":      round(throughput, 1),
            "equivalent_cycles":   round(eq_cycles, 1),
            # Investment
            "investment_eur":      round(investment, 2),
            "om_annual_eur":       round(om_annual, 2),
            # Financial KPIs
            "payback_yr":               payback,
            "npv_eur":                  round(npv_val, 2),
            "irr_pct":                  irr_val,
            "cashflows":                cash_flows.tolist(),
            # Battery lifecycle
            "battery_replacement_year": anni_vita if anni_vita <= anni else None,
            "replacement_capex_eur":    round(replacement_capex, 2),
        }

    # S_FV: proposed FV investment for clients without existing PV
    sfv = _compute_sfv_economics(case, profiles)
    if sfv:
        out["S_FV"] = sfv

    return out


# ── Internal helpers ──────────────────────────────────────────────────────────

def _reference_kpis(
    grid_kw:        np.ndarray,
    price:          np.ndarray,
    month:          np.ndarray,
    quota_kw_mese:  float = 0.0,
    direct_sc_kwh:  float = 0.0,
    pv_total_kwh:   float = 0.0,
    load_total_kwh: float = 0.0,
) -> dict:
    """Annual cost and peak summary for S1 / S2 (no BESS)."""
    n_days    = N_SLOTS // 96
    day_month = month[::96][:n_days]
    day_peaks = grid_kw.clip(min=0).reshape(n_days, 96).max(axis=1)

    import_kwh  = grid_kw.clip(min=0).sum() * SLOT_H
    export_kwh  = (-grid_kw).clip(min=0).sum() * SLOT_H
    energy_cost = (grid_kw.clip(min=0) * SLOT_H * price).sum()

    monthly_peak = np.zeros(12)
    for m in range(1, 13):
        mask = day_month == m
        if mask.any():
            monthly_peak[m - 1] = day_peaks[mask].max()

    annual_demand_charge = float(monthly_peak.sum()) * quota_kw_mese
    scr = direct_sc_kwh / pv_total_kwh * 100 if pv_total_kwh > 0 else 0.0
    ssr = direct_sc_kwh / load_total_kwh * 100 if load_total_kwh > 0 else 0.0

    return {
        "annual_grid_import_kwh":   round(import_kwh),
        "annual_grid_export_kwh":   round(export_kwh),
        "annual_energy_cost_eur":   round(energy_cost, 2),
        "annual_demand_charge_eur": round(annual_demand_charge, 2),
        "monthly_peak_kw":          [round(p, 1) for p in monthly_peak],
        "direct_selfcons_kwh":      round(direct_sc_kwh, 1),
        "scr_pct":                  round(scr, 1),
        "ssr_pct":                  round(ssr, 1),
    }


def _quota_potenza_saving(
    s2_grid:       np.ndarray,
    sc_grid:       np.ndarray,
    month:         np.ndarray,
    quota_kw_mese: float,
) -> float:
    """
    Annual saving from reduced monthly demand peaks.

    For each month: saving = max(0, s2_peak − sc_peak) × quota_potenza_eur_kw_mese.
    Uses import-only peaks (exports clipped to 0).
    """
    n_days    = N_SLOTS // 96
    day_month = month[::96][:n_days]
    s2_peaks  = s2_grid.clip(min=0).reshape(n_days, 96).max(axis=1)
    sc_peaks  = sc_grid.clip(min=0).reshape(n_days, 96).max(axis=1)

    saving = 0.0
    for m in range(1, 13):
        mask = day_month == m
        if not mask.any():
            continue
        reduction = max(0.0, float(s2_peaks[mask].max()) - float(sc_peaks[mask].max()))
        saving   += reduction * quota_kw_mese

    return saving


def _payback(cash_flows: np.ndarray) -> float | None:
    """
    Simple payback year with linear interpolation.
    Returns None if not reached within the analysis horizon.
    """
    cumulative = np.cumsum(cash_flows)
    for y in range(1, len(cash_flows)):
        if cumulative[y] >= 0:
            if cumulative[y - 1] < 0:
                frac = -cumulative[y - 1] / (cumulative[y] - cumulative[y - 1])
                return round(y - 1 + frac, 2)
            return float(y)
    return None


def _compute_sfv_economics(case: dict, profiles: dict) -> dict | None:
    """
    Economics for a proposed FV investment (S_FV scenario).

    Used when the client has no existing PV and wants to evaluate adding one.
    Requires case["pv_proposto"] and profiles["pv_kw_proposto"].

    Saving = slot-by-slot min(load, pv) × SLOT_H × price
    (direct self-consumption only; export is treated as zero value).
    """
    pv_prop = case.get("pv_proposto")
    if not pv_prop or "pv_kw_proposto" not in profiles:
        return None

    pv_kw   = np.asarray(profiles["pv_kw_proposto"])
    load_kw = np.asarray(profiles["load_kw"])
    price   = np.asarray(profiles["price_eur_kwh"])

    kwp          = pv_prop["kwp"]
    costo_kwp    = pv_prop.get("costo_eur_kwp", 800)
    anni         = pv_prop.get("anni_vita", 25)
    degradazione = pv_prop.get("degradazione_annua", 0.005)
    discount_rate = case.get("simulation", {}).get("tasso_sconto", 0.05)

    # Year-1 saving = price avoided by directly consuming FV output
    sc_kw      = np.minimum(load_kw, pv_kw)
    saving_y1  = float((sc_kw * SLOT_H * price).sum())
    direct_sc  = float(sc_kw.sum() * SLOT_H)
    pv_total   = float(pv_kw.sum() * SLOT_H)
    load_total = float(load_kw.sum() * SLOT_H)

    investment = kwp * costo_kwp

    cash_flows    = np.empty(anni + 1)
    cash_flows[0] = -investment
    for y in range(1, anni + 1):
        cash_flows[y] = saving_y1 * (1.0 - degradazione) ** (y - 1)

    payback  = _payback(cash_flows)
    npv_val  = float(npf.npv(discount_rate, cash_flows))
    try:
        irr_raw = npf.irr(cash_flows)
        irr_val = round(float(irr_raw) * 100.0, 2) if np.isfinite(irr_raw) else None
    except Exception:
        irr_val = None

    scr = direct_sc / pv_total   * 100 if pv_total   > 0 else 0.0
    ssr = direct_sc / load_total * 100 if load_total > 0 else 0.0

    return {
        "kwp":                 kwp,
        "investment_eur":      round(investment, 2),
        "costo_eur_kwp":       costo_kwp,
        "saving_y1_eur":       round(saving_y1, 2),
        "pv_total_kwh":        round(pv_total, 1),
        "direct_selfcons_kwh": round(direct_sc, 1),
        "scr_pct":             round(scr, 1),
        "ssr_pct":             round(ssr, 1),
        "payback_yr":          payback,
        "npv_eur":             round(npv_val, 2),
        "irr_pct":             irr_val,
        "cashflows":           cash_flows.tolist(),
    }


def _get_export_value(pv: dict, price: np.ndarray) -> float | np.ndarray:
    """
    Returns the per-slot value (€/kWh) of FV energy exported to the grid.

    Used to compute the opportunity cost of storing FV energy in the battery
    rather than exporting it — which is the correct basis for Layer 1.

    Regime mapping:
      "nessuno"          → 0.0  (no export incentive — current default)
      "ritiro_dedicato"  → ~85% of market price (GME PUN minus handling fee)
      "ssp"              → full import price (Scambio sul Posto: 1:1 virtual net metering)
      explicit value     → fv_export_value_eur_kwh overrides regime
    """
    explicit = pv.get("fv_export_value_eur_kwh")
    if explicit is not None:
        return float(explicit)
    regime = pv.get("fv_export_regime", "nessuno")
    if regime == "ssp":
        return price                  # full import price per slot
    if regime == "ritiro_dedicato":
        return price * 0.85           # approximate: market without taxes/spread
    return 0.0                        # "nessuno" — preserves current behaviour


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine.profile_builder import build_all_profiles
    from engine.bess_engine import simulate

    case_path = sys.argv[1] if len(sys.argv) > 1 else "cases/example_case.json"
    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(base_dir, case_path)) as f:
        case = json.load(f)

    profiles = build_all_profiles(case, base_dir)
    sim_results = simulate(case, profiles)
    econ = compute_economics(case, profiles, sim_results)

    print("── Reference scenarios ───────────────────────────────────────")
    for sc in ["S1", "S2"]:
        if sc not in econ:
            continue
        e = econ[sc]
        print(f"{sc}:  import={e['annual_grid_import_kwh']:>9,.0f} kWh/yr  "
              f"export={e['annual_grid_export_kwh']:>8,.0f} kWh/yr  "
              f"cost={e['annual_energy_cost_eur']:>10,.0f} €/yr")

    print("\n── BESS scenarios ────────────────────────────────────────────")
    for sc in ["S3", "S4"]:
        if sc not in econ:
            continue
        e = econ[sc]
        print(f"{sc}:  Layer1(FV)={e['saving_fv_eur']:>8,.0f} €  "
              f"Layer2(peak)={e['saving_quota_eur']:>7,.0f} €  "
              f"Layer3(shift)={e['shifting_margin_eur']:>8,.0f} €")
        print(f"    Total saving={e['annual_saving_eur']:>8,.0f} €/yr  "
              f"cycles={e['equivalent_cycles']:.0f}/yr  "
              f"payback={e['payback_yr']} yr  "
              f"NPV={e['npv_eur']:>10,.0f} €  "
              f"IRR={e['irr_pct']}%")
    print("OK")
