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
    cap_nom         = bess["capacita_nominale_kwh"]
    eta_d           = np.sqrt(bess["efficienza_roundtrip"])

    out = {}

    # S1 and S2 — reference cost summaries
    if "S1" in results:
        out["S1"] = _reference_kpis(results["S1"]["grid_kw"], price, month)

    s2_grid = None
    if "S2" in results:
        s2_grid = results["S2"]["grid_kw"]
        out["S2"] = _reference_kpis(s2_grid, price, month)

    # S3 and S4 — BESS economic analysis
    for sc in ["S3", "S4"]:
        if sc not in results:
            continue

        r_sc    = results[sc]
        grid_kw = r_sc["grid_kw"]

        # ── Layer 1: FV self-consumption saving ───────────────────────────────
        # discharged_fv_kwh[t] = DC kWh from PV-sourced SOC
        # AC energy delivered   = discharged_fv_kwh[t] * eta_d
        # Valued at spot price  = price[t]
        saving_fv = (r_sc["discharged_fv_kwh"] * eta_d * price).sum()

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
        if sc == "S4":
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
        om_annual  = investment * om_rate
        cash_flows = np.empty(anni + 1)
        cash_flows[0] = -investment
        for y in range(1, anni + 1):
            saving_y       = annual_saving * (1.0 - degradazione) ** (y - 1)
            cash_flows[y]  = saving_y - om_annual

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
            # Battery KPIs
            "throughput_kwh":      round(throughput, 1),
            "equivalent_cycles":   round(eq_cycles, 1),
            # Investment
            "investment_eur":      round(investment, 2),
            "om_annual_eur":       round(om_annual, 2),
            # Financial KPIs
            "payback_yr":          payback,
            "npv_eur":             round(npv_val, 2),
            "irr_pct":             irr_val,
            "cashflows":           cash_flows.tolist(),
        }

    return out


# ── Internal helpers ──────────────────────────────────────────────────────────

def _reference_kpis(grid_kw: np.ndarray, price: np.ndarray, month: np.ndarray) -> dict:
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

    return {
        "annual_grid_import_kwh": round(import_kwh),
        "annual_grid_export_kwh": round(export_kwh),
        "annual_energy_cost_eur": round(energy_cost, 2),
        "monthly_peak_kw":        [round(p, 1) for p in monthly_peak],
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
