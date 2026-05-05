"""
run_simulation.py
Entry point — chains profile builder, BESS engine, and economics.

Usage:
  python run_simulation.py
  python run_simulation.py cases/example_case.json

Output:
  results/<case_id>_results.json   — full arrays + KPIs, ready for dashboard
"""

import json
import os
import sys
import time

import numpy as np

from engine import build_all_profiles, simulate, compute_economics


def main() -> None:
    case_path = sys.argv[1] if len(sys.argv) > 1 else "cases/example_case.json"
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, case_path)

    if not os.path.exists(full_path):
        print(f"ERROR: case file not found: {full_path}")
        sys.exit(1)

    # ── Load case ─────────────────────────────────────────────────────────────
    with open(full_path) as f:
        case = json.load(f)
    case_id = case["meta"]["case_id"]

    print(f"Case      : {case_id}  ({case['meta']['nome_cliente']})")
    print(f"Site      : {case['meta']['nome_sito']}")
    print()

    # ── Step 1: Build profiles ────────────────────────────────────────────────
    t = time.perf_counter()
    print("1/3  Building profiles...", end=" ", flush=True)
    profiles = build_all_profiles(case, base_dir)
    print(f"done  ({time.perf_counter() - t:.2f}s)")

    # ── Step 2: Simulate ──────────────────────────────────────────────────────
    t = time.perf_counter()
    print("2/3  Running simulation...", end=" ", flush=True)
    sim_results = simulate(case, profiles)
    print(f"done  ({time.perf_counter() - t:.2f}s)")

    # ── Step 3: Economics ─────────────────────────────────────────────────────
    t = time.perf_counter()
    print("3/3  Computing economics...", end=" ", flush=True)
    econ = compute_economics(case, profiles, sim_results)
    print(f"done  ({time.perf_counter() - t:.2f}s)")

    # ── Assemble output ───────────────────────────────────────────────────────
    # Profiles: float arrays rounded to 4 dp; integer time-index kept as-is.
    _INT_KEYS = {"month", "hour", "dow", "slot_in_day"}
    profiles_out = {
        k: (v.tolist() if k in _INT_KEYS
            else np.round(v, 4).tolist() if isinstance(v, np.ndarray)
            else v)
        for k, v in profiles.items()
    }

    # Simulation arrays: round floats to 4 dp.
    simulation_out = {
        sc: {k: np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v
             for k, v in r.items()}
        for sc, r in sim_results.items()
    }

    output = {
        "meta":       case["meta"],
        "case":       case,
        "profiles":   profiles_out,
        "simulation": simulation_out,
        "economics":  econ,
    }

    # ── Write results.json ────────────────────────────────────────────────────
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{case_id}_results.json")

    with open(out_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nSaved     : {out_path}  ({size_mb:.1f} MB)")

    # ── Print summary ─────────────────────────────────────────────────────────
    _print_summary(econ, profiles)


def _print_summary(econ: dict, profiles: dict) -> None:
    print()
    print("── Profiles ────────────────────────────────────────────────────")
    load_kwh = profiles["load_kw"].sum() * 0.25
    pv_kwh   = profiles["pv_kw"].sum() * 0.25
    print(f"Load      : {load_kwh:>10,.0f} kWh/yr   peak {profiles['load_kw'].max():.1f} kW")
    print(f"PV        : {pv_kwh:>10,.0f} kWh/yr   peak {profiles['pv_kw'].max():.1f} kW")
    print(f"Price     :  mean {profiles['price_eur_kwh'].mean():.4f} €/kWh   "
          f"min {profiles['price_eur_kwh'].min():.4f}   max {profiles['price_eur_kwh'].max():.4f}")

    print()
    print("── Grid draw ───────────────────────────────────────────────────")
    for sc in ["S1", "S2", "S3", "S4"]:
        if sc not in econ:
            continue
        e = econ[sc]
        if "annual_grid_import_kwh" in e:
            exp = e.get("annual_grid_export_kwh", 0)
            print(f"{sc}  import {e['annual_grid_import_kwh']:>9,.0f} kWh/yr   "
                  f"export {exp:>8,.0f} kWh/yr   "
                  f"energy cost {e['annual_energy_cost_eur']:>9,.0f} €/yr")

    print()
    print("── BESS economics ──────────────────────────────────────────────")
    for sc in ["S3", "S4"]:
        if sc not in econ:
            continue
        e = econ[sc]
        print(f"{sc}  Layer1(FV) {e['saving_fv_eur']:>7,.0f} €   "
              f"Layer2(peak) {e['saving_quota_eur']:>6,.0f} €   "
              f"Layer3(shift) {e['shifting_margin_eur']:>7,.0f} €")
        pb = f"{e['payback_yr']} yr" if e["payback_yr"] else "not reached"
        print(f"   Total {e['annual_saving_eur']:>7,.0f} €/yr   "
              f"cycles {e['equivalent_cycles']:.0f}/yr   "
              f"payback {pb}   "
              f"NPV {e['npv_eur']:>9,.0f} €   "
              f"IRR {e['irr_pct']}%")


if __name__ == "__main__":
    main()
