# BESS Tool MVP — Project Foundation

## Objective

Build a local, usable simulation and business case tool for industrial BESS (Battery Energy Storage System) installations.

The tool answers one question:
> "If I install a BESS here, how can I use it, what value does it generate, and what size makes sense?"

---

## Target users

| User | Role | What they need |
|---|---|---|
| Carlo (consultant) | Runs the tool with the client | Fast, credible numbers in a meeting |
| Industrial client | Sees the output | Clear benefit and return on investment |
| Financiers | Reads the report | IRR, NPV, cash flow to evaluate financing |

---

## MVP scope — what is included

- One real commercial BESS reference product (G-MAX, ~255 €/kWh installed)
- Standardized case input file (JSON)
- Full simulation engine (S1 → S4), 15-minute slots, 35,040 per year
- Seasonal technical visualization (weekly representative profile per season)
- Separate KPI layers: self-consumption FV, peak shaving, shifting
- Business case output: payback, NPV, IRR

## MVP scope — what is NOT included

- Generic import from arbitrary customer files (PDF bills, CSV load curves)
- Full product catalog
- Full incentive / subsidy engine
- SaaS or multi-user architecture
- Bank-grade due diligence workflow
- Automatic sizing optimizer (sweep over sizes)
- Real-time market prices (MGP/GME)

---

## Scenarios

| ID | Description |
|---|---|
| S1 | Baseline — no PV, no BESS |
| S2 | Current state — PV active, no BESS |
| S3 | BESS for self-consumption only — charges from PV surplus, discharges on load |
| S4 | BESS multilayer — self-consumption + peak shaving + time shifting |

---

## Required outputs

### Technical
- Seasonal weekly profile charts: load, PV, BESS charge, BESS discharge, grid draw, SOC
- Annual technical KPIs: energy self-consumed, peak reduction, battery throughput, equivalent cycles

### Economic (layers kept separate)
- Saving from FV via BESS (self-consumption gain)
- Saving from peak shaving (power charge reduction)
- Shifting margin (night charge cost vs daytime discharge value)
- Total annual saving, payback, NPV, IRR

---

## Modelling principles

The BESS is not an abstract box. The model always uses:

| Parameter | Notes |
|---|---|
| Nominal capacity (kWh) | Commercial product size |
| Usable capacity | SOC min 5% — SOC max 95% |
| Charge/discharge power (kW) | C-rate limit |
| Round-trip efficiency | 90% (√0.90 applied both ways) |
| Equivalent annual cycles | Tracked from throughput |
| Operational limits | Hard constraints per slot |

### Peak shaving logic
Selective: only on "critical days" (daily peak ≥ 85% of monthly peak).
Clipping threshold calculated dynamically via bisection to use exactly the available energy.

### FV quota tracking
SOC tracks separately how much stored energy comes from PV vs grid.
Required to avoid double-counting across economic layers.

### Grid charging (shifting)
Only in winter months (Jan, Feb, Nov, Dec), only at night (00:00–07:00).
Power distributed evenly across the window to avoid creating artificial peaks.

---

## Validation case: Toninato

This is the reference case used to verify the engine is correct.

| Output | Target value |
|---|---|
| S3 annual saving | ~6,225 €/anno |
| S4 annual saving | ~7,010 €/anno |
| Investment (approx) | ~55,000 € |
| Simple payback | ~7.8 anni |

If the engine produces these numbers on the Toninato case, it is validated.

**Open item:** exact input parameters for Toninato (kWh/anno, kW peak, kWp PV, €/kWh, €/kW/month) — to be confirmed by Carlo before running validation.

---

## File architecture

```
bess 0.1/
├── CLAUDE.md                  ← operating brief for Claude in this project
├── PROJECT_FOUNDATION.md      ← this file
├── case_toninato.json         ← standardized case input (Toninato reference)
├── bess_engine.py             ← core simulation engine (physics + economics)
├── profile_builder.py         ← synthetic load profile: kWh/anno → 35,040 slots
├── pvgis.py                   ← PVGIS API client (PV production hourly)
├── simulatore.py              ← Streamlit UI (orchestrates all modules)
└── requirements.txt           ← Python dependencies
```

### Module responsibilities

| File | Does |
|---|---|
| `case_toninato.json` | All case parameters in one place — site, BESS, tariff, PVGIS |
| `bess_engine.py` | BESSConfig, SiteConfig, simulate_s3(), simulate_s4(), compute_kpi(), business_plan() |
| `profile_builder.py` | Generates synthetic industrial load profile from annual kWh + peak kW |
| `pvgis.py` | Calls PVGIS API, returns hourly PV production array for full year |
| `simulatore.py` | Streamlit pages: input → run → seasonal charts → KPIs → business case |

---

## Prior materials inventory

| Material | Decision |
|---|---|
| Obsidian BESS-04 (engine blueprint) | **Import** — full logic blueprint for bess_engine.py |
| Obsidian BESS-02 (validation numbers) | **Import** — S3/S4 targets, payback, CAPEX |
| Obsidian BESS-06 (roadmap) | **Reference** — deadlines and phase scope |
| `bess-dashboard/simulatore.py` | **Import UI structure only** — form layout, Plotly pattern |
| `bess-dashboard/requirements.txt` | **Import** — copy as starting point |
| `dashboard.html` | **Discard** — aesthetic reference only |
| `offerta.html` | **Discard** — aesthetic reference only |
| `hello.py` | **Discard** — first Python test, not relevant |

---

*Last updated: 2026-05-05 — foundation document, pre-implementation*
