"""
app.py
BESS Decision Tool — entry point web app.

Usage:
  streamlit run app.py

Two-step flow:
  Step 1: form — user describes site, consumption, PV, goals
  Step 2: results — recommended solution + business case + technical view

The existing engine (profile_builder, bess_engine, economics) is called
directly — no JSON files or CLI scripts needed.
"""

import json
import os
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

from engine import build_all_profiles, simulate, compute_economics

SLOT_H   = 0.25
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Italy bounding box for geocoding validation
_IT_LAT = (36.0, 47.5)
_IT_LON = (6.0,  18.5)

_SEASON_DAYS = {
    "Inverno (gennaio)":   21,
    "Primavera (aprile)": 105,
    "Estate (luglio)":    189,
    "Autunno (ottobre)":  280,
}
_DOW_LABELS = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _geocode(location: str) -> tuple[float, float] | None:
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": location, "format": "json", "limit": 1},
            headers={"User-Agent": "bess-simulator/0.1 (carlo.belluz@gmail.com)"},
            timeout=6,
        )
        results = resp.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception:
        pass
    return None


def _resolve_bess(bess: dict) -> dict:
    product_id = bess.get("product_id")
    if not product_id:
        return bess
    product_path = os.path.join(BASE_DIR, "products", f"{product_id}.json")
    if not os.path.exists(product_path):
        return bess
    with open(product_path) as f:
        product = json.load(f)
    defaults = product["simulation_defaults"]
    if "potenza_nominale_kw" in defaults:
        defaults.setdefault("potenza_carica_kw",  defaults["potenza_nominale_kw"])
        defaults.setdefault("potenza_scarica_kw", defaults["potenza_nominale_kw"])
    clean = {k: v for k, v in defaults.items()
             if not k.startswith("_") and not k.endswith("_fonte") and "override" not in k}
    return {**clean, **bess}


def _build_case(form: dict) -> dict:
    ha_pv      = form["ha_pv"] and form["kwp"] > 0
    lat        = form.get("lat") or 45.5
    lon        = form.get("lon") or 10.0
    has_lat    = form.get("lat") is not None
    pv_source  = "pvgis" if (ha_pv and has_lat) else "sintetico"
    cliente_id = form["nome_cliente"].lower().replace(" ", "-")[:20] or "cliente"

    case = {
        "meta": {
            "case_id":        f"app-{cliente_id}",
            "nome_cliente":   form["nome_cliente"] or "Cliente",
            "nome_sito":      form["location"] or "Sito",
            "data_creazione": "2026-05-06",
            "autore":         "app",
        },
        "site": {
            "consumo_annuo_kwh":       form["consumo_annuo_kwh"],
            "picco_potenza_kw":        form["picco_potenza_kw"],
            "lat":                     lat,
            "lon":                     lon,
            "ore_lavoro_giorno":       form["ore_lavoro_giorno"],
            "giorni_lavoro_settimana": form["giorni_lavoro_settimana"],
        },
        "pv": {
            "presente":               ha_pv,
            "kwp":                    form["kwp"] if ha_pv else 0,
            "profilo_source":         pv_source,
            "fv_export_regime":       "nessuno",
            "fv_export_value_eur_kwh": None,
        },
        "tariffs": {
            "market_price_series":      "prices/it_nord_2024.json",
            "supplier_spread_eur_kwh":  form["spread_eur_kwh"],
            "quota_potenza_eur_kw_mese": form["quota_potenza_eur_kw_mese"],
            "potenza_contrattuale_kw":  round(form["picco_potenza_kw"] * 1.1),
        },
        "bess": {
            "product_id":              "foxess-gmax-215",
            "costo_installato_eur_kwh": 255,
        },
        "simulation": {
            "scenari":               ["S1", "S2", "S3", "S4"],
            "anni_analisi":          20,
            "tasso_sconto":          0.05,
            "om_rate":               0.01,
            "degradazione_annua":    0.02,
            "soglia_critical_day_pct": 0.85,
        },
    }
    case["bess"] = _resolve_bess(case["bess"])

    # S_FV: proposed FV investment when client has no existing PV
    kwp_prop = form.get("kwp_proposto", 0)
    if not ha_pv and kwp_prop > 0:
        prop_source = "pvgis" if has_lat else "sintetico"
        case["pv_proposto"] = {
            "presente":               True,   # needed so build_pv_profile doesn't return zeros
            "kwp":                    kwp_prop,
            "profilo_source":         prop_source,
            "costo_eur_kwp":          form.get("costo_eur_kwp", 800),
            "anni_vita":              25,
            "degradazione_annua":     0.005,
        }

    return case


def _run_simulation(case: dict) -> tuple:
    profiles = build_all_profiles(case, BASE_DIR)
    sim      = simulate(case, profiles)
    econ     = compute_economics(case, profiles, sim)
    return profiles, sim, econ


def _pick_scenario(goal: str, ha_pv: bool) -> str:
    if goal == "Massimizzare l'autoconsumo FV" and ha_pv:
        return "S3"
    return "S4"


def _assumptions(form: dict, pv_source: str) -> dict:
    return {
        "carico": (
            f"Sintetico industriale — {form['consumo_annuo_kwh']:,.0f} kWh/anno, "
            f"{form['ore_lavoro_giorno']}h/gg, {form['giorni_lavoro_settimana']}gg/sett."
        ),
        "pv": (
            f"PVGIS 2020 — {form['location']}, {form['kwp']} kWp, inclinazione 30°, orientamento Sud"
            if pv_source == "pvgis" else
            "Sintetico — modello astronomico semplificato"
            if form["ha_pv"] else
            "Nessun impianto FV"
        ),
        "prezzi":     "ENTSO-E IT_NORD 2024 + spread fornitore",
        "confidenza": "Medio — dati di targa e bolletta, nessuna curva misurata",
    }


# ── Charts ────────────────────────────────────────────────────────────────────

def _chart_weekly(profiles: dict, sim: dict, scenario: str, day_start: int) -> None:
    s0 = day_start * 96
    s1 = s0 + 7 * 96
    t  = np.arange(7 * 96) * SLOT_H

    load   = np.array(profiles["load_kw"])[s0:s1]
    pv     = np.array(profiles["pv_kw"])[s0:s1]
    grid   = np.array(sim[scenario]["grid_kw"])[s0:s1]
    charge = np.array(sim[scenario]["bess_charge_kw"])[s0:s1]
    disc   = np.array(sim[scenario]["bess_discharge_kw"])[s0:s1]
    soc    = np.array(sim[scenario]["soc_kwh"])[s0:s1]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.06,
        subplot_titles=("Potenze (kW)", "SOC batteria (kWh)"),
    )
    fig.add_trace(go.Scatter(x=t, y=load,   name="Carico",       line=dict(color="#1565C0", width=2)),              row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=pv,     name="FV",           line=dict(color="#F9A825", width=2)),              row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=grid,   name="Rete",         line=dict(color="#757575", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=charge, name="Carica BESS",  line=dict(color="#2E7D32", width=1.5)),            row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=-disc,  name="Scarica BESS", line=dict(color="#C62828", width=1.5)),            row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=soc, name="SOC", fill="tozeroy",
        line=dict(color="#6A1B9A", width=1.5), fillcolor="rgba(106,27,154,0.12)",
    ), row=2, col=1)
    for d in range(1, 7):
        fig.add_vline(x=d * 24, line_dash="dash", line_color="rgba(0,0,0,0.12)")
    fig.update_xaxes(tickvals=[d * 24 + 12 for d in range(7)], ticktext=_DOW_LABELS, row=2, col=1)
    fig.update_yaxes(title_text="kW",  row=1, col=1)
    fig.update_yaxes(title_text="kWh", row=2, col=1)
    fig.update_layout(
        height=500, hovermode="x unified",
        margin=dict(t=40, b=10, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_cashflows(econ: dict, anni: int) -> None:
    years = list(range(anni + 1))
    fig   = go.Figure()
    for sc, color in [("S3", "#1565C0"), ("S4", "#2E7D32")]:
        if "cashflows" not in econ.get(sc, {}):
            continue
        cumcf = np.cumsum(econ[sc]["cashflows"]).tolist()
        fig.add_trace(go.Scatter(
            x=years, y=cumcf, name=sc, mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=4),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.25)")
    fig.update_layout(
        height=300, hovermode="x unified",
        margin=dict(t=10, b=30, l=60, r=10),
        xaxis_title="Anno", yaxis_title="€",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_layers(econ: dict) -> None:
    scenarios = [sc for sc in ["S3", "S4"] if "saving_fv_eur" in econ.get(sc, {})]
    if not scenarios:
        return
    fig = go.Figure()
    for label, key, color in [
        ("Layer 1 — Autoconsumo FV", "saving_fv_eur",       "#F9A825"),
        ("Layer 2 — Peak shaving",   "saving_quota_eur",    "#1565C0"),
        ("Layer 3 — Shifting",       "shifting_margin_eur", "#2E7D32"),
    ]:
        fig.add_trace(go.Bar(
            name=label, x=scenarios,
            y=[econ[sc][key] for sc in scenarios],
            marker_color=color,
        ))
    fig.update_layout(
        barmode="stack", height=280,
        margin=dict(t=10, b=30, l=60, r=10),
        yaxis_title="€/anno",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


def _data_quality_level(form: dict) -> tuple[int, str]:
    """Return (level 1-3, message string) based on available input data."""
    # Level 3: real hourly curve (not yet implemented in form)
    # Level 2: monthly consumption breakdown (not yet implemented in form)
    # Level 1: annual total only (current default)
    return (
        1,
        "**Risultato preliminare — qualità dati: bassa.** "
        "Il profilo di consumo è sintetico, ricavato dal dato annuale di bolletta. "
        "Accuratezza indicativa ±40%. Per un'analisi affidabile fornisci la curva di carico oraria.",
    )


def _chart_scenario_ladder(econ: dict, ha_pv: bool) -> None:
    """Waterfall chart — progressione del costo energetico annuo per scenario."""
    e1 = econ.get("S1", {})
    e2 = econ.get("S2", {})
    e3 = econ.get("S3", {})
    e4 = econ.get("S4", {})
    if not e1 or not e3:
        return

    s1_total = (e1.get("annual_energy_cost_eur", 0)
                + e1.get("annual_demand_charge_eur", 0))
    s2_total = (e2.get("annual_energy_cost_eur", 0)
                + e2.get("annual_demand_charge_eur", 0))
    s3_saving = e3.get("annual_saving_eur", 0)
    s4_saving = e4.get("annual_saving_eur", 0)
    s3_total  = s2_total - s3_saving
    s4_total  = s2_total - s4_saving

    if ha_pv:
        x       = ["Senza FV\n(baseline)", "Con FV", "FV + BESS\nautoconsumo", "FV + BESS\nmultilayer"]
        y       = [s1_total, s2_total - s1_total, s3_total - s2_total, s4_total - s3_total]
        measure = ["absolute", "relative", "relative", "relative"]
    else:
        x       = ["Situazione attuale", "Con BESS\nautoconsumo", "Con BESS\nmultilayer"]
        y       = [s1_total, s3_total - s1_total, s4_total - s3_total]
        measure = ["absolute", "relative", "relative"]

    text = []
    for m, v in zip(measure, y):
        if m == "absolute":
            text.append(f"{v:,.0f} €")
        else:
            text.append(f"−{abs(v):,.0f} €" if v < 0 else f"+{v:,.0f} €")

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measure,
        x=x, y=y,
        text=text, textposition="outside",
        connector=dict(line=dict(color="rgba(0,0,0,0.15)", width=1, dash="dot")),
        decreasing=dict(marker=dict(color="#2E7D32")),
        increasing=dict(marker=dict(color="#C62828")),
        totals=dict(marker=dict(color="#546E7A")),
    ))
    fig.update_layout(
        height=340,
        yaxis_title="€/anno (costo energetico totale)",
        margin=dict(t=30, b=20, l=70, r=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Advisory messages (Fix 3.3) ───────────────────────────────────────────────

def _advisory_message(econ: dict, form: dict, ha_pv: bool) -> tuple[str, str] | None:
    """
    Returns (message_text, streamlit_kind) for the most relevant advisory.
    kind: "info" | "warning" | "success" | "error"
    First matching condition wins — order is intentional.
    """
    e4    = econ.get("S4", {})
    e2    = econ.get("S2", {})
    e_sfv = econ.get("S_FV", {})

    npv           = e4.get("npv_eur", 0)
    payback       = e4.get("payback_yr")
    annual_saving = e4.get("annual_saving_eur", 0)
    saving_fv     = e4.get("saving_fv_eur", 0)
    saving_quota  = e4.get("saving_quota_eur", 0)
    investment    = e4.get("investment_eur", 0)
    scr_s4        = e4.get("scr_pct", 0)
    scr_s2        = e2.get("scr_pct", 0)

    # 1. No PV — recommend FV first
    if not ha_pv:
        sfv_npv = e_sfv.get("npv_eur", 0) if e_sfv else 0
        sfv_pb  = e_sfv.get("payback_yr") if e_sfv else None
        if sfv_npv > 0 and sfv_pb:
            return (
                f"**Prima di tutto, valuta un impianto FV.** "
                f"Senza produzione solare la batteria ha poco surplus da valorizzare. "
                f"Un FV da {e_sfv['kwp']} kWp ha NPV stimato **+{sfv_npv:,.0f} €** "
                f"con payback di {sfv_pb} anni. "
                f"Con FV + BESS insieme il rendimento complessivo migliorerà significativamente.",
                "info",
            )
        return (
            "**Attenzione: nessun impianto FV.** "
            "Senza produzione solare la batteria non può valorizzare l'autoconsumo. "
            "Il rendimento dipende quasi interamente dal peak shaving e dall'arbitraggio. "
            "Considera di valutare prima un impianto fotovoltaico.",
            "warning",
        )

    # 2. FV adds little — load mostly outside solar hours
    if ha_pv and scr_s2 < 38 and saving_fv < 1_500:
        return (
            f"**Gran parte dei consumi avviene fuori dalle ore di produzione solare** "
            f"(autoconsumo diretto FV: {scr_s2:.0f}%). "
            f"In questo profilo la batteria è quasi indispensabile per valorizzare il FV — "
            f"senza di essa gran parte dell'energia solare verrebbe ceduta a zero.",
            "info",
        )

    # 3. SCR already high — battery adds little SC value
    if ha_pv and scr_s2 > 75 and (scr_s4 - scr_s2) < 8:
        return (
            f"**L'autoconsumo FV senza batteria è già alto ({scr_s2:.0f}%).** "
            f"La batteria porta l'autoconsumo al {scr_s4:.0f}% (+{scr_s4 - scr_s2:.0f} punti). "
            f"Il vantaggio principale per questo sito è il **peak shaving**: "
            f"{saving_quota:,.0f} €/anno di riduzione quota potenza.",
            "info",
        )

    # 4. Peak shaving not effective
    if saving_quota < 200 and annual_saving > 0:
        return (
            f"**Il peak shaving ha effetto limitato su questo sito** ({saving_quota:,.0f} €/anno). "
            f"Il risparmio principale viene dall'autoconsumo FV ({saving_fv:,.0f} €/anno). "
            f"Possibili cause: picchi di breve durata oppure batteria già impegnata "
            f"nell'autoconsumo durante le ore di punta.",
            "warning",
        )

    # 5. NPV negative — identify main obstacle
    if npv < 0:
        if saving_quota < 500:
            obstacle = (
                f"il peak shaving genera solo {saving_quota:,.0f} €/anno "
                f"(quota potenza bassa o picchi difficili da tagliare)"
            )
        elif saving_fv < 1_500:
            obstacle = f"il surplus FV valorizzato è contenuto ({saving_fv:,.0f} €/anno)"
        else:
            obstacle = (
                f"il risparmio annuo ({annual_saving:,.0f} €/anno) "
                f"non è sufficiente a coprire l'investimento"
            )
        price_note = ""
        if saving_fv > 0 and investment > 0:
            target_saving = investment / 15
            price_factor  = max(1.0, (target_saving - saving_quota) / max(saving_fv, 1))
            pct_needed    = int((price_factor - 1) * 100)
            if pct_needed > 5:
                price_note = (
                    f" Se il prezzo dell'energia salisse del {pct_needed}%, "
                    f"il payback scenderebbe a circa 15 anni."
                )
        pb_str = f"{payback} anni" if payback else "non raggiunto in 20 anni"
        return (
            f"**Con i dati inseriti, la batteria non si ripaga in 20 anni** "
            f"(NPV: {npv:,.0f} €, payback: {pb_str}). "
            f"L'ostacolo principale: {obstacle}.{price_note}",
            "error",
        )

    # 6. Payback acceptable — positive message
    if payback and payback <= 12:
        layer_dom = "peak shaving" if saving_quota > saving_fv else "autoconsumo FV"
        return (
            f"**Progetto conveniente.** Payback di **{payback} anni**, "
            f"generato principalmente dal {layer_dom}. "
            f"NPV a 20 anni: **+{npv:,.0f} €**.",
            "success",
        )

    return None


# ── Sensitivity analysis (Fix 3.1) ────────────────────────────────────────────

def _display_sensitivity(econ: dict, best_sc: str, case: dict) -> None:
    """3-scenario sensitivity table: pessimistic / base / optimistic."""
    import numpy_financial as npf_s
    import pandas as pd

    e = econ.get(best_sc, {})
    if not e or e.get("annual_saving_eur", 0) <= 0:
        st.caption("Dati insufficienti per la sensitività.")
        return

    saving_fv    = e.get("saving_fv_eur", 0)
    saving_quota = e.get("saving_quota_eur", 0)
    investment   = e.get("investment_eur", 0)
    om_annual    = e.get("om_annual_eur", 0)
    anni         = case["simulation"]["anni_analisi"]
    disc         = case["simulation"]["tasso_sconto"]
    degr         = case["simulation"]["degradazione_annua"]
    anni_vita    = case["bess"].get("anni_vita", anni + 1)

    def _build_cf(s_adj, i_adj):
        cf   = np.empty(anni + 1)
        cf[0] = -i_adj
        repl  = i_adj if anni_vita <= anni else 0.0
        for y in range(1, anni + 1):
            exp   = y - 1 if (anni_vita > anni or y <= anni_vita) else y - anni_vita - 1
            cf[y] = s_adj * (1.0 - degr) ** exp - om_annual
        if anni_vita <= anni:
            cf[anni_vita] -= repl
        return cf

    def _pb(cf):
        cum = 0.0
        for y, v in enumerate(cf):
            cum += v
            if cum >= 0 and y > 0:
                return y
        return None

    rows = []
    for label, pf, cf_f in [
        ("Pessimistico (prezzo −20%, CAPEX +10%)", 0.80, 1.10),
        ("Base",                                   1.00, 1.00),
        ("Ottimistico (prezzo +20%, CAPEX −10%)",  1.20, 0.90),
    ]:
        s_adj = saving_fv * pf + saving_quota
        i_adj = investment * cf_f
        cf    = _build_cf(s_adj, i_adj)
        pb    = _pb(cf)
        npv_v = round(float(npf_s.npv(disc, cf)))
        rows.append({
            "Scenario":           label,
            "Risparmio/anno (€)": f"{s_adj:,.0f}",
            "Investimento (€)":   f"{i_adj:,.0f}",
            "Payback (anni)":     str(pb) if pb else "> 20",
            "NPV 20 anni (€)":    f"{npv_v:+,.0f}",
        })

    df = pd.DataFrame(rows).set_index("Scenario")
    st.dataframe(df, use_container_width=True)
    st.caption(
        "Il Layer 1 (autoconsumo FV) scala con il prezzo energia. "
        "Il Layer 2 (peak shaving) non scala con il prezzo — dipende dalla quota potenza. "
        "Layer 3 (arbitraggio) = 0 in MVP."
    )


# ── Step 1 — Input form ───────────────────────────────────────────────────────

def _page_input() -> None:
    st.markdown("# 🔋 BESS Decision Tool")
    st.markdown(
        "Descrivi la situazione energetica del tuo cliente. "
        "L'app ricostruisce lo scenario, simula l'installazione di una batteria "
        "e produce il business case."
    )
    st.divider()

    with st.form("input_form"):

        st.markdown("### 1 — Sito e consumi")
        col1, col2 = st.columns(2)
        with col1:
            nome_cliente = st.text_input("Nome cliente / azienda", placeholder="Es: Toninato S.r.l.")
            location     = st.text_input("Comune o indirizzo", placeholder="Es: Castelfranco Veneto, TV",
                                         help="Usato per stimare la produzione FV con PVGIS")
        with col2:
            consumo_annuo_kwh = st.number_input(
                "Consumo annuo (kWh)", min_value=1_000, max_value=10_000_000,
                value=350_000, step=5_000,
                help="Dalla bolletta: somma dei 12 mesi")
            picco_potenza_kw = st.number_input(
                "Picco / potenza contrattuale (kW)", min_value=10, max_value=5_000,
                value=150, step=5,
                help="Potenza massima del sito o potenza contrattuale")

        st.markdown("### 2 — Impianto fotovoltaico")
        col3, col4 = st.columns(2)
        with col3:
            ha_pv = st.checkbox("Il sito ha già un impianto FV", value=True)
        with col4:
            kwp = st.number_input(
                "Potenza FV esistente (kWp)", min_value=0, max_value=5_000,
                value=80, step=5,
                help="Potenza del FV già installato. Lascia 0 se assente")

        col3b, col4b = st.columns(2)
        with col3b:
            kwp_proposto = st.number_input(
                "FV da proporre (kWp) — se non ha FV", min_value=0, max_value=5_000,
                value=80, step=5,
                help="Usato solo se il sito non ha FV esistente. "
                     "Genera la valutazione dell'investimento FV (S_FV).")
        with col4b:
            costo_eur_kwp = st.number_input(
                "Costo FV chiavi in mano (€/kWp)", min_value=0, max_value=3_000,
                value=800, step=50,
                help="Range tipico C&I italiano 2025: 700–900 €/kWp (hardware + installazione)")

        st.markdown("### 3 — Orari operativi")
        col5, col6 = st.columns(2)
        with col5:
            ore_lavoro_giorno = st.number_input(
                "Ore lavoro / giorno", min_value=1, max_value=24, value=10,
                help="Turno di produzione tipico")
        with col6:
            giorni_lavoro_settimana = st.number_input(
                "Giorni lavoro / settimana", min_value=1, max_value=7, value=5)

        st.markdown("### 4 — Tariffe elettriche")
        col7, col8 = st.columns(2)
        with col7:
            spread_eur_kwh = st.number_input(
                "Spread fornitore (€/kWh)", min_value=0.0, max_value=0.5,
                value=0.095, step=0.005, format="%.3f",
                help="Componente commerciale aggiunta al prezzo mercato")
        with col8:
            quota_potenza = st.number_input(
                "Quota potenza (€/kW/mese)", min_value=0.0, max_value=50.0,
                value=12.0, step=0.5,
                help="Tutte le componenti di potenza aggregate (trasmissione + distribuzione)")

        st.markdown("### 5 — Obiettivo principale")
        goal = st.radio(
            "Cosa vuole ottenere il cliente?",
            options=[
                "Ridurre la bolletta elettrica",
                "Massimizzare l'autoconsumo FV",
                "Ridurre i picchi di potenza",
                "Valutare se conviene una batteria",
            ],
            index=0,
        )

        st.markdown("### 6 — Incentivi (opzionale)")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            t5_enable = st.checkbox(
                "Transizione 5.0 — credito d'imposta",
                value=False,
                help="Attiva per vedere il business case con il credito d'imposta (D.Lgs 19/2024). "
                     "Riduce il CAPEX netto nell'anno 0.",
            )
        with col_t2:
            t5_aliquota = st.number_input(
                "Aliquota credito d'imposta (%)",
                min_value=15, max_value=45, value=35, step=5,
                disabled=not t5_enable,
                help="Aliquota effettiva. Range tipico: 25–45% a seconda della fascia di risparmio energetico.",
            )

        st.divider()
        submitted = st.form_submit_button("Analizza →", type="primary", use_container_width=True)

    if not submitted:
        return

    if not nome_cliente.strip():
        st.warning("Inserisci il nome del cliente.")
        st.stop()

    # Geocode location → lat/lon (validated against Italy bounding box)
    lat, lon = None, None
    if location.strip():
        with st.spinner(f"Geocoding '{location}'..."):
            coords = _geocode(location.strip())
        if coords:
            lat_c, lon_c = coords
            if _IT_LAT[0] <= lat_c <= _IT_LAT[1] and _IT_LON[0] <= lon_c <= _IT_LON[1]:
                lat, lon = lat_c, lon_c
            else:
                st.warning(
                    f"Le coordinate trovate per '{location}' ({lat_c:.2f}°N, {lon_c:.2f}°E) "
                    "sono fuori dall'Italia. Verrà usato un profilo FV sintetico al posto di PVGIS."
                )
        else:
            st.warning(
                f"Posizione '{location}' non trovata. "
                "Uso coordinate default Nord Italia. "
                "PVGIS sarà meno preciso — prova con un nome più specifico."
            )

    form = {
        "nome_cliente":             nome_cliente.strip(),
        "location":                 location.strip() or "Sito",
        "consumo_annuo_kwh":        consumo_annuo_kwh,
        "picco_potenza_kw":         picco_potenza_kw,
        "ha_pv":                    ha_pv,
        "kwp":                      kwp if ha_pv else 0,
        "kwp_proposto":             kwp_proposto if not ha_pv else 0,
        "costo_eur_kwp":            costo_eur_kwp,
        "ore_lavoro_giorno":        ore_lavoro_giorno,
        "giorni_lavoro_settimana":  giorni_lavoro_settimana,
        "spread_eur_kwh":           spread_eur_kwh,
        "quota_potenza_eur_kw_mese": quota_potenza,
        "goal":                     goal,
        "lat":                      lat,
        "lon":                      lon,
        "t5_enable":                t5_enable,
        "t5_aliquota_pct":          t5_aliquota if t5_enable else 0,
    }

    case = _build_case(form)

    with st.spinner("Simulazione in corso..."):
        t0 = time.perf_counter()
        profiles, sim, econ = _run_simulation(case)
        elapsed = time.perf_counter() - t0

    st.session_state.update({
        "step":     "results",
        "form":     form,
        "case":     case,
        "profiles": profiles,
        "sim":      sim,
        "econ":     econ,
        "elapsed":  elapsed,
    })
    st.rerun()


# ── Step 2 — Results ──────────────────────────────────────────────────────────

def _page_results() -> None:
    form     = st.session_state["form"]
    case     = st.session_state["case"]
    profiles = st.session_state["profiles"]
    sim      = st.session_state["sim"]
    econ     = st.session_state["econ"]
    elapsed  = st.session_state.get("elapsed", 0)

    ha_pv     = form["ha_pv"] and form["kwp"] > 0
    pv_source = case["pv"]["profilo_source"]
    best_sc   = _pick_scenario(form["goal"], ha_pv)
    e_best    = econ.get(best_sc, {})
    assum     = _assumptions(form, pv_source)

    # ── Header + back button ──────────────────────────────────────────────────
    col_h, col_b = st.columns([6, 1])
    with col_h:
        st.markdown(f"# Soluzione per {form['nome_cliente']}")
    with col_b:
        st.markdown("<div style='padding-top:1.1rem'></div>", unsafe_allow_html=True)
        if st.button("← Modifica", use_container_width=True):
            st.session_state["step"] = "input"
            st.rerun()

    # ── Data quality banner (Fix 2.3) ─────────────────────────────────────────
    dq_level, dq_msg = _data_quality_level(form)
    st.warning(dq_msg, icon="⚠️")

    # ── Assumptions box ───────────────────────────────────────────────────────
    with st.expander("📋 Come abbiamo ricostruito lo scenario", expanded=False):
        st.markdown(f"- **Profilo di carico:** {assum['carico']}")
        st.markdown(f"- **Produzione FV:** {assum['pv']}")
        st.markdown(f"- **Prezzi energia:** {assum['prezzi']}")
        st.markdown(f"- **Livello di confidenza:** {assum['confidenza']}")
        st.caption(f"Simulazione completata in {elapsed:.1f}s · Scenario primario: {best_sc}")

    st.divider()

    # ── Product + primary KPIs ────────────────────────────────────────────────
    bess     = case["bess"]
    cap_kwh  = bess.get("capacita_nominale_kwh", 215)
    pow_kw   = bess.get("potenza_nominale_kw",   100)
    cost_kwh = bess.get("costo_installato_eur_kwh", 255)
    invest   = cap_kwh * cost_kwh
    sc_label = "multilayer (S4)" if best_sc == "S4" else "autoconsumo (S3)"

    st.markdown(f"### 🔋 Fox ESS G-MAX &nbsp; {cap_kwh} kWh / {pow_kw} kW &nbsp;·&nbsp; Strategia: {sc_label}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Investimento",    f"{invest:,.0f} €")
    c2.metric("Risparmio annuo", f"{e_best.get('annual_saving_eur', 0):,.0f} €/anno")
    pb = e_best.get("payback_yr")
    c3.metric("Payback",         f"{pb} anni" if pb else "— anni")
    c4.metric("NPV (20 anni)",   f"{e_best.get('npv_eur', 0):,.0f} €")
    irr = e_best.get("irr_pct")
    c5.metric("IRR",             f"{irr} %" if irr is not None else "—")

    # ── Advisory message (Fix 3.3) ────────────────────────────────────────────
    advisory = _advisory_message(econ, form, ha_pv)
    if advisory:
        msg_text, msg_kind = advisory
        {"info": st.info, "warning": st.warning,
         "success": st.success, "error": st.error}[msg_kind](msg_text)

    # ── S_FV: proposta impianto FV (Fix 2.1) ─────────────────────────────────
    e_sfv = econ.get("S_FV")
    if not ha_pv and e_sfv:
        st.divider()
        kwp_prop = e_sfv["kwp"]
        st.markdown(f"### Prima di tutto: valutazione impianto FV da {kwp_prop} kWp")
        st.info(
            "Il sito non ha un impianto FV esistente. "
            "Senza FV, la batteria ha poco surplus da valorizzare e il rendimento è molto basso. "
            "**Ti mostriamo prima la convenienza di un impianto FV da zero.**"
        )
        fv1, fv2, fv3, fv4, fv5 = st.columns(5)
        fv1.metric("Investimento FV",    f"{e_sfv['investment_eur']:,.0f} €",
                   f"{e_sfv['costo_eur_kwp']} €/kWp")
        fv2.metric("Risparmio anno 1",   f"{e_sfv['saving_y1_eur']:,.0f} €/anno")
        pb_fv = e_sfv.get("payback_yr")
        fv3.metric("Payback FV",         f"{pb_fv} anni" if pb_fv else "— anni")
        fv4.metric("NPV FV (25 anni)",   f"{e_sfv['npv_eur']:,.0f} €")
        irr_fv = e_sfv.get("irr_pct")
        fv5.metric("IRR FV",             f"{irr_fv} %" if irr_fv is not None else "—")
        st.caption(
            f"Producibilità stimata: {e_sfv['pv_total_kwh']:,.0f} kWh/anno · "
            f"Autoconsumo diretto: {e_sfv['direct_selfcons_kwh']:,.0f} kWh/anno "
            f"(SCR {e_sfv['scr_pct']}%, SSR {e_sfv['ssr_pct']}%) · "
            "Nessun valore di cessione in rete (regime 'nessuno'). "
            "Degradazione FV 0.5%/anno. Tasso sconto 5%."
        )

    st.divider()

    # ── Scenario ladder — progressione del costo energetico (Fix 2.2) ─────────
    st.markdown("#### Come si forma il risparmio")
    _chart_scenario_ladder(econ, ha_pv)

    e1  = econ.get("S1", {})
    e2  = econ.get("S2", {})
    e3b = econ.get("S3", {})
    e4b = econ.get("S4", {})
    s2_total = (e2.get("annual_energy_cost_eur", 0)
                + e2.get("annual_demand_charge_eur", 0))
    if ha_pv and e2:
        s1_total = (e1.get("annual_energy_cost_eur", 0)
                    + e1.get("annual_demand_charge_eur", 0))
        fv_saving = s1_total - s2_total
        st.caption(
            f"Il FV da {form.get('kwp', 0)} kWp risparmia **{fv_saving:,.0f} €/anno** sul costo totale. "
            f"La batteria (autoconsumo) aggiunge **{e3b.get('annual_saving_eur', 0):,.0f} €/anno**. "
            f"Il peak shaving porta il risparmio aggiuntivo a "
            f"**{e4b.get('annual_saving_eur', 0) - e3b.get('annual_saving_eur', 0):,.0f} €/anno**. "
            "Le barre verdi indicano risparmio; il grafico mostra il costo annuo totale (energia + quota potenza)."
        )
    else:
        st.caption(
            f"La batteria (autoconsumo) risparmia **{e3b.get('annual_saving_eur', 0):,.0f} €/anno**. "
            f"Il peak shaving aggiunge **{e4b.get('annual_saving_eur', 0) - e3b.get('annual_saving_eur', 0):,.0f} €/anno**. "
            "Le barre verdi indicano risparmio; il grafico mostra il costo annuo totale (energia + quota potenza)."
        )

    st.divider()

    # ── Energy profile summary ────────────────────────────────────────────────
    load_kw = np.array(profiles["load_kw"])
    pv_kw   = np.array(profiles["pv_kw"])
    price   = np.array(profiles["price_eur_kwh"])

    st.markdown("#### Profili annuali ricostruiti")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Consumo annuo",    f"{load_kw.sum() * SLOT_H:,.0f} kWh",
              f"picco {load_kw.max():.1f} kW")
    if ha_pv:
        p2.metric("Produzione FV", f"{pv_kw.sum() * SLOT_H:,.0f} kWh",
                  f"picco {pv_kw.max():.1f} kW")
    else:
        p2.metric("Produzione FV", "—", "Nessun impianto FV")
    e_s2 = econ.get("S2", {})
    e_s1 = econ.get("S1", {})
    s2_cost = e_s2.get("annual_energy_cost_eur", 0)
    s1_cost = e_s1.get("annual_energy_cost_eur", 0)
    p3.metric("Costo energia attuale (con FV)",
              f"{s2_cost:,.0f} €/anno" if ha_pv else f"{s1_cost:,.0f} €/anno",
              f"−{s1_cost - s2_cost:,.0f} € grazie al FV" if ha_pv else "senza FV")
    p4.metric("Prezzo medio energia",
              f"{price.mean():.4f} €/kWh",
              f"range {price.min():.3f}–{price.max():.3f}")

    st.divider()

    # ── Business case detail — tabs ────────────────────────────────────────────
    st.markdown("#### Business case di dettaglio")
    tab_labels = ["Risparmio per layer", "Flussi di cassa", "Vista tecnica", "Sensitività & Incentivi"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_labels)

    with tab1:
        _chart_layers(econ)
        st.caption(
            "Layer 1: risparmio FV via batteria (energia FV autoconsumata anziché esportata). "
            "Layer 2: riduzione quota potenza (peak shaving). "
            "Layer 3: arbitraggio prezzi orari (shifting). "
            "I layer derivano da un'unica simulazione — non sono indipendenti."
        )
        e3 = econ.get("S3", {})
        e4 = econ.get("S4", {})
        if e3 and e4:
            st.markdown("**S3 vs S4 — confronto scenari**")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("S3 risparmio", f"{e3.get('annual_saving_eur', 0):,.0f} €/anno")
            d2.metric("S4 risparmio", f"{e4.get('annual_saving_eur', 0):,.0f} €/anno")
            d3.metric("S3 NPV",       f"{e3.get('npv_eur', 0):,.0f} €")
            d4.metric("S4 NPV",       f"{e4.get('npv_eur', 0):,.0f} €")
            cap_l1 = e4.get("cap_l1_ottimale_kwh", 0)
            cap_sim = case["bess"].get("capacita_nominale_kwh", 215)
            sizing_note = ""
            if cap_l1 > 0 and cap_sim > cap_l1 * 1.30:
                sizing_note = (
                    f" | ⚠️ Taglia ottimale per autoconsumo stimata: ~{cap_l1} kWh — "
                    f"la G-MAX {cap_sim} kWh è {round(cap_sim/cap_l1*100-100)}% più grande."
                )
            st.caption(
                "S3 = solo autoconsumo FV.  "
                "S4 = multilayer (autoconsumo + peak shaving + shifting).  "
                f"Scenario primario mostrato sopra: **{best_sc}** "
                f"(scelto in base all'obiettivo: «{form['goal']}»)."
                + sizing_note
            )

    with tab2:
        _chart_cashflows(econ, case["simulation"]["anni_analisi"])
        repl_yr = e_best.get("battery_replacement_year")
        if repl_yr:
            repl_cost = e_best.get("replacement_capex_eur", 0)
            st.caption(f"Sostituzione batteria inclusa all'anno {repl_yr} ({repl_cost:,.0f} €).")
        else:
            st.caption("Orizzonte di analisi: 20 anni. Tasso di sconto: 5%.")

    with tab3:
        season = st.selectbox("Stagione", list(_SEASON_DAYS.keys()), key="season_sel")
        _chart_weekly(profiles, sim, best_sc, _SEASON_DAYS[season])
        st.caption(
            f"Vista tecnica settimanale — Scenario {best_sc}.  "
            "Carico, FV, potenza batteria e stato di carica (SOC) slot per slot."
        )

    with tab4:
        # ── Sensitivity analysis (Fix 3.1) ────────────────────────────────────
        st.markdown("##### Sensitività prezzi e CAPEX")
        _display_sensitivity(econ, best_sc, case)

        # ── Transizione 5.0 (Fix 3.4) ─────────────────────────────────────────
        t5_on      = form.get("t5_enable", False)
        t5_aliq    = form.get("t5_aliquota_pct", 0)
        if t5_on and t5_aliq > 0:
            st.markdown("---")
            st.markdown(f"##### Con Transizione 5.0 — credito d'imposta {t5_aliq}%")
            st.warning(
                "⚠️ Questo calcolo assume la qualificazione per il credito d'imposta Transizione 5.0 "
                "(D.Lgs 19/2024). **Verifica obbligatoria con un professionista abilitato.** "
                "L'aliquota effettiva dipende dalla fascia di risparmio energetico certificato.",
                icon="⚠️",
            )
            net_invest  = invest * (1.0 - t5_aliq / 100.0)
            base_cf     = list(e_best.get("cashflows", []))
            if base_cf:
                t5_cf    = base_cf.copy()
                t5_cf[0] = -net_invest
                # T5.0 applied to initial CAPEX only — replacement at year 15 stays at cost price

                import numpy_financial as _npf_t5
                t5_npv = round(float(_npf_t5.npv(case["simulation"]["tasso_sconto"], t5_cf)))
                # Simple payback on reduced CAPEX
                t5_pb = None
                cum = 0.0
                for yi, v in enumerate(t5_cf):
                    cum += v
                    if cum >= 0 and yi > 0:
                        t5_pb = yi
                        break
                try:
                    t5_irr_raw = _npf_t5.irr(t5_cf)
                    t5_irr = round(float(t5_irr_raw) * 100, 2) if float(t5_irr_raw) == t5_irr_raw else None
                except Exception:
                    t5_irr = None

                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric("Investimento netto",  f"{net_invest:,.0f} €",
                           f"−{invest - net_invest:,.0f} € di credito")
                tc2.metric("Risparmio annuo",      f"{e_best.get('annual_saving_eur', 0):,.0f} €/anno")
                tc3.metric("Payback (con T5.0)",   f"{t5_pb} anni" if t5_pb else "— anni")
                tc4.metric("NPV 20 anni (T5.0)",   f"{t5_npv:+,.0f} €")
                st.caption(
                    f"Il credito d'imposta riduce il CAPEX dell'anno 0 da {invest:,.0f} € "
                    f"a {net_invest:,.0f} €. I risparmi annui rimangono invariati."
                )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="BESS Decision Tool",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    if "step" not in st.session_state:
        st.session_state["step"] = "input"

    if st.session_state["step"] == "input":
        _page_input()
    else:
        _page_results()


if __name__ == "__main__":
    main()
