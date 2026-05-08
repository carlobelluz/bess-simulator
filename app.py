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

# Block 1 — Site Intake (parallel path, does not affect existing simulation flow)
try:
    from intake.case_builder import build_site_diagnostic as _build_sdi_block1
    _BLOCK1_AVAILABLE = True
except ImportError:
    _BLOCK1_AVAILABLE = False

# Block 1 — Diagnostic output layer
try:
    from intake.diagnostic import build_diagnostic as _build_diagnostic
    _DIAGNOSTIC_AVAILABLE = True
except ImportError:
    _DIAGNOSTIC_AVAILABLE = False

# Block 1 — Bill PDF parser (optional, requires GOOGLE_API_KEY)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
    from intake.bill_parser import parse_bill_pdf as _parse_bill_pdf, is_available as _bill_parser_available
    from intake.bill_extractor import normalize_bill as _normalize_bill, expand_parsed_bill as _expand_parsed_bill
    _PDF_PARSER_IMPORTED = True
except ImportError:
    _PDF_PARSER_IMPORTED = False
    def _bill_parser_available(): return False

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
_DOW_LABELS    = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
_PV_DEG_DEFAULT = 0.40   # %/anno — benchmark moduli cristallini moderni (IEA/NREL)


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


def _render_sdi_diagnostic(sdi: dict) -> None:
    def _v(field, default="N/D"):
        if isinstance(field, dict):
            return field.get("value", default)
        return field if field is not None else default

    def _fmt_kwh(v):
        return f"{v:,.0f} kWh" if isinstance(v, (int, float)) else "N/D"

    meta     = sdi.get("meta") or {}
    site_s   = sdi.get("site") or {}
    billing  = sdi.get("billing") or {}
    pv_ex    = sdi.get("pv_existing") or {}
    profiles = sdi.get("profiles") or {}
    warnings = sdi.get("warnings") or []

    macro_case = meta.get("macro_case", "N/D")
    dq_level   = meta.get("data_quality_level", "N/D")
    has_pv     = bool(pv_ex.get("presente", False))
    kwp_val    = _v(pv_ex.get("kwp")) if has_pv else None

    consumo_sito_f  = site_s.get("consumo_annuo_kwh") or {}
    consumo_billing = billing.get("consumo_annuo_derivato_kwh") or {}
    load_meta = profiles.get("load_kw") or {}
    pv_meta   = profiles.get("pv_kw") or {}

    # PV annual production from profiles cache
    pv_prod_kwh = None
    cache_file  = profiles.get("cache_file")
    if has_pv and cache_file:
        cache_abs = os.path.join(BASE_DIR, cache_file)
        if os.path.exists(cache_abs):
            try:
                with open(cache_abs) as _f:
                    _cached = json.load(_f)
                pv_arr = _cached.get("pv_kw") or []
                if pv_arr:
                    pv_prod_kwh = round(sum(pv_arr) * 0.25)
            except Exception:
                pass

    st.markdown("**Diagnostico Block 1**")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"**Macro-case:** `{macro_case}`  \n**Data quality:** L{dq_level}")
        pv_str = (f"Sì — {kwp_val} kWp" if kwp_val else "Sì") if has_pv else "No"
        st.markdown(f"**FV esistente:** {pv_str}")
        if pv_prod_kwh is not None:
            st.markdown(f"**Prod. FV stimata:** {pv_prod_kwh:,.0f} kWh/anno")

    with c2:
        st.markdown(f"**Consumo sito:**  \n{_fmt_kwh(_v(consumo_sito_f))}")
        st.caption(
            f"source: {consumo_sito_f.get('source', '—')} · "
            f"conf: {consumo_sito_f.get('confidence', '—')}"
        )
        st.markdown(f"**Prelievo netto (billing):**  \n{_fmt_kwh(_v(consumo_billing))}")
        st.caption(
            f"source: {consumo_billing.get('source', '—')} · "
            f"conf: {consumo_billing.get('confidence', '—')}"
        )
        if has_pv:
            _ac_kwh = load_meta.get("autoconsumo_fv_annuo_kwh")
            _ac_src = load_meta.get("autoconsumo_source", "estimated")
            _ac_label = "(dichiarato dall'utente)" if _ac_src == "user_input" else "(stimato)"
            if _ac_kwh is not None:
                st.markdown(
                    f"**Autoconsumo FV:**  \n{_ac_kwh:,.0f} kWh {_ac_label}"
                )

    with c3:
        st.markdown(f"**Profilo carico:**  \n`{load_meta.get('source', 'N/D')}`")
        st.caption(f"conf: {load_meta.get('confidence', '—')}")
        st.markdown(f"**Profilo FV:**  \n`{pv_meta.get('source', 'N/D')}`")
        n_w = len(warnings)
        st.markdown(f"**Warnings:** {n_w}")

    if warnings:
        for w in warnings:
            st.caption(f"⚠️ {w}")

    st.divider()


def _form_to_intake(form: dict) -> dict:
    picco    = form.get("picco_potenza_kw") or 0
    pot_contr = form.get("potenza_contrattuale_kw")

    # Costruisce raw_bill dai campi bolletta se il consumo periodo è disponibile
    raw_bill = None
    if form.get("consumo_kwh_periodo"):
        raw_bill = {
            "consumo_kwh":             form.get("consumo_kwh_periodo"),
            "potenza_contrattuale_kw": pot_contr,
            "costo_totale_eur":        form.get("costo_totale_eur"),
            "costo_energia_eur":       form.get("costo_energia_eur"),
            "costo_potenza_eur":       form.get("costo_potenza_eur"),
            "f1_kwh":                  form.get("f1_kwh"),
            "f2_kwh":                  form.get("f2_kwh"),
            "f3_kwh":                  form.get("f3_kwh"),
            "source":                  "form_entry",
        }

    return {
        "nome_cliente":              form.get("nome_cliente", ""),
        "comune":                    form.get("location", ""),
        "consumo_annuo_kwh":         form.get("consumo_annuo_kwh", 0),
        "picco_potenza_kw":          picco or None,
        "potenza_contrattuale_kw":   pot_contr or (round(picco * 1.1) if picco else None),
        "ha_pv":                     form.get("ha_pv", False),
        "kwp":                       form.get("kwp", 0),
        "ore_lavoro_giorno":         form.get("ore_lavoro_giorno"),
        "giorni_lavoro_settimana":   form.get("giorni_lavoro_settimana"),
        "spread_eur_kwh":            form.get("spread_eur_kwh"),
        "quota_potenza_eur_kw_mese": form.get("quota_potenza_eur_kw_mese"),
        "lat":                       form.get("lat"),
        "lon":                       form.get("lon"),
        "bills":                     [raw_bill] if raw_bill else [],
        "market_price_series":       "prices/it_nord_2024.json",
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
    price  = np.array(profiles["price_eur_kwh"])[s0:s1]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.06,
        subplot_titles=("Potenze (kW) — asse dx: prezzo energia (€/kWh)", "SOC batteria (kWh)"),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )
    fig.add_trace(go.Scatter(x=t, y=load,   name="Carico",       line=dict(color="#1565C0", width=2)),               row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=t, y=pv,     name="FV",           line=dict(color="#F9A825", width=2)),               row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=t, y=grid,   name="Rete",         line=dict(color="#757575", width=1.5, dash="dot")), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=t, y=charge, name="Carica BESS",  line=dict(color="#2E7D32", width=1.5)),             row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=t, y=-disc,  name="Scarica BESS", line=dict(color="#C62828", width=1.5)),             row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=t, y=price, name="Prezzo energia",
        line=dict(color="#FF6F00", width=1, dash="dot"), opacity=0.8,
    ), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(
        x=t, y=soc, name="SOC", fill="tozeroy",
        line=dict(color="#6A1B9A", width=1.5), fillcolor="rgba(106,27,154,0.12)",
    ), row=2, col=1)
    for d in range(1, 7):
        fig.add_vline(x=d * 24, line_dash="dash", line_color="rgba(0,0,0,0.12)")
    fig.update_xaxes(tickvals=[d * 24 + 12 for d in range(7)], ticktext=_DOW_LABELS, row=2, col=1)
    fig.update_yaxes(title_text="kW",     row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="€/kWh",  row=1, col=1, secondary_y=True,
                     showgrid=False, rangemode="tozero")
    fig.update_yaxes(title_text="kWh",    row=2, col=1)
    fig.update_layout(
        height=520, hovermode="x unified",
        margin=dict(t=40, b=10, l=60, r=60),
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

_IT_NORD_MARKET_AVG = 0.107   # stub per stima spread (stesso valore di bill_extractor)

_MESI_IT = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu",
             "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]

_DOW_LABELS_FULL = ["Lunedì", "Martedì", "Mercoledì", "Giovedì",
                    "Venerdì", "Sabato", "Domenica"]


_PROVINCE_IT: dict[str, str] = {
    "AG": "Agrigento", "AL": "Alessandria", "AN": "Ancona", "AO": "Aosta",
    "AP": "Ascoli Piceno", "AQ": "L'Aquila", "AR": "Arezzo", "AT": "Asti",
    "AV": "Avellino", "BA": "Bari", "BG": "Bergamo", "BI": "Biella",
    "BL": "Belluno", "BN": "Benevento", "BO": "Bologna", "BR": "Brindisi",
    "BS": "Brescia", "BT": "Barletta-Andria-Trani", "BZ": "Bolzano",
    "CA": "Cagliari", "CB": "Campobasso", "CE": "Caserta", "CH": "Chieti",
    "CL": "Caltanissetta", "CN": "Cuneo", "CO": "Como", "CR": "Cremona",
    "CS": "Cosenza", "CT": "Catania", "CZ": "Catanzaro", "EN": "Enna",
    "FC": "Forlì-Cesena", "FE": "Ferrara", "FG": "Foggia", "FI": "Firenze",
    "FM": "Fermo", "FR": "Frosinone", "GE": "Genova", "GO": "Gorizia",
    "GR": "Grosseto", "IM": "Imperia", "IS": "Isernia", "KR": "Crotone",
    "LC": "Lecco", "LE": "Lecce", "LI": "Livorno", "LO": "Lodi",
    "LT": "Latina", "LU": "Lucca", "MB": "Monza e Brianza", "MC": "Macerata",
    "ME": "Messina", "MI": "Milano", "MN": "Mantova", "MO": "Modena",
    "MS": "Massa-Carrara", "MT": "Matera", "NA": "Napoli", "NO": "Novara",
    "NU": "Nuoro", "OR": "Oristano", "PA": "Palermo", "PC": "Piacenza",
    "PD": "Padova", "PE": "Pescara", "PG": "Perugia", "PI": "Pisa",
    "PN": "Pordenone", "PO": "Prato", "PR": "Parma", "PT": "Pistoia",
    "PU": "Pesaro e Urbino", "PV": "Pavia", "PZ": "Potenza", "RA": "Ravenna",
    "RC": "Reggio Calabria", "RE": "Reggio Emilia", "RG": "Ragusa",
    "RI": "Rieti", "RM": "Roma", "RN": "Rimini", "RO": "Rovigo",
    "SA": "Salerno", "SI": "Siena", "SO": "Sondrio", "SP": "La Spezia",
    "SR": "Siracusa", "SS": "Sassari", "SU": "Sud Sardegna", "SV": "Savona",
    "TA": "Taranto", "TE": "Teramo", "TN": "Trento", "TO": "Torino",
    "TP": "Trapani", "TR": "Terni", "TS": "Trieste", "TV": "Treviso",
    "UD": "Udine", "VA": "Varese", "VB": "Verbano-Cusio-Ossola",
    "VC": "Vercelli", "VE": "Venezia", "VI": "Vicenza", "VR": "Verona",
    "VT": "Viterbo", "VV": "Vibo Valentia",
}


def _parse_indirizzo_it(raw: str) -> dict:
    """
    Struttura un indirizzo italiano nel formato tipico da bolletta:
    '<via> - <CAP> <COMUNE> (<PROV>)'
    Restituisce dict con keys: linea, cap, comune, provincia_sigla, provincia, raw.
    Se il pattern non corrisponde, mette tutto in linea e lascia gli altri vuoti.
    """
    import re as _re
    result = {"linea": "", "cap": "", "comune": "",
              "provincia_sigla": "", "provincia": "", "raw": raw or ""}
    if not raw:
        return result
    m = _re.match(
        r'^(.*?)\s*[-–]\s*(\d{5})\s+(.+?)\s+\(([A-Z]{2})\)\s*$',
        raw.strip(), _re.IGNORECASE,
    )
    if m:
        sigla = m.group(4).upper()
        result.update({
            "linea":           m.group(1).strip(),
            "cap":             m.group(2).strip(),
            "comune":          m.group(3).strip().title(),
            "provincia_sigla": sigla,
            "provincia":       _PROVINCE_IT.get(sigla, sigla),
        })
    else:
        result["linea"] = raw.strip()
    return result


def _periodo_label(periodo: str) -> str:
    if not periodo:
        return "?"
    try:
        parts = periodo.split("-")
        anno  = parts[0][-2:]
        mese  = int(parts[1]) - 1
        return f"{_MESI_IT[mese]} {anno}"
    except Exception:
        return periodo


def _pvgis_monthly(lat: float, lon: float, kwp: float,
                   tilt: int, azimuth: int) -> list | None:
    try:
        resp = requests.get(
            "https://re.jrc.ec.europa.eu/api/v5_2/PVcalc",
            params={
                "lat": lat, "lon": lon,
                "peakpower": kwp,
                "pvtechtechnology": "crystSi",
                "mountingplace": "building",
                "angle": tilt,
                "aspect": azimuth,
                "outputformat": "json",
                "loss": 14,
            },
            timeout=15,
        )
        data    = resp.json()
        monthly = data.get("outputs", {}).get("monthly", {}).get("fixed", [])
        if len(monthly) == 12:
            return [m.get("E_m", 0) for m in monthly]
    except Exception:
        pass
    return None


def _chart_monthly_energy(month_grid: list) -> go.Figure:
    """
    Stacked bar 12 mesi da month_grid.
    Barre verdi = observed (bolletta reale).
    Barre grigie tratteggiate = estimated (media estrapolata).
    F1/F2/F3 mostrate solo per i mesi observed con fasce disponibili.
    """
    labels  = [r["label"]        for r in month_grid]
    sources = [r.get("consumo_source", r.get("source", "empty")) for r in month_grid]
    totals  = [r["consumo_kwh"]  or 0 for r in month_grid]

    _obs_src = {"observed", "edited"}

    # Controlla se tutte le bollette OBSERVED hanno le fasce
    obs_with_bands = [
        r for r in month_grid
        if r.get("consumo_source", r.get("source", "")) in _obs_src
        and r["f1_kwh"] is not None and r["f2_kwh"] is not None and r["f3_kwh"] is not None
    ]
    show_bands = len(obs_with_bands) > 0

    fig = go.Figure()

    if show_bands:
        # Per observed con fasce: stacked F3+F2+F1; per estimated: barra grigia
        f3_obs  = [r["f3_kwh"]     if r.get("consumo_source", r.get("source", "")) in _obs_src and r["f3_kwh"] is not None else None for r in month_grid]
        f2_obs  = [r["f2_kwh"]     if r.get("consumo_source", r.get("source", "")) in _obs_src and r["f2_kwh"] is not None else None for r in month_grid]
        f1_obs  = [r["f1_kwh"]     if r.get("consumo_source", r.get("source", "")) in _obs_src and r["f1_kwh"] is not None else None for r in month_grid]
        tot_est = [totals[i] if sources[i] not in _obs_src else None for i in range(12)]
        tot_obs_nof = [totals[i] if sources[i] in _obs_src and month_grid[i]["f1_kwh"] is None else None for i in range(12)]

        fig.add_trace(go.Bar(x=labels, y=f3_obs,    name="F3 — fuori punta", marker_color="#A5D6A7"))
        fig.add_trace(go.Bar(x=labels, y=f2_obs,    name="F2 — intermedia",  marker_color="#FFB74D"))
        fig.add_trace(go.Bar(x=labels, y=f1_obs,    name="F1 — punta",       marker_color="#EF9A9A"))
        fig.add_trace(go.Bar(x=labels, y=tot_obs_nof, name="Osservato (totale)",
                             marker_color="#81C784", marker_pattern_shape=""))
        fig.add_trace(go.Bar(x=labels, y=tot_est,   name="Stimato (media)",
                             marker_color="#CFD8DC",
                             marker_pattern_shape="/", marker_pattern_fgcolor="#90A4AE"))
    else:
        col_map  = ["#66BB6A" if s in _obs_src else "#CFD8DC" for s in sources]
        fig.add_trace(go.Bar(x=labels, y=totals, name="Consumo kWh",
                             marker_color=col_map))

    fig.update_layout(
        barmode="stack", height=300,
        margin=dict(t=20, b=10, l=50, r=10),
        yaxis_title="kWh",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        annotations=[dict(
            text="🟢 osservato (bolletta) · ░ stimato (media estrapolata)",
            xref="paper", yref="paper", x=1, y=-0.16,
            showarrow=False, font=dict(size=11, color="grey"), xanchor="right",
        )],
    )
    return fig


def _chart_monthly_peaks(month_grid: list, pot_contrattuale: float) -> go.Figure:
    """
    Grafico picchi mensili da month_grid.
    Solo i mesi observed con picco_kw non-None ottengono la barra colorata.
    Linea tratteggiata = potenza contrattuale.
    """
    labels = [r["label"]   for r in month_grid]
    _pk_obs_src = {"observed", "edited"}
    peaks  = [r["picco_kw"] if r.get("picco_source", r.get("source", "")) in _pk_obs_src and r["picco_kw"] is not None
              else None for r in month_grid]

    cols = []
    for r in month_grid:
        if r.get("picco_source", r.get("source", "")) in _pk_obs_src and r["picco_kw"] is not None:
            cols.append("#1565C0")
        else:
            cols.append("rgba(0,0,0,0)")  # trasparente

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[p or 0 for p in peaks],
                         marker_color=cols, name="Picco mensile",
                         hovertemplate="%{x}: %{y:.0f} kW<extra></extra>"))
    if pot_contrattuale > 0:
        fig.add_hline(
            y=pot_contrattuale, line_dash="dash", line_color="#E53935",
            annotation_text=f"Contrattuale {pot_contrattuale:.0f} kW",
            annotation_position="top left",
        )
        # Evidenzia mesi dove il picco supera la contrattuale
        for i, r in enumerate(month_grid):
            if r["picco_kw"] and r["picco_kw"] > pot_contrattuale:
                fig.add_annotation(
                    x=r["label"], y=r["picco_kw"],
                    text="⚠️", showarrow=False, yshift=8, font=dict(size=14),
                )
    fig.update_layout(
        height=260, margin=dict(t=20, b=10, l=50, r=10),
        yaxis_title="kW", showlegend=False,
        annotations=fig.layout.annotations + (
            dict(text="Solo mesi con picco rilevato in bolletta",
                 xref="paper", yref="paper", x=1, y=-0.18,
                 showarrow=False, font=dict(size=11, color="grey"), xanchor="right"),
        ),
    )
    return fig


def _make_weekly_preview(params: dict, season: str) -> go.Figure:
    """
    Genera un vero profilo settimanale (672 slot = 7 giorni × 96 quart'ore).
    X axis = settimana intera Lun→Dom, con tick per giorno e sub-tick ogni 8h.
    La traccia è unica e continua. Weekend shaded.

    Parametri primari  : turni + giorni_attivi + carico_base_pct
    Parametri secondari: impatto_inv / est → season_factor
    Volume             : consumo_annuo_kwh (locked da Section 1+2) normalizza il profilo.
    """
    turni       = params.get("turni", "Singolo turno")
    giorni_idx  = params.get("giorni_attivi", list(range(5)))
    base_pct    = params.get("carico_base_pct", 15) / 100.0
    consumo_ann = params.get("consumo_annuo_kwh", 100_000)
    imp_inv     = params.get("impatto_invernale", "Neutro")
    imp_est     = params.get("impatto_estivo", "Riduce")

    # Finestre di attività dai parametri turno
    windows = []
    if turni == "Continuo 24h":
        windows = [(0, 96)]  # tutto il giorno
    elif turni == "Due turni":
        t1i = params.get("turno1_ini", 6)  * 4
        t1f = params.get("turno1_fine", 14) * 4
        t2i = params.get("turno2_ini", 14) * 4
        t2f = params.get("turno2_fine", 22) * 4
        windows = [(t1i, t1f), (t2i, t2f)]
    else:  # Singolo turno (default)
        ti  = params.get("ora_inizio", 7)  * 4
        tf  = params.get("ora_fine",   18) * 4
        windows = [(ti, tf)]

    # Season factor (secondary)
    _factor_map = {"Aumenta": 1.2, "Neutro": 1.0, "Riduce": 0.8}
    season_factor = {
        "inverno":   _factor_map.get(imp_inv, 1.0),
        "primavera": 1.0,
        "estate":    _factor_map.get(imp_est, 1.0),
        "autunno":   1.0,
    }.get(season, 1.0)

    # Costruisco matrice 7×96 dai parametri primari
    day_profile = np.ones(96) * base_pct
    for s, e in windows:
        day_profile[s:e] = 1.0

    weekly = np.zeros(7 * 96)
    for day in range(7):
        if day in giorni_idx:
            weekly[day * 96:(day + 1) * 96] = day_profile
        else:
            weekly[day * 96:(day + 1) * 96] = base_pct

    # Normalizzazione: il volume settimanale deve = consumo_ann / 52 × season_factor
    weekly_target_kwh = consumo_ann / 52 * season_factor
    current_kwh = weekly.sum() * SLOT_H
    if current_kwh > 0:
        weekly *= weekly_target_kwh / current_kwh

    # Asse X: 672 slot → tick ai confini di giorno + sub-tick ogni 8h
    x = list(range(672))
    day_labels = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
    tickvals, ticktext = [], []
    for d in range(7):
        tickvals.append(d * 96)
        ticktext.append(day_labels[d])
        for h_off in [32, 64]:  # 08:00, 16:00
            tickvals.append(d * 96 + h_off)
            ticktext.append(f"{h_off // 4:02d}:00")

    fig = go.Figure()

    # Shading weekend (Sab=5, Dom=6)
    for day in [5, 6]:
        fig.add_vrect(
            x0=day * 96, x1=(day + 1) * 96,
            fillcolor="#F5F5F5", opacity=0.6, layer="below", line_width=0,
        )

    # Linee verticali tra i giorni
    for day in range(1, 7):
        fig.add_vline(x=day * 96, line_width=1, line_color="#CCCCCC")

    fig.add_trace(go.Scatter(
        x=x, y=weekly.tolist(),
        mode="lines",
        line=dict(color="#1565C0", width=2),
        fill="tozeroy", fillcolor="rgba(21,101,192,0.10)",
        name="Carico sito",
        hovertemplate="kW: %{y:.1f}<extra></extra>",
    ))

    peak_kw = float(weekly.max())
    fig.add_hline(
        y=peak_kw, line_dash="dot", line_color="#E53935", line_width=1,
        annotation_text=f"Picco {peak_kw:.0f} kW",
        annotation_position="top right",
        annotation_font_size=11,
    )

    fig.update_layout(
        height=320, hovermode="x unified",
        margin=dict(t=30, b=10, l=55, r=10),
        yaxis=dict(title="kW", range=[0, peak_kw * 1.25 or 10]),
        xaxis=dict(
            tickvals=tickvals, ticktext=ticktext,
            tickfont=dict(size=10),
        ),
        showlegend=False,
    )
    return fig


def _chart_fv_daily_season(pv_monthly: list, season: str) -> go.Figure:
    """
    Giorno medio FV per stagione — forma a campana gaussiana sintetica,
    normalizzata alla media giornaliera di quella stagione.
    PVGIS restituisce kWh/mese; questo grafico stima la forma oraria.
    Badge: ○ stimato — forma sintetica (PVGIS dà solo totali mensili).
    """
    import math
    season_months = {
        "Inverno":   [11, 0, 1],   # dic-gen-feb (indici 0-based)
        "Primavera": [2, 3, 4],
        "Estate":    [5, 6, 7],
        "Autunno":   [8, 9, 10],
    }
    months = season_months.get(season, [0, 1, 2])
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Media giornaliera (kWh/giorno) per la stagione
    total_kwh = sum(pv_monthly[m] for m in months)
    total_days = sum(days_per_month[m] for m in months)
    daily_avg_kwh = total_kwh / max(total_days, 1)

    # Parametri campana per stagione: (centro_ora, larghezza_sigma_ore)
    bell_params = {
        "Inverno":   (12.5, 2.5),
        "Primavera": (13.0, 3.5),
        "Estate":    (13.0, 4.5),
        "Autunno":   (12.5, 3.0),
    }
    mu, sigma = bell_params.get(season, (13.0, 3.5))

    hours = [h + 0.5 for h in range(24)]
    raw   = [math.exp(-0.5 * ((h - mu) / sigma) ** 2) for h in hours]
    raw_kwh = sum(raw)
    if raw_kwh > 0:
        profile_kw = [v / raw_kwh * daily_avg_kwh / 1.0 for v in raw]  # kWh/h ≈ kW avg per ora
    else:
        profile_kw = [0.0] * 24

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"{h:02d}:00" for h in range(24)],
        y=profile_kw,
        marker_color="#F9A825",
        name=f"Produzione FV — giorno medio {season}",
    ))
    fig.update_layout(
        height=220,
        margin=dict(t=20, b=10, l=50, r=10),
        yaxis_title="kW medio",
        showlegend=False,
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
    )
    return fig


def _calc_econ_summary(
    econ_bills: list,
    quota_fissa_mese: float | None = None,
    month_grid: list | None = None,
) -> dict:
    """Aggrega KPI economici annualizzati, percentuali breakdown, badge epistemico e anomalie.

    month_grid — se fornito (display_grid dopo gli edit dell'utente in 1.3), viene usato
    per leggere consumo_kwh corretto al posto del valore raw della bolletta.
    I costi € restano sempre letti dalle bollette (non sono editabili in 1.3).
    """
    n = len(econ_bills)
    if n == 0:
        return {
            "n": 0, "ann_f": 1.0,
            "tot_ann": None, "en_ann": None, "pot_ann": None, "res_ann": None,
            "tot_raw": 0.0, "en_raw": 0.0, "pot_raw": 0.0, "res_raw": 0.0,
            "kwh_raw": 0.0, "avg_en_kwh": None, "quota_fissa_ann": None,
            "pct_en": None, "pct_pot": None, "pct_res": None,
            "anomalie": [], "badge": "⚪", "badge_text": "Nessuna bolletta",
        }

    # Lookup kWh corrente per periodo: usa display_grid se disponibile (riflette edit 1.3)
    kwh_by_periodo: dict[str, float] = {}
    if month_grid:
        for r in month_grid:
            p   = r.get("periodo")
            kwh = r.get("consumo_kwh")
            if p and kwh is not None:
                kwh_by_periodo[p] = float(kwh)

    ann_f = 12.0 / n
    tot_raw = en_raw = pot_raw = kwh_raw = 0.0
    anomalie: list[str] = []
    for b in econ_bills:
        d       = b["data"]
        periodo = b.get("periodo") or d.get("periodo") or ""
        tot_m   = d.get("costo_totale_eur") or 0.0
        en_m    = d.get("costo_energia_eur") or 0.0
        pot_m   = d.get("costo_potenza_eur") or 0.0
        tot_raw += tot_m
        en_raw  += en_m
        pot_raw += pot_m
        # kWh: legge da display_grid (verità corrente) se disponibile, altrimenti raw bill.
        # None check esplicito: 0 è un valore valido (mese a zero non deve cadere sul raw).
        _kwh_override = kwh_by_periodo.get(periodo)
        kwh_raw += _kwh_override if _kwh_override is not None else (d.get("consumo_kwh") or 0.0)
        if (tot_m - en_m - pot_m) < -0.5:
            anomalie.append(b.get("periodo") or d.get("periodo", "?"))
    res_raw = tot_raw - en_raw - pot_raw
    tot_ann = round(tot_raw * ann_f)
    en_ann  = round(en_raw  * ann_f)
    pot_ann = round(pot_raw * ann_f)
    res_ann = round(res_raw * ann_f)
    avg_en_kwh      = round(en_raw / kwh_raw, 4) if (kwh_raw > 0 and en_raw > 0) else None
    quota_fissa_ann = round(quota_fissa_mese * 12, 2) if quota_fissa_mese else None
    if tot_ann and tot_ann > 0:
        pct_en  = round(en_ann  / tot_ann * 100, 1)
        pct_pot = round(pot_ann / tot_ann * 100, 1)
        pct_res = round(100.0 - pct_en - pct_pot, 1)
    else:
        pct_en = pct_pot = pct_res = None
    if n >= 12:
        badge, badge_text = "🟢", "Osservato — 12 mesi disponibili"
    elif n >= 3:
        badge, badge_text = "🟡", f"Derivato — {n} mesi annualizzati ×{ann_f:.1f}"
    else:
        s = "bolletta" if n == 1 else "bollette"
        badge, badge_text = "🟡", f"Stima — {n} {s} (×{ann_f:.1f}), incertezza alta"
    return {
        "n": n, "ann_f": ann_f,
        "tot_ann": tot_ann, "en_ann": en_ann, "pot_ann": pot_ann, "res_ann": res_ann,
        "tot_raw": tot_raw, "en_raw": en_raw, "pot_raw": pot_raw, "res_raw": res_raw,
        "kwh_raw": kwh_raw, "avg_en_kwh": avg_en_kwh, "quota_fissa_ann": quota_fissa_ann,
        "pct_en": pct_en, "pct_pot": pct_pot, "pct_res": pct_res,
        "anomalie": anomalie, "badge": badge, "badge_text": badge_text,
    }


def _chart_cost_breakdown(es: dict) -> go.Figure | None:
    """Barchart orizzontale impilato: Energia / Potenza / Altri costi."""
    en  = es.get("en_ann") or 0
    pot = es.get("pot_ann") or 0
    res = es.get("res_ann") or 0
    if not (en or pot or res):
        return None
    pct_en  = es.get("pct_en")  or 0.0
    pct_pot = es.get("pct_pot") or 0.0
    pct_res = es.get("pct_res") or 0.0
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Energia (impattabile)",
        orientation="h", x=[en], y=[""],
        marker_color="#2196F3",
        hovertemplate=(f"<b>Energia</b>  {pct_en:.1f}%  ·  {en:,.0f} €/anno"
                       f"<br>Autoconsumo FV e shifting agiscono qui<extra></extra>"),
    ))
    fig.add_trace(go.Bar(
        name="Potenza (impattabile)",
        orientation="h", x=[pot], y=[""],
        marker_color="#FF9800",
        hovertemplate=(f"<b>Potenza</b>  {pct_pot:.1f}%  ·  {pot:,.0f} €/anno"
                       f"<br>Peak shaving riduce questa componente<extra></extra>"),
    ))
    fig.add_trace(go.Bar(
        name="Altri costi (non impattabili)",
        orientation="h", x=[res], y=[""],
        marker_color="#9E9E9E",
        hovertemplate=(f"<b>Altri costi</b>  {pct_res:.1f}%  ·  {res:,.0f} €/anno"
                       f"<br>Oneri sistema, imposte, partite varie"
                       f"<br>Derivato: totale − energia − potenza<extra></extra>"),
    ))
    fig.update_layout(
        barmode="stack", height=110,
        margin=dict(t=5, b=5, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=11)),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def _chart_annual_costs(econ_bills: list, month_grid: list | None = None, spread_eur_kwh: float | None = None) -> go.Figure | None:
    """Stacked bar 12 mesi: Energia / Potenza / Altri costi.
    Colori pieni = osservato; alpha 35% = stimato; arancio = anomalia.
    Se month_grid è fornito, i mesi stimati scalano proporzionalmente ai kWh mensili
    invece di usare una replica piatta della media."""
    if not econ_bills:
        return None
    from collections import Counter
    anni = []
    for b in econ_bills:
        p = b.get("periodo") or b["data"].get("periodo") or ""
        try:
            anni.append(int(p[:4]))
        except (ValueError, IndexError):
            pass
    if not anni:
        return None

    # Lookup kWh da display_grid (month_grid): include storico + edit utente.
    # Costruito prima del loop sui costi così obs_kwh usa la verità corrente.
    # None check esplicito: kwh=0 è valido e non deve essere ignorato.
    grid_kwh: dict[int, float] = {}
    if month_grid:
        for r in month_grid:
            m_idx = r.get("mese_idx")
            kwh   = r.get("consumo_kwh")
            if m_idx is not None and kwh is not None:
                grid_kwh[m_idx] = float(kwh)

    observed: dict[int, dict] = {}
    obs_kwh:  dict[int, float] = {}
    for b in econ_bills:
        p = b.get("periodo") or b["data"].get("periodo") or ""
        try:
            month = int(p[5:7])
        except (ValueError, IndexError):
            continue
        d   = b["data"]
        tot = d.get("costo_totale_eur") or 0.0
        en  = d.get("costo_energia_eur") or 0.0
        pot = d.get("costo_potenza_eur") or 0.0
        res = tot - en - pot
        observed[month] = {"en": en, "pot": pot, "res": res, "is_anomalia": res < -0.5}
        # kWh: legge da display_grid se disponibile (riflette edit 1.3), altrimenti raw bill
        _kwh_ov = grid_kwh.get(month)
        obs_kwh[month] = _kwh_ov if _kwh_ov is not None else (d.get("consumo_kwh") or 0.0)

    normal  = [v for v in observed.values() if not v["is_anomalia"]]
    avg_en  = sum(v["en"]  for v in normal) / len(normal) if normal else 0.0
    avg_pot = sum(v["pot"] for v in normal) / len(normal) if normal else 0.0
    avg_res = sum(v["res"] for v in normal) / len(normal) if normal else 0.0

    # Rate €/kWh per scalare i mesi stimati — usa obs_kwh (già aggiornato da display_grid)
    normal_months = [m for m, v in observed.items() if not v["is_anomalia"]]
    avg_kwh_obs = (sum(obs_kwh[m] for m in normal_months) / len(normal_months)
                   if normal_months else 0.0)
    rate_en  = avg_en  / avg_kwh_obs if avg_kwh_obs > 0 else 0.0
    rate_res = avg_res / avg_kwh_obs if avg_kwh_obs > 0 else 0.0

    _en_rate = spread_eur_kwh if (spread_eur_kwh is not None and spread_eur_kwh > 0) else rate_en

    en_v: list = []; pot_v: list = []; res_v: list = []
    est:  list = []; anom: list  = []
    for m in range(1, 13):
        if m in observed:
            obs = observed[m]
            # Componente energia: ricalcolata con il prezzo corrente usando i kWh reali del mese.
            # Potenza e altri costi restano dal dato fatturato (non dipendono dal prezzo energia).
            if _en_rate > 0 and obs_kwh.get(m, 0) > 0:
                en = obs_kwh[m] * _en_rate
            else:
                en = obs["en"]
            en_v.append(en); pot_v.append(obs["pot"]); res_v.append(obs["res"])
            est.append(False); anom.append(obs["is_anomalia"])
        else:
            # Mese stimato: tutti i valori da media + kWh del mese
            m_kwh = grid_kwh.get(m, avg_kwh_obs)
            if m_kwh > 0 and _en_rate > 0:
                en_est  = m_kwh * _en_rate
                pot_est = avg_pot
                res_est = max(0.0, m_kwh * rate_res)
            else:
                en_est  = avg_en
                pot_est = avg_pot
                res_est = max(0.0, avg_res)
            en_v.append(en_est); pot_v.append(pot_est); res_v.append(res_est)
            est.append(True); anom.append(False)

    def _rgba(h: str, a: float) -> str:
        r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
        return f"rgba({r},{g},{b},{a})"

    C_EN, C_POT, C_RES, C_ANOM = "#2196F3", "#FF9800", "#9E9E9E", "#FF5722"
    en_col  = [_rgba(C_EN,  0.35) if est[i] else C_EN  for i in range(12)]
    pot_col = [_rgba(C_POT, 0.35) if est[i] else C_POT for i in range(12)]
    res_col = [C_ANOM if anom[i] else (_rgba(C_RES, 0.35) if est[i] else C_RES) for i in range(12)]

    tips = []
    for i in range(12):
        stato = "⚠️ anomalia" if anom[i] else ("◻ stimato" if est[i] else "◼ osservato")
        tot_i = en_v[i] + pot_v[i] + res_v[i]
        note  = "<br><i>Dato reale — breakdown da interpretare con cautela</i>" if anom[i] else ""
        tips.append(
            f"<b>{_MESI_IT[i]}</b> ({stato})<br>"
            f"Energia: {en_v[i]:,.0f} €<br>"
            f"Potenza: {pot_v[i]:,.0f} €<br>"
            f"Altri costi: {res_v[i]:,.0f} €<br>"
            f"<b>Totale: {tot_i:,.0f} €</b>{note}"
        )

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Energia",     x=_MESI_IT, y=en_v,  marker_color=en_col,  hovertext=tips, hoverinfo="text"))
    fig.add_trace(go.Bar(name="Potenza",     x=_MESI_IT, y=pot_v, marker_color=pot_col, hovertext=tips, hoverinfo="text"))
    fig.add_trace(go.Bar(name="Altri costi", x=_MESI_IT, y=res_v, marker_color=res_col, hovertext=tips, hoverinfo="text"))

    n_obs = sum(1 for e in est if not e)
    n_est = 12 - n_obs
    fig.update_layout(
        barmode="stack", height=320,
        margin=dict(t=10, b=30 if n_est > 0 else 5, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        yaxis=dict(title="€", tickformat=",.0f"),
        xaxis=dict(title=None),
    )
    if n_est > 0:
        fig.add_annotation(
            text=f"◼ osservato ({n_obs} mesi)  ·  ◻ stimato ({n_est} mesi — proporzionale ai consumi mensili)",
            xref="paper", yref="paper", x=0, y=-0.12, showarrow=False,
            font=dict(size=10, color="#888"), align="left",
        )
    return fig


def _render_tariff_row(best_meta: dict, avg_en_kwh: float | None = None) -> None:
    """Blocco 3: tariffe contratto lette da bolletta (prezzo offerta, tipo, quota fissa, quota potenza)."""
    tipo = best_meta.get("tipo_prezzo")
    f0   = best_meta.get("f0_eur_kwh")
    f1   = best_meta.get("f1_eur_kwh")
    f2   = best_meta.get("f2_eur_kwh")
    f3   = best_meta.get("f3_eur_kwh")
    qf   = best_meta.get("quota_fissa_eur_mese")
    qp   = best_meta.get("quota_potenza_eur_kw_mese")

    pr_val = pr_lbl = pr_badge = None
    if tipo == "variabile_orario" and f0:
        pr_val, pr_lbl, pr_badge = f0, "Prezzo offerta (€/kWh)", "🟢 letto — PUN orario"
    elif any([f1, f2, f3]):
        vals   = [v for v in [f1, f2, f3] if v]
        pr_val = round(sum(vals) / len(vals), 4)
        pr_lbl, pr_badge = "Prezzo offerta (€/kWh)", "🟢 letto — media F1/F2/F3"
    elif avg_en_kwh:
        pr_val, pr_lbl, pr_badge = avg_en_kwh, "Costo medio energia (€/kWh)", "🟡 derivato — Σ costo en / Σ kWh"

    tipo_label = {
        "variabile_orario": "Variabile orario (PUN/GME)",
        "fisso_fasce":      "Fisso a fasce (F1/F2/F3)",
        "fisso_monorario":  "Fisso monorario",
    }.get(tipo) if tipo else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if pr_val is not None:
            st.metric(pr_lbl, f"{pr_val:.4f} €/kWh")
            st.caption(pr_badge)
    with c2:
        if tipo_label:
            st.metric("Tipo prezzo energia", tipo_label)
            st.caption("🟢 letto da bolletta — meccanismo di prezzo (distinto dalla struttura fasce)")
    with c3:
        if qf:
            st.metric("Quota fissa", f"{qf:.2f} €/mese")
            st.caption("🟢 letto da bolletta")
    with c4:
        if qp and 0.5 <= qp <= 50.0:
            st.metric("Quota potenza (rate)", f"{qp:.3f} €/kW/mese")
            st.caption("🟢 letto da bolletta")


def _calc_kwh_breakdown(es: dict, best_meta: dict, consumo_annuo: float) -> dict | None:
    """Composizione del costo per kWh: prezzo contrattuale, energia media, costo totale, overhead.

    Restituisce None se mancano dati sufficienti a produrre almeno un valore significativo.
    """
    avg_en  = es.get("avg_en_kwh")
    tot_ann = es.get("tot_ann")
    if not avg_en and not tot_ann:
        return None

    tipo = best_meta.get("tipo_prezzo")
    f0   = best_meta.get("f0_eur_kwh")
    f1   = best_meta.get("f1_eur_kwh")
    f2   = best_meta.get("f2_eur_kwh")
    f3   = best_meta.get("f3_eur_kwh")

    prezzo_contrattuale = None
    if tipo == "variabile_orario" and f0:
        prezzo_contrattuale = f0
    elif any([f1, f2, f3]):
        vals = [v for v in [f1, f2, f3] if v]
        prezzo_contrattuale = round(sum(vals) / len(vals), 4)

    costo_medio_tot_kwh = None
    if tot_ann and consumo_annuo > 0:
        costo_medio_tot_kwh = round(tot_ann / consumo_annuo, 4)

    overhead_kwh = None
    if costo_medio_tot_kwh is not None and avg_en is not None:
        raw_overhead = costo_medio_tot_kwh - avg_en
        overhead_kwh = round(max(0.0, raw_overhead), 4)

    note = None
    if avg_en and costo_medio_tot_kwh and costo_medio_tot_kwh > avg_en * 1.02:
        diff_pct = (costo_medio_tot_kwh - avg_en) / avg_en * 100
        note = (
            f"Il costo totale ({costo_medio_tot_kwh:.4f} €/kWh) è superiore al prezzo energia "
            f"({avg_en:.4f} €/kWh) del {diff_pct:.0f}%. "
            f"I {overhead_kwh:.4f} €/kWh di differenza sono quota potenza, oneri e imposte — "
            "costi fissi che esistono indipendentemente dai consumi."
        )

    return {
        "prezzo_contrattuale":     prezzo_contrattuale,
        "costo_medio_energia_kwh": avg_en,
        "costo_medio_tot_kwh":     costo_medio_tot_kwh,
        "overhead_kwh":            overhead_kwh,
        "note":                    note,
    }


def _classify_cost_nature(es: dict, best_meta: dict, pot_contrattuale: float) -> dict | None:
    """Classifica la spesa per natura economica: variabile con i consumi / legata alla potenza / strutturale.

    La quota_fissa (se disponibile) viene spostata dalla voce 'energia' alla voce 'strutturale'
    perché non dipende dai consumi.
    """
    tot_ann = es.get("tot_ann")
    en_ann  = es.get("en_ann")  or 0
    pot_ann = es.get("pot_ann") or 0
    res_ann = es.get("res_ann") or 0
    if not tot_ann or tot_ann <= 0:
        return None

    qf_mese = best_meta.get("quota_fissa_eur_mese")
    qf_ann  = (qf_mese * 12) if qf_mese else 0.0

    if qf_ann > 0 and qf_ann < en_ann:
        variabile = en_ann - qf_ann
        fisso     = res_ann + qf_ann
        nota = (
            f"La quota fissa fornitore ({qf_mese:.2f} €/mese · {qf_ann:,.0f} €/anno) "
            "è classificata come costo strutturale perché resta anche a consumi zero."
        )
    else:
        variabile = en_ann
        fisso     = res_ann
        nota      = None

    pct_variabile = round(variabile / tot_ann * 100, 1)
    pct_potenza   = round(pot_ann   / tot_ann * 100, 1)
    pct_fisso     = round(max(0.0, 100.0 - pct_variabile - pct_potenza), 1)

    return {
        "variabile":     round(variabile),
        "potenza":       round(pot_ann),
        "fisso":         round(fisso),
        "pct_variabile": pct_variabile,
        "pct_potenza":   pct_potenza,
        "pct_fisso":     pct_fisso,
        "nota":          nota,
    }


def _summarize_cost_pattern(econ_bills: list) -> str:
    """Frase di lettura sintetica sul pattern stagionale dei costi osservati.

    Opzione B: legge direttamente da econ_bills senza toccare _chart_annual_costs.
    Usa solo i mesi con costo_totale_eur > 0.
    """
    if not econ_bills:
        return ""

    month_totals: dict[int, float] = {}
    month_comp:   dict[int, dict]  = {}
    for b in econ_bills:
        p = b.get("periodo") or b["data"].get("periodo") or ""
        try:
            m = int(p[5:7])
        except (ValueError, IndexError):
            continue
        d   = b["data"]
        tot = d.get("costo_totale_eur") or 0.0
        if tot <= 0:
            continue
        en  = d.get("costo_energia_eur") or 0.0
        pot = d.get("costo_potenza_eur") or 0.0
        month_totals[m] = tot
        month_comp[m]   = {"en": en, "pot": pot, "res": max(0.0, tot - en - pot)}

    n_obs = len(month_totals)
    if n_obs == 0:
        return ""

    max_m    = max(month_totals, key=lambda k: month_totals[k])
    min_m    = min(month_totals, key=lambda k: month_totals[k])
    max_cost = month_totals[max_m]
    min_cost = month_totals[min_m]
    avg_cost = sum(month_totals.values()) / n_obs
    variaz   = (max_cost - min_cost) / avg_cost * 100 if avg_cost > 0 else 0

    comp          = month_comp[max_m]
    dominant_key  = max(comp, key=lambda k: comp[k])
    dominant_lbl  = {"en": "energia", "pot": "potenza", "res": "altri costi"}[dominant_key]

    # Lettura epistemicamente corretta: dipende da quanti mesi sono realmente osservati
    if n_obs == 1:
        return (
            f"◼ 1 mese osservato · ◻ 11 stimati — "
            "la distribuzione annuale è una stima basata su un solo mese reale. "
            "Non rispecchia ancora la stagionalità effettiva del sito."
        )

    if n_obs <= 5:
        return (
            f"◼ {n_obs} mesi osservati · ◻ {12 - n_obs} stimati — quadro parziale. "
            "I mesi senza bolletta sono ricostruzioni proporzionali, non dati reali."
        )

    # 6–11 mesi: lettura mista, con caveat
    prefix = f"◼ {n_obs} mesi osservati · ◻ {12 - n_obs} stimati."
    if max_m != min_m and variaz > 15:
        reading = (
            f"Mese più costoso: **{_MESI_IT[max_m - 1]}** ({max_cost:,.0f} €) · "
            f"meno costoso: **{_MESI_IT[min_m - 1]}** ({min_cost:,.0f} €) — "
            f"variazione del {variaz:.0f}% sulla media. "
            f"Componente dominante nei mesi di picco: **{dominant_lbl}**."
        )
        caveat = "Lettura parzialmente osservata — i mesi stimati potrebbero alterare il quadro."
    else:
        reading = f"Spesa mensile relativamente stabile (variazione {variaz:.0f}% tra il mese più e meno costoso)."
        caveat  = "Lettura parzialmente osservata."

    if n_obs >= 12:
        # 12 mesi: lettura assertiva senza caveat
        if max_m != min_m and variaz > 15:
            return (
                f"Mese più costoso: **{_MESI_IT[max_m - 1]}** ({max_cost:,.0f} €) · "
                f"meno costoso: **{_MESI_IT[min_m - 1]}** ({min_cost:,.0f} €) — "
                f"variazione del {variaz:.0f}% sulla media. "
                f"Componente dominante nei mesi di picco: **{dominant_lbl}**."
            )
        return f"Spesa mensile stabile durante l'anno (variazione {variaz:.0f}% tra il mese più e meno costoso)."

    return f"{prefix}  {reading}  {caveat}"


def _build_month_grid(ok_bills: list) -> list[dict]:
    """
    Restituisce sempre una lista di 12 dict (un mese per voce), ordinata Gen→Dic.
    Per i mesi con bolletta: source="observed" + dati reali.
    Per i mesi senza bolletta: source="estimated" + media mensile dalle bollette presenti.

    Ogni dict: {anno, mese_idx (1-12), periodo, label, consumo_kwh, f1_kwh, f2_kwh,
                f3_kwh, picco_kw, source}
    """
    # Determina anno di riferimento (il più frequente tra le bollette, o anno corrente)
    from collections import Counter
    import datetime

    anni = []
    for b in ok_bills:
        p = b.get("periodo") or b["data"].get("periodo") or ""
        try:
            anni.append(int(p[:4]))
        except (ValueError, IndexError):
            pass
    anno_ref = Counter(anni).most_common(1)[0][0] if anni else datetime.date.today().year

    # Indicizza bollette per mese — solo anno di riferimento
    observed: dict[int, dict] = {}
    all_data: list[dict] = []          # tutti gli anni, per la media
    for b in ok_bills:
        p = b.get("periodo") or b["data"].get("periodo") or ""
        try:
            yr   = int(p[:4])
            mese = int(p[5:7])
        except (ValueError, IndexError):
            continue
        all_data.append(b["data"])
        if yr == anno_ref:
            observed[mese] = b["data"]

    # Media mensile da TUTTE le bollette (anche anni diversi) per i mesi mancanti
    n_all = len(all_data)
    if n_all > 0:
        avg_consumo = sum(d.get("consumo_kwh") or 0 for d in all_data) / n_all
        avg_f1      = (sum(d.get("f1_kwh") or 0 for d in all_data) / n_all
                       if all(d.get("f1_kwh") is not None for d in all_data) else None)
        avg_f2      = (sum(d.get("f2_kwh") or 0 for d in all_data) / n_all
                       if all(d.get("f2_kwh") is not None for d in all_data) else None)
        avg_f3      = (sum(d.get("f3_kwh") or 0 for d in all_data) / n_all
                       if all(d.get("f3_kwh") is not None for d in all_data) else None)
    else:
        avg_consumo = avg_f1 = avg_f2 = avg_f3 = None

    grid = []
    for m in range(1, 13):
        periodo = f"{anno_ref}-{m:02d}"
        label   = f"{_MESI_IT[m - 1]} {str(anno_ref)[-2:]}"
        if m in observed:
            d = observed[m]
            grid.append({
                "anno": anno_ref, "mese_idx": m, "periodo": periodo, "label": label,
                "consumo_kwh": d.get("consumo_kwh"),
                "f1_kwh":      d.get("f1_kwh"),
                "f2_kwh":      d.get("f2_kwh"),
                "f3_kwh":      d.get("f3_kwh"),
                "picco_kw":    d.get("picco_kw"),
                "source":      "observed",
            })
        else:
            grid.append({
                "anno": anno_ref, "mese_idx": m, "periodo": periodo, "label": label,
                "consumo_kwh": round(avg_consumo) if avg_consumo is not None else None,
                "f1_kwh":      round(avg_f1, 1)   if avg_f1    is not None else None,
                "f2_kwh":      round(avg_f2, 1)   if avg_f2    is not None else None,
                "f3_kwh":      round(avg_f3, 1)   if avg_f3    is not None else None,
                "picco_kw":    None,
                "source":      "estimated" if n_all > 0 else "empty",
            })
    return grid


def _build_effective_model(
    month_grid: list,
    user_edits_kwh: dict,
    user_edits_peaks: dict,
) -> list:
    """Unisce month_grid con le correzioni utente da 1.3 e 1.4.

    Ogni dict risultante ha consumo_source e picco_source separati invece del
    singolo campo source — perché un mese può avere consumo osservato ma picco
    mancante, oppure solo uno dei due corretto manualmente.

    consumo_source: observed | estimated | empty | edited
    picco_source:   observed | missing | empty | edited
    """
    result = []
    for row in month_grid:
        r = dict(row)
        label = r["label"]
        raw_source = r.pop("source", "empty")

        # consumo_source e campi kWh
        if label in user_edits_kwh:
            consumo_source = "edited"
            e = user_edits_kwh[label]
            for fk in ("f1_kwh", "f2_kwh", "f3_kwh"):
                if fk in e:
                    r[fk] = e[fk]
            if all(r.get(k) is not None for k in ("f1_kwh", "f2_kwh", "f3_kwh")):
                r["consumo_kwh"] = round(r["f1_kwh"] + r["f2_kwh"] + r["f3_kwh"])
        else:
            consumo_source = raw_source

        # picco_source e picco_kw
        if label in user_edits_peaks:
            picco_source = "edited"
            r["picco_kw"] = user_edits_peaks[label]
        elif raw_source == "observed":
            picco_source = "observed" if r.get("picco_kw") is not None else "missing"
        else:
            picco_source = "empty"

        r["consumo_source"] = consumo_source
        r["picco_source"] = picco_source
        result.append(r)
    return result


# ── Sezione 2.3 helpers ────────────────────────────────────────────────────────

def _s2_fv_hourly_from_monthly(pv_monthly: list) -> list:
    """Profilo FV orario sintetico (8760 h) che rispetta i totali mensili PVGIS."""
    import math
    _SIGMA = 2.0
    _DAYS  = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    _gauss_daily_sum = sum(
        math.exp(-0.5 * ((h + 0.5 - 12.5) / _SIGMA) ** 2) for h in range(24)
    )
    fv = []
    for m, days in enumerate(_DAYS):
        daily_kwh = pv_monthly[m] / days if days > 0 else 0.0
        scale = daily_kwh / _gauss_daily_sum if _gauss_daily_sum > 0 else 0.0
        for _ in range(days):
            for h in range(24):
                fv.append(max(0.0, math.exp(-0.5 * ((h + 0.5 - 12.5) / _SIGMA) ** 2) * scale))
    return fv


def _s2_extract_f123_from_grid(grid: list) -> tuple:
    """Estrae F1/F2/F3 e picchi da effective_model (12 righe). Fallback 50/25/25 se F1/F2/F3 assenti."""
    f1 = [row.get("f1_kwh") or 0.0 for row in grid]
    f2 = [row.get("f2_kwh") or 0.0 for row in grid]
    f3 = [row.get("f3_kwh") or 0.0 for row in grid]
    if not any(f1) and not any(f2) and not any(f3):
        totals = [row.get("consumo_kwh") or 0.0 for row in grid]
        f1 = [v * 0.50 for v in totals]
        f2 = [v * 0.25 for v in totals]
        f3 = [v * 0.25 for v in totals]
    picchi = [row.get("picco_kw") or 0.0 for row in grid]
    return f1, f2, f3, picchi


def _s2_section_cache_key(kwp, tilt, azimuth, pv_monthly, user_constraints, grid) -> str:
    import hashlib, json
    grid_summary = [
        {"f1": row.get("f1_kwh"), "f2": row.get("f2_kwh"),
         "f3": row.get("f3_kwh"), "pk": row.get("picco_kw"),
         "c": row.get("consumo_kwh")}
        for row in (grid or [])
    ]
    payload = {
        "kwp": kwp, "tilt": tilt, "az": azimuth,
        "pv": pv_monthly, "uc": user_constraints, "grid": grid_summary,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _build_section_2_reconstruction(
    effective_model: list,
    pv_monthly: list,
    kwp: float,
    tilt: float,
    azimuth: float,
    user_constraints,
) -> dict | None:
    """
    Chiama reconcile() con i dati attuali del form (senza SDI completo).
    Ritorna dict con KPI e serie mensili, o None se dati insufficienti.
    """
    from intake.site_reconstruction import SiteEnergyState, reconcile

    if not effective_model or not pv_monthly or kwp <= 0:
        return None
    if not any(row.get("consumo_kwh") for row in effective_model):
        return None

    f1, f2, f3, picchi = _s2_extract_f123_from_grid(effective_model)
    fv_oraria = _s2_fv_hourly_from_monthly(pv_monthly)
    anno_ref = next((row.get("anno") for row in effective_model if row.get("anno")), 2025)
    quota_pot = float(st.session_state.get("intake_quota", 12.0) or 12.0)

    uc = user_constraints or {}
    user_ac_pct_ann = user_ac_pct_mens = user_sup_ann = user_sup_mens = None
    if uc.get("type") == "autoconsumo_pct" and uc.get("scope") == "annuo":
        user_ac_pct_ann = float(uc.get("annual_value") or 0)
    elif uc.get("type") == "autoconsumo_pct" and uc.get("scope") == "mensile":
        user_ac_pct_mens = [float(v) for v in uc.get("monthly_values") or [0] * 12]
    elif uc.get("type") == "surplus_kwh" and uc.get("scope") == "annuo":
        user_sup_ann = float(uc.get("annual_value") or 0)
    elif uc.get("type") == "surplus_kwh" and uc.get("scope") == "mensile":
        user_sup_mens = [float(v) for v in uc.get("monthly_values") or [0] * 12]

    state = SiteEnergyState(
        prelievo_f1_mensile=f1,
        prelievo_f2_mensile=f2,
        prelievo_f3_mensile=f3,
        picchi_mensili_kw=picchi,
        quota_potenza_eur_kw_mese=quota_pot,
        has_fv=True,
        fv_kwp=float(kwp),
        fv_tilt=float(tilt),
        fv_azimuth=float(azimuth),
        fv_oraria_pvgis_kw=fv_oraria,
        fv_mensile_pvgis_kwh=list(pv_monthly),
        anno_riferimento=anno_ref,
        user_autoconsumo_pct_annuo=user_ac_pct_ann,
        user_autoconsumo_pct_mensile=user_ac_pct_mens,
        user_surplus_kwh_annuo=user_sup_ann,
        user_surplus_kwh_mensile=user_sup_mens,
    )

    try:
        state = reconcile(state)
    except (ValueError, RuntimeError) as e:
        return {"error": str(e)}

    prelievo_mensile = [f1[m] + f2[m] + f3[m] for m in range(12)]
    fabb_ann = state.fabbisogno_annuo_kwh.value   if state.fabbisogno_annuo_kwh   else 0.0
    ac_ann   = state.autoconsumo_fv_annuo_kwh.value if state.autoconsumo_fv_annuo_kwh else 0.0
    sup_ann  = state.surplus_fv_annuo_kwh.value   if state.surplus_fv_annuo_kwh   else 0.0
    fv_tot   = sum(pv_monthly)
    spread   = float(st.session_state.get("intake_spread", 0.095) or 0.095)

    return {
        "fabbisogno_annuo":    fabb_ann,
        "autoconsumo_annuo":   ac_ann,
        "surplus_annuo":       sup_ann,
        "autoconsumo_pct":     (ac_ann / fv_tot * 100.0) if fv_tot > 0 else 0.0,
        "autosufficienza_pct": (ac_ann / fabb_ann * 100.0) if fabb_ann > 0 else 0.0,
        "costo_evitato_eur":   ac_ann * spread,
        "fabbisogno_mensile":  [v.value for v in (state.fabbisogno_mensile_kwh  or [])],
        "autoconsumo_mensile": [v.value for v in (state.autoconsumo_fv_mensile_kwh or [])],
        "surplus_mensile":     [v.value for v in (state.surplus_fv_mensile_kwh  or [])],
        "prelievo_mensile":    prelievo_mensile,
        "fv_mensile":          list(pv_monthly),
        "picchi_mensili":      picchi,
        "reconcile_mode":      state.reconcile_mode or "auto",
        "autoconsumo_source":  (
            state.autoconsumo_fv_annuo_kwh.source
            if state.autoconsumo_fv_annuo_kwh else "estimated"
        ),
        "overall_confidence":  state.overall_confidence or "medium",
        "band_match_error_pct": state.band_match_error_pct or {},
        "assumptions_active":  list(state.assumptions_active or []),
        "spread":              spread,
    }


def _fband_simple(day: int, slot: int) -> str:
    """Approssimazione fasce F1/F2/F3 per il blocco coerenza (senza festivi)."""
    h = slot // 4
    if day == 6:
        return "F3"
    if day == 5:
        return "F2" if 7 <= h < 23 else "F3"
    if 8 <= h < 19:
        return "F1"
    if 7 <= h < 8 or 19 <= h < 23:
        return "F2"
    return "F3"


def _render_multi_bill_upload() -> None:
    bills = st.session_state.setdefault("bills_uploaded", [])

    # Gestisci retry in sospeso (deve essere PRIMA del rendering)
    _retry_idx = st.session_state.pop("_retry_bill_idx", None)
    if _retry_idx is not None and _retry_idx < len(bills):
        _rb = bills[_retry_idx]
        _rb_bytes = _rb.get("file_bytes")
        if _rb_bytes:
            with st.spinner(f"Rianalisi {_rb['file_name']}…"):
                _parsed = _parse_bill_pdf(_rb_bytes)
            if "parse_error" in _parsed:
                st.session_state["bills_uploaded"][_retry_idx]["parse_error"] = _parsed["parse_error"]
            else:
                _new_bills, _site_meta = _expand_parsed_bill(_parsed, file_name=_rb["file_name"])
                st.session_state["bills_uploaded"].pop(_retry_idx)
                for _nb in _new_bills:
                    st.session_state["bills_uploaded"].append({
                        "file_name": _rb["file_name"],
                        "periodo":   _nb.get("periodo"),
                        "parse_ok":  True,
                        "data":      _nb,
                        "site_meta": _site_meta,
                    })
            st.rerun()

    # Lista bollette caricata — compatta, senza box pesanti
    n_ok = len([b for b in bills if b.get("parse_ok")])
    to_remove = None

    if bills:
        if n_ok > 0:
            periodi = sorted(
                b.get("periodo") or "" for b in bills if b.get("parse_ok") and b.get("periodo")
            )
            mesi_str = (
                f"{_periodo_label(periodi[0])} a {_periodo_label(periodi[-1])}"
                if periodi else "—"
            )
            st.caption(f"📊 **{n_ok} bollette** · {mesi_str} · copertura {n_ok}/12 mesi")

        # Contenitore scrollabile se ci sono molte bollette
        _list_height = min(36 * len(bills) + 16, 260)
        with st.container(height=_list_height, border=False):
            for i, bill in enumerate(bills):
                c_info, c_retry, c_btn = st.columns([11, 2, 1])
                with c_info:
                    if bill.get("parse_ok"):
                        d   = bill["data"]
                        kwh = (d.get("consumo_kwh") or 0)
                        kw  = (d.get("potenza_contrattuale_kw") or 0)
                        lbl = _periodo_label(bill.get("periodo"))
                        st.markdown(
                            f"<small><span style='color:#4CAF50'>●</span> "
                            f"<b>{lbl}</b> &nbsp; {kwh:,.0f} kWh &nbsp; {kw:.0f} kW</small>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<small><span style='color:#ef5350'>●</span> "
                            f"{bill['file_name']} — {bill.get('parse_error', 'errore')}</small>",
                            unsafe_allow_html=True,
                        )
                with c_retry:
                    if not bill.get("parse_ok") and bill.get("file_bytes"):
                        if st.button("↺ Rianalizza", key=f"retry_bill_{i}"):
                            st.session_state["_retry_bill_idx"] = i
                            st.rerun()
                with c_btn:
                    if st.button("×", key=f"rm_bill_{i}", help="Rimuovi"):
                        to_remove = i

    if to_remove is not None:
        st.session_state["bills_uploaded"].pop(to_remove)
        st.rerun()

    # Upload area
    st.caption("Carica una o più bollette PDF:")
    uploaded = st.file_uploader(
        "Bollette PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"multi_bill_uploader_{n_ok}",
        label_visibility="collapsed",
    )

    col_btn, col_warn = st.columns([2, 5])
    parser_ok = _bill_parser_available()
    with col_btn:
        do_parse = st.button(
            "Analizza bollette →",
            type="primary",
            disabled=(not uploaded or not parser_ok),
            key="parse_bills_btn",
        )
    if not parser_ok:
        with col_warn:
            st.warning("Parser non disponibile — Google API key mancante.", icon="⚠️")

    if uploaded and do_parse and parser_ok:
        # Mappa {periodo: source} per gestire upgrade storico→fatturato
        existing_map: dict[str, int] = {
            b["periodo"]: i
            for i, b in enumerate(st.session_state.get("bills_uploaded", []))
            if b.get("periodo")
        }
        existing_names = {b["file_name"] for b in bills}
        new_files = [f for f in uploaded if f.name not in existing_names]
        if not new_files:
            st.info("Tutte le bollette selezionate sono già presenti.")
        else:
            prog = st.progress(0, text="Analisi in corso…")
            for idx, f in enumerate(new_files):
                prog.progress((idx + 1) / len(new_files),
                              text=f"Analisi {f.name}…")
                try:
                    _file_bytes = f.read()
                    parsed = _parse_bill_pdf(_file_bytes)
                    if "parse_error" in parsed:
                        st.session_state["bills_uploaded"].append({
                            "file_name":  f.name, "periodo": None,
                            "parse_ok":   False, "data": {},
                            "parse_error": parsed["parse_error"],
                            "file_bytes": _file_bytes,
                        })
                    else:
                        new_bills, site_meta = _expand_parsed_bill(parsed, file_name=f.name)
                        duplicati  = []
                        sostituiti = []
                        aggiunti   = 0
                        for bill in new_bills:
                            periodo     = bill.get("periodo")
                            new_source  = bill.get("source", "")
                            new_entry   = {
                                "file_name": f.name,
                                "periodo":   periodo,
                                "parse_ok":  True,
                                "data":      bill,
                                "site_meta": site_meta,
                            }
                            if periodo and periodo in existing_map:
                                idx_existing = existing_map[periodo]
                                old_source   = st.session_state["bills_uploaded"][idx_existing]["data"].get("source", "")
                                # Upgrade: fatturato sostituisce storico
                                if new_source == "bill_fatturato" and old_source == "bill_storico":
                                    st.session_state["bills_uploaded"][idx_existing] = new_entry
                                    sostituiti.append(periodo)
                                    aggiunti += 1
                                else:
                                    duplicati.append(periodo)
                                continue
                            st.session_state["bills_uploaded"].append(new_entry)
                            if periodo:
                                existing_map[periodo] = len(st.session_state["bills_uploaded"]) - 1
                            aggiunti += 1
                        # Messaggio riepilogo
                        n_tot = len(new_bills)
                        nome  = site_meta.get("nome_cliente") or f.name
                        if aggiunti > 0:
                            periodi_ok = [
                                b["periodo"] for b in st.session_state["bills_uploaded"]
                                if b.get("file_name") == f.name and b.get("parse_ok")
                                   and b.get("periodo")
                            ]
                            if periodi_ok:
                                label = f"{periodi_ok[0]} – {periodi_ok[-1]}" if len(periodi_ok) > 1 else periodi_ok[0]
                            else:
                                label = ""
                            msg = f"✅ {aggiunti} periodi aggiunti ({label}) — {nome}"
                            if sostituiti:
                                msg += f"  ·  {len(sostituiti)} upgrade storico→fatturato"
                            if duplicati:
                                msg += f"  ·  {len(duplicati)} già presenti"
                            st.success(msg)
                        else:
                            st.warning(f"⚠️ {f.name}: tutti i {n_tot} periodi erano già presenti.")
                except Exception as exc:
                    st.session_state["bills_uploaded"].append({
                        "file_name": f.name, "periodo": None,
                        "parse_ok": False, "data": {},
                        "parse_error": str(exc),
                    })
            prog.empty()
            st.rerun()


def _page_block1_intake() -> None:
    bills    = st.session_state.setdefault("bills_uploaded", [])
    ok_bills = [b for b in bills if b.get("parse_ok")]

    # ── Header ────────────────────────────────────────────────────────────────
    col_h, col_leg = st.columns([4, 3])
    with col_h:
        st.markdown("# 📍 Intake Energetico Sito")
        st.caption(
            "Block 1 · Radiografia energetica — analisi del sito prima della simulazione BESS"
        )
    with col_leg:
        st.markdown(
            "<br><span style='font-size:0.85rem'>"
            "<span style='color:#2E7D32'>●</span> osservato &nbsp;"
            "<span style='color:#F9A825'>◐</span> derivato &nbsp;"
            "<span style='color:#9E9E9E'>○</span> stimato"
            "</span>",
            unsafe_allow_html=True,
        )

    # Progress bar: 3 contributi (bollette / FV / carico)
    step1 = len(ok_bills) > 0
    step2 = st.session_state.get("intake_ha_pv", False) and (
        st.session_state.get("intake_pv_monthly") is not None
        or not st.session_state.get("intake_ha_pv", False)
    )
    step3 = True  # ha sempre valori default
    n_done = int(step1) + 1 + int(step3)  # FV è opzionale
    st.progress(min(n_done / 3, 1.0))
    st.divider()

    # ════════════════════════════════════════════════════════════════
    # SEZIONE 1 — Bollette e assorbimento da rete
    # ════════════════════════════════════════════════════════════════
    st.markdown("## 1 · Bollette e assorbimento da rete")
    st.caption(
        "Carica le bollette elettriche del sito. "
        "Il sistema estrae automaticamente i dati energetici ed economici."
    )

    _render_multi_bill_upload()
    # Refresh dopo eventuale rerun da upload
    bills    = st.session_state.get("bills_uploaded", [])
    ok_bills = [b for b in bills if b.get("parse_ok")]

    # site_meta dal bill più recente (contiene quota_potenza, prezzi fascia, nome cliente, ecc.)
    _sorted_ok = sorted(ok_bills, key=lambda x: x.get("periodo") or "", reverse=True)
    _best_meta = next((b.get("site_meta") for b in _sorted_ok if b.get("site_meta")), {}) or {}

    st.divider()

    # 1.2 — Dati fissi sito
    st.markdown("#### 1.2 · Dati fissi sito")

    # Valori da bolletta
    pot_from_bill  = next(
        (b["data"].get("potenza_contrattuale_kw")
         for b in _sorted_ok if b["data"].get("potenza_contrattuale_kw")),
        None,
    )
    pot_disp_bill  = _best_meta.get("potenza_disponibile_kw")
    _raw_indirizzo = _best_meta.get("indirizzo_fornitura") or ""
    _addr          = _parse_indirizzo_it(_raw_indirizzo)

    # Pre-fill in session_state — solo se il campo è ancora vuoto
    def _prefill(key, val):
        if val and not st.session_state.get(key):
            st.session_state[key] = val

    _prefill("intake_nome",             _best_meta.get("nome_cliente"))
    _prefill("intake_pod",              _best_meta.get("codice_pod"))
    _prefill("intake_indirizzo_linea",  _addr["linea"])
    _prefill("intake_cap",              _addr["cap"])
    _prefill("intake_comune",           _addr["comune"])
    _prefill("intake_provincia",        _addr["provincia"])        # nome completo
    _prefill("intake_provincia_sigla",  _addr["provincia_sigla"])  # sigla per uso interno
    if pot_from_bill and not st.session_state.get("intake_pot_contr"):
        st.session_state["intake_pot_contr"] = int(round(pot_from_bill))
    if pot_disp_bill and not st.session_state.get("intake_pot_disp"):
        st.session_state["intake_pot_disp"] = int(round(pot_disp_bill))

    # Riga 1: nome + POD
    col_nome, col_pod = st.columns([3, 2])
    with col_nome:
        nome_cliente = st.text_input(
            "Nome cliente / azienda", key="intake_nome",
            placeholder="Es: Acciaierie Nord S.r.l.",
        )
    with col_pod:
        st.text_input(
            "Codice POD", key="intake_pod",
            placeholder="Es: IT001E12345678",
            help="Identificativo unico del punto di fornitura",
        )
        if _best_meta.get("codice_pod"):
            st.caption("🟢 da bolletta")

    # Riga 2: indirizzo linea
    _raw_help = (f"Indirizzo originale da bolletta: {_raw_indirizzo}"
                 if _raw_indirizzo and _addr["linea"] != _raw_indirizzo else "")
    st.text_input(
        "Indirizzo (via / numero)", key="intake_indirizzo_linea",
        placeholder="Es: Via Val Chiampo, SNC",
        help=_raw_help or "Indirizzo del punto di fornitura",
    )

    # Riga 3: CAP + comune + provincia (nome completo)
    col_cap, col_com, col_prov = st.columns([1, 3, 2])
    with col_cap:
        st.text_input("CAP", key="intake_cap", placeholder="36050")
    with col_com:
        st.text_input(
            "Comune", key="intake_comune",
            placeholder="Es: Montorso Vicentino",
            help="Usato per geocoding e PVGIS",
        )
        if _addr["comune"]:
            st.caption("🟢 da bolletta · usato per PVGIS")
        else:
            st.caption("📍 inserimento manuale")
    with col_prov:
        _prov_placeholder = (
            f"Es: Vicenza (VI)" if not _addr["provincia"] else
            f"{_addr['provincia']} ({_addr['provincia_sigla']})"
        )
        st.text_input(
            "Provincia", key="intake_provincia",
            placeholder="Es: Vicenza",
            help=(f"Sigla: {_addr['provincia_sigla']}" if _addr["provincia_sigla"] else
                  "Nome completo della provincia"),
        )
        if _addr["provincia_sigla"]:
            st.caption(f"🟢 {_addr['provincia_sigla']} da bolletta")

    # Riga 4: potenze + tipo tariffa
    col_pot, col_potd, col_tar = st.columns([1, 1, 2])
    with col_pot:
        pot_contrattuale = st.number_input(
            "Pot. contrattuale (kW)",
            min_value=0, max_value=5000,
            value=st.session_state.get("intake_pot_contr", 0),
            step=5, key="intake_pot_contr",
        )
        if pot_from_bill:
            st.caption("🟢 da bolletta")
        else:
            st.caption("⚪ manuale")
    with col_potd:
        st.number_input(
            "Pot. disponibile (kW)",
            min_value=0, max_value=5000,
            value=st.session_state.get("intake_pot_disp", 0),
            step=5, key="intake_pot_disp",
        )
        if pot_disp_bill:
            st.caption("🟢 da bolletta")
        else:
            st.caption("⚪ manuale")
    with col_tar:
        tipo_tariffa = st.selectbox(
            "Struttura fasce orarie",
            ["Trioraria F1-F2-F3", "Bioraria F1-F23", "Monoraria", "Non so"],
            key="intake_tariffa",
            help="Come il contatore raggruppa le ore per la misura dei consumi. Distinto dal meccanismo di prezzo (PUN variabile, fisso a fasce, ecc.).",
        )

    # Riga 5: dati annui da bolletta (sola lettura)
    _consumo_annuo = _best_meta.get("consumo_annuo_kwh")
    _spesa_annua   = _best_meta.get("spesa_annua_eur")
    if _consumo_annuo or _spesa_annua:
        _info_parts = []
        if _consumo_annuo:
            _info_parts.append(f"Consumo annuo: **{_consumo_annuo:,.0f} kWh**")
        if _spesa_annua:
            _info_parts.append(f"Spesa annua: **{_spesa_annua:,.0f} €**")
        st.caption("📊 Da bolletta (sola lettura) — " + " · ".join(_info_parts))

    # Costruisci griglia 12 mesi (usata da 1.3, 1.4 e Section 3)
    month_grid = _build_month_grid(ok_bills)

    # Inizializzati qui: garantiscono fallback sicuri se 1.3/1.4 non eseguono
    user_edits:    dict      = {}
    pk_edits:      dict      = {}
    effective_model: list    = []

    import math as _math
    import pandas as _pd

    def _valid(v):
        return v is not None and not (isinstance(v, float) and _math.isnan(v))

    _n_ok          = len(ok_bills)
    _reset_count   = st.session_state.get(f"_reset_{_n_ok}", 0)
    _editor_key    = f"monthly_editor_{_n_ok}_{_reset_count}"
    _edits_key     = f"user_edits_{_n_ok}"
    _months_labels = [r["label"] for r in month_grid]

    # 1.3 — Assorbimento mensile (sempre 12 mesi)
    st.markdown("#### 1.3 · Assorbimento mensile da rete")
    if ok_bills:
        # Placeholder per il grafico — riempito dopo l'editor così grafico e totali
        # riflettono le modifiche nello stesso render senza st.rerun()
        _chart_ph = st.empty()

        # Tabella editabile: 3 righe × 12 colonne (F1, F2, F3)
        # Passare sempre il parser puro come data=; Streamlit usa il suo stato interno
        # per gli edit. Il valore di ritorno (edited_df) contiene lo stato corrente.
        _parser_df = _pd.DataFrame(
            {r["label"]: [r.get("f1_kwh"), r.get("f2_kwh"), r.get("f3_kwh")]
             for r in month_grid},
            index=["F1 kWh", "F2 kWh", "F3 kWh"],
        )
        edited_df = st.data_editor(
            _parser_df,
            column_config={
                lbl: st.column_config.NumberColumn(lbl, min_value=0, max_value=10_000_000, format="%d")
                for lbl in _months_labels
            },
            use_container_width=True,
            key=_editor_key,
        )

        # Rileva modifiche confrontando il ritorno del widget col parser
        for _idx, _fk in enumerate(["f1_kwh", "f2_kwh", "f3_kwh"]):
            for _r in month_grid:
                _lbl = _r["label"]
                if _lbl not in edited_df.columns:
                    continue
                try:
                    _ev = edited_df[_lbl].iloc[_idx]
                except Exception:
                    continue
                _pv = _r.get(_fk)
                if _valid(_ev) and (_pv is None or abs(float(_ev) - float(_pv)) > 0.5):
                    user_edits.setdefault(_lbl, {})[_fk] = float(_ev)
        st.session_state[_edits_key] = user_edits

        # Placeholder per riga kWh tot — riempito dopo build di effective_model
        _total_ph = st.empty()

        _col_reset, _col_cap = st.columns([2, 5])
        with _col_reset:
            if st.button("↺ Reset dati parser", key="btn_reset_grid"):
                st.session_state[f"_reset_{_n_ok}"] = _reset_count + 1
                st.session_state.pop(_edits_key, None)
                st.rerun()
        with _col_cap:
            n_obs    = sum(1 for r in month_grid if r["source"] == "observed")
            n_est    = sum(1 for r in month_grid if r["source"] == "estimated")
            n_edited = sum(len(v) for v in user_edits.values())
            edit_note = f" · ✏️ {n_edited} valori corretti manualmente" if n_edited else ""
            st.caption(
                f"🟢 {n_obs} mesi da bolletta · 🟡 {n_est} stimati{edit_note} · "
                "F1/F2/F3 modificabili — kWh tot si ricalcola automaticamente"
            )
    else:
        st.info(
            "Carica almeno una bolletta per vedere l'assorbimento mensile. "
            "Puoi inserire il consumo annuo manualmente nella sezione sottostante."
        )
        consumo_manuale = st.number_input(
            "Consumo annuo stimato (kWh)", min_value=0, max_value=10_000_000,
            value=0, step=1000, key="intake_consumo_manual",
        )
        st.caption("⚪ stimato — inserimento manuale")

    # 1.4 — Picchi mensili (sempre 12 mesi, barre solo dove osservato)
    any_picco = any(r["picco_kw"] is not None for r in month_grid)
    if any_picco:
        st.markdown("#### 1.4 · Picchi mensili di potenza")

        _pk_reset_count = st.session_state.get(f"_pk_reset_{_n_ok}", 0)
        _pk_editor_key  = f"peak_editor_{_n_ok}_{_pk_reset_count}"
        _pk_edits_key   = f"peak_edits_{_n_ok}"

        # Placeholder: il grafico viene riempito dopo la tabella
        _pk_chart_ph = st.empty()

        # Tabella editabile: 1 riga × 12 colonne (Picco kW)
        _pk_parser_df = _pd.DataFrame(
            {r["label"]: [r.get("picco_kw")] for r in month_grid},
            index=["Picco kW"],
        )
        pk_edited_df = st.data_editor(
            _pk_parser_df,
            column_config={
                lbl: st.column_config.NumberColumn(lbl, min_value=0, max_value=5000, format="%.1f")
                for lbl in _months_labels
            },
            use_container_width=True,
            key=_pk_editor_key,
        )

        # Rileva modifiche rispetto al parser
        for _r in month_grid:
            _lbl = _r["label"]
            if _lbl not in pk_edited_df.columns:
                continue
            try:
                _ev = pk_edited_df[_lbl].iloc[0]
            except Exception:
                continue
            _pv = _r.get("picco_kw")
            if _valid(_ev):
                _new_val = float(_ev)
                if _pv is None or abs(_new_val - (_pv or 0)) > 0.05:
                    pk_edits[_lbl] = _new_val
        st.session_state[_pk_edits_key] = pk_edits

        # Reset + legenda
        _col_pk_r, _col_pk_cap = st.columns([2, 7])
        with _col_pk_r:
            if st.button("↺ Reset picchi", key="btn_reset_peaks"):
                st.session_state[f"_pk_reset_{_n_ok}"] = _pk_reset_count + 1
                st.session_state.pop(_pk_edits_key, None)
                st.rerun()
        with _col_pk_cap:
            _pk_n_edited = len(pk_edits)
            _pk_edit_note = f" · ✏️ {_pk_n_edited} valori corretti" if _pk_n_edited else ""
            st.caption(
                f"⚠️ = il picco supera la potenza contrattuale ({pot_contrattuale:.0f} kW)"
                f"{_pk_edit_note} · Picco kW modificabile"
            )
    elif ok_bills:
        st.caption("⚪ Picchi mensili non disponibili dalle bollette.")

    # Build point: effective_model = unica source of truth (month_grid + correzioni utente)
    effective_model = _build_effective_model(month_grid, user_edits, pk_edits)

    # Riempi i placeholder di 1.3 (chart + totale) e 1.4 (chart picchi)
    if ok_bills:
        _chart_ph.plotly_chart(
            _chart_monthly_energy(effective_model), use_container_width=True
        )
        _total_ph.dataframe(
            _pd.DataFrame(
                {r["label"]: [r["consumo_kwh"]] for r in effective_model},
                index=["kWh tot"],
            ),
            column_config={
                lbl: st.column_config.NumberColumn(lbl, format="%d")
                for lbl in _months_labels
            },
            use_container_width=True,
        )
    if any_picco:
        _pk_chart_ph.plotly_chart(
            _chart_monthly_peaks(effective_model, pot_contrattuale),
            use_container_width=True,
        )

    # 1.5 — Mappa economica bolletta
    if ok_bills:
        st.markdown("#### 1.5 · Mappa economica bolletta")

        # Filtro: solo fatturati con costo_totale non null
        _fat = [b for b in ok_bills
                if b["data"].get("source") == "bill_fatturato"
                and b["data"].get("costo_totale_eur") is not None]
        _econ_bills = _fat if _fat else [b for b in ok_bills
                                          if b["data"].get("costo_totale_eur") is not None]
        _qf_mese = _best_meta.get("quota_fissa_eur_mese")
        _es      = _calc_econ_summary(_econ_bills, _qf_mese, month_grid=effective_model)

        # ── BLOCCO A — Sintesi economica ─────────────────────────────────
        _sfx = " (stim.)" if _es["n"] < 12 else ""
        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            st.metric(f"Spesa annua{_sfx}",
                      f"{_es['tot_ann']:,.0f} €" if _es["tot_ann"] is not None else "—")
            st.caption("Stima annualizzata della spesa totale in bolletta")
        with ec2:
            st.metric(f"Energia{_sfx}",
                      f"{_es['en_ann']:,.0f} €" if _es["en_ann"] is not None else "—")
            st.caption("Energia acquistata dalla rete")
        with ec3:
            st.metric(f"Potenza{_sfx}",
                      f"{_es['pot_ann']:,.0f} €" if _es["pot_ann"] is not None else "—")
            st.caption("Capacità di prelievo e picchi")
        with ec4:
            st.metric(f"Altri costi{_sfx}",
                      f"{_es['res_ann']:,.0f} €" if _es["res_ann"] is not None else "—")
            st.caption("Oneri sistema, imposte, partite varie")
        st.caption(f"{_es['badge']} {_es['badge_text']}")

        # ── BLOCCO B — Dove vanno i soldi ────────────────────────────────
        st.markdown("**Dove vanno i soldi**")
        _fig_bd = _chart_cost_breakdown(_es)
        if _fig_bd:
            st.plotly_chart(_fig_bd, use_container_width=True)
        if _es["pct_en"] is not None:
            _en_v  = f"{_es['en_ann']:,.0f} €"  if _es["en_ann"]  is not None else "—"
            _pot_v = f"{_es['pot_ann']:,.0f} €" if _es["pot_ann"] is not None else "—"
            _res_v = f"{_es['res_ann']:,.0f} €" if _es["res_ann"] is not None else "—"
            st.markdown(
                f"| Voce | €/anno | % | Progetto BESS |\n"
                f"|---|---:|:---:|---|\n"
                f"| ⚡ **Energia** | {_en_v} | {_es['pct_en']:.0f}% |"
                f" 🟢 Autoconsumo FV · shifting tariffario |\n"
                f"| ⚡ **Potenza** | {_pot_v} | {_es['pct_pot']:.0f}% |"
                f" 🟢 Peak shaving abbassa il picco |\n"
                f"| 🔒 **Altri costi** | {_res_v} | {_es['pct_res']:.0f}% |"
                f" 🔴 Non impattabile — oneri, imposte |\n"
            )
        if _qf_mese:
            st.caption(
                f"ℹ️ La quota fissa fornitore ({_qf_mese:.2f} €/mese · "
                f"{_qf_mese * 12:,.0f} €/anno) è inclusa nella voce Energia — "
                "separata nel modello dati."
            )
        if _es["anomalie"]:
            _n_a = len(_es["anomalie"])
            st.warning(
                f"⚠️ {_n_a} mese/i con anomalia economica "
                f"({', '.join(_es['anomalie'][:3])}{'…' if _n_a > 3 else ''}): "
                "i residuali risultano negativi. "
                "Possibile credito, rettifica o parsing incompleto.",
            )

        # ── BLOCCO C — Composizione del costo per kWh ────────────────────
        _consumo_ann_c = (_es["kwh_raw"] * _es["ann_f"]) if _es.get("kwh_raw") and _es.get("ann_f") else 0.0
        _kwh_c = _calc_kwh_breakdown(_es, _best_meta, _consumo_ann_c)
        if _kwh_c:
            st.markdown("**Come si compone il tuo costo per kWh**")
            _cc1, _cc2, _cc3, _cc4 = st.columns(4)
            with _cc1:
                if _kwh_c["prezzo_contrattuale"] is not None:
                    st.metric("Prezzo contrattuale",
                              f"{_kwh_c['prezzo_contrattuale']:.4f} €/kWh")
                    st.caption("🟢 da tariffa — prezzo puro energia")
                else:
                    st.metric("Prezzo contrattuale", "—")
                    st.caption("⚪ non rilevato")
            with _cc2:
                if _kwh_c["costo_medio_energia_kwh"] is not None:
                    st.metric("Costo medio energia",
                              f"{_kwh_c['costo_medio_energia_kwh']:.4f} €/kWh")
                    st.caption("🟡 osservato — Σ costo energia / Σ kWh")
                else:
                    st.metric("Costo medio energia", "—")
                    st.caption("⚪ dati insufficienti")
            with _cc3:
                if _kwh_c["costo_medio_tot_kwh"] is not None:
                    st.metric("Costo totale /kWh",
                              f"{_kwh_c['costo_medio_tot_kwh']:.4f} €/kWh")
                    st.caption("🟡 derivato — Σ totale bolletta / Σ kWh")
                else:
                    st.metric("Costo totale /kWh", "—")
                    st.caption("⚪ dati insufficienti")
            with _cc4:
                if _kwh_c["overhead_kwh"] is not None:
                    st.metric("Overhead strutturale",
                              f"+{_kwh_c['overhead_kwh']:.4f} €/kWh")
                    st.caption("🟡 derivato — costi fissi spalmati sul kWh")
                else:
                    st.metric("Overhead strutturale", "—")
                    st.caption("⚪ dati insufficienti")
            if _kwh_c["note"]:
                st.info(_kwh_c["note"], icon="ℹ️")

        # ── BLOCCO D — Cosa cambia e cosa no ─────────────────────────────
        _nat = _classify_cost_nature(_es, _best_meta, pot_contrattuale)
        if _nat:
            st.markdown("**Cosa può cambiare — e cosa no**")
            _nd1, _nd2, _nd3 = st.columns(3)
            with _nd1:
                st.metric("Varia con i consumi",
                          f"{_nat['variabile']:,.0f} €/anno")
                st.caption(f"**{_nat['pct_variabile']:.0f}% della spesa**")
                st.markdown("**Il BESS agisce qui** — autoconsumo FV e shifting tariffario.")
            with _nd2:
                st.metric("Dipende dalla potenza",
                          f"{_nat['potenza']:,.0f} €/anno")
                st.caption(f"**{_nat['pct_potenza']:.0f}% della spesa**")
                st.markdown("**Il peak shaving agisce qui** — abbassa il picco di prelievo.")
            with _nd3:
                st.metric("Strutturale / quasi fisso",
                          f"{_nat['fisso']:,.0f} €/anno")
                st.caption(f"**{_nat['pct_fisso']:.0f}% della spesa**")
                st.markdown("**Il progetto non incide qui** — resta anche a consumi zero.")
            if _nat.get("nota"):
                st.caption(f"ℹ️ {_nat['nota']}")

        # ── BLOCCO E — Come è costruita questa bolletta ───────────────────
        _has_tariff = any(
            _best_meta.get(k)
            for k in ["f0_eur_kwh", "f1_eur_kwh", "tipo_prezzo",
                      "quota_fissa_eur_mese", "quota_potenza_eur_kw_mese"]
        )
        if _has_tariff or _es["avg_en_kwh"]:
            st.markdown("**Come è costruita questa bolletta — regole del contratto**")
            _render_tariff_row(_best_meta, _es["avg_en_kwh"])
            _f1_t = _best_meta.get("f1_eur_kwh")
            _f2_t = _best_meta.get("f2_eur_kwh")
            _f3_t = _best_meta.get("f3_eur_kwh")
            if _f1_t and _f2_t and _f3_t:
                _diff_pct = (_f1_t - _f3_t) / _f3_t * 100 if _f3_t > 0 else 0
                st.caption(
                    f"Struttura fasce: **F1** (ore picco) {_f1_t:.4f} €/kWh · "
                    f"**F2** (ore intermedie) {_f2_t:.4f} €/kWh · "
                    f"**F3** (notti/festivi) {_f3_t:.4f} €/kWh — "
                    f"differenziale F1/F3: +{_diff_pct:.0f}%"
                )
            elif _f1_t and _f3_t:
                _diff_pct = (_f1_t - _f3_t) / _f3_t * 100 if _f3_t > 0 else 0
                st.caption(
                    f"Struttura fasce: **F1** {_f1_t:.4f} €/kWh · "
                    f"**F3** {_f3_t:.4f} €/kWh — differenziale: +{_diff_pct:.0f}%"
                )

        # ── BLOCCO F — Distribuzione annuale dei costi ───────────────────
        # _sp_def calcolato prima del widget: serve sia al grafico (preletto da session_state)
        # sia al widget (come valore di default). Flag pattern per il reset.
        _sp_def = float(_es["avg_en_kwh"]) if _es["avg_en_kwh"] else 0.095
        _sp_def = max(0.0, min(_sp_def, 0.5))
        if st.session_state.pop("_reset_spread_flag", False):
            st.session_state["intake_spread"] = _sp_def
        elif _es["avg_en_kwh"] and st.session_state.get("intake_spread", 0.095) == 0.095:
            st.session_state["intake_spread"] = _sp_def

        _sp_raw = st.session_state.get("intake_spread")
        _sp_for_chart = float(_sp_raw) if _sp_raw is not None else _sp_def
        _fig_costs = _chart_annual_costs(_econ_bills, effective_model, spread_eur_kwh=_sp_for_chart)
        if _fig_costs:
            st.markdown("**Come si distribuisce la spesa durante l'anno**")
            st.plotly_chart(_fig_costs, use_container_width=True)
            _pattern = _summarize_cost_pattern(_econ_bills)
            if _pattern:
                st.caption(_pattern)

        # ── BLOCCO G — Parametri per la simulazione ───────────────────────
        st.markdown("---")
        st.markdown("**Parametri derivati — base per la simulazione BESS**")
        st.caption(
            "Valori estratti dalla bolletta e usati come input per il motore. "
            "Modifica solo se la tua tariffa reale è diversa da quella osservata."
        )
        _col_sp_in, _col_sp_rst = st.columns([5, 1])
        with _col_sp_in:
            spread_eur_kwh = st.number_input(
                "Prezzo medio energia (€/kWh)",
                min_value=0.0, max_value=0.5, value=_sp_def,
                step=0.005, format="%.3f", key="intake_spread",
            )
        with _col_sp_rst:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            if st.button("↺", key="btn_reset_spread", help="Ripristina valore da bolletta",
                         disabled=not bool(_es["avg_en_kwh"])):
                st.session_state["_reset_spread_flag"] = True
                st.rerun()
        if _es["avg_en_kwh"]:
            _changed = abs(spread_eur_kwh - _sp_def) > 0.0001
            if _changed:
                st.caption(
                    f"✏️ modificato — valore derivato originale: {_sp_def:.4f} €/kWh · "
                    "↺ per ripristinare"
                )
            else:
                st.caption(
                    f"🟡 derivato — Σ costo energia / Σ kWh = {_es['avg_en_kwh']:.4f} €/kWh"
                )
        else:
            st.caption("⚪ nessuna bolletta con costo energia — valore stimato")

        _qp_meta = _best_meta.get("quota_potenza_eur_kw_mese")
        if _qp_meta and 0.5 <= _qp_meta <= 50.0:
            _qp_def = float(_qp_meta)
        elif pot_contrattuale > 0 and _es["pot_raw"] > 0:
            _qp_der = round(_es["pot_raw"] / (pot_contrattuale * _es["n"]), 2)
            _qp_def = float(max(0.5, min(_qp_der, 50.0)))
        else:
            _qp_def = 12.0
        st.session_state["intake_quota"] = _qp_def
        quota_potenza = _qp_def
    else:
        # Nessuna bolletta — solo widget manuali
        _col_sp2, _col_qp2 = st.columns(2)
        with _col_sp2:
            spread_eur_kwh = st.number_input(
                "Prezzo medio energia (€/kWh)", min_value=0.0, max_value=0.5,
                value=0.095, step=0.005, format="%.3f", key="intake_spread",
            )
        with _col_qp2:
            quota_potenza = st.number_input(
                "Quota potenza (€/kW/mese)", min_value=0.0, max_value=50.0,
                value=12.0, step=0.5, key="intake_quota",
            )

    # Variabili di localizzazione: lette da session_state (widget 1.2 — nessuna variabile locale).
    comune           = st.session_state.get("intake_comune",         "") or ""
    _provincia_sigla = st.session_state.get("intake_provincia_sigla", "") or ""
    # Query di geocoding: "Comune, Sigla, Italy" per Nominatim.
    # Test empirici: questa forma è la più precisa — riduce l'errore da ~15 km a <1 km
    # perché la sigla disambigua il comune dalla provincia omonima.
    # Fallback senza sigla se non disponibile.
    if comune.strip() and _provincia_sigla.strip():
        _geo_query = f"{comune.strip()}, {_provincia_sigla.strip()}, Italy"
    elif comune.strip():
        _geo_query = f"{comune.strip()}, Italy"
    else:
        _geo_query = ""

    # ════════════════════════════════════════════════════════════════
    # SEZIONE 2 — FV esistente e ricostruzione consumo reale
    # ════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("## 2 · FV esistente e ricostruzione del consumo reale")

    ha_pv = st.checkbox(
        "Il sito ha un impianto FV esistente", key="intake_ha_pv"
    )

    if ha_pv:
        st.caption("Inserisci i dati dell'impianto FV per stimare la produzione solare del sito tramite PVGIS.")

        col_kwp, col_tilt, col_az = st.columns(3)
        with col_kwp:
            kwp = st.number_input(
                "Potenza FV (kWp)", min_value=0, max_value=5000,
                value=0, step=5, key="intake_kwp",
            )
        with col_tilt:
            tilt = st.slider("Inclinazione (°)", 0, 60, 30, key="intake_tilt")
        with col_az:
            azimuth = st.slider("Azimuth (°) — 0=Sud", -180, 180, 0,
                                key="intake_azimuth")

        col_pvgis_btn, _ = st.columns([2, 5])
        with col_pvgis_btn:
            calc_pvgis = st.button(
                "Calcola produzione FV con PVGIS →",
                key="intake_pvgis_btn",
                disabled=(kwp <= 0 or not _geo_query),
            )
        if kwp <= 0:
            st.caption("Inserisci la potenza FV (kWp) per attivare il calcolo PVGIS.")
        elif not _geo_query:
            st.caption("Inserisci il comune nella sezione 1.2 per attivare il calcolo PVGIS.")

        if calc_pvgis and _geo_query and kwp > 0:
            with st.spinner("Geocoding + PVGIS…"):
                coords = _geocode(_geo_query)
                if coords:
                    lt, ln = coords
                    pv_m = _pvgis_monthly(lt, ln, kwp, tilt, azimuth)
                    if pv_m:
                        st.session_state["intake_pv_monthly"] = pv_m
                        st.session_state["intake_pv_lat"] = lt
                        st.session_state["intake_pv_lon"] = ln
                        st.session_state.pop("_pvgis_geo_failed", None)
                    else:
                        st.session_state["_pvgis_geo_failed"] = _geo_query
                        st.warning("PVGIS non ha restituito dati per questa posizione.")
                else:
                    st.session_state["_pvgis_geo_failed"] = _geo_query

        # Fallback manuale: appare se il geocoding ha fallito
        _geo_failed_query = st.session_state.get("_pvgis_geo_failed")
        if _geo_failed_query:
            st.warning(
                f"Posizione **'{_geo_failed_query}'** non trovata automaticamente.  \n"
                "Scrivi il nome del comune nel campo qui sotto nel formato indicato e riprova."
            )
            _col_geo_in, _col_geo_btn = st.columns([4, 2])
            with _col_geo_in:
                _geo_manual = st.text_input(
                    "Posizione per PVGIS",
                    value=_geo_failed_query,
                    key="intake_geo_manual",
                    placeholder="Es: Vicenza, VI, Italy",
                    help="Formato consigliato: Comune, Sigla provincia, Italy",
                )
            with _col_geo_btn:
                st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
                _retry_pvgis = st.button("Riprova →", key="btn_geo_retry",
                                         disabled=(not _geo_manual.strip() or kwp <= 0))
            if _retry_pvgis and _geo_manual.strip():
                with st.spinner(f"Geocoding '{_geo_manual}'…"):
                    coords = _geocode(_geo_manual.strip())
                if coords:
                    lt, ln = coords
                    pv_m = _pvgis_monthly(lt, ln, kwp, tilt, azimuth)
                    if pv_m:
                        st.session_state["intake_pv_monthly"] = pv_m
                        st.session_state["intake_pv_lat"] = lt
                        st.session_state["intake_pv_lon"] = ln
                        st.session_state.pop("_pvgis_geo_failed", None)
                        st.rerun()
                    else:
                        st.session_state["_pvgis_geo_failed"] = _geo_manual.strip()
                        st.error("PVGIS non ha restituito dati. Prova con un comune vicino o una città più grande.")
                else:
                    st.session_state["_pvgis_geo_failed"] = _geo_manual.strip()
                    st.error(
                        f"'{_geo_manual}' non trovato. "
                        "Prova a scrivere il nome della città più grande della provincia "
                        "(es. 'Treviso, TV, Italy' invece del piccolo comune)."
                    )

        pv_monthly = st.session_state.get("intake_pv_monthly")
        if pv_monthly:

            # ── Correzione per età impianto ──────────────────────────────────
            # Flag reset degradazione (deve avvenire prima del widget)
            if st.session_state.pop("_reset_pv_deg_flag", False):
                st.session_state["intake_pv_deg"] = _PV_DEG_DEFAULT

            _col_eta, _col_deg, _col_deg_rst = st.columns([2, 3, 1])
            with _col_eta:
                _eta_anni = st.number_input(
                    "Età impianto (anni)", min_value=0, max_value=40,
                    value=int(st.session_state.get("intake_pv_eta", 0) or 0),
                    step=1, key="intake_pv_eta",
                )
            with _col_deg:
                _deg_pct = st.number_input(
                    "Degr. annua (%/anno)", min_value=0.0, max_value=2.0,
                    value=float(st.session_state.get("intake_pv_deg", _PV_DEG_DEFAULT)),
                    step=0.05, format="%.2f", key="intake_pv_deg",
                )
            with _col_deg_rst:
                st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
                if st.button("↺", key="btn_reset_pv_deg",
                             help=f"Ripristina default benchmark ({_PV_DEG_DEFAULT}%/anno)"):
                    st.session_state["_reset_pv_deg_flag"] = True
                    st.rerun()

            # Calcolo fattore correttivo — un solo punto, applicato una volta sola
            _riduzione  = min(_eta_anni * _deg_pct / 100, 0.40)   # cap: max 40% riduzione totale
            _fattore    = round(1.0 - _riduzione, 6)
            pv_monthly_raw = pv_monthly                            # PVGIS teorico — solo per caption
            pv_monthly  = [v * _fattore for v in pv_monthly_raw]  # corretto — sovrascrive scope locale

            pv_annual = sum(pv_monthly)

            # STANDBY autoconsumo — da riprendere in fase 2
            # net_kwh      = sum(b["data"].get("consumo_kwh", 0) for b in ok_bills) * 12.0 / len(ok_bills) if ok_bills else float(st.session_state.get("intake_consumo_manual") or 0)
            # self_pct     = st.slider("Percentuale di autoconsumo FV stimata (%)", 10, 80, 35, key="intake_self_pct")
            # self_consumed = pv_annual * self_pct / 100
            # real_load    = net_kwh + self_consumed
            # rc1, rc2, rc3 = st.columns(3)
            # rc1.metric("Prelievo dalla rete", f"{net_kwh:,.0f} kWh/anno")
            # rc2.metric(f"Autoconsumo FV (~{self_pct}%)", f"{self_consumed:,.0f} kWh/anno")
            # rc3.metric("Consumo reale sito", f"{real_load:,.0f} kWh/anno", delta=f"+{self_consumed:,.0f} rispetto alla bolletta")

            # ── KPI produzione ───────────────────────────────────────────────
            _pv_specific = pv_annual / kwp if kwp > 0 else 0
            _best_m_idx  = pv_monthly.index(max(pv_monthly))
            _worst_m_idx = pv_monthly.index(min(pv_monthly))
            _kpi1, _kpi2, _kpi3, _kpi4 = st.columns(4)
            with _kpi1:
                st.metric("Produzione annua", f"{pv_annual:,.0f} kWh/anno")
                if _eta_anni > 0:
                    _pv_teorico = sum(pv_monthly_raw)
                    st.caption(
                        f"🟡 corretto per età · PVGIS teorico: {_pv_teorico:,.0f} kWh/anno · "
                        f"riduzione: −{_riduzione*100:.1f}% "
                        f"({_eta_anni} anni × {_deg_pct:.2f}%/anno)"
                    )
                else:
                    st.caption("🟡 derivato — PVGIS JRC Europa · nessuna correzione età")
            with _kpi2:
                st.metric("Produzione specifica", f"{_pv_specific:,.0f} kWh/kWp/anno")
                st.caption("🟡 derivato — producibilità per kWp installato")
            with _kpi3:
                st.metric("Mese migliore", f"{_MESI_IT[_best_m_idx]} — {pv_monthly[_best_m_idx]:,.0f} kWh")
                st.caption("🟡 mese con maggiore irradiazione")
            with _kpi4:
                st.metric("Mese peggiore", f"{_MESI_IT[_worst_m_idx]} — {pv_monthly[_worst_m_idx]:,.0f} kWh")
                st.caption("🟡 mese con minore irradiazione")

            # ── 2.1 — Produzione mensile FV ──────────────────────────────────
            st.markdown("##### Produzione FV mensile (PVGIS)")
            fig_pv = go.Figure(go.Bar(
                x=_MESI_IT, y=pv_monthly,
                marker_color="#F9A825", name="Produzione FV",
            ))
            fig_pv.update_layout(
                height=200, margin=dict(t=10, b=10, l=50, r=10),
                yaxis_title="kWh/mese", showlegend=False,
            )
            st.plotly_chart(fig_pv, use_container_width=True, key="chart_pv_monthly")

            # Tabella mensile: kWh/mese e kWh/giorno
            _days_pm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            import pandas as _pd
            _df_pv = _pd.DataFrame(
                {_MESI_IT[i]: {"kWh/mese": round(pv_monthly[i]),
                               "kWh/giorno": round(pv_monthly[i] / _days_pm[i], 1)}
                 for i in range(12)}
            )
            st.dataframe(
                _df_pv,
                use_container_width=True,
                column_config={m: st.column_config.NumberColumn(m) for m in _MESI_IT},
            )

            # ── 2.2 — Giorno medio per stagione ──────────────────────────────
            st.markdown("##### Forma giornaliera stimata per stagione")
            st.caption(
                "⚪ stimato — PVGIS fornisce solo totali mensili. "
                "La forma oraria è una campana gaussiana centrata all'ora solare di picco."
            )
            season_sel = st.radio(
                "Stagione", ["Inverno", "Primavera", "Estate", "Autunno"],
                horizontal=True, key="intake_fv_season",
            )
            st.plotly_chart(
                _chart_fv_daily_season(pv_monthly, season_sel),
                use_container_width=True, key="chart_fv_daily",
            )

            # Numeri sintetici per la stagione selezionata
            import math as _math
            _s_months_map = {"Inverno": [11, 0, 1], "Primavera": [2, 3, 4],
                             "Estate":  [5, 6, 7],  "Autunno":   [8, 9, 10]}
            _s_mu_map     = {"Inverno": 12.5, "Primavera": 13.0, "Estate": 13.0, "Autunno": 12.5}
            _s_sigma_map  = {"Inverno": 2.5,  "Primavera": 3.5,  "Estate": 4.5,  "Autunno": 3.0}
            _sm  = _s_months_map[season_sel]
            _mu  = _s_mu_map[season_sel]
            _sig = _s_sigma_map[season_sel]
            _s_daily_kwh = (sum(pv_monthly[m] for m in _sm)
                            / sum(_days_pm[m] for m in _sm))
            _s_raw      = [_math.exp(-0.5 * ((h + 0.5 - _mu) / _sig) ** 2) for h in range(24)]
            _s_raw_sum  = sum(_s_raw)
            _s_peak_kw  = max(_s_raw) / _s_raw_sum * _s_daily_kwh if _s_raw_sum > 0 else 0
            _ora_h      = int(_mu)
            _ora_m      = "30" if _mu % 1 else "00"
            _ora_picco  = f"{_ora_h:02d}:{_ora_m}"
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Energia giorno medio", f"{_s_daily_kwh:.1f} kWh/giorno")
            sc2.metric("Ora di picco stimata", _ora_picco)
            sc3.metric("Potenza di picco", f"{_s_peak_kw:.2f} kW")
            st.caption("⚪ stimato — forma sintetica da totali mensili PVGIS")

            # ── Sezione 2.3: Ricostruzione del consumo del sito ─────────────
            _uc_for_s2 = st.session_state.get("intake_user_constraints")
            _s2_key = _s2_section_cache_key(
                kwp, tilt, azimuth, pv_monthly, _uc_for_s2, effective_model
            )
            _s2_cached = st.session_state.get("_s2_recon_cache") or {}
            if _s2_cached.get("key") != _s2_key:
                with st.spinner("Ricostruzione consumo del sito…"):
                    _s2_data = _build_section_2_reconstruction(
                        effective_model, pv_monthly, kwp, tilt, azimuth, _uc_for_s2
                    )
                st.session_state["_s2_recon_cache"] = {"key": _s2_key, "data": _s2_data}
            _recon = st.session_state["_s2_recon_cache"]["data"]

            st.divider()
            st.markdown("### Ricostruzione del consumo del sito")
            st.caption(
                "Combinando i dati di bolletta con la produzione FV, il modello stima quanta "
                "energia il sito consuma realmente, quanta arriva dalla rete e quanta dal FV."
            )

            if _recon is None:
                st.info(
                    "Per visualizzare la ricostruzione, completa la sezione 1 (bollette) "
                    "e calcola la produzione FV qui sopra."
                )
            elif "error" in _recon:
                st.error(f"Errore nella ricostruzione: {_recon['error']}")
            else:
                # ── 2.3.1 Frase narrativa ────────────────────────────────────
                _r_fabb   = _recon["fabbisogno_annuo"]
                _r_prel   = sum(_recon["prelievo_mensile"])
                _r_ac     = _recon["autoconsumo_annuo"]
                _r_fv_tot = sum(_recon["fv_mensile"])
                _r_surp   = _recon["surplus_annuo"]
                st.markdown(
                    f"**Il sito consuma {_r_fabb:,.0f} kWh/anno.** "
                    f"Di questi, {_r_prel:,.0f} kWh vengono acquistati dalla rete (bolletta) "
                    f"e {_r_ac:,.0f} kWh sono autoconsumati direttamente dal FV. "
                    f"Il FV produce {_r_fv_tot:,.0f} kWh/anno: {_r_ac:,.0f} consumati in loco, "
                    f"{_r_surp:,.0f} ceduti alla rete come surplus."
                )

                # ── 2.3.2 KPI principali ─────────────────────────────────────
                st.markdown("")
                _kp1, _kp2, _kp3 = st.columns(3)
                _ac_src = _recon.get("autoconsumo_source", "estimated")
                if _ac_src == "user_input":
                    _ac_dot, _ac_lbl = "🟢", "dichiarato dall'utente"
                else:
                    _ac_dot, _ac_lbl = "🟡", "stimato (modello calibrato)"

                with _kp1:
                    st.metric("Fabbisogno annuo sito", f"{_recon['fabbisogno_annuo']:,.0f} kWh")
                    st.caption("Energia totale consumata dal sito in un anno.")
                    st.caption("🟡 derivato (bolletta + FV)")
                with _kp2:
                    st.metric("Autoconsumo FV", f"{_recon['autoconsumo_annuo']:,.0f} kWh")
                    st.caption("Energia FV consumata in loco, senza passare dalla rete.")
                    st.caption(f"{_ac_dot} {_ac_lbl}")
                with _kp3:
                    st.metric("Surplus FV (rete)", f"{_recon['surplus_annuo']:,.0f} kWh")
                    st.caption("Energia FV prodotta in eccesso, ceduta alla rete.")
                    st.caption(f"{_ac_dot} {_ac_lbl}")

                # ── 2.3.3 KPI secondari ──────────────────────────────────────
                st.markdown("")
                _ks1, _ks2, _ks3 = st.columns(3)
                with _ks1:
                    st.metric("Autosufficienza energetica", f"{_recon['autosufficienza_pct']:.1f}%")
                    st.caption("Quota del fabbisogno coperta direttamente dal FV.")
                    st.caption("🟡 derivato")
                with _ks2:
                    st.metric("Autoconsumo FV %", f"{_recon['autoconsumo_pct']:.1f}%")
                    st.caption("Quota del FV prodotto consumata in loco.")
                    st.caption(f"{_ac_dot} {_ac_lbl}")
                with _ks3:
                    _spread_disp = _recon.get("spread", 0.095)
                    st.metric("Costo evitato dal FV", f"{_recon['costo_evitato_eur']:,.0f} €/anno")
                    st.caption("Risparmio annuo grazie all'autoconsumo FV.")
                    st.caption(f"🟡 derivato (×{_spread_disp:.3f} €/kWh)")

                # ── 2.3.4 Grafico mensile ────────────────────────────────────
                st.markdown("")
                st.markdown("**Bilancio energetico mensile**")
                st.caption(
                    "Click sulla legenda per nascondere/mostrare le serie. "
                    "Le barre sono impilate; le linee sono overlay (nascoste di default)."
                )
                _mesi_s2 = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu",
                            "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]
                _fig_s2 = go.Figure()
                _fig_s2.add_trace(go.Bar(
                    x=_mesi_s2, y=_recon["prelievo_mensile"],
                    name="Prelievo da rete", marker_color="#2E7D32",
                    hovertemplate="%{y:,.0f} kWh<extra></extra>",
                ))
                _fig_s2.add_trace(go.Bar(
                    x=_mesi_s2, y=_recon["autoconsumo_mensile"],
                    name="Autoconsumo FV", marker_color="#7CB342",
                    hovertemplate="%{y:,.0f} kWh<extra></extra>",
                ))
                _fig_s2.add_trace(go.Bar(
                    x=_mesi_s2, y=_recon["surplus_mensile"],
                    name="Surplus FV", marker_color="#F9A825",
                    hovertemplate="%{y:,.0f} kWh<extra></extra>",
                ))
                _fig_s2.add_trace(go.Scatter(
                    x=_mesi_s2, y=_recon["fv_mensile"],
                    name="FV totale prodotto", mode="lines+markers",
                    line=dict(color="#F9A825", width=2, dash="dash"),
                    marker=dict(size=6), visible="legendonly",
                    hovertemplate="FV: %{y:,.0f} kWh<extra></extra>",
                ))
                _fig_s2.add_trace(go.Scatter(
                    x=_mesi_s2, y=_recon["fabbisogno_mensile"],
                    name="Fabbisogno totale", mode="lines+markers",
                    line=dict(color="#1565C0", width=2, dash="dash"),
                    marker=dict(size=6), visible="legendonly",
                    hovertemplate="Fabbisogno: %{y:,.0f} kWh<extra></extra>",
                ))
                if any(p > 0 for p in _recon["picchi_mensili"]):
                    _fig_s2.add_trace(go.Scatter(
                        x=_mesi_s2, y=_recon["picchi_mensili"],
                        name="Picchi mensili (kW)", mode="lines+markers",
                        line=dict(color="#C62828", width=2),
                        marker=dict(size=6), yaxis="y2", visible="legendonly",
                        hovertemplate="Picco: %{y:.1f} kW<extra></extra>",
                    ))
                _fig_s2.update_layout(
                    barmode="stack", height=420,
                    margin=dict(l=10, r=10, t=10, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    yaxis=dict(title="kWh"),
                    yaxis2=dict(title="kW (picchi)", overlaying="y", side="right", showgrid=False),
                    hovermode="x unified",
                )
                st.plotly_chart(_fig_s2, use_container_width=True)

                # ── 2.3.5 Tabella mensile ────────────────────────────────────
                st.markdown("**Dettaglio mensile**")
                import pandas as _pd_s2
                _df_s2 = _pd_s2.DataFrame({
                    "Mese": _mesi_s2,
                    "Fabbisogno (kWh)":    [round(v) for v in _recon["fabbisogno_mensile"]],
                    "Prelievo rete (kWh)": [round(v) for v in _recon["prelievo_mensile"]],
                    "FV prodotto (kWh)":   [round(v) for v in _recon["fv_mensile"]],
                    "Autoconsumo (kWh)":   [round(v) for v in _recon["autoconsumo_mensile"]],
                    "Surplus (kWh)":       [round(v) for v in _recon["surplus_mensile"]],
                    "% AC del FV": [
                        f"{(ac / fv * 100):.0f}%" if fv > 0 else "—"
                        for ac, fv in zip(_recon["autoconsumo_mensile"], _recon["fv_mensile"])
                    ],
                })
                st.dataframe(_df_s2, hide_index=True, use_container_width=True)

                # ── 2.3.6 Expander dettagli tecnici ─────────────────────────
                with st.expander("Dettagli tecnici · qualità del modello", expanded=False):
                    _conf = _recon.get("overall_confidence", "medium")
                    _conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(_conf, "⚪")
                    st.markdown(f"**Confidenza globale**: {_conf_emoji} {_conf}")
                    _bme = _recon.get("band_match_error_pct") or {}
                    if _bme:
                        _errs_str = " · ".join(
                            f"{k.upper()}: {abs(v):.1f}%" for k, v in _bme.items()
                        )
                        _avg_err = sum(abs(v) for v in _bme.values()) / len(_bme)
                        st.markdown(f"**Errore F1/F2/F3** (medio {_avg_err:.1f}%): {_errs_str}")
                    _mode_s2 = _recon.get("reconcile_mode", "auto")
                    _mode_lbl = {
                        "auto": "Auto — bolletta + FV PVGIS",
                        "constrained_annual": "Vincolata annua — autoconsumo dichiarato",
                        "constrained_monthly": "Vincolata mensile — 12 mesi dichiarati",
                    }.get(_mode_s2, _mode_s2)
                    st.markdown(f"**Modalità modello**: {_mode_lbl}")
                    _assum = _recon.get("assumptions_active") or []
                    if _assum:
                        st.markdown("**Assunzioni attive:**")
                        for _a in _assum:
                            st.caption(f"• {_a}")

            # ── Expander: vincoli utente autoconsumo (Brief 6) ────────────────
            with st.expander("Hai dati specifici sull'autoconsumo?", expanded=False):
                st.caption(
                    "Se il cliente ha un'app PV plant o un report GSE con dati di "
                    "autoconsumo o energia immessa in rete, inseriscili qui per "
                    "migliorare la ricostruzione del modello."
                )
                _uc_type = st.radio(
                    "Tipo di dato disponibile",
                    options=["Nessuno", "% di autoconsumo FV", "kWh immessi in rete"],
                    index=0,
                    horizontal=True,
                    key="uc_type_radio",
                )
                _user_constraints = None

                if _uc_type == "% di autoconsumo FV":
                    _uc_scope = st.radio(
                        "Granularità",
                        options=["Annuo", "Mensile (tutti e 12)"],
                        index=0,
                        horizontal=True,
                        key="uc_pct_scope",
                    )
                    if _uc_scope == "Annuo":
                        _pct_ann = st.number_input(
                            "% autoconsumo annuo",
                            min_value=0.0, max_value=100.0,
                            value=65.0, step=1.0,
                            help="Percentuale di FV prodotto consumato in loco (vs. immesso in rete)",
                            key="uc_pct_ann",
                        )
                        _user_constraints = {
                            "type": "autoconsumo_pct",
                            "scope": "annuo",
                            "annual_value": _pct_ann,
                            "monthly_values": None,
                        }
                    else:
                        st.caption("Inserisci la % di autoconsumo per ciascun mese.")
                        _uc_cols1 = st.columns(6)
                        _uc_cols2 = st.columns(6)
                        _mesi_lab = ["Gen","Feb","Mar","Apr","Mag","Giu",
                                     "Lug","Ago","Set","Ott","Nov","Dic"]
                        _pct_monthly = []
                        for _m in range(12):
                            _col = _uc_cols1[_m % 6] if _m < 6 else _uc_cols2[_m % 6]
                            with _col:
                                _pct_monthly.append(st.number_input(
                                    _mesi_lab[_m], min_value=0.0, max_value=100.0,
                                    value=65.0, step=1.0, key=f"uc_pct_m{_m}",
                                ))
                        _user_constraints = {
                            "type": "autoconsumo_pct",
                            "scope": "mensile",
                            "annual_value": None,
                            "monthly_values": _pct_monthly,
                        }

                elif _uc_type == "kWh immessi in rete":
                    _uc_scope2 = st.radio(
                        "Granularità",
                        options=["Annuo", "Mensile (tutti e 12)"],
                        index=0,
                        horizontal=True,
                        key="uc_sup_scope",
                    )
                    if _uc_scope2 == "Annuo":
                        _sup_ann = st.number_input(
                            "kWh immessi in rete (annuo)",
                            min_value=0.0, max_value=1_000_000.0,
                            value=20_000.0, step=1000.0,
                            help="Energia FV ceduta alla rete (non autoconsumata)",
                            key="uc_sup_ann",
                        )
                        _user_constraints = {
                            "type": "surplus_kwh",
                            "scope": "annuo",
                            "annual_value": _sup_ann,
                            "monthly_values": None,
                        }
                    else:
                        st.caption("Inserisci i kWh immessi in rete per ciascun mese.")
                        _uc_cols3 = st.columns(6)
                        _uc_cols4 = st.columns(6)
                        _sup_monthly = []
                        for _m in range(12):
                            _col = _uc_cols3[_m % 6] if _m < 6 else _uc_cols4[_m % 6]
                            with _col:
                                _sup_monthly.append(st.number_input(
                                    f"{_mesi_lab[_m]} kWh",
                                    min_value=0.0, max_value=200_000.0,
                                    value=2_000.0, step=100.0,
                                    key=f"uc_sup_m{_m}",
                                ))
                        _user_constraints = {
                            "type": "surplus_kwh",
                            "scope": "mensile",
                            "annual_value": None,
                            "monthly_values": _sup_monthly,
                        }

                if _user_constraints:
                    st.session_state["intake_user_constraints"] = _user_constraints
                else:
                    st.session_state.pop("intake_user_constraints", None)

                st.markdown("")
                if st.button("✓ Applica vincolo", key="btn_apply_uc", type="primary"):
                    st.rerun()

        else:
            st.info(
                "Inserisci la potenza FV e clicca 'Calcola produzione FV con PVGIS →' "
                "per stimare la produzione solare del sito."
            )
    else:
        st.caption(
            "⚪ Sezione opzionale — attiva se il sito ha un impianto FV. "
            "Senza FV, il prelievo dalla rete coincide con il consumo reale."
        )

    # ════════════════════════════════════════════════════════════════
    # SEZIONE 3 — Allocazione temporale del carico
    # ════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("## 3 · Profilo di carico del sito")
    st.caption(
        "Questa sezione alloca temporalmente l'energia ricostruita nelle sezioni precedenti. "
        "**Non ridefinisce il volume** (locked da bollette + FV) — definisce *quando* viene consumata."
    )

    # ── 3.1 Struttura turni (parametri PRIMARI) ──────────────────────────────
    st.markdown("#### 3.1 · Struttura turni")
    col_t1, col_t2 = st.columns([2, 3])
    with col_t1:
        turni_sel = st.radio(
            "Struttura turni",
            ["Singolo turno", "Due turni", "Continuo 24h"],
            key="intake_turni",
            help="Definisce il template di attività giornaliero.",
        )
    with col_t2:
        if turni_sel == "Singolo turno":
            hc1, hc2 = st.columns(2)
            with hc1:
                ora_ini = st.selectbox(
                    "Inizio turno", list(range(24)), index=7,
                    format_func=lambda x: f"{x:02d}:00", key="intake_ora_ini",
                )
            with hc2:
                ora_fine_val = st.selectbox(
                    "Fine turno", list(range(1, 25)), index=10,
                    format_func=lambda x: f"{x:02d}:00" if x < 24 else "24:00",
                    key="intake_ora_fine",
                )
            t1_ini, t1_fine = ora_ini, ora_fine_val
            t2_ini, t2_fine = None, None

        elif turni_sel == "Due turni":
            hc1, hc2, hc3, hc4 = st.columns(4)
            with hc1:
                t1_ini = st.selectbox(
                    "T1 inizio", list(range(24)), index=5,
                    format_func=lambda x: f"{x:02d}:00", key="intake_t1_ini",
                )
            with hc2:
                t1_fine = st.selectbox(
                    "T1 fine", list(range(1, 25)), index=13,
                    format_func=lambda x: f"{x:02d}:00" if x < 24 else "24:00",
                    key="intake_t1_fine",
                )
            with hc3:
                t2_ini = st.selectbox(
                    "T2 inizio", list(range(24)), index=13,
                    format_func=lambda x: f"{x:02d}:00", key="intake_t2_ini",
                )
            with hc4:
                t2_fine = st.selectbox(
                    "T2 fine", list(range(1, 25)), index=21,
                    format_func=lambda x: f"{x:02d}:00" if x < 24 else "24:00",
                    key="intake_t2_fine",
                )
            ora_ini      = min(t1_ini, t2_ini)
            ora_fine_val = max(t1_fine, t2_fine)

        else:  # Continuo 24h
            st.info("Impianto attivo 24h/giorno — nessun orario da specificare.", icon="ℹ️")
            ora_ini, ora_fine_val = 0, 24
            t1_ini, t1_fine = 0, 24
            t2_ini, t2_fine = None, None

    # ── 3.2 Giorni e carico base (parametri PRIMARI) ─────────────────────────
    st.markdown("#### 3.2 · Giorni di attività e carico base")
    col_g, col_b = st.columns([3, 2])
    with col_g:
        giorni_sel = st.multiselect(
            "Giorni lavorativi", _DOW_LABELS_FULL,
            default=_DOW_LABELS_FULL[:5], key="intake_giorni",
        )
        giorni_idx = [_DOW_LABELS_FULL.index(g) for g in giorni_sel]
    with col_b:
        carico_base = st.slider(
            "Carico base fuori orario (%)", 0, 50, 15, key="intake_base_load",
            help="Consumi minimi fuori turno: guardiania, HVAC, server, linee in idle.",
        )

    # ── 3.3 Stagionalità (parametri SECONDARI) ───────────────────────────────
    st.markdown("#### 3.3 · Variazione stagionale")
    st.caption(
        "Raffina la distribuzione annuale del volume energetico tra le stagioni. "
        "Non modifica il totale annuo."
    )
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        impatto_inv = st.radio(
            "Inverno rispetto alla media",
            ["Aumenta", "Neutro", "Riduce"],
            index=1, horizontal=True, key="intake_inv",
        )
    with col_s2:
        impatto_est = st.radio(
            "Estate rispetto alla media",
            ["Aumenta", "Neutro", "Riduce"],
            index=2, horizontal=True, key="intake_est",
        )

    # ── Volume annuo base per la normalizzazione ──────────────────────────────
    # Section 3 alloca questo volume, non lo ridefinisce.
    pv_monthly_s3  = st.session_state.get("intake_pv_monthly")
    ha_pv_s3       = st.session_state.get("intake_ha_pv", False)
    _obs_eff = [r for r in effective_model if r["consumo_source"] in ("observed", "edited")]
    if _obs_eff:
        prelievo_rete_kwh = sum(r["consumo_kwh"] or 0 for r in _obs_eff) * 12.0 / len(_obs_eff)
    else:
        prelievo_rete_kwh = float(st.session_state.get("intake_consumo_manual") or 100_000)

    if ha_pv_s3 and pv_monthly_s3:
        autoconsumo_fv = sum(pv_monthly_s3) * 0.35
        consumo_chart  = prelievo_rete_kwh + autoconsumo_fv
    else:
        consumo_chart = prelievo_rete_kwh

    # ── 3.4 Profilo settimanale per stagione ─────────────────────────────────
    st.markdown("#### 3.4 · Profilo settimanale per stagione")
    st.caption(
        f"⚪ stimato — settimana tipo da turni + stagionalità, "
        f"normalizzata su **{consumo_chart:,.0f} kWh/anno** "
        f"{'(prelievo rete + autoconsumo FV)' if ha_pv_s3 and pv_monthly_s3 else '(da bollette)'}"
    )

    preview_params = {
        "turni":             turni_sel,
        "ora_inizio":        t1_ini,
        "ora_fine":          t1_fine,
        "turno2_ini":        t2_ini,
        "turno2_fine":       t2_fine,
        "giorni_attivi":     giorni_idx,
        "carico_base_pct":   carico_base,
        "consumo_annuo_kwh": consumo_chart,
        "impatto_invernale": impatto_inv,
        "impatto_estivo":    impatto_est,
    }

    tab_inv, tab_pri, tab_est, tab_aut = st.tabs(
        ["Inverno", "Primavera", "Estate", "Autunno"]
    )
    with tab_inv:
        st.plotly_chart(_make_weekly_preview(preview_params, "inverno"),
                        use_container_width=True, key="weekly_chart_inverno")
    with tab_pri:
        st.plotly_chart(_make_weekly_preview(preview_params, "primavera"),
                        use_container_width=True, key="weekly_chart_primavera")
    with tab_est:
        st.plotly_chart(_make_weekly_preview(preview_params, "estate"),
                        use_container_width=True, key="weekly_chart_estate")
    with tab_aut:
        st.plotly_chart(_make_weekly_preview(preview_params, "autunno"),
                        use_container_width=True, key="weekly_chart_autunno")

    # ── 3.5 Coerenza profilo ─────────────────────────────────────────────────
    st.markdown("#### 3.5 · Coerenza profilo")

    # Calcola distribuzione F1/F2/F3 dal profilo settimanale sintetico (primavera)
    _factor_map_s3 = {"Aumenta": 1.2, "Neutro": 1.0, "Riduce": 0.8}
    windows_s3 = []
    if turni_sel == "Continuo 24h":
        windows_s3 = [(0, 96)]
    elif turni_sel == "Due turni":
        windows_s3 = [(t1_ini * 4, t1_fine * 4), (t2_ini * 4, t2_fine * 4)]
    else:
        windows_s3 = [(t1_ini * 4, min(t1_fine * 4, 96))]

    day_prof_s3 = np.ones(96) * (carico_base / 100.0)
    for ws, we in windows_s3:
        day_prof_s3[ws:we] = 1.0
    weekly_s3 = np.zeros(672)
    for day in range(7):
        if day in giorni_idx:
            weekly_s3[day * 96:(day + 1) * 96] = day_prof_s3
        else:
            weekly_s3[day * 96:(day + 1) * 96] = carico_base / 100.0
    wk_sum = weekly_s3.sum() * SLOT_H
    if wk_sum > 0:
        weekly_s3 *= (consumo_chart / 52) / wk_sum

    kwh_f1 = kwh_f2 = kwh_f3 = 0.0
    for day in range(7):
        for slot in range(96):
            fb = _fband_simple(day, slot)
            kw = weekly_s3[day * 96 + slot]
            if fb == "F1":
                kwh_f1 += kw * SLOT_H
            elif fb == "F2":
                kwh_f2 += kw * SLOT_H
            else:
                kwh_f3 += kw * SLOT_H
    ft_mod = max(kwh_f1 + kwh_f2 + kwh_f3, 0.001)
    peak_mod = float(weekly_s3.max())

    col_coh1, col_coh2 = st.columns(2)
    with col_coh1:
        st.metric("Picco implicito del profilo", f"{peak_mod:.0f} kW",
                  help="Derivato dalla forma del profilo — non un input diretto.")
        if pot_contrattuale > 0 and peak_mod > pot_contrattuale * 1.05:
            st.warning(
                f"⚠️ Il profilo implica un picco di {peak_mod:.0f} kW "
                f"— superiore alla potenza contrattuale ({pot_contrattuale:.0f} kW). "
                "Verifica i turni o la potenza contrattuale.",
                icon="⚠️",
            )
    with col_coh2:
        st.markdown(
            f"**Distribuzione F1/F2/F3 modellata** (primavera, settimana tipo):  \n"
            f"F1 {kwh_f1/ft_mod*100:.0f}% · F2 {kwh_f2/ft_mod*100:.0f}% · F3 {kwh_f3/ft_mod*100:.0f}%"
        )
        if ok_bills:
            bill_f1 = sum(b["data"].get("f1_kwh") or 0 for b in ok_bills)
            bill_f2 = sum(b["data"].get("f2_kwh") or 0 for b in ok_bills)
            bill_f3 = sum(b["data"].get("f3_kwh") or 0 for b in ok_bills)
            bill_ft = bill_f1 + bill_f2 + bill_f3
            if bill_ft > 0:
                st.markdown(
                    f"**Da bolletta:** "
                    f"F1 {bill_f1/bill_ft*100:.0f}% · "
                    f"F2 {bill_f2/bill_ft*100:.0f}% · "
                    f"F3 {bill_f3/bill_ft*100:.0f}%"
                )
        else:
            st.caption("⚪ Nessuna bolletta per confronto fasce.")

    # ════════════════════════════════════════════════════════════════
    # FOOTER — Genera diagnosi
    # ════════════════════════════════════════════════════════════════
    st.divider()
    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        genera = st.button(
            "Genera diagnosi energetica →",
            type="primary",
            use_container_width=True,
        )
    with col_info:
        st.caption(
            "**Block 1** · Radiografia energetica del sito. "
            "La simulazione BESS viene dopo, dalla pagina diagnostica."
        )

    if not genera:
        return

    # ── Validazione ──────────────────────────────────────────────────────────
    has_consumo = (ok_bills or
                   (st.session_state.get("intake_consumo_manual") or 0) > 0)
    if not has_consumo:
        st.error("Carica almeno una bolletta o inserisci il consumo annuo manualmente.")
        return
    if not _geo_query:
        st.warning("Inserisci il comune nella sezione 1.2 per il geocoding.")
        return

    # ── Geocoding ────────────────────────────────────────────────────────────
    lat = st.session_state.get("intake_pv_lat")
    lon = st.session_state.get("intake_pv_lon")
    if not lat or not lon:
        with st.spinner(f"Geocoding '{_geo_query}'…"):
            coords = _geocode(_geo_query)
        if coords:
            lat_c, lon_c = coords
            if _IT_LAT[0] <= lat_c <= _IT_LAT[1] and _IT_LON[0] <= lon_c <= _IT_LON[1]:
                lat, lon = lat_c, lon_c
            else:
                st.warning(
                    f"Coordinate per '{_geo_query}' fuori dall'Italia. "
                    "PVGIS userà default Nord Italia."
                )

    # ── Costruisci intake_form ────────────────────────────────────────────────
    bills_data   = [b["data"] for b in ok_bills]
    consumo_ann  = int(round(consumo_chart))
    _eff_picchi  = [r["picco_kw"] for r in effective_model
                    if r["picco_source"] in ("observed", "edited") and r["picco_kw"] is not None]
    picco_est    = (int(round(max(_eff_picchi))) if _eff_picchi else
                    (int(round(pot_contrattuale * 0.9)) if pot_contrattuale > 0 else 0))
    kwp_pv       = int(st.session_state.get("intake_kwp", 0)) if ha_pv else 0

    def _safe_sum(key):
        v = sum(b.get(key) or 0 for b in bills_data)
        return v if v > 0 else None

    mensili_effettivi = [
        {
            "mese":        r.get("mese_idx"),
            "consumo_kwh": r.get("consumo_kwh"),
            "f1_kwh":      r.get("f1_kwh"),
            "f2_kwh":      r.get("f2_kwh"),
            "f3_kwh":      r.get("f3_kwh"),
            "picco_kw":    r.get("picco_kw") if r.get("picco_source") in ("observed", "edited") else None,
        }
        for r in (effective_model or [])
        if r.get("consumo_kwh") is not None and r.get("mese_idx") is not None
    ]

    _pv_eta_f  = int(st.session_state.get("intake_pv_eta", 0) or 0) if ha_pv else 0
    _pv_deg_f  = float(st.session_state.get("intake_pv_deg", _PV_DEG_DEFAULT)) if ha_pv else 0.0
    _pv_factor = round(1.0 - min(_pv_eta_f * _pv_deg_f / 100, 0.40), 6)

    intake_form = {
        "nome_cliente":              nome_cliente.strip() or "Sito",
        "comune":                    comune.strip(),
        "consumo_annuo_kwh":         consumo_ann,
        "picco_potenza_kw":          picco_est,
        "potenza_contrattuale_kw":   pot_contrattuale,
        "ha_pv":                     ha_pv,
        "kwp":                       kwp_pv,
        "kwp_proposto":              0,
        "costo_eur_kwp":             0,
        "ore_lavoro_giorno":         max(1, ora_fine_val - ora_ini),
        "giorni_lavoro_settimana":   max(1, len([d for d in giorni_idx if d < 5])),
        "spread_eur_kwh":            float(st.session_state.get("intake_spread", 0.095)),
        "quota_potenza_eur_kw_mese": float(st.session_state.get("intake_quota", 12.0)),
        "lat":                       lat,
        "lon":                       lon,
        "bills":                     bills_data,
        "market_price_series":       "prices/it_nord_2024.json",
        # Campi legacy per engine e build_case
        "location":                  comune.strip(),
        "goal":                      "Ridurre la bolletta elettrica",
        "t5_enable":                 False,
        "t5_aliquota_pct":           0,
        "consumo_kwh_periodo":       bills_data[0].get("consumo_kwh") if bills_data else consumo_ann / 12,
        "costo_totale_eur":          _safe_sum("costo_totale_eur"),
        "costo_energia_eur":         _safe_sum("costo_energia_eur"),
        "costo_potenza_eur":         _safe_sum("costo_potenza_eur"),
        "f1_kwh":                    _safe_sum("f1_kwh"),
        "f2_kwh":                    _safe_sum("f2_kwh"),
        "f3_kwh":                    _safe_sum("f3_kwh"),
        "user_constraints":          st.session_state.get("intake_user_constraints"),
        "mensili_effettivi":         mensili_effettivi if mensili_effettivi else None,
        "pv_degradation_factor":     _pv_factor if ha_pv else 1.0,
    }

    # ── Block 1: SDI + Diagnostica ────────────────────────────────────────────
    sdi        = None
    diagnostic = None
    if _BLOCK1_AVAILABLE:
        try:
            with st.spinner("Block 1: costruzione SDI…"):
                sdi = _build_sdi_block1(intake_form, BASE_DIR)
        except Exception as e:
            st.warning(f"Block 1 non disponibile: {e}")

    if _DIAGNOSTIC_AVAILABLE and sdi:
        try:
            diagnostic = _build_diagnostic(sdi, BASE_DIR)
        except Exception as e:
            st.warning(f"Diagnostica non disponibile: {e}")

    st.session_state.update({
        "step":       "diagnostic",
        "form":       intake_form,
        "sdi":        sdi,
        "diagnostic": diagnostic,
        "sim":        None,
        "econ":       None,
        "profiles":   None,
    })
    st.rerun()


def _render_bill_upload() -> None:
    """
    Sezione upload bolletta PDF — esterna al form Streamlit.
    Estrae i campi grezzi della bolletta e li salva in session_state["bill_prefill"].
    I campi vengono poi usati come valori iniziali dei widget nel form sottostante.
    """
    pf = st.session_state.get("bill_prefill", {})

    if not pf:
        st.caption("Hai il PDF della bolletta? Caricalo per compilare i campi automaticamente.")
        col_up, col_btn = st.columns([5, 1])
        with col_up:
            uploaded = st.file_uploader(
                "Bolletta elettrica (PDF)", type=["pdf"], label_visibility="collapsed"
            )
        with col_btn:
            st.markdown("<div style='padding-top:1.55rem'></div>", unsafe_allow_html=True)
            do_parse = st.button("Estrai →", type="primary", disabled=uploaded is None)

        if uploaded and do_parse:
            with st.spinner("Lettura bolletta con Gemini AI..."):
                parsed = _parse_bill_pdf(uploaded.read())
            if "parse_error" in parsed:
                st.error(f"Estrazione fallita: {parsed['parse_error']}")
                return
            new_bills, _sm = _expand_parsed_bill(parsed, file_name=uploaded.name)
            # Per il form legacy usa il periodo "fatturato" (o il primo disponibile)
            fatturati = [b for b in new_bills if b.get("source") == "bill_fatturato"]
            bill = fatturati[0] if fatturati else (new_bills[0] if new_bills else {})

            costo_medio   = bill.get("costo_medio_energia_eur_kwh") or 0
            spread        = round(costo_medio - _IT_NORD_MARKET_AVG, 3) if costo_medio else None
            spread        = max(0.0, min(spread, 0.30)) if spread is not None else None
            pot_contr     = bill.get("potenza_contrattuale_kw") or 0
            costo_pot_eur = bill.get("costo_potenza_eur") or 0
            quota_pot     = round(costo_pot_eur / pot_contr, 2) if pot_contr > 0 else None
            quota_pot     = max(0.0, min(quota_pot, 50.0)) if quota_pot is not None else None

            st.session_state["bill_prefill"] = {
                "periodo":                 bill.get("periodo"),
                "consumo_kwh":             bill.get("consumo_kwh"),
                "potenza_contrattuale_kw": bill.get("potenza_contrattuale_kw"),
                "picco_kw":                bill.get("picco_kw"),
                "costo_totale_eur":        bill.get("costo_totale_eur"),
                "costo_energia_eur":       bill.get("costo_energia_eur"),
                "costo_potenza_eur":       bill.get("costo_potenza_eur"),
                "f1_kwh":                  bill.get("f1_kwh"),
                "f2_kwh":                  bill.get("f2_kwh"),
                "f3_kwh":                  bill.get("f3_kwh"),
                "spread":                  spread,
                "quota_pot":               quota_pot,
            }
            st.session_state["bill_load_count"] = st.session_state.get("bill_load_count", 0) + 1
            st.rerun()
    else:
        col_ok, col_clear = st.columns([5, 1])
        with col_ok:
            st.success(
                f"Bolletta estratta: periodo **{pf.get('periodo', '?')}** · "
                f"Consumo **{(pf.get('consumo_kwh') or 0):,.0f} kWh** · "
                f"Potenza **{pf.get('potenza_contrattuale_kw', '?')} kW** · "
                f"Totale **{(pf.get('costo_totale_eur') or 0):,.2f} €**"
            )
        with col_clear:
            st.markdown("<div style='padding-top:0.4rem'></div>", unsafe_allow_html=True)
            if st.button("Ricarica", type="secondary"):
                st.session_state.pop("bill_prefill", None)
                st.session_state["bill_load_count"] = st.session_state.get("bill_load_count", 0) + 1
                st.rerun()


def _page_input() -> None:
    st.markdown("# 🔋 BESS Decision Tool")
    st.markdown(
        "Inserisci i dati della bolletta del cliente per simulare l'installazione di una batteria "
        "e valutare il business case."
    )
    st.divider()

    # ── Upload bolletta PDF (fuori dal form — Streamlit non permette file_uploader dentro form)
    if _bill_parser_available():
        st.markdown("### 📄 Bolletta elettrica")
        _render_bill_upload()
        st.divider()

    pf = st.session_state.get("bill_prefill", {})
    _form_key = f"input_form_{st.session_state.get('bill_load_count', 0)}"

    # Valori iniziali dai campi estratti (None → default sicuro)
    _consumo_val  = int(round(pf["consumo_kwh"]))       if pf.get("consumo_kwh")             else 0
    _pot_val      = int(round(pf["potenza_contrattuale_kw"])) if pf.get("potenza_contrattuale_kw") else 0
    _costo_val    = float(pf["costo_totale_eur"])        if pf.get("costo_totale_eur")         else 0.0
    _ce_val       = float(pf["costo_energia_eur"])       if pf.get("costo_energia_eur")        else 0.0
    _cp_val       = float(pf["costo_potenza_eur"])       if pf.get("costo_potenza_eur")        else 0.0
    _f1_val       = float(pf["f1_kwh"])                  if pf.get("f1_kwh")                  else 0.0
    _f2_val       = float(pf["f2_kwh"])                  if pf.get("f2_kwh")                  else 0.0
    _f3_val       = float(pf["f3_kwh"])                  if pf.get("f3_kwh")                  else 0.0
    _spread_val   = float(pf["spread"])                  if pf.get("spread") is not None       else 0.095
    _quota_val    = float(pf["quota_pot"])               if pf.get("quota_pot") is not None    else 12.0

    with st.form(_form_key):

        # ── Sezione 1: Dati bolletta ──────────────────────────────────────────
        st.markdown("### 1 — Dati bolletta")
        if not _bill_parser_available():
            st.caption("Inserisci i valori che trovi sulla bolletta elettrica del cliente.")

        col1, col2, col3 = st.columns(3)
        with col1:
            consumo_kwh = st.number_input(
                "Consumo periodo (kWh)",
                min_value=0, max_value=10_000_000, value=_consumo_val, step=100,
                help="Energia consumata nel mese (o trimestre) di fatturazione. "
                     "Il consumo annuo sarà stimato ×12."
            )
        with col2:
            potenza_contrattuale_kw = st.number_input(
                "Potenza contrattuale (kW)",
                min_value=0, max_value=5_000, value=_pot_val, step=5,
                help="Potenza disponibile o contrattuale riportata in bolletta"
            )
        with col3:
            costo_totale_eur = st.number_input(
                "Costo totale bolletta (€)",
                min_value=0.0, max_value=500_000.0, value=_costo_val,
                step=10.0, format="%.2f",
                help="Importo totale della bolletta (utile per derivare le tariffe)"
            )

        with st.expander("Dettaglio costi e fasce orarie (opzionale)"):
            dc1, dc2 = st.columns(2)
            with dc1:
                costo_energia_eur = st.number_input(
                    "Costo energia (€)", min_value=0.0, value=_ce_val,
                    step=10.0, format="%.2f",
                    help="Componente energia della bolletta (escluse quota potenza e oneri)"
                )
                costo_potenza_eur = st.number_input(
                    "Costo potenza (€)", min_value=0.0, value=_cp_val,
                    step=10.0, format="%.2f",
                    help="Quota potenza della bolletta"
                )
            with dc2:
                f1_kwh = st.number_input("F1 kWh (ore di punta, lun-ven 8-19)",
                    min_value=0.0, value=_f1_val, step=100.0)
                f2_kwh = st.number_input("F2 kWh (ore intermedie)",
                    min_value=0.0, value=_f2_val, step=100.0)
                f3_kwh = st.number_input("F3 kWh (fuori punta, notti e festivi)",
                    min_value=0.0, value=_f3_val, step=100.0)

        # ── Sezione 2: Sito e impianto ────────────────────────────────────────
        st.markdown("### 2 — Sito e impianto")
        col4, col5 = st.columns(2)
        with col4:
            nome_cliente = st.text_input(
                "Nome cliente / azienda", placeholder="Es: Acciaierie Nord S.r.l.")
            comune = st.text_input(
                "Comune o indirizzo", placeholder="Es: Castelfranco Veneto, TV",
                help="Usato per stimare la produzione FV con PVGIS")
        with col5:
            ha_pv = st.checkbox("Il sito ha già un impianto FV", value=True)
            kwp = st.number_input(
                "Potenza FV esistente (kWp)", min_value=0, max_value=5_000,
                value=80, step=5,
                help="Potenza del FV già installato. 0 se assente"
            )

        col6, col7 = st.columns(2)
        with col6:
            kwp_proposto = st.number_input(
                "FV da proporre (kWp) — se non ha FV", min_value=0, max_value=5_000,
                value=80, step=5,
                help="Usato solo se il sito non ha FV esistente. "
                     "Genera la valutazione dell'investimento FV (S_FV)."
            )
        with col7:
            costo_eur_kwp = st.number_input(
                "Costo FV chiavi in mano (€/kWp)", min_value=0, max_value=3_000,
                value=800, step=50,
                help="Range tipico C&I italiano 2025: 700–900 €/kWp"
            )

        # ── Sezione 3: Orari operativi ────────────────────────────────────────
        st.markdown("### 3 — Orari operativi")
        col8, col9 = st.columns(2)
        with col8:
            ore_lavoro_giorno = st.number_input(
                "Ore lavoro / giorno", min_value=1, max_value=24, value=10,
                help="Turno di produzione tipico"
            )
        with col9:
            giorni_lavoro_settimana = st.number_input(
                "Giorni lavoro / settimana", min_value=1, max_value=7, value=5
            )

        # ── Sezione 4: Tariffe ────────────────────────────────────────────────
        st.markdown("### 4 — Tariffe elettriche")
        col10, col11 = st.columns(2)
        with col10:
            spread_eur_kwh = st.number_input(
                "Spread fornitore (€/kWh)", min_value=0.0, max_value=0.5,
                value=_spread_val, step=0.005, format="%.3f",
                help="Componente commerciale sopra il prezzo mercato"
                     + (" · Stimato da bolletta" if pf.get("spread") is not None else "")
            )
        with col11:
            quota_potenza = st.number_input(
                "Quota potenza (€/kW/mese)", min_value=0.0, max_value=50.0,
                value=_quota_val, step=0.5,
                help="Tutte le componenti di potenza aggregate (trasmissione + distribuzione)"
                     + (" · Stimato da bolletta" if pf.get("quota_pot") is not None else "")
            )

        # ── Sezione 5: Obiettivo ──────────────────────────────────────────────
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

        # ── Sezione 6: Incentivi ──────────────────────────────────────────────
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
                help="Aliquota effettiva. Range tipico: 25–45%.",
            )

        st.divider()
        submitted = st.form_submit_button("Analizza →", type="primary", use_container_width=True)

    if not submitted:
        return

    # ── Validazione ───────────────────────────────────────────────────────────
    if not nome_cliente.strip():
        st.warning("Inserisci il nome del cliente.")
        st.stop()
    if consumo_kwh <= 0:
        st.warning("Inserisci il consumo del periodo dalla bolletta (kWh > 0).")
        st.stop()
    if potenza_contrattuale_kw <= 0:
        st.warning("Inserisci la potenza contrattuale dalla bolletta (kW > 0).")
        st.stop()

    # ── Derivazione campi per il motore ──────────────────────────────────────
    consumo_annuo_kwh = int(round(consumo_kwh * 12))
    picco_kw_bill     = pf.get("picco_kw") if pf else None
    picco_potenza_kw  = (
        int(round(picco_kw_bill)) if picco_kw_bill
        else int(round(potenza_contrattuale_kw * 0.9))
    )

    # Geocode comune → lat/lon
    lat, lon = None, None
    if comune.strip():
        with st.spinner(f"Geocoding '{comune}'..."):
            coords = _geocode(comune.strip())
        if coords:
            lat_c, lon_c = coords
            if _IT_LAT[0] <= lat_c <= _IT_LAT[1] and _IT_LON[0] <= lon_c <= _IT_LON[1]:
                lat, lon = lat_c, lon_c
            else:
                st.warning(
                    f"Le coordinate trovate per '{comune}' ({lat_c:.2f}°N, {lon_c:.2f}°E) "
                    "sono fuori dall'Italia. Verrà usato un profilo FV sintetico."
                )
        else:
            st.warning(
                f"Posizione '{comune}' non trovata. "
                "Uso coordinate default Nord Italia — PVGIS sarà meno preciso."
            )

    form = {
        "nome_cliente":              nome_cliente.strip(),
        "location":                  comune.strip() or "Sito",
        "consumo_annuo_kwh":         consumo_annuo_kwh,
        "picco_potenza_kw":          picco_potenza_kw,
        "ha_pv":                     ha_pv,
        "kwp":                       kwp if ha_pv else 0,
        "kwp_proposto":              kwp_proposto if not ha_pv else 0,
        "costo_eur_kwp":             costo_eur_kwp,
        "ore_lavoro_giorno":         ore_lavoro_giorno,
        "giorni_lavoro_settimana":   giorni_lavoro_settimana,
        "spread_eur_kwh":            spread_eur_kwh,
        "quota_potenza_eur_kw_mese": quota_potenza,
        "goal":                      goal,
        "lat":                       lat,
        "lon":                       lon,
        "t5_enable":                 t5_enable,
        "t5_aliquota_pct":           t5_aliquota if t5_enable else 0,
        # Campi grezzi bolletta (per Block 1 e audit)
        "consumo_kwh_periodo":       consumo_kwh,
        "potenza_contrattuale_kw":   potenza_contrattuale_kw,
        "costo_totale_eur":          costo_totale_eur   if costo_totale_eur   > 0 else None,
        "costo_energia_eur":         costo_energia_eur  if costo_energia_eur  > 0 else None,
        "costo_potenza_eur":         costo_potenza_eur  if costo_potenza_eur  > 0 else None,
        "f1_kwh":                    f1_kwh if f1_kwh > 0 else None,
        "f2_kwh":                    f2_kwh if f2_kwh > 0 else None,
        "f3_kwh":                    f3_kwh if f3_kwh > 0 else None,
    }

    case = _build_case(form)

    sdi = None
    if _BLOCK1_AVAILABLE:
        try:
            sdi = _build_sdi_block1(_form_to_intake(form), BASE_DIR)
        except Exception as _e:
            st.warning(f"Block 1 (intake) non disponibile: {_e}")

    diagnostic = None
    if _DIAGNOSTIC_AVAILABLE and sdi:
        try:
            diagnostic = _build_diagnostic(sdi, BASE_DIR)
        except Exception:
            pass

    with st.spinner("Simulazione in corso..."):
        t0 = time.perf_counter()
        profiles, sim, econ = _run_simulation(case)
        elapsed = time.perf_counter() - t0

    st.session_state.update({
        "step":       "diagnostic" if diagnostic else "results",
        "form":       form,
        "case":       case,
        "profiles":   profiles,
        "sim":        sim,
        "econ":       econ,
        "elapsed":    elapsed,
        "sdi":        sdi,
        "diagnostic": diagnostic,
    })
    st.rerun()


# ── Step 2 — Block 1 Diagnostic ──────────────────────────────────────────────

def _page_block1_diagnostic() -> None:
    form       = st.session_state["form"]
    sdi        = st.session_state.get("sdi") or {}
    diagnostic = st.session_state.get("diagnostic") or {}

    if not diagnostic:
        st.warning("Dati diagnostici non disponibili. Torna all'intake.")
        if st.button("← Torna all'intake"):
            st.session_state["step"] = "intake"
            st.rerun()
        return

    snap  = diagnostic.get("executive_snapshot") or {}
    ei    = diagnostic.get("energy_identity") or {}
    lb    = diagnostic.get("load_behavior") or {}
    pk    = diagnostic.get("power_peaks") or {}
    tb    = diagnostic.get("tariff_bands") or {}
    ep    = diagnostic.get("economic_picture") or {}
    dqr   = diagnostic.get("data_quality_report") or {}

    nome  = snap.get("nome_cliente") or form.get("nome_cliente", "Sito")
    dq    = snap.get("data_quality_level", 1)

    # ── Header ────────────────────────────────────────────────────────────────
    col_h, col_b = st.columns([6, 1])
    with col_h:
        st.markdown(f"# Diagnostica Sito — {nome}")
    with col_b:
        st.markdown("<div style='padding-top:1.1rem'></div>", unsafe_allow_html=True)
        if st.button("← Modifica intake", use_container_width=True):
            st.session_state["step"] = "intake"
            st.rerun()

    # ── Banner qualità dati ───────────────────────────────────────────────────
    reliability = snap.get("reliability_label", "N/D")
    _dq_icons   = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "🟢"}
    _dq_icon    = _dq_icons.get(dq, "⚪")
    st.info(
        f"{_dq_icon} **Affidabilità diagnostica: {reliability}** (Livello L{dq}/5)  ·  "
        f"{ei.get('data_basis', '')}",
        icon=None,
    )

    # ── 1. Executive Snapshot ─────────────────────────────────────────────────
    st.markdown("### Snapshot energetico")
    c1, c2, c3, c4 = st.columns(4)

    consumo = snap.get("annual_consumption_kwh")
    net     = snap.get("annual_net_draw_kwh")
    cost    = snap.get("annual_energy_cost_eur")
    avg_c   = snap.get("avg_cost_eur_kwh")

    c1.metric("Consumo sito stimato",
              f"{consumo:,.0f} kWh" if consumo else "N/D",
              help="Consumo annuo totale del sito, incluso l'autoconsumo FV")
    c2.metric("Prelievo netto dalla rete",
              f"{net:,.0f} kWh" if net else "N/D",
              help="Energia acquistata dalla rete (bolletta)")
    c3.metric("Costo energetico stimato",
              f"{cost:,.0f} €/anno" if cost else "N/D")
    c4.metric("Costo medio energia",
              f"{avg_c:.4f} €/kWh" if avg_c else "N/D")

    c5, c6, c7, c8 = st.columns(4)
    contracted = snap.get("contracted_power_kw")
    peak       = snap.get("peak_kw")
    has_pv     = snap.get("has_pv", False)
    kwp        = snap.get("pv_kwp")
    pv_prod    = snap.get("pv_annual_production_kwh")

    c5.metric("Potenza contrattuale", f"{contracted:.0f} kW" if contracted else "N/D")
    c6.metric("Picco di potenza",     f"{peak:.0f} kW"       if peak       else "N/D")
    c7.metric("FV esistente",
              f"{kwp:.0f} kWp" if (has_pv and kwp) else ("Sì" if has_pv else "No"))
    c8.metric("Produzione FV stimata",
              f"{pv_prod:,.0f} kWh/anno" if pv_prod else ("— kWh/anno" if has_pv else "—"))

    st.divider()

    # ── 2. Identità energetica ────────────────────────────────────────────────
    st.markdown("### Identità energetica del sito")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        rows = []
        sc = ei.get("pv_self_consumption_kwh")
        ex = ei.get("pv_export_kwh")
        cov = ei.get("pv_coverage_pct")

        if consumo:
            rows.append(("Consumo totale sito", f"{consumo:,.0f} kWh",
                         ei.get("site_consumption_source", "—"), ei.get("site_consumption_conf", "—")))
        if net:
            rows.append(("Prelievo netto rete", f"{net:,.0f} kWh", "billing", "—"))
        if pv_prod:
            rows.append(("Produzione FV", f"{pv_prod:,.0f} kWh", "pvgis/sintetico", "—"))
        if sc:
            rows.append(("Autoconsumo FV", f"{sc:,.0f} kWh ({cov:.0f}%)" if cov else f"{sc:,.0f} kWh", "calcolato", "—"))
        if ex is not None and has_pv:
            rows.append(("Stima immissione rete", f"{ex:,.0f} kWh", "calcolato", "—"))

        for label, val, src, conf in rows:
            st.markdown(f"**{label}:** {val}")
            st.caption(f"source: {src} · conf: {conf}")

    with col_right:
        if has_pv and consumo and pv_prod:
            grid_pct = round((net or 0) / consumo * 100, 1) if consumo > 0 else 0
            pv_sc_pct = cov or 0
            other_pct = max(0, 100 - grid_pct - pv_sc_pct)

            fig_pie = go.Figure(go.Bar(
                x=[net or 0, sc or 0],
                y=["Prelievo netto rete", "Autoconsumo FV"],
                orientation="h",
                marker_color=["#4C8BF5", "#F4C430"],
            ))
            fig_pie.update_layout(
                height=200, margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title="kWh/anno",
                yaxis_tickfont_size=11,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.markdown(f"**Origine profilo carico:** {ei.get('load_profile_origin', 'N/D')}")
            st.markdown(f"**Base dati:** {ei.get('data_basis', 'N/D')}")

    st.divider()

    # ── 3. Comportamento del carico ───────────────────────────────────────────
    if lb.get("available"):
        st.markdown("### Comportamento del carico")

        slot_labels = lb["slot_labels"]
        avg_load    = lb["avg_daily_load"]
        avg_grid    = lb["avg_daily_grid"]
        avg_pv      = lb["avg_daily_pv"]

        # Curva media giornaliera
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=slot_labels, y=avg_load,
            name="Carico sito (media)", line=dict(color="#4C8BF5", width=2),
        ))
        fig_daily.add_trace(go.Scatter(
            x=slot_labels, y=avg_grid,
            name="Prelievo rete (media)", line=dict(color="#FF6B6B", width=2, dash="dash"),
        ))
        if has_pv and max(avg_pv) > 0:
            fig_daily.add_trace(go.Scatter(
                x=slot_labels, y=avg_pv,
                name="Produzione FV (media)", line=dict(color="#F4C430", width=2),
            ))
        fig_daily.update_layout(
            title="Curva di carico media giornaliera (annuale)",
            xaxis_title="Ora del giorno", yaxis_title="Potenza (kW)",
            height=320, margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis=dict(tickmode="array",
                       tickvals=slot_labels[::16],
                       ticktext=slot_labels[::16]),
        )
        st.plotly_chart(fig_daily, use_container_width=True)

        # Vista stagionale
        seasonal = lb.get("seasonal") or {}
        if seasonal:
            season_names = {
                "winter": "Inverno (Dic–Feb)",
                "spring": "Primavera (Mar–Mag)",
                "summer": "Estate (Giu–Ago)",
                "autumn": "Autunno (Set–Nov)",
            }
            season_colors = {
                "winter": "#6495ED", "spring": "#90EE90",
                "summer": "#FF8C00", "autumn": "#CD853F",
            }
            fig_seas = go.Figure()
            for key, label in season_names.items():
                if key in seasonal:
                    fig_seas.add_trace(go.Scatter(
                        x=slot_labels,
                        y=seasonal[key]["load"],
                        name=label,
                        line=dict(color=season_colors[key], width=2),
                    ))
            fig_seas.update_layout(
                title="Profilo di carico medio per stagione",
                xaxis_title="Ora del giorno", yaxis_title="Potenza (kW)",
                height=300, margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                xaxis=dict(tickmode="array",
                           tickvals=slot_labels[::16],
                           ticktext=slot_labels[::16]),
            )
            st.plotly_chart(fig_seas, use_container_width=True)

        st.divider()

    # ── 4. Potenza e picchi ───────────────────────────────────────────────────
    if pk.get("available"):
        st.markdown("### Potenza e picchi")
        pk_c1, pk_c2, pk_c3, pk_c4 = st.columns(4)
        pk_c1.metric("Picco massimo",    f"{pk['max_peak_kw']:.0f} kW")
        pk_c2.metric("Picco P95",        f"{pk['p95_kw']:.0f} kW",
                     help="95% dei valori è sotto questa soglia")
        pk_c3.metric("Potenza contrattuale", f"{pk.get('contracted_power_kw', 0):.0f} kW")

        ratio = pk.get("peak_to_contracted_ratio")
        h80   = pk.get("hours_above_80pct_contracted")
        pk_c4.metric("Rapporto picco/contratto",
                     f"{ratio:.2f}×" if ratio else "N/D",
                     delta="ok" if (ratio and ratio < 1.0) else ("attenzione" if ratio else None),
                     delta_color="normal" if (ratio and ratio < 1.0) else "inverse")

        if h80 is not None:
            if h80 > 0:
                st.caption(f"⚡ {h80:.0f} ore/anno con potenza > 80% della contrattuale ({pk.get('contracted_power_kw', 0) * 0.8:.0f} kW)")
            else:
                st.caption("✅ Il carico non supera mai l'80% della potenza contrattuale.")

        st.divider()

    # ── 5. Fasce tariffarie ───────────────────────────────────────────────────
    if tb.get("available"):
        st.markdown("### Fasce tariffarie F1 / F2 / F3")
        tb_c1, tb_c2, tb_c3 = st.columns(3)
        tb_c1.metric("F1 — Ore di punta",    f"{tb['f1_pct']:.0f}%  ({tb['f1_kwh']:,.0f} kWh)")
        tb_c2.metric("F2 — Ore intermedie",  f"{tb['f2_pct']:.0f}%  ({tb['f2_kwh']:,.0f} kWh)")
        tb_c3.metric("F3 — Ore fuori punta", f"{tb['f3_pct']:.0f}%  ({tb['f3_kwh']:,.0f} kWh)")

        fig_bands = go.Figure(go.Bar(
            x=["F1 (punta)", "F2 (intermedia)", "F3 (fuori punta)"],
            y=[tb.get("f1_kwh", 0), tb.get("f2_kwh", 0), tb.get("f3_kwh", 0)],
            marker_color=["#E74C3C", "#F39C12", "#27AE60"],
            text=[f"{tb['f1_pct']:.0f}%", f"{tb['f2_pct']:.0f}%", f"{tb['f3_pct']:.0f}%"],
            textposition="outside",
        ))
        fig_bands.update_layout(
            yaxis_title="kWh/anno (da bolletta)", height=260,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_bands, use_container_width=True)

        if tb.get("f1f2f3_used_in_reconstruction"):
            st.caption("✅ F1/F2/F3 utilizzati nella ricostruzione del profilo di carico.")
        if tb.get("band_mismatch_warning"):
            st.caption(f"⚠️ {tb['band_mismatch_warning']}")

        st.divider()

    # ── 6. Quadro economico attuale ───────────────────────────────────────────
    if ep.get("available"):
        st.markdown("### Quadro economico attuale")
        ep_c1, ep_c2, ep_c3, ep_c4 = st.columns(4)
        ep_c1.metric("Costo annuo energia", f"{ep['annual_cost_eur']:,.0f} €")
        ep_c2.metric("Costo medio €/kWh",   f"{ep['avg_eur_kwh']:.4f} €/kWh" if ep.get("avg_eur_kwh") else "N/D")

        e_cost = ep.get("energy_cost_eur")
        p_cost = ep.get("power_cost_eur")
        ep_c3.metric("Componente energia",  f"{e_cost:,.0f} € ({ep.get('energy_share_pct', 0):.0f}%)" if e_cost else "N/D")
        ep_c4.metric("Componente potenza",  f"{p_cost:,.0f} € ({ep.get('power_share_pct', 0):.0f}%)" if p_cost else "N/D")

        spread = ep.get("spread_eur_kwh")
        quota  = ep.get("quota_potenza")
        if spread or quota:
            ep_c5, ep_c6, _, _ = st.columns(4)
            ep_c5.metric("Spread fornitore",      f"{spread:.4f} €/kWh" if spread else "N/D",
                         help=f"source: {ep.get('spread_source', '—')}")
            ep_c6.metric("Quota potenza",         f"{quota:.2f} €/kW/mese" if quota else "N/D",
                         help=f"source: {ep.get('quota_source', '—')}")

        st.divider()

    # ── 7. Affidabilità del diagnostico ──────────────────────────────────────
    st.markdown("### Affidabilità del diagnostico")
    col_obs, col_der, col_est = st.columns(3)

    def _fmt_fields(fields):
        return "  \n".join(f"· {f}" for f in fields) if fields else "—"

    col_obs.markdown("**Osservati (da bolletta o inseriti)**")
    col_obs.markdown(_fmt_fields(dqr.get("observed_fields", [])))
    col_der.markdown("**Derivati (calcolati da dati disponibili)**")
    col_der.markdown(_fmt_fields(dqr.get("derived_fields", [])))
    col_est.markdown("**Stimati (profilo sintetico o default)**")
    col_est.markdown(_fmt_fields(dqr.get("estimated_fields", [])))

    nota = dqr.get("nota_accuratezza")
    if nota:
        st.caption(f"📋 {nota}")

    warnings = dqr.get("warnings") or []
    if warnings:
        st.markdown("**Avvisi:**")
        for w in warnings:
            st.caption(f"⚠️ {w}")

    assumptions = dqr.get("assumptions") or []
    if assumptions:
        with st.expander(f"Assunzioni ({len(assumptions)})", expanded=False):
            for a in assumptions:
                st.markdown(f"**{a.get('campo', '')}:** {a.get('assunzione', '')}  \n"
                            f"*Rischio:* {a.get('rischio', '')}")

    st.divider()

    # ── Debug layer ───────────────────────────────────────────────────────────
    with st.expander("Dati tecnici / debug SDI", expanded=False):
        _render_sdi_diagnostic(sdi)
        st.json(sdi)

    # ── Bottone verso analisi BESS ────────────────────────────────────────────
    st.markdown("---")
    col_go, col_info_bess = st.columns([2, 5])
    with col_go:
        if st.button("Analisi BESS →", type="primary", use_container_width=True):
            form = st.session_state.get("form", {})
            if not st.session_state.get("sim"):
                with st.spinner("Simulazione BESS in corso…"):
                    t0 = time.perf_counter()
                    try:
                        case = _build_case(form)
                        profiles, sim, econ = _run_simulation(case)
                        elapsed = time.perf_counter() - t0
                        st.session_state.update({
                            "case": case, "profiles": profiles,
                            "sim": sim, "econ": econ, "elapsed": elapsed,
                        })
                    except Exception as e:
                        st.error(f"Errore simulazione: {e}")
                        st.stop()
            st.session_state["step"] = "results"
            st.rerun()
    with col_info_bess:
        st.caption(
            "Lancia la simulazione S1/S2/S3/S4 e calcola il business case BESS "
            "con i dati diagnostici del sito."
        )


# ── Step 3 — Results ──────────────────────────────────────────────────────────

def _page_results() -> None:
    form     = st.session_state.get("form", {})
    case     = st.session_state.get("case")
    profiles = st.session_state.get("profiles")
    sim      = st.session_state.get("sim")
    econ     = st.session_state.get("econ")
    elapsed  = st.session_state.get("elapsed", 0)

    if not sim or not case:
        st.error("Simulazione non disponibile. Torna alla diagnostica e clicca 'Analisi BESS →'.")
        if st.button("← Diagnostica"):
            st.session_state["step"] = "diagnostic"
            st.rerun()
        return

    ha_pv     = form["ha_pv"] and form["kwp"] > 0
    pv_source = case["pv"]["profilo_source"]
    best_sc   = _pick_scenario(form["goal"], ha_pv)
    e_best    = econ.get(best_sc, {})
    assum     = _assumptions(form, pv_source)

    # ── Header + back buttons ─────────────────────────────────────────────────
    col_h, col_d, col_b = st.columns([5, 1, 1])
    with col_h:
        st.markdown(f"# Soluzione per {form['nome_cliente']}")
    with col_d:
        st.markdown("<div style='padding-top:1.1rem'></div>", unsafe_allow_html=True)
        if st.session_state.get("diagnostic") and st.button("← Diagnostica", use_container_width=True):
            st.session_state["step"] = "diagnostic"
            st.rerun()
    with col_b:
        st.markdown("<div style='padding-top:1.1rem'></div>", unsafe_allow_html=True)
        if st.button("← Modifica intake", use_container_width=True):
            st.session_state["step"] = "intake"
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

    # ── Block 1 debug expander ────────────────────────────────────────────────
    sdi = st.session_state.get("sdi")
    if sdi:
        with st.expander("Block 1 — Site Diagnostic (debug)", expanded=False):
            _render_sdi_diagnostic(sdi)
            st.json(sdi)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="BESS Decision Tool",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    if "step" not in st.session_state:
        st.session_state["step"] = "intake"

    step = st.session_state["step"]
    if step == "intake":
        _page_block1_intake()
    elif step == "input":
        _page_input()
    elif step == "diagnostic":
        _page_block1_diagnostic()
    else:
        _page_results()


if __name__ == "__main__":
    main()
