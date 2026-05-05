"""
dashboard.py
Streamlit dashboard — visualises BESS simulation results.

Run:
  streamlit run dashboard.py
"""

import json
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

SLOT_H = 0.25

# Representative week start days for seasonal views (0-indexed, Monday-aligned)
_SEASON_DAYS = {
    "Inverno  (gen, sett. 4)":   21,
    "Primavera (apr, sett. 16)": 105,
    "Estate   (lug, sett. 28)":  189,
    "Autunno  (ott, sett. 41)":  280,
}

_DOW_LABELS = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="BESS Simulator",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    data = _load_results()
    if data is None:
        st.error("Nessun file risultati trovato. Esegui prima: `python run_simulation.py`")
        st.stop()

    meta     = data["meta"]
    case     = data["case"]
    profiles = {k: np.array(v) if isinstance(v, list) else v
                for k, v in data["profiles"].items()}
    sim      = {sc: {k: np.array(v) if isinstance(v, list) else v
                     for k, v in r.items()}
                for sc, r in data["simulation"].items()}
    econ     = data["economics"]

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔋 BESS Simulator")
        st.caption("MVP v0.1")
        st.divider()

        st.markdown(f"**{meta['nome_cliente']}**  \n{meta['nome_sito']}")
        st.caption(f"Case: {meta['case_id']}  |  {meta['data_creazione']}")
        st.divider()

        bess_sc = st.radio(
            "Scenario BESS",
            options=["S3", "S4"],
            index=1,
            format_func=lambda s: f"{s} — autoconsumo" if s == "S3"
                                  else f"{s} — multilayer",
        )
        st.divider()

        season = st.selectbox("Vista stagionale", list(_SEASON_DAYS.keys()))

    # ── Page ─────────────────────────────────────────────────────────────────
    st.markdown(f"# {meta['nome_cliente']}")
    st.caption(meta["nome_sito"])

    _section_kpis(profiles, econ, bess_sc, case)

    st.divider()
    st.subheader(f"Comportamento tecnico — {season}  ·  Scenario {bess_sc}")
    _section_weekly(profiles, sim, bess_sc, _SEASON_DAYS[season])

    st.divider()
    st.subheader("Analisi economica")
    _section_economics(econ, case)


# ── KPI section ───────────────────────────────────────────────────────────────

def _section_kpis(profiles, econ, bess_sc, case) -> None:
    e_bess = econ.get(bess_sc, {})
    e_s1   = econ.get("S1", {})
    e_s2   = econ.get("S2", {})

    st.markdown("#### Profili annuali")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Consumo annuo",
              f"{profiles['load_kw'].sum() * SLOT_H:,.0f} kWh",
              f"picco {profiles['load_kw'].max():.1f} kW")
    c2.metric("Produzione FV",
              f"{profiles['pv_kw'].sum() * SLOT_H:,.0f} kWh",
              f"picco {profiles['pv_kw'].max():.1f} kW")
    s1_cost = e_s1.get("annual_energy_cost_eur", 0)
    s2_cost = e_s2.get("annual_energy_cost_eur", 0)
    c3.metric("Costo energia con FV (S2)",
              f"{s2_cost:,.0f} €/yr",
              f"−{s1_cost - s2_cost:,.0f} € vs S1")
    c4.metric("Prezzo medio energia",
              f"{profiles['price_eur_kwh'].mean():.4f} €/kWh",
              f"range {profiles['price_eur_kwh'].min():.3f} – "
              f"{profiles['price_eur_kwh'].max():.3f}")

    if "annual_saving_eur" not in e_bess:
        return

    st.markdown(f"#### KPI batteria — Scenario {bess_sc}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Investimento",    f"{e_bess['investment_eur']:,.0f} €")
    c2.metric("Risparmio annuo", f"{e_bess['annual_saving_eur']:,.0f} €/yr")
    pb = e_bess.get("payback_yr")
    c3.metric("Payback",         f"{pb} yr" if pb else "— yr")
    c4.metric("NPV",             f"{e_bess['npv_eur']:,.0f} €")
    irr = e_bess.get("irr_pct")
    c5.metric("IRR",             f"{irr} %" if irr is not None else "—")
    c6.metric("Cicli equiv.",    f"{e_bess['equivalent_cycles']:.0f} /yr")


# ── Weekly technical chart ────────────────────────────────────────────────────

def _section_weekly(profiles, sim, bess_sc, day_start: int) -> None:
    s0 = day_start * 96
    s1 = s0 + 7 * 96
    t  = np.arange(7 * 96) * SLOT_H   # hours 0 … 167.75

    load   = profiles["load_kw"][s0:s1]
    pv     = profiles["pv_kw"][s0:s1]
    grid   = sim[bess_sc]["grid_kw"][s0:s1]
    charge = sim[bess_sc]["bess_charge_kw"][s0:s1]
    disc   = sim[bess_sc]["bess_discharge_kw"][s0:s1]
    soc    = sim[bess_sc]["soc_kwh"][s0:s1]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=("Potenze  (kW)", "SOC batteria  (kWh)"),
    )

    fig.add_trace(go.Scatter(x=t, y=load,   name="Carico",       line=dict(color="#1565C0", width=2)),     row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=pv,     name="FV",           line=dict(color="#F9A825", width=2)),     row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=grid,   name="Rete",         line=dict(color="#757575", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=charge, name="Carica BESS",  line=dict(color="#2E7D32", width=1.5)),   row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=-disc,  name="Scarica BESS", line=dict(color="#C62828", width=1.5)),   row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=soc, name="SOC", fill="tozeroy",
        line=dict(color="#6A1B9A", width=1.5),
        fillcolor="rgba(106,27,154,0.12)",
    ), row=2, col=1)

    for d in range(1, 7):
        fig.add_vline(x=d * 24, line_dash="dash", line_color="rgba(0,0,0,0.12)")

    fig.update_xaxes(
        tickvals=[d * 24 + 12 for d in range(7)],
        ticktext=_DOW_LABELS,
        row=2, col=1,
    )
    fig.update_yaxes(title_text="kW",  row=1, col=1)
    fig.update_yaxes(title_text="kWh", row=2, col=1)
    fig.update_layout(
        height=520,
        margin=dict(t=40, b=10, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Economics section ─────────────────────────────────────────────────────────

def _section_economics(econ, case) -> None:
    col_l, col_r = st.columns(2)

    # Left — savings breakdown by layer
    with col_l:
        st.markdown("**Risparmio annuo per layer  (S3 vs S4)**")
        scenarios = [sc for sc in ["S3", "S4"] if "saving_fv_eur" in econ.get(sc, {})]
        if scenarios:
            fig_bar = go.Figure()
            layer_defs = [
                ("Layer 1 — Autoconsumo FV",  "saving_fv_eur",       "#F9A825"),
                ("Layer 2 — Peak shaving",     "saving_quota_eur",    "#1565C0"),
                ("Layer 3 — Shifting",         "shifting_margin_eur", "#2E7D32"),
            ]
            for label, key, color in layer_defs:
                vals = [econ[sc][key] for sc in scenarios]
                fig_bar.add_trace(go.Bar(name=label, x=scenarios, y=vals,
                                          marker_color=color))
            fig_bar.update_layout(
                barmode="stack", height=300,
                margin=dict(t=10, b=30, l=60, r=10),
                yaxis_title="€/anno",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # Right — cumulative cash flows
    with col_r:
        st.markdown("**Flussi di cassa cumulativi**")
        anni  = case["simulation"]["anni_analisi"]
        years = list(range(anni + 1))
        fig_cf = go.Figure()
        cf_colors = {"S3": "#1565C0", "S4": "#2E7D32"}
        for sc in ["S3", "S4"]:
            if "cashflows" not in econ.get(sc, {}):
                continue
            cumcf = np.cumsum(econ[sc]["cashflows"]).tolist()
            fig_cf.add_trace(go.Scatter(
                x=years, y=cumcf, name=sc,
                mode="lines+markers",
                line=dict(color=cf_colors.get(sc, "#999"), width=2),
                marker=dict(size=4),
            ))
        fig_cf.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.25)")
        fig_cf.update_layout(
            height=300,
            margin=dict(t=10, b=30, l=60, r=10),
            xaxis_title="Anno",
            yaxis_title="€",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
        )
        st.plotly_chart(fig_cf, use_container_width=True)

    # Detail expanders for S3 and S4
    for sc in ["S3", "S4"]:
        e = econ.get(sc, {})
        if "saving_fv_eur" not in e:
            continue
        pb = f"{e['payback_yr']} anni" if e["payback_yr"] else "non raggiunto"
        with st.expander(f"Dettaglio {sc}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Investimento",     f"{e['investment_eur']:,.0f} €")
            c1.metric("O&M annuo",        f"{e['om_annual_eur']:,.0f} €/yr")
            c2.metric("Layer 1 (FV)",     f"{e['saving_fv_eur']:,.0f} €/yr")
            c2.metric("Layer 2 (picco)",  f"{e['saving_quota_eur']:,.0f} €/yr")
            c2.metric("Layer 3 (shift)",  f"{e['shifting_margin_eur']:,.0f} €/yr")
            c3.metric("Payback",          pb)
            c3.metric("NPV",              f"{e['npv_eur']:,.0f} €")
            irr = e.get("irr_pct")
            c3.metric("IRR",              f"{irr} %" if irr is not None else "—")


# ── Loader ────────────────────────────────────────────────────────────────────

def _load_results() -> dict | None:
    base        = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base, "results")
    if not os.path.isdir(results_dir):
        return None
    files = sorted(
        [f for f in os.listdir(results_dir) if f.endswith("_results.json")],
        key=lambda f: os.path.getmtime(os.path.join(results_dir, f)),
        reverse=True,
    )
    if not files:
        return None
    with open(os.path.join(results_dir, files[0])) as f:
        return json.load(f)


if __name__ == "__main__":
    main()
