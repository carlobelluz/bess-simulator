"""
intake/site_reconstruction.py
Ricostruzione dello stato energetico del sito da dati di bolletta + PVGIS.

Public API:
  SiteEnergyState   — dataclass contenitore di tutto lo stato
  TrackedValue      — wrapper valore + source + confidence
  reconcile(state)  — orchestratore: ricalcola tutti i campi derivati

Modalità supportate in Brief 1:
  "auto" — profilo sintetico calibrato su F1/F2/F3 mensili + picchi

Modalità vincolate (user_autoconsumo_pct_*, user_surplus_*) → NotImplementedError (Brief 3).

Identità fondamentale del modello:
  fabbisogno_energetico = prelievo_rete + autoconsumo_fv
  fv_prodotta           = autoconsumo_fv + surplus_fv

Convenzioni:
  - Tutte le energie in kWh, le potenze in kW
  - Array mensili: 12 elementi, 0=gennaio … 11=dicembre
  - Profili quartorari: 35.040 slot (anno standard non bisestile)
  - Slot duration: 15 min → 0.25 h
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal, Optional

import numpy as np
from scipy.optimize import minimize

from intake.tariff_bands import build_band_masks, italian_holidays


# ── Tipi ──────────────────────────────────────────────────────────────────────

SourceType     = Literal["observed", "derived", "estimated", "user_input"]
ConfidenceType = Literal["high", "medium", "low"]

_ARCHETIPI = (
    "industrial_single_shift",
    "industrial_double_shift",
    "industrial_continuous",
    "commercial_office",
    "mixed",
)

_N_SLOTS = 35_040   # 365 × 96 slot/giorno
_SLOT_H  = 0.25     # ore per slot


# ── Dataclass di supporto ─────────────────────────────────────────────────────

@dataclass
class TrackedValue:
    """Valore numerico con tracciatura di origine e confidenza."""
    value:      float | None
    source:     SourceType
    confidence: ConfidenceType


# ── Dataclass principale ──────────────────────────────────────────────────────

@dataclass
class SiteEnergyState:
    """
    Stato energetico completo del sito.

    I campi di input (prelievo_*, picchi_*, fv_*) vengono popolati dall'esterno
    prima di chiamare reconcile(). I campi di output (fabbisogno_*, autoconsumo_*,
    surplus_*, profili quartorari, diagnostica) vengono popolati da reconcile().
    """

    # ── Inputs bolletta ────────────────────────────────────────────────────────
    prelievo_f1_mensile: list[float] = field(default_factory=lambda: [0.0] * 12)
    prelievo_f2_mensile: list[float] = field(default_factory=lambda: [0.0] * 12)
    prelievo_f3_mensile: list[float] = field(default_factory=lambda: [0.0] * 12)
    picchi_mensili_kw:   list[float] = field(default_factory=lambda: [0.0] * 12)

    quota_potenza_eur_kw_mese: float = 0.0

    # ── Inputs FV ──────────────────────────────────────────────────────────────
    has_fv:               bool                   = False
    fv_kwp:               float                  = 0.0
    fv_tilt:              float                  = 30.0
    fv_azimuth:           float                  = 0.0
    fv_oraria_pvgis_kw:   Optional[np.ndarray]   = None   # 8760 valori
    fv_mensile_pvgis_kwh: Optional[list[float]]  = None   # 12 valori

    # ── Vincoli utente opzionali ───────────────────────────────────────────────
    user_autoconsumo_pct_annuo:    Optional[float]       = None
    user_autoconsumo_pct_mensile:  Optional[list[float]] = None
    user_surplus_kwh_annuo:        Optional[float]       = None
    user_surplus_kwh_mensile:      Optional[list[float]] = None

    # ── Parametri di forma ─────────────────────────────────────────────────────
    archetype:           Optional[str]  = None
    shift_start:         int            = 7
    shift_end:           int            = 19
    giorni_lavorativi:   list[bool]     = field(
        default_factory=lambda: [True] * 5 + [False] * 2
    )
    carico_base_pct:           float                                       = 0.15
    stagionalita_inverno: Literal["aumenta", "neutro", "riduce"]           = "neutro"
    stagionalita_estate:  Literal["aumenta", "neutro", "riduce"]           = "neutro"

    anno_riferimento: int = 2025

    # ── Outputs ricostruiti ────────────────────────────────────────────────────
    fabbisogno_annuo_kwh:          Optional[TrackedValue]       = None
    fabbisogno_mensile_kwh:        Optional[list[TrackedValue]] = None

    autoconsumo_fv_annuo_kwh:      Optional[TrackedValue]       = None
    autoconsumo_fv_mensile_kwh:    Optional[list[TrackedValue]] = None

    surplus_fv_annuo_kwh:          Optional[TrackedValue]       = None
    surplus_fv_mensile_kwh:        Optional[list[TrackedValue]] = None

    autoconsumo_pct_annuo:         Optional[TrackedValue]       = None
    autoconsumo_pct_mensile:       Optional[list[TrackedValue]] = None

    # Profili quartorari (35040 slot)
    load_profile_qh_kw:       Optional[np.ndarray] = None
    fv_profile_qh_kw:         Optional[np.ndarray] = None
    grid_profile_qh_kw:       Optional[np.ndarray] = None
    autoconsumo_profile_qh_kw: Optional[np.ndarray] = None

    # ── Diagnostica ────────────────────────────────────────────────────────────
    archetype_inferred:         Optional[str]   = None
    archetype_confidence_score: Optional[float] = None
    band_match_error_pct:       Optional[dict]  = None    # {'f1': %, 'f2': %, 'f3': %}
    band_match_error_monthly:   Optional[list]  = None    # 12 dict
    peak_match_error_pct_monthly: Optional[list[float]] = None
    overall_confidence:         Optional[ConfidenceType] = None
    assumptions_active:         list[str]       = field(default_factory=list)
    calibration_loss:           Optional[float] = None
    reconcile_mode:             Optional[str]   = None


# ── Helper privati ────────────────────────────────────────────────────────────

def _build_time_index(year: int) -> dict:
    """
    Costruisce metadati temporali per ogni slot quartorario dell'anno.

    Returns dict:
      month     : ndarray(35040,) int8, 1-12
      hour      : ndarray(35040,) int8, 0-23
      dow       : ndarray(35040,) int8, 0=Lun..6=Dom
      is_holiday: ndarray(35040,) bool
      band_masks: dict {'f1','f2','f3'} → ndarray(35040,) bool
    """
    n        = _N_SLOTS
    holidays = italian_holidays(year)
    start    = datetime(year, 1, 1)

    months     = np.zeros(n, dtype=np.int8)
    hours      = np.zeros(n, dtype=np.int8)
    dows       = np.zeros(n, dtype=np.int8)
    is_holiday = np.zeros(n, dtype=bool)

    for i in range(n):
        dt = start + timedelta(minutes=i * 15)
        months[i]     = dt.month
        hours[i]      = dt.hour
        dows[i]       = dt.weekday()
        is_holiday[i] = dt.date() in holidays

    raw_masks = build_band_masks(year, slot_minutes=15)
    band_masks = {k: v[:n].copy() for k, v in raw_masks.items()}

    return {
        "month":      months,
        "hour":       hours,
        "dow":        dows,
        "is_holiday": is_holiday,
        "band_masks": band_masks,
    }


def _infer_archetype(state: SiteEnergyState) -> tuple[str, float]:
    """
    Inferisce l'archetipo di consumo dal pattern F1/F2/F3 annuale.
    Restituisce (archetype_key, confidence_score 0-1).
    """
    f1_ann = sum(state.prelievo_f1_mensile)
    f2_ann = sum(state.prelievo_f2_mensile)
    f3_ann = sum(state.prelievo_f3_mensile)
    total  = f1_ann + f2_ann + f3_ann

    if total <= 0:
        return "mixed", 0.40

    pct_f1 = f1_ann / total
    pct_f2 = f2_ann / total
    pct_f3 = f3_ann / total
    f3_f1_ratio = f3_ann / max(f1_ann, 1.0)

    if pct_f3 > 0.40 and f3_f1_ratio > 0.70:
        return "industrial_continuous", 0.85
    if pct_f1 > 0.50:
        return "industrial_single_shift", 0.80
    if pct_f1 > 0.45 and pct_f3 < 0.25:
        return "commercial_office", 0.75
    if pct_f1 > 0.30 and pct_f2 > 0.20 and pct_f3 > 0.30:
        return "industrial_double_shift", 0.70
    return "mixed", 0.50


def _base_daily_shape(archetype: str) -> np.ndarray:
    """
    Restituisce la forma adimensionale del giorno feriale tipo (24 float).
    Media approssimativa diurna ≈ 1.0, notte ≈ 0.15.
    """
    if archetype == "industrial_single_shift":
        shape = [
            0.15, 0.15, 0.15, 0.15, 0.15, 0.15,   # 0-5
            0.40, 0.75,                              # 6-7 rampa
            1.00, 1.00, 1.00, 1.00,                 # 8-11 plateau
            0.75, 0.75,                              # 12-13 pranzo
            0.90, 0.90, 0.90, 0.90,                 # 14-17 plateau pomeriggio
            0.55, 0.30,                              # 18-19 discesa
            0.15, 0.15, 0.15, 0.15,                 # 20-23 notte
        ]
    elif archetype == "industrial_double_shift":
        shape = [
            0.15, 0.15, 0.15, 0.15, 0.15, 0.15,   # 0-5
            0.70, 0.90,                              # 6-7 rampa
            1.00, 1.00, 1.00, 1.00,                 # 8-11
            0.85, 0.85,                              # 12-13
            1.00, 1.00, 1.00, 1.00,                 # 14-17
            0.90, 0.90, 0.90, 0.90,                 # 18-21 secondo turno
            0.50, 0.20,                              # 22-23 discesa
        ]
    elif archetype == "industrial_continuous":
        shape = [
            0.80, 0.80, 0.80, 0.80, 0.80, 0.80,   # 0-5 notte
            0.85, 0.90,                              # 6-7
            1.00, 1.00, 1.00, 1.00,                 # 8-11
            0.95, 0.95,                              # 12-13
            1.00, 1.00, 1.00, 1.00,                 # 14-17
            0.95, 0.90, 0.85, 0.82,                 # 18-21
            0.80, 0.80,                              # 22-23
        ]
    elif archetype == "commercial_office":
        shape = [
            0.15, 0.15, 0.15, 0.15, 0.15, 0.15,   # 0-5
            0.15, 0.40,                              # 6-7
            0.70, 0.95,                              # 7-8 rampa
            1.00, 1.00, 1.00, 1.00,                 # 9-12  (ore 9-13)
            0.90, 0.90, 0.90, 0.90,                 # 13-16 (ore 13-17)
            0.70, 0.40,                              # 17-18 discesa
            0.15, 0.15, 0.15, 0.15,                 # 19-23
        ]
    else:  # mixed — media single_shift + continuous
        ss = _base_daily_shape("industrial_single_shift")
        co = _base_daily_shape("industrial_continuous")
        shape = [(a + b) / 2 for a, b in zip(ss, co)]

    return np.array(shape, dtype=np.float64)


def _build_synthetic_load(
    state: SiteEnergyState,
    params: dict,
    time_index: dict,
) -> np.ndarray:
    """
    Costruisce il profilo quartorario di carico del sito (35040 slot, kW).
    Parametri: base_load_kw, production_amp_kw, winter_extra_kw.
    """
    base_shape_24 = _base_daily_shape(state.archetype)  # type: ignore[arg-type]

    n            = _N_SLOTS
    months       = time_index["month"]
    hours        = time_index["hour"]
    dows         = time_index["dow"]
    is_holiday   = time_index["is_holiday"]

    base_load    = max(float(params["base_load_kw"]),       0.0)
    prod_amp     = max(float(params["production_amp_kw"]),  0.0)
    winter_extra = max(float(params["winter_extra_kw"]),    0.0)

    # Componente base + boost invernale
    is_winter   = np.isin(months, [12, 1, 2])
    is_shoulder = np.isin(months, [3, 11])
    base        = np.full(n, base_load)
    base[is_winter]   += winter_extra
    base[is_shoulder] += winter_extra * 0.4

    # Forma produttiva
    prod_shape     = base_shape_24[hours]
    weekend_mask   = (dows >= 5) | is_holiday
    weekend_factor = np.where(weekend_mask, 0.20, 1.0)

    # Stagionalità ±15% sulla componente produttiva
    seasonal = np.ones(n)
    if state.stagionalita_inverno == "aumenta":
        seasonal[is_winter] *= 1.15
    elif state.stagionalita_inverno == "riduce":
        seasonal[is_winter] *= 0.85

    is_summer = np.isin(months, [6, 7, 8])
    if state.stagionalita_estate == "aumenta":
        seasonal[is_summer] *= 1.15
    elif state.stagionalita_estate == "riduce":
        seasonal[is_summer] *= 0.85

    production = prod_shape * prod_amp * weekend_factor * seasonal
    profile    = base + production
    return np.maximum(profile, 1.0)


def _compute_self_consumption(
    load_kw: np.ndarray,
    fv_kw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcola autoconsumo, surplus e prelievo da rete slot per slot.
    Returns: (autoconsumo_kw, surplus_kw, grid_kw)
    """
    autoconsumo = np.minimum(load_kw, fv_kw)
    surplus     = np.maximum(0.0, fv_kw - load_kw)
    grid        = load_kw - autoconsumo
    return autoconsumo, surplus, grid


def _aggregate_monthly(profile_qh_kw: np.ndarray, time_index: dict) -> list[float]:
    """Aggrega un profilo quartorario in 12 valori mensili (kWh)."""
    months = time_index["month"]
    return [
        float(profile_qh_kw[months == (m + 1)].sum() * _SLOT_H)
        for m in range(12)
    ]


def _calibrate_load(
    state: SiteEnergyState,
    time_index: dict,
    mode: str = "auto",
    target_ac: dict | None = None,
) -> dict:
    """
    Ottimizzazione Nelder-Mead: trova i 3 parametri di profilo che minimizzano
    lo scarto tra prelievo modellato e prelievo bolletta su 36 osservazioni
    (12 mesi × 3 fasce F1/F2/F3). In modalità vincolata aggiunge un termine
    soft sull'autoconsumo (peso 5.0).
    """
    months     = time_index["month"]
    band_masks = time_index["band_masks"]
    fv_qh      = state.fv_profile_qh_kw  # già calcolato prima di _calibrate_load

    def loss(p: np.ndarray) -> float:
        ac_penalty = 0.0  # inizializzato come prima riga, prima di qualunque condizione

        params = {
            "base_load_kw":       p[0],
            "production_amp_kw":  p[1],
            "winter_extra_kw":    p[2],
        }
        load_kw = _build_synthetic_load(state, params, time_index)
        grid_kw = np.maximum(load_kw - fv_qh, 0.0) if state.has_fv else load_kw

        loss_val = 0.0
        for m in range(12):
            mm = months == (m + 1)
            for band, target in (
                ("f1", state.prelievo_f1_mensile[m]),
                ("f2", state.prelievo_f2_mensile[m]),
                ("f3", state.prelievo_f3_mensile[m]),
            ):
                stimato = float(grid_kw[mm & band_masks[band]].sum() * _SLOT_H)
                denom   = max(target, 100.0)
                loss_val += ((stimato - target) / denom) ** 2

        # Penalità picco mensile (peso 0.1)
        peak_pen = 0.0
        for m in range(12):
            mm = months == (m + 1)
            tgt = state.picchi_mensili_kw[m]
            if tgt > 0 and mm.any():
                stim_peak = float(grid_kw[mm].max())
                peak_pen += ((stim_peak - tgt) / max(tgt, 10.0)) ** 2

        # Penalità vincolo autoconsumo (peso 5.0)
        if target_ac is not None and state.has_fv:
            ac_qh = np.minimum(load_kw, fv_qh)
            if mode == "constrained_annual" and target_ac["annuo"] is not None:
                ac_stim = float(ac_qh.sum() * _SLOT_H)
                denom   = max(target_ac["annuo"], 1000.0)
                ac_penalty = ((ac_stim - target_ac["annuo"]) / denom) ** 2
            elif mode == "constrained_monthly" and target_ac["mensile"] is not None:
                for m in range(12):
                    mm = months == (m + 1)
                    ac_stim_m = float(ac_qh[mm].sum() * _SLOT_H)
                    denom_m   = max(target_ac["mensile"][m], 100.0)
                    ac_penalty += ((ac_stim_m - target_ac["mensile"][m]) / denom_m) ** 2

        # Penalità parametri negativi
        penalty = sum(1e6 * v ** 2 for v in p if v < 0)

        return loss_val + 0.1 * peak_pen + 5.0 * ac_penalty + penalty

    # Stima iniziale basata sui dati di bolletta
    f3_total    = sum(state.prelievo_f3_mensile)
    annual_net  = (sum(state.prelievo_f1_mensile)
                   + sum(state.prelievo_f2_mensile)
                   + f3_total)
    n_f3_slots  = float(band_masks["f3"].sum())
    hours_f3    = n_f3_slots * _SLOT_H
    base_init   = f3_total / hours_f3 if hours_f3 > 0 else 5.0

    annual_base = base_init * 8760
    annual_prod = max(annual_net - annual_base, 1000.0)
    # ore produttive stimate (giorni lavorativi × ore turno)
    prod_hours  = 250 * 11 * 0.7
    prod_amp_init = annual_prod / prod_hours if prod_hours > 0 else 10.0

    # boost invernale: differenza dic vs lug (mesi 0-based: 11 vs 6)
    dec_net = (state.prelievo_f1_mensile[11]
               + state.prelievo_f2_mensile[11]
               + state.prelievo_f3_mensile[11])
    jul_net = (state.prelievo_f1_mensile[6]
               + state.prelievo_f2_mensile[6]
               + state.prelievo_f3_mensile[6])
    winter_extra_init = max(0.0, (dec_net - jul_net) / (24 * 31))

    x0     = np.array([base_init, prod_amp_init, winter_extra_init])
    result = minimize(
        loss, x0, method="Nelder-Mead",
        options={"maxiter": 200, "xatol": 0.1, "fatol": 1e-4},
    )

    return {
        "base_load_kw":      float(max(result.x[0], 0.0)),
        "production_amp_kw": float(max(result.x[1], 0.0)),
        "winter_extra_kw":   float(max(result.x[2], 0.0)),
        "loss":              float(result.fun),
    }


def _compute_overall_confidence(state: SiteEnergyState) -> ConfidenceType:
    """Stima la confidenza complessiva della ricostruzione."""
    if state.archetype_confidence_score is None or state.band_match_error_pct is None:
        return "low"

    avg_err = float(np.mean([
        abs(state.band_match_error_pct.get(b, 0.0))
        for b in ("f1", "f2", "f3")
    ]))

    if state.archetype_confidence_score >= 0.75 and avg_err < 5.0:
        return "high"
    if state.archetype_confidence_score >= 0.60 and avg_err < 12.0:
        return "medium"
    return "low"


# ── Helpers modalità vincolate ───────────────────────────────────────────────

def _classify_constraints(state: SiteEnergyState) -> str:
    """Esamina i campi user_* e ritorna la modalità: 'auto', 'constrained_annual', 'constrained_monthly'."""
    has_ac_ann  = state.user_autoconsumo_pct_annuo   is not None
    has_ac_mens = state.user_autoconsumo_pct_mensile is not None
    has_sup_ann = state.user_surplus_kwh_annuo       is not None
    has_sup_men = state.user_surplus_kwh_mensile     is not None

    if not any([has_ac_ann, has_ac_mens, has_sup_ann, has_sup_men]):
        return "auto"

    has_ac  = has_ac_ann or has_ac_mens
    has_sup = has_sup_ann or has_sup_men
    if has_ac and has_sup:
        raise ValueError(
            "Vincoli incompatibili: % autoconsumo e kWh immessi sono mutuamente "
            "esclusivi. Fornire uno solo dei due tipi."
        )
    if has_ac_ann and has_ac_mens:
        raise ValueError("Vincoli autoconsumo: forniti sia annuo che mensile. Sceglierne uno solo.")
    if has_sup_ann and has_sup_men:
        raise ValueError("Vincoli surplus: forniti sia annuo che mensile. Sceglierne uno solo.")

    if has_ac_mens:
        v = state.user_autoconsumo_pct_mensile
        if len(v) != 12 or any(x is None for x in v):
            raise ValueError("Vincolo autoconsumo mensile: devono essere forniti tutti e 12 i mesi.")
        return "constrained_monthly"
    if has_sup_men:
        v = state.user_surplus_kwh_mensile
        if len(v) != 12 or any(x is None for x in v):
            raise ValueError("Vincolo surplus mensile: devono essere forniti tutti e 12 i mesi.")
        return "constrained_monthly"
    return "constrained_annual"


def _user_to_target_autoconsumo(state: SiteEnergyState) -> dict:
    """
    Converte i vincoli utente in target uniformi espressi in kWh autoconsumo.

    Ritorna {"annuo": float|None, "mensile": list[float]|None, "warnings": list[str]}.
    """
    fv_ann  = sum(state.fv_mensile_pvgis_kwh) if state.fv_mensile_pvgis_kwh else 0.0
    fv_mens = state.fv_mensile_pvgis_kwh or [0.0] * 12
    out: dict = {"annuo": None, "mensile": None, "warnings": []}

    if state.user_autoconsumo_pct_annuo is not None:
        pct = state.user_autoconsumo_pct_annuo
        if not (0.0 <= pct <= 100.0):
            raise ValueError(f"% autoconsumo annuo fuori range [0,100]: {pct}")
        out["annuo"] = pct / 100.0 * fv_ann

    elif state.user_surplus_kwh_annuo is not None:
        sup = state.user_surplus_kwh_annuo
        if sup < 0:
            raise ValueError(f"Surplus annuo negativo: {sup}")
        ac = fv_ann - sup
        if ac < 0:
            ac = 0.0
            out["warnings"].append(
                f"Surplus dichiarato {sup:,.0f} kWh > FV prodotto {fv_ann:,.0f} kWh. "
                "Autoconsumo saturato a 0."
            )
        out["annuo"] = ac

    elif state.user_autoconsumo_pct_mensile is not None:
        out["mensile"] = []
        for m in range(12):
            pct = state.user_autoconsumo_pct_mensile[m]
            if not (0.0 <= pct <= 100.0):
                raise ValueError(f"% autoconsumo mese {m+1} fuori range [0,100]: {pct}")
            out["mensile"].append(pct / 100.0 * fv_mens[m])
        out["annuo"] = sum(out["mensile"])

    elif state.user_surplus_kwh_mensile is not None:
        out["mensile"] = []
        for m in range(12):
            sup = state.user_surplus_kwh_mensile[m]
            if sup < 0:
                raise ValueError(f"Surplus mese {m+1} negativo: {sup}")
            ac = fv_mens[m] - sup
            if ac < 0:
                ac = 0.0
                out["warnings"].append(
                    f"Mese {m+1}: surplus {sup:,.0f} kWh > FV {fv_mens[m]:,.0f} kWh. "
                    "Autoconsumo saturato a 0."
                )
            out["mensile"].append(ac)
        out["annuo"] = sum(out["mensile"])

    return out


# ── Funzione pubblica ─────────────────────────────────────────────────────────

def reconcile(state: SiteEnergyState) -> SiteEnergyState:
    """
    Ricalcola tutti i campi derivati di SiteEnergyState.

    Modalità supportate:
      "auto"                — profilo sintetico calibrato su F1/F2/F3 mensili + picchi
      "constrained_annual"  — vincolo annuo utente (% autoconsumo o kWh surplus)
      "constrained_monthly" — vincolo mensile completo (12 valori)

    Non modifica gli array di input in-place.
    """
    # ── 1. Validazione ────────────────────────────────────────────────────────
    if state.has_fv and state.fv_oraria_pvgis_kw is None:
        raise ValueError(
            "SiteEnergyState.fv_oraria_pvgis_kw è None ma has_fv=True. "
            "Fornire il profilo PVGIS orario (8760 valori) prima di chiamare reconcile()."
        )
    if state.has_fv and state.fv_oraria_pvgis_kw is not None:
        if len(state.fv_oraria_pvgis_kw) != 8760:
            raise ValueError(
                f"fv_oraria_pvgis_kw deve avere 8760 elementi, "
                f"trovati {len(state.fv_oraria_pvgis_kw)}."
            )

    state.reconcile_mode  = _classify_constraints(state)
    picchi_ok             = sum(state.picchi_mensili_kw) > 0
    state.assumptions_active = []

    # ── 2. Time index ─────────────────────────────────────────────────────────
    ti = _build_time_index(state.anno_riferimento)

    # ── 3. Espansione FV → quartoraria ────────────────────────────────────────
    if state.has_fv and state.fv_oraria_pvgis_kw is not None:
        fv_qh = np.repeat(state.fv_oraria_pvgis_kw, 4)[:_N_SLOTS].copy()
    else:
        fv_qh = np.zeros(_N_SLOTS)
    state.fv_profile_qh_kw = fv_qh

    # ── 4. Inferenza archetipo ────────────────────────────────────────────────
    arch_key, arch_score = _infer_archetype(state)
    state.archetype_inferred         = arch_key
    state.archetype_confidence_score = arch_score
    if state.archetype is None:
        state.archetype = arch_key

    state.assumptions_active.append(
        f"Archetipo inferito: {state.archetype_inferred} "
        f"(confidenza {arch_score*100:.0f}%)"
    )

    # ── 5. Calibrazione ───────────────────────────────────────────────────────
    target_ac = None
    if state.reconcile_mode in ("constrained_annual", "constrained_monthly"):
        target_ac = _user_to_target_autoconsumo(state)
        state.assumptions_active.extend(target_ac["warnings"])

    cal = _calibrate_load(state, ti, mode=state.reconcile_mode, target_ac=target_ac)
    state.calibration_loss = cal["loss"]

    state.assumptions_active.append(
        "Profilo sintetico calibrato su 36 vincoli (12 mesi × 3 fasce F1/F2/F3)"
    )
    if not picchi_ok:
        state.assumptions_active.append(
            "⚠ Picchi mensili non forniti — calibrazione picco disabilitata"
        )

    if state.reconcile_mode == "constrained_annual" and target_ac is not None:
        fv_ann_ref2 = sum(state.fv_mensile_pvgis_kwh) if state.fv_mensile_pvgis_kwh else 1.0
        state.assumptions_active.append(
            f"Vincolo utente attivo: autoconsumo annuo = "
            f"{target_ac['annuo']:,.0f} kWh "
            f"({target_ac['annuo'] / max(fv_ann_ref2, 1.0) * 100:.1f}% del FV prodotto)"
        )
    elif state.reconcile_mode == "constrained_monthly":
        state.assumptions_active.append(
            "Vincolo utente attivo: autoconsumo mensile dettagliato (12 mesi)"
        )

    # ── 6. Profilo load finale ────────────────────────────────────────────────
    load_qh = _build_synthetic_load(state, cal, ti)
    state.load_profile_qh_kw = load_qh

    # Aggiungi assunzione sulla forma giornaliera
    _shape_label = {
        "industrial_single_shift": "singola gobba (turno 8-17)",
        "industrial_double_shift": "doppia gobba (turno 6-22)",
        "industrial_continuous":   "plateau continuo",
        "commercial_office":       "singola gobba (uffici 9-17)",
        "mixed":                   "mista (media singolo+continuo)",
    }.get(state.archetype, state.archetype)
    state.assumptions_active.append(f"Forma giornaliera: {_shape_label}")

    # ── 7. Self-consumption ───────────────────────────────────────────────────
    ac_qh, surplus_qh, grid_qh = _compute_self_consumption(load_qh, fv_qh)
    state.autoconsumo_profile_qh_kw = ac_qh
    state.grid_profile_qh_kw        = grid_qh

    # ── 8. Aggregazione mensile e annua ───────────────────────────────────────
    load_monthly   = _aggregate_monthly(load_qh,    ti)
    ac_monthly     = _aggregate_monthly(ac_qh,      ti)
    surplus_monthly = _aggregate_monthly(surplus_qh, ti)
    grid_monthly   = _aggregate_monthly(grid_qh,    ti)

    # ── Hard enforcement vincoli utente ──────────────────────────────────────
    # NB: load_monthly, ac_monthly, surplus_monthly già calcolati sopra da _aggregate_monthly.
    # In auto mode (target_ac is None) questo blocco non li sovrascrive.
    prelievo_mensile = [
        state.prelievo_f1_mensile[m] + state.prelievo_f2_mensile[m] + state.prelievo_f3_mensile[m]
        for m in range(12)
    ]
    fv_mensile_ref = state.fv_mensile_pvgis_kwh or [0.0] * 12

    if target_ac is not None:
        if state.reconcile_mode == "constrained_annual":
            ac_model_ann = sum(ac_monthly)
            if ac_model_ann > 0:
                scale = target_ac["annuo"] / ac_model_ann
                ac_monthly = [v * scale for v in ac_monthly]
            else:
                fv_ann_ref = sum(fv_mensile_ref)
                ac_monthly = [
                    target_ac["annuo"] * fv_mensile_ref[m] / fv_ann_ref if fv_ann_ref > 0 else 0.0
                    for m in range(12)
                ]
            surplus_monthly = [max(0.0, fv_mensile_ref[m] - ac_monthly[m]) for m in range(12)]
            fabb_monthly    = [prelievo_mensile[m] + ac_monthly[m] for m in range(12)]
        elif state.reconcile_mode == "constrained_monthly":
            ac_monthly      = list(target_ac["mensile"])
            surplus_monthly = [max(0.0, fv_mensile_ref[m] - ac_monthly[m]) for m in range(12)]
            fabb_monthly    = [prelievo_mensile[m] + ac_monthly[m] for m in range(12)]
        else:
            fabb_monthly = [load_monthly[m] for m in range(12)]
    else:
        # modalità auto: ac_monthly e surplus_monthly intatti dall'aggregazione
        fabb_monthly = [load_monthly[m] for m in range(12)]

    # Source/confidence in base al mode
    if state.reconcile_mode == "auto":
        ac_source, ac_conf           = "estimated", "medium"
        fabb_source, fabb_conf       = "estimated", "medium"
        monthly_source, monthly_conf = "estimated", "medium"
    else:
        ac_source, ac_conf           = "user_input", "high"
        fabb_source, fabb_conf       = "derived",    "high"
        if state.reconcile_mode == "constrained_monthly":
            monthly_source, monthly_conf = "user_input", "high"
        else:
            monthly_source, monthly_conf = "estimated",  "medium"

    # Prepara TrackedValue mensili
    state.fabbisogno_mensile_kwh = [
        TrackedValue(v, fabb_source, fabb_conf) for v in fabb_monthly
    ]
    state.autoconsumo_fv_mensile_kwh = [
        TrackedValue(v, monthly_source, monthly_conf) for v in ac_monthly
    ]
    state.surplus_fv_mensile_kwh = [
        TrackedValue(v, monthly_source, monthly_conf) for v in surplus_monthly
    ]

    fabb_ann    = sum(fabb_monthly)
    ac_ann      = sum(ac_monthly)
    surplus_ann = sum(surplus_monthly)

    state.fabbisogno_annuo_kwh     = TrackedValue(fabb_ann,    fabb_source, fabb_conf)
    state.autoconsumo_fv_annuo_kwh = TrackedValue(ac_ann,      ac_source,   ac_conf)
    state.surplus_fv_annuo_kwh     = TrackedValue(surplus_ann, ac_source,   ac_conf)

    # % autoconsumo = autoconsumo / fv_prodotta (se FV presente)
    fv_ann = sum(state.fv_mensile_pvgis_kwh) if state.fv_mensile_pvgis_kwh else float(fv_qh.sum() * _SLOT_H)
    if fv_ann > 0:
        pct_ann = ac_ann / fv_ann * 100.0
        state.autoconsumo_pct_annuo = TrackedValue(pct_ann, ac_source, ac_conf)
        state.autoconsumo_pct_mensile = [
            TrackedValue(
                ac_monthly[m] / max(fv_mensile_ref[m], 1.0) * 100.0,
                monthly_source, monthly_conf,
            )
            for m in range(12)
        ]
    else:
        state.autoconsumo_pct_annuo   = TrackedValue(0.0, ac_source, ac_conf)
        state.autoconsumo_pct_mensile = [TrackedValue(0.0, ac_source, ac_conf)] * 12

    # ── 9. Diagnostica ────────────────────────────────────────────────────────
    prelievo_bolletta_ann = {
        "f1": sum(state.prelievo_f1_mensile),
        "f2": sum(state.prelievo_f2_mensile),
        "f3": sum(state.prelievo_f3_mensile),
    }

    # Errore per fascia mensile
    band_err_monthly = []
    for m in range(12):
        mm   = ti["month"] == (m + 1)
        errs = {}
        for band, tgt_list in (
            ("f1", state.prelievo_f1_mensile),
            ("f2", state.prelievo_f2_mensile),
            ("f3", state.prelievo_f3_mensile),
        ):
            stimato = float(grid_qh[mm & ti["band_masks"][band]].sum() * _SLOT_H)
            target  = tgt_list[m]
            errs[band] = (stimato - target) / max(target, 100.0) * 100.0
        band_err_monthly.append(errs)
    state.band_match_error_monthly = band_err_monthly

    # Errore medio annuo per fascia
    state.band_match_error_pct = {
        band: float(np.mean([e[band] for e in band_err_monthly]))
        for band in ("f1", "f2", "f3")
    }

    # Errore picco mensile
    state.peak_match_error_pct_monthly = []
    for m in range(12):
        mm  = ti["month"] == (m + 1)
        tgt = state.picchi_mensili_kw[m]
        if tgt > 0 and mm.any():
            stim = float(grid_qh[mm].max())
            err  = (stim - tgt) / tgt * 100.0
        else:
            err = 0.0
        state.peak_match_error_pct_monthly.append(err)

    # Confidenza complessiva
    state.overall_confidence = _compute_overall_confidence(state)

    # Assunzione stagionalità
    stag_inv = state.stagionalita_inverno
    stag_est = state.stagionalita_estate
    if stag_inv == "neutro" and stag_est == "neutro":
        state.assumptions_active.append("Stagionalità: neutra (nessuna modulazione stagionale)")
    else:
        state.assumptions_active.append(
            f"Stagionalità: inverno {stag_inv}, estate {stag_est} (±15% componente produttiva)"
        )

    return state

