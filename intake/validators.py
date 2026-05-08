"""
intake/validators.py
Field validation and data-quality scoring for Block 1 intake.

Functions:
  validate_form(form)           → list[str]   hard errors — intake blocked if non-empty
  validate_form_warnings(form)  → list[str]   soft warnings — shown but do not block
  validate_billing(bills)       → list[str]   soft warnings on bill list
  compute_quality_level(form)   → int (1–5)
  determine_macro_case(form)    → str ("A"|"B"|"C"|"D")
"""

from __future__ import annotations


# ── Constants ─────────────────────────────────────────────────────────────────

_KWH_MIN      = 1_000
_KWH_MAX      = 10_000_000
_POT_MIN      = 10
_POT_MAX      = 5_000
_SPREAD_MIN   = -0.02
_SPREAD_MAX   = 0.30
_QUOTA_MIN    = 0.0
_QUOTA_MAX    = 50.0
_COST_KWH_MIN = 0.05
_COST_KWH_MAX = 0.50


# ── Public API ─────────────────────────────────────────────────────────────────

def validate_form(form: dict) -> list[str]:
    """
    Hard validation.  Returns a list of error strings.
    If any errors are present the intake cannot proceed.

    Fields are validated ONLY if present in the form dict, except
    consumo_annuo_kwh which is always required.
    spread_eur_kwh and quota_potenza_eur_kw_mese are optional at intake time
    because Block 1 may derive them later from billing data.
    """
    errors: list[str] = []

    # consumo_annuo_kwh — always required
    kwh = form.get("consumo_annuo_kwh", 0)
    if not (_KWH_MIN <= kwh <= _KWH_MAX):
        errors.append(
            f"Consumo annuo ({kwh:,.0f} kWh) fuori range [{_KWH_MIN:,}–{_KWH_MAX:,}]."
        )

    # picco_potenza_kw — validate only if present
    picco = form.get("picco_potenza_kw")
    if picco is not None and not (_POT_MIN <= picco <= _POT_MAX):
        errors.append(
            f"Potenza picco ({picco} kW) fuori range [{_POT_MIN}–{_POT_MAX}]."
        )

    # spread_eur_kwh — validate only if present (may be derived from billing)
    spread = form.get("spread_eur_kwh")
    if spread is not None and not (_SPREAD_MIN <= spread <= _SPREAD_MAX):
        errors.append(
            f"Spread fornitore ({spread:.3f} €/kWh) fuori range "
            f"[{_SPREAD_MIN}–{_SPREAD_MAX}]."
        )

    # quota_potenza_eur_kw_mese — validate only if present (may be derived from billing)
    quota = form.get("quota_potenza_eur_kw_mese")
    if quota is not None and not (_QUOTA_MIN <= quota <= _QUOTA_MAX):
        errors.append(
            f"Quota potenza ({quota} €/kW/mese) fuori range [{_QUOTA_MIN}–{_QUOTA_MAX}]."
        )

    # PV consistency — always checked when either field is present
    ha_pv = form.get("ha_pv", False)
    kwp   = form.get("kwp", 0)
    if ha_pv and kwp <= 0:
        errors.append("FV indicato come presente ma kWp = 0.")
    if not ha_pv and kwp > 0:
        errors.append("FV indicato come assente ma kWp > 0.")

    return errors


def validate_form_warnings(form: dict) -> list[str]:
    """
    Soft validation.  Returns a list of warning strings.
    Warnings are shown to the user but do not block intake.
    """
    warnings: list[str] = []

    picco = form.get("picco_potenza_kw")
    contr = form.get("potenza_contrattuale_kw")
    if picco is not None and contr is not None and contr > 0:
        if picco > contr * 1.5:
            warnings.append(
                f"Picco ({picco} kW) supera del 50% la potenza contrattuale ({contr} kW). "
                "Verifica i dati — potrebbe essere corretto per siti con avviamenti pesanti "
                "o picchi brevi non limitati contrattualmente."
            )

    return warnings


def validate_billing(bills: list[dict]) -> list[str]:
    """
    Soft validation on a list of normalized bill dicts.
    Returns warnings (non-blocking).
    Each bill is a dict as returned by bill_extractor.normalize_bill().
    """
    warnings: list[str] = []

    if not bills:
        warnings.append(
            "Nessuna bolletta inserita — consumo annuo è un'ipotesi manuale."
        )
        return warnings

    if len(bills) == 1:
        warnings.append(
            "Solo 1 bolletta disponibile. Consumo annuo extrapolato ×12 — "
            "stagionalità non catturata. Accuratezza indicativa ±40%."
        )

    for i, b in enumerate(bills, 1):
        kwh = b.get("consumo_kwh", 0)
        if kwh <= 0:
            warnings.append(f"Bolletta {i}: consumo kWh mancante o zero.")

        costo = b.get("costo_medio_energia_eur_kwh")
        if costo is not None and not (_COST_KWH_MIN <= costo <= _COST_KWH_MAX):
            warnings.append(
                f"Bolletta {i}: costo medio energia ({costo:.3f} €/kWh) "
                f"fuori range [{_COST_KWH_MIN}–{_COST_KWH_MAX}]. Verifica i dati."
            )

        f1 = b.get("f1_kwh")
        f2 = b.get("f2_kwh")
        f3 = b.get("f3_kwh")
        if all(v is not None for v in (f1, f2, f3)):
            bands_total = f1 + f2 + f3
            if kwh > 0 and abs(bands_total - kwh) > kwh * 0.05:
                warnings.append(
                    f"Bolletta {i}: F1+F2+F3 ({bands_total:,.0f} kWh) "
                    f"differisce dal totale ({kwh:,.0f} kWh) di più del 5%."
                )

    return warnings


def compute_quality_level(form: dict) -> int:
    """
    Returns data quality level 1–5.

    Mapping:
      L1 : annual kWh only (single bill or manual input)
      L2 : L1 + qualitative operational info (working hours and days present)
      L3 : L1-2 + monthly breakdown (12 complete bills OR F1/F2/F3 annual bands)
      L4 : real measured load curve uploaded
      L5 : L4 + well-defined PV context (tilt and azimuth explicitly provided)

    Note on L3: having 3–11 bills does NOT reach L3, because partial-year
    extrapolation still lacks seasonal detail.  F1/F2/F3 gives time-band
    information (not monthly), but is accepted as L3-equivalent for now.
    """
    has_real_curve   = bool(form.get("load_curve_file"))
    has_12_bills     = len(form.get("bills", [])) >= 12
    has_f1f2f3       = _has_f1f2f3_annual(form)
    has_monthly_data = has_12_bills or has_f1f2f3

    has_operational = (
        form.get("ore_lavoro_giorno") is not None
        and form.get("giorni_lavoro_settimana") is not None
    )
    has_pv_rich = (
        form.get("ha_pv")
        and form.get("kwp", 0) > 0
        and form.get("pv_tilt") is not None
        and form.get("pv_azimuth") is not None
    )

    if has_real_curve and has_pv_rich:
        return 5
    if has_real_curve:
        return 4
    if has_monthly_data:
        return 3
    if has_operational:
        return 2
    return 1


def determine_macro_case(form: dict) -> str:
    """
    Returns "A", "B", "C", or "D" based on available data.

    A: no PV, no real load curve
    B: existing PV, no real load curve
    C: no PV, real load curve present
    D: existing PV + real load curve present
    """
    has_pv    = form.get("ha_pv", False) and form.get("kwp", 0) > 0
    has_curve = bool(form.get("load_curve_file"))

    if has_curve and has_pv:
        return "D"
    if has_curve:
        return "C"
    if has_pv:
        return "B"
    return "A"


# ── Internal helpers ───────────────────────────────────────────────────────────

def _has_f1f2f3_annual(form: dict) -> bool:
    """True if all three annual F1/F2/F3 time-band values are present and non-zero."""
    f1 = form.get("f1_kwh")
    f2 = form.get("f2_kwh")
    f3 = form.get("f3_kwh")
    return all(v is not None and v > 0 for v in (f1, f2, f3))
