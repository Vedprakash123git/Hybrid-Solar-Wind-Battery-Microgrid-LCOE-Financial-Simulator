"""
financial.py
============
Standard project-finance calculations for the microgrid: LCOE, NPV, and IRR,
built on an explicit year-by-year cash-flow model. This mirrors what
PySAM's `Lcoefcr`/`Cashloan` financial models compute, just implemented
directly so the whole pipeline has no external dependency beyond
numpy/pandas.

Definitions used:
  LCOE (real, discounted) =  NPV(annual costs, years 1..N)
                              -----------------------------
                              NPV(annual energy delivered, years 1..N)

    i.e. the constant $/kWh price that, if received on every kWh delivered
    for the life of the project, exactly recovers all discounted costs
    (CapEx + discounted OpEx). This is the standard "financial LCOE"
    definition used across NREL/PySAM and utility-scale project finance.

  NPV = -CapEx + sum_{t=1..N} (Revenue_t - OpEx_t) / (1 + r)^t

    where Revenue_t = energy delivered in year t * tariff (or PPA price),
    optionally escalated, and energy delivered degrades year over year to
    reflect panel/turbine/battery aging.
"""

import numpy as np


def annual_energy_series(
    first_year_energy_kwh,
    project_life_years,
    solar_degradation_pct=0.5,
    wind_degradation_pct=0.3,
    solar_share=0.5,
):
    """
    Builds a (project_life_years,) array of annual delivered energy (kWh),
    degrading a blended solar/wind rate each year off the first-year value.
    `solar_share` is the fraction of first-year energy attributable to
    solar (rest assumed wind) purely to blend the two degradation rates.
    """
    blended_degradation = solar_share * solar_degradation_pct + (1 - solar_share) * wind_degradation_pct
    years = np.arange(1, project_life_years + 1)
    factor = (1 - blended_degradation / 100.0) ** (years - 1)
    return first_year_energy_kwh * factor


def lcoe_real(
    capex,
    first_year_opex,
    first_year_energy_kwh,
    discount_rate,
    project_life_years,
    opex_escalation_pct=2.5,
    solar_degradation_pct=0.5,
    wind_degradation_pct=0.3,
    solar_share=0.5,
    battery_replacement_schedule=None,
):
    """
    Computes LCOE ($/kWh) using the discounted-cost / discounted-energy
    definition.

    battery_replacement_schedule: optional dict {year: cost} for one-off
    battery augmentation/replacement capex in specific years (e.g. a
    mid-life battery swap), added to that year's cost stream.
    """
    years = np.arange(1, project_life_years + 1)
    energy = annual_energy_series(
        first_year_energy_kwh, project_life_years,
        solar_degradation_pct, wind_degradation_pct, solar_share,
    )
    opex = first_year_opex * (1 + opex_escalation_pct / 100.0) ** (years - 1)

    if battery_replacement_schedule:
        for yr, cost in battery_replacement_schedule.items():
            if 1 <= yr <= project_life_years:
                opex[yr - 1] += cost

    discount_factors = 1.0 / (1 + discount_rate) ** years

    npv_costs = capex + np.sum(opex * discount_factors)
    npv_energy = np.sum(energy * discount_factors)

    lcoe = npv_costs / npv_energy
    return lcoe, {
        "annual_energy_kwh": energy,
        "annual_opex": opex,
        "npv_costs": npv_costs,
        "npv_energy_kwh": npv_energy,
    }


def project_npv(
    capex,
    first_year_opex,
    first_year_energy_kwh,
    tariff_per_kwh,
    discount_rate,
    project_life_years,
    tariff_escalation_pct=0.0,
    opex_escalation_pct=2.5,
    solar_degradation_pct=0.5,
    wind_degradation_pct=0.3,
    solar_share=0.5,
    battery_replacement_schedule=None,
):
    """
    Computes project NPV ($) given a tariff/PPA price per kWh (either an
    avoided-cost of grid power for a bill-savings analysis, or a feed-in
    tariff / PPA price for revenue). Returns (npv, cash_flow_series, irr).
    """
    years = np.arange(1, project_life_years + 1)
    energy = annual_energy_series(
        first_year_energy_kwh, project_life_years,
        solar_degradation_pct, wind_degradation_pct, solar_share,
    )
    opex = first_year_opex * (1 + opex_escalation_pct / 100.0) ** (years - 1)
    if battery_replacement_schedule:
        for yr, cost in battery_replacement_schedule.items():
            if 1 <= yr <= project_life_years:
                opex[yr - 1] += cost

    tariff = tariff_per_kwh * (1 + tariff_escalation_pct / 100.0) ** (years - 1)
    revenue = energy * tariff
    net_cash_flow = revenue - opex

    discount_factors = 1.0 / (1 + discount_rate) ** years
    npv = -capex + np.sum(net_cash_flow * discount_factors)

    full_cash_flows = np.concatenate(([-capex], net_cash_flow))
    irr = _irr(full_cash_flows)

    return npv, full_cash_flows, irr


def _irr(cash_flows, guess=0.08, tol=1e-6, max_iter=200):
    """Simple Newton's-method IRR solver (avoids a numpy-financial dependency)."""
    rate = guess
    for _ in range(max_iter):
        periods = np.arange(len(cash_flows))
        npv = np.sum(cash_flows / (1 + rate) ** periods)
        d_npv = np.sum(-periods * cash_flows / (1 + rate) ** (periods + 1))
        if abs(d_npv) < 1e-12:
            break
        new_rate = rate - npv / d_npv
        if not np.isfinite(new_rate):
            return np.nan
        if abs(new_rate - rate) < tol:
            return new_rate
        rate = new_rate
    return rate if abs(npv) < 1 else np.nan
