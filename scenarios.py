"""
scenarios.py
============
Runs multi-variable sensitivity sweeps across CapEx, OpEx, and tariff/PPA
price assumptions, and across capacity-sizing choices (solar/wind/battery
mix), producing tidy DataFrames ready for heatmaps or tornado charts.
"""

import itertools
import numpy as np
import pandas as pd

from . import financial as fin


def capex_opex_tariff_sweep(
    base_capex,
    base_opex,
    first_year_energy_kwh,
    discount_rate,
    project_life_years,
    capex_multipliers=(0.8, 0.9, 1.0, 1.1, 1.2),
    opex_multipliers=(0.8, 1.0, 1.2),
    tariff_values=(0.06, 0.08, 0.10, 0.12),
    **lcoe_kwargs,
):
    """
    Full-factorial sweep over CapEx multiplier x OpEx multiplier x tariff.
    Returns a DataFrame with one row per combination: LCOE and NPV.
    """
    rows = []
    for cx, ox, tariff in itertools.product(capex_multipliers, opex_multipliers, tariff_values):
        capex = base_capex * cx
        opex = base_opex * ox

        lcoe, _ = fin.lcoe_real(
            capex, opex, first_year_energy_kwh, discount_rate, project_life_years,
            **lcoe_kwargs,
        )
        npv, _, irr = fin.project_npv(
            capex, opex, first_year_energy_kwh, tariff, discount_rate, project_life_years,
            **lcoe_kwargs,
        )
        rows.append(
            {
                "capex_multiplier": cx,
                "opex_multiplier": ox,
                "tariff_per_kwh": tariff,
                "capex": capex,
                "opex_year1": opex,
                "lcoe_per_kwh": lcoe,
                "npv": npv,
                "irr": irr,
            }
        )
    return pd.DataFrame(rows)


def capacity_sizing_sweep(
    solar_kw_options,
    wind_kw_options,
    battery_kwh_options,
    generation_fn,
    dispatch_fn,
    financial_fn,
):
    """
    Generic capacity-sizing sweep. Caller supplies:
      generation_fn(solar_kw, wind_kw) -> (gen_profile_kw, first_year_kwh)
      dispatch_fn(gen_profile_kw, battery_kwh) -> dispatch summary dict
      financial_fn(solar_kw, wind_kw, battery_kwh, dispatch_summary) -> dict with lcoe, npv, capex

    Returns a tidy DataFrame, one row per (solar, wind, battery) combination.
    """
    rows = []
    for s_kw, w_kw, b_kwh in itertools.product(solar_kw_options, wind_kw_options, battery_kwh_options):
        gen_profile, first_year_kwh = generation_fn(s_kw, w_kw)
        dispatch_summ = dispatch_fn(gen_profile, b_kwh)
        result = financial_fn(s_kw, w_kw, b_kwh, dispatch_summ)
        rows.append(
            {
                "solar_kw": s_kw,
                "wind_kw": w_kw,
                "battery_kwh": b_kwh,
                **result,
            }
        )
    return pd.DataFrame(rows)
