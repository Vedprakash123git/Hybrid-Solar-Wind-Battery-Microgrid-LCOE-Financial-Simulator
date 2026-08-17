"""
battery.py
==========
Rule-based hourly dispatch of a battery energy storage system (BESS) against
a net-generation-minus-load signal. This is the standard "greedy" dispatch
strategy used in most first-pass microgrid feasibility studies:

  1. If renewable generation exceeds load, use the surplus to charge the
     battery (up to its power and energy headroom). Anything left over is
     curtailed (or exported to the grid, if grid-tied).
  2. If load exceeds renewable generation, discharge the battery to cover
     the deficit (up to its power and energy availability). Anything left
     over is unmet load (off-grid) or imported from the grid (grid-tied).

This is intentionally simple and deterministic (no optimization / MPC) so
it's transparent and auditable -- exactly what you want for a first
capacity-sizing pass. A natural v2 extension is to replace `dispatch()`
with a linear program (e.g. via PuLP or scipy.optimize.linprog) that
minimizes cost or unmet load over a rolling horizon.
"""

import numpy as np
import pandas as pd


class Battery:
    def __init__(
        self,
        energy_capacity_kwh,
        power_rating_kw,
        round_trip_efficiency=0.90,
        soc_min_frac=0.10,
        soc_max_frac=0.95,
        initial_soc_frac=0.50,
        degradation_pct_per_year=1.5,
    ):
        self.energy_capacity_kwh = energy_capacity_kwh
        self.power_rating_kw = power_rating_kw
        self.rte = round_trip_efficiency
        self.charge_eff = np.sqrt(round_trip_efficiency)
        self.discharge_eff = np.sqrt(round_trip_efficiency)
        self.soc_min_frac = soc_min_frac
        self.soc_max_frac = soc_max_frac
        self.initial_soc_frac = initial_soc_frac
        self.degradation_pct_per_year = degradation_pct_per_year


def dispatch(generation_kw, load_kw, battery: Battery, grid_tied=False):
    """
    Runs hour-by-hour dispatch for one representative year.

    Parameters
    ----------
    generation_kw : array (8760,) combined solar+wind AC output per hour
    load_kw       : array (8760,) load per hour
    battery       : Battery instance
    grid_tied     : if True, any residual deficit is met by grid import and
                    any residual surplus is exported to the grid (both
                    tracked separately). If False (off-grid), residual
                    deficit becomes unserved/unmet load and residual surplus
                    is curtailed.

    Returns
    -------
    pandas.DataFrame with one row per hour and columns:
      generation_kw, load_kw, battery_soc_kwh, battery_charge_kw,
      battery_discharge_kw, curtailed_kw, unmet_load_kw,
      grid_import_kw, grid_export_kw
    """
    n = len(generation_kw)
    soc = battery.initial_soc_frac * battery.energy_capacity_kwh
    soc_min = battery.soc_min_frac * battery.energy_capacity_kwh
    soc_max = battery.soc_max_frac * battery.energy_capacity_kwh

    soc_series = np.zeros(n)
    charge_series = np.zeros(n)
    discharge_series = np.zeros(n)
    curtailed = np.zeros(n)
    unmet_load = np.zeros(n)
    grid_import = np.zeros(n)
    grid_export = np.zeros(n)

    for t in range(n):
        net = generation_kw[t] - load_kw[t]  # + surplus, - deficit

        if net >= 0:
            # Try to charge battery with surplus
            max_charge_power = min(battery.power_rating_kw, net)
            headroom_kwh = soc_max - soc
            max_charge_by_energy = headroom_kwh / battery.charge_eff  # kW for 1 hr
            charge_kw = max(0.0, min(max_charge_power, max_charge_by_energy))
            soc += charge_kw * battery.charge_eff
            leftover = net - charge_kw
            charge_series[t] = charge_kw
            if grid_tied:
                grid_export[t] = leftover
            else:
                curtailed[t] = leftover
        else:
            deficit = -net
            max_discharge_power = min(battery.power_rating_kw, deficit)
            available_kwh = soc - soc_min
            max_discharge_by_energy = available_kwh * battery.discharge_eff
            discharge_kw = max(0.0, min(max_discharge_power, max_discharge_by_energy))
            soc -= discharge_kw / battery.discharge_eff
            leftover = deficit - discharge_kw
            discharge_series[t] = discharge_kw
            if grid_tied:
                grid_import[t] = leftover
            else:
                unmet_load[t] = leftover

        soc_series[t] = soc

    df = pd.DataFrame(
        {
            "generation_kw": generation_kw,
            "load_kw": load_kw,
            "battery_soc_kwh": soc_series,
            "battery_charge_kw": charge_series,
            "battery_discharge_kw": discharge_series,
            "curtailed_kw": curtailed,
            "unmet_load_kw": unmet_load,
            "grid_import_kw": grid_import,
            "grid_export_kw": grid_export,
        }
    )
    return df


def dispatch_summary(df, grid_tied=False):
    """Quick scalar summary stats (annual energy totals in kWh) from a dispatch DataFrame."""
    summary = {
        "total_generation_kwh": df["generation_kw"].sum(),
        "total_load_kwh": df["load_kw"].sum(),
        "renewable_direct_use_kwh": (
            df["generation_kw"].sum()
            - df["battery_charge_kw"].sum()
            - df["curtailed_kw"].sum()
            - df["grid_export_kw"].sum()
        ),
        "battery_discharge_kwh": df["battery_discharge_kw"].sum(),
        "curtailed_kwh": df["curtailed_kw"].sum(),
        "unmet_load_kwh": df["unmet_load_kw"].sum(),
        "grid_import_kwh": df["grid_import_kw"].sum(),
        "grid_export_kwh": df["grid_export_kw"].sum(),
    }
    total_load = summary["total_load_kwh"]
    served_kwh = total_load - summary["unmet_load_kwh"]
    summary["renewable_fraction_of_load"] = (
        served_kwh - summary["grid_import_kwh"]
    ) / total_load if total_load > 0 else np.nan
    summary["load_served_fraction"] = served_kwh / total_load if total_load > 0 else np.nan
    return summary
