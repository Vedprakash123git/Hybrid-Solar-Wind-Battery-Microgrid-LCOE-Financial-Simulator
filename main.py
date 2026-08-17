"""
main.py
=======
End-to-end run of the hybrid solar-wind-battery microgrid simulator:

  1. Simulate hourly solar + wind generation and a facility load for one
     representative year (8,760 hours).
  2. Run rule-based battery dispatch to compute the served/curtailed/
     unmet-load balance.
  3. Compute base-case LCOE and project NPV/IRR.
  4. Run a capacity-sizing sweep (solar/wind/battery mix) and a
     CapEx/OpEx/tariff financial sensitivity sweep.
  5. Save plots (dispatch week, generation mix, LCOE heatmap, sizing
     sensitivity) and CSV outputs to outputs/.

Run with:  python -m src.main   (from the project root)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import generation as gen
from src import battery as batt
from src import financial as fin
from src import scenarios as scn

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})

# ---------------------------------------------------------------------------
# 1. SITE / SYSTEM ASSUMPTIONS  (edit these to model a different project)
# ---------------------------------------------------------------------------
SITE = dict(
    latitude_deg=19.07,          # Mumbai, India (edit for your site)
    peak_load_kw=500.0,          # facility peak demand
)

SYSTEM = dict(
    solar_kw_dc=600.0,           # solar nameplate DC capacity
    wind_kw=400.0,               # wind fleet nameplate capacity
    battery_kwh=800.0,           # battery energy capacity
    battery_power_kw=250.0,      # battery power rating
    grid_tied=True,              # False = off-grid microgrid
)

COSTS = dict(
    solar_capex_per_kw=750.0,     # $/kW DC installed (utility-scale-ish, 2025-26 range)
    wind_capex_per_kw=1300.0,     # $/kW installed (small/mid wind fleet)
    battery_capex_per_kwh=280.0,  # $/kWh installed (Li-ion BESS, 2025-26 range)
    bos_soft_cost_frac=0.12,      # balance-of-system + soft costs, as frac of hardware capex
    solar_opex_per_kw_yr=12.0,    # $/kW-yr O&M
    wind_opex_per_kw_yr=40.0,     # $/kW-yr O&M
    battery_opex_per_kwh_yr=5.0,  # $/kWh-yr O&M
)

FINANCE = dict(
    discount_rate=0.08,
    project_life_years=25,
    tariff_per_kwh=0.09,          # PPA / feed-in tariff, or avoided grid cost
    opex_escalation_pct=2.5,
    solar_degradation_pct=0.5,
    wind_degradation_pct=0.3,
    battery_replacement_schedule={12: SYSTEM["battery_kwh"] * 150.0},  # $150/kWh mid-life augmentation in yr 12
)


def build_capex(solar_kw, wind_kw, battery_kwh):
    hardware = (
        solar_kw * COSTS["solar_capex_per_kw"]
        + wind_kw * COSTS["wind_capex_per_kw"]
        + battery_kwh * COSTS["battery_capex_per_kwh"]
    )
    return hardware * (1 + COSTS["bos_soft_cost_frac"])


def build_opex(solar_kw, wind_kw, battery_kwh):
    return (
        solar_kw * COSTS["solar_opex_per_kw_yr"]
        + wind_kw * COSTS["wind_opex_per_kw_yr"]
        + battery_kwh * COSTS["battery_opex_per_kwh_yr"]
    )


def run_base_case():
    print("=" * 70)
    print("HYBRID SOLAR-WIND-BATTERY MICROGRID SIMULATION - BASE CASE")
    print("=" * 70)

    # --- 1. Generation & load profiles -------------------------------------------------
    solar_kw = gen.solar_profile_kw(SYSTEM["solar_kw_dc"], latitude_deg=SITE["latitude_deg"], seed=1)
    wind_kw, wind_speed = gen.wind_profile_kw(SYSTEM["wind_kw"], seed=2)
    load_kw = gen.load_profile_kw(SITE["peak_load_kw"], seed=3)
    combined_gen_kw = solar_kw + wind_kw

    # --- 2. Battery dispatch -------------------------------------------------------------
    battery = batt.Battery(
        energy_capacity_kwh=SYSTEM["battery_kwh"],
        power_rating_kw=SYSTEM["battery_power_kw"],
    )
    dispatch_df = batt.dispatch(combined_gen_kw, load_kw, battery, grid_tied=SYSTEM["grid_tied"])
    summary = batt.dispatch_summary(dispatch_df, grid_tied=SYSTEM["grid_tied"])

    print("\n--- Annual Energy Balance ---")
    for k, v in summary.items():
        if "fraction" in k:
            print(f"  {k:32s}: {v:8.1%}")
        else:
            print(f"  {k:32s}: {v:12,.0f} kWh")

    # --- 3. Base-case financials -----------------------------------------------------------
    capex = build_capex(SYSTEM["solar_kw_dc"], SYSTEM["wind_kw"], SYSTEM["battery_kwh"])
    opex = build_opex(SYSTEM["solar_kw_dc"], SYSTEM["wind_kw"], SYSTEM["battery_kwh"])
    first_year_energy = summary["total_generation_kwh"] - summary["curtailed_kwh"]
    solar_share = solar_kw.sum() / combined_gen_kw.sum()

    lcoe, lcoe_detail = fin.lcoe_real(
        capex, opex, first_year_energy,
        FINANCE["discount_rate"], FINANCE["project_life_years"],
        opex_escalation_pct=FINANCE["opex_escalation_pct"],
        solar_degradation_pct=FINANCE["solar_degradation_pct"],
        wind_degradation_pct=FINANCE["wind_degradation_pct"],
        solar_share=solar_share,
        battery_replacement_schedule=FINANCE["battery_replacement_schedule"],
    )
    npv, cash_flows, irr = fin.project_npv(
        capex, opex, first_year_energy, FINANCE["tariff_per_kwh"],
        FINANCE["discount_rate"], FINANCE["project_life_years"],
        opex_escalation_pct=FINANCE["opex_escalation_pct"],
        solar_degradation_pct=FINANCE["solar_degradation_pct"],
        wind_degradation_pct=FINANCE["wind_degradation_pct"],
        solar_share=solar_share,
        battery_replacement_schedule=FINANCE["battery_replacement_schedule"],
    )

    print("\n--- Base-Case Financials ---")
    print(f"  {'Total CapEx':32s}: ${capex:14,.0f}")
    print(f"  {'Year-1 OpEx':32s}: ${opex:14,.0f}")
    print(f"  {'LCOE':32s}: ${lcoe:14.4f} /kWh")
    print(f"  {'Assumed tariff/PPA price':32s}: ${FINANCE['tariff_per_kwh']:14.4f} /kWh")
    print(f"  {'Project NPV':32s}: ${npv:14,.0f}")
    print(f"  {'Project IRR':32s}: {irr:14.2%}" if np.isfinite(irr) else f"  {'Project IRR':32s}: n/a")

    return dict(
        solar_kw=solar_kw, wind_kw=wind_kw, wind_speed=wind_speed, load_kw=load_kw,
        dispatch_df=dispatch_df, summary=summary, capex=capex, opex=opex,
        first_year_energy=first_year_energy, solar_share=solar_share,
        lcoe=lcoe, lcoe_detail=lcoe_detail, npv=npv, cash_flows=cash_flows, irr=irr,
    )


def run_sensitivity(base):
    print("\n" + "=" * 70)
    print("SENSITIVITY: CapEx x OpEx x Tariff sweep")
    print("=" * 70)

    sweep_df = scn.capex_opex_tariff_sweep(
        base_capex=base["capex"],
        base_opex=base["opex"],
        first_year_energy_kwh=base["first_year_energy"],
        discount_rate=FINANCE["discount_rate"],
        project_life_years=FINANCE["project_life_years"],
        capex_multipliers=(0.8, 0.9, 1.0, 1.1, 1.2),
        opex_multipliers=(0.8, 1.0, 1.2),
        tariff_values=(0.06, 0.08, 0.10, 0.12, 0.14),
        opex_escalation_pct=FINANCE["opex_escalation_pct"],
        solar_degradation_pct=FINANCE["solar_degradation_pct"],
        wind_degradation_pct=FINANCE["wind_degradation_pct"],
        solar_share=base["solar_share"],
    )
    sweep_df.to_csv(os.path.join(OUT_DIR, "capex_opex_tariff_sweep.csv"), index=False)
    print(f"  LCOE range across sweep: ${sweep_df.lcoe_per_kwh.min():.4f} - ${sweep_df.lcoe_per_kwh.max():.4f} /kWh")
    print(f"  NPV range across sweep:  ${sweep_df.npv.min():,.0f} - ${sweep_df.npv.max():,.0f}")

    print("\n" + "=" * 70)
    print("SENSITIVITY: Capacity sizing sweep (solar / wind / battery mix)")
    print("=" * 70)

    def generation_fn(solar_kw_dc, wind_kw_cap):
        s = gen.solar_profile_kw(solar_kw_dc, latitude_deg=SITE["latitude_deg"], seed=1)
        w, _ = gen.wind_profile_kw(wind_kw_cap, seed=2)
        return s + w, (s + w).sum()

    load_kw = base["load_kw"]

    def dispatch_fn(gen_profile, battery_kwh):
        battery = batt.Battery(energy_capacity_kwh=battery_kwh,
                                power_rating_kw=max(battery_kwh * 0.3, 1.0))
        df = batt.dispatch(gen_profile, load_kw, battery, grid_tied=SYSTEM["grid_tied"])
        return batt.dispatch_summary(df, grid_tied=SYSTEM["grid_tied"])

    def financial_fn(solar_kw_dc, wind_kw_cap, battery_kwh, dsum):
        capex = build_capex(solar_kw_dc, wind_kw_cap, battery_kwh)
        opex = build_opex(solar_kw_dc, wind_kw_cap, battery_kwh)
        first_yr_energy = dsum["total_generation_kwh"] - dsum["curtailed_kwh"]
        solar_frac = 0.5 if (solar_kw_dc + wind_kw_cap) == 0 else solar_kw_dc / (solar_kw_dc + wind_kw_cap + 1e-9)
        lcoe, _ = fin.lcoe_real(
            capex, opex, max(first_yr_energy, 1.0),
            FINANCE["discount_rate"], FINANCE["project_life_years"],
            opex_escalation_pct=FINANCE["opex_escalation_pct"],
            solar_degradation_pct=FINANCE["solar_degradation_pct"],
            wind_degradation_pct=FINANCE["wind_degradation_pct"],
            solar_share=solar_frac,
        )
        return dict(
            capex=capex, opex=opex,
            lcoe_per_kwh=lcoe,
            renewable_fraction=dsum["renewable_fraction_of_load"],
            unmet_load_kwh=dsum["unmet_load_kwh"],
        )

    sizing_df = scn.capacity_sizing_sweep(
        solar_kw_options=[300, 450, 600, 750, 900],
        wind_kw_options=[200, 300, 400, 500],
        battery_kwh_options=[400, 800, 1200],
        generation_fn=generation_fn,
        dispatch_fn=dispatch_fn,
        financial_fn=financial_fn,
    )
    sizing_df.to_csv(os.path.join(OUT_DIR, "capacity_sizing_sweep.csv"), index=False)
    best = sizing_df.loc[sizing_df.lcoe_per_kwh.idxmin()]
    print(f"  Lowest-LCOE configuration in sweep:")
    print(f"    Solar: {best.solar_kw:.0f} kW | Wind: {best.wind_kw:.0f} kW | Battery: {best.battery_kwh:.0f} kWh")
    print(f"    LCOE: ${best.lcoe_per_kwh:.4f}/kWh | Renewable fraction: {best.renewable_fraction:.1%}")

    return sweep_df, sizing_df


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

def make_plots(base, sweep_df, sizing_df):
    df = base["dispatch_df"]

    # --- Plot 1: One representative week of dispatch (peak summer week) ---
    week_start = 24 * 172  # ~ day 172, mid-year
    week = slice(week_start, week_start + 24 * 7)
    fig, ax = plt.subplots(figsize=(11, 5))
    t = np.arange(24 * 7)
    ax.fill_between(t, 0, df["generation_kw"].values[week], color="#2E86AB", alpha=0.25, label="Solar+Wind generation")
    ax.plot(t, df["load_kw"].values[week], color="#333333", lw=2, label="Load")
    ax.plot(t, df["battery_soc_kwh"].values[week] / SYSTEM["battery_kwh"] * SITE["peak_load_kw"],
            color="#F18F01", lw=1.5, ls="--", label="Battery SOC (scaled)")
    ax.bar(t, df["battery_discharge_kw"].values[week], color="#C73E1D", alpha=0.6, width=1.0, label="Battery discharge")
    ax.bar(t, -df["battery_charge_kw"].values[week], color="#3B7A57", alpha=0.6, width=1.0, label="Battery charge")
    ax.set_xlabel("Hour of week")
    ax.set_ylabel("kW")
    ax.set_title("Representative Week: Hybrid Dispatch Profile")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dispatch_week.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: Annual generation mix + energy balance ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["Solar", "Wind"]
    vals = [base["solar_kw"].sum(), base["wind_kw"].sum()]
    axes[0].pie(vals, labels=labels, autopct="%1.0f%%", colors=["#F6C90E", "#2E86AB"], startangle=90)
    axes[0].set_title("Annual Generation Mix (kWh)")

    s = base["summary"]
    balance_labels = ["Direct use", "Via battery", "Curtailed", "Unmet/Grid import"]
    balance_vals = [
        s["renewable_direct_use_kwh"],
        s["battery_discharge_kwh"],
        s["curtailed_kwh"],
        s["unmet_load_kwh"] + s["grid_import_kwh"],
    ]
    balance_vals = [max(v, 0) for v in balance_vals]
    axes[1].bar(balance_labels, balance_vals, color=["#3B7A57", "#F18F01", "#999999", "#C73E1D"])
    axes[1].set_ylabel("kWh / year")
    axes[1].set_title("Annual Energy Balance")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "generation_mix_and_balance.png"), dpi=150)
    plt.close(fig)

    # --- Plot 3: LCOE heatmap (CapEx multiplier vs Tariff, at OpEx=1.0x) ---
    pivot = sweep_df[sweep_df.opex_multiplier == 1.0].pivot(
        index="capex_multiplier", columns="tariff_per_kwh", values="lcoe_per_kwh"
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"${c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.1f}x" for r in pivot.index])
    ax.set_xlabel("Tariff / PPA price ($/kWh)")
    ax.set_ylabel("CapEx multiplier")
    ax.set_title("LCOE ($/kWh) Sensitivity\n(OpEx at base case)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=8.5)
    fig.colorbar(im, ax=ax, label="LCOE ($/kWh)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "lcoe_heatmap.png"), dpi=150)
    plt.close(fig)

    # --- Plot 4: NPV vs CapEx multiplier, one line per tariff ---
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = sweep_df[sweep_df.opex_multiplier == 1.0]
    for tariff, grp in sub.groupby("tariff_per_kwh"):
        grp = grp.sort_values("capex_multiplier")
        ax.plot(grp.capex_multiplier, grp.npv / 1000, marker="o", label=f"${tariff:.2f}/kWh")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("CapEx multiplier")
    ax.set_ylabel("Project NPV ($ thousands)")
    ax.set_title("NPV Sensitivity to CapEx and Tariff/PPA Price")
    ax.legend(title="Tariff", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "npv_sensitivity.png"), dpi=150)
    plt.close(fig)

    # --- Plot 5: Capacity sizing LCOE heatmap (solar vs wind, at mid battery) ---
    mid_batt = sorted(sizing_df.battery_kwh.unique())[len(sizing_df.battery_kwh.unique()) // 2]
    pivot2 = sizing_df[sizing_df.battery_kwh == mid_batt].pivot(
        index="wind_kw", columns="solar_kw", values="lcoe_per_kwh"
    )
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(pivot2.values, cmap="RdYlGn_r", aspect="auto", origin="lower")
    ax.set_xticks(range(len(pivot2.columns)))
    ax.set_xticklabels([f"{c:.0f}" for c in pivot2.columns])
    ax.set_yticks(range(len(pivot2.index)))
    ax.set_yticklabels([f"{r:.0f}" for r in pivot2.index])
    ax.set_xlabel("Solar capacity (kW DC)")
    ax.set_ylabel("Wind capacity (kW)")
    ax.set_title(f"LCOE ($/kWh) by Solar/Wind Sizing\n(Battery = {mid_batt:.0f} kWh)")
    for i in range(pivot2.shape[0]):
        for j in range(pivot2.shape[1]):
            ax.text(j, i, f"{pivot2.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="LCOE ($/kWh)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "capacity_sizing_heatmap.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved 5 plots + 2 CSVs to: {OUT_DIR}")


def write_report(base, sweep_df, sizing_df):
    best = sizing_df.loc[sizing_df.lcoe_per_kwh.idxmin()]
    path = os.path.join(OUT_DIR, "results_summary.md")
    with open(path, "w") as f:
        f.write(f"""# Hybrid Solar-Wind-Battery Microgrid — Results Summary

## Base-Case System
- Solar: {SYSTEM['solar_kw_dc']:.0f} kW DC | Wind: {SYSTEM['wind_kw']:.0f} kW | Battery: {SYSTEM['battery_kwh']:.0f} kWh / {SYSTEM['battery_power_kw']:.0f} kW
- Site peak load: {SITE['peak_load_kw']:.0f} kW | Mode: {'Grid-tied' if SYSTEM['grid_tied'] else 'Off-grid'}

## Annual Energy Balance
| Metric | Value |
|---|---|
| Total generation | {base['summary']['total_generation_kwh']:,.0f} kWh |
| Total load | {base['summary']['total_load_kwh']:,.0f} kWh |
| Renewable fraction of load | {base['summary']['renewable_fraction_of_load']:.1%} |
| Curtailed energy | {base['summary']['curtailed_kwh']:,.0f} kWh |
| Grid import | {base['summary']['grid_import_kwh']:,.0f} kWh |
| Grid export | {base['summary']['grid_export_kwh']:,.0f} kWh |

## Base-Case Financials
| Metric | Value |
|---|---|
| Total CapEx | ${base['capex']:,.0f} |
| Year-1 OpEx | ${base['opex']:,.0f} |
| **LCOE** | **${base['lcoe']:.4f} / kWh** |
| Assumed tariff/PPA | ${FINANCE['tariff_per_kwh']:.4f} / kWh |
| **Project NPV** | **${base['npv']:,.0f}** |
| Project IRR | {base['irr']:.2%} |
| Discount rate | {FINANCE['discount_rate']:.1%} |
| Project life | {FINANCE['project_life_years']} years |

## Sensitivity Highlights
- LCOE ranges from **${sweep_df.lcoe_per_kwh.min():.4f}** to **${sweep_df.lcoe_per_kwh.max():.4f}**/kWh
  across the CapEx (0.8x–1.2x) × OpEx (0.8x–1.2x) sweep.
- Project NPV turns negative when CapEx multiplier and tariff assumptions
  fall below breakeven — see `lcoe_heatmap.png` / `npv_sensitivity.png`.
- Best LCOE found in the capacity-sizing sweep: **${best.lcoe_per_kwh:.4f}/kWh**
  at Solar={best.solar_kw:.0f} kW, Wind={best.wind_kw:.0f} kW, Battery={best.battery_kwh:.0f} kWh
  (renewable fraction {best.renewable_fraction:.1%}).

## Files in this output folder
- `dispatch_week.png` — representative week of hourly dispatch
- `generation_mix_and_balance.png` — annual solar/wind mix and energy balance
- `lcoe_heatmap.png` — LCOE sensitivity to CapEx multiplier × tariff
- `npv_sensitivity.png` — NPV vs CapEx multiplier, by tariff
- `capacity_sizing_heatmap.png` — LCOE by solar/wind capacity mix
- `capex_opex_tariff_sweep.csv` — full sweep data (raw)
- `capacity_sizing_sweep.csv` — full sizing sweep data (raw)
""")
    print(f"Saved report to: {path}")


if __name__ == "__main__":
    base = run_base_case()
    sweep_df, sizing_df = run_sensitivity(base)
    make_plots(base, sweep_df, sizing_df)
    write_report(base, sweep_df, sizing_df)
    print("\nDone.")
