# Hybrid Solar-Wind-Battery Microgrid — Results Summary

## Base-Case System
- Solar: 600 kW DC | Wind: 400 kW | Battery: 800 kWh / 250 kW
- Site peak load: 500 kW | Mode: Grid-tied

## Annual Energy Balance
| Metric | Value |
|---|---|
| Total generation | 1,865,113 kWh |
| Total load | 2,520,207 kWh |
| Renewable fraction of load | 70.8% |
| Curtailed energy | 0 kWh |
| Grid import | 734,810 kWh |
| Grid export | 52,468 kWh |

## Base-Case Financials
| Metric | Value |
|---|---|
| Total CapEx | $1,337,280 |
| Year-1 OpEx | $27,200 |
| **LCOE** | **$0.0906 / kWh** |
| Assumed tariff/PPA | $0.0900 / kWh |
| **Project NPV** | **$-10,736** |
| Project IRR | 7.90% |
| Discount rate | 8.0% |
| Project life | 25 years |

## Sensitivity Highlights
- LCOE ranges from **$0.0705** to **$0.1057**/kWh
  across the CapEx (0.8x–1.2x) × OpEx (0.8x–1.2x) sweep.
- Project NPV turns negative when CapEx multiplier and tariff assumptions
  fall below breakeven — see `lcoe_heatmap.png` / `npv_sensitivity.png`.
- Best LCOE found in the capacity-sizing sweep: **$0.0769/kWh**
  at Solar=900 kW, Wind=200 kW, Battery=400 kWh
  (renewable fraction 64.8%).

## Files in this output folder
- `dispatch_week.png` — representative week of hourly dispatch
- `generation_mix_and_balance.png` — annual solar/wind mix and energy balance
- `lcoe_heatmap.png` — LCOE sensitivity to CapEx multiplier × tariff
- `npv_sensitivity.png` — NPV vs CapEx multiplier, by tariff
- `capacity_sizing_heatmap.png` — LCOE by solar/wind capacity mix
- `capex_opex_tariff_sweep.csv` — full sweep data (raw)
- `capacity_sizing_sweep.csv` — full sizing sweep data (raw)
