"""
generation.py
=============
Synthesizes realistic 8,760-hour (one full year) generation profiles for a
solar PV array and a wind turbine fleet from a small set of site parameters.

This plays the role that a PySAM `Pvwattsv8` + `Windpower` run would play,
but is self-contained (no external weather files / SSC binary needed), which
makes the whole pipeline runnable anywhere. The physics is simplified but
directionally correct:

  Solar:
    - Clear-sky irradiance modeled from solar geometry (declination, hour
      angle) for the site's latitude -> a daily bell curve that widens/
      narrows and shifts amplitude with season.
    - Stochastic cloud-cover attenuation (autocorrelated, so cloudy days
      cluster instead of flickering hour to hour, like real weather).
    - DC->AC derate (inverter efficiency, wiring, soiling) via a single
      `system_derate` factor, matching PVWatts' "derate" convention.

  Wind:
    - Hourly wind speed drawn from a Weibull distribution (the standard
      wind-resource distribution) with a mild diurnal (windier at night/
      early morning, a well-documented boundary-layer effect) and seasonal
      (windier in winter) pattern layered on top, plus autocorrelation so
      wind speed doesn't teleport between hours.
    - Speed -> power via a standard 3-region turbine power curve
      (cut-in, cubic ramp to rated, flat rated, cut-out).

Swap-in point for real data / PySAM:
  Replace `solar_profile_kw()` / `wind_profile_kw()` internals with a call to
  `PySAM.Pvwattsv8` / `PySAM.Windpower` fed by an actual TMY weather file for
  the site, or with historical wind-speed data run through your EN 639 wind
  forecasting model. Everything downstream (battery dispatch, LCOE, NPV)
  is agnostic to where the 8,760-length arrays came from.
"""

import numpy as np


HOURS_PER_YEAR = 8760


def _autocorrelated_noise(n_hours, rng, tau_hours=6.0, sigma=1.0):
    """
    Generate mean-zero AR(1) noise with a given correlation time constant.
    Used so that cloud cover / wind gusts persist over several hours instead
    of being independent white noise each hour (which looks unrealistic).
    """
    phi = np.exp(-1.0 / tau_hours)
    noise = np.zeros(n_hours)
    innovation_sigma = sigma * np.sqrt(1 - phi**2)
    eps = rng.normal(0, innovation_sigma, n_hours)
    for t in range(1, n_hours):
        noise[t] = phi * noise[t - 1] + eps[t]
    return noise


def solar_profile_kw(
    capacity_kw_dc,
    latitude_deg=19.07,       # default: Mumbai
    system_derate=0.86,       # PVWatts-style combined derate (inverter, wiring, soiling)
    cloud_sigma=0.28,         # cloud-attenuation volatility
    seed=1,
):
    """
    Returns an (8760,) array of AC solar output in kW for a system with the
    given DC nameplate capacity.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(HOURS_PER_YEAR)
    day_of_year = hours // 24
    hour_of_day = hours % 24

    lat = np.radians(latitude_deg)
    declination = np.radians(23.45) * np.sin(2 * np.pi * (284 + day_of_year) / 365.0)

    # Hour angle: 0 at solar noon, +/-180 deg over the day
    hour_angle = np.radians(15.0 * (hour_of_day - 12.0))

    # Solar elevation angle from standard solar-geometry formula
    sin_elev = (
        np.sin(lat) * np.sin(declination)
        + np.cos(lat) * np.cos(declination) * np.cos(hour_angle)
    )
    elevation = np.arcsin(np.clip(sin_elev, -1, 1))
    elevation_deg = np.degrees(elevation)

    # Clear-sky irradiance proxy: zero below horizon, scales with sin(elevation)
    # (a simplified but standard proxy for clear-sky global horizontal irradiance)
    clear_sky = np.clip(sin_elev, 0, None) ** 1.15

    # Cloud attenuation: autocorrelated noise -> multiplicative factor in [0.15, 1.0]
    cloud_noise = _autocorrelated_noise(HOURS_PER_YEAR, rng, tau_hours=5.0, sigma=cloud_sigma)
    cloud_factor = np.clip(1.0 - np.abs(cloud_noise), 0.15, 1.0)
    # Occasional monsoon-like heavy-cloud stretches for realism
    storm_mask = rng.random(HOURS_PER_YEAR) < 0.01
    storm_decay = np.exp(-np.abs(np.arange(-24, 24)) / 6.0)
    storm_series = np.zeros(HOURS_PER_YEAR)
    for idx in np.where(storm_mask)[0]:
        lo, hi = max(0, idx - 24), min(HOURS_PER_YEAR, idx + 24)
        storm_series[lo:hi] += storm_decay[lo - idx + 24: hi - idx + 24]
    cloud_factor *= np.clip(1.0 - 0.6 * np.clip(storm_series, 0, 1), 0.25, 1.0)

    ac_kw = capacity_kw_dc * clear_sky * cloud_factor * system_derate
    return np.clip(ac_kw, 0, capacity_kw_dc)  # inverter clipping ceiling


def wind_profile_kw(
    capacity_kw,
    weibull_k=2.0,
    weibull_scale_ms=7.5,     # mean wind speed ~ scale * Gamma(1+1/k)
    cut_in_ms=3.5,
    rated_ms=12.0,
    cut_out_ms=25.0,
    seed=2,
):
    """
    Returns an (8760,) array of wind-farm AC output in kW for a fleet with
    the given nameplate capacity, using a standard cut-in/rated/cut-out
    power curve driven by an autocorrelated Weibull-ish wind-speed series.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(HOURS_PER_YEAR)
    hour_of_day = hours % 24
    day_of_year = hours // 24

    # Base Weibull draw (via inverse-CDF on a uniform, then smoothed/correlated)
    u = rng.random(HOURS_PER_YEAR)
    base_speed = weibull_scale_ms * (-np.log(1 - u)) ** (1.0 / weibull_k)

    # Smooth it so wind speed doesn't jump discontinuously hour to hour:
    # blend each draw with an autocorrelated process anchored at the same mean.
    corr = _autocorrelated_noise(HOURS_PER_YEAR, rng, tau_hours=4.0, sigma=1.0)
    smoothed_speed = base_speed + corr * (weibull_scale_ms * 0.25)

    # Diurnal effect: boundary layer decouples at night -> higher wind speeds
    # overnight/early morning, lower in the afternoon.
    diurnal = 1.0 - 0.12 * np.cos(2 * np.pi * (hour_of_day - 4) / 24.0)
    # Seasonal effect: mild winter-peak, summer-trough pattern
    seasonal = 1.0 + 0.10 * np.cos(2 * np.pi * (day_of_year - 15) / 365.0)

    wind_speed = np.clip(smoothed_speed * diurnal * seasonal, 0, None)

    # Standard cubic-ramp power curve
    power_frac = np.zeros(HOURS_PER_YEAR)
    ramp_mask = (wind_speed >= cut_in_ms) & (wind_speed < rated_ms)
    rated_mask = (wind_speed >= rated_ms) & (wind_speed < cut_out_ms)
    power_frac[ramp_mask] = (
        (wind_speed[ramp_mask] ** 3 - cut_in_ms**3) / (rated_ms**3 - cut_in_ms**3)
    )
    power_frac[rated_mask] = 1.0
    power_frac = np.clip(power_frac, 0, 1)

    ac_kw = capacity_kw * power_frac
    return ac_kw, wind_speed


def load_profile_kw(
    peak_load_kw,
    base_load_frac=0.35,
    seed=3,
):
    """
    Synthesizes a representative commercial/microgrid load profile: a base
    load plus a daytime working-hours bump, a mild weekday/weekend
    difference, and small random noise. Replace with a real metered load
    (e.g. a facility's interval data) when available.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(HOURS_PER_YEAR)
    hour_of_day = hours % 24
    day_of_week = (hours // 24) % 7  # 0=Mon ... 5,6 = weekend

    daytime_bump = np.clip(np.sin(np.pi * (hour_of_day - 7) / 14.0), 0, 1) ** 0.8
    weekend_factor = np.where(day_of_week >= 5, 0.6, 1.0)

    base = base_load_frac * peak_load_kw
    variable = (1 - base_load_frac) * peak_load_kw * daytime_bump * weekend_factor
    noise = rng.normal(0, 0.04 * peak_load_kw, HOURS_PER_YEAR)

    load = np.clip(base + variable + noise, 0.05 * peak_load_kw, peak_load_kw)
    return load
