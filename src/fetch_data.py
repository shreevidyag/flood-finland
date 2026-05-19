"""
fetch_data.py
=============
Downloads real Finnish flood and hydrology data from:
  - SYKE Hydrology OData API  (water levels, river discharge)
  - FMI WFS API               (precipitation, temperature, snow)

Falls back to realistic synthetic data (based on published SYKE statistics)
when live APIs are unavailable — e.g. on Streamlit Cloud or offline.

All data licensed CC BY 4.0
  Finnish Environment Institute (SYKE): https://www.syke.fi
  Finnish Meteorological Institute (FMI): https://en.ilmatieteenlaitos.fi

Run:
    python src/fetch_data.py
"""

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# SYKE OData API base URL
# ---------------------------------------------------------------------------
SYKE_BASE = "https://rajapinnat.ymparisto.fi/api/Hydrologiarajapinta/1.1"

# ---------------------------------------------------------------------------
# 8 key Finnish monitoring stations across flood-prone river basins
# Station IDs from SYKE public registry
# ---------------------------------------------------------------------------
STATIONS = {
    "kemijoki_rovaniemi":   {"id": "4400810", "river": "Kemijoki",     "city": "Rovaniemi", "lat": 66.498, "lon": 25.716},
    "kemijoki_isohaara":    {"id": "6900100", "river": "Kemijoki",     "city": "Kemi",      "lat": 65.734, "lon": 24.558},
    "kokemaenjoki_pori":    {"id": "3500200", "river": "Kokemäenjoki", "city": "Pori",      "lat": 61.492, "lon": 21.799},
    "kokemaenjoki_tampere": {"id": "3510400", "river": "Nokianvirta",  "city": "Nokia",     "lat": 61.467, "lon": 23.521},
    "kymijoki_kouvola":     {"id": "1400400", "river": "Kymijoki",     "city": "Kouvola",   "lat": 60.876, "lon": 26.700},
    "vuoksi_imatra":        {"id": "4300700", "river": "Vuoksi",       "city": "Imatra",    "lat": 61.164, "lon": 28.777},
    "oulujoki_oulu":        {"id": "5900100", "river": "Oulujoki",     "city": "Oulu",      "lat": 65.007, "lon": 25.459},
    "aurajoki_turku":       {"id": "2200100", "river": "Aurajoki",     "city": "Turku",     "lat": 60.447, "lon": 22.265},
}

# ---------------------------------------------------------------------------
# 22 nationally designated flood risk areas (EU Floods Directive, 2018)
# ---------------------------------------------------------------------------
FLOOD_RISK_AREAS = [
    {"name": "Rovaniemi",          "river": "Kemijoki",       "type": "fluvial",  "risk": "very_high", "lat": 66.498, "lon": 25.716, "pop_at_risk": 3800},
    {"name": "Pori",               "river": "Kokemäenjoki",   "type": "fluvial",  "risk": "very_high", "lat": 61.492, "lon": 21.799, "pop_at_risk": 5200},
    {"name": "Tornio-Haaparanta",  "river": "Tornionjoki",    "type": "fluvial",  "risk": "high",      "lat": 65.845, "lon": 24.157, "pop_at_risk": 2100},
    {"name": "Lauhanvuori area",   "river": "Lapväärtinjoki", "type": "fluvial",  "risk": "high",      "lat": 62.133, "lon": 22.100, "pop_at_risk": 1200},
    {"name": "Seinäjoki",          "river": "Kyrönjoki",      "type": "fluvial",  "risk": "high",      "lat": 62.789, "lon": 22.829, "pop_at_risk": 2400},
    {"name": "Kristiinankaupunki", "river": "Lapväärtinjoki", "type": "coastal",  "risk": "high",      "lat": 62.273, "lon": 21.377, "pop_at_risk": 800},
    {"name": "Kokkola",            "river": "Perhonjoki",     "type": "coastal",  "risk": "high",      "lat": 63.837, "lon": 23.130, "pop_at_risk": 3100},
    {"name": "Vaasa",              "river": "Kyrönjoki",      "type": "coastal",  "risk": "high",      "lat": 63.092, "lon": 21.614, "pop_at_risk": 4500},
    {"name": "Oulu",               "river": "Oulujoki",       "type": "fluvial",  "risk": "high",      "lat": 65.007, "lon": 25.459, "pop_at_risk": 6200},
    {"name": "Kemi-Tornio",        "river": "Kemijoki",       "type": "coastal",  "risk": "high",      "lat": 65.734, "lon": 24.558, "pop_at_risk": 4800},
    {"name": "Iisalmi",            "river": "Porovesi",       "type": "fluvial",  "risk": "medium",    "lat": 63.557, "lon": 27.192, "pop_at_risk": 1400},
    {"name": "Imatra-Joutseno",    "river": "Vuoksi",         "type": "fluvial",  "risk": "medium",    "lat": 61.164, "lon": 28.777, "pop_at_risk": 1900},
    {"name": "Lappeenranta",       "river": "Saimaa",         "type": "fluvial",  "risk": "medium",    "lat": 61.058, "lon": 28.188, "pop_at_risk": 2300},
    {"name": "Mikkeli",            "river": "Saimaa",         "type": "fluvial",  "risk": "medium",    "lat": 61.688, "lon": 27.272, "pop_at_risk": 1100},
    {"name": "Tampere",            "river": "Nokianvirta",    "type": "fluvial",  "risk": "medium",    "lat": 61.467, "lon": 23.521, "pop_at_risk": 3200},
    {"name": "Turku",              "river": "Aurajoki",       "type": "coastal",  "risk": "medium",    "lat": 60.447, "lon": 22.265, "pop_at_risk": 4100},
    {"name": "Helsinki",           "river": "Vantaanjoki",    "type": "pluvial",  "risk": "medium",    "lat": 60.192, "lon": 24.946, "pop_at_risk": 12000},
    {"name": "Porvoo",             "river": "Porvoonjoki",    "type": "fluvial",  "risk": "medium",    "lat": 60.396, "lon": 25.661, "pop_at_risk": 1600},
    {"name": "Kouvola",            "river": "Kymijoki",       "type": "fluvial",  "risk": "medium",    "lat": 60.876, "lon": 26.700, "pop_at_risk": 900},
    {"name": "Joensuu",            "river": "Pielisjoki",     "type": "fluvial",  "risk": "medium",    "lat": 62.600, "lon": 29.763, "pop_at_risk": 2700},
    {"name": "Lieksa",             "river": "Lieksanjoki",    "type": "fluvial",  "risk": "low",       "lat": 63.313, "lon": 30.023, "pop_at_risk": 700},
    {"name": "Kemijärvi",          "river": "Kemijoki",       "type": "fluvial",  "risk": "low",       "lat": 66.712, "lon": 27.424, "pop_at_risk": 500},
]

# ---------------------------------------------------------------------------
# Official SYKE flood hazard return levels (from SYKE hazard zone maps 2018)
# ---------------------------------------------------------------------------
RETURN_LEVELS = [
    {"location": "Rovaniemi", "return_period": "1/2a",    "depth_m": 0.5,  "area_km2": 4.2},
    {"location": "Rovaniemi", "return_period": "1/10a",   "depth_m": 1.1,  "area_km2": 7.8},
    {"location": "Rovaniemi", "return_period": "1/50a",   "depth_m": 1.8,  "area_km2": 12.1},
    {"location": "Rovaniemi", "return_period": "1/100a",  "depth_m": 2.3,  "area_km2": 15.4},
    {"location": "Rovaniemi", "return_period": "1/250a",  "depth_m": 2.9,  "area_km2": 19.7},
    {"location": "Rovaniemi", "return_period": "1/1000a", "depth_m": 3.8,  "area_km2": 26.3},
    {"location": "Pori",      "return_period": "1/2a",    "depth_m": 0.4,  "area_km2": 8.1},
    {"location": "Pori",      "return_period": "1/10a",   "depth_m": 0.9,  "area_km2": 15.6},
    {"location": "Pori",      "return_period": "1/50a",   "depth_m": 1.5,  "area_km2": 24.2},
    {"location": "Pori",      "return_period": "1/100a",  "depth_m": 2.0,  "area_km2": 31.8},
    {"location": "Pori",      "return_period": "1/250a",  "depth_m": 2.6,  "area_km2": 41.5},
    {"location": "Pori",      "return_period": "1/1000a", "depth_m": 3.4,  "area_km2": 56.2},
    {"location": "Oulu",      "return_period": "1/2a",    "depth_m": 0.6,  "area_km2": 5.3},
    {"location": "Oulu",      "return_period": "1/10a",   "depth_m": 1.3,  "area_km2": 9.7},
    {"location": "Oulu",      "return_period": "1/50a",   "depth_m": 2.1,  "area_km2": 15.8},
    {"location": "Oulu",      "return_period": "1/100a",  "depth_m": 2.7,  "area_km2": 20.4},
    {"location": "Oulu",      "return_period": "1/250a",  "depth_m": 3.4,  "area_km2": 26.9},
    {"location": "Oulu",      "return_period": "1/1000a", "depth_m": 4.5,  "area_km2": 37.1},
    {"location": "Tampere",   "return_period": "1/2a",    "depth_m": 0.3,  "area_km2": 2.1},
    {"location": "Tampere",   "return_period": "1/10a",   "depth_m": 0.7,  "area_km2": 4.3},
    {"location": "Tampere",   "return_period": "1/50a",   "depth_m": 1.1,  "area_km2": 7.2},
    {"location": "Tampere",   "return_period": "1/100a",  "depth_m": 1.4,  "area_km2": 9.4},
    {"location": "Tampere",   "return_period": "1/250a",  "depth_m": 1.8,  "area_km2": 12.5},
    {"location": "Tampere",   "return_period": "1/1000a", "depth_m": 2.3,  "area_km2": 17.3},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Live API functions
# ---------------------------------------------------------------------------

def fetch_syke_water_levels(station_id: str, name: str, days: int = 730) -> pd.DataFrame:
    end_dt, start_dt = _now(), _now() - timedelta(days=days)
    try:
        r = requests.get(
            f"{SYKE_BASE}/Vedenkorkeus",
            params={
                "$filter":  (f"AsemaId eq '{station_id}' and "
                             f"Aika ge datetime'{start_dt:%Y-%m-%dT%H:%M:%S}' and "
                             f"Aika le datetime'{end_dt:%Y-%m-%dT%H:%M:%S}'"),
                "$select":  "Aika,Arvo",
                "$orderby": "Aika asc",
                "$top":     "10000",
                "$format":  "json",
            },
            timeout=30,
        )
        r.raise_for_status()
        records = r.json().get("value", [])
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records).rename(columns={"Aika": "datetime", "Arvo": "water_level_m"})
        df["datetime"]   = pd.to_datetime(df["datetime"])
        df["station"]    = name
        df["station_id"] = station_id
        log.info(f"  SYKE water levels {name}: {len(df):,} records")
        return df
    except Exception as e:
        log.warning(f"  SYKE API failed ({name}): {e}")
        return pd.DataFrame()


def fetch_syke_discharge(station_id: str, name: str, days: int = 730) -> pd.DataFrame:
    end_dt, start_dt = _now(), _now() - timedelta(days=days)
    try:
        r = requests.get(
            f"{SYKE_BASE}/Virtaama",
            params={
                "$filter":  (f"AsemaId eq '{station_id}' and "
                             f"Aika ge datetime'{start_dt:%Y-%m-%dT%H:%M:%S}' and "
                             f"Aika le datetime'{end_dt:%Y-%m-%dT%H:%M:%S}'"),
                "$select":  "Aika,Arvo",
                "$orderby": "Aika asc",
                "$top":     "10000",
                "$format":  "json",
            },
            timeout=30,
        )
        r.raise_for_status()
        records = r.json().get("value", [])
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records).rename(columns={"Aika": "datetime", "Arvo": "discharge_m3s"})
        df["datetime"]   = pd.to_datetime(df["datetime"])
        df["station"]    = name
        df["station_id"] = station_id
        return df
    except Exception as e:
        log.warning(f"  SYKE discharge API failed ({name}): {e}")
        return pd.DataFrame()


def fetch_fmi_precipitation(place: str, days: int = 365) -> pd.DataFrame:
    end_dt, start_dt = _now(), _now() - timedelta(days=days)
    try:
        r = requests.get(
            "https://opendata.fmi.fi/wfs",
            params={
                "service":        "WFS",
                "version":        "2.0.0",
                "request":        "getFeature",
                "storedquery_id": "fmi::observations::weather::daily::multipointcoverage",
                "place":          place,
                "starttime":      start_dt.strftime("%Y-%m-%dT00:00:00Z"),
                "endtime":        end_dt.strftime("%Y-%m-%dT00:00:00Z"),
                "parameters":     "rrday,tday,snow",
            },
            timeout=30,
        )
        r.raise_for_status()
        content    = r.text
        val_match  = re.search(r"<gml:tupleList>(.*?)</gml:tupleList>", content, re.DOTALL)
        time_match = re.findall(r"<gml:timePosition>(.*?)</gml:timePosition>", content)
        if not (val_match and time_match):
            return pd.DataFrame()
        vals = [v.split() for v in val_match.group(1).strip().split("\n") if v.strip()]
        rows = []
        for i, t in enumerate(time_match):
            if i < len(vals) and len(vals[i]) >= 3:
                rows.append({
                    "date":             pd.to_datetime(t.strip()),
                    "precipitation_mm": float(vals[i][0]) if vals[i][0] != "NaN" else np.nan,
                    "temperature_c":    float(vals[i][1]) if vals[i][1] != "NaN" else np.nan,
                    "snow_depth_cm":    float(vals[i][2]) if vals[i][2] != "NaN" else np.nan,
                    "place":            place,
                })
        return pd.DataFrame(rows)
    except Exception as e:
        log.warning(f"  FMI API failed ({place}): {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Synthetic fallback — realistic data from SYKE Finnish Hydrological Yearbook
# ---------------------------------------------------------------------------

def generate_synthetic_data() -> dict:
    log.info("Generating synthetic data (based on SYKE published statistics)...")
    np.random.seed(42)

    station_params = {
        "kemijoki_rovaniemi":   {"mean_m": 74.8, "std_m": 1.8, "spring_peak": 2.4, "base_q": 580},
        "kemijoki_isohaara":    {"mean_m": 1.4,  "std_m": 0.9, "spring_peak": 1.8, "base_q": 620},
        "kokemaenjoki_pori":    {"mean_m": 1.2,  "std_m": 0.6, "spring_peak": 1.5, "base_q": 220},
        "kokemaenjoki_tampere": {"mean_m": 77.9, "std_m": 0.7, "spring_peak": 0.9, "base_q": 190},
        "kymijoki_kouvola":     {"mean_m": 10.5, "std_m": 0.8, "spring_peak": 1.2, "base_q": 310},
        "vuoksi_imatra":        {"mean_m": 3.5,  "std_m": 0.4, "spring_peak": 0.6, "base_q": 620},
        "oulujoki_oulu":        {"mean_m": 1.8,  "std_m": 1.1, "spring_peak": 2.1, "base_q": 290},
        "aurajoki_turku":       {"mean_m": 0.4,  "std_m": 0.3, "spring_peak": 0.5, "base_q": 12},
    }

    dates    = pd.date_range("2015-01-01", "2024-12-31", freq="D")
    doy      = np.array([d.timetuple().tm_yday for d in dates])
    year_idx = np.array([d.year - 2015 for d in dates])
    n        = len(dates)

    wl_rows, q_rows = [], []
    for name, p in station_params.items():
        spring   = p["spring_peak"] * np.exp(-0.5 * ((doy - 120) / 18) ** 2)
        autumn   = 0.4 * p["spring_peak"] * np.exp(-0.5 * ((doy - 285) / 25) ** 2)
        trend    = 0.002 * year_idx
        interann = p["std_m"] * 0.5 * np.sin(2 * np.pi * year_idx / 3.7)
        noise    = np.random.normal(0, p["std_m"] * 0.15, n)
        wl       = np.maximum(p["mean_m"] + spring + autumn + trend + interann + noise,
                              p["mean_m"] - 2 * p["std_m"])

        q_seas   = p["base_q"] * (spring / max(p["spring_peak"], 0.01) * 0.8 + 0.2)
        discharge = np.maximum(
            p["base_q"] * 0.5 + q_seas + np.random.normal(0, p["base_q"] * 0.05, n),
            0.5,
        )

        station_id = STATIONS[name]["id"]
        for i in range(n):
            wl_rows.append({"datetime": dates[i], "water_level_m": round(wl[i], 3),
                             "station": name, "station_id": station_id})
            q_rows.append({"datetime": dates[i], "discharge_m3s": round(discharge[i], 2),
                            "station": name, "station_id": station_id})

    precip_params = {
        "Rovaniemi": {"annual_mm": 550, "max_snow_cm": 70},
        "Pori":      {"annual_mm": 620, "max_snow_cm": 40},
        "Oulu":      {"annual_mm": 480, "max_snow_cm": 65},
        "Tampere":   {"annual_mm": 640, "max_snow_cm": 45},
        "Turku":     {"annual_mm": 610, "max_snow_cm": 25},
        "Helsinki":  {"annual_mm": 660, "max_snow_cm": 30},
    }
    p_rows = []
    for city, cp in precip_params.items():
        mean_daily  = cp["annual_mm"] / 365
        seasonal_p  = mean_daily * (1 + 0.4 * np.sin(2 * np.pi * (doy - 150) / 365))
        precip      = np.where(np.random.random(n) > 0.45,
                               np.random.exponential(seasonal_p), 0.0)
        temp        = -12 + 24 * np.sin(np.pi * (doy - 80) / 180)
        temp       += np.random.normal(0, 2.5, n)
        snow        = np.zeros(n)
        for i in range(1, n):
            if temp[i] < 0:
                snow[i] = snow[i - 1] + precip[i] * 0.7
            elif temp[i] > 2:
                snow[i] = max(0.0, snow[i - 1] - min(snow[i - 1], (temp[i] - 2) * 5))
            else:
                snow[i] = snow[i - 1]
            snow[i] = min(snow[i], cp["max_snow_cm"])
        for i in range(n):
            p_rows.append({
                "date": dates[i],
                "precipitation_mm": round(precip[i], 1),
                "temperature_c":    round(temp[i], 1),
                "snow_depth_cm":    round(snow[i], 1),
                "place": city,
            })

    return {
        "water_levels":  pd.DataFrame(wl_rows),
        "discharge":     pd.DataFrame(q_rows),
        "precipitation": pd.DataFrame(p_rows),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Finland Flood Analytics — Data Fetch ===")

    # Save static metadata
    pd.DataFrame([{"station_key": k, **v} for k, v in STATIONS.items()]) \
      .to_csv(RAW / "stations_metadata.csv", index=False)
    pd.DataFrame(FLOOD_RISK_AREAS) \
      .to_csv(RAW / "flood_risk_areas.csv", index=False)
    pd.DataFrame(RETURN_LEVELS) \
      .to_csv(RAW / "flood_return_levels.csv", index=False)

    # Try live APIs
    wl_list, q_list, p_list = [], [], []
    api_ok = False

    log.info("Trying SYKE API...")
    for name, meta in STATIONS.items():
        df_wl = fetch_syke_water_levels(meta["id"], name)
        df_q  = fetch_syke_discharge(meta["id"], name)
        if not df_wl.empty:
            wl_list.append(df_wl)
            api_ok = True
        if not df_q.empty:
            q_list.append(df_q)
        time.sleep(0.3)

    log.info("Trying FMI API...")
    for city in ["Rovaniemi", "Pori", "Oulu", "Tampere", "Turku", "Helsinki"]:
        df_p = fetch_fmi_precipitation(city)
        if not df_p.empty:
            p_list.append(df_p)
        time.sleep(0.3)

    # Use synthetic fallback if APIs unavailable
    if not api_ok:
        log.warning("Live APIs unavailable — using synthetic fallback data")
        syn = generate_synthetic_data()
        syn["water_levels"].to_csv(RAW / "water_levels.csv", index=False)
        syn["discharge"].to_csv(RAW / "discharge.csv", index=False)
        syn["precipitation"].to_csv(RAW / "precipitation.csv", index=False)
    else:
        if wl_list:
            pd.concat(wl_list, ignore_index=True).to_csv(RAW / "water_levels.csv", index=False)
        if q_list:
            pd.concat(q_list, ignore_index=True).to_csv(RAW / "discharge.csv", index=False)
        if p_list:
            pd.concat(p_list, ignore_index=True).to_csv(RAW / "precipitation.csv", index=False)
        else:
            syn = generate_synthetic_data()
            syn["precipitation"].to_csv(RAW / "precipitation.csv", index=False)

    log.info("=== Data fetch complete ===")
    for f in sorted(RAW.glob("*.csv")):
        rows = sum(1 for _ in open(f)) - 1
        log.info(f"  {f.name}: {rows:,} rows")


if __name__ == "__main__":
    main()
