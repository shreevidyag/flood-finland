"""
analysis.py
===========
Complete statistical analysis pipeline for Finnish flood data.

Steps:
  1  Flood event detection       (P95 threshold, min 2 days)
  2  Spring peak extraction      (kevättulva — March to June)
  3  Gumbel return period model  (EV-I distribution)
  4  Mann-Kendall trend test     (+ Sen's slope)
  5  Seasonal statistics         (monthly mean, std, percentiles)
  6  Precipitation-discharge     (Pearson correlation)

Run:
    python src/analysis.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_water_levels() -> pd.DataFrame:
    p = RAW / "water_levels.csv"
    if not p.exists():
        raise FileNotFoundError(f"Run fetch_data.py first — missing {p}")
    return pd.read_csv(p, parse_dates=["datetime"]).dropna(subset=["water_level_m"])


def load_discharge() -> pd.DataFrame:
    p = RAW / "discharge.csv"
    if not p.exists():
        raise FileNotFoundError(f"Run fetch_data.py first — missing {p}")
    return pd.read_csv(p, parse_dates=["datetime"]).dropna(subset=["discharge_m3s"])


def load_precipitation() -> pd.DataFrame:
    p = RAW / "precipitation.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, parse_dates=["date"]).dropna(subset=["precipitation_mm"])


def load_risk_areas()    -> pd.DataFrame:
    return pd.read_csv(RAW / "flood_risk_areas.csv")

def load_return_levels() -> pd.DataFrame:
    return pd.read_csv(RAW / "flood_return_levels.csv")

def load_stations()      -> pd.DataFrame:
    return pd.read_csv(RAW / "stations_metadata.csv")


# ---------------------------------------------------------------------------
# Step 1 — Flood event detection
# ---------------------------------------------------------------------------

def detect_flood_events(df: pd.DataFrame,
                         percentile: float = 95.0,
                         min_days: int = 2) -> pd.DataFrame:
    """
    Find periods where water level exceeds the station P95 threshold
    for at least min_days consecutive days.
    """
    events = []
    for station, grp in df.groupby("station"):
        grp       = grp.sort_values("datetime").reset_index(drop=True)
        threshold = float(np.percentile(grp["water_level_m"].dropna(), percentile))
        grp["above"] = grp["water_level_m"] >= threshold
        grp["grp"]   = (grp["above"] != grp["above"].shift()).cumsum()

        for _, seg in grp[grp["above"]].groupby("grp"):
            if len(seg) >= min_days:
                events.append({
                    "station":       station,
                    "start_date":    seg["datetime"].min(),
                    "end_date":      seg["datetime"].max(),
                    "duration_days": len(seg),
                    "peak_level_m":  round(float(seg["water_level_m"].max()), 3),
                    "mean_level_m":  round(float(seg["water_level_m"].mean()), 3),
                    "threshold_m":   round(threshold, 3),
                    "excess_m":      round(float(seg["water_level_m"].max()) - threshold, 3),
                })

    return (pd.DataFrame(events).sort_values("start_date").reset_index(drop=True)
            if events else pd.DataFrame())


# ---------------------------------------------------------------------------
# Step 2 — Spring flood peaks (kevättulva)
# ---------------------------------------------------------------------------

def extract_spring_peaks(df: pd.DataFrame) -> pd.DataFrame:
    """Annual spring flood peak per station, March–June window."""
    df         = df.copy()
    df["year"] = df["datetime"].dt.year
    df["month"]= df["datetime"].dt.month
    spring     = df[df["month"].between(3, 6)]
    rows       = []
    for (station, year), grp in spring.groupby(["station", "year"]):
        if grp.empty:
            continue
        idx = grp["water_level_m"].idxmax()
        rows.append({
            "station":      station,
            "year":         year,
            "peak_date":    grp.loc[idx, "datetime"],
            "peak_level_m": round(float(grp.loc[idx, "water_level_m"]), 3),
            "peak_doy":     int(grp.loc[idx, "datetime"].timetuple().tm_yday),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3 — Gumbel EV-I return periods
# ---------------------------------------------------------------------------

def gumbel_return_period(annual_maxima: np.ndarray) -> pd.DataFrame:
    """
    Fit Gumbel (Extreme Value Type I) to annual maxima.
    Returns estimated levels for 2 … 1000 year return periods.
    """
    loc, scale = stats.gumbel_r.fit(annual_maxima)
    rows = []
    for rp in [2, 5, 10, 20, 50, 100, 250, 500, 1000]:
        level = stats.gumbel_r.ppf(1 - 1.0 / rp, loc=loc, scale=scale)
        rows.append({
            "return_period_years": rp,
            "exceedance_prob":     round(1.0 / rp, 6),
            "estimated_level_m":  round(float(level), 3),
            "gumbel_loc":         round(float(loc), 3),
            "gumbel_scale":       round(float(scale), 3),
        })
    return pd.DataFrame(rows)


def compute_return_periods(df: pd.DataFrame) -> pd.DataFrame:
    peaks, results = extract_spring_peaks(df), []
    for station, grp in peaks.groupby("station"):
        if len(grp) < 5:
            continue
        rp = gumbel_return_period(grp["peak_level_m"].values)
        rp["station"] = station
        results.append(rp)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ---------------------------------------------------------------------------
# Step 4 — Mann-Kendall trend + Sen's slope
# ---------------------------------------------------------------------------

def mann_kendall_trend(series: np.ndarray) -> dict:
    """Non-parametric trend test. Returns MK statistic, p-value, Sen's slope."""
    n = len(series)
    s = sum(
        (1 if series[j] > series[i] else -1)
        for i in range(n - 1)
        for j in range(i + 1, n)
        if series[j] != series[i]
    )
    var_s = n * (n - 1) * (2 * n + 5) / 18
    z     = ((s - 1) / var_s ** 0.5 if s > 0 else
             (s + 1) / var_s ** 0.5 if s < 0 else 0.0)
    p     = float(2 * (1 - stats.norm.cdf(abs(z))))
    slope = float(np.median([
        (series[j] - series[i]) / (j - i)
        for i in range(n - 1)
        for j in range(i + 1, n)
    ]))
    return {
        "mk_statistic":       round(z, 4),
        "p_value":            round(p, 4),
        "sen_slope_per_year": round(slope, 6),
        "trend": ("increasing"           if z > 0 and p < 0.05 else
                  "decreasing"           if z < 0 and p < 0.05 else
                  "no_significant_trend"),
    }


def analyze_trends(df: pd.DataFrame) -> pd.DataFrame:
    df         = df.copy()
    df["year"] = df["datetime"].dt.year
    annual     = df.groupby(["station", "year"])["water_level_m"].mean().reset_index()
    rows = []
    for station, grp in annual.groupby("station"):
        grp = grp.sort_values("year")
        if len(grp) < 6:
            continue
        mk = mann_kendall_trend(grp["water_level_m"].values)
        rows.append({"station": station, "n_years": len(grp),
                     "mean_level_m": round(grp["water_level_m"].mean(), 3),
                     "std_level_m":  round(grp["water_level_m"].std(), 3), **mk})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 5 — Seasonal statistics
# ---------------------------------------------------------------------------

def compute_seasonal_stats(df: pd.DataFrame) -> pd.DataFrame:
    df          = df.copy()
    df["month"] = df["datetime"].dt.month
    names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    rows = []
    for (station, month), grp in df.groupby(["station", "month"]):
        rows.append({
            "station":      station,
            "month":        month,
            "month_name":   names[month],
            "mean_level_m": round(grp["water_level_m"].mean(), 3),
            "std_level_m":  round(grp["water_level_m"].std(), 3),
            "p95_level_m":  round(grp["water_level_m"].quantile(0.95), 3),
            "p05_level_m":  round(grp["water_level_m"].quantile(0.05), 3),
            "n_obs":        len(grp),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 6 — Precipitation–discharge correlation
# ---------------------------------------------------------------------------

STATION_CITY = {
    "kemijoki_rovaniemi":   "Rovaniemi",
    "kemijoki_isohaara":    "Rovaniemi",
    "kokemaenjoki_pori":    "Pori",
    "kokemaenjoki_tampere": "Tampere",
    "kymijoki_kouvola":     None,
    "vuoksi_imatra":        None,
    "oulujoki_oulu":        "Oulu",
    "aurajoki_turku":       "Turku",
}


def compute_precip_discharge_corr(q_df: pd.DataFrame, p_df: pd.DataFrame) -> pd.DataFrame:
    if p_df.empty:
        return pd.DataFrame()
    q_df, p_df = q_df.copy(), p_df.copy()
    q_df["ym"] = q_df["datetime"].dt.to_period("M")
    p_df["ym"] = p_df["date"].dt.to_period("M")
    mq = q_df.groupby(["station", "ym"])["discharge_m3s"].mean()
    mp = p_df.groupby(["place",   "ym"])["precipitation_mm"].sum()
    rows = []
    for station, city in STATION_CITY.items():
        if not city:
            continue
        if (station not in mq.index.get_level_values(0) or
                city not in mp.index.get_level_values(0)):
            continue
        common = mq[station].index.intersection(mp[city].index)
        if len(common) < 12:
            continue
        r, pval = stats.pearsonr(mp[city][common].values, mq[station][common].values)
        rows.append({"station": station, "city": city, "n_months": len(common),
                     "pearson_r": round(r, 3), "p_value": round(pval, 4),
                     "significant": pval < 0.05})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Risk summary
# ---------------------------------------------------------------------------

STATION_AREA = {
    "kemijoki_rovaniemi":   "Rovaniemi",
    "kemijoki_isohaara":    "Kemi-Tornio",
    "kokemaenjoki_pori":    "Pori",
    "kokemaenjoki_tampere": "Tampere",
    "kymijoki_kouvola":     "Kouvola",
    "vuoksi_imatra":        "Imatra-Joutseno",
    "oulujoki_oulu":        "Oulu",
    "aurajoki_turku":       "Turku",
}


def compute_risk_summary(events: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    s = events.groupby("station").agg(
        n_events=("start_date", "count"),
        avg_duration=("duration_days", "mean"),
        max_peak=("peak_level_m", "max"),
        total_days=("duration_days", "sum"),
    ).reset_index()
    s["area_name"] = s["station"].map(STATION_AREA)
    return s.merge(risk, left_on="area_name", right_on="name", how="left")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_analysis() -> dict:
    log.info("=== Finland Flood Analytics — Analysis Pipeline ===")

    wl     = load_water_levels()
    q      = load_discharge()
    precip = load_precipitation()
    risk   = load_risk_areas()

    log.info(f"Loaded: {len(wl):,} water level rows, {wl['station'].nunique()} stations")

    log.info("[1/6] Detecting flood events...")
    events = detect_flood_events(wl)
    events.to_csv(PROC / "flood_events.csv", index=False)
    log.info(f"  → {len(events)} events")

    log.info("[2/6] Extracting spring peaks...")
    peaks = extract_spring_peaks(wl)
    peaks.to_csv(PROC / "spring_peaks.csv", index=False)
    log.info(f"  → {len(peaks)} annual peaks")

    log.info("[3/6] Computing Gumbel return periods...")
    rp = compute_return_periods(wl)
    rp.to_csv(PROC / "return_periods.csv", index=False)
    log.info(f"  → {len(rp)} rows for {rp['station'].nunique()} stations")

    log.info("[4/6] Mann-Kendall trend analysis...")
    trends = analyze_trends(wl)
    trends.to_csv(PROC / "trends.csv", index=False)
    for _, row in trends.iterrows():
        log.info(f"  {row['station']:30s}  {row['trend']}  (p={row['p_value']:.3f})")

    log.info("[5/6] Seasonal statistics...")
    seasonal = compute_seasonal_stats(wl)
    seasonal.to_csv(PROC / "seasonal_stats.csv", index=False)
    log.info(f"  → {len(seasonal)} station-month records")

    log.info("[6/6] Precipitation-discharge correlation...")
    corr = compute_precip_discharge_corr(q, precip)
    corr.to_csv(PROC / "precip_discharge_corr.csv", index=False)
    if not corr.empty:
        for _, row in corr.iterrows():
            log.info(f"  {row['station']:30s} r={row['pearson_r']:.2f} p={row['p_value']:.3f}")

    rs = compute_risk_summary(events, risk)
    rs.to_csv(PROC / "risk_summary.csv", index=False)

    log.info("=== Analysis complete ===")
    for f in sorted(PROC.glob("*.csv")):
        rows = sum(1 for _ in open(f)) - 1
        log.info(f"  {f.name}: {rows:,} rows")

    return {"flood_events": events, "spring_peaks": peaks, "return_periods": rp,
            "trends": trends, "seasonal_stats": seasonal, "corr": corr}


if __name__ == "__main__":
    run_analysis()
