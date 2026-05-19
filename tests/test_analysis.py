"""
test_analysis.py
================
Unit tests for the Finland Flood Analytics pipeline.
Run:  pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis import (
    detect_flood_events,
    extract_spring_peaks,
    gumbel_return_period,
    mann_kendall_trend,
    compute_seasonal_stats,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def wl_data():
    """10 years of synthetic daily water levels with realistic spring peaks."""
    np.random.seed(0)
    dates  = pd.date_range("2015-01-01", "2024-12-31", freq="D")
    doy    = np.array([d.timetuple().tm_yday for d in dates])
    spring = 2.0 * np.exp(-0.5 * ((doy - 120) / 18) ** 2)
    wl     = 74.5 + spring + np.random.normal(0, 0.2, len(dates))
    return pd.DataFrame({
        "datetime":      dates,
        "water_level_m": wl,
        "station":       "test_station",
        "station_id":    "0000",
    })


# ---------------------------------------------------------------------------
# Flood detection
# ---------------------------------------------------------------------------

class TestFloodDetection:

    def test_returns_dataframe(self, wl_data):
        assert isinstance(detect_flood_events(wl_data), pd.DataFrame)

    def test_required_columns(self, wl_data):
        cols = detect_flood_events(wl_data).columns
        for c in ("station","start_date","end_date","duration_days",
                  "peak_level_m","threshold_m","excess_m"):
            assert c in cols

    def test_peaks_above_threshold(self, wl_data):
        ev = detect_flood_events(wl_data, percentile=95)
        assert (ev["peak_level_m"] >= ev["threshold_m"]).all()

    def test_min_duration_respected(self, wl_data):
        ev = detect_flood_events(wl_data, min_days=3)
        assert (ev["duration_days"] >= 3).all()

    def test_excess_non_negative(self, wl_data):
        ev = detect_flood_events(wl_data)
        assert (ev["excess_m"] >= 0).all()


# ---------------------------------------------------------------------------
# Spring peaks
# ---------------------------------------------------------------------------

class TestSpringPeaks:

    def test_returns_dataframe(self, wl_data):
        assert isinstance(extract_spring_peaks(wl_data), pd.DataFrame)

    def test_required_columns(self, wl_data):
        cols = extract_spring_peaks(wl_data).columns
        for c in ("station","year","peak_date","peak_level_m","peak_doy"):
            assert c in cols

    def test_one_peak_per_station_year(self, wl_data):
        peaks  = extract_spring_peaks(wl_data)
        counts = peaks.groupby(["station","year"]).size()
        assert (counts == 1).all()

    def test_peaks_in_spring_window(self, wl_data):
        peaks = extract_spring_peaks(wl_data)
        assert peaks["peak_doy"].between(60, 181).all()


# ---------------------------------------------------------------------------
# Gumbel return periods
# ---------------------------------------------------------------------------

class TestGumbel:

    def test_nine_return_periods(self):
        data = np.array([74.2,75.1,76.4,74.8,77.2,75.5,76.8,74.5,76.1,75.9])
        assert len(gumbel_return_period(data)) == 9

    def test_levels_monotone_increasing(self):
        data   = np.array([74.2,75.1,76.4,74.8,77.2,75.5,76.8,74.5,76.1,75.9])
        levels = gumbel_return_period(data)["estimated_level_m"].values
        assert all(levels[i] <= levels[i+1] for i in range(len(levels)-1))

    def test_correct_exceedance_prob(self):
        data  = np.random.RandomState(1).normal(75, 1, 50)
        df    = gumbel_return_period(data)
        ep100 = df[df["return_period_years"]==100]["exceedance_prob"].values[0]
        assert abs(ep100 - 0.01) < 1e-9


# ---------------------------------------------------------------------------
# Mann-Kendall
# ---------------------------------------------------------------------------

class TestMannKendall:

    def test_detects_increasing_trend(self):
        result = mann_kendall_trend(np.arange(1.0, 11.0))
        assert result["trend"] == "increasing"
        assert result["p_value"] < 0.05
        assert result["sen_slope_per_year"] > 0

    def test_all_keys_present(self):
        result = mann_kendall_trend(np.array([1.0,2.0,1.5,3.0]))
        assert set(result) == {"mk_statistic","p_value","sen_slope_per_year","trend"}

    def test_trend_value_is_valid_string(self):
        result = mann_kendall_trend(np.random.RandomState(7).normal(5,1,30))
        assert result["trend"] in {"increasing","decreasing","no_significant_trend"}


# ---------------------------------------------------------------------------
# Seasonal stats
# ---------------------------------------------------------------------------

class TestSeasonalStats:

    def test_twelve_months_per_station(self, wl_data):
        result = compute_seasonal_stats(wl_data)
        assert (result.groupby("station")["month"].count() == 12).all()

    def test_required_columns(self, wl_data):
        result = compute_seasonal_stats(wl_data)
        for c in ("mean_level_m","std_level_m","p95_level_m","p05_level_m"):
            assert c in result.columns

    def test_p95_above_mean(self, wl_data):
        result = compute_seasonal_stats(wl_data)
        assert (result["p95_level_m"] >= result["mean_level_m"]).all()

    def test_p05_below_mean(self, wl_data):
        result = compute_seasonal_stats(wl_data)
        assert (result["p05_level_m"] <= result["mean_level_m"]).all()

    def test_spring_higher_than_winter(self, wl_data):
        by_month = compute_seasonal_stats(wl_data).groupby("month")["mean_level_m"].mean()
        assert by_month[4] > by_month[1]   # April > January
        assert by_month[5] > by_month[1]   # May   > January
