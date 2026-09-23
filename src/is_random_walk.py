import numpy as np
import pandas as pd
from scipy import stats

from src.analyse import load_data, prepare_data

# =========================================================================
# Constants: DTD banding and sampling-regime lookup
# =========================================================================

DEFAULT_DTD_BAND_BINS = [
    0, 7, 14, 30, 60, 90, 180, 270, 365
]
DEFAULT_DTD_BAND_LABELS = [
    "1-7", "8-14", "15-30", "31-60", "61-90", "91-180", "181-270", "271-365",
]

DEFAULT_DTD_REGIME_BINS = [0, 14, 60, 365]
DEFAULT_DTD_REGIME_LABELS = ["hourly", "six_hourly", "daily"]

# Which sampling regime the bands are in
BAND_TO_REGIME = {
    "1-7": "hourly",
    "8-14": "hourly",
    "15-30": "six_hourly",
    "31-60": "six_hourly",
    "61-90": "daily",
    "91-180": "daily",
    "181-270": "daily",
    "271-365": "daily",
}

REGIME_OBS_PER_DAY = {"hourly": 24, "six_hourly": 4, "daily": 1}

NATIVE_VR_QS = (2, 4, 8)
NATIVE_LB_M = 5
CALENDAR_DAY_LAGS = (1,)

def get_regime_for_dtd_band(band_label,band_col):
    """Get the sampling regime for a given DTD band label."""

    if band_col == "dtd_regime":
        return band_label
    return BAND_TO_REGIME[band_label]
    
def calendar_days_to_lags(days,obs_per_day,min_lag=1):
    """Convert calendar days to lags in the sampling regime."""

    raw = [int(round(days * obs_per_day))]
    return max(min_lag, raw)

def flight_switch_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every (route, departure_date) series, count how many unique
    (departure_at, arrival_at) pairs ever appear, and how many times the
    identity changes row-to-row. Checking if switching is rare.
    """
    working = df.sort_values(["route", "departure_date", "observed_at"]).copy()
    working["_flight_key"] = list(zip(working["departure_at"], working["arrival_at"]))

    summary = (
        working.groupby(["route", "departure_date"])
        .agg(n_obs=("price", "size"), n_distinct_flights=("_flight_key", "nunique"))
        .reset_index()
    )

    def _count_switches(s: pd.Series) -> int:

        return int((s != s.shift(1)).sum()) - 1

    switches = (
        working.groupby(["route", "departure_date"])["_flight_key"]
        .apply(_count_switches)
        .reset_index(name="n_switches")
    )
    summary = summary.merge(switches, on=["route", "departure_date"])
    summary["ever_switched"] = summary["n_distinct_flights"] > 1
    return summary


def print_switch_diagnostics_report(summary: pd.DataFrame) -> None:
    n_series = len(summary)
    n_switched = int(summary["ever_switched"].sum())
    pct = 100 * n_switched / n_series if n_series else 0.0

    print(f"(route, departure_date) series total : {n_series}")
    print(f"Series with >1 distinct flight       : {n_switched} ({pct:.1f}%)")
    print()
    print("Per-route switch rate:")
    route_stats = summary.groupby("route").agg(
        n_series=("ever_switched", "size"),
        pct_ever_switched=("ever_switched", lambda s: 100 * s.mean()),
        median_switches=("n_switches", "median"),
        max_switches=("n_switches", "max"),
    )
    print(route_stats.to_string())


# =========================================================================
# DTD banding
# =========================================================================

def add_dtd_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Divide the booking horizon into 8 interpretable, regime-safe bands."""
    df = df.copy()
    df["dtd_band"] = pd.cut(
        df["days_to_departure"],
        bins=DEFAULT_DTD_BAND_BINS,
        labels=DEFAULT_DTD_BAND_LABELS,
    )
    return df


def add_dtd_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the booking horizon into the 3 actual sampling regimes."""
    df = df.copy()
    df["dtd_regime"] = pd.cut(
        df["days_to_departure"],
        bins=DEFAULT_DTD_REGIME_BINS,
        labels=DEFAULT_DTD_REGIME_LABELS,
    )
    return df
