import numpy as np
import pandas as pd
from scipy import stats

from src.analyse import load_data, prepare_data

ALPHA = 0.05
NOISE_FLOOR_PCT = ALPHA * 100

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
NATIVE_LB_LAGS = 5
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

# =========================================================================
# GET SERIES
# =========================================================================

def get_completed_series(df, today=None):
    """Retrieve routes where departure date has already passed"""

    if today is None:
        today = pd.Timestamp.now().normalize()
    else:
        today = pd.Timestamp(today).normalize()
    
    working = df.copy()
    working["departure_date"] = pd.to_datetime(working["departure_date"])
    return working[working["departure_date"] < today].copy()

def build_daily_panel(df):
    """Table considering only observations taken at 18:00 UTC - so daily collection regime """

    working = df.copy()
    working['observed_at'] = pd.to_datetime(working['observed_at'], utc=True)
    working['observation_hour'] =  working['observed_at'].dt.hour

    daily = working[working['observation_hour'] == 18]
    return daily.sort_values(['route','departure_date','observed_at'])

# =========================================================================
# CHECK FOR FLIGHT SWITCHING
# =========================================================================

def flight_switch_diagnostics(daily_df):
    """
    For every (route, departure_date) series, count how many unique
    (departure_at, arrival_at) pairs ever appear, checking if switching is rare.
    """
    key = ['route','departure_date']
    working = daily_df.sort_values(key + ['observed_at']).copy()
    working["_flight_key"] = list(zip(working["departure_at"], working["arrival_at"]))

    per_series = (
        working.groupby(key)
        .agg(n_obs=("price", "size"), n_distinct_flights=("_flight_key", "nunique"))
        .reset_index()
    )
    per_series['ever_switched'] = per_series['n_distinct_flights'] > 1

    per_route = per_series.groupby('route').agg(
        n_series = ('ever_switched','size'),
        n_ever_switched = ('ever_switched','sum'),
    )
    per_route['pct_ever_switched'] = 100 * per_route['n_ever_switched'] / per_route['n_series']

    return per_series, per_route

# =========================================================================
# BUILD RETURN SERIES
# =========================================================================

def build_return_series(daily_df, min_return_obs=5, exclude_switched=False):
    """Build a return series from the daily DataFrame so log of price(n)/price(n-1)"""

    key = ['route', 'departure_date']
    working = daily_df.sort_values(key + ['observed_at']).copy()
    g = working.groupby(key)

    working['previous_price'] = g['price'].shift(1)
    working['previous_departure_at'] = g['departure_at'].shift(1)
    working['previous_arrival_at'] = g['arrival_at'].shift(1)

    has_previous = working['previous_price'].notna()
    working['log_return'] = np.where(
        has_previous,
        np.log(working['price'] / working['previous_price']),
        np.nan,
    )

    working['ever_switched'] = has_previous & (
        working['departure_at'] != working['previous_departure_at']
    ) | (
        working['arrival_at'] != working['previous_arrival_at']
    )

    if exclude_switched:
        working.loc[working["ever_switched"], "log_return"] = np.nan

    working = working.drop(columns=['previous_price', 'previous_departure_at', 'previous_arrival_at'])

    series_dict = {}
    for key_tuple, group in working.groupby(key):
        n_valid = group['log_return'].notna().sum()
        if n_valid >= min_return_obs:
            series_dict[key_tuple] = group.reset_index(drop=True)

    return series_dict

# =========================================================================
# STATISTICAL TESTS
# =========================================================================

def ols_fit_and_t_test(x, y):
    """"Fit an OLS regression and return the y-intercept (beta0), slope (beta1), beta2(see if squared term is significant),
      p-values, se of betas, t-values, r-squared
      y = beta0 + beta1 * x + beta2 * x^2 + error
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # keep only observations where both x and y are valid
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    n = len(x)

    # Need more observations than coefficients
    # beta0, beta1, beta2
    if n < 4:
        return None

    # Design matrix:
    # column 1 = intercept
    # column 2 = x
    # column 3 = x^2
    X = np.column_stack([
        np.ones(n),
        x,
        x**2,
    ])

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None

    # OLS coefficient estimates
    beta = XtX_inv @ X.T @ y

    # Fitted values and residuals
    y_pred = X @ beta
    residuals = y - y_pred

    # Residual degrees of freedom
    dof = n - X.shape[1]

    if dof <= 0:
        return None

    # Estimated error variance
    sigma2 = np.sum(residuals**2) / dof

    # Standard errors of beta0, beta1, beta2
    se_beta = np.sqrt(
        sigma2 * np.diag(XtX_inv)
    )

    # t-statistics
    t_stats = beta / se_beta

    # Two-sided p-values
    p_values = 2 * (
        1 - stats.t.cdf(
            np.abs(t_stats),
            df=dof,
        )
    )

    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    r_squared = (
        1 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    return {
        "beta0": beta[0],
        "beta1": beta[1],
        "beta2": beta[2],

        "se_beta0": se_beta[0],
        "se_beta1": se_beta[1],
        "se_beta2": se_beta[2],

        "t_beta0": t_stats[0],
        "t_beta1": t_stats[1],
        "t_beta2": t_stats[2],

        "p_beta0": p_values[0],
        "p_beta1": p_values[1],
        "p_beta2": p_values[2],

        "n": n,
        "dof": dof,
        "r_squared": r_squared,
    }
    
def variance_ratio_test(returns, q):
    """Lo-Mackinlay variance ratio test for random walk hypothesis
    (for one flight) Returns VR and test statistic for VR(q) 
    and p-value. carries out homoskedastic test and 
    heteroskedastic test.
    """

    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 2 * q:
        return None

    mu = np.mean(r)

    # make returns series of q-period returns
    r_q = np.array([np.sum(r[t - q:t]) for t in range(q, n + 1)])

    # normalising factor for variance of q-period returns
    m = q * (n - q + 1) * (1 - q / n)

    var_q = np.sum((r_q - q * mu) ** 2) / m
    var_1 = np.sum((r - mu) ** 2) / (n - 1)
    vr = var_q / var_1

    # homoskedastic 
    phi = (2 * (2 * q - 1) * (q - 1)) / (3 * q * n)
    z_hom = (vr - 1) / np.sqrt(phi) if phi > 0 else np.nan
    p_hom = 2 * (1 - stats.norm.cdf(np.abs(z_hom))) if not np.isnan(z_hom) else np.nan

    # heteroskedasticity-robust
    denom = np.sum((r - mu) ** 2) ** 2
    delta = np.zeros(q)
    for j in range(1, q):
        num = n * np.sum(((r[j:] - mu) ** 2) * ((r[:-j] - mu) ** 2))
        delta[j] = num / denom if denom > 0 else np.nan

    theta = np.sum([((2 * (q - j) / q) ** 2) * delta[j] for j in range(1, q)])
    z_rob = (vr - 1) / np.sqrt(theta) if theta > 0 else np.nan
    p_rob = 2 * (1 - stats.norm.cdf(np.abs(z_rob))) if not np.isnan(z_rob) else np.nan

    return {
        "q": q, "n": n, "vr": vr,
        "z_hom": z_hom, "p_hom": p_hom,
        "z_rob": z_rob, "p_rob": p_rob,
    }

def ljung_box_test(returns, lags=5):
    """Ljung-Box test for autocorrelation in returns series.
    Returns test statistic and p-value for given number of lags.
    Run on return and return^2 to test for autocorrelation and
    possible volatility clustering.
    """

    # Remove NaN values from returns series
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]

    n = len(r)

    # Check if there are enough observations for the specified number of lags
    if n < lags + 2:
        return None

    mean_r = np.mean(r)

    # Ensure that the variance is not zero 
    c0 = np.sum((r - mean_r) ** 2)
    if c0 == 0:
        return None

    # Calculate autocorrelations for the specified number of lags
    rho = np.array([
        np.sum(
            (r[k:] - mean_r) *
            (r[:-k] - mean_r)
        ) / c0
        for k in range(1, lags + 1)
    ])

    # Calculate the Ljung-Box Q statistic
    q_stat = (
        n * (n + 2)
        * np.sum([
            rho[k - 1] ** 2 / (n - k)
            for k in range(1, lags + 1)
        ])
    )

    # Calculate the p-value using the chi-squared distribution
    p_value = stats.chi2.sf(
        q_stat,
        df=lags
    )

    return {
        "Q": q_stat,
        "df": lags,
        "p": p_value,
        "n": n,
    }

# =========================================================================
# APPLY HYPOTHESIS TESTS TO COMPLETE FLIGHT SERIES
# =========================================================================


def run_tier1_tests_per_flight(series_dict, vr_qs=(2, 4, 8), lb_lags=5):
    """
    Applies OLS, the VR test (at each q in vr_qs), and Ljung-Box (returns
    and squared returns) to each completed flight individually. Returns
    one row per (route, departure_date)
    """
    rows = []

    for (route, departure_date), group in series_dict.items():
        valid = group.dropna(subset=["log_return"])
        r = valid["log_return"].values
        dtd = valid["days_to_departure"].values

        row = {
            "route": route,
            "departure_date": departure_date,
            "n_obs": len(valid),
        }

        ols_result = ols_fit_and_t_test(dtd, r)

        if ols_result is not None:
            row.update({
            "ols_beta0": ols_result["beta0"],
            "ols_beta1": ols_result["beta1"],
            "ols_beta2": ols_result["beta2"],

            "ols_se_beta0": ols_result["se_beta0"],
            "ols_se_beta1": ols_result["se_beta1"],
            "ols_se_beta2": ols_result["se_beta2"],

            "ols_t_beta0": ols_result["t_beta0"],
            "ols_t_beta1": ols_result["t_beta1"],
            "ols_t_beta2": ols_result["t_beta2"],

            "ols_p_beta0": ols_result["p_beta0"],
            "ols_p_beta1": ols_result["p_beta1"],
            "ols_p_beta2": ols_result["p_beta2"],

            "ols_r_squared": ols_result["r_squared"],
            "ols_n": ols_result["n"],
            })

        for q in vr_qs:
            vr_result = variance_ratio_test(r, q)
            if vr_result is not None:
                row.update({
                    f"vr_{q}_value": vr_result["vr"],
                    f"vr_{q}_z_hom": vr_result["z_hom"],
                    f"vr_{q}_p_hom": vr_result["p_hom"],
                    f"vr_{q}_z_rob": vr_result["z_rob"],
                    f"vr_{q}_p_rob": vr_result["p_rob"],
                    f"vr_{q}_n": vr_result["n"],
                })

        lb_returns = ljung_box_test(r, lags=lb_lags)
        if lb_returns is not None:
            row.update({
                "lb_returns_Q": lb_returns["Q"],
                "lb_returns_df": lb_returns["df"],
                "lb_returns_p": lb_returns["p"],
            })

        lb_sq = ljung_box_test(r ** 2, lags=lb_lags)
        if lb_sq is not None:
            row.update({
                "lb_sq_Q": lb_sq["Q"],
                "lb_sq_df": lb_sq["df"],
                "lb_sq_p": lb_sq["p"],
            })

        rows.append(row)

    return pd.DataFrame(rows)

# =========================================================================
# ROUTRE-LEVEL SUMMARY (PERCENT + RAW COUNTS + NOISE FLOOR)
# =========================================================================

def route_summary(per_flight_df, vr_qs=(2, 4, 8)):
    """
    Merge per-flight table so one row per route: % (and raw count)
    of that route's completed departures that were statistically
    significant at alpha=0.05, split by direction for two-tailed
    tests (OLS slope sign, VR above/below 1), and an overall
    significant-either-direction figure. Every percentage should be read
    next to the noise floor (~5%) printed alongside it in main().
    """
    summaries = []

    for route, g in per_flight_df.groupby("route"):
        n_total = len(g)
        row = {"route": route, "n_completed_departures": n_total}

        # beta1
        if "ols_p_beta1" in g.columns and "ols_p_beta2" in g.columns:

            valid_beta1 = g["ols_p_beta1"].notna()
            n_valid_beta1 = int(valid_beta1.sum())

            sig_beta1 = valid_beta1 & (g["ols_p_beta1"] < ALPHA)

            neg_beta1 = sig_beta1 & (g["ols_beta1"] < 0)
            pos_beta1 = sig_beta1 & (g["ols_beta1"] > 0)

            row["ols_n_valid_beta1"] = n_valid_beta1
            row["ols_pct_sig_beta1"] = (
                100 * sig_beta1.sum() / n_valid_beta1
                if n_valid_beta1 else np.nan
            )
            row["ols_n_sig_beta1"] = int(sig_beta1.sum())

            row["ols_pct_sig_negative_beta1"] = (
                100 * neg_beta1.sum() / n_valid_beta1
                if n_valid_beta1 else np.nan
            )
            row["ols_n_sig_negative_beta1"] = int(neg_beta1.sum())

            row["ols_pct_sig_positive_beta1"] = (
                100 * pos_beta1.sum() / n_valid_beta1
                if n_valid_beta1 else np.nan
            )
            row["ols_n_sig_positive_beta1"] = int(pos_beta1.sum())

            # beta2
            valid_beta2 = g["ols_p_beta2"].notna()
            n_valid_beta2 = int(valid_beta2.sum())

            sig_beta2 = valid_beta2 & (g["ols_p_beta2"] < ALPHA)

            neg_beta2 = sig_beta2 & (g["ols_beta2"] < 0)
            pos_beta2 = sig_beta2 & (g["ols_beta2"] > 0)

            row["ols_n_valid_beta2"] = n_valid_beta2
            row["ols_pct_sig_beta2"] = (
                100 * sig_beta2.sum() / n_valid_beta2
                if n_valid_beta2 else np.nan
            )
            row["ols_n_sig_beta2"] = int(sig_beta2.sum())

            row["ols_pct_sig_negative_beta2"] = (
                100 * neg_beta2.sum() / n_valid_beta2
                if n_valid_beta2 else np.nan
            )
            row["ols_n_sig_negative_beta2"] = int(neg_beta2.sum())

            row["ols_pct_sig_positive_beta2"] = (
                100 * pos_beta2.sum() / n_valid_beta2
                if n_valid_beta2 else np.nan
            )
            row["ols_n_sig_positive_beta2"] = int(pos_beta2.sum())

        for q in vr_qs:
            pcol, vcol = f"vr_{q}_p_rob", f"vr_{q}_value"
            if pcol in g.columns:
                valid = g[pcol].notna()
                n_valid = int(valid.sum())
                if n_valid > 0:
                    sig = valid & (g[pcol] < ALPHA)
                    momentum = sig & (g[vcol] > 1)
                    reversion = sig & (g[vcol] < 1)
                    row[f"vr_{q}_n_valid"] = n_valid
                    row[f"vr_{q}_pct_sig"] = 100 * sig.sum() / n_valid
                    row[f"vr_{q}_n_sig"] = int(sig.sum())
                    row[f"vr_{q}_pct_momentum"] = 100 * momentum.sum() / n_valid
                    row[f"vr_{q}_n_momentum"] = int(momentum.sum())
                    row[f"vr_{q}_pct_reversion"] = 100 * reversion.sum() / n_valid
                    row[f"vr_{q}_n_reversion"] = int(reversion.sum())

        for label, pcol in [("returns", "lb_returns_p"), ("squared", "lb_sq_p")]:
            if pcol in g.columns:
                valid = g[pcol].notna()
                n_valid = int(valid.sum())
                if n_valid > 0:
                    sig = valid & (g[pcol] < ALPHA)
                    row[f"lb_{label}_n_valid"] = n_valid
                    row[f"lb_{label}_pct_sig"] = 100 * sig.sum() / n_valid
                    row[f"lb_{label}_n_sig"] = int(sig.sum())

        summaries.append(row)

    return pd.DataFrame(summaries)

def main():
    print("=" * 78)
    print("AIR ALPHA - Tier 0/1 tests on completed departures")
    print(f"Significance level (two-tailed): alpha = {ALPHA}")
    print(f"Expected 'significant' rate under pure noise: ~{NOISE_FLOOR_PCT:.1f}%")
    print("=" * 78)

    df = prepare_data(load_data())
    completed = get_completed_series(df)
    daily = build_daily_panel(completed)

    n_completed = daily.groupby(["route", "departure_date"]).ngroups
    print(f"\nCompleted (route, departure_date) series found: {n_completed}")
    print(daily.groupby("route")["departure_date"].nunique().to_string())

    _, switch_by_route = flight_switch_diagnostics(daily)
    print("\n--- Flight-switch diagnostic (completed series) ---")
    print(switch_by_route.to_string())

    #for exclude in [False, True]:
    #label = (
     #       "SWITCHES EXCLUDED (strict same-flight)"
      #      if exclude else
       #     "SWITCHES INCLUDED (default: one continuous track)"
        #)
    #print("\n" + "=" * 78)
    #print(label)
    #print("=" * 78)

    print("\n" + "=" * 78)
    print("CONTINUOUS OBSERVED FARE SERIES")
    print("=" * 78)

    series_dict = build_return_series(daily, min_return_obs=5, exclude_switched=False)
    print(f"Completed series with >= min_return_obs valid daily returns: {len(series_dict)}")

    per_flight = run_tier1_tests_per_flight(series_dict)
    print("\n--- Per-flight results (first 10 rows) ---")
    print(per_flight.head(10).to_string())

    summary = route_summary(per_flight)
    print("\n--- Route-level summary ---")
    print(f"(Compare every 'pct' column above against the ~{NOISE_FLOOR_PCT:.1f}% noise floor.)")
    print(summary.to_string())


if __name__ == "__main__":
    main()