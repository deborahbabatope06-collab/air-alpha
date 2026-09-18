from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.analyse import load_data, prepare_data

OUTPUT_DIR = Path("outputs/figures")


def get_daily_panel(df):
    """
    get daily observations 
    (sampled at 18:00/collections for dtd > 60)
    """

    # get max dtd for each collection in run (14,60,365)
    run_max_dtd = (df.groupby("observed_at")["days_to_departure"].max())

    # get timestamps for daily observations (dtd > 60)
    daily_timestamps = (run_max_dtd[run_max_dtd > 60].index)

    daily_panel = df[
    df["observed_at"].isin(daily_timestamps)].copy()

    return daily_panel

def add_dtd_bands(df):
    """
    Divide the booking horizon into
    interpretable groups.
    """

    df = df.copy()

    bins = [
        0,
        7,
        14,
        30,
        60,
        90,
        180,
        270,
        365,
    ]

    labels = [
        "1-7",
        "8-14",
        "15-30",
        "31-60",
        "61-90",
        "91-180",
        "181-270",
        "271-365",
    ]

    df["dtd_band"] = pd.cut(
        df["days_to_departure"],
        bins=bins,
        labels=labels,
    )

    return df

def print_booking_curve_summary(df):
    """
    Describe the fare distribution at
    different booking horizons.
    """

    summary = (
        df.groupby(
            "dtd_band",
            observed=True,
        )
        .agg(
            observations=("price", "size"),
            mean_price=("price", "mean"),
            median_price=("price", "median"),
            std_price=("price", "std"),
            min_price=("price", "min"),
            max_price=("price", "max"),
        )
        .round(2)
    )

    print("\n==============================")
    print("BOOKING HORIZON SUMMARY")
    print("==============================")

    print(summary)

def print_route_summary(df):
    """
    Compare price distributions by route.
    """

    summary = (
        df.groupby("route")
        .agg(
            observations=("price", "size"),
            mean_price=("price", "mean"),
            median_price=("price", "median"),
            std_price=("price", "std"),
        )
        .round(2)
        .sort_values("median_price")
    )

    print("\n==============================")
    print("ROUTE PRICE SUMMARY")
    print("==============================")

    print(summary)

def print_calendar_summary(df):
    """
    Examine fare distributions by departure weekday
    and departure month.
    """

    weekday_summary = (
        df.groupby("departure_weekday")
        .agg(
            observations=("price", "size"),
            mean_price=("price", "mean"),
            median_price=("price", "median"),
            std_price=("price", "std"),
        )
        .round(2)
    )

    month_summary = (
        df.groupby("departure_month")
        .agg(
            observations=("price", "size"),
            mean_price=("price", "mean"),
            median_price=("price", "median"),
            std_price=("price", "std"),
        )
        .round(2)
        .sort_index()
    )

    print("\n==============================")
    print("DEPARTURE WEEKDAY SUMMARY")
    print("==============================")

    print(weekday_summary)

    print("\n==============================")
    print("DEPARTURE MONTH SUMMARY")
    print("==============================")

    print(month_summary)

def print_holiday_summary(df):
    """
    Examine fare distributions across travel-demand
    holiday states.
    """

    summary = (
        df.groupby(
            [
                "uk_school_holiday",
                "major_holiday",
                "major_holiday_window_7d",
            ]
        )
        .agg(
            observations=("price", "size"),
            mean_price=("price", "mean"),
            median_price=("price", "median"),
            std_price=("price", "std"),
        )
        .round(2)
    )

    print("\n==============================")
    print("HOLIDAY / TRAVEL-DEMAND SUMMARY")
    print("==============================")

    print(summary)

def print_major_holiday_summary(df):
    """
    Compare fare distributions during named
    major holiday periods.
    """

    holiday_data = df[
        df["major_holiday_name"].notna()
    ]

    summary = (
        holiday_data
        .groupby("major_holiday_name")
        .agg(
            observations=("price", "size"),
            mean_price=("price", "mean"),
            median_price=("price", "median"),
            std_price=("price", "std"),
        )
        .round(2)
    )

    print("\n==============================")
    print("MAJOR HOLIDAY SUMMARY")
    print("==============================")

    print(summary)

def print_school_holiday_summary(df):
    """
    Compare fare distributions during named
    school holiday periods.
    """

    holiday_data = df[
        df["school_holiday_name"].notna()
    ]

    summary = (
        holiday_data
        .groupby("school_holiday_name")
        .agg(
            observations=("price", "size"),
            mean_price=("price", "mean"),
            median_price=("price", "median"),
            std_price=("price", "std"),
        )
        .round(2)
    )

    print("\n==============================")
    print("SCHOOL HOLIDAY SUMMARY")
    print("==============================")

    print(summary)

def plot_median_price_by_route_dtd(df):
    """
    Plot median fare for each route and
    dtd value.
    """

    route_dtd_medians = (
        df.groupby(["route", "days_to_departure"])
        ["price"]
        .median()
        .reset_index()
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(12, 7)
    )

    for route in sorted(
        route_dtd_medians["route"].unique()
    ):

        route_data = route_dtd_medians[
            route_dtd_medians["route"] == route
        ].sort_values(
            "days_to_departure"
        )

        plt.plot(
            route_data["days_to_departure"],
            route_data["price"],
            label=route,
        )

    plt.xlabel(
        "Days to departure"
    )

    plt.ylabel(
        "Median cheapest fare (£)"
    )

    plt.title(
        "Median Ryanair Fare vs Days to Departure by Route"
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "median_price_vs_dtd_by_route.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved figure: {output_path}"
    )

def plot_school_holiday_distributions(df):
    """
    Compare fare distributions for each UK school-holiday
    period against departures outside school holidays.
    """

    plot_data = df.copy()

    plot_data["school_holiday_group"] = "Normal"

    plot_data.loc[
        plot_data["uk_school_holiday"],
        "school_holiday_group"
    ] = plot_data.loc[
        plot_data["uk_school_holiday"],
        "school_holiday_name"
    ]

    school_holidays = sorted(
        plot_data.loc[
            plot_data["uk_school_holiday"],
            "school_holiday_name"
        ]
        .dropna()
        .unique()
    )

    groups = ["Normal"] + school_holidays

    data = [
        plot_data.loc[
            plot_data["school_holiday_group"] == group,
            "price"
        ]
        for group in groups
    ]

    plt.figure(
        figsize=(12, 7)
    )

    plt.boxplot(
        data,
        tick_labels=groups,
        showfliers=True,
    )

    plt.xlabel(
        "Departure period"
    )

    plt.ylabel(
        "Fare (£)"
    )

    plt.title(
        "Fare Distribution by UK School-Holiday Period"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "fare_distribution_by_school_holiday.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved figure: {output_path}"
    )

def plot_major_holiday_distributions(df):
    """
    Compare fare distributions for each major holiday
    against departures outside major-holiday periods.
    """

    plot_data = df.copy()

    plot_data["major_holiday_group"] = "Normal"

    plot_data.loc[
        plot_data["major_holiday"],
        "major_holiday_group"
    ] = plot_data.loc[
        plot_data["major_holiday"],
        "major_holiday_name"
    ]

    major_holidays = sorted(
        plot_data.loc[
            plot_data["major_holiday"],
            "major_holiday_name"
        ]
        .dropna()
        .unique()
    )

    groups = ["Normal"] + major_holidays

    data = [
        plot_data.loc[
            plot_data["major_holiday_group"] == group,
            "price"
        ]
        for group in groups
    ]

    plt.figure(
        figsize=(10, 7)
    )

    plt.boxplot(
        data,
        tick_labels=groups,
        showfliers=True,
    )

    plt.xlabel(
        "Departure period"
    )

    plt.ylabel(
        "Fare (£)"
    )

    plt.title(
        "Fare Distribution by Major Holiday Period"
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "fare_distribution_by_major_holiday.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved figure: {output_path}"
    )

def plot_holiday_price_by_dtd(df):
    """
    Plot median fare against days to departure,
    separated by travel-demand state.
    """

    plot_data = df.copy()

    plot_data["holiday_state"] = "Normal"

    plot_data.loc[
        plot_data["major_holiday_window_7d"],
        "holiday_state"
    ] = "Major holiday window"

    plot_data.loc[
        plot_data["major_holiday"],
        "holiday_state"
    ] = "Major holiday"

    plot_data.loc[
        plot_data["uk_school_holiday"],
        "holiday_state"
    ] = "UK school holiday"

    states = [
        "Normal",
        "Major holiday window",
        "Major holiday",
        "UK school holiday",
    ]

    plt.figure(
        figsize=(11, 7)
    )

    for state in states:

        state_data = plot_data[
            plot_data["holiday_state"] == state
        ]

        median_by_dtd = (
            state_data
            .groupby("days_to_departure")["price"]
            .median()
            .sort_index()
        )

        plt.plot(
            median_by_dtd.index,
            median_by_dtd.values,
            label=state,
        )

    plt.xlabel(
        "Days to departure"
    )

    plt.ylabel(
        "Median fare (£)"
    )

    plt.title(
        "Median Fare vs Days to Departure by Travel-Demand State"
    )

    plt.legend()

    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "median_price_vs_dtd_by_holiday_state.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"Saved figure: {output_path}"
    )

def main():

    df = load_data()

    df = prepare_data(df)

    daily_panel = get_daily_panel(df)

    daily_panel = add_dtd_bands(
        daily_panel
    )

    print("\n==============================")
    print("EDA DATASET")
    print("==============================")

    print(
        f"All observations: "
        f"{len(df)}"
    )

    print(
        f"Daily-panel observations: "
        f"{len(daily_panel)}"
    )

    print(
        f"Full daily snapshots: "
        f"{daily_panel['observed_at'].nunique()}"
    )

    print_booking_curve_summary(
        daily_panel
    )

    print_route_summary(
        daily_panel
    )

    print_calendar_summary(
        daily_panel
    )

    print_holiday_summary(
        daily_panel
    )
    
    print_major_holiday_summary(
    daily_panel
    )

    print_school_holiday_summary(
    daily_panel
    )
    
    plot_school_holiday_distributions(
    daily_panel
    )

    plot_major_holiday_distributions(
        daily_panel
    )

    plot_median_price_by_route_dtd(
        daily_panel
    )

    plot_holiday_price_by_dtd(
        daily_panel
    )

if __name__ == "__main__":
    main()

