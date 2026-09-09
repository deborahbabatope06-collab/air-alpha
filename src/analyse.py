import pandas as pd
import sqlite3
from pathlib import Path
ANALYSIS_DB_PATH = Path(
    "data/analysis/airfares.db"
)

# load fare observations
def load_data():

    with sqlite3.connect(ANALYSIS_DB_PATH) as conn:

        df = pd.read_sql_query(
            """
            SELECT *
            FROM fare_observations
            ORDER BY observed_at
            """,
            conn,
        )

    return df


def prepare_data(df):
    """
    Convert raw database fields into
    analysis-ready variables.
    """

    df = df.copy()

    # -----------------------------------
    # Observation timestamp
    # -----------------------------------

    # observed_at time stored in UTC.
    df["observed_at"] = pd.to_datetime(
        df["observed_at"],
        utc=True,
    )

    df["observation_date"] = (
        df["observed_at"].dt.tz_localize(None).dt.normalize()
    )

    df["observation_hour"] = (
        df["observed_at"].dt.hour
    )

    # -----------------------------------
    # Departure dates/times
    # -----------------------------------

    df["departure_date"] = pd.to_datetime(
        df["departure_date"],
    )

    df["departure_at"] = pd.to_datetime(
        df["departure_at"],
    )

    df["arrival_at"] = pd.to_datetime(
        df["arrival_at"]
    )

    # -----------------------------------
    # Days to departure
    # -----------------------------------

    observation_day = (
        df["observed_at"]
        .dt.tz_localize(None)
        .dt.normalize()
    )

    df["days_to_departure"] = (
        df["departure_date"]
        - observation_day
    ).dt.days

    # sampling frequency
    def sampling_frequency(days):
        if days <= 14:
            return "hourly"
        elif days <= 60:
            return "six_hourly"
        else:
            return "daily"

    df["sampling_frequency"] = (
        df["days_to_departure"].apply(sampling_frequency)
    )

    # -----------------------------------
    # Route
    # -----------------------------------

    df["route"] = (
        df["origin"]
        + "-"
        + df["destination"]
    )

    # -----------------------------------
    # Departure calendar variables
    # -----------------------------------

    df["departure_weekday"] = (
        df["departure_date"]
        .dt.day_name()
    )

    df["departure_weekday_number"] = (
        df["departure_date"]
        .dt.dayofweek
    )

    df["departure_month"] = (
        df["departure_date"]
        .dt.month
    )

    df["departure_hour"] = (
        df["departure_at"]
        .dt.hour
    )

    # -----------------------------------
    # Expected sampling regime
    # -----------------------------------

    df["sampling_regime"] = pd.cut(
        df["days_to_departure"],
        bins=[
            0,
            14,
            60,
            365,
        ],
        labels=[
            "hourly",
            "six_hourly",
            "daily",
        ],
    )

    return df


def validate_data(df):
    """
    Perform basic integrity checks on
    the research dataset.
    """

    print("\n==============================")
    print("DATA VALIDATION")
    print("==============================")

    print(
        f"Number of observations: "
        f"{len(df):,}"
    )

    print(
        f"Number of routes: "
        f"{df['route'].nunique()}"
    )

    print(
        f"Number of departure dates: "
        f"{df['departure_date'].nunique()}"
    )

    print(
        f"Earliest observation: "
        f"{df['observed_at'].min()}"
    )

    print(
        f"Latest observation: "
        f"{df['observed_at'].max()}"
    )

    print(
        f"Minimum price: "
        f"£{df['price'].min():.2f}"
    )

    print(
        f"Maximum price: "
        f"£{df['price'].max():.2f}"
    )

    print(
        f"Median price: "
        f"£{df['price'].median():.2f}"
    )

    print(
        f"Mean price: "
        f"£{df['price'].mean():.2f}"
    )

    print(
        f"Minimum days to departure: "
        f"{df['days_to_departure'].min()}"
    )

    print(
        f"Maximum days to departure: "
        f"{df['days_to_departure'].max()}"
    )

    print(
        f"Missing prices: "
        f"{df['price'].isna().sum()}"
    )

    print(
        f"Missing departure times: "
        f"{df['departure_at'].isna().sum()}"
    )

    print(
        f"Non-GBP observations: "
        f"{(df['currency'] != 'GBP').sum()}"
    )

    print(
        f"Non-positive prices: "
        f"{(df['price'] <= 0).sum()}"
    )

    duplicates = df.duplicated(
        subset=[
            "observed_at",
            "origin",
            "destination",
            "departure_date",
        ]
    ).sum()

    print(
        f"Duplicate observations: "
        f"{duplicates}"
    )

    print(
        f"Unique observation timestamps: "
        f"{df['observed_at'].nunique()}"
    )

    print("\nObservations by sampling regime:")
    print(
        df["sampling_frequency"].value_counts()
    )


def route_summary(df):
    """
    Summarise observations and prices
    for each route.
    """

    print("\n==============================")
    print("ROUTE SUMMARY")
    print("==============================")

    summary = (
        df.groupby("route")
        .agg(
            observations=(
                "price",
                "size",
            ),
            collection_times=(
                "observed_at",
                "nunique",
            ),
            departure_dates=(
                "departure_date",
                "nunique",
            ),
            mean_price=(
                "price",
                "mean",
            ),
            median_price=(
                "price",
                "median",
            ),
            min_price=(
                "price",
                "min",
            ),
            max_price=(
                "price",
                "max",
            ),
            min_dtd=(
                "days_to_departure",
                "min",
            ),
            max_dtd=(
                "days_to_departure",
                "max",
            ),
        )
        .round(2)
        .sort_index()
    )

    print(summary)


def sampling_summary(df):
    """
    Show how many observations belong to
    each intended sampling regime.
    """

    print("\n==============================")
    print("SAMPLING REGIME SUMMARY")
    print("==============================")

    summary = (
        df.groupby(
            "sampling_regime",
            observed=True,
        )
        .agg(
            observations=(
                "price",
                "size",
            ),
            collection_times=(
                "observed_at",
                "nunique",
            ),
            departure_dates=(
                "departure_date",
                "nunique",
            ),
            routes=(
                "route",
                "nunique",
            ),
        )
    )

    print(summary)


def collection_summary(df):
    """
    Summarise individual collection runs.

    Each collector run receives one observed_at
    timestamp, so this lets us inspect how many
    observations each run produced.
    """

    print("\n==============================")
    print("COLLECTION RUN SUMMARY")
    print("==============================")

    summary = (
        df.groupby("observed_at")
        .agg(
            observations=(
                "price",
                "size",
            ),
            routes=(
                "route",
                "nunique",
            ),
            minimum_dtd=(
                "days_to_departure",
                "min",
            ),
            maximum_dtd=(
                "days_to_departure",
                "max",
            ),
        )
        .sort_index()
    )

    print(
        summary.tail(20)
    )


def show_sample(df):
    """
    Display a small analysis-ready sample.
    """

    print("\n==============================")
    print("SAMPLE")
    print("==============================")

    columns = [
        "observed_at",
        "route",
        "departure_date",
        "days_to_departure",
        "sampling_frequency",
        "departure_weekday",
        "departure_hour",
        "price",
    ]

    print(
        df[columns].head(10)
    )


def main():

    df = load_data()

    if df.empty:

        print(
            "Database contains no observations."
        )

        return

    df = prepare_data(df)

    validate_data(df)

    route_summary(df)

    sampling_summary(df)

    collection_summary(df)

    show_sample(df)


if __name__ == "__main__":
    main()