import pandas as pd
import sqlite3
from pathlib import Path

ANALYSIS_DB_PATH = Path(
    "data/analysis/airfares.db"
)

CALENDAR_PATH = Path(
    "config/travel_demand_events.csv"
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

def load_calendar_events():
    """
    Load recurring travel-demand events.

    CSV contains month/day definitions - the year is generated from the
    departure dates in the airfare dataset.
    """

    events = pd.read_csv(
        CALENDAR_PATH
    )

    return events

def add_calendar_features(df):
    """
    Add travel-demand calendar variables based on
    the departure date.

    Event dates are generated for the years present
    in the airfare dataset.
    """

    events = load_calendar_events()

    df = df.copy()

    # Candidate calendar features.
    df["uk_school_holiday"] = False
    df["school_holiday_name"] = pd.NA

    df["major_holiday"] = False
    df["major_holiday_window_7d"] = False
    df["major_holiday_name"] = pd.NA

    # Generate enough years to cover the dataset.
    min_year = (
        df["departure_date"]
        .dt.year
        .min()
    )

    max_year = (
        df["departure_date"]
        .dt.year
        .max()
    )

    generated_events = []

    for year in range(
        min_year - 1,
        max_year + 2,
    ):

        for _, event in events.iterrows():

            start_year = year
            end_year = year

            # Detect events that cross New Year.
            if (
                int(event["end_month"])
                < int(event["start_month"])
                or (
                    int(event["end_month"])
                    == int(event["start_month"])
                    and int(event["end_day"])
                    < int(event["start_day"])
                )
            ):
                end_year = year + 1

            start_date = pd.Timestamp(
                year=start_year,
                month=int(event["start_month"]),
                day=int(event["start_day"]),
            )

            end_date = pd.Timestamp(
                year=end_year,
                month=int(event["end_month"]),
                day=int(event["end_day"]),
            )

            generated_events.append(
                {
                    "event_name": event["event_name"],
                    "event_type": event["event_type"],
                    "market_scope": event["market_scope"],
                    "start_date": start_date,
                    "end_date": end_date,
                }
            )

    generated_events = pd.DataFrame(
        generated_events
    )

    for _, event in generated_events.iterrows():

        start_date = event["start_date"]
        end_date = event["end_date"]

        # Exact event period.
        in_event = (
            (df["departure_date"] >= start_date)
            &
            (df["departure_date"] <= end_date)
        )

        # Seven-day window around the event.
        in_window = (
            (
                df["departure_date"]
                >= start_date - pd.Timedelta(days=7)
            )
            &
            (
                df["departure_date"]
                <= end_date + pd.Timedelta(days=7)
            )
        )

        # UK school holidays.
        if (
            event["market_scope"] == "UK"
            and event["event_type"] == "school_holiday"
        ):

            df.loc[
                in_event,
                "uk_school_holiday"
            ] = True

            df.loc[
                in_event,
                "school_holiday_name"
            ] = event["event_name"]

        # Major holidays applying to all routes.
        if (
            event["market_scope"] == "ALL"
            and event["event_type"] == "major_holiday"
        ):

            df.loc[
                in_event,
                "major_holiday"
            ] = True

            df.loc[
                in_window,
                "major_holiday_window_7d"
            ] = True

            df.loc[
                in_event,
                "major_holiday_name"
            ] = event["event_name"]

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

    df=add_calendar_features(df)

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

    print(
        "\nTravel-demand event counts:"
    )

    print(
        f"UK school holiday: "
        f"{df['uk_school_holiday'].sum():,}"
    )

    print(
    f"Named school-holiday observations: "
    f"{df['school_holiday_name'].notna().sum():,}"
    )

    print(
        f"Major holiday: "
        f"{df['major_holiday'].sum():,}"
    )

    print(
    f"Major-holiday window (7d): "
    f"{df['major_holiday_window_7d'].sum():,}"
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
        .reset_index()
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
        "uk_school_holiday",
        "school_holiday_name",
        "major_holiday",
        "major_holiday_window_7d",
        "major_holiday_name",
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