import pandas as pd

from src.database import connect


def load_data():
    """
    Load all fare observations from SQLite
    into a pandas DataFrame.
    """

    with connect() as conn:

        df = pd.read_sql_query(
            """
            SELECT *
            FROM fare_observations
            """,
            conn,
        )

    return df


def prepare_data(df):
    """
    Convert dates into useful formats and
    create basic research variables.
    """

    df = df.copy()

    # Convert stored text dates to pandas dates.
    df["observation_date"] = pd.to_datetime(
        df["observation_date"]
    )

    df["departure_date"] = pd.to_datetime(
        df["departure_date"]
    )

    df["departure_at"] = pd.to_datetime(
        df["departure_at"]
    )

    # Days until departure.
    df["days_to_departure"] = (
        df["departure_date"]
        - df["observation_date"]
    ).dt.days

    # Route identifier.
    df["route"] = (
        df["origin"]
        + "-"
        + df["destination"]
    )

    # Day of week on which the flight departs.
    df["departure_weekday"] = (
        df["departure_date"]
        .dt.day_name()
    )

    # Month of departure.
    df["departure_month"] = (
        df["departure_date"]
        .dt.month
    )

    # Hour at which the cheapest flight departs.
    df["departure_hour"] = (
        df["departure_at"]
        .dt.hour
    )

    return df


def validate_data(df):
    """
    Print basic checks to make sure the
    collected dataset looks sensible.
    """

    print("\n==============================")
    print("DATA VALIDATION")
    print("==============================")

    print(
        f"Number of observations: "
        f"{len(df)}"
    )

    print(
        f"Number of routes: "
        f"{df['route'].nunique()}"
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
        f"Average price: "
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

    duplicates = df.duplicated(
        subset=[
            "observation_date",
            "origin",
            "destination",
            "departure_date",
        ]
    ).sum()

    print(
        f"Duplicate observations: "
        f"{duplicates}"
    )


def route_summary(df):
    """
    Show basic statistics for each route.
    """

    print("\n==============================")
    print("ROUTE SUMMARY")
    print("==============================")

    summary = (
        df.groupby("route")
        .agg(
            observations=("price", "size"),
            mean_price=("price", "mean"),
            median_price=("price", "median"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            max_dtd=(
                "days_to_departure",
                "max",
            ),
        )
        .round(2)
        .sort_index()
    )

    print(summary)


def show_sample(df):
    """
    Display a few useful columns.
    """

    print("\n==============================")
    print("SAMPLE")
    print("==============================")

    columns = [
        "observation_date",
        "route",
        "departure_date",
        "days_to_departure",
        "departure_weekday",
        "departure_hour",
        "price",
    ]

    print(
        df[columns].head(10)
    )


def main():

    df = load_data()

    df = prepare_data(df)

    validate_data(df)

    route_summary(df)

    show_sample(df)


if __name__ == "__main__":
    main()