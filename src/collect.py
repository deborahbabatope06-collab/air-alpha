from datetime import (
    date,
    datetime,
    timezone,
)

import sys
import time

import pandas as pd
import requests

from src.database import (
    create_tables,
    insert_observations,
)

from src.ryanair import (
    get_monthly_fares,
    parse_fares,
)

ROUTES_FILE = "config/routes.csv"

REQUEST_DELAY_SECONDS = 0.5


COLLECTION_MODES = {

    # 1-14 dbd
    "hourly": {
        "max_dtd": 14,
        "number_of_months": 2,
    },

    # 1-60 dbd
    "six_hourly": {
        "max_dtd": 60,
        "number_of_months": 3,
    },

    # 1-365 dbd
    "daily": {
        "max_dtd": 365,
        "number_of_months": 13,
    },
}


def get_month_starts(
    number_of_months,
):

    today = date.today()

    months = []

    year = today.year
    month = today.month

    for i in range(number_of_months):

        new_month = month + i
        new_year = year

        while new_month > 12:

            new_month -= 12
            new_year += 1

        months.append(
            f"{new_year:04d}-"
            f"{new_month:02d}-01"
        )

    return months


def days_to_departure(
    departure_date,
):

    departure = date.fromisoformat(
        departure_date
    )

    return (
        departure
        - date.today()
    ).days

# collect data for one route within requested booking horizon
def collect_route(
    origin,
    destination,
    max_dtd,
    number_of_months,
):

    observations = []

    months = get_month_starts(
        number_of_months
    )

    for month in months:

        print(
            f"Fetching {origin} -> "
            f"{destination} | {month}"
        )

        try:

            data = get_monthly_fares(
                origin=origin,
                destination=destination,
                month=month,
                currency="GBP",
            )

            fares = parse_fares(data)

            for fare in fares:

                dtd = days_to_departure(
                    fare["departure_date"]
                )

                if 1 <= dtd <= max_dtd:

                    fare["origin"] = origin
                    fare["destination"] = (
                        destination
                    )

                    observations.append(
                        fare
                    )

        except requests.RequestException as error:

            print(
                f"Request failed: "
                f"{origin} -> {destination} "
                f"{month}: {error}"
            )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    return observations

# collect data within requested horizon for every route in routes.csv
def collect_all_routes(
    max_dtd,
    number_of_months,
):
    
    routes = pd.read_csv(
        ROUTES_FILE
    )

    all_observations = []

    for _, route in routes.iterrows():

        origin = route["origin"]
        destination = route["destination"]

        print(
            f"\n--- "
            f"{origin} -> {destination} "
            f"---"
        )

        route_observations = (
            collect_route(
                origin=origin,
                destination=destination,
                max_dtd=max_dtd,
                number_of_months=number_of_months,
            )
        )

        print(
            f"Found "
            f"{len(route_observations)} "
            f"valid observations."
        )

        all_observations.extend(
            route_observations
        )

    return all_observations


def main():

    # requires a collection mode.
    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python -m src.collect "
            "[hourly|six_hourly|daily]"
        )

        return

    mode = sys.argv[1]

    if mode not in COLLECTION_MODES:

        print(
            f"Unknown collection mode: "
            f"{mode}"
        )

        print(
            "Choose: hourly, "
            "six_hourly, or daily"
        )

        return

    settings = COLLECTION_MODES[
        mode
    ]

    max_dtd = settings["max_dtd"]

    number_of_months = settings[
        "number_of_months"
    ]

    print(
        "\n=============================="
    )

    print(
        f"COLLECTION MODE: "
        f"{mode.upper()}"
    )

    print(
        f"DTD RANGE: 1-{max_dtd}"
    )

    print(
        "=============================="
    )

    create_tables()

    # One timestamp represents this
    # entire collection snapshot.
    observed_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    observations = collect_all_routes(
        max_dtd=max_dtd,
        number_of_months=number_of_months,
    )

    print(
        "\n=============================="
    )

    print(
        "COLLECTION COMPLETE"
    )

    print(
        "=============================="
    )

    print(
        f"Mode: {mode}"
    )

    print(
        f"Total observations: "
        f"{len(observations)}"
    )

    if not observations:

        print(
            "Nothing to save."
        )

        return

    insert_observations(
        observations=observations,
        observed_at=observed_at,
    )

    print(
        f"Saved "
        f"{len(observations)} "
        f"observations."
    )


if __name__ == "__main__":
    main()