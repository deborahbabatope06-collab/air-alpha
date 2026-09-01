import requests


BASE_URL = "https://www.ryanair.com/api/farfnd/v4"


def get_monthly_fares(
    origin: str,
    destination: str,
    month: str,
    currency: str = "GBP",
):
    """
    Retrieve Ryanair's cheapest available fare
    for each departure day in a given month.
    """

    url = (
        f"{BASE_URL}/oneWayFares/"
        f"{origin}/{destination}/cheapestPerDay"
    )

    params = {
        "outboundMonthOfDate": month,
        "currency": currency,
    }

    response = requests.get(
        url,
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    return response.json()


def parse_fares(data):
    """
    Convert Ryanair's raw JSON response into
    clean fare observations.
    """

    fares = (
        data
        .get("outbound", {})
        .get("fares", [])
    )

    parsed = []

    for fare in fares:

        price_data = fare.get("price")

        if price_data is not None:

            price = price_data.get("value")

            currency = price_data.get(
                "currencyCode"
            )

        else:

            price = None
            currency = None

        observation = {

            "departure_date":
                fare.get("day"),

            "departure_at":
                fare.get("departureDate"),

            "arrival_at":
                fare.get("arrivalDate"),

            "price":
                price,

            "currency":
                currency,

            "sold_out":
                fare.get(
                    "soldOut",
                    False,
                ),

            "unavailable":
                fare.get(
                    "unavailable",
                    False,
                ),
        }

        if validate_fare(observation):

            parsed.append(
                observation
            )

    return parsed


def validate_fare(fare):
    """
    Check that a fare is suitable for
    our research dataset.
    """

    if fare["departure_date"] is None:
        return False

    if fare["unavailable"]:
        return False

    if fare["sold_out"]:
        return False

    if fare["price"] is None:
        return False

    if fare["price"] <= 0:
        return False

    if fare["currency"] != "GBP":
        return False

    return True


def main():
    """
    Simple test using one route/month.
    """

    data = get_monthly_fares(
        origin="STN",
        destination="VIE",
        month="2026-09-01",
        currency="GBP",
    )

    fares = parse_fares(data)

    for fare in fares:
        print(fare)


if __name__ == "__main__":
    main()