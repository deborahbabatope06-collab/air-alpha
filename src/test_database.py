from src.database import connect, create_tables


def test_database():

    # Make sure table exists
    create_tables()

    # Insert fake airfare
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO observations (
                observed_at,
                flight_id,
                origin,
                destination,
                departure_at,
                airline,
                flight_number,
                price,
                currency,
                cabin,
                fare_type,
                stops,
                source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-08-30T10:00:00",
                "FR1234_2026-10-20",
                "STN",
                "VIE",
                "2026-10-20T08:15:00",
                "Ryanair",
                "FR1234",
                39.99,
                "GBP",
                "ECONOMY",
                "BASIC",
                0,
                "TEST",
            ),
        )

    # Read it back
    with connect() as conn:
        result = conn.execute(
            """
            SELECT flight_number, origin, destination, price
            FROM observations
            WHERE flight_id = ?
            """,
            ("FR1234_2026-10-20",),
        ).fetchone()

    print(result)

    assert result is not None
    assert result[0] == "FR1234"
    assert result[1] == "STN"
    assert result[2] == "VIE"
    assert result[3] == 39.99

    print("Database test passed!")


if __name__ == "__main__":
    test_database()