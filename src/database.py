import sqlite3
from pathlib import Path


DB_PATH = Path("data/airfares.db")

# connect to the SQLite database
def connect():

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return sqlite3.connect(DB_PATH)

# create ryanair fare observations table
def create_tables():

    with connect() as conn:

        conn.execute("""
        CREATE TABLE IF NOT EXISTS fare_observations (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            observed_at TEXT NOT NULL,

            origin TEXT NOT NULL,
            destination TEXT NOT NULL,

            departure_date TEXT NOT NULL,
            departure_at TEXT,
            arrival_at TEXT,

            price REAL NOT NULL,
            currency TEXT NOT NULL,

            sold_out INTEGER NOT NULL,
            unavailable INTEGER NOT NULL,

            source TEXT NOT NULL,

            UNIQUE (
                observed_at,
                origin,
                destination,
                departure_date
            )
        )
        """)

    # insert a timestamped snapshot of fares
def insert_observations(
    observations,
    observed_at,
    ):

    with connect() as conn:

        for fare in observations:

            conn.execute(
                """
                INSERT OR IGNORE INTO fare_observations (

                    observed_at,

                    origin,
                    destination,

                    departure_date,
                    departure_at,
                    arrival_at,

                    price,
                    currency,

                    sold_out,
                    unavailable,

                    source
                )

                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observed_at,

                    fare["origin"],
                    fare["destination"],

                    fare["departure_date"],
                    fare["departure_at"],
                    fare["arrival_at"],

                    fare["price"],
                    fare["currency"],

                    int(fare["sold_out"]),
                    int(fare["unavailable"]),

                    "RYANAIR_FARE_FINDER",
                ),
            )


def main():

    create_tables()
    print("Database created successfully.")

if __name__ == "__main__":
    main()