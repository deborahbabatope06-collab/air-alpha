from src.database import (
    connect,
    DB_PATH,
)

def main():

    print(
        f"Database path: "
        f"{DB_PATH.resolve()}"
    )

    with connect() as conn:

        # show every table in the database
        tables = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        ).fetchall()

        print("\nTables:")

        for table in tables:
            print(table[0])

        # check whether table exists
        table_exists = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'fare_observations'
            """
        ).fetchone()

        if table_exists is None:

            print(
                "\nERROR: fare_observations "
                "table does not exist."
            )

            return

        # count rows
        count = conn.execute(
            """
            SELECT COUNT(*)
            FROM fare_observations
            """
        ).fetchone()[0]

        print(
            f"\nNumber of observations: "
            f"{count}"
        )

        # show table columns
        columns = conn.execute(
            """
            PRAGMA table_info(
                fare_observations
            )
            """
        ).fetchall()

        print("\nColumns:")

        for column in columns:
            print(column[1])

        # show sample rows
        rows = conn.execute(
            """
            SELECT *
            FROM fare_observations
            ORDER BY id
            LIMIT 10
            """
        ).fetchall()

        print("\nFirst 10 observations:")

        for row in rows:
            print(row)


if __name__ == "__main__":
    main()