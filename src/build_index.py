import os
import sqlite3
import pandas as pd

from common import name_prefix, address_prefix


# ---------------------------------------------------------
# File locations
# ---------------------------------------------------------

SOURCE_FILES = [
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv"
]

DATABASE_FILE = "experiments/entity_index.db"


# ---------------------------------------------------------
# Create the database
# ---------------------------------------------------------

def create_database():

    # Create the experiments folder if it does not exist.
    os.makedirs("experiments", exist_ok=True)

    # Remove an old experiment database so that we start clean.
    if os.path.exists(DATABASE_FILE):
        os.remove(DATABASE_FILE)

    print("Creating database...")
    connection = sqlite3.connect(DATABASE_FILE)

    cursor = connection.cursor()

    # Store only the fields required for candidate generation.
    cursor.execute("""
        CREATE TABLE entities (
            entity_id TEXT PRIMARY KEY,
            country TEXT,
            business_name TEXT,
            business_address TEXT,
            name_prefix TEXT,
            address_prefix TEXT
        )
    """)

    # Indexes make candidate lookup much faster.
    # Country is included because the challenge contains multiple countries.
    cursor.execute("""
        CREATE INDEX idx_name_block
        ON entities(country, name_prefix)
    """)

    cursor.execute("""
        CREATE INDEX idx_address_block
        ON entities(country, address_prefix)
    """)

    connection.commit()

    # -----------------------------------------------------
    # Read Source 2 and Source 3 in chunks.
    # -----------------------------------------------------

    for file in SOURCE_FILES:

        print("\nProcessing:", file)

        for chunk in pd.read_csv(
            file,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=100000
        ):

            # Create the blocking keys.
            chunk["name_prefix"] = chunk[
                "business_name"
            ].map(name_prefix)

            chunk["address_prefix"] = chunk[
                "business_address"
            ].map(address_prefix)

            # Keep only the columns required by our index.
            rows = chunk[
                [
                    "entity_id",
                    "country",
                    "business_name",
                    "business_address",
                    "name_prefix",
                    "address_prefix"
                ]
            ]

            # Insert the current chunk into SQLite.
            rows.to_sql(
                "entities",
                connection,
                if_exists="append",
                index=False
            )

            print(
                "  inserted:",
                len(rows),
                "rows"
            )

    connection.commit()

    # Get the final number of indexed records.
    cursor.execute("SELECT COUNT(*) FROM entities")
    total = cursor.fetchone()[0]

    connection.close()

    print("\n" + "=" * 60)
    print("INDEX CREATION COMPLETE")
    print("=" * 60)
    print("Indexed Source 2 + Source 3 records:", total)
    print("Database:", DATABASE_FILE)


if __name__ == "__main__":
    create_database()