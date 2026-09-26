import os
import sqlite3
import pandas as pd
from common import normalize_text, compact_text

DATABASE_FILE = "experiments/test_entity_index.db"

SOURCE_FILES = [
    "dataset/test/test_source2.tsv",
    "dataset/test/test_source3.tsv"
]


def make_name_key(name):
    """Create the main blocking key from the business name."""
    return compact_text(name)[:8]


def make_address_key(address):
    """Create the main blocking key from the business address."""
    return compact_text(address)[:12]


def make_name_word_key(name):
    """Create a blocking key from the first two normalized name words."""
    text = normalize_text(name)

    if not text:
        return ""

    words = text.split()
    return " ".join(words[:2])


def build_database():

    os.makedirs("experiments", exist_ok=True)

    if os.path.exists(DATABASE_FILE):
        os.remove(DATABASE_FILE)

    connection = sqlite3.connect(DATABASE_FILE)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE entities (
            entity_id TEXT PRIMARY KEY,
            country TEXT,
            business_name TEXT,
            business_address TEXT,
            name_key TEXT,
            address_key TEXT,
            name_word_key TEXT
        )
    """)

    cursor.execute("""
        CREATE INDEX idx_name_key
        ON entities(country, name_key)
    """)

    cursor.execute("""
        CREATE INDEX idx_address_key
        ON entities(country, address_key)
    """)

    cursor.execute("""
        CREATE INDEX idx_name_word_key
        ON entities(country, name_word_key)
    """)

    connection.commit()

    for file in SOURCE_FILES:

        print("\nProcessing:", file)

        for chunk in pd.read_csv(
            file,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=100000
        ):

            chunk["name_key"] = chunk["business_name"].map(make_name_key)

            chunk["address_key"] = chunk["business_address"].map(
                make_address_key
            )

            chunk["name_word_key"] = chunk["business_name"].map(
                make_name_word_key
            )

            rows = chunk[
                [
                    "entity_id",
                    "country",
                    "business_name",
                    "business_address",
                    "name_key",
                    "address_key",
                    "name_word_key"
                ]
            ]

            rows.to_sql(
                "entities",
                connection,
                if_exists="append",
                index=False
            )

            print("Inserted:", len(rows))

    connection.commit()

    total = cursor.execute(
        "SELECT COUNT(*) FROM entities"
    ).fetchone()[0]

    connection.close()

    print("\n" + "=" * 60)
    print("TEST INDEX CREATED")
    print("=" * 60)
    print("Total indexed records:", total)
    print("Database:", DATABASE_FILE)


if __name__ == "__main__":
    build_database()