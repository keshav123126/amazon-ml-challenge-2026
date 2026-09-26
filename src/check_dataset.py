import os
import pandas as pd


# These are the files we need to understand before building the model.
files = [
    "dataset/train/train_source1.tsv",
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv",
    "dataset/train/train_ground_truth.tsv",
    "dataset/test/test_source1.tsv",
    "dataset/test/test_source2.tsv",
    "dataset/test/test_source3.tsv"
]


def check_source(file):
    print("\n" + "=" * 70)
    print(file)

    total_rows = 0
    sample = None

    # Store missing-value counts for each column.
    missing = {}

    # Store country frequencies.
    countries = {}

    # Read the file in chunks because these files are very large.
    for chunk in pd.read_csv(file, sep="\t", chunksize=100000):

        total_rows += len(chunk)

        # Keep a few rows so that we can see what the data looks like.
        if sample is None:
            sample = chunk.head(3)

        # Count missing values.
        for col in chunk.columns:
            if col not in missing:
                missing[col] = 0

            missing[col] += chunk[col].isna().sum()

        # Count countries if the file contains the country column.
        if "country" in chunk.columns:

            counts = chunk["country"].fillna("MISSING").value_counts()

            for country, count in counts.items():
                countries[country] = countries.get(country, 0) + count

    print("Total rows:", total_rows)
    print("Columns:", list(sample.columns))

    print("\nSample rows:")
    print(sample.to_string(index=False))

    print("\nMissing values:")
    for col, count in missing.items():
        print(f"{col}: {count}")

    if countries:
        print("\nCountries:")

        for country, count in sorted(
            countries.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"{country}: {count}")


def check_ground_truth(file):
    print("\n" + "=" * 70)
    print(file)

    total_rows = 0
    zero_matches = 0

    # Example:
    # 0 matches -> 5000 entities
    # 1 match   -> 15000 entities
    # 2 matches -> 3000 entities
    match_counts = {}

    # Ground truth can also be large, so read it in chunks.
    for chunk in pd.read_csv(
        file,
        sep="\t",
        chunksize=100000,
        keep_default_na=False
    ):

        total_rows += len(chunk)

        for value in chunk["matched_entity_ids"]:

            # Empty value means that this S1 entity has no match.
            if value == "":
                count = 0
            else:
                count = len(value.split(","))

            if count == 0:
                zero_matches += 1

            match_counts[count] = match_counts.get(count, 0) + 1

    print("Total rows:", total_rows)
    print("Entities with zero matches:", zero_matches)

    print("\nMatch-count distribution:")

    for count in sorted(match_counts):
        print(f"{count} matches: {match_counts[count]}")


# Check every dataset file.
for file in files:

    if not os.path.exists(file):
        print("\nFILE NOT FOUND:", file)
        continue

    if "ground_truth" in file:
        check_ground_truth(file)
    else:
        check_source(file)