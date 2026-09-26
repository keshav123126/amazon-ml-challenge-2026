import sqlite3
import pandas as pd

from common import name_prefix, address_prefix


DATABASE_FILE = "experiments/entity_index.db"


# ---------------------------------------------------------
# Open the blocking database.
# ---------------------------------------------------------

connection = sqlite3.connect(DATABASE_FILE)


# ---------------------------------------------------------
# Load training Source 1.
#
# We only test a sample first because the complete training
# set contains more than 2.2 million S1 entities.
# ---------------------------------------------------------

print("Loading Source 1...")

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

# 20,000 records are enough for our first blocking experiment.
sample = s1.sample(
    min(20000, len(s1)),
    random_state=42
)

print("Testing S1 records:", len(sample))


# ---------------------------------------------------------
# Load ground truth.
#
# Ground truth tells us which S2/S3 records are actually
# matched with every S1 record.
# ---------------------------------------------------------

print("Loading ground truth...")

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

ground_truth = ground_truth.set_index(
    "source1_entity_id"
)


# ---------------------------------------------------------
# Statistics that we want to measure.
# ---------------------------------------------------------

entities_with_match = 0
entities_found = 0

total_candidates = 0
maximum_candidates = 0


# ---------------------------------------------------------
# Check every sampled S1 entity.
# ---------------------------------------------------------

for _, row in sample.iterrows():

    s1_id = row["entity_id"]

    # Get the true S2/S3 IDs for this S1.
    true_value = ground_truth.loc[
        s1_id,
        "matched_entity_ids"
    ]

    # Empty ground truth means this business has no match.
    if not true_value:
        continue

    true_ids = set(
        true_value.split(",")
    )

    entities_with_match += 1

    candidates = set()

    country = row["country"]

    # ---------------------------------------------
    # Candidate block based on business name.
    # ---------------------------------------------

    nkey = name_prefix(
        row["business_name"]
    )

    if nkey:

        result = connection.execute(
            """
            SELECT entity_id
            FROM entities
            WHERE country = ?
            AND name_prefix = ?
            """,
            (country, nkey)
        )

        for item in result:
            candidates.add(item[0])

    # ---------------------------------------------
    # Candidate block based on address.
    # ---------------------------------------------

    akey = address_prefix(
        row["business_address"]
    )

    if akey:

        result = connection.execute(
            """
            SELECT entity_id
            FROM entities
            WHERE country = ?
            AND address_prefix = ?
            """,
            (country, akey)
        )

        for item in result:
            candidates.add(item[0])

    # Keep statistics about candidate size.
    total_candidates += len(candidates)

    maximum_candidates = max(
        maximum_candidates,
        len(candidates)
    )

    # Check whether at least one true match was retrieved.
    if true_ids.intersection(candidates):
        entities_found += 1


# ---------------------------------------------------------
# Display the blocking results.
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("BLOCKING RECALL RESULTS")
print("=" * 60)

print(
    "S1 entities with true matches:",
    entities_with_match
)

print(
    "S1 entities where a true match was found:",
    entities_found
)

if entities_with_match > 0:

    recall = (
        entities_found /
        entities_with_match
    )

    print(
        "Blocking recall:",
        round(recall, 4)
    )

print(
    "Average candidates per matched S1:",
    round(
        total_candidates /
        max(entities_with_match, 1),
        2
    )
)

print(
    "Maximum candidates:",
    maximum_candidates
)


connection.close()