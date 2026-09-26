import os
import sqlite3
import random

import pandas as pd
import numpy as np

from rapidfuzz.fuzz import (
    ratio,
    token_set_ratio,
    WRatio
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from common import normalize_text, compact_text


DATABASE_FILE = "experiments/better_entity_index.db"

GROUND_TRUTH_FILE = "dataset/train/train_ground_truth.tsv"
SOURCE1_FILE = "dataset/train/train_source1.tsv"

MODEL_FILE = "experiments/matching_model.pkl"


# ---------------------------------------------------------
# Blocking functions
# ---------------------------------------------------------

def name_key(name):
    """Return a compact prefix used for candidate generation."""
    return compact_text(name)[:8]


def address_key(address):
    """Return a compact address prefix used for candidate generation."""
    return compact_text(address)[:12]


def name_word_key(name):
    """Return the first two normalized words of a business name."""
    text = normalize_text(name)

    if not text:
        return ""

    return " ".join(text.split()[:2])


# ---------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------

def get_candidates(connection, row):
    """
    Generate candidates using several blocking keys.

    We intentionally avoid comparing a Source 1 record against
    every Source 2/3 record because the dataset contains more
    than ten million noisy records.
    """

    candidates = set()

    country = row["country"]

    nk = name_key(row["business_name"])

    if nk:
        result = connection.execute(
            """
            SELECT entity_id
            FROM entities
            WHERE country = ?
            AND name_key = ?
            LIMIT 500
            """,
            (country, nk)
        )

        candidates.update(x[0] for x in result)

    ak = address_key(row["business_address"])

    if ak:
        result = connection.execute(
            """
            SELECT entity_id
            FROM entities
            WHERE country = ?
            AND address_key = ?
            LIMIT 500
            """,
            (country, ak)
        )

        candidates.update(x[0] for x in result)

    nwk = name_word_key(row["business_name"])

    if nwk:
        result = connection.execute(
            """
            SELECT entity_id
            FROM entities
            WHERE country = ?
            AND name_word_key = ?
            LIMIT 500
            """,
            (country, nwk)
        )

        candidates.update(x[0] for x in result)

    return candidates


# ---------------------------------------------------------
# Similarity features
# ---------------------------------------------------------

def create_features(s1_name, s1_address, s2_name, s2_address):
    """
    Convert two business records into numerical ML features.

    The model uses several different similarities because no
    single string metric works well for every type of noise.
    """

    n1 = normalize_text(s1_name)
    n2 = normalize_text(s2_name)

    a1 = normalize_text(s1_address)
    a2 = normalize_text(s2_address)

    cn1 = compact_text(s1_name)
    cn2 = compact_text(s2_name)

    ca1 = compact_text(s1_address)
    ca2 = compact_text(s2_address)

    features = [

        # Exact normalized comparisons.
        int(n1 == n2),
        int(a1 == a2),

        # Compact comparison handles punctuation and spaces.
        int(cn1 == cn2),
        int(ca1 == ca2),

        # Business-name similarity.
        ratio(n1, n2) / 100.0,
        token_set_ratio(n1, n2) / 100.0,
        WRatio(n1, n2) / 100.0,

        # Address similarity.
        ratio(a1, a2) / 100.0,
        token_set_ratio(a1, a2) / 100.0,
        WRatio(a1, a2) / 100.0,

        # Length differences help distinguish accidental
        # high-similarity matches from genuine matches.
        abs(len(cn1) - len(cn2)),
        abs(len(ca1) - len(ca2)),

        # Prefix agreement.
        int(cn1[:8] == cn2[:8]),
        int(ca1[:12] == ca2[:12])
    ]

    return features


# ---------------------------------------------------------
# Load source records
# ---------------------------------------------------------

def load_source_records():

    print("Loading Source 1...")

    s1 = pd.read_csv(
        SOURCE1_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    # Dictionary lookup is much faster than repeatedly reading
    # the TSV file during candidate generation.
    records = {}

    for _, row in s1.iterrows():

        records[row["entity_id"]] = (
            row["business_name"],
            row["business_address"],
            row["country"]
        )

    return records


# ---------------------------------------------------------
# Build supervised training examples
# ---------------------------------------------------------

def build_training_data(sample_size=15000):

    connection = sqlite3.connect(DATABASE_FILE)

    s1 = load_source_records()

    print("Loading ground truth...")

    ground_truth = pd.read_csv(
        GROUND_TRUTH_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    # We do not need all 2.2 million entities to train the
    # first model. A representative sample is sufficient and
    # keeps training time manageable.
    sample_ids = random.Random(42).sample(
        list(s1.keys()),
        min(sample_size, len(s1))
    )

    rows = []

    positive_count = 0
    negative_count = 0

    print("Generating training pairs...")

    for index, s1_id in enumerate(sample_ids):

        if index % 500 == 0:
            print(
                "Processed:",
                index,
                "/",
                len(sample_ids)
            )

        s1_name, s1_address, country = s1[s1_id]

        gt_row = ground_truth[
            ground_truth["source1_entity_id"] == s1_id
        ]

        if len(gt_row) == 0:
            continue

        true_value = gt_row.iloc[0]["matched_entity_ids"]

        true_ids = set()

        if true_value:
            true_ids = set(true_value.split(","))

        candidates = get_candidates(
            connection,
            {
                "business_name": s1_name,
                "business_address": s1_address,
                "country": country
            }
        )

        # Keep true matches even if a blocking key missed them.
        # This allows the model to learn genuine positive examples.
        candidates.update(true_ids)

        # Limit negative examples so one S1 record with a huge
        # candidate group does not dominate the training set.
        negatives = [
            candidate
            for candidate in candidates
            if candidate not in true_ids
        ]

        random.Random(index).shuffle(negatives)

        negatives = negatives[:5]

        # Create positive examples.
        for candidate in true_ids:

            result = connection.execute(
                """
                SELECT business_name, business_address
                FROM entities
                WHERE entity_id = ?
                """,
                (candidate,)
            ).fetchone()

            if result is None:
                continue

            features = create_features(
                s1_name,
                s1_address,
                result[0],
                result[1]
            )

            rows.append(features + [1])

            positive_count += 1

        # Create negative examples.
        for candidate in negatives:

            result = connection.execute(
                """
                SELECT business_name, business_address
                FROM entities
                WHERE entity_id = ?
                """,
                (candidate,)
            ).fetchone()

            if result is None:
                continue

            features = create_features(
                s1_name,
                s1_address,
                result[0],
                result[1]
            )

            rows.append(features + [0])

            negative_count += 1

    connection.close()

    print("\nPositive pairs:", positive_count)
    print("Negative pairs:", negative_count)

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

def train_model():

    data = build_training_data()

    if len(data) == 0:
        raise RuntimeError("No training pairs were created.")

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    # A 0.5 threshold is not necessarily appropriate for this
    # challenge because false positive matches are expensive.
    for threshold in [0.50, 0.60, 0.70, 0.80, 0.90]:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f05 = (
            1.25 * precision * recall /
            (0.25 * precision + recall)
            if precision + recall > 0
            else 0
        )

        print(
            "Threshold:",
            threshold,
            "Precision:",
            round(precision, 4),
            "Recall:",
            round(recall, 4),
            "F0.5:",
            round(f05, 4)
        )

    # Save the trained model for inference.
    import joblib

    os.makedirs("experiments", exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    print("\nModel saved to:", MODEL_FILE)


if __name__ == "__main__":
    train_model()