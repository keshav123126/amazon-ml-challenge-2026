import os
import csv
import sqlite3
import time
import shutil
import joblib

from concurrent.futures import ProcessPoolExecutor

from rapidfuzz.fuzz import ratio, token_set_ratio, WRatio

from common import normalize_text, compact_text


# ============================================================
# FILES
# ============================================================

DATABASE_FILE = "experiments/test_entity_index.db"
MODEL_FILE = "experiments/matching_model.pkl"

TEST_SOURCE1 = "dataset/test/test_source1.tsv"

OUTPUT_MATCHING = "output/matching_results.tsv"
OUTPUT_CANDIDATES = "output/candidate_pairs.tsv"

TEMP_DIR = "output/temp_inference"


# ============================================================
# SETTINGS
# ============================================================

# Your PC has 16 logical CPUs.
# We leave a few CPUs free for Windows/SQLite.
WORKERS = 4

# Number of Source1 records handled by one worker job.
CHUNK_SIZE = 10000

# IMPORTANT:
# Full dataset. Do NOT put 10000 here.
TEST_LIMIT = None

THRESHOLD = 0.70


# ============================================================
# GLOBAL WORKER OBJECTS
# ============================================================

# Each worker gets its own database connection and model.
worker_connection = None
worker_model = None


# ============================================================
# TEXT FUNCTIONS
# ============================================================

def prepare_text(text):
    """
    Normalize text once and also create its compact version.
    """

    normalized = normalize_text(text)

    compact = normalized.replace(" ", "")

    return normalized, compact


def name_key(name):
    """
    Name blocking key.
    """

    return compact_text(name)[:8]


def address_key(address):
    """
    Address blocking key.
    """

    return compact_text(address)[:12]


def name_word_key(name):
    """
    First two normalized words of the business name.
    """

    text = normalize_text(name)

    if not text:
        return ""

    return " ".join(text.split()[:2])


# ============================================================
# WORKER INITIALIZATION
# ============================================================

def initialize_worker():
    """
    Runs once when a worker process starts.

    Every worker gets:
    - its own SQLite connection
    - its own loaded ML model
    """

    global worker_connection
    global worker_model

    worker_model = joblib.load(
        MODEL_FILE
    )

    worker_connection = sqlite3.connect(
        DATABASE_FILE
    )

    # SQLite is only being read.
    worker_connection.execute(
        "PRAGMA query_only = ON"
    )

    # Give SQLite a reasonable memory cache.
    worker_connection.execute(
        "PRAGMA cache_size = -100000"
    )

    worker_connection.execute(
        "PRAGMA temp_store = MEMORY"
    )


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def get_candidates(
    country,
    nk,
    ak,
    nwk
):
    """
    Generate candidates using the three indexed blocking strategies.
    Each block is limited separately.
    """

    global worker_connection

    candidates = {}

    # --------------------------------------------------------
    # Name block
    # --------------------------------------------------------

    if nk:
        rows = worker_connection.execute(
            """
            SELECT entity_id, business_name, business_address
            FROM entities
            WHERE country = ?
              AND name_key = ?
            LIMIT 300
            """,
            (country, nk)
        ).fetchall()

        for row in rows:
            candidates[row[0]] = row

    # --------------------------------------------------------
    # Address block
    # --------------------------------------------------------

    if ak:
        rows = worker_connection.execute(
            """
            SELECT entity_id, business_name, business_address
            FROM entities
            WHERE country = ?
              AND address_key = ?
            LIMIT 300
            """,
            (country, ak)
        ).fetchall()

        for row in rows:
            candidates[row[0]] = row

    # --------------------------------------------------------
    # Name-word block
    # --------------------------------------------------------

    if nwk:
        rows = worker_connection.execute(
            """
            SELECT entity_id, business_name, business_address
            FROM entities
            WHERE country = ?
              AND name_word_key = ?
            LIMIT 300
            """,
            (country, nwk)
        ).fetchall()

        for row in rows:
            candidates[row[0]] = row

    return list(candidates.values())

# ============================================================
# FEATURES
# ============================================================

def create_features(
    s1_name_norm,
    s1_name_compact,
    s1_address_norm,
    s1_address_compact,
    s2_name_norm,
    s2_name_compact,
    s2_address_norm,
    s2_address_compact
):
    """
    Exactly the same 14 features used by the trained model.
    """

    return [

        # Exact normalized name
        int(
            s1_name_norm == s2_name_norm
        ),

        # Exact normalized address
        int(
            s1_address_norm == s2_address_norm
        ),

        # Exact compact name
        int(
            s1_name_compact == s2_name_compact
        ),

        # Exact compact address
        int(
            s1_address_compact == s2_address_compact
        ),

        # Name fuzzy features
        ratio(
            s1_name_norm,
            s2_name_norm
        ) / 100.0,

        token_set_ratio(
            s1_name_norm,
            s2_name_norm
        ) / 100.0,

        WRatio(
            s1_name_norm,
            s2_name_norm
        ) / 100.0,

        # Address fuzzy features
        ratio(
            s1_address_norm,
            s2_address_norm
        ) / 100.0,

        token_set_ratio(
            s1_address_norm,
            s2_address_norm
        ) / 100.0,

        WRatio(
            s1_address_norm,
            s2_address_norm
        ) / 100.0,

        # Length differences
        abs(
            len(s1_name_compact)
            -
            len(s2_name_compact)
        ),

        abs(
            len(s1_address_compact)
            -
            len(s2_address_compact)
        ),

        # Prefix agreement
        int(
            s1_name_compact[:8]
            ==
            s2_name_compact[:8]
        ),

        int(
            s1_address_compact[:12]
            ==
            s2_address_compact[:12]
        )
    ]


# ============================================================
# PROCESS ONE CHUNK
# ============================================================

def process_chunk(args):
    """
    Process one chunk of Source1 records.

    The worker:
        blocking
        -> fuzzy pre-filter
        -> feature creation
        -> ML prediction

    Only candidates actually passed to the ML model are
    written to candidate_pairs.tsv.
    """

    chunk_number, rows, column_index = args

    global worker_model

    matching_path = os.path.join(
        TEMP_DIR,
        f"matching_{chunk_number:05d}.tsv"
    )

    candidate_path = os.path.join(
        TEMP_DIR,
        f"candidate_{chunk_number:05d}.tsv"
    )

    matching_file = open(
        matching_path,
        "w",
        encoding="utf-8",
        newline="",
        buffering=1024 * 1024
    )

    candidate_file = open(
        candidate_path,
        "w",
        encoding="utf-8",
        newline="",
        buffering=1024 * 1024
    )

    matching_writer = csv.writer(
        matching_file,
        delimiter="\t"
    )

    candidate_writer = csv.writer(
        candidate_file,
        delimiter="\t"
    )

    processed = 0
    candidate_count = 0
    match_count = 0

    for row in rows:

        s1_id = row[
            column_index["entity_id"]
        ]

        country = row[
            column_index["country"]
        ]

        business_name = row[
            column_index["business_name"]
        ]

        business_address = row[
            column_index["business_address"]
        ]

        # ----------------------------------------------------
        # Prepare Source1 text once.
        # ----------------------------------------------------

        s1_name_norm, s1_name_compact = prepare_text(
            business_name
        )

        s1_address_norm, s1_address_compact = prepare_text(
            business_address
        )

        # ----------------------------------------------------
        # Blocking keys.
        # ----------------------------------------------------

        nk = s1_name_compact[:8]

        ak = s1_address_compact[:12]

        nwk = name_word_key(
            business_name
        )

        # ----------------------------------------------------
        # Candidate generation.
        # ----------------------------------------------------

        candidates = get_candidates(
            country,
            nk,
            ak,
            nwk
        )

        if not candidates:

            candidate_writer.writerow(
                [s1_id, ""]
            )

            matching_writer.writerow(
                [s1_id, ""]
            )

            processed += 1

            continue

        # ----------------------------------------------------
        # Cheap fuzzy pre-filter.
        #
        # This is the same filtering idea as your previous
        # working version.
        # ----------------------------------------------------

        feature_rows = []
        candidate_ids = []

        for (
            candidate_id,
            candidate_name,
            candidate_address
        ) in candidates:

            candidate_name_norm = normalize_text(
                candidate_name
            )

            candidate_address_norm = normalize_text(
                candidate_address
            )

            name_score = ratio(
                s1_name_norm,
                candidate_name_norm
            )

            address_score = ratio(
                s1_address_norm,
                candidate_address_norm
            )

            exact_match = (
                s1_name_norm == candidate_name_norm
                or
                s1_address_norm == candidate_address_norm
            )

            # Only candidates surviving this stage are
            # actually sent to the trained ML model.

            if (
                exact_match
                or
                name_score >= 45
                or
                address_score >= 45
            ):

                candidate_name_compact = (
                    candidate_name_norm.replace(
                        " ",
                        ""
                    )
                )

                candidate_address_compact = (
                    candidate_address_norm.replace(
                        " ",
                        ""
                    )
                )

                features = create_features(
                    s1_name_norm,
                    s1_name_compact,
                    s1_address_norm,
                    s1_address_compact,
                    candidate_name_norm,
                    candidate_name_compact,
                    candidate_address_norm,
                    candidate_address_compact
                )

                feature_rows.append(
                    features
                )

                candidate_ids.append(
                    candidate_id
                )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # candidate_ids here are EXACTLY the candidates
        # that will be passed to the ML model.
        # ----------------------------------------------------

        candidate_writer.writerow(
            [
                s1_id,
                ",".join(
                    candidate_ids
                )
            ]
        )

        candidate_count += len(
            candidate_ids
        )

        # ----------------------------------------------------
        # ML prediction.
        # ----------------------------------------------------

        if not feature_rows:

            matching_writer.writerow(
                [s1_id, ""]
            )

            processed += 1

            continue

        probabilities = worker_model.predict_proba(
            feature_rows
        )[:, 1]

        matches = []

        for candidate_id, probability in zip(
            candidate_ids,
            probabilities
        ):

            if probability >= THRESHOLD:

                matches.append(
                    (
                        candidate_id,
                        probability
                    )
                )

        # Highest probability first.

        matches.sort(
            key=lambda x: x[1],
            reverse=True
        )

        final_ids = [
            item[0]
            for item in matches
        ]

        matching_writer.writerow(
            [
                s1_id,
                ",".join(final_ids)
            ]
        )

        match_count += len(
            final_ids
        )

        processed += 1

    matching_file.close()
    candidate_file.close()

    return (
        chunk_number,
        matching_path,
        candidate_path,
        processed,
        candidate_count,
        match_count
    )


# ============================================================
# READ SOURCE1 IN CHUNKS
# ============================================================

def chunk_generator():

    source1_file = open(
        TEST_SOURCE1,
        "r",
        encoding="utf-8",
        newline=""
    )

    reader = csv.reader(
        source1_file,
        delimiter="\t"
    )

    header = next(reader)

    column_index = {
        name: index
        for index, name in enumerate(header)
    }

    chunk = []
    chunk_number = 0

    for row in reader:

        chunk.append(row)

        if len(chunk) >= CHUNK_SIZE:

            yield (
                chunk_number,
                chunk,
                column_index
            )

            chunk_number += 1
            chunk = []

    if chunk:

        yield (
            chunk_number,
            chunk,
            column_index
        )

    source1_file.close()


# ============================================================
# MERGE TEMPORARY FILES
# ============================================================

def merge_outputs(results):

    print()
    print("=" * 60)
    print("MERGING WORKER OUTPUTS")
    print("=" * 60)

    results.sort(
        key=lambda x: x[0]
    )

    with open(
        OUTPUT_MATCHING,
        "w",
        encoding="utf-8",
        newline="",
        buffering=1024 * 1024
    ) as final_matching:

        writer = csv.writer(
            final_matching,
            delimiter="\t"
        )

        writer.writerow(
            [
                "source1_entity_id",
                "matched_entity_ids"
            ]
        )

        for result in results:

            matching_path = result[1]

            with open(
                matching_path,
                "r",
                encoding="utf-8"
            ) as part:

                shutil.copyfileobj(
                    part,
                    final_matching
                )

    with open(
        OUTPUT_CANDIDATES,
        "w",
        encoding="utf-8",
        newline="",
        buffering=1024 * 1024
    ) as final_candidates:

        writer = csv.writer(
            final_candidates,
            delimiter="\t"
        )

        writer.writerow(
            [
                "source1_entity_id",
                "candidate_entity_ids"
            ]
        )

        for result in results:

            candidate_path = result[2]

            with open(
                candidate_path,
                "r",
                encoding="utf-8"
            ) as part:

                shutil.copyfileobj(
                    part,
                    final_candidates
                )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    os.makedirs(
        "output",
        exist_ok=True
    )

    # Delete old temporary files.
    if os.path.exists(TEMP_DIR):

        shutil.rmtree(
            TEMP_DIR
        )

    os.makedirs(
        TEMP_DIR,
        exist_ok=True
    )

    print("=" * 60)
    print("AMAZON ML CHALLENGE - FINAL INFERENCE")
    print("=" * 60)

    print(
        "Workers:",
        WORKERS
    )

    print(
        "Chunk size:",
        CHUNK_SIZE
    )

    print(
        "Threshold:",
        THRESHOLD
    )

    print()
    print(
        "Starting FULL test inference..."
    )

    results = []

    # --------------------------------------------------------
    # Process chunks in parallel.
    # --------------------------------------------------------

    with ProcessPoolExecutor(
        max_workers=WORKERS,
        initializer=initialize_worker
    ) as executor:

        jobs = executor.map(
            process_chunk,
            chunk_generator()
        )

        completed = 0
        total_processed = 0
        total_candidates = 0
        total_matches = 0

        for result in jobs:

            results.append(
                result
            )

            completed += 1

            total_processed += result[3]
            total_candidates += result[4]
            total_matches += result[5]

            elapsed = (
                time.time()
                -
                start_time
            )

            speed = (
                total_processed
                /
                max(elapsed, 1)
            )

            print(
                f"Chunks: {completed} | "
                f"Records: {total_processed} | "
                f"Speed: {speed:.1f} records/sec | "
                f"Candidates: {total_candidates} | "
                f"Matches: {total_matches} | "
                f"Time: {elapsed / 60:.1f} min"
            )

    # --------------------------------------------------------
    # Merge all worker files.
    # --------------------------------------------------------

    merge_outputs(
        results
    )

    # --------------------------------------------------------
    # Remove temporary files.
    # --------------------------------------------------------

    shutil.rmtree(
        TEMP_DIR
    )

    elapsed = (
        time.time()
        -
        start_time
    )

    print()
    print("=" * 60)
    print("FULL INFERENCE COMPLETE")
    print("=" * 60)

    print(
        "Total records:",
        total_processed
    )

    print(
        "Average final candidates:",
        round(
            total_candidates
            /
            max(total_processed, 1),
            2
        )
    )

    print(
        "Total predicted matches:",
        total_matches
    )

    print(
        "Total time:",
        round(
            elapsed / 60,
            2
        ),
        "minutes"
    )

    print()
    print(
        "Created:"
    )

    print(
        OUTPUT_MATCHING
    )

    print(
        OUTPUT_CANDIDATES
    )


if __name__ == "__main__":
    main()