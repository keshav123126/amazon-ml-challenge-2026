import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import run_inference as inf


# =========================
# FINAL SETTINGS
# =========================

inf.WORKERS = 4
inf.CHUNK_SIZE = 10000
inf.TEST_LIMIT = None
inf.THRESHOLD = 0.70

TEMP_DIR = inf.TEMP_DIR
OUTPUT_MATCHING = inf.OUTPUT_MATCHING
OUTPUT_CANDIDATES = inf.OUTPUT_CANDIDATES

PROGRESS_FILE = os.path.join(
    TEMP_DIR,
    "completed_chunks.txt"
)


# =========================
# PROGRESS FUNCTIONS
# =========================

def load_completed_chunks():
    completed = set()

    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line:
                    completed.add(int(line))

    return completed


def save_completed_chunk(chunk_number):
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(str(chunk_number) + "\n")
        f.flush()
        os.fsync(f.fileno())


# =========================
# OUTPUT SETUP
# =========================

def setup_outputs(completed):
    os.makedirs("output", exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # If this is a completely new run
    if not completed:

        with open(OUTPUT_MATCHING, "w", encoding="utf-8") as f:
            f.write(
                "source1_entity_id\tmatched_entity_ids\n"
            )

        with open(OUTPUT_CANDIDATES, "w", encoding="utf-8") as f:
            f.write(
                "source1_entity_id\tcandidate_entity_ids\n"
            )


# =========================
# MERGE
# =========================

def merge_chunk(result):
    (
        chunk_number,
        matching_path,
        candidate_path,
        processed,
        candidate_count,
        match_count
    ) = result

    # Make sure both files exist before touching final output
    if not os.path.exists(matching_path):
        raise FileNotFoundError(matching_path)

    if not os.path.exists(candidate_path):
        raise FileNotFoundError(candidate_path)

    # Append matching results
    with open(matching_path, "rb") as src:
        with open(OUTPUT_MATCHING, "ab") as dst:
            shutil.copyfileobj(
                src,
                dst,
                1024 * 1024
            )

    # Append candidate results
    with open(candidate_path, "rb") as src:
        with open(OUTPUT_CANDIDATES, "ab") as dst:
            shutil.copyfileobj(
                src,
                dst,
                1024 * 1024
            )

    # Only delete temporary files after successful merge
    os.remove(matching_path)
    os.remove(candidate_path)

    # Mark chunk completed only after both files
    # have successfully been merged and deleted.
    save_completed_chunk(chunk_number)

    print(
        f"Completed chunk {chunk_number:05d} | "
        f"rows={processed:,} | "
        f"candidates={candidate_count:,} | "
        f"matches={match_count:,}"
    )


# =========================
# MAIN
# =========================

def main():

    os.makedirs(TEMP_DIR, exist_ok=True)

    completed = load_completed_chunks()

    setup_outputs(completed)

    print("=" * 60)
    print("FINAL AMAZON ML INFERENCE")
    print("=" * 60)

    print(f"Workers       : {inf.WORKERS}")
    print(f"Chunk size    : {inf.CHUNK_SIZE}")
    print(f"Threshold     : {inf.THRESHOLD}")
    print(f"Completed     : {len(completed)} chunks")
    print("=" * 60)
    print()

    pending = {}

    with ProcessPoolExecutor(
        max_workers=inf.WORKERS,
        initializer=inf.initialize_worker
    ) as executor:

        for chunk in inf.chunk_generator():

            chunk_number = chunk[0]

            # Already completed in an earlier run
            if chunk_number in completed:
                continue

            # If a complete-looking temporary chunk already exists
            # from an interrupted run, use it instead of recomputing.
            matching_path = os.path.join(
                TEMP_DIR,
                f"matching_{chunk_number:05d}.tsv"
            )

            candidate_path = os.path.join(
                TEMP_DIR,
                f"candidate_{chunk_number:05d}.tsv"
            )

            if (
                os.path.exists(matching_path)
                and os.path.exists(candidate_path)
            ):
                print(
                    f"Found existing chunk "
                    f"{chunk_number:05d}, merging it..."
                )

                fake_result = (
                    chunk_number,
                    matching_path,
                    candidate_path,
                    0,
                    0,
                    0
                )

                merge_chunk(fake_result)

                continue

            future = executor.submit(
                inf.process_chunk,
                chunk
            )

            pending[future] = chunk_number

            # Keep only 4 chunks running
            if len(pending) >= inf.WORKERS:

                done = next(
                    as_completed(pending)
                )

                result = done.result()

                del pending[done]

                merge_chunk(result)

        # Finish remaining chunks
        while pending:

            done = next(
                as_completed(pending)
            )

            result = done.result()

            del pending[done]

            merge_chunk(result)

    print()
    print("=" * 60)
    print("FINAL INFERENCE COMPLETE")
    print("=" * 60)

    print(
        f"Matching file  : {OUTPUT_MATCHING}"
    )

    print(
        f"Candidate file : {OUTPUT_CANDIDATES}"
    )

    print()


if __name__ == "__main__":
    main()