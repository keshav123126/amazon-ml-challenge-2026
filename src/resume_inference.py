import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

from run_inference import (
    WORKERS,
    TEMP_DIR,
    OUTPUT_MATCHING,
    OUTPUT_CANDIDATES,
    chunk_generator,
    process_chunk,
    initialize_worker,
)


def merge_chunk(result):
    chunk_number, matching_path, candidate_path, processed, candidate_count, match_count = result

    with open(matching_path, "rb") as src:
        with open(OUTPUT_MATCHING, "ab") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)

    with open(candidate_path, "rb") as src:
        with open(OUTPUT_CANDIDATES, "ab") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)

    os.remove(matching_path)
    os.remove(candidate_path)

    print(
        f"Merged chunk {chunk_number:05d} | "
        f"processed={processed:,} | "
        f"candidates={candidate_count:,} | "
        f"matches={match_count:,}"
    )


def main():

    os.makedirs("output", exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Start completely fresh.
    with open(OUTPUT_MATCHING, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")

    with open(OUTPUT_CANDIDATES, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")

    print("Starting fresh inference...")
    print(f"Workers: {WORKERS}")
    print("Temporary chunks are deleted immediately after merging.")
    print()

    next_chunk = 0
    pending = {}

    with ProcessPoolExecutor(
        max_workers=WORKERS,
        initializer=initialize_worker
    ) as executor:

        for chunk in chunk_generator():

            future = executor.submit(
                process_chunk,
                chunk
            )

            pending[future] = chunk[0]

            # Keep only WORKERS jobs in flight.
            if len(pending) >= WORKERS:

                done = next(
                    as_completed(pending)
                )

                result = done.result()

                del pending[done]

                merge_chunk(result)

                next_chunk += 1

        # Finish remaining jobs.
        while pending:

            done = next(
                as_completed(pending)
            )

            result = done.result()

            del pending[done]

            merge_chunk(result)

            next_chunk += 1

    print()
    print("========================================")
    print("INFERENCE COMPLETE")
    print("========================================")
    print()
    print(f"Matching:   {OUTPUT_MATCHING}")
    print(f"Candidates: {OUTPUT_CANDIDATES}")


if __name__ == "__main__":
    main()